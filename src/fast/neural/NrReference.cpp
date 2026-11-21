#include "NrReference.h"

#include <cmath>
#include <functional>
#include <thread>

#include "NrNumeric.h"

namespace Fast::Neural {

namespace {

using Tokens = std::vector<float>; // [rows][channels]

void ParallelFor(size_t count, const std::function<void(size_t)>& body) {
    const size_t workers =
        std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), std::max<size_t>(count, 1));
    std::vector<std::thread> threads;
    for (size_t w = 0; w < workers; ++w) {
        threads.emplace_back([&, w] {
            for (size_t i = w; i < count; i += workers) {
                body(i);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
}

// One output element of an FP8 GEMM: `a` and `w` are K contiguous values, `seed` the scaled skip (a half).
float Dot(Arithmetic arithmetic, const float* a, const float* w, uint32_t K, float seed, uint32_t partition) {
    if (arithmetic == Arithmetic::Fp16) {
        float sum = 0.0f; // the WMMA's f32 accumulator; published as a half, then the skip is a half add
        for (uint32_t k = 0; k < K; ++k) {
            sum += a[k] * w[k];
        }
        return Num::RoundF16(Num::RoundF16(sum) + seed);
    }
    // The K order is part of the result, and so is where the skip enters: it seeds the accumulator. A partition
    // (the ViT's split-K) accumulates from zero and the partial sums are added as halves, in order.
    float sums = seed, total = 0.0f;
    for (uint32_t k = 0; k < K; k += 16) {
        sums = Num::Fp8Dot16(a + k, w + k, sums);
        if (partition && ((k + 16) % partition == 0 || k + 16 >= K)) {
            total = k < partition ? sums : Num::RoundF16(total + sums);
            sums = 0.0f;
        }
    }
    return partition ? total : sums;
}

struct Gemm {
    const Tokens* mInput = nullptr;
    uint32_t mInStride = 0, mInCol = 0;
    uint32_t mRows = 0, mK = 0, mN = 0;
    const float* mWeights = nullptr; // [K][N]
    uint32_t mPartition = 0;
    const Tokens* mResidual = nullptr; // [rows][N at mResCol], scaled per column
    uint32_t mResStride = 0, mResCol = 0;
    const float* mScale = nullptr;
    bool mSilu = false;
    Tokens* mRaw = nullptr;       // the half result
    Tokens* mQuantized = nullptr; // its E4M3 publication
    uint32_t mOutStride = 0, mOutCol = 0;
};

void RunGemm(Arithmetic arithmetic, const Gemm& g) {
    std::vector<float> columns((size_t)g.mK * g.mN); // [n][k]: a column's K weights contiguous
    for (uint32_t k = 0; k < g.mK; ++k) {
        for (uint32_t n = 0; n < g.mN; ++n) {
            columns[(size_t)n * g.mK + k] = g.mWeights[(size_t)k * g.mN + n];
        }
    }
    ParallelFor(g.mRows, [&](size_t row) {
        const float* a = g.mInput->data() + row * g.mInStride + g.mInCol;
        for (uint32_t n = 0; n < g.mN; ++n) {
            float seed = 0.0f;
            if (g.mResidual) {
                seed = Num::RoundF16((*g.mResidual)[row * g.mResStride + g.mResCol + n] * g.mScale[n]);
            }
            float value = Dot(arithmetic, a, &columns[(size_t)n * g.mK], g.mK, seed, g.mPartition);
            if (g.mSilu) {
                value = Num::CubicSilu(value);
            }
            if (g.mRaw) {
                (*g.mRaw)[row * g.mOutStride + g.mOutCol + n] = value;
            }
            if (g.mQuantized) {
                (*g.mQuantized)[row * g.mOutStride + g.mOutCol + n] = Num::Fp8Domain(value);
            }
        }
    });
}

// The two f16 GEMMs (input adapter, RGBA head): half operands, eight-product steps on the tensor cores.
Tokens GemmF16(Arithmetic arithmetic, const Tokens& input, uint32_t rows, uint32_t K, uint32_t N,
               const std::vector<float>& weights) {
    Tokens out((size_t)rows * N);
    ParallelFor(rows, [&](size_t row) {
        float a[8], b[8];
        for (uint32_t n = 0; n < N; ++n) {
            float value = 0.0f;
            if (arithmetic == Arithmetic::Fp16) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < K; ++k) {
                    sum += input[row * K + k] * weights[(size_t)k * N + n];
                }
                value = Num::RoundF16(sum);
            } else {
                for (uint32_t base = 0; base < K; base += 8) {
                    for (uint32_t i = 0; i < 8; ++i) {
                        a[i] = input[row * K + base + i];
                        b[i] = weights[(size_t)(base + i) * N + n];
                    }
                    value = Num::F16Dot8(a, b, value);
                }
            }
            out[row * N + n] = value;
        }
    });
    return out;
}

Tokens Quantized(const Tokens& raw) {
    Tokens q(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        q[i] = Num::Fp8Domain(raw[i]);
    }
    return q;
}

// 1 / |v| over 32 halves. Window blocks: r[c] = fma(v[c], v[c], f16(v[c+16]^2)); the ViT keeps the low square in
// f32 and rounds the sum once. Then a stride 8/4/2/1 tree, every level a half.
float InverseNorm(const float* v, bool vit) {
    float r[16];
    for (int c = 0; c < 16; ++c) {
        const float highSquare = Num::RoundF16(v[c + 16] * v[c + 16]);
        if (vit) {
            r[c] = Num::RoundF16(v[c] * v[c] + highSquare); // f32 square, f32 add, one rounding to half
        } else {
            r[c] = Num::RoundF16((float)((double)v[c] * (double)v[c] + (double)highSquare));
        }
    }
    for (int stride = 8; stride > 0; stride >>= 1) {
        for (int c = 0; c < stride; ++c) {
            r[c] = Num::AddF16(r[c], r[c + stride]);
        }
    }
    return Num::RoundF16(1.0f / std::sqrt(r[0]));
}

// Window attention: raw QKV [tokens][3C] (head-major Q | K | V) -> attended E4 [tokens][C].
Tokens WindowAttention(Arithmetic arithmetic, const Tokens& qkv, uint32_t width, uint32_t height, uint32_t channels,
                       uint32_t phase, const std::vector<float>& prior, const std::vector<float>& qScale) {
    const uint32_t heads = channels / 32, tokens = width * height, stride = channels * 3;
    // Cosine normalization: Q scaled by the learned temperature, K to unit length, V only quantized. A zero row is
    // 0 * inf: it publishes as +0.
    Tokens normalized(qkv.size());
    ParallelFor(tokens, [&](size_t token) {
        for (uint32_t head = 0; head < heads; ++head) {
            const float* q = &qkv[token * stride + head * 96];
            float* out = &normalized[token * stride + head * 96];
            const float qNorm = InverseNorm(q, false), kNorm = InverseNorm(q + 32, false);
            const float scale = Num::RoundF16(qScale[head]);
            for (int c = 0; c < 32; ++c) {
                out[c] = Num::Fp8Domain(Num::RoundF16(Num::RoundF16(q[c] * qNorm) * scale));
                out[32 + c] = Num::Fp8Domain(Num::RoundF16(q[32 + c] * kNorm));
                out[64 + c] = Num::Fp8Domain(q[64 + c]);
            }
        }
    });

    uint32_t shiftX, shiftY;
    WindowPhase(phase, shiftX, shiftY);
    const uint32_t windowsX = (width + shiftX + 7) / 8, windowsY = (height + shiftY + 7) / 8;
    Tokens attended((size_t)tokens * channels, 0.0f);
    ParallelFor((size_t)windowsX * windowsY * heads, [&](size_t job) {
        const uint32_t head = (uint32_t)(job % heads);
        const uint32_t wx = (uint32_t)(job / heads) % windowsX, wy = (uint32_t)(job / heads) / windowsX;
        const int originX = (int)(wx * 8) - (int)shiftX, originY = (int)(wy * 8) - (int)shiftY;
        // The 64 tokens in tiled ("physical") order: four 4x4 tiles. Tokens outside the field are zero vectors
        // whose exponential still enters the softmax denominator.
        int source[64];
        float q[64][32] = {}, k[64][32] = {}, v[64][32] = {};
        for (uint32_t p = 0; p < 64; ++p) {
            const int x = originX + (int)(((p >> 4) & 1) * 4 + (p & 3)),
                      y = originY + (int)((p >> 5) * 4 + ((p >> 2) & 3));
            source[p] = x < 0 || y < 0 || x >= (int)width || y >= (int)height ? -1 : y * (int)width + x;
            if (source[p] >= 0) {
                const float* row = &normalized[(size_t)source[p] * stride + head * 96];
                std::copy(row, row + 32, q[p]);
                std::copy(row + 32, row + 64, k[p]);
                std::copy(row + 64, row + 96, v[p]);
            }
        }
        const float* headPrior = &prior[(size_t)head * 4096];
        for (uint32_t query = 0; query < 64; ++query) {
            if (source[query] < 0) {
                continue;
            }
            float e[64], weight[64];
            for (uint32_t key = 0; key < 64; ++key) {
                float score;
                if (arithmetic == Arithmetic::Fp8) { // the prior is the C operand of the first step
                    score = Num::Fp8Dot16(q[query], k[key], headPrior[query * 64 + key]);
                    score = Num::Fp8Dot16(q[query] + 16, k[key] + 16, score);
                } else {
                    float sum = 0.0f;
                    for (int c = 0; c < 32; ++c) {
                        sum += q[query][c] * k[key][c];
                    }
                    score = Num::RoundF16(Num::RoundF16(sum) + headPrior[query * 64 + key]);
                }
                e[key] = Num::WindowExpWeight(score);
            }
            float total;
            if (arithmetic == Arithmetic::Fp8) {
                // b[g][j] = e[g+16j] + e[g+16j+8]; t[g] = ((b0+b1)+b2)+b3; then evens and odds separately.
                auto pair = [&](uint32_t pairIndex, uint32_t parity) {
                    const uint32_t key = pairIndex * 2 + parity;
                    const float b01 = Num::AddF16(e[key], e[key + 8]), b23 = Num::AddF16(e[key + 16], e[key + 24]);
                    const float b45 = Num::AddF16(e[key + 32], e[key + 40]),
                                b67 = Num::AddF16(e[key + 48], e[key + 56]);
                    return Num::AddF16(Num::AddF16(Num::AddF16(b01, b23), b45), b67);
                };
                const float even =
                    Num::AddF16(Num::AddF16(Num::AddF16(pair(0, 0), pair(1, 0)), pair(2, 0)), pair(3, 0));
                const float odd = Num::AddF16(Num::AddF16(Num::AddF16(pair(0, 1), pair(1, 1)), pair(2, 1)), pair(3, 1));
                total = Num::AddF16(even, odd);
            } else {
                // The kernel's order: two lanes per row, each summing eight keys of every key tile in four lanes.
                float partial[2];
                for (uint32_t part = 0; part < 2; ++part) {
                    float lane[4] = { 0, 0, 0, 0 };
                    for (uint32_t tile = 0; tile < 4; ++tile) {
                        for (uint32_t i = 0; i < 8; ++i) {
                            lane[i & 3] = Num::AddF16(lane[i & 3], e[tile * 16 + part * 8 + i]);
                        }
                    }
                    partial[part] = Num::AddF16(Num::AddF16(lane[0], lane[2]), Num::AddF16(lane[1], lane[3]));
                }
                total = Num::AddF16(partial[0], partial[1]);
            }
            const float reciprocal = Num::RoundF16(1.0f / total);
            for (uint32_t key = 0; key < 64; ++key) {
                weight[key] = Num::Fp8Domain(Num::RoundF16(e[key] * reciprocal));
            }
            float* out = &attended[(size_t)source[query] * channels + head * 32];
            for (int c = 0; c < 32; ++c) {
                float value = 0.0f;
                if (arithmetic == Arithmetic::Fp8) {
                    float column[64];
                    for (uint32_t i = 0; i < 64; ++i) {
                        column[i] = v[i][c];
                    }
                    value = Dot(arithmetic, weight, column, 64, 0.0f, 0);
                } else {
                    float sum = 0.0f;
                    for (uint32_t key = 0; key < 64; ++key) {
                        sum += weight[key] * v[key][c];
                    }
                    value = Num::RoundF16(sum);
                }
                out[c] = Num::Fp8Domain(value);
            }
        }
    });
    return attended;
}

// The ViT's attention over every level-5 token: no prior, an extra sqrt(32) on Q, unnormalized E4M3 weights with
// the reciprocal applied to the value accumulator, and padding rows that still count in the denominator.
Tokens GlobalAttention(Arithmetic arithmetic, const Tokens& qkv, uint32_t tokens, uint32_t padded,
                       const std::vector<float>& qScale) {
    const uint32_t channels = 1024, heads = 32, stride = channels * 3;
    Tokens normalized((size_t)padded * stride, 0.0f);
    ParallelFor(tokens, [&](size_t token) {
        const float headScale = Num::RoundF16(std::sqrt(32.0f));
        for (uint32_t head = 0; head < heads; ++head) {
            const float* q = &qkv[token * stride + head * 96];
            float* out = &normalized[token * stride + head * 96];
            const float qNorm = InverseNorm(q, true), kNorm = InverseNorm(q + 32, true);
            const float learned = Num::RoundF16(qScale[head]);
            for (int c = 0; c < 32; ++c) {
                const float nq = Num::RoundF16(Num::RoundF16(Num::RoundF16(q[c] * qNorm) * headScale) * learned);
                out[c] = Num::Fp8Domain(nq);
                out[32 + c] = Num::Fp8Domain(Num::RoundF16(q[32 + c] * kNorm));
                out[64 + c] = Num::Fp8Domain(q[64 + c]);
            }
        }
    });
    Tokens attended((size_t)tokens * channels);
    ParallelFor((size_t)tokens * heads, [&](size_t job) {
        const uint32_t head = (uint32_t)(job % heads), query = (uint32_t)(job / heads);
        const float* q = &normalized[(size_t)query * stride + head * 96];
        std::vector<float> e(padded), weight(padded);
        for (uint32_t key = 0; key < padded; ++key) {
            const float* k = &normalized[(size_t)key * stride + head * 96 + 32];
            float score;
            if (arithmetic == Arithmetic::Fp8) {
                score = Dot(arithmetic, q, k, 32, 0.0f, 0);
            } else {
                float sum = 0.0f;
                for (int c = 0; c < 32; ++c) {
                    sum += q[c] * k[c];
                }
                score = Num::RoundF16(sum);
            }
            e[key] = Num::VitExpWeight(score);
            weight[key] = Num::Fp8Domain(e[key]);
        }
        // The window tree per block of 64 keys, the blocks added in order; then the padding rows' exp(0) leaves.
        float total = 0.0f;
        for (uint32_t base = 0; base < padded; base += 64) {
            auto pair = [&](uint32_t pairIndex, uint32_t parity) {
                const float* s = &e[base + pairIndex * 2 + parity];
                return Num::AddF16(Num::AddF16(Num::AddF16(Num::AddF16(s[0], s[8]), Num::AddF16(s[16], s[24])),
                                               Num::AddF16(s[32], s[40])),
                                   Num::AddF16(s[48], s[56]));
            };
            const float even = Num::AddF16(Num::AddF16(Num::AddF16(pair(0, 0), pair(1, 0)), pair(2, 0)), pair(3, 0));
            const float odd = Num::AddF16(Num::AddF16(Num::AddF16(pair(0, 1), pair(1, 1)), pair(2, 1)), pair(3, 1));
            total = Num::AddF16(total, Num::AddF16(even, odd));
        }
        if (padded > tokens) {
            total = Num::RoundF16(total - Num::RoundF16(Num::VitExpWeight(0.0f) * (float)(padded - tokens)));
        }
        const float reciprocal = Num::RoundF16(1.0f / total);
        float* out = &attended[(size_t)query * channels + head * 32];
        std::vector<float> column(padded);
        for (int c = 0; c < 32; ++c) {
            for (uint32_t key = 0; key < padded; ++key) {
                column[key] = normalized[(size_t)key * stride + head * 96 + 64 + c];
            }
            float value = 0.0f;
            if (arithmetic == Arithmetic::Fp8) {
                value = Dot(arithmetic, weight.data(), column.data(), padded, 0.0f, 0);
            } else {
                float sum = 0.0f;
                for (uint32_t key = 0; key < padded; ++key) {
                    sum += weight[key] * column[key];
                }
                value = Num::RoundF16(sum);
            }
            out[c] = Num::Fp8Domain(Num::RoundF16(value * reciprocal));
        }
    });
    return attended;
}

// 2x2 box pool of the raw half output, ((a + b) + (c + d)) * 0.25 with every step a half, then E4M3. Output pixels
// whose quad is not inside the input (a level rounded up to 4) are zero.
Tokens Pool(const Tokens& raw, uint32_t inWidth, uint32_t inHeight, uint32_t outWidth, uint32_t outHeight,
            uint32_t channels) {
    Tokens out((size_t)outWidth * outHeight * channels, 0.0f);
    ParallelFor(outHeight, [&](size_t oy) {
        for (uint32_t ox = 0; ox < outWidth; ++ox) {
            const uint32_t sx = ox * 2, sy = (uint32_t)oy * 2;
            if (sx + 1 >= inWidth || sy + 1 >= inHeight) {
                continue;
            }
            const float* a = &raw[((size_t)sy * inWidth + sx) * channels];
            const float* c = &raw[((size_t)(sy + 1) * inWidth + sx) * channels];
            for (uint32_t ch = 0; ch < channels; ++ch) {
                const float sum =
                    Num::AddF16(Num::AddF16(a[ch], a[channels + ch]), Num::AddF16(c[ch], c[channels + ch]));
                out[(oy * outWidth + ox) * channels + ch] = Num::Fp8Domain(Num::RoundF16(sum * 0.25f));
            }
        }
    });
    return out;
}

// Decoder merge: nearest 2x upsample of the low-level projection + skip * scale, as one FMA, then a half.
Tokens UpsampleMerge(const Tokens& projection, const Tokens& skip, const std::vector<float>& scale, uint32_t lowWidth,
                     uint32_t width, uint32_t height, uint32_t channels) {
    Tokens raw((size_t)width * height * channels);
    ParallelFor(height, [&](size_t y) {
        for (uint32_t x = 0; x < width; ++x) {
            const float* low = &projection[((y >> 1) * lowWidth + (x >> 1)) * channels];
            const float* high = &skip[(y * width + x) * channels];
            for (uint32_t c = 0; c < channels; ++c) {
                raw[(y * width + x) * channels + c] = Num::RoundF16(std::fma(high[c], scale[c], low[c]));
            }
        }
    });
    return raw;
}

struct BlockOutput {
    Tokens mE4, mRaw;
};

} // namespace

struct Reference::Impl {
    const Model& mModel;
    const Geometry& mGeometry;
    Arithmetic mArithmetic;
    mutable uint32_t mPhase[7] = {};                             // levels 0-5, and 6 for the field
    std::map<std::string, std::vector<float>>* mDebug = nullptr; // block 0's intermediates, when asked for

    // FFN -> QKV -> window attention -> projection, with the two scaled skips. `ffnSkip` overrides the FFN's skip
    // source with a raw half tensor (blocks 0, 66 and 70).
    BlockOutput WindowBlock(const Tokens& state, const Tokens* ffnSkip, const BlockLayout& l, const Tensor& t,
                            int level) const {
        const uint32_t C = l.mChannels;
        const uint32_t width = level == 6 ? mGeometry.mFullWidth : mGeometry.mWidth[level];
        const uint32_t height = level == 6 ? mGeometry.mFullHeight : mGeometry.mHeight[level];
        const uint32_t rows = width * height;
        const uint32_t phase = mPhase[level]++;
        const std::vector<float> ffnScale = mModel.Halves(t, l.mFfnScale, C),
                                 attnScale = mModel.Halves(t, l.mAttnScale, C);
        const Tokens& skip = ffnSkip ? *ffnSkip : state;
        Tokens ffnRaw((size_t)rows * C), ffnE4((size_t)rows * C);

        if (!l.mExperts) { // dense 32 -> 128 -> 32, SiLU after the first layer
            const std::vector<float> w1 = mModel.Fp8Matrix(t, l.mExpand, 32, 128),
                                     w2 = mModel.Fp8Matrix(t, l.mContract, 128, 32);
            Tokens hidden((size_t)rows * 128);
            Gemm expand;
            expand.mInput = &state;
            expand.mInStride = 32;
            expand.mRows = rows;
            expand.mK = 32;
            expand.mN = 128;
            expand.mWeights = w1.data();
            expand.mSilu = true;
            expand.mQuantized = &hidden;
            expand.mOutStride = 128;
            RunGemm(mArithmetic, expand);
            Gemm contract;
            contract.mInput = &hidden;
            contract.mInStride = 128;
            contract.mRows = rows;
            contract.mK = 128;
            contract.mN = 32;
            contract.mWeights = w2.data();
            contract.mResidual = &skip;
            contract.mResStride = 32;
            contract.mScale = ffnScale.data();
            contract.mRaw = &ffnRaw;
            contract.mQuantized = &ffnE4;
            contract.mOutStride = 32;
            RunGemm(mArithmetic, contract);
            if (mDebug && level == 6 && phase == 0) {
                (*mDebug)["transition-b0-hidden"] = hidden;
                (*mDebug)["transition-b0-ffn"] = ffnE4;
            }
        } else { // C / 32 paths C -> 128 -> 32 (every path sees the whole state), concatenated, then W3 with the skip
            const uint32_t E = l.mExperts;
            const std::vector<float> w1 = mModel.Fp8Matrix(t, l.mExpand, E * C, 128);
            const std::vector<float> w2 = mModel.Fp8Matrix(t, l.mContract, E * 128, 32),
                                     w3 = mModel.Fp8Matrix(t, l.mW3, C, C);
            Tokens hidden((size_t)rows * E * 128), narrow((size_t)rows * C);
            for (uint32_t e = 0; e < E; ++e) {
                Gemm expand;
                expand.mInput = &state;
                expand.mInStride = C;
                expand.mRows = rows;
                expand.mK = C;
                expand.mN = 128;
                expand.mWeights = &w1[(size_t)e * C * 128];
                expand.mSilu = true;
                expand.mQuantized = &hidden;
                expand.mOutStride = E * 128;
                expand.mOutCol = e * 128;
                RunGemm(mArithmetic, expand);
                Gemm contract;
                contract.mInput = &hidden;
                contract.mInStride = E * 128;
                contract.mInCol = e * 128;
                contract.mRows = rows;
                contract.mK = 128;
                contract.mN = 32;
                contract.mWeights = &w2[(size_t)e * 128 * 32];
                contract.mQuantized = &narrow;
                contract.mOutStride = C;
                contract.mOutCol = e * 32;
                RunGemm(mArithmetic, contract);
            }
            Gemm mix;
            mix.mInput = &narrow;
            mix.mInStride = C;
            mix.mRows = rows;
            mix.mK = C;
            mix.mN = C;
            mix.mWeights = w3.data();
            mix.mResidual = &skip;
            mix.mResStride = C;
            mix.mScale = ffnScale.data();
            mix.mRaw = &ffnRaw;
            mix.mQuantized = &ffnE4;
            mix.mOutStride = C;
            RunGemm(mArithmetic, mix);
        }

        const std::vector<float> wqkv = mModel.Fp8Matrix(t, l.mQkv, C, 3 * C);
        Tokens qkv((size_t)rows * 3 * C);
        Gemm project;
        project.mInput = &ffnE4;
        project.mInStride = C;
        project.mRows = rows;
        project.mK = C;
        project.mN = 3 * C;
        project.mWeights = wqkv.data();
        project.mRaw = &qkv;
        project.mOutStride = 3 * C;
        RunGemm(mArithmetic, project);
        const Tokens attended =
            WindowAttention(mArithmetic, qkv, width, height, C, phase, mModel.Prior(t, l.mPrior, l.mHeads),
                            mModel.Floats(t, l.mQScale, l.mHeads));
        if (mDebug && level == 6 && phase == 0) {
            (*mDebug)["transition-b0-attended"] = attended;
        }

        // The 32-channel blocks add the raw half FFN result as the attention skip, the wider ones its E4M3 publication.
        const std::vector<float> wproj = mModel.Fp8Matrix(t, l.mProjection, C, C);
        BlockOutput out{ Tokens((size_t)rows * C), Tokens((size_t)rows * C) };
        Gemm projection;
        projection.mInput = &attended;
        projection.mInStride = C;
        projection.mRows = rows;
        projection.mK = C;
        projection.mN = C;
        projection.mWeights = wproj.data();
        projection.mResidual = l.mExperts ? &ffnE4 : &ffnRaw;
        projection.mResStride = C;
        projection.mScale = attnScale.data();
        projection.mRaw = &out.mRaw;
        projection.mQuantized = &out.mE4;
        projection.mOutStride = C;
        RunGemm(mArithmetic, projection);
        return out;
    }

    // The 512 stage: one 512 -> 512 layer, eight 64 -> 256 -> 64 branches (SiLU on the 256-wide middle only), one
    // 512 -> 512 contraction carrying the skip; then 16-head window attention. Four tensor records.
    BlockOutput SplitBlock(const Tokens& state, int block) const {
        const uint32_t C = 512, width = mGeometry.mWidth[4], height = mGeometry.mHeight[4], rows = width * height;
        const uint32_t phase = mPhase[4]++;
        const Tensor& branches = mModel.Get(block, 0);
        const Tensor& contract = mModel.Get(block, 1);
        const Tensor& attention = mModel.Get(block, 2);
        const Tensor& projectionTensor = mModel.Get(block, 3);
        const std::vector<float> w1 = mModel.Fp8Matrix(branches, 0, 512, 512);
        const std::vector<float> w2 = mModel.Fp8Matrix(branches, 512 * 512, 8 * 64, 256);
        const std::vector<float> w3 = mModel.Fp8Matrix(branches, 512 * 512 + 8 * 64 * 256, 8 * 256, 64);
        Tokens branch((size_t)rows * 512), middle((size_t)rows * 2048), layer0((size_t)rows * 512),
            ffnE4((size_t)rows * 512);
        Gemm first;
        first.mInput = &state;
        first.mInStride = 512;
        first.mRows = rows;
        first.mK = 512;
        first.mN = 512;
        first.mWeights = w1.data();
        first.mQuantized = &branch;
        first.mOutStride = 512;
        RunGemm(mArithmetic, first);
        for (uint32_t b = 0; b < 8; ++b) {
            Gemm up;
            up.mInput = &branch;
            up.mInStride = 512;
            up.mInCol = b * 64;
            up.mRows = rows;
            up.mK = 64;
            up.mN = 256;
            up.mWeights = &w2[(size_t)b * 64 * 256];
            up.mSilu = true;
            up.mQuantized = &middle;
            up.mOutStride = 2048;
            up.mOutCol = b * 256;
            RunGemm(mArithmetic, up);
            Gemm down;
            down.mInput = &middle;
            down.mInStride = 2048;
            down.mInCol = b * 256;
            down.mRows = rows;
            down.mK = 256;
            down.mN = 64;
            down.mWeights = &w3[(size_t)b * 256 * 64];
            down.mQuantized = &layer0;
            down.mOutStride = 512;
            down.mOutCol = b * 64;
            RunGemm(mArithmetic, down);
        }
        const std::vector<float> wc = mModel.Fp8Matrix(contract, 0, 512, 512),
                                 ffnScale = mModel.Halves(contract, 512 * 512, 512);
        Gemm mix;
        mix.mInput = &layer0;
        mix.mInStride = 512;
        mix.mRows = rows;
        mix.mK = 512;
        mix.mN = 512;
        mix.mWeights = wc.data();
        mix.mResidual = &state;
        mix.mResStride = 512;
        mix.mScale = ffnScale.data();
        mix.mQuantized = &ffnE4;
        mix.mOutStride = 512;
        RunGemm(mArithmetic, mix);

        const std::vector<float> wqkv = mModel.Fp8Matrix(attention, 0, 512, 1536);
        Tokens qkv((size_t)rows * 1536);
        Gemm project;
        project.mInput = &ffnE4;
        project.mInStride = 512;
        project.mRows = rows;
        project.mK = 512;
        project.mN = 1536;
        project.mWeights = wqkv.data();
        project.mRaw = &qkv;
        project.mOutStride = 1536;
        RunGemm(mArithmetic, project);
        const Tokens attended =
            WindowAttention(mArithmetic, qkv, width, height, 512, phase, mModel.Prior(attention, 512 * 1536, 16),
                            mModel.Floats(attention, 512 * 1536 + 16 * 8192, 16));
        const std::vector<float> wproj = mModel.Fp8Matrix(projectionTensor, 0, 512, 512);
        const std::vector<float> attnScale = mModel.Halves(projectionTensor, 512 * 512, 512);
        BlockOutput out{ Tokens((size_t)rows * 512), Tokens((size_t)rows * 512) };
        Gemm projection;
        projection.mInput = &attended;
        projection.mInStride = 512;
        projection.mRows = rows;
        projection.mK = 512;
        projection.mN = 512;
        projection.mWeights = wproj.data();
        projection.mResidual = &ffnE4;
        projection.mResStride = 512;
        projection.mScale = attnScale.data();
        projection.mRaw = &out.mRaw;
        projection.mQuantized = &out.mE4;
        projection.mOutStride = 512;
        RunGemm(mArithmetic, projection);
        return out;
    }

    // A ViT block: dense 1024 -> 4096 -> 1024 FFN, global attention, projection. Its GEMMs are split-K on the
    // tensor cores (contract 1024, qkv 512, projection 256), which is part of the Fp8 result.
    Tokens VitBlock(const Tokens& state, int block) const {
        const uint32_t tokens = mGeometry.Tokens(5);
        const Tensor& expandTensor = mModel.Get(block, 0);
        const Tensor& contractTensor = mModel.Get(block, 1);
        const Tensor& attention = mModel.Get(block, 2); // the per-head scales come first in this one
        const Tensor& projectionTensor = mModel.Get(block, 4);
        const std::vector<float> w1 = mModel.Fp8Matrix(expandTensor, 0, 1024, 4096);
        Tokens expanded((size_t)tokens * 4096), ffnE4((size_t)tokens * 1024), qkv((size_t)tokens * 3072);
        Gemm expand;
        expand.mInput = &state;
        expand.mInStride = 1024;
        expand.mRows = tokens;
        expand.mK = 1024;
        expand.mN = 4096;
        expand.mWeights = w1.data();
        expand.mSilu = true;
        expand.mQuantized = &expanded;
        expand.mOutStride = 4096;
        RunGemm(mArithmetic, expand);
        const std::vector<float> w2 = mModel.Fp8Matrix(contractTensor, 0, 4096, 1024);
        const std::vector<float> ffnScale = mModel.Halves(contractTensor, 4096 * 1024, 1024);
        Gemm contract;
        contract.mInput = &expanded;
        contract.mInStride = 4096;
        contract.mRows = tokens;
        contract.mK = 4096;
        contract.mN = 1024;
        contract.mWeights = w2.data();
        contract.mPartition = 1024;
        contract.mResidual = &state;
        contract.mResStride = 1024;
        contract.mScale = ffnScale.data();
        contract.mQuantized = &ffnE4;
        contract.mOutStride = 1024;
        RunGemm(mArithmetic, contract);
        const std::vector<float> wqkv = mModel.Fp8Matrix(attention, 32 * 4, 1024, 3072);
        Gemm project;
        project.mInput = &ffnE4;
        project.mInStride = 1024;
        project.mRows = tokens;
        project.mK = 1024;
        project.mN = 3072;
        project.mWeights = wqkv.data();
        project.mPartition = 512;
        project.mRaw = &qkv;
        project.mOutStride = 3072;
        RunGemm(mArithmetic, project);
        const Tokens attended =
            GlobalAttention(mArithmetic, qkv, tokens, mGeometry.PaddedVitTokens(), mModel.Floats(attention, 0, 32));
        const std::vector<float> wproj = mModel.Fp8Matrix(projectionTensor, 0, 1024, 1024);
        const std::vector<float> attnScale = mModel.Halves(projectionTensor, 1024 * 1024, 1024);
        Tokens out((size_t)tokens * 1024);
        Gemm projection;
        projection.mInput = &attended;
        projection.mInStride = 1024;
        projection.mRows = tokens;
        projection.mK = 1024;
        projection.mN = 1024;
        projection.mWeights = wproj.data();
        projection.mPartition = 256;
        projection.mResidual = &ffnE4;
        projection.mResStride = 1024;
        projection.mScale = attnScale.data();
        projection.mQuantized = &out;
        projection.mOutStride = 1024;
        RunGemm(mArithmetic, projection);
        return out;
    }
};

Reference::Reference(const Model& model, const Geometry& geometry, Arithmetic arithmetic)
    : mModel(model), mGeometry(geometry), mArithmetic(arithmetic) {
}

std::vector<float> Reference::Run(const std::vector<float>& features,
                                  std::map<std::string, std::vector<float>>* boundaries, int lastBlock) const {
    const Geometry& g = mGeometry;
    const uint32_t field = g.FieldTokens();
    if (features.size() != (size_t)field * 16) {
        throw std::runtime_error("reference: features must be [field][16]");
    }
    Impl impl{ mModel, mGeometry, mArithmetic };
    if (mCaptureBlock0 && boundaries) {
        impl.mDebug = boundaries;
    }
    auto keep = [&](const std::string& name, const Tokens& tensor) {
        if (boundaries) {
            (*boundaries)[name] = tensor;
        }
    };
    auto done = [&](int block) { return block >= lastBlock; };

    // Block 0 at full resolution: f32 features -> halves -> the f16 16 -> 32 input adapter. Its half output is the
    // FFN's skip, its E4M3 publication the block's state.
    const Tensor& pre = mModel.Get(0);
    const BlockLayout preLayout = PreLayout();
    Tokens half(features.size());
    for (size_t i = 0; i < features.size(); ++i) {
        half[i] = Num::RoundF16(features[i]);
    }
    const Tokens adapterRaw =
        GemmF16(mArithmetic, half, field, 16, 32, mModel.F16Matrix(pre, preLayout.mInputAdapter, 16, 32));
    if (impl.mDebug) {
        (*impl.mDebug)["transition-b0-adapter"] = Quantized(adapterRaw);
    }
    const BlockOutput block0 = impl.WindowBlock(Quantized(adapterRaw), &adapterRaw, preLayout, pre, 6);
    keep("block-0", block0.mE4);
    if (done(0)) {
        return {};
    }

    // Encoder. Each stage ends with a 2x2 pool of the last block's raw output and a C -> 2C GEMM into the next.
    Tokens skips[5];
    Tokens state = Pool(block0.mRaw, g.mFullWidth, g.mFullHeight, g.mWidth[0], g.mHeight[0], 32);
    keep("transition-0-1", state);
    for (const Stage& stage : kEncoderStages) {
        const uint32_t C = stage.mChannels;
        Tokens raw;
        for (int block = stage.mFirst; block <= stage.mLast; ++block) {
            BlockOutput out =
                C == 512 ? impl.SplitBlock(state, block)
                         : impl.WindowBlock(state, nullptr, StandardLayout(C), mModel.Get(block), stage.mLevel);
            state = std::move(out.mE4);
            raw = std::move(out.mRaw);
            keep("block-" + std::to_string(block), state);
            if (done(block)) {
                return {};
            }
        }
        skips[stage.mLevel] = state;
        const int next = stage.mLevel + 1;
        const Tokens pooled =
            Pool(raw, g.mWidth[stage.mLevel], g.mHeight[stage.mLevel], g.mWidth[next], g.mHeight[next], C);
        const std::vector<float> w = C == 512
                                         ? mModel.Fp8Matrix(mModel.Get(30, 4), 0, 512, 1024)
                                         : mModel.Fp8Matrix(mModel.Get(stage.mLast), StandardLayout(C).mEnd, C, 2 * C);
        Tokens widened((size_t)g.Tokens(next) * 2 * C);
        Gemm widen;
        widen.mInput = &pooled;
        widen.mInStride = C;
        widen.mRows = g.Tokens(next);
        widen.mK = C;
        widen.mN = 2 * C;
        widen.mWeights = w.data();
        widen.mQuantized = &widened;
        widen.mOutStride = 2 * C;
        RunGemm(mArithmetic, widen);
        state = std::move(widened);
        keep("transition-" + std::to_string(stage.mLast) + "-" + std::to_string(stage.mLast + 1), state);
    }

    // ViT.
    for (int block = 31; block <= 38; ++block) {
        state = impl.VitBlock(state, block);
        keep("block-" + std::to_string(block), state);
        if (done(block)) {
            return {};
        }
    }

    // Decoder. Each stage starts with a 2C -> C GEMM at the low level (half output), a nearest 2x upsample and the
    // scaled encoder skip. "Block 39" is that merge for the 512 stage and nothing else.
    Tokens stageRaw;
    for (const Stage& stage : kDecoderStages) {
        const uint32_t C = stage.mChannels;
        const int low = stage.mLevel + 1;
        const BlockLayout first = C == 512 ? BlockLayout{} : UpsampleLayout(C);
        const Tensor& transition = mModel.Get(C == 512 ? 39 : stage.mFirst);
        const std::vector<float> w = mModel.Fp8Matrix(transition, C == 512 ? 0 : first.mUpsample, 2 * C, C);
        Tokens projection((size_t)g.Tokens(low) * C);
        Gemm narrow;
        narrow.mInput = &state;
        narrow.mInStride = 2 * C;
        narrow.mRows = g.Tokens(low);
        narrow.mK = 2 * C;
        narrow.mN = C;
        narrow.mWeights = w.data();
        narrow.mPartition = C == 512 ? 256 : 0;
        narrow.mRaw = &projection;
        narrow.mOutStride = C;
        RunGemm(mArithmetic, narrow);
        const std::vector<float> scale = mModel.Halves(transition, C == 512 ? 1024 * 512 : first.mTransitionScale, C);
        const Tokens mergedRaw = UpsampleMerge(projection, skips[stage.mLevel], scale, g.mWidth[low],
                                               g.mWidth[stage.mLevel], g.mHeight[stage.mLevel], C);
        state = Quantized(mergedRaw);
        keep("merge-" + std::to_string(stage.mFirst), state);
        if (C == 512) {
            keep("block-39", state);
            if (done(39)) {
                return {};
            }
        }
        for (int block = stage.mFirst; block <= stage.mLast; ++block) {
            BlockOutput out;
            if (C == 512) {
                out = impl.SplitBlock(state, block);
            } else {
                // The first 32-channel decoder block takes the merge's half value as its FFN skip.
                const bool head = block == stage.mFirst;
                out = impl.WindowBlock(state, head && C == 32 ? &mergedRaw : nullptr, head ? first : StandardLayout(C),
                                       mModel.Get(block), stage.mLevel);
            }
            state = std::move(out.mE4);
            keep("block-" + std::to_string(block), state);
            if (done(block)) {
                return {};
            }
        }
    }

    // Block 70 at full resolution: the post blend of the upsampled block 69 and block 0's output, then the head.
    const Tensor& post = mModel.Get(70);
    const BlockLayout postLayout = PostLayout();
    const std::vector<float> inputScale = mModel.Halves(post, postLayout.mInputScale, 32);
    const std::vector<float> adapterScale = mModel.Halves(post, postLayout.mAdapterScale, 32);
    Tokens blendRaw((size_t)field * 32);
    ParallelFor(g.mFullHeight, [&](size_t y) {
        for (uint32_t x = 0; x < g.mFullWidth; ++x) {
            const float* up = &state[((y >> 1) * g.mWidth[0] + (x >> 1)) * 32];
            const float* adapter = &block0.mE4[(y * g.mFullWidth + x) * 32];
            for (uint32_t c = 0; c < 32; ++c) {
                const float scaled = Num::RoundF16(up[c] * inputScale[c]);
                blendRaw[(y * g.mFullWidth + x) * 32 + c] =
                    Num::RoundF16(std::fma(adapter[c], adapterScale[c], scaled));
            }
        }
    });
    keep("merge-70", Quantized(blendRaw));
    const BlockOutput block70 = impl.WindowBlock(Quantized(blendRaw), &blendRaw, postLayout, post, 6);
    keep("block-70-raw", block70.mRaw);
    return GemmF16(mArithmetic, block70.mRaw, field, 32, 4, mModel.F16Matrix(post, postLayout.mHead, 32, 4));
}

std::vector<float> Reference::Features(const Geometry& g, const std::vector<float>& proxyRgba,
                                       const Conditioning& conditioning) {
    if (proxyRgba.size() != (size_t)g.mValidWidth * g.mValidHeight * 4) {
        throw std::runtime_error("reference: the proxy must be RGBA f32 at the valid size");
    }
    auto hashUniform = [](uint32_t value) {
        uint32_t mixed = value;
        mixed = (mixed >> ((mixed >> 28) + 4)) ^ mixed;
        mixed *= 0x108ef2d9u;
        const uint32_t integer = ((mixed >> 30) ^ (mixed >> 8)) + 1u;
        return (float)integer * Num::F32FromBits(0x33800000u);
    };
    std::vector<float> features((size_t)g.FieldTokens() * 16);
    for (uint32_t y = 0; y < g.mFullHeight; ++y) {
        for (uint32_t x = 0; x < g.mFullWidth; ++x) {
            // Outside the valid rectangle the image is mirrored (no edge repeat); the noise hashes the padded
            // coordinate.
            const uint32_t sx = x < g.mValidWidth ? x : 2 * g.mValidWidth - x - 2;
            const uint32_t sy = y < g.mValidHeight ? y : 2 * g.mValidHeight - y - 2;
            const float* rgba = &proxyRgba[((size_t)sy * g.mValidWidth + sx) * 4];
            uint32_t base = (x * 0x8da6b343u) ^ (conditioning.mSeed * 0x9e3779b9u) ^ (y * 0xd8163841u) ^ 0x243f6a88u;
            base = (base >> ((base >> 28) + 4)) ^ base;
            base *= 0x108ef2d9u;
            base = (base >> 22) ^ base;
            const float u0 = hashUniform(base * 0x2c9277b5u + 0xac564b05u),
                        u1 = hashUniform(base * 0xfa6dc5f9u + 0x4712a88eu);
            const float u2 = hashUniform(base * 0xcaa5b80du + 0x21dd796bu),
                        u3 = hashUniform(base * 0x83232c31u + 0x3463e0acu);
            const float ln2 = Num::F32FromBits(0x3f317218u), tau = Num::F32FromBits(0x40c90fdbu);
            const float radius0 = std::sqrt(std::log2(u0) * ln2 * -2.0f),
                        radius1 = std::sqrt(std::log2(u2) * ln2 * -2.0f);
            float* out = &features[((size_t)y * g.mFullWidth + x) * 16];
            out[0] = Num::RoundF16(radius0 * std::cos(u1 * tau));
            out[1] = Num::RoundF16(radius0 * std::sin(u1 * tau));
            out[2] = Num::RoundF16(radius1 * std::cos(u3 * tau));
            out[3] = 1.0f;
            for (int c = 0; c < 3; ++c) {
                out[4 + c] = out[7 + c] = Num::RoundF16(Num::RoundF16(Num::RoundF16(rgba[c]) - 0.5f) * 0.125f);
            }
            const float structure = conditioning.mLocalStructure;
            out[10] = conditioning.mStyle / 128.0f;
            out[11] = Num::RoundF16(conditioning.mLocalTone);
            out[12] = Num::RoundF16(conditioning.mAutoMask ? 1.0f : structure);
            out[13] = Num::RoundF16(conditioning.mAutoMask
                                        ? (conditioning.mSkinStructure < 0.0f ? structure : conditioning.mSkinStructure)
                                        : -1.0f);
            out[14] = Num::RoundF16(conditioning.mAutoMask ? structure : -1.0f);
            out[15] = 0.0f;
        }
    }
    return features;
}

} // namespace Fast::Neural
