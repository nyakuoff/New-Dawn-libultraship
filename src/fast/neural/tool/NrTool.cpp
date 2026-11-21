// nd-nr: the standalone runner of New Dawn's neural rendering network.
//
//   nd-nr devices                                    the Vulkan devices and what each lacks, if anything
//   nd-nr extract <nvngx_dlssnr.dll> <model dir>     build the model directory from the player's own file
//   nd-nr synth <model dir> [seed]                   a stand-in model: plausible random weights, real layouts
//   nd-nr reference <model dir> <width> <height> [--fp8] [--last N]
//                                                    the network on the CPU (slow), statistics per boundary
//   nd-nr verify <model dir> <width> <height> [--fp8] [--image file.ppm]
//                                                    the GPU network against the CPU reference, step by step
//   nd-nr bench <model dir> <width> <height> [--runs N] [--no-tune] [--reuse-vit]
//                                                    milliseconds per frame and per stage on this GPU
//   nd-nr fixture <model dir> <width> <height> <out dir> [--image file.ppm] [--bisect]
//                                                    the FP8 CPU reference's frame as an OpenDLSS-NR parity fixture,
//                                                    so that project's own checker can confirm it bit for bit
//   nd-nr sensitivity <model dir> <width> <height> [--image file.ppm]
//                                                    how far a one-ulp input change moves the output (CPU only)
//   nd-nr selftest                                   the arithmetic the kernels rely on, on the CPU
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>

#include "NrGraph.h"
#include "NrModel.h"
#include "NrNumeric.h"
#include "NrReference.h"

using namespace Fast::Neural;

namespace Fast::Neural {
void SynthesizeModel(const std::filesystem::path& directory, uint32_t seed);
}

namespace {

// A deterministic test image: smooth gradients with some structure, sRGB code values in 0..1.
std::vector<float> TestProxy(uint32_t width, uint32_t height) {
    std::vector<float> rgba((size_t)width * height * 4);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const float u = (float)x / (float)width, v = (float)y / (float)height;
            float* p = &rgba[((size_t)y * width + x) * 4];
            p[0] = 0.5f + 0.4f * std::sin(9.0f * u + 3.0f * v);
            p[1] = 0.5f + 0.4f * std::sin(5.0f * v - 2.0f * u * u);
            p[2] = 0.25f + 0.5f * u * v + (((x / 16) + (y / 16)) & 1 ? 0.1f : 0.0f);
            p[3] = 1.0f;
        }
    }
    return rgba;
}

// A binary PPM (P6, 8 bits) as RGBA f32 code values, box-resampled to `width` x `height`.
std::vector<float> LoadPpm(const std::string& path, uint32_t width, uint32_t height) {
    std::ifstream file(path, std::ios::binary);
    std::string magic;
    uint32_t w = 0, h = 0, peak = 0;
    file >> magic >> w >> h >> peak;
    file.get();
    if (!file || magic != "P6" || peak != 255 || !w || !h) {
        throw std::runtime_error("cannot read " + path + " (expected a binary 8-bit PPM)");
    }
    std::vector<uint8_t> pixels((size_t)w * h * 3);
    file.read((char*)pixels.data(), (std::streamsize)pixels.size());
    std::vector<float> rgba((size_t)width * height * 4, 1.0f);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const uint32_t x0 = x * w / width, x1 = std::max(x0 + 1, (x + 1) * w / width);
            const uint32_t y0 = y * h / height, y1 = std::max(y0 + 1, (y + 1) * h / height);
            double sum[3] = {};
            for (uint32_t sy = y0; sy < y1; ++sy) {
                for (uint32_t sx = x0; sx < x1; ++sx) {
                    for (int c = 0; c < 3; ++c) {
                        sum[c] += pixels[((size_t)sy * w + sx) * 3 + c];
                    }
                }
            }
            for (int c = 0; c < 3; ++c) {
                rgba[((size_t)y * width + x) * 4 + c] = (float)(sum[c] / (255.0 * (x1 - x0) * (y1 - y0)));
            }
        }
    }
    return rgba;
}

void SavePpm(const std::string& path, const std::vector<float>& rgb, uint32_t width, uint32_t height) {
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    for (float v : rgb) {
        file.put((char)(uint8_t)std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
    }
}

void Describe(const std::string& name, const std::vector<float>& tensor) {
    double sum = 0, squares = 0;
    size_t zeros = 0, saturated = 0, nonFinite = 0;
    for (float v : tensor) {
        if (!std::isfinite(v)) {
            nonFinite += 1;
            continue;
        }
        sum += v;
        squares += (double)v * v;
        zeros += v == 0.0f;
        saturated += std::fabs(v) >= 448.0f;
    }
    printf("  %-18s %9zu values  mean %+.4f  rms %.4f  zero %.1f%%  saturated %.2f%%  non-finite %zu\n", name.c_str(),
           tensor.size(), sum / tensor.size(), std::sqrt(squares / tensor.size()), 100.0 * zeros / tensor.size(),
           100.0 * saturated / tensor.size(), nonFinite);
}

struct Difference {
    double mSnrDb = 0.0;     // signal power of the reference over the power of the difference
    double mDiffering = 0.0; // share of values that are not identical
};

Difference Compare(const std::vector<float>& reference, const std::vector<float>& value) {
    double signal = 0, error = 0;
    size_t differing = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double d = (double)reference[i] - value[i];
        signal += (double)reference[i] * reference[i];
        error += d * d;
        differing += reference[i] != value[i];
    }
    return { error == 0 ? 999.0 : 10.0 * std::log10(signal / error), (double)differing / (double)reference.size() };
}

// The displayed image of a head: clamp(proxy + rgb / 4, 0, 1) in the proxy's code space, over the valid rectangle.
std::vector<float> Compose(const Geometry& g, const std::vector<float>& proxy, const std::vector<float>& head) {
    std::vector<float> rgb((size_t)g.mValidWidth * g.mValidHeight * 3);
    for (uint32_t y = 0; y < g.mValidHeight; ++y) {
        for (uint32_t x = 0; x < g.mValidWidth; ++x) {
            for (int c = 0; c < 3; ++c) {
                rgb[((size_t)y * g.mValidWidth + x) * 3 + c] =
                    std::clamp(proxy[((size_t)y * g.mValidWidth + x) * 4 + c] +
                                   head[((size_t)y * g.mFullWidth + x) * 4 + c] * 0.25f,
                               0.0f, 1.0f);
            }
        }
    }
    return rgb;
}

double PsnrDb(const std::vector<float>& a, const std::vector<float>& b) {
    double error = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        error += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
    }
    return error == 0 ? 999.0 : 10.0 * std::log10((double)a.size() / error);
}

int SelfTest() {
    // quantE4 of nr_common.glsl against the reference E4M3 conversion, for every non-NaN half.
    size_t bad = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        if ((h & 0x7fff) > 0x7c00) {
            continue;
        }
        const float mag = std::min(std::fabs(Num::F16ToF32((uint16_t)h)), 448.0f);
        const float lift = Num::F16ToF32((uint16_t)((Num::F16Bits(std::max(mag, 0.015625f)) & 0x7c00) + 0x1e00));
        const uint16_t result = Num::F16Bits(Num::RoundF16(mag + lift) - lift) | (h & 0x8000);
        const uint8_t code = Num::E4m3FromF16Bits((uint16_t)h);
        const uint16_t expect = Num::F16Bits(Num::E4m3ToF32(code & 0x7f)) | (code & 0x80 ? 0x8000 : 0);
        bad += result != expect;
    }
    printf("E4M3 publication by lift-and-drop, every non-NaN half: %zu mismatches\n", bad);
    // The packed exponentials against the reference's bit tricks.
    size_t badExp = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        if ((h & 0x7c00) == 0x7c00) {
            continue;
        }
        const float s = Num::F16ToF32((uint16_t)h);
        float affine = std::clamp(Num::RoundF16(std::fma(s, 0.044921875f, 1.30078125f)), 1.03125f, 1.5693359375f);
        badExp += (uint16_t)(((Num::F16Bits(affine) & 0x07ff) << 5) ^ 0x8000) != Num::F16Bits(Num::WindowExpWeight(s));
        const float slope = Num::RoundF16(0.08953946828842163f), offset = Num::RoundF16(1.7093614339828491f);
        affine = std::clamp(Num::RoundF16((float)((double)s * slope + offset)), 1.439453125f, 1.9775390625f);
        badExp += (uint16_t)((Num::F16Bits(affine) & 0x03ff) << 4) != Num::F16Bits(Num::VitExpWeight(s));
    }
    printf("packed window / ViT exponentials, every finite half: %zu mismatches\n", badExp);
    return bad || badExp;
}

// Stage of a segment for the timing table.
std::string StageOf(const Graph::Segment& segment) {
    int block = -1;
    if (sscanf(segment.mLabel.c_str(), "block %d", &block) != 1) {
        return "transitions";
    }
    if (block == 0 || block == 70)
        return "full resolution, 32 ch (0, 70)";
    if (block <= 4 || block >= 66)
        return "level 0, 32 ch (1-4, 66-69)";
    if (block <= 8 || block >= 62)
        return "level 1, 64 ch (5-8, 62-65)";
    if (block <= 14 || block >= 56)
        return "level 2, 128 ch (9-14, 56-61)";
    if (block <= 22 || block >= 48)
        return "level 3, 256 ch (15-22, 48-55)";
    if (block <= 30 || block >= 40)
        return "level 4, 512 ch (23-30, 40-47)";
    return "level 5, ViT (31-38)";
}

int Bench(const std::string& modelPath, uint32_t width, uint32_t height, int runs, bool tune, bool reuseVit) {
    const Model model(modelPath);
    const Geometry g = Geometry::FromValid(width, height);
    Context context(getenv("NR_GPU"));
    Graph graph(context, model, g);
    graph.SetFeatures(Reference::Features(g, TestProxy(width, height), {}));
    printf(
        "%s\n%ux%u -> field %ux%u, level 5 %ux%u; %.1f GMAC, %zu dispatches, weights %.0f MiB, activations %.0f MiB\n",
        context.Info().mName.c_str(), width, height, g.mFullWidth, g.mFullHeight, g.mWidth[5], g.mHeight[5],
        graph.Gmac(), graph.Dispatches(), graph.WeightBytes() / 1048576.0, graph.ActivationBytes() / 1048576.0);
    if (tune) {
        graph.Tune();
    }
    const size_t segments = graph.Segments().size();
    std::vector<double> totals;
    std::vector<std::vector<double>> perSegment;
    for (int run = 0; run < runs + 3; ++run) { // three warm-up frames: pipelines, clocks
        VkCommandBuffer commands = context.Begin();
        graph.Record(commands, 0, SIZE_MAX, true, reuseVit && run > 0);
        context.Submit(commands, true);
        if (run < 3) {
            continue;
        }
        std::vector<double> ms(segments);
        for (size_t s = 0; s < segments; ++s) {
            ms[s] = context.ElapsedMs((uint32_t)s, (uint32_t)s + 1);
        }
        perSegment.push_back(ms);
        totals.push_back(context.ElapsedMs(0, (uint32_t)segments));
    }
    const size_t best = std::min_element(totals.begin(), totals.end()) - totals.begin();
    std::vector<double> sorted = totals;
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::string> order;
    std::map<std::string, std::pair<double, double>> stages;
    for (size_t s = 0; s < segments; ++s) {
        const std::string stage = StageOf(graph.Segments()[s]);
        if (!stages.count(stage)) {
            order.push_back(stage);
        }
        stages[stage].first += perSegment[best][s];
        stages[stage].second += graph.Segments()[s].mGmac;
    }
    for (const std::string& stage : order) {
        printf("  %-34s %7.3f ms  %6.1f GMAC\n", stage.c_str(), stages[stage].first, stages[stage].second);
    }
    printf("  frame%s: min %.2f ms, median %.2f ms over %d runs\n", reuseVit ? " (ViT reused)" : "", sorted.front(),
           sorted[sorted.size() / 2], runs);
    return 0;
}

int Verify(const std::string& modelPath, uint32_t width, uint32_t height, bool fp8, const std::string& image,
           const std::string& saveStem) {
    const Model model(modelPath);
    const Geometry g = Geometry::FromValid(width, height);
    const std::vector<float> proxy = image.empty() ? TestProxy(width, height) : LoadPpm(image, width, height);
    const std::vector<float> features = Reference::Features(g, proxy, {});
    printf("%ux%u -> field %ux%u\n", width, height, g.mFullWidth, g.mFullHeight);

    std::map<std::string, std::vector<float>> ref16, ref8;
    auto start = std::chrono::steady_clock::now();
    ref16["head"] = Reference(model, g, Arithmetic::Fp16).Run(features, &ref16);
    printf("CPU reference, FP16 semantics: %.1f s\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    if (fp8) {
        start = std::chrono::steady_clock::now();
        ref8["head"] = Reference(model, g, Arithmetic::Fp8).Run(features, &ref8);
        printf("CPU reference, exact FP8 arithmetic: %.1f s\n",
               std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }

    if (fp8) {
        // How fast two arithmetics drift apart when each runs on its own output: the yardstick for reading any
        // free-running comparison (including another implementation's) as rounding noise or as a real difference.
        printf("\nfree running, FP16 reference against FP8 reference (SNR dB):");
        for (int block : { 0, 4, 8, 14, 22, 30, 31, 38, 39, 47, 55, 61, 65, 69 }) {
            const std::string name = "block-" + std::to_string(block);
            printf(" %d:%.1f", block, Compare(ref8.at(name), ref16.at(name)).mSnrDb);
        }
        printf(" head:%.1f\n", Compare(ref8.at("head"), ref16.at("head")).mSnrDb);
    }

    Context context(getenv("NR_GPU"));
    Graph graph(context, model, g);
    graph.SetFeatures(features);
    auto runSegment = [&](size_t s) {
        VkCommandBuffer commands = context.Begin();
        graph.Record(commands, s, 1);
        context.Submit(commands, true);
    };
    auto stateOf = [&](const Graph::Segment& segment) {
        return segment.mBoundary == "head" ? graph.Head() : graph.Read(*segment.mState);
    };

    // 1. One step at a time, each fed the reference's own inputs: a difference names the step that made it.
    //    Against the FP16 reference this is the implementation check; against FP8 it is the price of the arithmetic.
    //    A whole pass per reference: a step also reads half tensors its predecessor left (the pools, the merges).
    auto pass = [&](const std::map<std::string, std::vector<float>>& reference) {
        std::vector<Difference> differences;
        for (size_t s = 0; s < graph.Segments().size(); ++s) {
            const Graph::Segment& segment = graph.Segments()[s];
            for (const auto& [activation, boundary] : segment.mInputs) {
                graph.Write(*activation, reference.at(boundary));
            }
            runSegment(s);
            differences.push_back(Compare(reference.at(segment.mBoundary), stateOf(segment)));
        }
        return differences;
    };
    const std::vector<Difference> d16 = pass(ref16), d8 = fp8 ? pass(ref8) : std::vector<Difference>();
    printf("\nstep by step (each step fed the reference's inputs): SNR dB / share of values differing\n");
    printf("  %-18s %-22s %s\n", "step", "GPU vs FP16 reference", fp8 ? "GPU vs FP8 reference" : "");
    double worst16 = 999.0, worst8 = 999.0, sum8 = 0.0;
    for (size_t s = 0; s < graph.Segments().size(); ++s) {
        worst16 = std::min(worst16, d16[s].mSnrDb);
        printf("  %-18s %6.1f dB %6.2f%%", graph.Segments()[s].mLabel.c_str(), d16[s].mSnrDb,
               100.0 * d16[s].mDiffering);
        if (fp8) {
            worst8 = std::min(worst8, d8[s].mSnrDb);
            sum8 += std::min(d8[s].mSnrDb, 60.0);
            printf("      %6.1f dB %6.2f%%", d8[s].mSnrDb, 100.0 * d8[s].mDiffering);
        }
        printf("\n");
    }
    printf("  worst step against the FP16 reference: %.1f dB\n", worst16);
    if (fp8) {
        printf("  against the FP8 reference: worst step %.1f dB, mean %.1f dB\n", worst8,
               sum8 / (double)graph.Segments().size());
    }

    // 2. The whole frame, free running: every step on the GPU's own output. The drift per boundary is to be read
    //    against the references' own drift above.
    graph.SetFeatures(features);
    printf("\nfree running, GPU against FP16 reference (SNR dB):");
    for (size_t s = 0; s < graph.Segments().size(); ++s) {
        const Graph::Segment& segment = graph.Segments()[s];
        runSegment(s);
        int block = -1;
        if (sscanf(segment.mBoundary.c_str(), "block-%d", &block) == 1 &&
            (block == 0 || block == 4 || block == 8 || block == 14 || block == 22 || block == 30 || block == 31 ||
             block == 38 || block == 39 || block == 47 || block == 55 || block == 61 || block == 65 || block == 69)) {
            printf(" %d:%.1f", block, Compare(ref16.at(segment.mBoundary), stateOf(segment)).mSnrDb);
        }
    }
    printf("\n");
    // And once more as one submission, which is how it runs for real; it must give the same head.
    const std::vector<float> stepped = graph.Head();
    {
        VkCommandBuffer commands = context.Begin();
        graph.Record(commands);
        context.Submit(commands, true);
    }
    const std::vector<float> head = graph.Head();
    printf("  one submission against step by step: %s\n", head == stepped ? "identical heads" : "DIFFERENT heads");
    const std::vector<float> image16 = Compose(g, proxy, ref16.at("head")), imageGpu = Compose(g, proxy, head);
    {
        std::vector<float> zero(head.size(), 0.0f);
        printf("\nwhole frame, free running (the network moves the image by %.1f dB PSNR from its input):\n",
               PsnrDb(Compose(g, proxy, zero), image16));
    }
    const Difference h16 = Compare(ref16.at("head"), head);
    printf("  head, GPU vs FP16 reference:  %6.1f dB SNR;  composed image %6.1f dB PSNR\n", h16.mSnrDb,
           PsnrDb(image16, imageGpu));
    if (fp8) {
        const std::vector<float> image8 = Compose(g, proxy, ref8.at("head"));
        const Difference h8 = Compare(ref8.at("head"), head), hRef = Compare(ref8.at("head"), ref16.at("head"));
        printf("  head, GPU vs FP8 reference:   %6.1f dB SNR;  composed image %6.1f dB PSNR\n", h8.mSnrDb,
               PsnrDb(image8, imageGpu));
        printf("  head, FP16 vs FP8 on the CPU: %6.1f dB SNR;  composed image %6.1f dB PSNR\n", hRef.mSnrDb,
               PsnrDb(image8, image16));
        if (!saveStem.empty()) {
            SavePpm(saveStem + "-fp8.ppm", image8, g.mValidWidth, g.mValidHeight);
        }
    }
    if (!saveStem.empty()) {
        std::vector<float> input((size_t)g.mValidWidth * g.mValidHeight * 3);
        for (size_t i = 0; i < input.size() / 3; ++i) {
            std::copy(&proxy[i * 4], &proxy[i * 4] + 3, &input[i * 3]);
        }
        SavePpm(saveStem + "-input.ppm", input, g.mValidWidth, g.mValidHeight);
        SavePpm(saveStem + "-gpu.ppm", imageGpu, g.mValidWidth, g.mValidHeight);
    }
    return 0;
}

// The exact FP8 reference's frame in the fixture format of OpenDLSS-NR's `parity` (README "Fixtures"): the input
// features, the E4M3 bytes of the 75 comparable boundaries and the f32 head. Its WebGPU port, which is verified
// against native captures, then says whether this reference computes the same bytes.
int Fixture(const std::string& modelPath, uint32_t width, uint32_t height, const std::filesystem::path& out,
            const std::string& image, bool bisect) {
    const Model model(modelPath);
    const Geometry g = Geometry::FromValid(width, height);
    const std::vector<float> features =
        Reference::Features(g, image.empty() ? TestProxy(width, height) : LoadPpm(image, width, height), {});
    std::map<std::string, std::vector<float>> boundaries;
    Reference reference(model, g, Arithmetic::Fp8);
    reference.mCaptureBlock0 = bisect;
    const std::vector<float> head = reference.Run(features, &boundaries);
    std::filesystem::create_directories(out);
    auto write = [&](const std::string& name, const void* data, size_t bytes) {
        std::ofstream(out / name, std::ios::binary).write((const char*)data, (std::streamsize)bytes);
    };
    write("features.f32", features.data(), features.size() * 4);
    write("head.f32", head.data(), head.size() * 4);
    auto shapeOf = [&](int block, uint32_t& w, uint32_t& h, uint32_t& c) {
        if (block == 0) {
            w = g.mFullWidth;
            h = g.mFullHeight;
            c = 32;
            return;
        }
        static const int kLastOfLevel[6] = { 4, 8, 14, 22, 30, 38 };
        int level = 0;
        const int mirrored = block <= 38 ? block
                                         : (block <= 47   ? 30
                                            : block <= 55 ? 22
                                            : block <= 61 ? 14
                                            : block <= 65 ? 8
                                                          : 4);
        while (mirrored > kLastOfLevel[level])
            ++level;
        w = g.mWidth[level];
        h = g.mHeight[level];
        c = 32u << level;
    };
    std::string blocks, transitions;
    for (const auto& [name, tensor] : boundaries) {
        int block = 0, from = 0, to = 0;
        uint32_t w = 0, h = 0, c = 0;
        const bool isBlock = sscanf(name.c_str(), "block-%d", &block) == 1 && name.find("raw") == std::string::npos;
        const bool isTransition = sscanf(name.c_str(), "transition-%d-%d", &from, &to) == 2 && from != 30;
        const bool isProbe = name.rfind("transition-b0-", 0) == 0; // block 0's intermediates, at full resolution
        if ((!isBlock || block > 69) && !isTransition && !isProbe) {
            continue;
        }
        if (isProbe) {
            w = g.mFullWidth;
            h = g.mFullHeight;
            c = (uint32_t)(tensor.size() / g.FieldTokens());
        } else {
            shapeOf(isBlock ? block : to, w, h, c);
        }
        if (tensor.size() != (size_t)w * h * c) {
            throw std::runtime_error("fixture: unexpected shape for " + name);
        }
        std::vector<uint8_t> codes(tensor.size());
        for (size_t i = 0; i < tensor.size(); ++i) {
            codes[i] = Num::E4m3FromF16Bits(Num::F16Bits(tensor[i]));
        }
        write(name + ".e4", codes.data(), codes.size());
        char entry[256];
        if (isBlock) {
            snprintf(entry, sizeof entry,
                     "%s{\"block\":%d,\"width\":%u,\"height\":%u,\"channels\":%u,\"file\":\"%s.e4\"}",
                     blocks.empty() ? "" : ",", block, w, h, c, name.c_str());
            blocks += entry;
        } else {
            snprintf(entry, sizeof entry,
                     "%s{\"id\":\"%s\",\"width\":%u,\"height\":%u,\"channels\":%u,\"file\":\"%s.e4\"}",
                     transitions.empty() ? "" : ",", name.c_str() + strlen("transition-"), w, h, c, name.c_str());
            transitions += entry;
        }
    }
    std::ofstream(out / "manifest.json")
        << "{\"sourceDimensions\":[" << width << "," << height << "],\"fullDimensions\":[" << g.mFullWidth << ","
        << g.mFullHeight
        << "],\"inputFeatures\":{\"file\":\"features.f32\"},\"checks\":[\"boundaries\",\"head\"],\"blocks\":[" << blocks
        << "],\"transitions\":[" << transitions << "],\"referenceHead\":{\"file\":\"head.f32\"}}\n";
    printf("fixture -> %s\n", out.string().c_str());
    return 0;
}

int Usage() {
    fprintf(stderr, "usage: nd-nr devices | selftest\n"
                    "       nd-nr extract <nvngx_dlssnr.dll> <model dir>\n"
                    "       nd-nr synth <model dir> [seed]\n"
                    "       nd-nr reference <model dir> <width> <height> [--fp8] [--last N]\n"
                    "       nd-nr verify <model dir> <width> <height> [--fp8] [--image file.ppm] [--save stem]\n"
                    "       nd-nr bench <model dir> <width> <height> [--runs N] [--no-tune] [--reuse-vit]\n"
                    "       nd-nr fixture <model dir> <width> <height> <out dir> [--image file.ppm] [--bisect]\n"
                    "       nd-nr sensitivity <model dir> <width> <height> [--image file.ppm]\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) try {
    const std::string command = argc > 1 ? argv[1] : "";
    auto flag = [&](const char* name) {
        for (int i = 2; i < argc; ++i) {
            if (!strcmp(argv[i], name)) {
                return true;
            }
        }
        return false;
    };
    auto option = [&](const char* name, const char* fallback) -> std::string {
        for (int i = 2; i + 1 < argc; ++i) {
            if (!strcmp(argv[i], name)) {
                return argv[i + 1];
            }
        }
        return fallback;
    };
    if (command == "selftest") {
        return SelfTest();
    }
    if (command == "devices") {
        for (const DeviceInfo& d : Context::Enumerate()) {
            printf("%s (%04x:%04x), %.1f GiB: %s%s%s\n", d.mName.c_str(), d.mVendor, d.mDevice,
                   d.mVramBytes / 1073741824.0,
                   d.mMissing.empty() ? "FP16 path ready" : ("lacks " + d.mMissing).c_str(),
                   d.mFloat8 ? ", has FP8" : "", d.mExternalFd ? ", fd interop" : "");
        }
        return 0;
    }
    if (command == "extract" && argc == 4) {
        const ExtractSummary s = ExtractModel(argv[2], argv[3]);
        printf("%u tensors, %u blocks, %llu payload bytes -> %s\n", s.mTensors, s.mBlocks,
               (unsigned long long)s.mPayloadBytes, argv[3]);
        return 0;
    }
    if (command == "synth" && argc >= 3) {
        SynthesizeModel(argv[2], argc > 3 ? (uint32_t)atoi(argv[3]) : 1u);
        printf("synthetic model -> %s\n", argv[2]);
        return 0;
    }
    if (argc < 5) {
        return Usage();
    }
    const uint32_t width = (uint32_t)atoi(argv[3]), height = (uint32_t)atoi(argv[4]);
    if (command == "sensitivity") {
        // The network's own noise floor: how far the head moves when a sparse subset of the inputs changes by one
        // half-float ulp, same arithmetic. Any comparison between two arithmetics is to be read against this.
        const Model model(argv[2]);
        const Geometry g = Geometry::FromValid(width, height);
        const std::string image = option("--image", "");
        const std::vector<float> proxy = image.empty() ? TestProxy(width, height) : LoadPpm(image, width, height);
        std::vector<float> features = Reference::Features(g, proxy, {});
        const std::vector<float> base = Reference(model, g, Arithmetic::Fp16).Run(features);
        for (size_t at : { (size_t)7, (size_t)101, (size_t)1009 }) { // one ulp on every `at`-th input value
            std::vector<float> nudged = features;
            for (size_t i = 0; i < nudged.size(); i += at) {
                nudged[i] = Num::F16ToF32((uint16_t)(Num::F16Bits(nudged[i]) + 1));
            }
            const std::vector<float> head = Reference(model, g, Arithmetic::Fp16).Run(nudged);
            double m0 = 0, m1 = 0, s3 = 0, e3 = 0, srgb = 0, ergb = 0;
            for (size_t i = 0; i < head.size() / 4; ++i) {
                m0 += base[i * 4 + 3];
                m1 += head[i * 4 + 3];
                s3 += (double)base[i * 4 + 3] * base[i * 4 + 3];
                e3 += ((double)base[i * 4 + 3] - head[i * 4 + 3]) * ((double)base[i * 4 + 3] - head[i * 4 + 3]);
                for (int c = 0; c < 3; ++c) {
                    srgb += (double)base[i * 4 + c] * base[i * 4 + c];
                    ergb += ((double)base[i * 4 + c] - head[i * 4 + c]) * ((double)base[i * 4 + c] - head[i * 4 + c]);
                }
            }
            const size_t n = head.size() / 4;
            printf("one ulp on every %zu-th input value: head RGB SNR %.1f dB, logit SNR %.1f dB (mean %.3f -> %.3f)\n",
                   at, 10 * std::log10(srgb / ergb), 10 * std::log10(s3 / e3), m0 / n, m1 / n);
        }
        return 0;
    }
    if (command == "reference") {
        const Arithmetic arithmetic = flag("--fp8") ? Arithmetic::Fp8 : Arithmetic::Fp16;
        const Model model(argv[2]);
        const Geometry g = Geometry::FromValid(width, height);
        printf("%ux%u -> field %ux%u\n", g.mValidWidth, g.mValidHeight, g.mFullWidth, g.mFullHeight);
        const std::vector<float> features = Reference::Features(g, TestProxy(g.mValidWidth, g.mValidHeight), {});
        std::map<std::string, std::vector<float>> boundaries;
        const auto start = std::chrono::steady_clock::now();
        const std::vector<float> head =
            Reference(model, g, arithmetic).Run(features, &boundaries, atoi(option("--last", "70").c_str()));
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        for (int block = 0; block <= 69; ++block) {
            auto it = boundaries.find("block-" + std::to_string(block));
            if (it != boundaries.end()) {
                Describe(it->first, it->second);
            }
        }
        if (!head.empty()) {
            Describe("head", head);
        }
        printf("%s reference: %.1f s\n", arithmetic == Arithmetic::Fp8 ? "FP8" : "FP16", seconds);
        return 0;
    }
    if (command == "fixture" && argc >= 6) {
        return Fixture(argv[2], width, height, argv[5], option("--image", ""), flag("--bisect"));
    }
    if (command == "verify") {
        return Verify(argv[2], width, height, flag("--fp8"), option("--image", ""), option("--save", ""));
    }
    if (command == "bench") {
        return Bench(argv[2], width, height, atoi(option("--runs", "12").c_str()), !flag("--no-tune"),
                     flag("--reuse-vit"));
    }
    return Usage();
} catch (const std::exception& error) {
    fprintf(stderr, "nd-nr: %s\n", error.what());
    return 1;
}
