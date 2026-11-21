// A stand-in model for development without anyone's weights: the 153 tensor records of the 71-block network with
// their real sizes (re-derived from the layouts, which is also a check of the layouts), filled with random values of
// plausible magnitude so activations neither vanish nor saturate. It means nothing as an image; it exercises every
// kernel, every layout offset and every publication.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <random>

#include <nlohmann/json.hpp>

#include "NrLayout.h"
#include "NrModel.h"
#include "NrNumeric.h"
#include "NrReference.h"
#include "NrSha256.h"

namespace Fast::Neural {

namespace {

struct Record {
    std::string mName;
    int mBlock, mLayer;
    std::string mParameter;
    std::vector<uint8_t> mBytes;
};

// Where a block's output is set: its projection matrix and the attention skip scale.
struct OutputStage {
    size_t mRecord = 0;
    uint32_t mWeights = 0, mWeightCount = 0, mScale = 0, mChannels = 0;
};

// Gain of the layers that add onto a skip: with skip scales just under one this keeps activations of order one.
constexpr float kOut = 0.6f;

struct Filler {
    std::mt19937 mRng;

    void Weights(Record& r, uint32_t offset, uint32_t K, uint32_t N, float gain = 0.8f) {
        std::normal_distribution<float> d(0.0f, gain / std::sqrt((float)K));
        for (uint32_t i = 0; i < K * N; ++i) {
            r.mBytes.at(offset + i) = Num::E4m3FromF16Bits(Num::F16Bits(d(mRng)));
        }
    }
    void Half(Record& r, uint32_t offset, float value) {
        const uint16_t bits = Num::F16Bits(value);
        r.mBytes.at(offset) = (uint8_t)bits;
        r.mBytes.at(offset + 1) = (uint8_t)(bits >> 8);
    }
    void Scales(Record& r, uint32_t offset, uint32_t count, float low = 0.9f, float high = 1.0f) {
        std::uniform_real_distribution<float> d(low, high);
        for (uint32_t i = 0; i < count; ++i) {
            Half(r, offset + i * 2, d(mRng));
        }
    }
    void Normals(Record& r, uint32_t offset, uint32_t halves, float sigma) {
        std::normal_distribution<float> d(0.0f, sigma);
        for (uint32_t i = 0; i < halves; ++i) {
            Half(r, offset + i * 2, d(mRng));
        }
    }
    void Floats(Record& r, uint32_t offset, uint32_t count, float low, float high) {
        std::uniform_real_distribution<float> d(low, high);
        for (uint32_t i = 0; i < count; ++i) {
            const float value = d(mRng);
            memcpy(&r.mBytes.at(offset + i * 4), &value, 4);
        }
    }

    // Everything a window block's record holds from the FFN scale on.
    void Attention(Record& r, const BlockLayout& l) {
        const uint32_t C = l.mChannels;
        Scales(r, l.mFfnScale, C);
        Weights(r, l.mQkv, C, 3 * C, 1.5f);
        Normals(r, l.mPrior, l.mHeads * 4096, 1.0f);
        Floats(r, l.mQScale, l.mHeads, 3.0f, 6.0f);
        Weights(r, l.mProjection, C, C, kOut);
        Scales(r, l.mAttnScale, C);
    }
    void Ffn(Record& r, const BlockLayout& l) {
        const uint32_t C = l.mChannels;
        if (!l.mExperts) {
            Weights(r, l.mExpand, 32, 128, 1.0f);
            Weights(r, l.mContract, 128, 32, kOut);
        } else {
            Weights(r, l.mExpand, l.mExperts * C, 128, std::sqrt((float)l.mExperts)); // K is C per expert
            Weights(r, l.mContract, l.mExperts * 128, 32, std::sqrt((float)l.mExperts));
            Weights(r, l.mW3, C, C, kOut);
        }
    }
};

} // namespace

void SynthesizeModel(const std::filesystem::path& directory, uint32_t seed) {
    Filler fill{ std::mt19937(seed) };
    std::vector<Record> records;
    records.reserve(Model::kTensors); // references into it are kept while filling
    std::map<int, OutputStage> outputs;
    auto add = [&](int block, int layer, const char* parameter, uint32_t bytes) -> Record& {
        records.push_back({ "block" + std::to_string(block) + ".layer" + std::to_string(layer) + "." + parameter, block,
                            layer, parameter, std::vector<uint8_t>(bytes, 0) });
        return records.back();
    };
    auto windowBlock = [&](int block, BlockLayout l, uint32_t extra) -> Record& {
        Record& r = add(block, 0, "layer", l.mEnd + extra);
        fill.Ffn(r, l);
        fill.Attention(r, l);
        outputs[block] = { records.size() - 1, l.mProjection, l.mChannels * l.mChannels, l.mAttnScale, l.mChannels };
        return r;
    };
    auto splitBlock = [&](int block) {
        Record& branches = add(block, 0, "layer", 524288);
        fill.Weights(branches, 0, 512, 512);
        fill.Weights(branches, 512 * 512, 8 * 64, 256, 1.2f * std::sqrt(8.0f));
        fill.Weights(branches, 512 * 512 + 8 * 64 * 256, 8 * 256, 64, std::sqrt(8.0f));
        Record& contract = add(block, 1, "layer", 263168);
        fill.Weights(contract, 0, 512, 512, kOut);
        fill.Scales(contract, 512 * 512, 512);
        Record& attention = add(block, 2, "layer", 917568);
        fill.Weights(attention, 0, 512, 1536, 1.5f);
        fill.Normals(attention, 512 * 1536, 16 * 4096, 1.0f);
        fill.Floats(attention, 512 * 1536 + 16 * 8192, 16, 3.0f, 6.0f);
        Record& projection = add(block, 3, "layer", 263168);
        fill.Weights(projection, 0, 512, 512, kOut);
        fill.Scales(projection, 512 * 512, 512);
        outputs[block] = { records.size() - 1, 0, 512 * 512, 512 * 512, 512 };
    };

    { // block 0: the f16 input adapter sits inside the record
        const BlockLayout l = PreLayout();
        Record& r = windowBlock(0, l, 16);
        fill.Normals(r, l.mInputAdapter, 16 * 32, 0.6f);
    }
    for (const Stage& stage : kEncoderStages) {
        const uint32_t C = stage.mChannels;
        for (int block = stage.mFirst; block <= stage.mLast; ++block) {
            if (C == 512) {
                splitBlock(block);
                continue;
            }
            const BlockLayout l = StandardLayout(C);
            if (block != stage.mLast) {
                windowBlock(block, l, 16);
            } else { // the next stage's C -> 2C matrix follows (and, for 32 channels only, 16 bytes of padding)
                Record& r = windowBlock(block, l, C * 2 * C + (C == 32 ? 16 : 0));
                fill.Weights(r, l.mEnd, C, 2 * C, 1.6f); // the pool halves the signal
            }
        }
    }
    fill.Weights(add(30, 4, "layer", 524304), 0, 512, 1024, 1.6f);
    for (int block = 31; block <= 38; ++block) {
        fill.Weights(add(block, 0, "layer", 4194320), 0, 1024, 4096, 1.0f);
        Record& contract = add(block, 1, "layer", 4196352);
        fill.Weights(contract, 0, 4096, 1024, kOut);
        fill.Scales(contract, 4096 * 1024, 1024);
        Record& attention = add(block, 2, "layer", 3145856); // head scales first
        fill.Floats(attention, 0, 32, 0.6f, 1.2f);
        fill.Weights(attention, 128, 1024, 3072, 1.5f);
        add(block, 3, "layer", 2);
        Record& projection = add(block, 4, "layer", 1050624);
        fill.Weights(projection, 0, 1024, 1024, kOut);
        fill.Scales(projection, 1024 * 1024, 1024);
        outputs[block] = { records.size() - 1, 0, 1024 * 1024, 1024 * 1024, 1024 };
    }
    { // block 39: the 1024 -> 512 matrix and the skip merge's scale
        Record& r = add(39, 0, "layer", 525312);
        fill.Weights(r, 0, 1024, 512, 0.5f);
        fill.Scales(r, 1024 * 512, 512);
    }
    for (const Stage& stage : kDecoderStages) {
        const uint32_t C = stage.mChannels;
        for (int block = stage.mFirst; block <= stage.mLast; ++block) {
            if (C == 512) {
                splitBlock(block);
            } else if (block == stage.mFirst) {
                const BlockLayout l = UpsampleLayout(C);
                Record& r = windowBlock(block, l, 16);
                fill.Weights(r, l.mUpsample, 2 * C, C, 0.5f);
                fill.Scales(r, l.mTransitionScale, C);
            } else {
                windowBlock(block, StandardLayout(C), 16);
            }
        }
    }
    { // block 70: the post blend scales and the f16 RGBA head
        const BlockLayout l = PostLayout();
        Record& r = windowBlock(70, l, 0);
        fill.Scales(r, l.mInputScale, 32);
        fill.Scales(r, l.mAdapterScale, 32);
        fill.Normals(r, l.mHead, 512, 0.15f);
        fill.Half(add(70, 0, "blend_scale", 2), 0, 0.73974609375f);
    }

    uint64_t payload = 0;
    for (const Record& r : records) {
        payload += r.mBytes.size();
    }
    if (records.size() != Model::kTensors || payload != 147683778ull) {
        throw std::runtime_error(
            "synthetic model: the layouts do not add up to the network's 153 records / 147683778 bytes (" +
            std::to_string(records.size()) + " / " + std::to_string(payload) + ")");
    }

    auto write = [&] {
        nlohmann::json tensors = nlohmann::json::array();
        std::vector<uint8_t> blob;
        for (const Record& r : records) {
            tensors.push_back({ { "name", r.mName },
                                { "block", r.mBlock },
                                { "layer", r.mLayer },
                                { "parameter", r.mParameter },
                                { "stage", "all" },
                                { "stageOffset", blob.size() },
                                { "byteLength", r.mBytes.size() } });
            blob.insert(blob.end(), r.mBytes.begin(), r.mBytes.end());
            blob.resize((blob.size() + 15) & ~size_t(15));
        }
        nlohmann::json manifest = {
            { "source", { { "synthetic", true }, { "seed", seed } } },
            { "totals",
              { { "blockCount", Model::kBlocks }, { "tensorCount", records.size() }, { "payloadBytes", payload } } },
            { "stages", nlohmann::json::array({ { { "id", "all" },
                                                  { "file", "weights.bin" },
                                                  { "packedByteLength", blob.size() },
                                                  { "sha256", Sha256Hex(blob.data(), blob.size()) } } }) },
            { "tensors", tensors },
        };
        std::filesystem::create_directories(directory / "model");
        std::ofstream(directory / "model" / "weights.bin", std::ios::binary | std::ios::trunc)
            .write((const char*)blob.data(), (std::streamsize)blob.size());
        std::ofstream(directory / "manifest.json", std::ios::trunc) << manifest.dump(1) << "\n";
    };
    write();

    // A random network of this depth either explodes or dies out: every layer is (nearly) homogeneous, so a block's
    // gain compounds. Calibrate: run the network, and scale each block's output stage by the gain it showed, until
    // every block boundary is of order one.
    const Geometry geometry = Geometry::FromValid(256, 256);
    std::vector<float> proxy((size_t)256 * 256 * 4, 0.5f);
    std::mt19937 rng(seed + 1);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (float& v : proxy) {
        v = uniform(rng);
    }
    const std::vector<float> features = Reference::Features(geometry, proxy, {});
    auto rms = [](const std::vector<float>& t) {
        double squares = 0;
        for (float v : t) {
            squares += (double)v * v;
        }
        return std::sqrt(squares / (double)t.size());
    };
    for (int pass = 0; pass < 12; ++pass) {
        std::map<std::string, std::vector<float>> boundaries;
        {
            const Model model(directory, false);
            Reference(model, geometry, Arithmetic::Fp16).Run(features, &boundaries, 70);
        }
        double previous = 1.5, worst = 1.0;
        for (int block = 0; block <= 70; ++block) {
            auto found = boundaries.find(block == 70 ? "block-70-raw" : "block-" + std::to_string(block));
            auto stage = outputs.find(block);
            if (found == boundaries.end() || stage == outputs.end()) {
                continue; // block 39 is a merge
            }
            const double level = std::max(rms(found->second), 1e-6);
            worst = std::max(worst, std::max(level / 1.5, 1.5 / level));
            // Bring this block back to the level of its input; the first one sets the level.
            const float correction = (float)std::clamp(previous / level, 0.2, 5.0);
            previous = block == 0 ? 1.5 : level;
            Record& r = records[stage->second.mRecord];
            for (uint32_t i = 0; i < stage->second.mWeightCount; ++i) {
                uint8_t& code = r.mBytes[stage->second.mWeights + i];
                code = Num::E4m3FromF16Bits(Num::F16Bits(Num::E4m3ToF32(code) * correction));
            }
            for (uint32_t c = 0; c < stage->second.mChannels; ++c) {
                const uint8_t* p = &r.mBytes[stage->second.mScale + c * 2];
                fill.Half(r, stage->second.mScale + c * 2, Num::F16ToF32((uint16_t)(p[0] | (p[1] << 8))) * correction);
            }
        }
        write();
        if (worst < 3.0) {
            break;
        }
    }
}

} // namespace Fast::Neural
