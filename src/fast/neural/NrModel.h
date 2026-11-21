// The DLSS-NR model directory: what the player's own nvngx_dlssnr.dll becomes on their machine, and how its tensor
// records turn into plain matrices.
//
// Nothing in this tree is NVIDIA's. ExtractModel reads the file the player points at and writes a cache beside
// their other generated data; Model reads that cache. The record framing follows lmxxf/dlss5-on-amd-9070xt-porting
// and the tensor layouts OpenDLSS-NR's nr_model.cpp (both MIT, see THIRD_PARTY.md).
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace Fast::Neural {

struct Tensor {
    std::string mName;
    int mBlock = 0;
    int mLayer = 0;
    std::string mParameter;
    const uint8_t* mBytes = nullptr; // into the model's storage
    uint32_t mByteLength = 0;
};

class Model {
  public:
    static constexpr uint32_t kBlocks = 71;
    static constexpr uint32_t kTensors = 153;

    // Throws std::runtime_error with a message fit for a status line.
    explicit Model(const std::filesystem::path& directory, bool verifyHash = true);

    const Tensor& Get(int block, int layer = 0, const char* parameter = "layer") const;

    // E4M3 weights are stored as tensor-core B fragments, with the K rows in the chained activation order. This
    // returns plain values [K][N] in natural channel order (the NaN code decodes to 0, as the reference's).
    std::vector<float> Fp8Matrix(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t N) const;
    // The two f16 matrices (16 -> 32 input adapter, 32 -> 4 head), [K][N].
    std::vector<float> F16Matrix(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t N) const;
    // The attention prior, [head][64 query][64 key] with both tokens in tiled order (4x4 tiles of 16).
    std::vector<float> Prior(const Tensor& tensor, uint32_t byteOffset, uint32_t heads) const;
    // `count` f16 values (the per-channel skip scales).
    std::vector<float> Halves(const Tensor& tensor, uint32_t byteOffset, uint32_t count) const;
    // `count` f32 values (the per-head attention scales).
    std::vector<float> Floats(const Tensor& tensor, uint32_t byteOffset, uint32_t count) const;

    // Natural window token (row-major in the 8x8 window) -> tiled token.
    static uint32_t TiledToken(uint32_t token) {
        const uint32_t x = token & 7, y = token >> 3;
        return (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3);
    }

  private:
    std::vector<uint8_t> mStorage;
    std::map<std::string, Tensor> mTensors;
};

struct ExtractSummary {
    uint32_t mTensors = 0;
    uint32_t mBlocks = 0;
    uint64_t mPayloadBytes = 0;
    std::string mSourceSha256;
};

// Builds `directory` (manifest.json + model/weights.bin) from the player's DLL. Throws std::runtime_error with a
// message fit for a status line when the file is not what it should be; writes nothing in that case.
ExtractSummary ExtractModel(const std::filesystem::path& dll, const std::filesystem::path& directory);

} // namespace Fast::Neural
