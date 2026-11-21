#include "NrModel.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "NrNumeric.h"
#include "NrSha256.h"

namespace Fast::Neural {

namespace {

std::vector<uint8_t> ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot read " + path.filename().string());
    }
    const std::streamsize size = file.tellg();
    file.seekg(0);
    std::vector<uint8_t> bytes((size_t)size);
    file.read((char*)bytes.data(), size);
    if (!file) {
        throw std::runtime_error("cannot read " + path.filename().string());
    }
    return bytes;
}

bool SameHex(std::string a, std::string b) {
    std::transform(a.begin(), a.end(), a.begin(), [](unsigned char c) { return std::tolower(c); });
    std::transform(b.begin(), b.end(), b.begin(), [](unsigned char c) { return std::tolower(c); });
    return a == b;
}

// Every FP8 GEMM's A operand is indexed by a rotation of index bits 1-3 within each group of 32 channels; this is
// its inverse, applied to the K rows so activations stay in natural channel order (OpenDLSS-NR nr_model.cpp).
uint32_t InversePackedInputIndex(uint32_t k) {
    const uint32_t base = k & ~31u, within = k & 31u;
    return base + (within & 17u) + ((within & 2u) << 1) + ((within & 4u) << 1) + ((within & 8u) >> 2);
}

// Byte of weight (k, n) inside a matrix stored as m16n8k32 B fragment pairs.
uint32_t PackedWeightIndex(uint32_t k, uint32_t n, uint32_t outputChannels) {
    const uint32_t kTile = k >> 5, kIn = k & 31;
    const uint32_t nTile = n >> 7, nIn = n & 127;
    const uint32_t nHalf = nIn >> 6, nGroup = (nIn & 63) >> 4, nInGroup = nIn & 15;
    const uint32_t lane = ((nInGroup & 7) << 2) | ((kIn & 15) >> 2);
    const uint32_t byteInLane = ((nInGroup >> 3) << 3) | ((kIn >> 4) << 2) | (kIn & 3);
    return kTile * outputChannels * 32 + nTile * 4096 + nHalf * 2048 + nGroup * 512 + lane * 16 + byteInLane;
}

// Half index of an f16 weight inside a matrix stored as m16n8k16 B fragment pairs, tiled 16x16 in (k, n).
uint32_t PackedF16WeightIndex(uint32_t k, uint32_t n, uint32_t outputChannels) {
    const uint32_t nTiles = (outputChannels + 15) / 16;
    const uint32_t tile = (k >> 4) * nTiles + (n >> 4);
    const uint32_t kk = k & 15, nn = n & 15;
    const uint32_t lane = ((nn & 7) << 2) | ((kk & 7) >> 1);
    const uint32_t fragment = (kk >= 8 ? 2 : 0) + (kk & 1);
    return tile * 256 + lane * 8 + ((nn >> 3) & 1) * 4 + fragment;
}

void Require(const Tensor& tensor, uint64_t end, const char* what) {
    if (end > tensor.mByteLength) {
        throw std::runtime_error(std::string("model: ") + what + " exceeds tensor " + tensor.mName);
    }
}

} // namespace

Model::Model(const std::filesystem::path& directory, bool verifyHash) {
    nlohmann::json manifest;
    {
        std::ifstream file(directory / "manifest.json");
        if (!file) {
            throw std::runtime_error("no model has been built yet");
        }
        manifest = nlohmann::json::parse(file, nullptr, false);
    }
    if (manifest.is_discarded() || !manifest.contains("stages") || !manifest.contains("tensors")) {
        throw std::runtime_error("the model cache is damaged (manifest); build it again");
    }
    try {
        if (manifest["totals"]["blockCount"].get<uint32_t>() != kBlocks) {
            throw std::runtime_error("the model is not the 71-block network this build supports");
        }
        // One contiguous store; stages are concatenated in manifest order.
        std::map<std::string, size_t> stageBase;
        std::map<std::string, size_t> stageSize;
        for (const nlohmann::json& stage : manifest["stages"]) {
            std::vector<uint8_t> bytes = ReadFile(directory / "model" / stage["file"].get<std::string>());
            if (bytes.size() != stage["packedByteLength"].get<size_t>() ||
                (verifyHash && !SameHex(Sha256Hex(bytes.data(), bytes.size()), stage["sha256"].get<std::string>()))) {
                throw std::runtime_error("the model cache is damaged (weights); build it again");
            }
            const std::string id = stage["id"].get<std::string>();
            stageBase[id] = mStorage.size();
            stageSize[id] = bytes.size();
            mStorage.insert(mStorage.end(), bytes.begin(), bytes.end());
        }
        for (const nlohmann::json& entry : manifest["tensors"]) {
            Tensor tensor;
            tensor.mName = entry["name"].get<std::string>();
            tensor.mBlock = entry["block"].get<int>();
            tensor.mLayer = entry["layer"].get<int>();
            tensor.mParameter = entry["parameter"].get<std::string>();
            tensor.mByteLength = entry["byteLength"].get<uint32_t>();
            const std::string stage = entry["stage"].get<std::string>();
            const size_t offset = entry["stageOffset"].get<size_t>();
            if (!stageBase.count(stage) || offset + tensor.mByteLength > stageSize[stage]) {
                throw std::runtime_error("the model cache is damaged (tensor table); build it again");
            }
            tensor.mBytes = mStorage.data() + stageBase[stage] + offset;
            mTensors[tensor.mName] = tensor;
        }
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error("the model cache is damaged (manifest); build it again");
    }
    if (mTensors.size() != kTensors) {
        throw std::runtime_error("the model is not the 71-block network this build supports");
    }
}

const Tensor& Model::Get(int block, int layer, const char* parameter) const {
    const std::string name = "block" + std::to_string(block) + ".layer" + std::to_string(layer) + "." + parameter;
    auto it = mTensors.find(name);
    if (it == mTensors.end()) {
        throw std::runtime_error("model: missing tensor " + name);
    }
    return it->second;
}

std::vector<float> Model::Fp8Matrix(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t N) const {
    if (K % 32 || N % 16) {
        throw std::runtime_error("model: FP8 matrix shape in " + tensor.mName);
    }
    Require(tensor, (uint64_t)byteOffset + (uint64_t)K * N, "an FP8 matrix");
    std::vector<float> plain((size_t)K * N);
    for (uint32_t j = 0; j < K; ++j) {
        const uint32_t k = InversePackedInputIndex(j);
        for (uint32_t n = 0; n < N; ++n) {
            const uint8_t code = tensor.mBytes[byteOffset + PackedWeightIndex(k, n, N)];
            plain[(size_t)j * N + n] = (code & 0x7f) == 0x7f ? 0.0f : Num::E4m3ToF32(code);
        }
    }
    return plain;
}

std::vector<float> Model::F16Matrix(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t N) const {
    std::vector<float> plain((size_t)K * N);
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t n = 0; n < N; ++n) {
            const uint32_t at = byteOffset + PackedF16WeightIndex(k, n, N) * 2;
            Require(tensor, (uint64_t)at + 2, "an f16 matrix");
            plain[(size_t)k * N + n] = Num::F16ToF32((uint16_t)(tensor.mBytes[at] | (tensor.mBytes[at + 1] << 8)));
        }
    }
    return plain;
}

std::vector<float> Model::Prior(const Tensor& tensor, uint32_t byteOffset, uint32_t heads) const {
    Require(tensor, (uint64_t)byteOffset + (uint64_t)heads * 8192, "an attention prior");
    std::vector<float> prior((size_t)heads * 4096);
    for (uint32_t head = 0; head < heads; ++head) {
        for (uint32_t q = 0; q < 64; ++q) {
            for (uint32_t k = 0; k < 64; ++k) {
                // 16x16 fragment tiles, both token indices tiled.
                const uint32_t m = q & 15, n = k & 15;
                const uint32_t lane = ((m & 7) << 2) | ((n & 7) >> 1);
                const uint32_t fragment = (m >= 8 ? 2 : 0) + (n & 1);
                const uint32_t half = (q >> 4) * 1024 + (k >> 4) * 256 + lane * 8 + (n >> 3) * 4 + fragment;
                const uint32_t at = byteOffset + head * 8192 + half * 2;
                prior[((size_t)head * 64 + q) * 64 + k] =
                    Num::F16ToF32((uint16_t)(tensor.mBytes[at] | (tensor.mBytes[at + 1] << 8)));
            }
        }
    }
    return prior;
}

std::vector<float> Model::Halves(const Tensor& tensor, uint32_t byteOffset, uint32_t count) const {
    Require(tensor, (uint64_t)byteOffset + (uint64_t)count * 2, "a scale vector");
    std::vector<float> values(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* p = tensor.mBytes + byteOffset + i * 2;
        values[i] = Num::F16ToF32((uint16_t)(p[0] | (p[1] << 8)));
    }
    return values;
}

std::vector<float> Model::Floats(const Tensor& tensor, uint32_t byteOffset, uint32_t count) const {
    Require(tensor, (uint64_t)byteOffset + (uint64_t)count * 4, "an attention scale");
    std::vector<float> values(count);
    memcpy(values.data(), tensor.mBytes + byteOffset, (size_t)count * 4);
    return values;
}

// ---- Extraction ----------------------------------------------------------------------------------------------------

namespace {

// Bounds-checked little-endian reads over a file the player supplied.
struct Reader {
    const std::vector<uint8_t>& mBytes;
    const char* mWhat;

    void Need(uint64_t offset, uint64_t size) const {
        if (offset > mBytes.size() || size > mBytes.size() - offset) {
            throw std::runtime_error(mWhat);
        }
    }
    uint16_t U16(uint64_t offset) const {
        Need(offset, 2);
        return (uint16_t)(mBytes[offset] | (mBytes[offset + 1] << 8));
    }
    uint32_t U32(uint64_t offset) const {
        Need(offset, 4);
        uint32_t value;
        memcpy(&value, mBytes.data() + offset, 4);
        return value;
    }
    uint64_t U64(uint64_t offset) const {
        Need(offset, 8);
        uint64_t value;
        memcpy(&value, mBytes.data() + offset, 8);
        return value;
    }
};

// File offset and size of the RT_RCDATA resource named `name` (first language).
std::pair<uint64_t, uint64_t> FindRcData(const std::vector<uint8_t>& pe, const std::u16string& name) {
    const Reader r{ pe, "this file is not a Windows DLL" };
    if (r.U16(0) != 0x5a4d) {
        throw std::runtime_error(r.mWhat);
    }
    const uint64_t nt = r.U32(0x3c);
    if (r.U32(nt) != 0x00004550) {
        throw std::runtime_error(r.mWhat);
    }
    const uint32_t sections = r.U16(nt + 6), optionalSize = r.U16(nt + 20);
    const uint64_t optional = nt + 24;
    const uint16_t magic = r.U16(optional);
    if (magic != 0x10b && magic != 0x20b) {
        throw std::runtime_error(r.mWhat);
    }
    const uint32_t resourceRva = r.U32(optional + (magic == 0x20b ? 112 : 96) + 2 * 8);
    const std::runtime_error notOurs("this DLL has no DLSS-NR weights (expected nvngx_dlssnr.dll)");
    if (!resourceRva) {
        throw notOurs;
    }
    auto offsetOf = [&](uint32_t rva) -> uint64_t {
        for (uint32_t i = 0; i < sections; ++i) {
            const uint64_t s = optional + optionalSize + (uint64_t)i * 40;
            const uint32_t virtualSize = r.U32(s + 8), virtualAddress = r.U32(s + 12), rawSize = r.U32(s + 16),
                           raw = r.U32(s + 20);
            if (rva >= virtualAddress && rva - virtualAddress < std::max(virtualSize, rawSize)) {
                return (uint64_t)raw + (rva - virtualAddress);
            }
        }
        throw std::runtime_error(r.mWhat);
    };
    const uint64_t root = offsetOf(resourceRva);
    // One level of the resource directory: the entry whose id or name matches (any entry when both are empty).
    auto child = [&](uint64_t directory, uint32_t id, const std::u16string* wanted) -> uint32_t {
        const uint32_t entries = (uint32_t)r.U16(directory + 12) + r.U16(directory + 14);
        for (uint32_t i = 0; i < entries; ++i) {
            const uint32_t key = r.U32(directory + 16 + (uint64_t)i * 8),
                           data = r.U32(directory + 20 + (uint64_t)i * 8);
            if (!wanted && id == 0) {
                return data;
            }
            if (key & 0x80000000u) {
                if (!wanted) {
                    continue;
                }
                const uint64_t at = root + (key & 0x7fffffffu);
                const uint32_t length = r.U16(at);
                if (length != wanted->size()) {
                    continue;
                }
                bool same = true;
                for (uint32_t c = 0; c < length && same; ++c) {
                    same = r.U16(at + 2 + (uint64_t)c * 2) == (uint16_t)(*wanted)[c];
                }
                if (same) {
                    return data;
                }
            } else if (!wanted && key == id) {
                return data;
            }
        }
        throw notOurs;
    };
    uint32_t node = child(root, 10 /* RT_RCDATA */, nullptr);
    node = child(root + (node & 0x7fffffffu), 0, &name);
    node = child(root + (node & 0x7fffffffu), 0, nullptr);
    if (node & 0x80000000u) {
        throw notOurs;
    }
    const uint64_t offset = offsetOf(r.U32(root + node)), size = r.U32(root + node + 4);
    r.Need(offset, size);
    return { offset, size };
}

} // namespace

ExtractSummary ExtractModel(const std::filesystem::path& dll, const std::filesystem::path& directory) {
    const std::vector<uint8_t> pe = ReadFile(dll);
    const auto [archiveAt, archiveSize] = FindRcData(pe, u"WEIGHTS_HT");
    const std::vector<uint8_t> archive(pe.begin() + (ptrdiff_t)archiveAt,
                                       pe.begin() + (ptrdiff_t)(archiveAt + archiveSize));
    const Reader r{ archive, "the weights in this DLL are not laid out as expected (unsupported DLSS-NR build)" };
    if (r.U64(0) != archive.size()) {
        throw std::runtime_error(r.mWhat);
    }

    // Records: u64 name length, name, u64 body span; body = u64 span, u64 payload size, u32 type, payload, 5 x u32.
    nlohmann::json tensors = nlohmann::json::array();
    std::vector<uint8_t> blob;
    ExtractSummary summary;
    const std::regex pattern("block([0-9]+)\\.layer([0-9]+)\\.([A-Za-z0-9_]+)");
    for (uint64_t cursor = 8; cursor < archive.size();) {
        const uint64_t nameLength = r.U64(cursor);
        if (nameLength < 1 || nameLength > 256) {
            throw std::runtime_error(r.mWhat);
        }
        r.Need(cursor + 8, nameLength);
        const std::string name((const char*)archive.data() + cursor + 8, (size_t)nameLength);
        cursor += 8 + nameLength;
        const uint64_t span = r.U64(cursor), body = cursor + 8;
        const uint64_t payloadSize = r.U64(body + 8);
        if (payloadSize > archive.size() || span != payloadSize + 40) {
            throw std::runtime_error(r.mWhat);
        }
        r.Need(body, span);
        std::smatch match;
        if (!std::regex_match(name, match, pattern)) {
            throw std::runtime_error(r.mWhat);
        }
        const int block = std::stoi(match[1].str());
        tensors.push_back({ { "name", name },
                            { "block", block },
                            { "layer", std::stoi(match[2].str()) },
                            { "parameter", match[3].str() },
                            { "stage", "all" },
                            { "stageOffset", blob.size() },
                            { "byteLength", payloadSize } });
        blob.insert(blob.end(), archive.begin() + (ptrdiff_t)(body + 20),
                    archive.begin() + (ptrdiff_t)(body + 20 + payloadSize));
        blob.resize((blob.size() + 15) & ~size_t(15));
        summary.mTensors += 1;
        summary.mBlocks = std::max(summary.mBlocks, (uint32_t)block + 1);
        summary.mPayloadBytes += payloadSize;
        cursor = body + span;
    }
    if (summary.mBlocks != Model::kBlocks || summary.mTensors != Model::kTensors) {
        throw std::runtime_error("this DLL carries a different DLSS-NR network (" + std::to_string(summary.mBlocks) +
                                 " blocks); this build supports the 71-block one of version 310.8.0");
    }
    summary.mSourceSha256 = Sha256Hex(pe.data(), pe.size());

    nlohmann::json manifest = {
        { "source", { { "sha256", summary.mSourceSha256 }, { "resourceBytes", archiveSize } } },
        { "totals",
          { { "blockCount", summary.mBlocks },
            { "tensorCount", summary.mTensors },
            { "payloadBytes", summary.mPayloadBytes } } },
        { "stages", nlohmann::json::array({ { { "id", "all" },
                                              { "file", "weights.bin" },
                                              { "packedByteLength", blob.size() },
                                              { "sha256", Sha256Hex(blob.data(), blob.size()) } } }) },
        { "tensors", tensors },
    };
    std::filesystem::create_directories(directory / "model");
    {
        std::ofstream file(directory / "model" / "weights.bin", std::ios::binary | std::ios::trunc);
        file.write((const char*)blob.data(), (std::streamsize)blob.size());
        if (!file) {
            throw std::runtime_error("cannot write the model cache (disk full?)");
        }
    }
    // The manifest goes last: a cache without one is "not built", never half-built.
    std::ofstream file(directory / "manifest.json", std::ios::trunc);
    file << manifest.dump(1) << "\n";
    if (!file) {
        throw std::runtime_error("cannot write the model cache (disk full?)");
    }
    return summary;
}

} // namespace Fast::Neural
