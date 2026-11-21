// The DLSS-NR network as FP16 cooperative-matrix kernels (the AMD / generic path): the 71 blocks of the graph as
// a list of compute dispatches over half-float activations, with the weights re-laid out into one arena.
//
// The data flow is NrReference's (which documents it); this file decides how it runs: which kernel computes what,
// which tensors are materialized, how the weights are laid out for the loads RADV generates.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "NrLayout.h"
#include "NrModel.h"
#include "NrVk.h"

namespace Fast::Neural {

class Graph {
  public:
    // A half-float tensor [rows][channels]. Spatial ones (width > 0) hold their tokens in 4x4 pixel tiles.
    struct Activation {
        std::string mName;
        Buffer mBuffer;
        uint32_t mWidth = 0, mHeight = 0;
        uint32_t mRows = 0, mChannels = 0;
    };

    // One step of the frame: a block, or the transition between two stages.
    struct Segment {
        std::string mLabel;
        std::string mBoundary;        // the reference's name for the state this step leaves ("" if none)
        Activation* mState = nullptr; // where that state is once the step has run
        // What the step reads from earlier steps, by the boundary name that produced it (for checking one step
        // at a time against the reference).
        std::vector<std::pair<Activation*, std::string>> mInputs;
        size_t mFirstOp = 0, mOpCount = 0;
        double mGmac = 0.0; // multiply-accumulates, in units of 1e9
        bool mVit = false;  // one of the eight global-attention blocks (skippable: ViT reuse)
    };

    Graph(Context& context, const Model& model, const Geometry& geometry);
    ~Graph();
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    const Geometry& GetGeometry() const {
        return mGeometry;
    }
    const std::vector<Segment>& Segments() const {
        return mSegments;
    }
    double Gmac() const;
    uint64_t WeightBytes() const {
        return mArenaBytes;
    }
    uint64_t ActivationBytes() const;
    size_t Dispatches() const {
        return mOps.size();
    }

    // Input: [field][16] f32 in row-major token order. Output: the head, [field][4].
    void SetFeatures(const std::vector<float>& features);
    std::vector<float> Head();

    // Records segments [first, first + count) into `commands`. With `stamps`, a timestamp follows each segment
    // (query i + 1 after segment first + i; query 0 before the first). `reuseVit` leaves the ViT's eight blocks out:
    // the decoder then reads the global context of the last frame that ran them.
    void Record(VkCommandBuffer commands, size_t first = 0, size_t count = SIZE_MAX, bool stamps = false,
                bool reuseVit = false);

    // The column-tile count of each GEMM shape, chosen by timing it on this GPU (a few hundred milliseconds).
    void Tune();

    // Tensors in row-major token order, as the reference holds them.
    std::vector<float> Read(const Activation& activation);
    void Write(const Activation& activation, const std::vector<float>& values);

  private:
    struct Op {
        enum class Kind { Gemm, Block32, Attention, VitNormalize, VitAttention, Ops } mKind;
        const Buffer* mBuffers[6] = {}; // bindings 0-5; 1 is the weight arena unless set
        VkDescriptorSet mSet = VK_NULL_HANDLE;
        uint32_t mPush[16] = {};
        uint32_t mPushBytes = 0;
        uint32_t mGroups[3] = { 1, 1, 1 };
        uint32_t mSpec = 0;                 // Gemm, Block32: flags; Ops: mode
        uint32_t mRows = 0, mK = 0, mN = 0; // Gemm: the shape (its pipeline and x count depend on the tile choice)
    };
    using GemmKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>; // rows, K, N, flags, batches

    Activation* Allocate(const std::string& name, uint32_t width, uint32_t height, uint32_t channels,
                         uint32_t rows = 0);
    uint32_t AddMatrix(const std::vector<float>& plain, uint32_t K, uint32_t N, uint32_t batchK = 0);
    uint32_t AddHalves(const std::vector<float>& values);
    void Begin(const std::string& label);
    void End(const std::string& boundary, Activation* state);
    void Input(Activation* activation, const std::string& boundary);

    struct GemmArgs {
        uint32_t mRows = 0, mK = 0, mN = 0, mFlags = 0;
        Activation* mInput = nullptr;
        uint32_t mInCol = 0;
        Activation* mOutput = nullptr;    // the half (or, with F_QUANT alone, its E4M3 publication)
        Activation* mQuantized = nullptr; // F_DUAL: the E4M3 publication
        uint32_t mOutCol = 0;
        uint32_t mWeights = 0;
        Activation* mResidual = nullptr;
        uint32_t mScale = 0;
        uint32_t mBatches = 1, mBatchIn = 0, mBatchOut = 0;
        uint32_t mValidN = 0; // 0: all N
    };
    void AddGemm(const GemmArgs& args);
    void AddBlock32(const BlockLayout& layout, const Tensor& tensor, int level, Activation* state, Activation* skip,
                    Activation* out, Activation* raw);
    void AddAttention(int level, uint32_t channels, uint32_t phase, Activation* qkv, Activation* out, uint32_t prior,
                      uint32_t scales);
    void AddOps(uint32_t mode, Activation* in, Activation* skip, Activation* out, Activation* raw, uint32_t inWidth4,
                uint32_t inWidth, uint32_t inHeight, uint32_t scaleA, uint32_t scaleB);
    void AddWideBlock(const BlockLayout& layout, const Tensor& tensor, int level, Activation* state, Activation* out,
                      Activation* raw);
    void AddSplitBlock(int block, Activation* state, Activation* out, Activation* raw);
    void AddVitBlock(int block, Activation* state, Activation* out);
    void Build();

    Pipeline& GemmPipeline(uint32_t K, uint32_t tiles, uint32_t flags);
    uint32_t TilesFor(const Op& op) const;
    void Dispatch(VkCommandBuffer commands, const Op& op);

    Context& mContext;
    const Model& mModel;
    Geometry mGeometry;
    std::vector<std::unique_ptr<Activation>> mActivations;
    std::map<std::string, Activation*> mByName;
    std::vector<uint16_t> mArena; // released once uploaded
    uint64_t mArenaBytes = 0;
    Buffer mWeights;
    Buffer mFeatures;
    Activation* mHead = nullptr;
    std::vector<Op> mOps;
    std::vector<Segment> mSegments;
    uint32_t mPhase[7] = {};
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Pipeline> mGemmPipelines;
    std::map<std::pair<int, uint32_t>, Pipeline> mPipelines; // (kind, specialization)
    std::map<GemmKey, uint32_t> mTiles;
};

} // namespace Fast::Neural
