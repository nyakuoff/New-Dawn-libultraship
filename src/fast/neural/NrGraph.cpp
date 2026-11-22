#include "NrGraph.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "NrNumeric.h"
#include "NrSpirv.inc"

namespace Fast::Neural {

namespace {

constexpr uint32_t kResidual = 1, kSilu = 2, kQuant = 4, kDual = 8; // gemm.comp
constexpr uint32_t kSkip = 1, kRawOut = 2;                          // block32.comp
constexpr uint32_t kFeatures = 0, kPool = 1, kMerge = 2, kPost = 3; // ops.comp

uint32_t TiledIndex(uint32_t x, uint32_t y, uint32_t width) {
    return ((y >> 2) * (width >> 2) + (x >> 2)) * 16 + (y & 3) * 4 + (x & 3);
}

} // namespace

Graph::Graph(Context& context, const Model& model, const Geometry& geometry, bool externalFeatures)
    : mContext(context), mModel(model), mGeometry(geometry), mExternalFeatures(externalFeatures) {
    Build();
    mArenaBytes = (uint64_t)mArena.size() * 2;
    mWeights = mContext.CreateBuffer(mArenaBytes);
    mContext.Upload(mWeights, 0, mArena.data(), mArenaBytes);
    mArena = {};

    // Descriptor sets, now that the arena exists: it is binding 1 of every kernel.
    for (Op& op : mOps) {
        if (!op.mBuffers[1]) {
            op.mBuffers[1] = &mWeights;
        }
        op.mSet = mContext.Bind(std::vector<const Buffer*>(op.mBuffers, op.mBuffers + 6));
    }
}

Graph::~Graph() {
    mContext.WaitIdle();
    for (auto& activation : mActivations) {
        mContext.DestroyBuffer(activation->mBuffer);
    }
    mContext.DestroyBuffer(mWeights);
    mContext.DestroyBuffer(mFeatures);
    for (auto& [key, pipeline] : mGemmPipelines) {
        mContext.DestroyPipeline(pipeline);
    }
    for (auto& [key, pipeline] : mPipelines) {
        mContext.DestroyPipeline(pipeline);
    }
}

double Graph::Gmac() const {
    double total = 0.0;
    for (const Segment& segment : mSegments) {
        total += segment.mGmac;
    }
    return total;
}

uint64_t Graph::ActivationBytes() const {
    uint64_t total = mFeatures.mSize;
    for (const auto& activation : mActivations) {
        total += activation->mBuffer.mSize;
    }
    return total;
}

Graph::Activation* Graph::Allocate(const std::string& name, uint32_t width, uint32_t height, uint32_t channels,
                                   uint32_t rows) {
    auto found = mByName.find(name);
    if (found != mByName.end()) {
        return found->second;
    }
    auto activation = std::make_unique<Activation>();
    activation->mName = name;
    activation->mWidth = width;
    activation->mHeight = height;
    activation->mRows = rows ? rows : width * height;
    activation->mChannels = channels;
    // Rows are whole 16-token tiles; one spare tile row keeps the 64-row GEMM groups inside the buffer.
    activation->mBuffer = mContext.CreateBuffer(((uint64_t)AlignUp(activation->mRows, 64)) * channels * 2);
    mByName[name] = activation.get();
    mActivations.push_back(std::move(activation));
    return mActivations.back().get();
}

// A [K][N] matrix (or `K / batchK` of them, stacked) into the arena as k16-tile-major halves: [batch][K/16][N][16].
// A 16x16 B tile is then 512 contiguous bytes and each lane's column two 16-byte reads.
uint32_t Graph::AddMatrix(const std::vector<float>& plain, uint32_t K, uint32_t N, uint32_t batchK) {
    if (!batchK) {
        batchK = K;
    }
    const size_t base = mArena.size();
    mArena.resize(base + (size_t)K * N, 0);
    for (uint32_t j = 0; j < K; ++j) {
        const uint32_t batch = j / batchK, k = j % batchK;
        const size_t tile = base + (size_t)batch * batchK * N + (size_t)(k / 16) * N * 16;
        for (uint32_t n = 0; n < N; ++n) {
            mArena[tile + (size_t)n * 16 + (k & 15)] = Num::F16Bits(plain[(size_t)j * N + n]);
        }
    }
    return (uint32_t)base;
}

uint32_t Graph::AddHalves(const std::vector<float>& values) {
    const size_t base = mArena.size();
    mArena.resize(base + AlignUp((uint32_t)values.size(), 16), 0);
    for (size_t i = 0; i < values.size(); ++i) {
        mArena[base + i] = Num::F16Bits(values[i]);
    }
    return (uint32_t)base;
}

void Graph::Begin(const std::string& label) {
    Segment segment;
    segment.mLabel = label;
    segment.mFirstOp = mOps.size();
    mSegments.push_back(segment);
}

void Graph::End(const std::string& boundary, Activation* state) {
    Segment& segment = mSegments.back();
    segment.mBoundary = boundary;
    segment.mState = state;
    segment.mOpCount = mOps.size() - segment.mFirstOp;
}

void Graph::Input(Activation* activation, const std::string& boundary) {
    mSegments.back().mInputs.push_back({ activation, boundary });
}

// ---- Kernels
// ---------------------------------------------------------------------------------------------------------

void Graph::AddGemm(const GemmArgs& g) {
    Op op;
    op.mKind = Op::Kind::Gemm;
    op.mSpec = g.mFlags;
    op.mRows = g.mRows;
    op.mK = g.mK;
    op.mN = g.mN;
    const uint32_t push[14] = { g.mRows,
                                g.mN,
                                g.mValidN ? g.mValidN : g.mN,
                                g.mInput->mChannels,
                                g.mInCol,
                                g.mOutput->mChannels,
                                g.mOutCol,
                                g.mResidual ? g.mResidual->mChannels : 0,
                                0,
                                g.mWeights,
                                g.mScale,
                                g.mBatchIn,
                                g.mBatchOut,
                                g.mK * g.mN };
    memcpy(op.mPush, push, sizeof push);
    op.mPushBytes = sizeof push;
    op.mGroups[1] = AlignUp(g.mRows, 64) / 64;
    op.mGroups[2] = g.mBatches;
    op.mBuffers[0] = &g.mInput->mBuffer;
    op.mBuffers[2] = &g.mOutput->mBuffer;
    op.mBuffers[3] = g.mResidual ? &g.mResidual->mBuffer : nullptr;
    op.mBuffers[5] = g.mQuantized ? &g.mQuantized->mBuffer : nullptr;
    mOps.push_back(op);
    mSegments.back().mGmac += (double)g.mRows * g.mK * g.mN * g.mBatches * 1e-9;
}

void Graph::AddBlock32(const BlockLayout& l, const Tensor& t, int level, Activation* state, Activation* skip,
                       Activation* out, Activation* raw) {
    const uint32_t width = state->mWidth, height = state->mHeight;
    uint32_t shiftX, shiftY;
    WindowPhase(mPhase[level]++, shiftX, shiftY);
    Op op;
    op.mKind = Op::Kind::Block32;
    op.mSpec = (skip ? kSkip : 0) | (raw ? kRawOut : 0);
    struct {
        uint32_t mSize4[2], mShift[2], mW1, mW2, mQkv, mProj, mFfnScale, mAttnScale, mPrior;
        float mQScale;
    } push = { { width / 4, height / 4 },
               { shiftX / 4, shiftY / 4 },
               AddMatrix(mModel.Fp8Matrix(t, l.mExpand, 32, 128), 32, 128),
               AddMatrix(mModel.Fp8Matrix(t, l.mContract, 128, 32), 128, 32),
               AddMatrix(mModel.Fp8Matrix(t, l.mQkv, 32, 96), 32, 96),
               AddMatrix(mModel.Fp8Matrix(t, l.mProjection, 32, 32), 32, 32),
               AddHalves(mModel.Halves(t, l.mFfnScale, 32)),
               AddHalves(mModel.Halves(t, l.mAttnScale, 32)),
               AddHalves(mModel.Prior(t, l.mPrior, 1)),
               mModel.Floats(t, l.mQScale, 1)[0] };
    memcpy(op.mPush, &push, sizeof push);
    op.mPushBytes = sizeof push;
    op.mGroups[0] = (width + shiftX + 7) / 8;
    op.mGroups[1] = (height + shiftY + 7) / 8;
    op.mBuffers[0] = &state->mBuffer;
    op.mBuffers[2] = &out->mBuffer;
    op.mBuffers[3] = skip ? &skip->mBuffer : nullptr;
    op.mBuffers[4] = raw ? &raw->mBuffer : nullptr;
    mOps.push_back(op);
    mSegments.back().mGmac += (double)width * height * 16384 * 1e-9;
}

void Graph::AddAttention(int level, uint32_t channels, uint32_t phase, Activation* qkv, Activation* out, uint32_t prior,
                         uint32_t scales) {
    const uint32_t width = mGeometry.mWidth[level], height = mGeometry.mHeight[level];
    uint32_t shiftX, shiftY;
    WindowPhase(phase, shiftX, shiftY);
    Op op;
    op.mKind = Op::Kind::Attention;
    const uint32_t push[7] = { width / 4, height / 4, shiftX / 4, shiftY / 4, channels, prior, scales };
    memcpy(op.mPush, push, sizeof push);
    op.mPushBytes = sizeof push;
    op.mGroups[0] = (width + shiftX + 7) / 8;
    op.mGroups[1] = (height + shiftY + 7) / 8;
    op.mGroups[2] = channels / 32;
    op.mBuffers[0] = &qkv->mBuffer;
    op.mBuffers[2] = &out->mBuffer;
    mOps.push_back(op);
    mSegments.back().mGmac += (double)width * height * (channels / 32) * 4096 * 1e-9;
}

void Graph::AddOps(uint32_t mode, Activation* in, Activation* skip, Activation* out, Activation* raw, uint32_t inWidth4,
                   uint32_t inWidth, uint32_t inHeight, uint32_t scaleA, uint32_t scaleB) {
    Op op;
    op.mKind = Op::Kind::Ops;
    op.mSpec = mode;
    const uint32_t groups = out->mRows * out->mChannels / 8;
    op.mGroups[0] = std::min(AlignUp(groups, 64) / 64, 2048u);
    op.mGroups[1] = (groups + op.mGroups[0] * 64 - 1) / (op.mGroups[0] * 64);
    const uint32_t push[11] = { groups,
                                op.mGroups[0] * 64,
                                out->mChannels,
                                out->mWidth / 4,
                                inWidth4,
                                0,
                                inWidth,
                                inHeight,
                                scaleA,
                                scaleB,
                                raw && mode == kMerge ? 1u : 0u };
    memcpy(op.mPush, push, sizeof push);
    op.mPushBytes = sizeof push;
    op.mBuffers[0] = in ? &in->mBuffer : &mFeatures;
    op.mBuffers[2] = &out->mBuffer;
    op.mBuffers[3] = skip ? &skip->mBuffer : nullptr;
    op.mBuffers[5] = raw ? &raw->mBuffer : nullptr;
    mOps.push_back(op);
}

// ---- Blocks
// ----------------------------------------------------------------------------------------------------------

// C / 32 expert paths C -> 128 -> 32 (every path reads the whole state), concatenated, then W3 with the scaled skip;
// QKV; window attention; projection with the FFN's E4M3 publication as the skip.
void Graph::AddWideBlock(const BlockLayout& l, const Tensor& t, int level, Activation* state, Activation* out,
                         Activation* raw) {
    const uint32_t C = l.mChannels, E = l.mExperts, width = mGeometry.mWidth[level], height = mGeometry.mHeight[level];
    const uint32_t rows = width * height;
    const std::string stage = "level " + std::to_string(level) + " ";
    Activation* hidden = Allocate(stage + "hidden", width, height, E * 128);
    Activation* narrow = Allocate(stage + "narrow", width, height, C);
    Activation* ffn = Allocate(stage + "ffn", width, height, C);
    Activation* qkv = Allocate(stage + "qkv", width, height, 3 * C);
    Activation* attended = Allocate(stage + "attended", width, height, C);

    GemmArgs expand;
    expand.mRows = rows;
    expand.mK = C;
    expand.mN = 128;
    expand.mFlags = kSilu | kQuant;
    expand.mInput = state;
    expand.mOutput = hidden;
    expand.mBatches = E;
    expand.mBatchOut = 128;
    expand.mWeights = AddMatrix(mModel.Fp8Matrix(t, l.mExpand, E * C, 128), E * C, 128, C);
    AddGemm(expand);
    GemmArgs contract;
    contract.mRows = rows;
    contract.mK = 128;
    contract.mN = 32;
    contract.mFlags = kQuant;
    contract.mInput = hidden;
    contract.mOutput = narrow;
    contract.mBatches = E;
    contract.mBatchIn = 128;
    contract.mBatchOut = 32;
    contract.mWeights = AddMatrix(mModel.Fp8Matrix(t, l.mContract, E * 128, 32), E * 128, 32, 128);
    AddGemm(contract);
    GemmArgs mix;
    mix.mRows = rows;
    mix.mK = C;
    mix.mN = C;
    mix.mFlags = kResidual | kQuant;
    mix.mInput = narrow;
    mix.mOutput = ffn;
    mix.mResidual = state;
    mix.mWeights = AddMatrix(mModel.Fp8Matrix(t, l.mW3, C, C), C, C);
    mix.mScale = AddHalves(mModel.Halves(t, l.mFfnScale, C));
    AddGemm(mix);

    GemmArgs project;
    project.mRows = rows;
    project.mK = C;
    project.mN = 3 * C;
    project.mInput = ffn;
    project.mOutput = qkv;
    project.mWeights = AddMatrix(mModel.Fp8Matrix(t, l.mQkv, C, 3 * C), C, 3 * C);
    AddGemm(project);
    std::vector<float> scales = mModel.Floats(t, l.mQScale, l.mHeads);
    AddAttention(level, C, mPhase[level]++, qkv, attended, AddHalves(mModel.Prior(t, l.mPrior, l.mHeads)),
                 AddHalves(scales));

    GemmArgs projection;
    projection.mRows = rows;
    projection.mK = C;
    projection.mN = C;
    projection.mFlags = kResidual | (raw ? kDual : kQuant);
    projection.mInput = attended;
    projection.mResidual = ffn;
    projection.mOutput = raw ? raw : out;
    projection.mQuantized = raw ? out : nullptr;
    projection.mWeights = AddMatrix(mModel.Fp8Matrix(t, l.mProjection, C, C), C, C);
    projection.mScale = AddHalves(mModel.Halves(t, l.mAttnScale, C));
    AddGemm(projection);
}

// The 512 stage: 512 -> 512, eight 64 -> 256 -> 64 branches (SiLU on the middle only), a 512 -> 512 contraction with
// the skip; then 16-head window attention and the projection.
void Graph::AddSplitBlock(int block, Activation* state, Activation* out, Activation* raw) {
    const uint32_t width = mGeometry.mWidth[4], height = mGeometry.mHeight[4], rows = width * height;
    const Tensor& branches = mModel.Get(block, 0);
    const Tensor& contractTensor = mModel.Get(block, 1);
    const Tensor& attention = mModel.Get(block, 2);
    const Tensor& projectionTensor = mModel.Get(block, 3);
    Activation* branch = Allocate("level 4 branch", width, height, 512);
    Activation* middle = Allocate("level 4 middle", width, height, 2048);
    Activation* layer0 = Allocate("level 4 layer0", width, height, 512);
    Activation* ffn = Allocate("level 4 ffn", width, height, 512);
    Activation* qkv = Allocate("level 4 qkv", width, height, 1536);
    Activation* attended = Allocate("level 4 attended", width, height, 512);

    GemmArgs first;
    first.mRows = rows;
    first.mK = 512;
    first.mN = 512;
    first.mFlags = kQuant;
    first.mInput = state;
    first.mOutput = branch;
    first.mWeights = AddMatrix(mModel.Fp8Matrix(branches, 0, 512, 512), 512, 512);
    AddGemm(first);
    GemmArgs up;
    up.mRows = rows;
    up.mK = 64;
    up.mN = 256;
    up.mFlags = kSilu | kQuant;
    up.mInput = branch;
    up.mOutput = middle;
    up.mBatches = 8;
    up.mBatchIn = 64;
    up.mBatchOut = 256;
    up.mWeights = AddMatrix(mModel.Fp8Matrix(branches, 512 * 512, 8 * 64, 256), 8 * 64, 256, 64);
    AddGemm(up);
    GemmArgs down;
    down.mRows = rows;
    down.mK = 256;
    down.mN = 64;
    down.mFlags = kQuant;
    down.mInput = middle;
    down.mOutput = layer0;
    down.mBatches = 8;
    down.mBatchIn = 256;
    down.mBatchOut = 64;
    down.mWeights = AddMatrix(mModel.Fp8Matrix(branches, 512 * 512 + 8 * 64 * 256, 8 * 256, 64), 8 * 256, 64, 256);
    AddGemm(down);
    GemmArgs mix;
    mix.mRows = rows;
    mix.mK = 512;
    mix.mN = 512;
    mix.mFlags = kResidual | kQuant;
    mix.mInput = layer0;
    mix.mOutput = ffn;
    mix.mResidual = state;
    mix.mWeights = AddMatrix(mModel.Fp8Matrix(contractTensor, 0, 512, 512), 512, 512);
    mix.mScale = AddHalves(mModel.Halves(contractTensor, 512 * 512, 512));
    AddGemm(mix);

    GemmArgs project;
    project.mRows = rows;
    project.mK = 512;
    project.mN = 1536;
    project.mInput = ffn;
    project.mOutput = qkv;
    project.mWeights = AddMatrix(mModel.Fp8Matrix(attention, 0, 512, 1536), 512, 1536);
    AddGemm(project);
    AddAttention(4, 512, mPhase[4]++, qkv, attended, AddHalves(mModel.Prior(attention, 512 * 1536, 16)),
                 AddHalves(mModel.Floats(attention, 512 * 1536 + 16 * 8192, 16)));

    GemmArgs projection;
    projection.mRows = rows;
    projection.mK = 512;
    projection.mN = 512;
    projection.mFlags = kResidual | (raw ? kDual : kQuant);
    projection.mInput = attended;
    projection.mResidual = ffn;
    projection.mOutput = raw ? raw : out;
    projection.mQuantized = raw ? out : nullptr;
    projection.mWeights = AddMatrix(mModel.Fp8Matrix(projectionTensor, 0, 512, 512), 512, 512);
    projection.mScale = AddHalves(mModel.Halves(projectionTensor, 512 * 512, 512));
    AddGemm(projection);
}

// A ViT block: dense 1024 -> 4096 -> 1024 FFN with the skip, QKV, normalization, global attention, projection with
// the FFN's publication as the skip. `out` may be `state` (the block's own input is dead by its last GEMM).
void Graph::AddVitBlock(int block, Activation* state, Activation* out) {
    const uint32_t width = mGeometry.mWidth[5], height = mGeometry.mHeight[5], tokens = width * height;
    const uint32_t padded = mGeometry.PaddedVitTokens();
    const Tensor& expandTensor = mModel.Get(block, 0);
    const Tensor& contractTensor = mModel.Get(block, 1);
    const Tensor& attention = mModel.Get(block, 2);
    const Tensor& projectionTensor = mModel.Get(block, 4);
    Activation* expanded = Allocate("vit expanded", width, height, 4096);
    Activation* ffn = Allocate("vit ffn", width, height, 1024);
    Activation* qkv = Allocate("vit qkv", width, height, 3072);
    Activation* queries = Allocate("vit q", width, height, 1024);
    Activation* keys = Allocate("vit k", 0, 0, 1024, padded);   // rows past `tokens` stay zero: the padding keys
    Activation* values = Allocate("vit v", 0, 0, padded, 1024); // transposed: [channel][key]
    Activation* attended = Allocate("vit attended", width, height, 1024);

    GemmArgs expand;
    expand.mRows = tokens;
    expand.mK = 1024;
    expand.mN = 4096;
    expand.mFlags = kSilu | kQuant;
    expand.mInput = state;
    expand.mOutput = expanded;
    expand.mWeights = AddMatrix(mModel.Fp8Matrix(expandTensor, 0, 1024, 4096), 1024, 4096);
    AddGemm(expand);
    GemmArgs contract;
    contract.mRows = tokens;
    contract.mK = 4096;
    contract.mN = 1024;
    contract.mFlags = kResidual | kQuant;
    contract.mInput = expanded;
    contract.mOutput = ffn;
    contract.mResidual = state;
    contract.mWeights = AddMatrix(mModel.Fp8Matrix(contractTensor, 0, 4096, 1024), 4096, 1024);
    contract.mScale = AddHalves(mModel.Halves(contractTensor, 4096 * 1024, 1024));
    AddGemm(contract);
    GemmArgs project;
    project.mRows = tokens;
    project.mK = 1024;
    project.mN = 3072;
    project.mInput = ffn;
    project.mOutput = qkv;
    project.mWeights = AddMatrix(mModel.Fp8Matrix(attention, 32 * 4, 1024, 3072), 1024, 3072);
    AddGemm(project);

    Op normalize;
    normalize.mKind = Op::Kind::VitNormalize;
    const uint32_t normalizePush[3] = { tokens, padded, AddHalves(mModel.Floats(attention, 0, 32)) };
    memcpy(normalize.mPush, normalizePush, sizeof normalizePush);
    normalize.mPushBytes = sizeof normalizePush;
    normalize.mGroups[0] = tokens * 32 / 64;
    normalize.mBuffers[0] = &qkv->mBuffer;
    normalize.mBuffers[2] = &queries->mBuffer;
    normalize.mBuffers[3] = &keys->mBuffer;
    normalize.mBuffers[4] = &values->mBuffer;
    mOps.push_back(normalize);
    Op attend;
    attend.mKind = Op::Kind::VitAttention;
    const uint32_t attendPush[2] = { tokens, padded };
    memcpy(attend.mPush, attendPush, sizeof attendPush);
    attend.mPushBytes = sizeof attendPush;
    attend.mGroups[0] = 32;
    attend.mGroups[1] = AlignUp(tokens, 64) / 64;
    attend.mBuffers[0] = &queries->mBuffer;
    attend.mBuffers[1] = &keys->mBuffer; // this kernel reads no weights: binding 1 is its K
    attend.mBuffers[2] = &attended->mBuffer;
    attend.mBuffers[3] = &values->mBuffer;
    mOps.push_back(attend);
    mSegments.back().mGmac += (double)tokens * padded * 32 * 32 * 2 * 1e-9;

    GemmArgs projection;
    projection.mRows = tokens;
    projection.mK = 1024;
    projection.mN = 1024;
    projection.mFlags = kResidual | kQuant;
    projection.mInput = attended;
    projection.mOutput = out;
    projection.mResidual = ffn;
    projection.mWeights = AddMatrix(mModel.Fp8Matrix(projectionTensor, 0, 1024, 1024), 1024, 1024);
    projection.mScale = AddHalves(mModel.Halves(projectionTensor, 1024 * 1024, 1024));
    AddGemm(projection);
}

// ---- The graph
// ---------------------------------------------------------------------------------------------------------

void Graph::Build() {
    const Geometry& g = mGeometry;
    const uint32_t FW = g.mFullWidth, FH = g.mFullHeight;
    mFeatures = mContext.CreateBuffer(mExternalFeatures ? 16 : (uint64_t)g.FieldTokens() * 16 * 4);

    // Block 0 at full resolution: features -> halves -> the f16 16 -> 32 input adapter (its half output is the FFN's
    // skip, its E4M3 publication the state) -> the block. Its E4M3 output is kept for block 70, its half output
    // feeds the pool.
    const Tensor& pre = mModel.Get(0);
    const BlockLayout preLayout = PreLayout();
    Activation* featureHalves = Allocate("features", FW, FH, 16);
    Activation* adapterRaw = Allocate("adapter raw", FW, FH, 32);
    Activation* adapter = Allocate("adapter", FW, FH, 32);
    Activation* block0 = Allocate("block 0", FW, FH, 32);
    Activation* fieldRaw = Allocate("field raw", FW, FH, 32);
    Begin("block 0");
    if (!mExternalFeatures) {
        AddOps(kFeatures, nullptr, nullptr, featureHalves, nullptr, 0, 0, 0, 0, 0);
    }
    {
        GemmArgs a;
        a.mRows = g.FieldTokens();
        a.mK = 16;
        a.mN = 32;
        a.mFlags = kDual;
        a.mInput = featureHalves;
        a.mOutput = adapterRaw;
        a.mQuantized = adapter;
        a.mWeights = AddMatrix(mModel.F16Matrix(pre, preLayout.mInputAdapter, 16, 32), 16, 32);
        AddGemm(a);
    }
    AddBlock32(preLayout, pre, 6, adapter, adapterRaw, block0, fieldRaw);
    End("block-0", block0);

    // Encoder. A stage's last block also leaves its half output, which is pooled into the next level and widened.
    Activation* skips[5] = {};
    Activation* state = Allocate("level 0 input", g.mWidth[0], g.mHeight[0], 32);
    Begin("transition 0-1");
    AddOps(kPool, fieldRaw, nullptr, state, nullptr, FW / 4, FW, FH, 0, 0);
    End("transition-0-1", state);
    std::string previous = "transition-0-1";
    for (const Stage& stage : kEncoderStages) {
        const uint32_t C = stage.mChannels, width = g.mWidth[stage.mLevel], height = g.mHeight[stage.mLevel];
        const std::string name = "level " + std::to_string(stage.mLevel) + " ";
        Activation* ping[2] = { Allocate(name + "state a", width, height, C),
                                Allocate(name + "state b", width, height, C) };
        Activation* raw = Allocate(name + "raw", width, height, C);
        skips[stage.mLevel] = Allocate(name + "skip", width, height, C);
        for (int block = stage.mFirst; block <= stage.mLast; ++block) {
            const bool last = block == stage.mLast;
            Activation* out = last ? skips[stage.mLevel] : ping[(block - stage.mFirst) & 1];
            Begin("block " + std::to_string(block));
            Input(state, previous);
            if (C == 32) {
                AddBlock32(StandardLayout(32), mModel.Get(block), stage.mLevel, state, nullptr, out,
                           last ? raw : nullptr);
            } else if (C == 512) {
                AddSplitBlock(block, state, out, last ? raw : nullptr);
            } else {
                AddWideBlock(StandardLayout(C), mModel.Get(block), stage.mLevel, state, out, last ? raw : nullptr);
            }
            previous = "block-" + std::to_string(block);
            End(previous, out);
            state = out;
        }
        const int next = stage.mLevel + 1;
        Activation* pooled = Allocate(name + "pooled", g.mWidth[next], g.mHeight[next], C);
        Activation* widened =
            Allocate("level " + std::to_string(next) + " input", g.mWidth[next], g.mHeight[next], 2 * C);
        Begin("transition " + std::to_string(stage.mLast) + "-" + std::to_string(stage.mLast + 1));
        AddOps(kPool, raw, nullptr, pooled, nullptr, width / 4, width, height, 0, 0);
        GemmArgs widen;
        widen.mRows = g.Tokens(next);
        widen.mK = C;
        widen.mN = 2 * C;
        widen.mFlags = kQuant;
        widen.mInput = pooled;
        widen.mOutput = widened;
        widen.mWeights =
            C == 512 ? AddMatrix(mModel.Fp8Matrix(mModel.Get(30, 4), 0, 512, 1024), 512, 1024)
                     : AddMatrix(mModel.Fp8Matrix(mModel.Get(stage.mLast), StandardLayout(C).mEnd, C, 2 * C), C, 2 * C);
        AddGemm(widen);
        previous = "transition-" + std::to_string(stage.mLast) + "-" + std::to_string(stage.mLast + 1);
        End(previous, widened);
        state = widened;
    }

    // ViT. Its state is its own buffer, not the encoder's output: a frame that skips these eight steps (ViT reuse)
    // then decodes with the previous frame's global context.
    Activation* vit = Allocate("vit state", g.mWidth[5], g.mHeight[5], 1024);
    for (int block = 31; block <= 38; ++block) {
        Begin("block " + std::to_string(block));
        Input(state, previous);
        AddVitBlock(block, state, vit);
        mSegments.back().mVit = true;
        previous = "block-" + std::to_string(block);
        End(previous, vit);
        state = vit;
    }

    // Decoder. Each stage starts with a 2C -> C GEMM at the low level (half output), a nearest 2x upsample and the
    // scaled encoder skip; the first 32-channel block takes the merge's half value as its FFN skip.
    static const int kSkipBlock[5] = { 4, 8, 14, 22, 30 };
    for (const Stage& stage : kDecoderStages) {
        const uint32_t C = stage.mChannels, width = g.mWidth[stage.mLevel], height = g.mHeight[stage.mLevel];
        const int low = stage.mLevel + 1;
        const std::string name = "level " + std::to_string(stage.mLevel) + " ";
        const BlockLayout first = C == 512 ? BlockLayout{} : UpsampleLayout(C);
        const Tensor& transition = mModel.Get(C == 512 ? 39 : stage.mFirst);
        Activation* projection = Allocate(name + "projection", g.mWidth[low], g.mHeight[low], C);
        Activation* merged = Allocate(name + "merged", width, height, C);
        Activation* mergedRaw = C == 32 ? Allocate(name + "merged raw", width, height, C) : nullptr;
        Begin("merge " + std::to_string(stage.mFirst));
        Input(state, previous);
        Input(skips[stage.mLevel], "block-" + std::to_string(kSkipBlock[stage.mLevel]));
        GemmArgs narrow;
        narrow.mRows = g.Tokens(low);
        narrow.mK = 2 * C;
        narrow.mN = C;
        narrow.mInput = state;
        narrow.mOutput = projection;
        narrow.mWeights = AddMatrix(mModel.Fp8Matrix(transition, C == 512 ? 0 : first.mUpsample, 2 * C, C), 2 * C, C);
        AddGemm(narrow);
        AddOps(kMerge, projection, skips[stage.mLevel], merged, mergedRaw, g.mWidth[low] / 4, 0, 0,
               AddHalves(mModel.Halves(transition, C == 512 ? 1024 * 512 : first.mTransitionScale, C)), 0);
        previous = C == 512 ? "block-39" : "merge-" + std::to_string(stage.mFirst);
        End(previous, merged);
        state = merged;

        Activation* ping[2] = { Allocate(name + "state a", width, height, C),
                                Allocate(name + "state b", width, height, C) };
        for (int block = stage.mFirst; block <= stage.mLast; ++block) {
            Activation* out = ping[(block - stage.mFirst) & 1];
            const bool head = block == stage.mFirst;
            Begin("block " + std::to_string(block));
            Input(state, previous);
            if (C == 32) {
                AddBlock32(head ? first : StandardLayout(32), mModel.Get(block), stage.mLevel, state,
                           head ? mergedRaw : nullptr, out, nullptr);
            } else if (C == 512) {
                AddSplitBlock(block, state, out, nullptr);
            } else {
                AddWideBlock(head ? first : StandardLayout(C), mModel.Get(block), stage.mLevel, state, out, nullptr);
            }
            previous = "block-" + std::to_string(block);
            End(previous, out);
            state = out;
        }
    }

    // Block 70 at full resolution: the post blend of the upsampled block 69 and block 0's output; the block; the f16
    // 32 -> 4 head on its half output. The head is stored as [field][8] halves, four of them valid.
    const Tensor& post = mModel.Get(70);
    const BlockLayout postLayout = PostLayout();
    // The adapter's two full-resolution tensors are dead since block 0: the blend takes their place, and block 70
    // publishes its (unused) E4M3 output over its own input, which a window kernel may do: a window reads only
    // its own tokens, and before it writes them.
    Activation* blend = adapter;
    Activation* blendRaw = adapterRaw;
    Begin("post blend");
    Input(state, previous);
    Input(block0, "block-0");
    AddOps(kPost, state, block0, blend, blendRaw, g.mWidth[0] / 4, 0, 0,
           AddHalves(mModel.Halves(post, postLayout.mInputScale, 32)),
           AddHalves(mModel.Halves(post, postLayout.mAdapterScale, 32)));
    End("merge-70", blend);
    mHead = Allocate("head", FW, FH, 8);
    Begin("block 70");
    Input(blend, "merge-70");
    AddBlock32(postLayout, post, 6, blend, blendRaw, blend, fieldRaw);
    {
        std::vector<float> weights = mModel.F16Matrix(post, postLayout.mHead, 32, 4), paddedWeights(32 * 16, 0.0f);
        for (uint32_t k = 0; k < 32; ++k) {
            std::copy(&weights[k * 4], &weights[k * 4] + 4, &paddedWeights[k * 16]);
        }
        GemmArgs head;
        head.mRows = g.FieldTokens();
        head.mK = 32;
        head.mN = 16;
        head.mValidN = 8;
        head.mInput = fieldRaw;
        head.mOutput = mHead;
        head.mWeights = AddMatrix(paddedWeights, 32, 16);
        AddGemm(head);
    }
    End("head", mHead);
}

// ---- Running
// ---------------------------------------------------------------------------------------------------------

Pipeline& Graph::GemmPipeline(uint32_t K, uint32_t tiles, uint32_t flags) {
    auto key = std::tuple{ K, tiles, flags };
    auto found = mGemmPipelines.find(key);
    if (found == mGemmPipelines.end()) {
        found =
            mGemmPipelines.emplace(key, mContext.CreateCompute(kGemmSpv, std::size(kGemmSpv), { K, tiles, flags }, 32))
                .first;
    }
    return found->second;
}

uint32_t Graph::TilesFor(const Op& op) const {
    auto found = mTiles.find(GemmKey{ op.mRows, op.mK, op.mN, op.mSpec, op.mGroups[2] });
    if (found != mTiles.end()) {
        return found->second;
    }
    const uint32_t columns = op.mN / 16; // untuned: as many tiles as divide the matrix, up to eight
    return columns % 8 == 0 ? 8 : columns % 4 == 0 ? 4 : columns % 2 == 0 ? 2 : 1;
}

void Graph::Dispatch(VkCommandBuffer commands, const Op& op) {
    Pipeline* pipeline = nullptr;
    uint32_t groupsX = op.mGroups[0];
    if (op.mKind == Op::Kind::Gemm) {
        const uint32_t tiles = TilesFor(op);
        pipeline = &GemmPipeline(op.mK, tiles, op.mSpec);
        groupsX = op.mN / (tiles * 16);
    } else {
        auto key = std::pair{ (int)op.mKind, op.mSpec };
        auto found = mPipelines.find(key);
        if (found == mPipelines.end()) {
            Pipeline created;
            switch (op.mKind) {
                case Op::Kind::Block32:
                    created = mContext.CreateCompute(kBlock32Spv, std::size(kBlock32Spv), { op.mSpec }, 32);
                    break;
                case Op::Kind::Attention:
                    created = mContext.CreateCompute(kAttentionSpv, std::size(kAttentionSpv), {}, 32);
                    break;
                case Op::Kind::VitNormalize:
                    created = mContext.CreateCompute(kVitNormalizeSpv, std::size(kVitNormalizeSpv), {}, 0);
                    break;
                case Op::Kind::VitAttention:
                    created = mContext.CreateCompute(kVitAttentionSpv, std::size(kVitAttentionSpv), {}, 32);
                    break;
                default:
                    created = mContext.CreateCompute(kOpsSpv, std::size(kOpsSpv), { op.mSpec }, 0);
                    break;
            }
            found = mPipelines.emplace(key, created).first;
        }
        pipeline = &found->second;
    }
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->mPipeline);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, mContext.Layout(), 0, 1, &op.mSet, 0, nullptr);
    vkCmdPushConstants(commands, mContext.Layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, op.mPushBytes, op.mPush);
    vkCmdDispatch(commands, groupsX, op.mGroups[1], op.mGroups[2]);
    mContext.Barrier(commands);
}

void Graph::Record(VkCommandBuffer commands, size_t first, size_t count, bool stamps, bool reuseVit) {
    const size_t end = std::min(mSegments.size(), count == SIZE_MAX ? mSegments.size() : first + count);
    if (stamps) {
        mContext.ResetStamps(commands);
        mContext.Stamp(commands, 0);
    }
    for (size_t s = first; s < end; ++s) {
        const Segment& segment = mSegments[s];
        for (size_t i = 0; i < segment.mOpCount && !(reuseVit && segment.mVit); ++i) {
            Dispatch(commands, mOps[segment.mFirstOp + i]);
        }
        if (stamps) {
            mContext.Stamp(commands, (uint32_t)(s - first + 1));
        }
    }
}

void Graph::Tune() {
    std::map<GemmKey, const Op*> shapes;
    for (const Op& op : mOps) {
        if (op.mKind == Op::Kind::Gemm) {
            shapes.emplace(GemmKey{ op.mRows, op.mK, op.mN, op.mSpec, op.mGroups[2] }, &op);
        }
    }
    // How many waves a dispatch makes and how much weight traffic each carries both depend on the tile count, and
    // the best choice is not monotonic in either: measure. The GPU clocks down between short bursts, so each
    // candidate runs several times back to back and the best of three submissions counts.
    for (const auto& [key, op] : shapes) {
        double bestMs = 1e30;
        uint32_t bestTiles = 1;
        for (uint32_t tiles : { 1u, 2u, 4u, 8u }) {
            if ((op->mN / 16) % tiles) {
                continue;
            }
            mTiles[key] = tiles;
            double ms = 1e30;
            for (int attempt = 0; attempt < 3; ++attempt) {
                VkCommandBuffer commands = mContext.Begin();
                mContext.ResetStamps(commands);
                mContext.Stamp(commands, 0);
                for (int repeat = 0; repeat < 8; ++repeat) {
                    Dispatch(commands, *op);
                }
                mContext.Stamp(commands, 1);
                mContext.Submit(commands, true);
                ms = std::min(ms, mContext.ElapsedMs(0, 1));
            }
            if (ms < bestMs) {
                bestMs = ms;
                bestTiles = tiles;
            }
        }
        mTiles[key] = bestTiles;
    }
}

Graph::Tuning Graph::GetTuning() const {
    Tuning tuning;
    for (const auto& [key, tiles] : mTiles) {
        tuning.push_back(
            { std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key), std::get<4>(key), tiles });
    }
    return tuning;
}

bool Graph::SetTuning(const Tuning& tuning) {
    std::map<GemmKey, uint32_t> tiles;
    for (const auto& entry : tuning) {
        const bool valid = entry[5] == 1 || entry[5] == 2 || entry[5] == 4 || entry[5] == 8;
        if (!valid || (entry[2] / 16) % entry[5]) {
            return false;
        }
        tiles[GemmKey{ entry[0], entry[1], entry[2], entry[3], entry[4] }] = entry[5];
    }
    for (const Op& op : mOps) {
        if (op.mKind == Op::Kind::Gemm && !tiles.count(GemmKey{ op.mRows, op.mK, op.mN, op.mSpec, op.mGroups[2] })) {
            return false;
        }
    }
    mTiles = std::move(tiles);
    return true;
}

void Graph::SetFeatures(const std::vector<float>& features) {
    if (features.size() != (size_t)mGeometry.FieldTokens() * 16) {
        throw std::runtime_error("neural rendering: features must be [field][16]");
    }
    mContext.Upload(mFeatures, 0, features.data(), features.size() * 4);
}

std::vector<float> Graph::Read(const Activation& a) {
    std::vector<uint16_t> halves((size_t)a.mRows * a.mChannels);
    mContext.Download(a.mBuffer, 0, halves.data(), halves.size() * 2);
    std::vector<float> values(halves.size());
    for (uint32_t row = 0; row < a.mRows; ++row) {
        // Spatial tensors are tiled on the GPU and row-major for everyone else.
        const uint32_t source = a.mWidth ? TiledIndex(row % a.mWidth, row / a.mWidth, a.mWidth) : row;
        for (uint32_t c = 0; c < a.mChannels; ++c) {
            values[(size_t)row * a.mChannels + c] = Num::F16ToF32(halves[(size_t)source * a.mChannels + c]);
        }
    }
    return values;
}

void Graph::Write(const Activation& a, const std::vector<float>& values) {
    if (values.size() != (size_t)a.mRows * a.mChannels) {
        throw std::runtime_error("neural rendering: tensor size mismatch for " + a.mName);
    }
    std::vector<uint16_t> halves(values.size());
    for (uint32_t row = 0; row < a.mRows; ++row) {
        const uint32_t target = a.mWidth ? TiledIndex(row % a.mWidth, row / a.mWidth, a.mWidth) : row;
        for (uint32_t c = 0; c < a.mChannels; ++c) {
            halves[(size_t)target * a.mChannels + c] = Num::F16Bits(values[(size_t)row * a.mChannels + c]);
        }
    }
    mContext.Upload(a.mBuffer, 0, halves.data(), halves.size() * 2);
}

std::vector<float> Graph::Head() {
    const std::vector<float> wide = Read(*mHead);
    std::vector<float> head((size_t)mHead->mRows * 4);
    for (uint32_t row = 0; row < mHead->mRows; ++row) {
        std::copy(&wide[(size_t)row * 8], &wide[(size_t)row * 8] + 4, &head[(size_t)row * 4]);
    }
    return head;
}

} // namespace Fast::Neural
