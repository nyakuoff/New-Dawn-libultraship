#include "NrRuntime.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <thread>

#include "NrGraph.h"
#include "NrModel.h"
#include "NrNumeric.h"
#include "NrSpirv.inc"
#include "NrVk.h"

namespace Fast::Neural {

namespace {
constexpr uint32_t kMaxSpread = 8; // frames one run may be spread over
}

struct Runtime::Impl {
    std::atomic<State> mState{ State::Idle };
    mutable std::mutex mStatusMutex;
    std::string mStatus;
    std::thread mWorker;

    // Built by the worker, used by the rendering thread once Ready.
    std::unique_ptr<Context> mContext;
    std::unique_ptr<Graph> mGraph;
    Geometry mGeometry;
    float mBlendScale = 0.0f;
    SharedBuffer mProxy, mHistory, mResidual;
    VkSemaphore mSemaphore = VK_NULL_HANDLE;
    int mSemaphoreFd = -1;
    VkFence mFence = VK_NULL_HANDLE;
    Pipeline mFeatures, mComposite;
    VkDescriptorSet mFeaturesSet = VK_NULL_HANDLE, mCompositeSet = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> mSegments; // one per graph segment, recorded once
    std::vector<double> mSegmentMs;

    // The run in flight.
    VkCommandBuffer mBegin = VK_NULL_HANDLE, mEnd = VK_NULL_HANDLE;
    std::vector<std::vector<VkCommandBuffer>> mParts;
    std::vector<uint32_t> mRan; // the segments of this run
    size_t mNextPart = 0;
    bool mWaitRenderer = false;
    bool mVitPrimed = false;

    void Release() {
        if (!mContext) {
            return;
        }
        mContext->WaitIdle();
        for (VkCommandBuffer commands : mSegments) {
            mContext->FreeRecorded(commands);
        }
        mContext->FreeRecorded(mBegin);
        mContext->FreeRecorded(mEnd);
        mSegments.clear();
        mBegin = mEnd = VK_NULL_HANDLE;
        mParts.clear();
        mGraph.reset();
        mContext->DestroyPipeline(mFeatures);
        mContext->DestroyPipeline(mComposite);
        mContext->DestroyBuffer(mProxy.mBuffer);
        mContext->DestroyBuffer(mHistory.mBuffer);
        mContext->DestroyBuffer(mResidual.mBuffer);
        mContext->DestroySemaphore(mSemaphore);
        mContext->DestroyFence(mFence);
        mSemaphore = VK_NULL_HANDLE;
        mFence = VK_NULL_HANDLE;
        mContext->ResetDescriptors();
    }
};

bool Runtime::Request::operator==(const Request& o) const {
    return mModel == o.mModel && mExtractFrom == o.mExtractFrom && mWidth == o.mWidth && mHeight == o.mHeight &&
           mHaveUuid == o.mHaveUuid && memcmp(mUuid, o.mUuid, sizeof mUuid) == 0;
}

Runtime::Runtime() : mImpl(std::make_unique<Impl>()) {
}

Runtime::~Runtime() {
    if (mImpl->mWorker.joinable()) {
        mImpl->mWorker.join();
    }
    mImpl->Release();
}

Runtime::State Runtime::GetState() const {
    return mImpl->mState.load();
}

std::string Runtime::Status() const {
    std::lock_guard lock(mImpl->mStatusMutex);
    return mImpl->mStatus;
}

void Runtime::SetStatus(const std::string& status) {
    std::lock_guard lock(mImpl->mStatusMutex);
    mImpl->mStatus = status;
}

void Runtime::Load(const Request& request) {
    if (mImpl->mWorker.joinable()) {
        mImpl->mWorker.join();
    }
    mRequest = request;
    mBusy = false;
    mImpl->mState = State::Working;
    mImpl->mWorker = std::thread([this, request] {
        try {
            Work(request);
            mImpl->mState = State::Ready;
        } catch (const std::exception& error) {
            SetStatus(error.what());
            mImpl->mState = State::Failed;
        }
    });
}

void Runtime::Work(const Request& request) {
    Impl& m = *mImpl;
    m.Release();
    if (!request.mExtractFrom.empty()) {
        SetStatus("Reading the network out of " + request.mExtractFrom.filename().string() + "...");
        ExtractModel(request.mExtractFrom, request.mModel);
    }
    SetStatus("Loading the network...");
    const Model model(request.mModel);
    if (!m.mContext) {
        m.mContext = std::make_unique<Context>(nullptr, request.mHaveUuid ? request.mUuid : nullptr);
    }
    Context& context = *m.mContext;
    m.mGeometry = Geometry::FromValid(request.mWidth, request.mHeight);
    const Geometry& g = m.mGeometry;
    m.mBlendScale = model.Halves(model.Get(70, 0, "blend_scale"), 0, 1)[0];
    m.mGraph = std::make_unique<Graph>(context, model, g, true);
    Graph& graph = *m.mGraph;
    SetStatus("Tuning the kernels for " + context.Info().mName + "...");
    graph.Tune();

    const VkDeviceSize imageBytes = (VkDeviceSize)g.mValidWidth * g.mValidHeight * 8;
    m.mProxy = context.CreateSharedBuffer(imageBytes);
    m.mHistory = context.CreateSharedBuffer(imageBytes);
    m.mResidual = context.CreateSharedBuffer(imageBytes);
    m.mSemaphore = context.CreateSharedSemaphore(m.mSemaphoreFd);
    m.mFence = context.CreateFence();
    m.mFeatures = context.CreateCompute(kFeaturesSpv, std::size(kFeaturesSpv), {}, 0);
    m.mComposite = context.CreateCompute(kCompositeSpv, std::size(kCompositeSpv), {}, 0);
    m.mFeaturesSet = context.Bind({ &m.mProxy.mBuffer, nullptr, &graph.Find("features").mBuffer, &m.mHistory.mBuffer });
    m.mCompositeSet = context.Bind(
        { &graph.Find("head").mBuffer, nullptr, &m.mResidual.mBuffer, &m.mProxy.mBuffer, &m.mHistory.mBuffer });

    // Every segment as its own command buffer, so a run is any split of them over the frames it is given; and what
    // each costs on this GPU, which decides the split. The best of a few passes counts: the first wakes the GPU up.
    const size_t segments = graph.Segments().size();
    if (2 * segments > Context::kQueries) {
        throw std::runtime_error("neural rendering: more segments than timestamps");
    }
    m.mSegmentMs.assign(segments, 1e30);
    mFullMs = 1e30;
    for (int pass = 0; pass < 4; ++pass) {
        VkCommandBuffer commands = context.Begin();
        graph.Record(commands, 0, SIZE_MAX, true);
        context.Submit(commands, true);
        for (size_t s = 0; s < segments; ++s) {
            m.mSegmentMs[s] = std::min(m.mSegmentMs[s], context.ElapsedMs((uint32_t)s, (uint32_t)s + 1));
        }
        mFullMs = std::min(mFullMs, context.ElapsedMs(0, (uint32_t)segments));
    }
    // Each one carries its own two stamps (2s, 2s + 1): a run's time is the sum over its segments, whatever the
    // renderer got to do between them.
    for (size_t s = 0; s < segments; ++s) {
        VkCommandBuffer commands = context.Begin();
        context.Stamp(commands, 2 * (uint32_t)s);
        graph.Record(commands, s, 1);
        context.Stamp(commands, 2 * (uint32_t)s + 1);
        context.End(commands);
        m.mSegments.push_back(commands);
    }
    m.mVitPrimed = false;
    char line[160];
    snprintf(line, sizeof line, "Ready on %s: %ux%u, %.1f ms a run", context.Info().mName.c_str(), g.mValidWidth,
             g.mValidHeight, mFullMs);
    SetStatus(line);
}

Runtime::Shared Runtime::TakeShared() {
    Impl& m = *mImpl;
    Shared shared;
    shared.mProxyFd = m.mProxy.mFd;
    shared.mHistoryFd = m.mHistory.mFd;
    shared.mResidualFd = m.mResidual.mFd;
    shared.mSemaphoreFd = m.mSemaphoreFd;
    shared.mAllocationBytes = m.mProxy.mAllocationSize;
    shared.mImageBytes = (uint64_t)m.mGeometry.mValidWidth * m.mGeometry.mValidHeight * 8;
    m.mProxy.mFd = m.mHistory.mFd = m.mResidual.mFd = m.mSemaphoreFd = -1;
    return shared;
}

void Runtime::Start(const Run& run) {
    Impl& m = *mImpl;
    Context& context = *m.mContext;
    const Geometry& g = m.mGeometry;
    context.FreeRecorded(m.mBegin);
    context.FreeRecorded(m.mEnd);

    m.mBegin = context.Begin();
    context.ResetStamps(m.mBegin);
    {
        const uint32_t tokens = g.FieldTokens();
        const uint32_t groupsX = std::min(AlignUp(tokens, 64) / 64, 2048u);
        const uint32_t groupsY = (tokens + groupsX * 64 - 1) / (groupsX * 64);
        struct {
            uint32_t tokens, rowStride, width4, validWidth, validHeight, seed, historyValid;
            float style, tone, structure;
        } push{ tokens,    groupsX * 64,           g.mFullWidth / 4, g.mValidWidth, g.mValidHeight,
                run.mSeed, run.mHistory ? 1u : 0u, run.mStyle,       run.mTone,     run.mStructure };
        vkCmdBindPipeline(m.mBegin, VK_PIPELINE_BIND_POINT_COMPUTE, m.mFeatures.mPipeline);
        vkCmdBindDescriptorSets(m.mBegin, VK_PIPELINE_BIND_POINT_COMPUTE, context.Layout(), 0, 1, &m.mFeaturesSet, 0,
                                nullptr);
        vkCmdPushConstants(m.mBegin, context.Layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
        vkCmdDispatch(m.mBegin, groupsX, groupsY, 1);
        context.Barrier(m.mBegin);
    }
    context.End(m.mBegin);

    m.mEnd = context.Begin();
    {
        struct {
            uint32_t validWidth, validHeight, width4, historyValid;
            float blendScale;
        } push{ g.mValidWidth, g.mValidHeight, g.mFullWidth / 4, run.mHistory ? 1u : 0u, m.mBlendScale };
        vkCmdBindPipeline(m.mEnd, VK_PIPELINE_BIND_POINT_COMPUTE, m.mComposite.mPipeline);
        vkCmdBindDescriptorSets(m.mEnd, VK_PIPELINE_BIND_POINT_COMPUTE, context.Layout(), 0, 1, &m.mCompositeSet, 0,
                                nullptr);
        vkCmdPushConstants(m.mEnd, context.Layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
        vkCmdDispatch(m.mEnd, (g.mValidWidth + 7) / 8, (g.mValidHeight + 7) / 8, 1);
        context.Barrier(m.mEnd);
    }
    context.End(m.mEnd);

    // The segments of this run, cut into parts of about equal cost; each part is one job.
    const auto& segments = m.mGraph->Segments();
    const bool reuseVit = run.mReuseVit && m.mVitPrimed;
    m.mVitPrimed = true;
    double total = 0.0;
    for (size_t s = 0; s < segments.size(); ++s) {
        total += reuseVit && segments[s].mVit ? 0.0 : m.mSegmentMs[s];
    }
    const uint32_t spread = std::clamp(run.mSpread, 1u, kMaxSpread);
    m.mParts.assign(spread, {});
    m.mRan.clear();
    double done = 0.0;
    for (size_t s = 0; s < segments.size(); ++s) {
        if (reuseVit && segments[s].mVit) {
            continue;
        }
        const uint32_t part = std::min(spread - 1, (uint32_t)((done + 0.5 * m.mSegmentMs[s]) / total * spread));
        m.mParts[part].push_back(m.mSegments[s]);
        m.mRan.push_back((uint32_t)s);
        done += m.mSegmentMs[s];
    }
    m.mParts.front().insert(m.mParts.front().begin(), m.mBegin); // resets the stamps, so before the first of them
    m.mParts.back().push_back(m.mEnd);
    m.mNextPart = 0;
    m.mWaitRenderer = run.mWaitRenderer;
    mBusy = true;
    Advance();
}

void Runtime::Advance() {
    Impl& m = *mImpl;
    if (!mBusy || m.mNextPart >= m.mParts.size()) {
        return;
    }
    const bool first = m.mNextPart == 0, last = m.mNextPart + 1 == m.mParts.size();
    m.mContext->SubmitRecorded(m.mParts[m.mNextPart], first && m.mWaitRenderer ? m.mSemaphore : VK_NULL_HANDLE,
                               last ? m.mFence : VK_NULL_HANDLE);
    ++m.mNextPart;
}

bool Runtime::Finished() {
    Impl& m = *mImpl;
    if (!mBusy || m.mNextPart < m.mParts.size() || !m.mContext->FenceSignaled(m.mFence)) {
        return false;
    }
    m.mContext->ResetFence(m.mFence);
    mLastMs = 0.0;
    for (uint32_t s : m.mRan) {
        mLastMs += m.mContext->ElapsedMs(2 * s, 2 * s + 1);
    }
    mBusy = false;
    return true;
}

void Runtime::Wait() {
    while (mBusy && !Finished()) {
        Advance();
        if (mImpl->mNextPart >= mImpl->mParts.size()) {
            mImpl->mContext->WaitIdle();
        }
    }
}

void Runtime::WriteInputs(const std::vector<uint16_t>& proxy, const std::vector<uint16_t>& history) {
    mImpl->mContext->Upload(mImpl->mProxy.mBuffer, 0, proxy.data(), proxy.size() * 2);
    mImpl->mContext->Upload(mImpl->mHistory.mBuffer, 0, history.data(), history.size() * 2);
}

std::vector<uint16_t> Runtime::ReadResidual() {
    std::vector<uint16_t> halves((size_t)mImpl->mGeometry.mValidWidth * mImpl->mGeometry.mValidHeight * 4);
    mImpl->mContext->Download(mImpl->mResidual.mBuffer, 0, halves.data(), halves.size() * 2);
    return halves;
}

std::vector<float> Runtime::ReadHead() {
    return mImpl->mGraph->Head();
}

std::vector<float> Runtime::ReadFeatures() {
    return mImpl->mGraph->Read(mImpl->mGraph->Find("features"));
}

} // namespace Fast::Neural
