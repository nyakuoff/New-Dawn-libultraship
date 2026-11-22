// The network as the renderer drives it: loaded in the background, fed from buffers the renderer's own API fills,
// run as a list of recorded command buffers that one frame or several submit.
//
// A run reads two images (proxy, history) and writes one (the residual); all three are rgba16f [height][width]
// buffers, top row first, that the renderer imports (GL_EXT_memory_object_fd). Vulkan stays out of this header: the
// renderer includes it beside its own API.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Fast::Neural {

class Runtime {
  public:
    enum class State { Idle, Working, Ready, Failed };

    struct Request {
        std::filesystem::path mModel;       // the model directory, in the player's cache
        std::filesystem::path mExtractFrom; // not empty: build that directory from this nvngx_dlssnr.dll first
        uint32_t mWidth = 0, mHeight = 0;   // the image the network sees
        bool mHaveUuid = false;             // run on the renderer's GPU (GL_DEVICE_UUID_EXT), else on the best one
        uint8_t mUuid[16] = {};

        bool operator==(const Request& other) const;
    };

    // File descriptors the renderer imports once (it owns them from then on) and the size of each allocation.
    struct Shared {
        int mProxyFd = -1, mHistoryFd = -1, mResidualFd = -1, mSemaphoreFd = -1;
        uint64_t mAllocationBytes = 0;
        uint64_t mImageBytes = 0; // width * height * 8
    };

    struct Run {
        uint32_t mSeed = 0;        // the noise lanes: the frame counter
        float mStyle = 0.0f;       // 0 off, 1 natural, 2 cinematic
        float mTone = 1.0f;        // 0..1
        float mStructure = 1.0f;   // 0..1
        bool mHistory = false;     // the history buffer holds one (its alpha says where)
        bool mReuseVit = false;    // skip the eight global-attention blocks: the decoder reads the last run's
        uint32_t mSpread = 1;      // submit over this many Advance() calls
        bool mWaitRenderer = true; // the first job waits for the renderer's semaphore (it just wrote the inputs)
    };

    Runtime();
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // Starts loading in the background; GetState() and Status() follow it. Not while Working.
    void Load(const Request& request);
    State GetState() const;
    std::string Status() const; // one line for the menu
    const Request& Loaded() const {
        return mRequest;
    }

    // Ready only, all from the thread that renders.
    Shared TakeShared();
    bool Busy() const {
        return mBusy;
    }
    void Start(const Run& run); // records the run and submits its first part
    void Advance();             // submits the next part, if one is left
    void Hurry();               // submits every part that is left (a run nobody is waiting for any more)
    bool Finished();            // true once per run, when the residual buffer holds its result
    void Wait();                // blocks until the run in flight has finished (tools, shutdown)
    double LastMs() const {     // GPU time of the last finished run, without what ran between its parts
        return mLastMs;
    }
    double FullMs() const { // the whole network measured at load, every block
        return mFullMs;
    }

    // For tools without a renderer: the image buffers through the CPU, as halves.
    void WriteInputs(const std::vector<uint16_t>& proxy, const std::vector<uint16_t>& history);
    std::vector<uint16_t> ReadResidual();
    std::vector<float> ReadFeatures(); // [field][16], row-major
    std::vector<float> ReadHead();     // [field][4], row-major

  private:
    struct Impl;
    void Work(const Request& request);
    void SetStatus(const std::string& status);

    std::unique_ptr<Impl> mImpl;
    Request mRequest;
    bool mBusy = false;
    double mLastMs = 0.0, mFullMs = 0.0;
};

} // namespace Fast::Neural
