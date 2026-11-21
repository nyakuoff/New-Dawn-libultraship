// Vulkan compute context of the neural renderer: one device, one queue, storage buffers, compute pipelines.
// The loader is opened at run time (volk), so the game neither links against Vulkan nor needs it to start.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <volk.h>

namespace Fast::Neural {

struct VkError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Buffer {
    VkBuffer mBuffer = VK_NULL_HANDLE;
    VkDeviceMemory mMemory = VK_NULL_HANDLE;
    VkDeviceSize mSize = 0;
};

struct Pipeline {
    VkPipeline mPipeline = VK_NULL_HANDLE;
    VkShaderModule mModule = VK_NULL_HANDLE;
};

// What the FP16 path needs from a device, and what the device says about itself.
struct DeviceInfo {
    std::string mName;
    uint32_t mVendor = 0;
    uint32_t mDevice = 0;
    uint8_t mUuid[VK_UUID_SIZE] = {};
    bool mCoopMatF16 = false; // 16x16x16, f16 x f16 -> f32, subgroup scope
    bool mFloat8 = false;     // VK_EXT_shader_float8 (the FP8 path)
    bool mExternalFd = false; // VK_KHR_external_memory_fd + VK_KHR_external_semaphore_fd
    uint32_t mMinSubgroup = 0, mMaxSubgroup = 0;
    uint64_t mVramBytes = 0;
    std::string mMissing; // empty when the FP16 path can run; else what is absent
};

class Context {
  public:
    static constexpr uint32_t kBindings = 8;

    // Throws VkError with a message fit for a status line when the loader, a device or a feature is missing.
    // `deviceNameFilter` (a substring) or `uuid` select the GPU; default: the first discrete one that qualifies.
    explicit Context(const char* deviceNameFilter = nullptr, const uint8_t* uuid = nullptr);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    static std::vector<DeviceInfo> Enumerate();

    const DeviceInfo& Info() const {
        return mInfo;
    }
    VkDevice Device() const {
        return mDevice;
    }
    VkPipelineLayout Layout() const {
        return mPipelineLayout;
    }

    Buffer CreateBuffer(VkDeviceSize size);
    void DestroyBuffer(Buffer& buffer);
    void Upload(const Buffer& buffer, VkDeviceSize offset, const void* data, VkDeviceSize size);
    void Download(const Buffer& buffer, VkDeviceSize offset, void* data, VkDeviceSize size);
    void Fill(const Buffer& buffer, uint32_t value = 0);

    // `requiredSubgroupSize` 0 leaves the driver's choice (shaders that use no subgroup operations).
    Pipeline CreateCompute(const uint32_t* spirv, size_t words, const std::vector<uint32_t>& specialization,
                           uint32_t requiredSubgroupSize);
    void DestroyPipeline(Pipeline& pipeline);

    // A descriptor set with `buffers` in bindings 0..; null entries get a dummy. Sets live until ResetDescriptors.
    VkDescriptorSet Bind(const std::vector<const Buffer*>& buffers);
    void ResetDescriptors();

    VkCommandBuffer Begin();
    void Submit(VkCommandBuffer commands, bool wait);
    void WaitIdle();
    void Barrier(VkCommandBuffer commands);

    // Timestamps: Stamp() writes query `index`; ElapsedMs reads two after the queue is idle.
    static constexpr uint32_t kQueries = 256;
    void ResetStamps(VkCommandBuffer commands);
    void Stamp(VkCommandBuffer commands, uint32_t index);
    double ElapsedMs(uint32_t first, uint32_t second);

  private:
    static DeviceInfo Describe(VkInstance instance, VkPhysicalDevice device);
    uint32_t MemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags) const;
    Buffer CreateBufferOfType(VkDeviceSize size, VkMemoryPropertyFlags flags);

    VkInstance mInstance = VK_NULL_HANDLE;
    VkPhysicalDevice mPhysical = VK_NULL_HANDLE;
    VkDevice mDevice = VK_NULL_HANDLE;
    uint32_t mQueueFamily = 0;
    VkQueue mQueue = VK_NULL_HANDLE;
    VkCommandPool mCommandPool = VK_NULL_HANDLE;
    VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout mSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout mPipelineLayout = VK_NULL_HANDLE;
    VkQueryPool mQueryPool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mMemoryProperties{};
    float mTimestampPeriod = 1.0f;
    DeviceInfo mInfo;
    Buffer mDummy;
    Buffer mStaging;
    void* mStagingMap = nullptr;
    std::vector<VkCommandBuffer> mPending;
};

} // namespace Fast::Neural
