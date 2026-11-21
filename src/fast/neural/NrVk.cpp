#include "NrVk.h"

#include <algorithm>
#include <cstring>

namespace Fast::Neural {

namespace {

void Check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw VkError(std::string(what) + " failed (VkResult " + std::to_string((int)result) + ")");
    }
}

bool HasExtension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    for (const VkExtensionProperties& extension : extensions) {
        if (!strcmp(extension.extensionName, name)) {
            return true;
        }
    }
    return false;
}

VkInstance CreateInstance() {
    if (volkInitialize() != VK_SUCCESS) {
        throw VkError("no Vulkan loader on this system");
    }
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "New Dawn neural rendering";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo info{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    info.pApplicationInfo = &app;
    VkInstance instance;
    Check(vkCreateInstance(&info, nullptr, &instance), "vkCreateInstance (Vulkan 1.3)");
    volkLoadInstanceOnly(instance);
    return instance;
}

constexpr VkDeviceSize kStagingBytes = 64ull << 20;

} // namespace

DeviceInfo Context::Describe(VkInstance instance, VkPhysicalDevice device) {
    DeviceInfo info;
    VkPhysicalDeviceIDProperties id{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES, &id
    };
    VkPhysicalDeviceProperties2 properties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroup };
    vkGetPhysicalDeviceProperties2(device, &properties);
    info.mName = properties.properties.deviceName;
    info.mVendor = properties.properties.vendorID;
    info.mDevice = properties.properties.deviceID;
    memcpy(info.mUuid, id.deviceUUID, VK_UUID_SIZE);
    info.mMinSubgroup = subgroup.minSubgroupSize;
    info.mMaxSubgroup = subgroup.maxSubgroupSize;

    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(device, &memory);
    for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
        if (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            info.mVramBytes += memory.memoryHeaps[i].size;
        }
    }

    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
    info.mFloat8 = HasExtension(extensions, "VK_EXT_shader_float8");
    info.mExternalFd = HasExtension(extensions, "VK_KHR_external_memory_fd") &&
                       HasExtension(extensions, "VK_KHR_external_semaphore_fd");

    if (properties.properties.apiVersion < VK_API_VERSION_1_3) {
        info.mMissing = "Vulkan 1.3";
        return info;
    }
    if (!HasExtension(extensions, "VK_KHR_cooperative_matrix")) {
        info.mMissing = "VK_KHR_cooperative_matrix";
        return info;
    }
    uint32_t combinations = 0;
    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device, &combinations, nullptr);
    std::vector<VkCooperativeMatrixPropertiesKHR> matrices(combinations,
                                                           { VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR });
    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device, &combinations, matrices.data());
    for (const VkCooperativeMatrixPropertiesKHR& m : matrices) {
        if (m.MSize == 16 && m.NSize == 16 && m.KSize == 16 && m.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            m.BType == VK_COMPONENT_TYPE_FLOAT16_KHR && m.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            m.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR && m.scope == VK_SCOPE_SUBGROUP_KHR) {
            info.mCoopMatF16 = true;
        }
    }
    if (!info.mCoopMatF16) {
        info.mMissing = "16x16x16 half-float cooperative matrices with a 32-bit accumulator";
        return info;
    }
    if (info.mMinSubgroup > 32 || info.mMaxSubgroup < 32) {
        info.mMissing = "a subgroup size of 32";
        return info;
    }

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR matrix{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
    };
    VkPhysicalDeviceVulkan13Features f13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &matrix };
    VkPhysicalDeviceVulkan12Features f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13 };
    VkPhysicalDeviceVulkan11Features f11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12 };
    VkPhysicalDeviceFeatures2 features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11 };
    vkGetPhysicalDeviceFeatures2(device, &features);
    if (!matrix.cooperativeMatrix || !f12.shaderFloat16 || !f11.storageBuffer16BitAccess || !f13.subgroupSizeControl ||
        !f13.computeFullSubgroups || !f12.shaderSubgroupExtendedTypes || !f13.maintenance4) {
        info.mMissing = "half-float shaders with subgroup size control";
    }
    (void)instance;
    return info;
}

std::vector<DeviceInfo> Context::Enumerate() {
    VkInstance instance = CreateInstance();
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    std::vector<DeviceInfo> infos;
    for (VkPhysicalDevice device : devices) {
        infos.push_back(Describe(instance, device));
    }
    vkDestroyInstance(instance, nullptr);
    return infos;
}

Context::Context(const char* deviceNameFilter, const uint8_t* uuid) {
    mInstance = CreateInstance();
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(mInstance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(mInstance, &count, devices.data());

    std::string refused = count ? "" : "no Vulkan device";
    int bestRank = -1;
    for (VkPhysicalDevice device : devices) {
        DeviceInfo info = Describe(mInstance, device);
        if (uuid && memcmp(uuid, info.mUuid, VK_UUID_SIZE) != 0) {
            continue;
        }
        if (deviceNameFilter && info.mName.find(deviceNameFilter) == std::string::npos) {
            continue;
        }
        if (!info.mMissing.empty()) {
            refused = info.mName + " lacks " + info.mMissing;
            continue;
        }
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        int rank = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
        if (rank > bestRank) {
            bestRank = rank;
            mPhysical = device;
            mInfo = info;
            mTimestampPeriod = properties.limits.timestampPeriod;
        }
    }
    if (!mPhysical) {
        vkDestroyInstance(mInstance, nullptr);
        throw VkError(refused.empty() ? "the renderer's GPU is not a Vulkan device" : refused);
    }

    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysical, &families, nullptr);
    std::vector<VkQueueFamilyProperties> queues(families);
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysical, &families, queues.data());
    for (uint32_t i = 0; i < families; ++i) {
        if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            mQueueFamily = i;
            break;
        }
    }

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR matrix{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
    };
    matrix.cooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &matrix };
    f13.subgroupSizeControl = VK_TRUE;
    f13.computeFullSubgroups = VK_TRUE;
    f13.maintenance4 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13 };
    f12.shaderFloat16 = VK_TRUE;
    f12.shaderSubgroupExtendedTypes = VK_TRUE;
    f12.vulkanMemoryModel = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12 };
    f11.storageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceFeatures2 features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11 };

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queue.queueFamilyIndex = mQueueFamily;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    std::vector<const char*> extensions = { "VK_KHR_cooperative_matrix" };
    if (mInfo.mExternalFd) {
        extensions.push_back("VK_KHR_external_memory_fd");
        extensions.push_back("VK_KHR_external_semaphore_fd");
    }
    VkDeviceCreateInfo deviceInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features };
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queue;
    deviceInfo.enabledExtensionCount = (uint32_t)extensions.size();
    deviceInfo.ppEnabledExtensionNames = extensions.data();
    Check(vkCreateDevice(mPhysical, &deviceInfo, nullptr, &mDevice), "vkCreateDevice");
    volkLoadDevice(mDevice);
    vkGetDeviceQueue(mDevice, mQueueFamily, 0, &mQueue);
    vkGetPhysicalDeviceMemoryProperties(mPhysical, &mMemoryProperties);

    VkCommandPoolCreateInfo pool{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = mQueueFamily;
    Check(vkCreateCommandPool(mDevice, &pool, nullptr, &mCommandPool), "vkCreateCommandPool");

    const uint32_t maxSets = 4096;
    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maxSets * kBindings };
    VkDescriptorPoolCreateInfo descriptors{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    descriptors.maxSets = maxSets;
    descriptors.poolSizeCount = 1;
    descriptors.pPoolSizes = &poolSize;
    Check(vkCreateDescriptorPool(mDevice, &descriptors, nullptr, &mDescriptorPool), "vkCreateDescriptorPool");

    VkDescriptorSetLayoutBinding bindings[kBindings];
    for (uint32_t i = 0; i < kBindings; ++i) {
        bindings[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    }
    VkDescriptorSetLayoutCreateInfo setLayout{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    setLayout.bindingCount = kBindings;
    setLayout.pBindings = bindings;
    Check(vkCreateDescriptorSetLayout(mDevice, &setLayout, nullptr, &mSetLayout), "vkCreateDescriptorSetLayout");

    VkPushConstantRange range{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 128 };
    VkPipelineLayoutCreateInfo layout{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &mSetLayout;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &range;
    Check(vkCreatePipelineLayout(mDevice, &layout, nullptr, &mPipelineLayout), "vkCreatePipelineLayout");

    VkQueryPoolCreateInfo queries{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    queries.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queries.queryCount = kQueries;
    Check(vkCreateQueryPool(mDevice, &queries, nullptr, &mQueryPool), "vkCreateQueryPool");

    mDummy = CreateBuffer(4096);
    mStaging =
        CreateBufferOfType(kStagingBytes, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Check(vkMapMemory(mDevice, mStaging.mMemory, 0, VK_WHOLE_SIZE, 0, &mStagingMap), "vkMapMemory");
}

Context::~Context() {
    if (!mDevice) {
        return;
    }
    vkDeviceWaitIdle(mDevice);
    DestroyBuffer(mDummy);
    DestroyBuffer(mStaging);
    vkDestroyQueryPool(mDevice, mQueryPool, nullptr);
    vkDestroyPipelineLayout(mDevice, mPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(mDevice, mSetLayout, nullptr);
    vkDestroyDescriptorPool(mDevice, mDescriptorPool, nullptr);
    vkDestroyCommandPool(mDevice, mCommandPool, nullptr);
    vkDestroyDevice(mDevice, nullptr);
    vkDestroyInstance(mInstance, nullptr);
}

uint32_t Context::MemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags) const {
    for (uint32_t i = 0; i < mMemoryProperties.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (mMemoryProperties.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    throw VkError("no suitable Vulkan memory type");
}

Buffer Context::CreateBufferOfType(VkDeviceSize size, VkMemoryPropertyFlags flags) {
    Buffer buffer;
    buffer.mSize = (size + 4095) & ~VkDeviceSize(4095);
    VkBufferCreateInfo info{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    info.size = buffer.mSize;
    info.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    Check(vkCreateBuffer(mDevice, &info, nullptr, &buffer.mBuffer), "vkCreateBuffer");
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(mDevice, buffer.mBuffer, &requirements);
    VkMemoryAllocateInfo allocation{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = MemoryType(requirements.memoryTypeBits, flags);
    VkResult result = vkAllocateMemory(mDevice, &allocation, nullptr, &buffer.mMemory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(mDevice, buffer.mBuffer, nullptr);
        throw VkError("not enough video memory (" + std::to_string(size >> 20) + " MiB requested)");
    }
    Check(vkBindBufferMemory(mDevice, buffer.mBuffer, buffer.mMemory, 0), "vkBindBufferMemory");
    return buffer;
}

Buffer Context::CreateBuffer(VkDeviceSize size) {
    Buffer buffer = CreateBufferOfType(size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mStaging.mBuffer) {
        Fill(buffer);
    }
    return buffer;
}

void Context::DestroyBuffer(Buffer& buffer) {
    if (buffer.mBuffer) {
        vkDestroyBuffer(mDevice, buffer.mBuffer, nullptr);
        vkFreeMemory(mDevice, buffer.mMemory, nullptr);
    }
    buffer = {};
}

void Context::Fill(const Buffer& buffer, uint32_t value) {
    VkCommandBuffer commands = Begin();
    vkCmdFillBuffer(commands, buffer.mBuffer, 0, VK_WHOLE_SIZE, value);
    Submit(commands, true);
}

void Context::Upload(const Buffer& buffer, VkDeviceSize offset, const void* data, VkDeviceSize size) {
    for (VkDeviceSize done = 0; done < size;) {
        const VkDeviceSize chunk = std::min(size - done, kStagingBytes);
        memcpy(mStagingMap, (const uint8_t*)data + done, chunk);
        VkCommandBuffer commands = Begin();
        VkBufferCopy region{ 0, offset + done, chunk };
        vkCmdCopyBuffer(commands, mStaging.mBuffer, buffer.mBuffer, 1, &region);
        Submit(commands, true);
        done += chunk;
    }
}

void Context::Download(const Buffer& buffer, VkDeviceSize offset, void* data, VkDeviceSize size) {
    for (VkDeviceSize done = 0; done < size;) {
        const VkDeviceSize chunk = std::min(size - done, kStagingBytes);
        VkCommandBuffer commands = Begin();
        VkBufferCopy region{ offset + done, 0, chunk };
        vkCmdCopyBuffer(commands, buffer.mBuffer, mStaging.mBuffer, 1, &region);
        Submit(commands, true);
        memcpy((uint8_t*)data + done, mStagingMap, chunk);
        done += chunk;
    }
}

Pipeline Context::CreateCompute(const uint32_t* spirv, size_t words, const std::vector<uint32_t>& specialization,
                                uint32_t requiredSubgroupSize) {
    Pipeline pipeline;
    VkShaderModuleCreateInfo module{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    module.codeSize = words * 4;
    module.pCode = spirv;
    Check(vkCreateShaderModule(mDevice, &module, nullptr, &pipeline.mModule), "vkCreateShaderModule");

    std::vector<VkSpecializationMapEntry> entries;
    for (uint32_t i = 0; i < specialization.size(); ++i) {
        entries.push_back({ i, i * 4, 4 });
    }
    VkSpecializationInfo constants{ (uint32_t)entries.size(), entries.data(), specialization.size() * 4,
                                    specialization.data() };
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
    };
    subgroup.requiredSubgroupSize = requiredSubgroupSize;
    VkComputePipelineCreateInfo info{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = pipeline.mModule;
    info.stage.pName = "main";
    info.stage.pSpecializationInfo = &constants;
    if (requiredSubgroupSize) {
        info.stage.pNext = &subgroup;
        info.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }
    info.layout = mPipelineLayout;
    VkResult result = vkCreateComputePipelines(mDevice, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline.mPipeline);
    if (result != VK_SUCCESS) {
        vkDestroyShaderModule(mDevice, pipeline.mModule, nullptr);
        Check(result, "vkCreateComputePipelines");
    }
    return pipeline;
}

void Context::DestroyPipeline(Pipeline& pipeline) {
    if (pipeline.mPipeline) {
        vkDestroyPipeline(mDevice, pipeline.mPipeline, nullptr);
        vkDestroyShaderModule(mDevice, pipeline.mModule, nullptr);
    }
    pipeline = {};
}

VkDescriptorSet Context::Bind(const std::vector<const Buffer*>& buffers) {
    VkDescriptorSetAllocateInfo allocation{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocation.descriptorPool = mDescriptorPool;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &mSetLayout;
    VkDescriptorSet set;
    Check(vkAllocateDescriptorSets(mDevice, &allocation, &set), "vkAllocateDescriptorSets");
    VkDescriptorBufferInfo infos[kBindings];
    VkWriteDescriptorSet writes[kBindings];
    for (uint32_t i = 0; i < kBindings; ++i) {
        const Buffer* buffer = i < buffers.size() && buffers[i] ? buffers[i] : &mDummy;
        infos[i] = { buffer->mBuffer, 0, VK_WHOLE_SIZE };
        writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(mDevice, kBindings, writes, 0, nullptr);
    return set;
}

void Context::ResetDescriptors() {
    vkResetDescriptorPool(mDevice, mDescriptorPool, 0);
}

VkCommandBuffer Context::Begin() {
    VkCommandBufferAllocateInfo allocation{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocation.commandPool = mCommandPool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkCommandBuffer commands;
    Check(vkAllocateCommandBuffers(mDevice, &allocation, &commands), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    Check(vkBeginCommandBuffer(commands, &begin), "vkBeginCommandBuffer");
    return commands;
}

void Context::Submit(VkCommandBuffer commands, bool wait) {
    Check(vkEndCommandBuffer(commands), "vkEndCommandBuffer");
    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    Check(vkQueueSubmit(mQueue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
    mPending.push_back(commands);
    if (wait) {
        WaitIdle();
    }
}

void Context::WaitIdle() {
    Check(vkQueueWaitIdle(mQueue), "vkQueueWaitIdle");
    if (!mPending.empty()) {
        vkFreeCommandBuffers(mDevice, mCommandPool, (uint32_t)mPending.size(), mPending.data());
        mPending.clear();
    }
}

void Context::Barrier(VkCommandBuffer commands) {
    VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &barrier, 0, nullptr, 0, nullptr);
}

void Context::ResetStamps(VkCommandBuffer commands) {
    vkCmdResetQueryPool(commands, mQueryPool, 0, kQueries);
}

void Context::Stamp(VkCommandBuffer commands, uint32_t index) {
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, mQueryPool, index);
}

double Context::ElapsedMs(uint32_t first, uint32_t second) {
    uint64_t stamps[2] = {};
    Check(vkGetQueryPoolResults(mDevice, mQueryPool, first, 1, 8, &stamps[0], 8, VK_QUERY_RESULT_64_BIT),
          "vkGetQueryPoolResults");
    Check(vkGetQueryPoolResults(mDevice, mQueryPool, second, 1, 8, &stamps[1], 8, VK_QUERY_RESULT_64_BIT),
          "vkGetQueryPoolResults");
    return double(stamps[1] - stamps[0]) * mTimestampPeriod * 1e-6;
}

} // namespace Fast::Neural
