#define NOMINMAX

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#ifndef _WIN32
#include <dlfcn.h>
#endif

#include <any>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <list>
#include <stack>
#include <chrono>
#include "fast/resource/type/Light.h"

#ifndef _LANGUAGE_C
#define _LANGUAGE_C
#endif
#include "fast/debug/GfxDebugger.h"
#include "fast/types.h"
#include <string>

#include "fast/interpreter.h"
#include "fast/newdawn_wind.h"
#include "fast/lus_gbi.h"
#include "fast/newdawn_postfx.h"
#include "fast/backends/gfx_window_manager_api.h"
#include "fast/backends/gfx_rendering_api.h"

#include "ship/window/gui/Gui.h"
#include "ship/resource/ResourceManager.h"
#include "ship/utils/Utils.h"
#include "ship/resource/archive/Archive.h"
#include <nlohmann/json.hpp>
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"

#include "libultraship/libultra/os.h"

#include <spdlog/fmt/fmt.h>

#ifdef _WIN32
#include <windows.h>
#endif

std::stack<std::string> currentDir;

#define SEG_ADDR(seg, addr) (addr | (seg << 24) | 1)
#define SUPPORT_CHECK(x) assert(x)

// SCALE_M_N: upscale/downscale M-bit integer to N-bit
#define SCALE_5_8(VAL_) (((VAL_) * 0xFF) / 0x1F)
#define SCALE_8_5(VAL_) ((((VAL_) + 4) * 0x1F) / 0xFF)
#define SCALE_4_8(VAL_) ((VAL_) * 0x11)
#define SCALE_8_4(VAL_) ((VAL_) / 0x11)
#define SCALE_3_8(VAL_) ((VAL_) * 0x24)
#define SCALE_8_3(VAL_) ((VAL_) / 0x24)

// Based off the current set native dimensions or active framebuffer
#define HALF_SCREEN_WIDTH(activeFb) ((mFbActive ? activeFb->second.orig_width : mNativeDimensions.width) / 2)
#define HALF_SCREEN_HEIGHT(activeFb) ((mFbActive ? activeFb->second.orig_height : mNativeDimensions.height) / 2)

// Ratios for current window dimensions or active framebuffer scaled size
#define RATIO_X(activeFb, dims) \
    ((mFbActive ? activeFb->second.applied_width : dims.width) / (2.0f * HALF_SCREEN_WIDTH(activeFb)))
#define RATIO_Y(activeFb, dims) \
    ((mFbActive ? activeFb->second.applied_height : dims.height) / (2.0f * HALF_SCREEN_HEIGHT(activeFb)))

#define TEXTURE_CACHE_MAX_SIZE 1024

namespace Fast {

static UcodeHandlers ucode_handler_index = ucode_f3dex2;

const static uint32_t f3dex2AttrHandler[] = {
    F3DEX2_G_MTX_PROJECTION, F3DEX2_G_MTX_LOAD,  F3DEX2_G_MTX_PUSH,  F3DEX_G_MTX_NOPUSH,
    F3DEX2_G_CULL_FRONT,     F3DEX2_G_CULL_BACK, F3DEX2_G_CULL_BOTH,
};

const static uint32_t f3dexAttrHandler[] = { F3DEX_G_MTX_PROJECTION, F3DEX_G_MTX_LOAD,   F3DEX_G_MTX_PUSH,
                                             F3DEX_G_MTX_NOPUSH,     F3DEX_G_CULL_FRONT, F3DEX_G_CULL_BACK,
                                             F3DEX_G_CULL_BOTH };

static constexpr std::array ucode_attr_handlers = {
    &f3dexAttrHandler,  // ucode_f3db
    &f3dexAttrHandler,  // ucode_f3d
    &f3dexAttrHandler,  // ucode_f3dex
    &f3dexAttrHandler,  // ucode_f3exb
    &f3dex2AttrHandler, // ucode_f3ex2
    &f3dex2AttrHandler, // ucode_s2dex
};

static uint32_t get_attr(Attribute attr) {
    const auto ucode_map = ucode_attr_handlers[ucode_handler_index];
    // assert(ucode_map->contains(attr) && "Attribute not found in the current ucode handler");
    return (*ucode_map)[attr];
}

static std::string GetPathWithoutFileName(char* filePath) {
    size_t len = strlen(filePath);

    for (size_t i = len - 1; (long)i >= 0; i--) {
        if (filePath[i] == '/' || filePath[i] == '\\') {
            return std::string(filePath).substr(0, i);
        }
    }

    return filePath;
}

constexpr size_t MAX_TRI_BUFFER = 256;

Interpreter::Interpreter() {
    mRsp = new RSP();
    mRdp = new RDP();
    mBufVbo = new float[MAX_TRI_BUFFER * (48 * 3)];
}

Interpreter::~Interpreter() {
    delete mRsp;
    delete mRdp;
    delete[] mBufVbo;
}

static std::weak_ptr<Interpreter> mInstance;
// Set a cached pointer to the instance so we don't need to go through the window every time
void GfxSetInstance(std::shared_ptr<Interpreter> gfx) {
    mInstance = gfx;
}

// N64 prim_depth is 15-bit (0 near, 0x7FFF far).
static constexpr float N64_PRIM_DEPTH_MAX = 32767.0f;

void Interpreter::Flush() {
    if (mBufVboLen > 0) {
        if (mPerPixelLightState.serial != mPerPixelLightUploadedSerial) {
            mRapi->SetPerPixelLighting(mPerPixelLightState);
            mPerPixelLightUploadedSerial = mPerPixelLightState.serial;
        }
        mRapi->SetCurrentPrimDepth((float)mRdp->prim_depth / N64_PRIM_DEPTH_MAX);
        mRapi->SetStencilMark(mBufMark);
        mRapi->DrawTriangles(mBufVbo, mBufVboLen, mBufVboNumTris);
        mBufVboLen = 0;
        mBufVboNumTris = 0;
    }
}

ShaderProgram* Interpreter::LookupOrCreateShaderProgram(uint64_t id0, uint64_t id1) {
    ShaderProgram* prg = mRapi->LookupShader(id0, id1);
    if (prg == nullptr) {
        mRapi->UnloadShader(mRenderingState.mShaderProgram);
        prg = mRapi->CreateAndLoadNewShader(id0, id1);
        mRenderingState.mShaderProgram = prg;
    }
    return prg;
}

const char* Interpreter::CCMUXtoStr(uint32_t ccmux) {
    static constexpr std::array tbl = {
        "G_CCMUX_COMBINED",
        "G_CCMUX_TEXEL0",
        "G_CCMUX_TEXEL1",
        "G_CCMUX_PRIMITIVE",
        "G_CCMUX_SHADE",
        "G_CCMUX_ENVIRONMENT",
        "G_CCMUX_1",
        "G_CCMUX_COMBINED_ALPHA",
        "G_CCMUX_TEXEL0_ALPHA",
        "G_CCMUX_TEXEL1_ALPHA",
        "G_CCMUX_PRIMITIVE_ALPHA",
        "G_CCMUX_SHADE_ALPHA",
        "G_CCMUX_ENV_ALPHA",
        "G_CCMUX_LOD_FRACTION",
        "G_CCMUX_PRIM_LOD_FRAC",
        "G_CCMUX_K5",
    };
    if (ccmux > tbl.size()) {
        return "G_CCMUX_0";
    }
    return tbl[ccmux];
}

// Seems unused
const char* Interpreter::ACMUXtoStr(uint32_t acmux) {
    static constexpr std::array tbl = {
        "G_ACMUX_COMBINED or G_ACMUX_LOD_FRACTION",
        "G_ACMUX_TEXEL0",
        "G_ACMUX_TEXEL1",
        "G_ACMUX_PRIMITIVE",
        "G_ACMUX_SHADE",
        "G_ACMUX_ENVIRONMENT",
        "G_ACMUX_1 or G_ACMUX_PRIM_LOD_FRAC",
        "G_ACMUX_0",
    };
    return tbl[acmux];
}

void Interpreter::GenerateCC(ColorCombiner* comb, const ColorCombinerKey& key) {
    const bool is2Cyc = (key.options & SHADER_OPT(_2CYC)) != 0;

    uint8_t c[2][2][4];
    uint64_t shaderId0 = 0;
    uint64_t shaderId1 = key.options;
    uint8_t shaderInputMapping[2][7] = { { 0 } };
    bool usedTextures[2]{};
    for (uint32_t i = 0; i < 2 && (i == 0 || is2Cyc); i++) {
        uint32_t rgbA = (key.combine_mode >> (i * 28)) & 0xf;
        uint32_t rgbB = (key.combine_mode >> (i * 28 + 4)) & 0xf;
        uint32_t rgbC = (key.combine_mode >> (i * 28 + 8)) & 0x1f;
        uint32_t rgbD = (key.combine_mode >> (i * 28 + 13)) & 7;
        uint32_t alphaA = (key.combine_mode >> (i * 28 + 16)) & 7;
        uint32_t alphaB = (key.combine_mode >> (i * 28 + 16 + 3)) & 7;
        uint32_t alphaC = (key.combine_mode >> (i * 28 + 16 + 6)) & 7;
        uint32_t alphaD = (key.combine_mode >> (i * 28 + 16 + 9)) & 7;

        if (rgbA >= 8) {
            rgbA = G_CCMUX_0;
        }
        if (rgbB >= 8) {
            rgbB = G_CCMUX_0;
        }
        if (rgbC >= 16) {
            rgbC = G_CCMUX_0;
        }
        if (rgbD == 7) {
            rgbD = G_CCMUX_0;
        }

        if (rgbA == rgbB || rgbC == G_CCMUX_0) {
            // Normalize
            rgbA = G_CCMUX_0;
            rgbB = G_CCMUX_0;
            rgbC = G_CCMUX_0;
        }
        if (alphaA == alphaB || alphaC == G_ACMUX_0) {
            // Normalize
            alphaA = G_ACMUX_0;
            alphaB = G_ACMUX_0;
            alphaC = G_ACMUX_0;
        }
        if (i == 1) {
            if (rgbA != G_CCMUX_COMBINED && rgbB != G_CCMUX_COMBINED && rgbC != G_CCMUX_COMBINED &&
                rgbD != G_CCMUX_COMBINED) {
                // First cycle RGB not used, so clear it away
                c[0][0][0] = c[0][0][1] = c[0][0][2] = c[0][0][3] = G_CCMUX_0;
            }
            if (rgbC != G_CCMUX_COMBINED_ALPHA && alphaA != G_ACMUX_COMBINED && alphaB != G_ACMUX_COMBINED &&
                alphaD != G_ACMUX_COMBINED) {
                // First cycle ALPHA not used, so clear it away
                c[0][1][0] = c[0][1][1] = c[0][1][2] = c[0][1][3] = G_ACMUX_0;
            }
        }

        c[i][0][0] = rgbA;
        c[i][0][1] = rgbB;
        c[i][0][2] = rgbC;
        c[i][0][3] = rgbD;
        c[i][1][0] = alphaA;
        c[i][1][1] = alphaB;
        c[i][1][2] = alphaC;
        c[i][1][3] = alphaD;
    }
    if (!is2Cyc) {
        for (uint32_t i = 0; i < 2; i++) {
            for (uint32_t k = 0; k < 4; k++) {
                c[1][i][k] = i == 0 ? G_CCMUX_0 : G_ACMUX_0;
            }
        }

        // In 1-cycle mode, TEXEL1 returns the same value as TEXEL0.
        // Remap combiner inputs so the shader samples from the correct slot.
        // Ex: TEXEL1/TEXEL1_ALPHA → TEXEL0/TEXEL0_ALPHA
        for (uint32_t k = 0; k < 4; k++) {
            if (c[0][0][k] == G_CCMUX_TEXEL1)
                c[0][0][k] = G_CCMUX_TEXEL0;
            if (c[0][0][k] == G_CCMUX_TEXEL1_ALPHA)
                c[0][0][k] = G_CCMUX_TEXEL0_ALPHA;
            if (c[0][1][k] == G_ACMUX_TEXEL1)
                c[0][1][k] = G_ACMUX_TEXEL0;
        }
    }
    {
        uint8_t inputNumber[32] = { 0 };
        uint32_t nextInputNumber = SHADER_INPUT_1;
        for (uint32_t i = 0; i < 2 && (i == 0 || is2Cyc); i++) {
            for (uint32_t j = 0; j < 4; j++) {
                // Mux values 6/7/15 are overloaded by slot. Only B (value 6 = CENTER, 7 = K4)
                // and C (value 6 = SCALE, 15 = K5) carry chroma-key/convert inputs; A and D
                // reuse those values for unrelated constants.
                if (j == 1 && c[i][0][j] == G_CCMUX_CENTER) {
                    c[i][0][j] = G_CCMUX_KEY_CENTER;
                } else if (j == 1 && c[i][0][j] == G_CCMUX_K4) {
                    c[i][0][j] = G_CCMUX_CONVERT_K4;
                } else if (j == 2 && c[i][0][j] == G_CCMUX_SCALE) {
                    c[i][0][j] = G_CCMUX_KEY_SCALE;
                } else if (j == 2 && c[i][0][j] == G_CCMUX_K5) {
                    c[i][0][j] = G_CCMUX_CONVERT_K5;
                }
                uint32_t val = 0;
                switch (c[i][0][j]) {
                    case G_CCMUX_0:
                        val = SHADER_0;
                        break;
                    case G_CCMUX_1:
                        val = SHADER_1;
                        break;
                    case G_CCMUX_TEXEL0:
                        val = SHADER_TEXEL0;
                        // Set the opposite texture when reading from the second cycle color options
                        if (i == 0) {
                            usedTextures[0] = true;
                        } else {
                            usedTextures[1] = true;
                        }
                        break;
                    case G_CCMUX_TEXEL1:
                        val = SHADER_TEXEL1;
                        if (i == 0) {
                            usedTextures[1] = true;
                        } else {
                            usedTextures[0] = true;
                        }
                        break;
                    case G_CCMUX_TEXEL0_ALPHA:
                        val = SHADER_TEXEL0A;
                        if (i == 0) {
                            usedTextures[0] = true;
                        } else {
                            usedTextures[1] = true;
                        }
                        break;
                    case G_CCMUX_TEXEL1_ALPHA:
                        val = SHADER_TEXEL1A;
                        if (i == 0) {
                            usedTextures[1] = true;
                        } else {
                            usedTextures[0] = true;
                        }
                        break;
                    case G_CCMUX_NOISE:
                        val = SHADER_NOISE;
                        break;
                    case G_CCMUX_PRIMITIVE:
                    case G_CCMUX_PRIMITIVE_ALPHA:
                    case G_CCMUX_PRIM_LOD_FRAC:
                    case G_CCMUX_SHADE:
                    case G_CCMUX_ENVIRONMENT:
                    case G_CCMUX_ENV_ALPHA:
                    case G_CCMUX_LOD_FRACTION:
                    case G_CCMUX_KEY_CENTER:
                    case G_CCMUX_KEY_SCALE:
                    case G_CCMUX_CONVERT_K4:
                    case G_CCMUX_CONVERT_K5:
                        if (inputNumber[c[i][0][j]] == 0) {
                            shaderInputMapping[0][nextInputNumber - 1] = c[i][0][j];
                            inputNumber[c[i][0][j]] = nextInputNumber++;
                        }
                        val = inputNumber[c[i][0][j]];
                        break;
                    case G_CCMUX_COMBINED:
                        val = SHADER_COMBINED;
                        break;
                    default:
                        fprintf(stderr, "Unsupported ccmux: %d\n", c[i][0][j]);
                        break;
                }
                shaderId0 |= (uint64_t)val << (i * 32 + j * 4);
            }
        }
    }
    {
        uint8_t inputNumber[16] = { 0 };
        uint32_t nextInputNumber = SHADER_INPUT_1;
        for (uint32_t i = 0; i < 2; i++) {
            for (uint32_t j = 0; j < 4; j++) {
                uint32_t val = 0;
                switch (c[i][1][j]) {
                    case G_ACMUX_0:
                        val = SHADER_0;
                        break;
                    case G_ACMUX_TEXEL0:
                        val = SHADER_TEXEL0;
                        // Set the opposite texture when reading from the second cycle color options
                        if (i == 0) {
                            usedTextures[0] = true;
                        } else {
                            usedTextures[1] = true;
                        }
                        break;
                    case G_ACMUX_TEXEL1:
                        val = SHADER_TEXEL1;
                        if (i == 0) {
                            usedTextures[1] = true;
                        } else {
                            usedTextures[0] = true;
                        }
                        break;
                    case G_ACMUX_LOD_FRACTION:
                        // case G_ACMUX_COMBINED: same numerical value
                        if (j != 2) {
                            val = SHADER_COMBINED;
                            break;
                        }
                        c[i][1][j] = G_CCMUX_LOD_FRACTION;
                        [[fallthrough]]; // for G_ACMUX_LOD_FRACTION
                    case G_ACMUX_1:
                        // case G_ACMUX_PRIM_LOD_FRAC: same numerical value
                        if (j != 2) {
                            val = SHADER_1;
                            break;
                        }
                        [[fallthrough]]; // for G_ACMUX_PRIM_LOD_FRAC
                    case G_ACMUX_PRIMITIVE:
                    case G_ACMUX_SHADE:
                    case G_ACMUX_ENVIRONMENT:
                        if (inputNumber[c[i][1][j]] == 0) {
                            shaderInputMapping[1][nextInputNumber - 1] = c[i][1][j];
                            inputNumber[c[i][1][j]] = nextInputNumber++;
                        }
                        val = inputNumber[c[i][1][j]];
                        break;
                }
                shaderId0 |= (uint64_t)val << (i * 32 + 16 + j * 4);
            }
        }
    }
    comb->shader_id0 = shaderId0;
    comb->shader_id1 = shaderId1;
    comb->usedTextures[0] = usedTextures[0];
    comb->usedTextures[1] = usedTextures[1];
    // comb->prg = gfx_lookup_or_create_mShaderProgram(shader_id0, shader_id1);
    memcpy(comb->shader_input_mapping, shaderInputMapping, sizeof(shaderInputMapping));
    CCFeatures features{};
    gfx_cc_get_features(shaderId0, shaderId1, &features);
    comb->materialShadeMask = NewDawnMaterialShadeMask(features, comb->materialBlend);
    uint32_t shadeMask = 0;
    for (int j = 0; j < 7; ++j) {
        if (shaderInputMapping[0][j] == G_CCMUX_SHADE)
            shadeMask |= 1u << j;
    }
    comb->materialShadeMask &= shadeMask;
}

ColorCombiner* Interpreter::LookupOrCreateColorCombiner(const ColorCombinerKey& key) {
    if (mPrevCombiner != mColorCombinerPool.end() && mPrevCombiner->first == key) {
        return &mPrevCombiner->second;
    }
    mPrevCombiner = mColorCombinerPool.find(key);
    if (mPrevCombiner != mColorCombinerPool.end()) {
        return &mPrevCombiner->second;
    }
    Flush();
    mPrevCombiner = mColorCombinerPool.insert(std::make_pair(key, ColorCombiner())).first;
    GenerateCC(&mPrevCombiner->second, key);
    return &mPrevCombiner->second;
}

void Interpreter::SetResolvedResourceCacheEnabled(bool enabled) {
    mResolvedResourceCacheEnabled = enabled;
    if (!enabled) {
        mResolvedResourceCache.clear();
    }
}

// Texture binds resolve the same paths every frame; skip the resource
// manager's string/hash/mutex work by memoizing on the pointer.
std::shared_ptr<Ship::IResource> Interpreter::ResolveResourceCached(const char* path) {
    if (path == nullptr) {
        return nullptr;
    }
    if (!mResolvedResourceCacheEnabled) {
        return Ship::Context::GetRawInstance()->GetResourceManager()->LoadResourceProcess(path);
    }
    auto it = mResolvedResourceCache.find(path);
    if (it != mResolvedResourceCache.end()) {
        return it->second;
    }
    auto res = Ship::Context::GetRawInstance()->GetResourceManager()->LoadResourceProcess(path);
    // Only memoize a hit. The resource manager caches its own misses, so re-asking
    // for one is cheap, and CacheExternalResource can turn a path that missed into a
    // valid resource at runtime. A memoized null would outlive the resource itself.
    if (res != nullptr) {
        mResolvedResourceCache[path] = res;
    }
    return res;
}

// New Dawn: alternate (HD) textures are large -- the 4K pack's skyboxes are 64 MiB each -- and
// reading them from the archive on the render thread is what stalls the first frame of a scene.
// Start the read on the resource pool and keep drawing the original texture until it arrives:
// the draw is still correct, just lower detail for a frame or two. The resource manager cache
// serves the alternate texture from the next frame on.
std::shared_ptr<Ship::IResource> Interpreter::ResolveTextureResource(const char* path, bool& usedFallback) {
    usedFallback = false;
    if (path == nullptr) {
        return nullptr;
    }
    auto rm = Ship::Context::GetRawInstance()->GetResourceManager();
    if (!mAsyncAltLoadEnabled) {
        return rm->LoadResourceProcess(path);
    }
    std::string_view pathView(path);
    if (!rm->IsAltAssetsEnabled() || pathView.starts_with(Ship::IResource::gAltAssetPrefix)) {
        return rm->LoadResourceProcess(path);
    }
    // Match LoadResourceProcess: the texture address may carry the "__OTR__" signature prefix.
    static constexpr std::string_view kOtrSignature = "__OTR__";
    if (pathView.starts_with(kOtrSignature)) {
        pathView.remove_prefix(kOtrSignature.size());
    }
    if (mResolvedResourceCacheEnabled) {
        auto memo = mResolvedResourceCache.find(path);
        if (memo != mResolvedResourceCache.end()) {
            return memo->second;
        }
    }

    std::string altPath = Ship::IResource::gAltAssetPrefix + std::string(pathView);

    auto ready = mReadyAltTextures.find(altPath);
    if (ready != mReadyAltTextures.end()) {
        if (mResolvedResourceCacheEnabled) {
            mResolvedResourceCache[path] = ready->second;
        }
        return ready->second;
    }
    if (mMissingAltTextures.find(altPath) != mMissingAltTextures.end()) {
        return rm->LoadResourceProcess(path, true);
    }

    // Whatever we draw until the alternate texture arrives has to exist: a mod can ship an
    // alternate texture for a path the base archives never had (3DS Link, new equipment, ...).
    // There is nothing to fall back to in that case, so take the old synchronous path instead of
    // returning a null texture. The check is an archive index lookup, not a load.
    const bool altLoadStarted = mPendingAltTextures.find(altPath) != mPendingAltTextures.end();
    const bool baseExists = rm->GetArchiveManager()->HasFile(std::string(pathView));
    if (!altLoadStarted && !baseExists) {
        return rm->LoadResourceProcess(path);
    }

    // Background reads are best-effort: keep only a few in flight so they do not starve the render
    // thread's own work (uploading and converting the textures that are already available).
    static constexpr size_t kMaxAltLoadsInFlight = 4;
    if (!altLoadStarted && mPendingAltTextures.size() < kMaxAltLoadsInFlight) {
        mPendingAltTextures[altPath] = rm->LoadResourceAsync(altPath, true);
    }

    usedFallback = true;
    return rm->LoadResourceProcess(path, true);
}

void Interpreter::PumpAltTextureLoads() {
    // Escape hatch: the previous synchronous path, with no background reads and no fallback frames.
    mAsyncAltLoadEnabled =
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Textures.AsyncAltLoad", 1) != 0;
    if (!mAsyncAltLoadEnabled) {
        mPendingAltTextures.clear();
        mReadyAltTextures.clear();
        mMissingAltTextures.clear();
        return;
    }
    if (mPendingAltTextures.empty()) {
        return;
    }
    for (auto it = mPendingAltTextures.begin(); it != mPendingAltTextures.end();) {
        if (it->second.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            ++it;
            continue;
        }
        auto resource = it->second.get();
        if (resource != nullptr) {
            mReadyAltTextures[it->first] = resource;
        } else {
            mMissingAltTextures.insert(it->first);
        }
        it = mPendingAltTextures.erase(it);
    }
}

// Companions are optional: flat shading is valid while reads/validation are pending. Albedo
// resolution still uses ResolveTextureResource, including its synchronous mod-only rule.
void Interpreter::PumpMaterialLoads() {
    for (auto it = mPendingMaterials.begin(); it != mPendingMaterials.end();) {
        if (it->second.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            ++it;
            continue;
        }
        mMaterials[it->first] = it->second.get();
        it = mPendingMaterials.erase(it);
    }
}

bool Interpreter::BindMaterialMaps(int tile, int albedoSlot) {
    const auto& loaded = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index];
    auto albedo = loaded.raw_tex_metadata.resource;
    // Full-image companions cannot describe a cropped/subloaded tile or a framebuffer mirror.
    if (!albedo || mFbTextures.contains((uintptr_t)loaded.addr) || loaded.addr != albedo->ImageData ||
        !(albedo->Flags & (TEX_FLAG_LOAD_AS_RAW | TEX_FLAG_LOAD_AS_IMG)) ||
        loaded.raw_tex_metadata.h_byte_scale != albedo->HByteScale ||
        loaded.raw_tex_metadata.v_pixel_scale != albedo->VPixelScale) {
        return false;
    }
    const auto& path = albedo->GetInitData()->Path;
    if (!path.starts_with("alt/scenes/"))
        return false;
    auto found = mMaterials.find(albedo.get());
    if (found == mMaterials.end()) {
        if (mPendingMaterials.contains(albedo.get()) || mPendingMaterials.size() >= 4)
            return false;
        auto rm = Ship::Context::GetRawInstance()->GetResourceManager();
        if (!rm->GetArchiveManager()->HasFile(path + "_n") || !rm->GetArchiveManager()->HasFile(path + "_r")) {
            mMaterials.emplace(albedo.get(), MaterialMaps{ albedo, nullptr, nullptr });
            return false;
        }
        auto archive = rm->GetArchiveManager()->GetArchiveFromFile(path + "_n");
        if (!archive || archive != rm->GetArchiveManager()->GetArchiveFromFile(path + "_r")) {
            mMaterials.emplace(albedo.get(), MaterialMaps{ albedo, nullptr, nullptr });
            return false;
        }
        auto normal = rm->LoadResourceAsync(path + "_n", true);
        auto roughness = rm->LoadResourceAsync(path + "_r", true);
        // Hashing a large albedo and reading the small recipe must never block a draw. Resource
        // reads themselves share the existing resource pool used by asynchronous alt textures.
        mPendingMaterials.emplace(
            albedo.get(), std::async(std::launch::async, [albedo, normal, roughness, path, archive]() {
                MaterialMaps result{ albedo, nullptr, nullptr };
                try {
                    auto n = std::dynamic_pointer_cast<Fast::Texture>(normal.get());
                    auto r = std::dynamic_pointer_cast<Fast::Texture>(roughness.get());
                    auto valid = [](const std::shared_ptr<Fast::Texture>& t) {
                        return t && t->Type == Fast::TextureType::GrayscaleAlpha16bpp && t->Flags == 0 &&
                               t->Width > 0 && t->Height > 0 && t->ImageDataSize == uint64_t(t->Width) * t->Height * 2;
                    };
                    if (!valid(n) || !valid(r) || n->Width != r->Width || n->Height != r->Height)
                        return result;
                    auto file = archive->LoadFile("newdawn-materials.json");
                    if (!file || !file->Buffer)
                        return result;
                    auto metadata = nlohmann::json::parse(file->Buffer->begin(), file->Buffer->end());
                    for (const auto& entry : metadata.at("materials")) {
                        if ("alt/" + entry.at("resource").get<std::string>() != path)
                            continue;
                        // FNV-1a of decoded RGBA8 plus dimensions/scales ties maps to the winning
                        // albedo without another archive read. SHA-256 of the original stays in the recipe.
                        if (albedo->ImageDataSize != uint64_t(albedo->Width) * albedo->Height * 4 ||
                            entry.at("source_size") != nlohmann::json::array({ albedo->Width, albedo->Height }) ||
                            entry.at("output_size") != nlohmann::json::array({ n->Width, n->Height }) ||
                            entry.at("source_h_byte_scale").get<float>() != albedo->HByteScale ||
                            entry.at("source_v_pixel_scale").get<float>() != albedo->VPixelScale)
                            return result;
                        uint64_t hash = 14695981039346656037ULL;
                        for (uint32_t i = 0; i < albedo->ImageDataSize; ++i)
                            hash = (hash ^ albedo->ImageData[i]) * 1099511628211ULL;
                        if (std::stoull(entry.at("source_pixels_fnv1a64").get<std::string>(), nullptr, 16) != hash)
                            return result;
                        result.normal = n;
                        result.roughness = r;
                        SPDLOG_INFO("NewDawn material ready: {}", path);
                        break;
                    }
                } catch (const std::exception& e) { SPDLOG_WARN("NewDawn material skipped {}: {}", path, e.what()); }
                return result;
            }));
        return false;
    }
    const MaterialMaps& maps = found->second;
    if (!maps.normal || !maps.roughness)
        return false;
    const TextureCacheNode* base = mRenderingState.mTextures[albedoSlot];
    if (!base)
        return false;
    const uint8_t cms = base->second.cms, cmt = base->second.cmt;
    // These slots go idle immediately when a draw is ineligible; they still participate in LRU accounting.
    for (int slot = MATERIAL_NORMAL_TEXTURE; slot <= MATERIAL_ROUGHNESS_TEXTURE; ++slot) {
        const auto& texture = slot == MATERIAL_NORMAL_TEXTURE ? maps.normal : maps.roughness;
        TextureCacheKey key{ texture->ImageData, {}, 0, 0, 0, texture->ImageDataSize };
        if (!mRenderingState.mTextures[slot] || mRenderingState.mTextures[slot]->first != key) {
            Flush();
            if (!TextureCacheLookup(slot, key)) {
                mRapi->UploadMaterialTexture(texture->ImageData, texture->Width, texture->Height);
                mRapi->BuildMipmaps();
            }
        }
        auto& state = mRenderingState.mTextures[slot]->second;
        if (!state.linear_filter || state.cms != cms || state.cmt != cmt) {
            Flush();
            mRapi->SetSamplerParameters(slot, true, cms, cmt);
            state.linear_filter = true;
            state.cms = cms;
            state.cmt = cmt;
        }
        UpdateTextureCacheSize(slot);
    }
    return true;
}

void Interpreter::TextureCacheClear() {
    for (const auto& entry : mTextureCache.map) {
        mTextureCache.free_texture_ids.push_back(entry.second.texture_id);
    }
    mTextureCache.map.clear();
    mTextureCache.lru.clear();
    mTextureCache.total_bytes = 0;
    mTextureCache.eviction_count = 0;
    mResolvedResourceCache.clear();
    mPendingMaterials.clear();
    mMaterials.clear();
    // Pre-allocate buckets so the map never rehashes during normal operation.
    // Rehashing invalidates all iterators, including those stored in LRU entries.
    mTextureCache.map.reserve(TEXTURE_CACHE_MAX_SIZE);
    // Null rendering-state pointers — they pointed into map nodes that are now freed.
    std::fill(std::begin(mRenderingState.mTextures), std::end(mRenderingState.mTextures), nullptr);
}

void Interpreter::ShaderCacheClear() {
    mRapi->ClearShaderCache();
}

bool Interpreter::TextureCacheLookup(int i, const TextureCacheKey& key) {
    TextureCacheMap::iterator it = mTextureCache.map.find(key);
    TextureCacheNode** n = &mRenderingState.mTextures[i];

    if (it != mTextureCache.map.end()) {
        mRapi->SelectTexture(i, it->second.texture_id);
        *n = &*it;
        mTextureCache.lru.splice(mTextureCache.lru.end(), mTextureCache.lru,
                                 it->second.lru_location); // move to back
        return true;
    }

    // Make room for the new entry. The byte budget is enforced after the upload, when the
    // backend knows the texture's dimensions and mip state.
    while (mTextureCache.map.size() >= TEXTURE_CACHE_MAX_SIZE) {
        if (!EvictOneTextureCacheEntry(nullptr)) {
            break;
        }
    }

    uint32_t texture_id;
    if (!mTextureCache.free_texture_ids.empty()) {
        texture_id = mTextureCache.free_texture_ids.back();
        mTextureCache.free_texture_ids.pop_back();
    } else {
        texture_id = mRapi->NewTexture();
    }

    it = mTextureCache.map.insert(std::make_pair(key, TextureCacheValue())).first;
    TextureCacheNode* node = &*it;
    node->second.texture_id = texture_id;
    node->second.byte_size = 0;
    node->second.lru_location = mTextureCache.lru.insert(mTextureCache.lru.end(), { it });

    mRapi->SelectTexture(i, texture_id);
    mRapi->SetSamplerParameters(i, false, 0, 0);
    *n = node;
    return false;
}

bool Interpreter::TextureCacheNodeInUse(const TextureCacheNode* node) const {
    if (node->second.shadow_pin == mShadowSerial) {
        return true; // a recorded foliage caster samples this texture in the shadow pass later this frame
    }
    for (int i = 0; i < MATERIAL_TEXTURE_SLOTS; i++) {
        if (mRenderingState.mTextures[i] == node) {
            return true;
        }
    }
    return false;
}

bool Interpreter::EvictOneTextureCacheEntry(const TextureCacheNode* protect) {
    for (auto lruIt = mTextureCache.lru.begin(); lruIt != mTextureCache.lru.end(); ++lruIt) {
        TextureCacheNode* node = &*lruIt->it;
        if (node == protect || TextureCacheNodeInUse(node)) {
            continue;
        }

        mTextureCache.total_bytes -= node->second.byte_size;
        mTextureCache.free_texture_ids.push_back(node->second.texture_id);
        for (int i = 0; i < MATERIAL_TEXTURE_SLOTS; i++) {
            if (mRenderingState.mTextures[i] == node) {
                mRenderingState.mTextures[i] = nullptr;
            }
        }
        mTextureCache.map.erase(lruIt->it);
        mTextureCache.lru.erase(lruIt);
        mTextureCache.eviction_count++;

        if (mTextureCacheDebug && (mTextureCache.eviction_count == 1 || mTextureCache.eviction_count % 64 == 0)) {
            SPDLOG_INFO("NewDawn texture cache: evicted {} textures, {} entries, {:.1f}/{:.1f} MiB",
                        mTextureCache.eviction_count, mTextureCache.map.size(),
                        (double)mTextureCache.total_bytes / (1024.0 * 1024.0),
                        (double)mTextureCacheBudgetBytes / (1024.0 * 1024.0));
        }
        return true;
    }
    return false;
}

void Interpreter::EnforceTextureCacheBudget(const TextureCacheNode* protect) {
    while (mTextureCache.map.size() > TEXTURE_CACHE_MAX_SIZE) {
        if (!EvictOneTextureCacheEntry(protect)) {
            break;
        }
    }
    if (mTextureCacheBudgetBytes == 0) {
        return;
    }
    while (mTextureCache.total_bytes > mTextureCacheBudgetBytes) {
        if (!EvictOneTextureCacheEntry(protect)) {
            break;
        }
    }
}

void Interpreter::UpdateTextureCacheSize(int textureSlot) {
    TextureCacheNode* node = mRenderingState.mTextures[textureSlot];
    if (node == nullptr) {
        return;
    }
    uint64_t byteSize = mRapi->GetTextureByteSize(node->second.texture_id);
    if (byteSize != node->second.byte_size) {
        mTextureCache.total_bytes -= node->second.byte_size;
        node->second.byte_size = byteSize;
        mTextureCache.total_bytes += byteSize;
    }
    EnforceTextureCacheBudget(node);
}

std::string_view Interpreter::GetBaseTexturePath(std::string_view path) {
    if (path.starts_with(Ship::IResource::gAltAssetPrefix)) {
        return path.substr(Ship::IResource::gAltAssetPrefix.length());
    }

    return path;
}

void Interpreter::TextureCacheDelete(const uint8_t* origAddr) {
    while (mTextureCache.map.bucket_count() > 0) {
        TextureCacheKey key = { origAddr, { 0 }, 0, 0, 0 }; // bucket index only depends on the address
        size_t bucket = mTextureCache.map.bucket(key);
        bool again = false;
        for (auto it = mTextureCache.map.begin(bucket); it != mTextureCache.map.end(bucket); ++it) {
            if (it->first.texture_addr == origAddr) {
                for (int j = 0; j < MATERIAL_TEXTURE_SLOTS; j++) {
                    if (mRenderingState.mTextures[j] == &*it)
                        mRenderingState.mTextures[j] = nullptr;
                }
                mTextureCache.lru.erase(it->second.lru_location);
                mTextureCache.total_bytes -= it->second.byte_size;
                mTextureCache.free_texture_ids.push_back(it->second.texture_id);
                mTextureCache.map.erase(it->first);
                again = true;
                break;
            }
        }
        if (!again) {
            break;
        }
    }
}

static bool Invert4x4(const float m[4][4], float out[16]) {
    float a[4][8];
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            a[row][col] = m[row][col];
            a[row][col + 4] = (row == col) ? 1.0f : 0.0f;
        }
    }
    for (int col = 0; col < 4; col++) {
        int pivot = col;
        for (int row = col + 1; row < 4; row++) {
            if (fabsf(a[row][col]) > fabsf(a[pivot][col])) {
                pivot = row;
            }
        }
        if (fabsf(a[pivot][col]) < 1e-8f) {
            return false;
        }
        if (pivot != col) {
            for (int k = 0; k < 8; k++) {
                std::swap(a[pivot][k], a[col][k]);
            }
        }
        const float invPivot = 1.0f / a[col][col];
        for (int k = 0; k < 8; k++) {
            a[col][k] *= invPivot;
        }
        for (int row = 0; row < 4; row++) {
            if (row == col) {
                continue;
            }
            const float factor = a[row][col];
            if (factor == 0.0f) {
                continue;
            }
            for (int k = 0; k < 8; k++) {
                a[row][k] -= factor * a[col][k];
            }
        }
    }
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            out[row * 4 + col] = a[row][col + 4];
        }
    }
    return true;
}

void Interpreter::BeginShadowRecording(bool enabled, bool captureCamera) {
    mShadowRecording = enabled;
    mShadowOnlyTris = 0;
    mShadowOnlyNs = 0;
    mShadowVertices.clear();
    mShadowOverflowLogged = false;
    mShadowSerial++; // expires last frame's texture pins
    mShadowFoliage = enabled &&
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Shadows.Foliage", 0) != 0;
    mShadowFoliageByTex.clear();
    mShadowFoliageFloats = 0;
    mShadowExactProj = false;
    if (!enabled && !captureCamera) {
        return;
    }
    if (enabled && mShadowVertices.capacity() < 3 * 1024 * 1024) {
        mShadowVertices.reserve(3 * 1024 * 1024);
    }

    mShadowExactProj = Invert4x4(mRsp->P_matrix, mShadowInvViewProj);
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            mShadowViewProj[row * 4 + col] = mRsp->P_matrix[row][col];
        }
    }
    const float ndcXScale = AdjXForAspectRatio(1.0f);
    mShadowNdcXScale = (ndcXScale > 1e-6f) ? (1.0f / ndcXScale) : 1.0f;
    mShadowFlipY = mRapi->GetClipParameters().invertY ? -1.0f : 1.0f;
}

void Interpreter::FillShadowCameraParams(NewDawnPostFxParams& params) const {
    params.shadowUseExactProj = mShadowExactProj ? 1 : 0;
    params.shadowNdcXScale = mShadowNdcXScale;
    params.shadowFlipY = mShadowFlipY;
    memcpy(params.shadowViewProj, mShadowViewProj, sizeof(mShadowViewProj));
    memcpy(params.shadowInvViewProj, mShadowInvViewProj, sizeof(mShadowInvViewProj));
}

void Interpreter::EndShadowRecording() {
    mShadowRecording = false;
}

void Interpreter::SubmitShadowCasters() {
    if (const int dbg = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Shadows.CoverageDebug", 0)) {
        // 1: a line every 30 frames; 2: every frame, with the camera yaw (the measurement fixture of docs/CASCADED_SHADOWS.md)
        static uint32_t sFrame = 0;
        if (dbg >= 2 || sFrame++ % 30 == 0) {
            float a[4] = { 0, 0, -1, 1 }, b[4] = { 0, 0, 1, 1 }, wa[4] = {}, wb[4] = {};
            for (int j = 0; j < 4; j++) {
                for (int i = 0; i < 4; i++) {
                    wa[j] += a[i] * mShadowInvViewProj[i * 4 + j];
                    wb[j] += b[i] * mShadowInvViewProj[i * 4 + j];
                }
            }
            const float fx = wb[0] / wb[3] - wa[0] / wa[3], fz = wb[2] / wb[3] - wa[2] / wa[3];
            SPDLOG_INFO("NewDawn shadow casters: {} triangles ({} from the coverage pass) yaw {:.2f}",
                        mShadowVertices.size() / 9, mShadowOnlyTris, std::atan2(fx, fz) * 57.29578f);
            static double sNs = 0.0;
            static int sN = 0;
            sNs += mShadowOnlyNs;
            if (++sN == 120) {
                SPDLOG_INFO("NewDawn coverage pass CPU: {:.4f} ms/frame (120 frames)", sNs / sN / 1.0e6);
                sNs = 0.0;
                sN = 0;
            }
        }
    }
    mRapi->SetShadowCasterGeometry(mShadowVertices.empty() ? nullptr : mShadowVertices.data(),
                                   (uint32_t)(mShadowVertices.size() / 3));
    mShadowFoliageVerts.clear();
    mShadowFoliageBatches.clear();
    for (const auto& [textureId, floats] : mShadowFoliageByTex) {
        mShadowFoliageBatches.push_back(
            { textureId, (uint32_t)(mShadowFoliageVerts.size() / 5), (uint32_t)(floats.size() / 5) });
        mShadowFoliageVerts.insert(mShadowFoliageVerts.end(), floats.begin(), floats.end());
    }
    mRapi->SetShadowFoliageGeometry(mShadowFoliageVerts.data(), mShadowFoliageBatches.data(),
                                    (uint32_t)mShadowFoliageBatches.size());
}

// Same UV derivation as the main draw (post-texgen, tile shift/offset, linear-filter half texel), so the shadow's
// cut-out lines up with the leaf the player sees.
void Interpreter::RecordShadowFoliageTriangle(const LoadedVertex* const v[3], uint32_t tile, uint32_t texWidth,
                                              uint32_t texHeight, TextureCacheNode* node) {
    constexpr size_t kMaxFoliageFloats = 5 * 1024 * 1024;
    if (mShadowFoliageFloats + 15 > kMaxFoliageFloats || texWidth == 0 || texHeight == 0) {
        return;
    }
    node->second.shadow_pin = mShadowSerial;
    auto& out = mShadowFoliageByTex[node->second.texture_id];
    const int shifts = mRdp->texture_tile[tile].shifts;
    const int shiftt = mRdp->texture_tile[tile].shiftt;
    const bool linear = (mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) != G_TF_POINT;
    for (int i = 0; i < 3; i++) {
        float u = v[i]->u / 32.0f;
        float t = v[i]->v / 32.0f;
        if (shifts != 0) {
            u = shifts <= 10 ? u / (1 << shifts) : u * (1 << (16 - shifts));
        }
        if (shiftt != 0) {
            t = shiftt <= 10 ? t / (1 << shiftt) : t * (1 << (16 - shiftt));
        }
        u -= mRdp->texture_tile[tile].uls / 4.0f;
        t -= mRdp->texture_tile[tile].ult / 4.0f;
        if (linear) {
            u += 0.5f;
            t += 0.5f;
        }
        out.push_back(v[i]->view_pos[0]);
        out.push_back(v[i]->view_pos[1]);
        out.push_back(v[i]->view_pos[2]);
        out.push_back(u / texWidth);
        out.push_back(t / texHeight);
    }
    mShadowFoliageFloats += 15;
}

// True when the alpha combiner output is the tile-0 texel alpha (optionally modulated by a constant or vertex
// alpha), so a texel-alpha test in the shadow pass reproduces the cutout. Anything fed by TEXEL1, LOD or the
// combined-colour of an unrelated cycle is left out.
static bool AlphaCombinerUsesTexel0(uint64_t combineMode, bool twoCycle) {
    bool texel0 = false;
    for (uint32_t cycle = 0; cycle < (twoCycle ? 2u : 1u); cycle++) {
        for (uint32_t k = 0; k < 4; k++) {
            const uint32_t mux = (combineMode >> (cycle * 28 + 16 + k * 3)) & 7;
            // In the second cycle the hardware swaps the texel inputs: TEXEL1 reads slot 0 there.
            const bool slot0 = twoCycle ? (cycle == 0 ? mux == G_ACMUX_TEXEL0 : mux == G_ACMUX_TEXEL1)
                                        : (mux == G_ACMUX_TEXEL0 || mux == G_ACMUX_TEXEL1);
            const bool slot1 = twoCycle && (cycle == 0 ? mux == G_ACMUX_TEXEL1 : mux == G_ACMUX_TEXEL0);
            if (slot1 || (k == 2 && (mux == G_ACMUX_LOD_FRACTION || mux == G_ACMUX_PRIM_LOD_FRAC))) {
                return false;
            }
            texel0 |= slot0;
        }
    }
    return texel0;
}

// ---- New Dawn VFX: native effect sprites are recognised by texture path, recorded in clip space and drawn again by the
// renderer as HD HDR sprites. Everything here is render-side; no game state is read beyond the draw stream.
enum { VFX_NONE = 0, VFX_FLAME = 1, VFX_GLOW = 2, VFX_DUST = 3, VFX_BOMB = 4, VFX_HIT = 5, VFX_SPARKLE = 6, VFX_SPARK = 7,
       VFX_CANDLE = 8 /* recorded as a flame whose alpha is forced: its draw leaves prim alpha at 0 */,
       VFX_ARROW_FIRE = 9, VFX_ARROW_ICE = 10, VFX_ARROW_LIGHT = 11 /* whole 3D model, accumulated into one instance */,
       VFX_WATER_SPLASH = 12, VFX_WATER_RIPPLE = 13 /* native draw kept, only dimmed (see mVfxDim) */ };

void Interpreter::BeginVfxRecording(bool enabled) {
    mVfx = enabled;
    mVfxInst.clear();
    mVfxTriSerial = mVfxLastTri = 0;
    mVfxLastClass = 0;
    mVfxArrowPts.clear();
    mVfxArrowInst = -1;
    mVfxDebug = enabled && Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Vfx.Debug", 0);
}

uint8_t Interpreter::VfxClassify(std::string_view path) {
    // Prefix -> class. Numbered animation frames (gEffDust1..8Tex) carry their frame in the high nibble.
    static const std::pair<std::string_view, uint8_t> kTable[] = {
        { "gEffFireTex", VFX_FLAME },
        { "gEffEnemyDeathFlame", VFX_FLAME },
        { "gameplay_keep_Tex_01A8B0", VFX_CANDLE }, // gameplay_keep_DL_01ACF0: En_Light's small flame (Inn candles)
        { "gameplay_keep_Tex_01AAB0", VFX_CANDLE },
        { "gameplay_keep_Tex_08FBE0", VFX_GLOW }, // Lights_DrawGlow: the hard-edged light disc
        { "gEffDust", VFX_DUST },
        { "gEffHitmarkDust", VFX_DUST },
        { "gEffBombExplosion", VFX_BOMB },
        { "gEffHitmarkRed", VFX_HIT },
        { "gEffHitmarkWhiteMetal", VFX_HIT },
        { "gElectricSpark", VFX_SPARK },
        // Water splashes/ripples (EffectSsGSplash/GRipple) are opaque-white native quads that read blown out in HDR; they are
        // not replaced, only dimmed at draw time. Kept out of the dust class (frame digits) on purpose.
        { "gEffWaterSplash", VFX_WATER_SPLASH },
        { "gEffWaterRipple", VFX_WATER_RIPPLE },
        { "gFireArrow", VFX_ARROW_FIRE }, // Tex and MaskTex: the 43-vertex glow model
        { "gIceArrow", VFX_ARROW_ICE },
        { "gLightArrow", VFX_ARROW_LIGHT },
        // Sparkles are deliberately NOT classified: the native small pink four-point stars are crisper than the Kenney
        // star_01 replacement (docs/VFX.md). Re-add { "gameplay_keep_Tex_054AF0", VFX_SPARKLE } with a better sprite to enable.
        // { "gameplay_keep_Tex_054AF0", VFX_SPARKLE }, // gEffSparklesDL: fairy sparkles, rupee/item glitter
    };
    uint8_t cls = VFX_NONE;
    for (const auto& [name, c] : kTable) {
        const size_t pos = path.find(name);
        if (pos == std::string_view::npos) {
            continue;
        }
        cls = c;
        const size_t digit = pos + name.size();
        if (c >= VFX_DUST && c < VFX_WATER_SPLASH && digit < path.size() && path[digit] >= '1' && path[digit] <= '8') {
            cls |= (uint8_t)((path[digit] - '1') << 4);
        }
        break;
    }
    if (mVfxDebug) {
        static std::unordered_set<std::string> seen;
        // Vfx.DebugAll=1 logs every texture path (misclassification hunting); otherwise only gEff* ones.
        static const bool all = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Vfx.DebugAll", 0);
        if ((all || path.find("gEff") != std::string_view::npos) && seen.insert(std::string(path)).second) {
            SPDLOG_INFO("VFXDBG texture {} class {}", path, cls);
        }
    }
    return cls;
}

void Interpreter::RecordVfxTriangle(const LoadedVertex* const v[3], uint8_t cls, float frame) {
    float minx = 1e9f, maxx = -1e9f, miny = 1e9f, maxy = -1e9f, minz = 1e9f, maxz = -1e9f, w = 0, wx = 0, wy = 0, wz = 0;
    for (int i = 0; i < 3; i++) {
        if (v[i]->w < 1.0f) {
            return;
        }
        const float x = v[i]->x / v[i]->w, y = v[i]->y / v[i]->w;
        minx = std::min(minx, x), maxx = std::max(maxx, x), miny = std::min(miny, y), maxy = std::max(maxy, y);
        minz = std::min(minz, v[i]->z / v[i]->w), maxz = std::max(maxz, v[i]->z / v[i]->w);
        w += v[i]->w / 3.0f;
        wx += v[i]->view_pos[0] / 3.0f, wy += v[i]->view_pos[1] / 3.0f, wz += v[i]->view_pos[2] / 3.0f;
    }
    const uint32_t serial = mVfxTriSerial;
    bool merged = false;
    if (!mVfxInst.empty() && mVfxLastClass == cls && mVfxLastTri + 1 == serial && mVfxLastCount == 1) {
        for (int i = 0; i < 3 && !merged; i++)
            for (int j = 0; j < 3; j++)
                merged |= v[i] == mVfxLastVerts[j];
    }
    float* inst;
    if (merged) {
        inst = &mVfxInst[mVfxInst.size() - NEWDAWN_VFX_FLOATS];
        const float ox0 = inst[1] - inst[5], ox1 = inst[1] + inst[5], oy0 = inst[2] - inst[6], oy1 = inst[2] + inst[6];
        minx = std::min(minx, ox0), maxx = std::max(maxx, ox1), miny = std::min(miny, oy0), maxy = std::max(maxy, oy1);
        // Average the two triangles' centres: the quad centre.
        inst[12] = (inst[12] + wx) * 0.5f, inst[13] = (inst[13] + wy) * 0.5f, inst[14] = (inst[14] + wz) * 0.5f;
    } else {
        if (mVfxInst.size() / NEWDAWN_VFX_FLOATS >= 2048)
            return;
        mVfxInst.resize(mVfxInst.size() + NEWDAWN_VFX_FLOATS, 0.0f);
        inst = &mVfxInst[mVfxInst.size() - NEWDAWN_VFX_FLOATS];
        inst[0] = cls;
        inst[12] = wx, inst[13] = wy, inst[14] = wz;
        inst[8] = mRdp->prim_color.r / 255.0f, inst[9] = mRdp->prim_color.g / 255.0f;
        inst[10] = mRdp->prim_color.b / 255.0f, inst[11] = mRdp->prim_color.a / 255.0f;
        inst[16] = mRdp->env_color.r / 255.0f, inst[17] = mRdp->env_color.g / 255.0f;
        inst[18] = mRdp->env_color.b / 255.0f, inst[19] = mRdp->env_color.a / 255.0f;
    }
    inst[1] = (minx + maxx) * 0.5f, inst[2] = (miny + maxy) * 0.5f, inst[3] = (minz + maxz) * 0.5f;
    inst[4] = w, inst[5] = (maxx - minx) * 0.5f, inst[6] = (maxy - miny) * 0.5f;
    // Stable per-source seed: the world position quantised to 4 units (never uses RNG or game state).
    inst[7] = fmodf(floorf(inst[12] / 4.0f) * 12.9898f + floorf(inst[13] / 4.0f) * 78.233f +
                        floorf(inst[14] / 4.0f) * 37.719f,
                    1000.0f);
    mVfxLastClass = cls, mVfxLastTri = serial;
    mVfxLastCount = merged ? 2 : 1; // a quad is exactly two triangles
    for (int i = 0; i < 3; i++)
        mVfxLastVerts[i] = v[i];
    inst[15] = frame;
}

// A magic arrow is a 3D model, not a quad: every triangle of one draw is folded into a single instance describing the
// model's screen-space axis. A=[cls, centre xy, z], B=[w, half-length vector xy (NDC), seed], [15] = half-width (NDC y).
void Interpreter::RecordVfxArrow(const LoadedVertex* const v[3], uint8_t cls) {
    if (v[0]->w < 1.0f && v[1]->w < 1.0f && v[2]->w < 1.0f) {
        return;
    }
    const uint32_t serial = mVfxTriSerial;
    const bool fresh = mVfxArrowInst < 0 || serial - mVfxArrowLastTri > 6 ||
                       mVfxInst[mVfxArrowInst] != (float)cls;
    if (fresh) {
        if (mVfxInst.size() / NEWDAWN_VFX_FLOATS >= 2048) {
            return;
        }
        mVfxArrowPts.clear();
        mVfxArrowInst = (int64_t)mVfxInst.size();
        mVfxInst.resize(mVfxInst.size() + NEWDAWN_VFX_FLOATS, 0.0f);
        float* n = &mVfxInst[mVfxArrowInst];
        n[0] = cls;
        n[8] = mRdp->prim_color.r / 255.0f, n[9] = mRdp->prim_color.g / 255.0f;
        n[10] = mRdp->prim_color.b / 255.0f, n[11] = mRdp->prim_color.a / 255.0f;
        n[16] = mRdp->env_color.r / 255.0f, n[17] = mRdp->env_color.g / 255.0f;
        n[18] = mRdp->env_color.b / 255.0f, n[19] = mRdp->env_color.a / 255.0f;
    }
    mVfxArrowLastTri = serial;
    float* inst = &mVfxInst[mVfxArrowInst];
    float zmin = 1e9f;
    for (int i = 0; i < 3; i++) {
        if (v[i]->w < 1.0f) {
            continue; // the part of the model behind the near plane is clipped by the game; keep the visible vertices
        }
        mVfxArrowPts.push_back(v[i]->x / v[i]->w);
        mVfxArrowPts.push_back(v[i]->y / v[i]->w);
        mVfxArrowPts.push_back(v[i]->z / v[i]->w);
        mVfxArrowPts.push_back(v[i]->w);
        mVfxArrowPts.push_back(v[i]->view_pos[0]);
        mVfxArrowPts.push_back(v[i]->view_pos[1]);
        mVfxArrowPts.push_back(v[i]->view_pos[2]);
        zmin = std::min(zmin, v[i]->z / v[i]->w);
    }
    // Principal axis of the projected points (the arrow is a long ellipsoid), then extents along and across it.
    const size_t count = mVfxArrowPts.size() / 7;
    double mx = 0, my = 0, mw = 0, wx = 0, wy = 0, wz = 0, mz = 0;
    for (size_t i = 0; i < count; i++) {
        const float* p = &mVfxArrowPts[i * 7];
        mx += p[0], my += p[1], mz += p[2], mw += p[3], wx += p[4], wy += p[5], wz += p[6];
    }
    mx /= count, my /= count, mz /= count, mw /= count, wx /= count, wy /= count, wz /= count;
    double sxx = 0, sxy = 0, syy = 0;
    for (size_t i = 0; i < count; i++) {
        const float* p = &mVfxArrowPts[i * 7];
        const double dx = p[0] - mx, dy = p[1] - my;
        sxx += dx * dx, sxy += dx * dy, syy += dy * dy;
    }
    const double ang = 0.5 * atan2(2.0 * sxy, sxx - syy);
    const float ax = (float)cos(ang), ay = (float)sin(ang);
    float lo = 1e9f, hi = -1e9f, wid = 0.0f;
    for (size_t i = 0; i < count; i++) {
        const float* p = &mVfxArrowPts[i * 7];
        const float dx = p[0] - (float)mx, dy = p[1] - (float)my;
        const float along = dx * ax + dy * ay, across = -dx * ay + dy * ax;
        lo = std::min(lo, along), hi = std::max(hi, along), wid = std::max(wid, fabsf(across));
    }
    const float mid = 0.5f * (lo + hi), half = 0.5f * (hi - lo);
    inst[1] = (float)mx + ax * mid, inst[2] = (float)my + ay * mid, inst[3] = (float)mz;
    inst[4] = (float)mw, inst[5] = ax * half, inst[6] = ay * half;
    inst[12] = (float)wx, inst[13] = (float)wy, inst[14] = (float)wz;
    inst[7] = fmodf(floorf(inst[12] / 4.0f) * 12.9898f + floorf(inst[13] / 4.0f) * 78.233f +
                        floorf(inst[14] / 4.0f) * 37.719f, 1000.0f);
    inst[15] = wid;
    // Burst test in 3D (view space): the screen-space axis is foreshortened whenever the arrow points at or away from the
    // camera (aiming), which used to look like the impact's near-round model. Store half-width / half-length of the 3D
    // point cloud in env.a: flight/aim stay thin (< 0.5), the game's impact scale-up (x/z up to 64 vs y 8) is >> 1.
    {
        double c[3] = { 0, 0, 0 };
        for (size_t i = 0; i < count; i++)
            for (int k = 0; k < 3; k++)
                c[k] += mVfxArrowPts[i * 7 + 4 + k] / count;
        double cov[3][3] = {};
        for (size_t i = 0; i < count; i++) {
            double d[3];
            for (int k = 0; k < 3; k++)
                d[k] = mVfxArrowPts[i * 7 + 4 + k] - c[k];
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    cov[a][b] += d[a] * d[b];
        }
        double ax3[3] = { 1, 0.7, 0.3 };
        for (int it = 0; it < 12; it++) {
            double n[3];
            double len = 0;
            for (int a = 0; a < 3; a++) {
                n[a] = cov[a][0] * ax3[0] + cov[a][1] * ax3[1] + cov[a][2] * ax3[2];
                len += n[a] * n[a];
            }
            len = sqrt(len) + 1e-12;
            for (int a = 0; a < 3; a++)
                ax3[a] = n[a] / len;
        }
        double along3 = 0, across3 = 0;
        for (size_t i = 0; i < count; i++) {
            double d[3], proj = 0, r2 = 0;
            for (int k = 0; k < 3; k++) {
                d[k] = mVfxArrowPts[i * 7 + 4 + k] - c[k];
                proj += d[k] * ax3[k];
            }
            for (int k = 0; k < 3; k++)
                r2 += (d[k] - proj * ax3[k]) * (d[k] - proj * ax3[k]);
            along3 = std::max(along3, fabs(proj)), across3 = std::max(across3, sqrt(r2));
        }
        inst[19] = (float)(across3 / std::max(along3, 1e-3));
    }
}

void Interpreter::SubmitVfx() {
    if (mVfxDebug) {
        static int sLogged = 0;
        const int dbg = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Vfx.Debug", 0);
        for (size_t i = 0; i + NEWDAWN_VFX_FLOATS <= mVfxInst.size() && sLogged < (dbg >= 2 ? 200 : 60); i += NEWDAWN_VFX_FLOATS, ++sLogged) {
            const float* f = &mVfxInst[i];
            if (dbg >= 2 && f[0] != 6.0f) {
                continue; // Debug=2: sparkles only
            }
            SPDLOG_INFO("VFXDBG inst cls={:.0f} world={:.0f},{:.0f},{:.0f} w={:.0f} ndc={:.3f},{:.3f} half={:.3f},{:.3f} wid={:.3f} prim={:.2f},{:.2f},{:.2f},{:.2f}",
                        f[0], f[12], f[13], f[14], f[4], f[1], f[2], f[5], f[6], f[15], f[8], f[9], f[10], f[11]);
        }
    }
    mRapi->SetVfxSprites(mVfxInst.empty() ? nullptr : mVfxInst.data(), (uint32_t)(mVfxInst.size() / NEWDAWN_VFX_FLOATS));
}

// ---- New Dawn grass: ground triangles whose texture is classified "grass" (tools/newdawn/grass_textures.py) are recorded in
// world space and scattered with render-only tufts by the renderer at the opaque-end marker. Nothing here reads or changes
// game state beyond the draw stream.
namespace {
const char* const kGrassTextures[] = {
#include "fast/newdawn_grass_textures.inc"
};
const char* const kSwayTextures[] = {
#include "fast/newdawn_sway_textures.inc"
};
} // namespace

void Interpreter::BeginGrassRecording(bool enabled) {
    mGrass = enabled;
    mGrassTris.clear();
    mGrassDebug = enabled && Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Grass.Debug", 0);
}

uint16_t Interpreter::GrassClassify(std::string_view path) {
    static const std::unordered_map<std::string_view, uint16_t> table = [] {
        std::unordered_map<std::string_view, uint16_t> t;
        for (size_t i = 0; i < sizeof(kGrassTextures) / sizeof(kGrassTextures[0]); i++) {
            t.emplace(kGrassTextures[i], (uint16_t)(i + 1));
        }
        return t;
    }();
    const size_t slash = path.rfind('/');
    const auto found = table.find(slash == std::string_view::npos ? path : path.substr(slash + 1));
    return found == table.end() ? 0 : found->second;
}

// Only upright geometry sways: floors, lawns and near-level cut-outs are never displaced (room geometry: |n.y| >= 0.75,
// within about 41 degrees of level; foliage actors, whose drooping leaf cards are tilted: |n.y| >= 0.95), so ground
// triangles that share edges with plants cannot open cracks.
static bool WindTriangleUpright(const LoadedVertex* const v[3], float maxNormalY) {
    const float* a = v[0]->view_pos;
    const float* b = v[1]->view_pos;
    const float* c = v[2]->view_pos;
    const float e1[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, e2[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
    const float n[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
    const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    const float ext = std::max({ a[1], b[1], c[1] }) - std::min({ a[1], b[1], c[1] });
    return len >= 1e-3f && std::fabs(n[1]) < maxNormalY * len && ext > 0.5f;
}

bool Interpreter::SwayClassify(std::string_view path) {
    static const std::unordered_set<std::string_view> table(std::begin(kSwayTextures), std::end(kSwayTextures));
    const size_t slash = path.rfind('/');
    return table.count(slash == std::string_view::npos ? path : path.substr(slash + 1)) != 0;
}

void Interpreter::RecordGrassTriangle(const LoadedVertex* const v[3]) {
    constexpr size_t kMaxGrassTriangles = 200000;
    if (mGrassTris.size() >= kMaxGrassTriangles * NEWDAWN_GRASS_TRI_FLOATS) {
        return;
    }
    const float* a = v[0]->view_pos;
    const float* b = v[1]->view_pos;
    const float* c = v[2]->view_pos;
    const float e1[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, e2[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
    const float n[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
    const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    // Ground only: walls, banks and degenerate slivers get no grass (|n.y| >= 0.85: within about 32 degrees of level).
    if (len < 1e-3f || std::fabs(n[1]) < 0.85f * len) {
        return;
    }
    for (int i = 0; i < 3; i++) {
        mGrassTris.insert(mGrassTris.end(), v[i]->view_pos, v[i]->view_pos + 3);
    }
}

// ---- New Dawn interior polish: window and emissive textures (tools/newdawn: newdawn_interior_textures.inc, listed by
// resource basename) are recorded in world space, merged into openings and emissive lights, and handed to the renderer in
// the post-fx params at the opaque-end marker. Emissive draws also write a stencil class (see Flush). Read-only.
namespace {
enum { INTERIOR_NONE = 0, INTERIOR_WINDOW = 1, INTERIOR_GLASS = 2, INTERIOR_LAMP = 3 };
struct InteriorTexture {
    const char* name;
    uint8_t kind;
    float a, b, c, d; // window: inset u, inset v, panes u, panes v. glass/lamp: tint rgb, light intensity (0 = no light)
};
const InteriorTexture kInteriorTextures[] = {
#include "fast/newdawn_interior_textures.inc"
};

struct InteriorCluster {
    uint8_t idx;
    float n[3];
    float mn[3], mx[3];
    bool cullBack;
    std::vector<float> verts;
};
} // namespace

void Interpreter::BeginInteriorRecording(bool enabled, bool debug) {
    mInterior = enabled;
    mInteriorDebug = enabled && debug;
    mInteriorTris.clear();
    mInteriorBBox[0] = mInteriorBBox[1] = mInteriorBBox[2] = 1e9f;
    mInteriorBBox[3] = mInteriorBBox[4] = mInteriorBBox[5] = -1e9f;
}

uint8_t Interpreter::InteriorClassify(std::string_view path) {
    static const std::unordered_map<std::string_view, uint8_t> table = [] {
        std::unordered_map<std::string_view, uint8_t> t;
        for (size_t i = 0; i < sizeof(kInteriorTextures) / sizeof(kInteriorTextures[0]); i++) {
            t.emplace(kInteriorTextures[i].name, (uint8_t)(i + 1));
        }
        return t;
    }();
    const size_t slash = path.rfind('/');
    const auto found = table.find(slash == std::string_view::npos ? path : path.substr(slash + 1));
    return found == table.end() ? 0 : found->second;
}

void Interpreter::RecordInteriorTriangle(const LoadedVertex* const v[3], uint8_t idx) {
    constexpr size_t kMaxTriangles = 4096;
    if (mInteriorTris.size() >= kMaxTriangles * 11) {
        return;
    }
    for (int i = 0; i < 3; i++) {
        mInteriorTris.insert(mInteriorTris.end(), v[i]->view_pos, v[i]->view_pos + 3);
    }
    mInteriorTris.push_back((float)idx);
    const uint32_t cull_back = get_attr(CULL_BACK);
    mInteriorTris.push_back((mRsp->geometry_mode & cull_back) != 0 ? 1.0f : 0.0f);
}

void Interpreter::SubmitInterior(NewDawnPostFxParams& params) {
    // The game side already put the render-only windows (kind 2) of the scene in front; recorded openings follow.
    params.emissiveLightCount = 0;
    // Developer test opening, "dx,dy,dz,nx,ny,nz,halfWidth,halfHeight" relative to the camera (gNewDawn.Interior.TestWindow).
    {
        std::string test = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetString("gNewDawn.Interior.TestWindow", "");
        const bool absolute = !test.empty() && test[0] == 'a'; // "a:" prefix: world coordinates instead of eye-relative
        if (absolute) {
            test.erase(0, 2);
        }
        float t[8];
        if (!test.empty() && sscanf(test.c_str(), "%f,%f,%f,%f,%f,%f,%f,%f", &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], &t[6], &t[7]) == 8 &&
            params.interiorEnabled && params.interiorWindowCount < NEWDAWN_INTERIOR_WINDOWS) {
            float* o = params.interiorWindows[params.interiorWindowCount++];
            const float nl = std::max(std::sqrt(t[3] * t[3] + t[4] * t[4] + t[5] * t[5]), 1e-3f);
            for (int a = 0; a < 3; a++) {
                o[a] = (absolute ? 0.0f : params.giCameraEye[a]) + t[a];
                o[3 + a] = t[3 + a] / nl;
            }
            o[6] = o[5], o[7] = 0.0f, o[8] = -o[3];
            const float ul = std::max(std::sqrt(o[6] * o[6] + o[8] * o[8]), 1e-3f);
            o[6] /= ul, o[8] /= ul;
            o[9] = t[6], o[10] = t[7], o[11] = 2, o[12] = 2, o[13] = 2;
        }
    }
    if (mInteriorDebug) {
        static uint32_t sBox = 0;
        if (sBox++ % 120 == 0) {
            SPDLOG_INFO("INTDBG room bbox {:.0f},{:.0f},{:.0f} .. {:.0f},{:.0f},{:.0f} eye {:.0f},{:.0f},{:.0f}", mInteriorBBox[0], mInteriorBBox[1], mInteriorBBox[2], mInteriorBBox[3], mInteriorBBox[4], mInteriorBBox[5], params.giCameraEye[0], params.giCameraEye[1], params.giCameraEye[2]);
        }
    }
    if (mInteriorTris.empty()) {
        return;
    }
    // Merge the triangles of one texture that lie in one plane and touch (a window is usually two triangles, sometimes
    // a frame of several).
    std::vector<InteriorCluster> windows, lamps;
    const size_t triCount = mInteriorTris.size() / 11;
    for (size_t t = 0; t < triCount; t++) {
        const float* f = &mInteriorTris[t * 11];
        const uint8_t idx = (uint8_t)f[9];
        const InteriorTexture& tex = kInteriorTextures[idx - 1];
        const float e1[3] = { f[3] - f[0], f[4] - f[1], f[5] - f[2] }, e2[3] = { f[6] - f[0], f[7] - f[1], f[8] - f[2] };
        float n[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
        const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len < 1.0f) {
            continue;
        }
        for (float& c : n) {
            c /= len;
        }
        float mn[3], mx[3];
        for (int a = 0; a < 3; a++) {
            mn[a] = std::min({ f[a], f[3 + a], f[6 + a] });
            mx[a] = std::max({ f[a], f[3 + a], f[6 + a] });
        }
        auto& list = tex.kind == INTERIOR_WINDOW ? windows : lamps;
        InteriorCluster* hit = nullptr;
        for (auto& c : list) {
            if (c.idx != idx) {
                continue;
            }
            const bool planar = tex.kind != INTERIOR_WINDOW || std::fabs(n[0] * c.n[0] + n[1] * c.n[1] + n[2] * c.n[2]) > 0.9f;
            bool touches = true;
            for (int a = 0; a < 3; a++) {
                touches &= mn[a] <= c.mx[a] + 8.0f && mx[a] >= c.mn[a] - 8.0f;
            }
            if (planar && touches) {
                hit = &c;
                break;
            }
        }
        if (!hit) {
            list.push_back({ idx, { n[0], n[1], n[2] }, { mn[0], mn[1], mn[2] }, { mx[0], mx[1], mx[2] },
                             f[10] != 0.0f, {} });
            hit = &list.back();
        }
        for (int a = 0; a < 3; a++) {
            hit->mn[a] = std::min(hit->mn[a], mn[a]);
            hit->mx[a] = std::max(hit->mx[a], mx[a]);
        }
        if (hit->verts.size() < 96) {
            hit->verts.insert(hit->verts.end(), f, f + 9);
        }
    }

    if (mInteriorDebug) {
        static uint32_t sLog = 0;
        if (sLog++ % 120 == 0) {
            SPDLOG_INFO("INTDBG submit: {} triangles, {} window clusters, {} lamp clusters", triCount, windows.size(), lamps.size());
        }
    }
    struct Opening {
        float data[NEWDAWN_INTERIOR_WINDOW_FLOATS];
        float area;
    };
    std::vector<Opening> openings;
    for (const auto& c : windows) {
        const InteriorTexture& tex = kInteriorTextures[c.idx - 1];
        float n[3] = { c.n[0], c.n[1], c.n[2] };
        const float centre[3] = { (c.mn[0] + c.mx[0]) * 0.5f, (c.mn[1] + c.mx[1]) * 0.5f, (c.mn[2] + c.mx[2]) * 0.5f };
        // Front face looks into the room. Double-sided quads face the camera's side instead.
        if (!c.cullBack) {
            const float* eye = params.giCameraEye;
            if ((eye[0] - centre[0]) * n[0] + (eye[1] - centre[1]) * n[1] + (eye[2] - centre[2]) * n[2] < 0.0f) {
                for (float& x : n) {
                    x = -x;
                }
            }
        }
        // Horizontal axis along the wall; skylights and other near-horizontal openings use the world X axis.
        float u[3] = { n[2], 0.0f, -n[0] };
        float ul = std::sqrt(u[0] * u[0] + u[2] * u[2]);
        if (std::fabs(n[1]) > 0.7f || ul < 1e-3f) {
            const float d = n[0];
            u[0] = 1.0f - n[0] * d, u[1] = -n[1] * d, u[2] = -n[2] * d;
            ul = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        }
        for (float& x : u) {
            x /= ul;
        }
        const float vv[3] = { n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0] };
        float u0 = 1e9f, u1 = -1e9f, v0 = 1e9f, v1 = -1e9f;
        for (size_t i = 0; i + 2 < c.verts.size(); i += 3) {
            const float d[3] = { c.verts[i] - centre[0], c.verts[i + 1] - centre[1], c.verts[i + 2] - centre[2] };
            const float pu = d[0] * u[0] + d[1] * u[1] + d[2] * u[2], pv = d[0] * vv[0] + d[1] * vv[1] + d[2] * vv[2];
            u0 = std::min(u0, pu), u1 = std::max(u1, pu), v0 = std::min(v0, pv), v1 = std::max(v1, pv);
        }
        const float hu = (u1 - u0) * 0.5f * (1.0f - 2.0f * tex.a), hv = (v1 - v0) * 0.5f * (1.0f - 2.0f * tex.b);
        if (mInteriorDebug) {
            SPDLOG_INFO("INTDBG cluster {} bbox {:.0f},{:.0f},{:.0f}..{:.0f},{:.0f},{:.0f} half {:.0f}x{:.0f}", tex.name, c.mn[0], c.mn[1], c.mn[2], c.mx[0], c.mx[1], c.mx[2], hu, hv);
        }
        if (hu < 7.0f || hv < 7.0f) {
            continue;
        }
        Opening o = {};
        const float cu = (u0 + u1) * 0.5f, cv = (v0 + v1) * 0.5f;
        for (int a = 0; a < 3; a++) {
            o.data[a] = centre[a] + u[a] * cu + vv[a] * cv;
            o.data[3 + a] = n[a];
            o.data[6 + a] = u[a];
        }
        o.data[9] = hu, o.data[10] = hv, o.data[11] = tex.c, o.data[12] = tex.d, o.data[13] = (float)tex.kind;
        o.area = hu * hv;
        o.data[14] = o.area;
        if (mInteriorDebug) {
            SPDLOG_INFO("INTDBG opening tex {} centre {:.0f},{:.0f},{:.0f} n {:.2f},{:.2f},{:.2f} half {:.0f}x{:.0f} cullBack {}",
                        kInteriorTextures[c.idx - 1].name, o.data[0], o.data[1], o.data[2], n[0], n[1], n[2], hu, hv,
                        c.cullBack);
        }
        openings.push_back(o);
    }
    std::sort(openings.begin(), openings.end(), [](const Opening& a, const Opening& b) {
        return a.area != b.area ? a.area > b.area : a.data[0] < b.data[0];
    });
    for (size_t i = 0; i < openings.size() && params.interiorWindowCount < NEWDAWN_INTERIOR_WINDOWS; i++) {
        memcpy(params.interiorWindows[params.interiorWindowCount], openings[i].data, sizeof(openings[i].data));
        params.interiorWindowCount++;
    }

    // Emissive lights: nearest-first, one per cluster, only for textures that ask for a light.
    struct Lamp {
        float data[8];
        float dist;
    };
    std::vector<Lamp> lights;
    // Bit 0 of params.flags: an emissive texture was drawn this frame (the renderer skips the emission pass otherwise).
    if (!lamps.empty()) {
        params.flags |= 1u;
    }
    for (const auto& c : lamps) {
        const InteriorTexture& tex = kInteriorTextures[c.idx - 1];
        if (tex.d <= 0.0f) {
            continue;
        }
        Lamp l = {};
        const float ext = std::max({ c.mx[0] - c.mn[0], c.mx[1] - c.mn[1], c.mx[2] - c.mn[2] });
        for (int a = 0; a < 3; a++) {
            l.data[a] = (c.mn[a] + c.mx[a]) * 0.5f + c.n[a] * 8.0f;
        }
        l.data[3] = std::clamp(ext * 3.0f, 70.0f, 260.0f);
        l.data[4] = tex.a, l.data[5] = tex.b, l.data[6] = tex.c;
        l.data[7] = tex.kind == INTERIOR_GLASS ? -tex.d : tex.d; // negative: lit only at night
        const float dx = l.data[0] - params.giCameraEye[0], dy = l.data[1] - params.giCameraEye[1],
                    dz = l.data[2] - params.giCameraEye[2];
        l.dist = dx * dx + dy * dy + dz * dz;
        // One light per fixture: skip clusters close to a nearer one already taken.
        bool dup = false;
        for (const auto& o : lights) {
            const float ex = o.data[0] - l.data[0], ey = o.data[1] - l.data[1], ez = o.data[2] - l.data[2];
            dup |= ex * ex + ey * ey + ez * ez < 50.0f * 50.0f;
        }
        if (!dup) {
            lights.push_back(l);
        }
    }
    std::sort(lights.begin(), lights.end(), [](const Lamp& a, const Lamp& b) { return a.dist < b.dist; });
    for (size_t i = 0; i < lights.size() && i < NEWDAWN_EMISSIVE_LIGHTS; i++) {
        memcpy(params.emissiveLights[i], lights[i].data, sizeof(lights[i].data));
        params.emissiveLightCount++;
    }
}

void Interpreter::SubmitGrass() {
    if (mGrassDebug) {
        static int frame = 0;
        if (++frame % 600 == 1) {
            const size_t n = mGrassTris.size() / NEWDAWN_GRASS_TRI_FLOATS;
            std::vector<std::pair<float, size_t>> big;
            for (size_t i = 0; i < n; i++) {
                const float* t = &mGrassTris[i * NEWDAWN_GRASS_TRI_FLOATS];
                big.emplace_back(std::fabs((t[3] - t[0]) * (t[8] - t[2]) - (t[5] - t[2]) * (t[6] - t[0])), i);
            }
            std::sort(big.rbegin(), big.rend());
            SPDLOG_INFO("GRASSDBG {} triangles recorded", n);
            for (size_t k = 0; k < std::min<size_t>(12, big.size()); k++) {
                const float* t = &mGrassTris[big[k].second * NEWDAWN_GRASS_TRI_FLOATS];
                SPDLOG_INFO("GRASSDBG tri area {:.0f} centre {:.0f},{:.0f},{:.0f}", big[k].first * 0.5f,
                            (t[0] + t[3] + t[6]) / 3, (t[1] + t[4] + t[7]) / 3, (t[2] + t[5] + t[8]) / 3);
            }
        }
    }
    mRapi->SetGrassTriangles(mGrassTris.empty() ? nullptr : mGrassTris.data(),
                             (uint32_t)(mGrassTris.size() / NEWDAWN_GRASS_TRI_FLOATS));
}

// Mean colour (gamma space, alpha-weighted) and mean alpha of the texture loaded for a tile, from its CPU data. Cached
// per data pointer and palette; about a thousand texels are sampled. Unknown formats give neutral grey.
void Interpreter::GiTextureAverage(int tile, float rgba[4]) {
    const auto& lt = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index];
    const Fast::Texture* tex = lt.raw_tex_metadata.resource.get();
    rgba[0] = rgba[1] = rgba[2] = 0.5f;
    rgba[3] = 1.0f;
    if (tex == nullptr || tex->ImageData == nullptr || tex->Width == 0 || tex->Height == 0) {
        return;
    }
    const bool paletted = tex->Type == TextureType::Palette4bpp || tex->Type == TextureType::Palette8bpp;
    const uint8_t* palette = paletted ? mRdp->palettes[0] : nullptr;
    const uint64_t paletteKey = paletted ? (uint64_t)(uintptr_t)mRdp->palette_dram_addr[0] * 0x9E3779B97F4A7C15ull : 0;
    const uint64_t key = (uint64_t)(uintptr_t)tex->ImageData ^ paletteKey ^ mRdp->texture_tile[tile].palette;
    if (auto found = mGiTexAverage.find(key); found != mGiTexAverage.end()) {
        std::copy_n(found->second.data(), 4, rgba);
        return;
    }
    const uint8_t* data = tex->ImageData;
    const uint32_t w = tex->Width, h = tex->Height, pixels = w * h;
    uint32_t bits = 0; // per pixel
    switch (tex->Type) {
        case TextureType::RGBA32bpp: bits = 32; break;
        case TextureType::RGBA16bpp: case TextureType::GrayscaleAlpha16bpp: bits = 16; break;
        case TextureType::Palette8bpp: case TextureType::Grayscale8bpp: case TextureType::GrayscaleAlpha8bpp: bits = 8; break;
        case TextureType::Palette4bpp: case TextureType::Grayscale4bpp: case TextureType::GrayscaleAlpha4bpp: bits = 4; break;
        default: break;
    }
    if (bits == 0 || (uint64_t)pixels * bits / 8 > tex->ImageDataSize || (paletted && palette == nullptr)) {
        return; // not cached: a palette may be loaded later
    }
    const auto rgba16 = [](uint16_t c, float out[4]) {
        out[0] = (float)(c >> 11) / 31.0f;
        out[1] = (float)((c >> 6) & 31) / 31.0f;
        out[2] = (float)((c >> 1) & 31) / 31.0f;
        out[3] = (float)(c & 1);
    };
    const uint32_t step = std::max(1u, pixels / 1024u);
    double sum[4] = {};
    uint32_t count = 0;
    for (uint32_t i = step / 2; i < pixels; i += step, ++count) {
        float c[4] = { 0.5f, 0.5f, 0.5f, 1.0f };
        const uint32_t nibble = (data[i / 2] >> ((i & 1) ? 0 : 4)) & 15;
        switch (tex->Type) {
            case TextureType::RGBA32bpp:
                for (int k = 0; k < 4; ++k) c[k] = data[i * 4 + k] / 255.0f;
                break;
            case TextureType::RGBA16bpp:
                rgba16((uint16_t)((data[i * 2] << 8) | data[i * 2 + 1]), c);
                break;
            case TextureType::Palette8bpp: {
                // Same lookup as ImportTextureCi8 (two 128-entry halves, big endian RGBA5551).
                const uint8_t* half = mRdp->palettes[data[i] / 128];
                if (half != nullptr) rgba16((uint16_t)((half[(data[i] % 128) * 2] << 8) | half[(data[i] % 128) * 2 + 1]), c);
                break;
            }
            case TextureType::Palette4bpp: {
                const uint8_t* pal = mRdp->palettes[mRdp->texture_tile[tile].palette / 8];
                const uint32_t idx = (mRdp->texture_tile[tile].palette % 8) * 16 + nibble;
                if (pal != nullptr) rgba16((uint16_t)((pal[idx * 2] << 8) | pal[idx * 2 + 1]), c);
                break;
            }
            case TextureType::Grayscale8bpp: c[0] = c[1] = c[2] = data[i] / 255.0f; break;
            case TextureType::Grayscale4bpp: c[0] = c[1] = c[2] = nibble / 15.0f; break;
            case TextureType::GrayscaleAlpha16bpp: c[0] = c[1] = c[2] = data[i * 2] / 255.0f; c[3] = data[i * 2 + 1] / 255.0f; break;
            case TextureType::GrayscaleAlpha8bpp: c[0] = c[1] = c[2] = (data[i] >> 4) / 15.0f; c[3] = (data[i] & 15) / 15.0f; break;
            case TextureType::GrayscaleAlpha4bpp: c[0] = c[1] = c[2] = (nibble >> 1) / 7.0f; c[3] = (float)(nibble & 1); break;
            default: break;
        }
        for (int k = 0; k < 3; ++k) sum[k] += c[k] * c[3];
        sum[3] += c[3];
    }
    if (count > 0 && sum[3] > 1e-3) {
        for (int k = 0; k < 3; ++k) rgba[k] = (float)(sum[k] / sum[3]);
        rgba[3] = (float)(sum[3] / count);
    }
    if (mGiTexAverage.size() > 8192) {
        mGiTexAverage.clear();
    }
    mGiTexAverage[key] = { rgba[0], rgba[1], rgba[2], rgba[3] };
}

void Interpreter::RecordGiTriangle(const LoadedVertex* v1, const LoadedVertex* v2, const LoadedVertex* v3) {
    constexpr size_t kMaxGiTriangles = 400000;
    if (mGiTris.size() >= kMaxGiTriangles * NEWDAWN_GI_TRI_FLOATS) {
        return;
    }
    const float* a = v1->view_pos;
    const float* b = v2->view_pos;
    const float* c = v3->view_pos;
    const float e1[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, e2[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
    float n[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
    const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len < 1e-4f) {
        return;
    }
    // Front faces wind counter-clockwise on screen; mirrored culling and front-face culling turn the visible side over.
    const bool flip = ((mRsp->extra_geometry_mode & G_EX_INVERT_CULLING) != 0) !=
                      ((mRsp->geometry_mode & get_attr(CULL_BOTH)) == get_attr(CULL_FRONT));
    for (float& x : n) {
        x /= flip ? -len : len;
    }
    // Albedo: the colour combiner evaluated once on the average texture colours, with the light term set to white
    // (lit geometry) or to the baked vertex colour (unlit geometry). The display-space value is used as the
    // reflectance as it is: the game's textures are painted dark (decoded to linear, wood and stone come out at
    // 0.02-0.1 and would bounce next to nothing), and their display values sit where real albedos do.
    const bool twoCycle = (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE)) == G_CYC_2CYCLE;
    float tex[2][4];
    GiTextureAverage(mRdp->first_tile_index, tex[0]);
    if (twoCycle) {
        GiTextureAverage(mRdp->first_tile_index + 1 < 8 ? mRdp->first_tile_index + 1 : mRdp->first_tile_index, tex[1]);
    } else {
        std::copy_n(tex[0], 4, tex[1]);
    }
    float shade[3] = { 1.0f, 1.0f, 1.0f };
    if ((mRsp->geometry_mode & G_LIGHTING) == 0) {
        shade[0] = (v1->color.r + v2->color.r + v3->color.r) / 765.0f;
        shade[1] = (v1->color.g + v2->color.g + v3->color.g) / 765.0f;
        shade[2] = (v1->color.b + v2->color.b + v3->color.b) / 765.0f;
    }
    const float prim[4] = { mRdp->prim_color.r / 255.0f, mRdp->prim_color.g / 255.0f, mRdp->prim_color.b / 255.0f,
                            mRdp->prim_color.a / 255.0f };
    const float env[4] = { mRdp->env_color.r / 255.0f, mRdp->env_color.g / 255.0f, mRdp->env_color.b / 255.0f,
                           mRdp->env_color.a / 255.0f };
    float combined[3] = {};
    for (int cycle = 0; cycle < (twoCycle ? 2 : 1); ++cycle) {
        const uint32_t m = (uint32_t)(mRdp->combine_mode >> (cycle * 28));
        const uint32_t sel[4] = { m & 0xf, (m >> 4) & 0xf, (m >> 8) & 0x1f, (m >> 13) & 7 };
        const float* t0 = tex[cycle]; // the second cycle sees the two texels swapped
        const float* t1 = tex[1 - cycle];
        float next[3];
        for (int ch = 0; ch < 3; ++ch) {
            float v[4];
            for (int k = 0; k < 4; ++k) {
                const uint32_t x = sel[k];
                switch (x) {
                    case G_CCMUX_COMBINED: v[k] = combined[ch]; break;
                    case G_CCMUX_TEXEL0: v[k] = t0[ch]; break;
                    case G_CCMUX_TEXEL1: v[k] = t1[ch]; break;
                    case G_CCMUX_PRIMITIVE: v[k] = prim[ch]; break;
                    case G_CCMUX_SHADE: v[k] = shade[ch]; break;
                    case G_CCMUX_ENVIRONMENT: v[k] = env[ch]; break;
                    default:
                        if (k == 0) { // A: 6 = 1, 7 = noise
                            v[k] = x == 6 ? 1.0f : (x == 7 ? 0.5f : 0.0f);
                        } else if (k == 2) { // C: the alpha inputs; LOD fraction 0 is the close-up texture
                            v[k] = x == G_CCMUX_COMBINED_ALPHA ? 1.0f : x == G_CCMUX_TEXEL0_ALPHA ? t0[3]
                                 : x == G_CCMUX_TEXEL1_ALPHA ? t1[3] : x == G_CCMUX_PRIMITIVE_ALPHA ? prim[3]
                                 : x == G_CCMUX_SHADE_ALPHA ? 1.0f : x == G_CCMUX_ENV_ALPHA ? env[3]
                                 : x == G_CCMUX_PRIM_LOD_FRAC ? mRdp->prim_lod_fraction / 255.0f : 0.0f;
                        } else { // B: key/convert constants; D: 6 = 1
                            v[k] = (k == 3 && x == 6) ? 1.0f : 0.0f;
                        }
                        break;
                }
            }
            next[ch] = (v[0] - v[1]) * v[2] + v[3];
        }
        std::copy_n(next, 3, combined);
    }
    float albedo[3];
    for (int ch = 0; ch < 3; ++ch) {
        albedo[ch] = std::clamp(combined[ch], 0.0f, 0.9f);
    }
    // Unlit room geometry is the game's baked lighting: the renderer treats it as a light source of its own colour.
    const float baked = (mRsp->geometry_mode & G_LIGHTING) == 0 ? 1.0f : 0.0f;
    const LoadedVertex* verts[3] = { v1, v2, v3 };
    for (const LoadedVertex* v : verts) {
        mGiTris.insert(mGiTris.end(), v->view_pos, v->view_pos + 3);
        mGiTris.insert(mGiTris.end(), albedo, albedo + 3);
        mGiTris.insert(mGiTris.end(), n, n + 3);
        mGiTris.push_back(baked);
    }
}

void Interpreter::RecordShadowTriangle(const LoadedVertex* v1, const LoadedVertex* v2, const LoadedVertex* v3) {
    constexpr size_t kMaxShadowVertices = 2 * 1024 * 1024;
    if (mShadowVertices.size() + 9 > kMaxShadowVertices * 3) {
        if (!mShadowOverflowLogged) {
            mShadowOverflowLogged = true;
            SPDLOG_WARN("NewDawn shadows: caster buffer overflow, truncating this frame");
        }
        return;
    }
    const LoadedVertex* verts[3] = { v1, v2, v3 };
    for (const LoadedVertex* v : verts) {
        mShadowVertices.push_back(v->view_pos[0]);
        mShadowVertices.push_back(v->view_pos[1]);
        mShadowVertices.push_back(v->view_pos[2]);
    }
}

// Pick the per-line byte width for texture decode. Prefer the DRAM stride from
// loaded_texture when it looks like real per-line info (differs from total size).
// Fall back to the TMEM tile stride when loaded sizes match total (LoadBlock with
// width=1, where line_size == full_image_line_size == size).
static uint32_t GetEffectiveLineSize(uint32_t lineSizeBytes, uint32_t fullImageLineSizeBytes, uint32_t sizeBytes,
                                     uint32_t tileLineSizeBytes) {
    if ((lineSizeBytes != sizeBytes || fullImageLineSizeBytes != sizeBytes) && lineSizeBytes > 0) {
        return lineSizeBytes;
    }
    return tileLineSizeBytes;
}

static uint32_t GetTileSizeFromCoordinates(float low, float high) {
    // An unset tile (high <= low) defines no region; return 0 so callers skip the tile-region clamp
    // instead of collapsing the texture to the phantom 1-texel size the +4 formula would yield.
    if (high <= low) {
        return 0;
    }
    return static_cast<uint32_t>(lroundf((high - low + 4.0f) / 4.0f));
}

// Loaded size is HD-scaled but the tile region is raw N64 texels; normalize before comparing.
static bool IsPyramidLike(uint32_t width, uint32_t height, uint32_t tileW, uint32_t tileH, float hByteScale,
                          float vPixelScale) {
    const float hs = hByteScale > 0.0f ? hByteScale : 1.0f;
    const float vs = vPixelScale > 0.0f ? vPixelScale : 1.0f;
    const uint32_t loadedPixels = static_cast<uint32_t>(width / hs) * static_cast<uint32_t>(height / vs);
    const uint32_t renderedPixels = tileW * tileH;
    return renderedPixels > 0 && loadedPixels > renderedPixels && loadedPixels * 8 < renderedPixels * 13; // < 1.625x
}

void Interpreter::ImportTextureRgba16(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureRgba16: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(line_size_bytes, fullImageLineSizeBytes, sizeBytes,
                                               mRdp->texture_tile[tile].line_size_bytes);
    uint32_t width = widthBytes / 2;
    uint32_t height = widthBytes > 0 ? sizeBytes / widthBytes : 0;

    // Clamp to the rendered region only when the loaded buffer is ~1.33x of it (mipmap
    // pyramid signature). Window-scrolling tiles have loaded ≈ rendered or loaded >> rendered;
    // skip both. CLAMP wrap mode always opts in.
    uint32_t tile_w = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].uls, mRdp->texture_tile[tile].lrs);
    uint32_t tile_h = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].ult, mRdp->texture_tile[tile].lrt);
    bool pyramidLike = IsPyramidLike(width, height, tile_w, tile_h, metadata->h_byte_scale, metadata->v_pixel_scale);
    bool clampS = (mRdp->texture_tile[tile].cms & G_TX_CLAMP) != 0;
    bool clampT = (mRdp->texture_tile[tile].cmt & G_TX_CLAMP) != 0;
    // A masked axis wraps every 2^mask texels, so trim an over-loaded texture back to that.
    // Skip a mask smaller than the tile region though - that's stale tile state, not a real load.
    uint32_t maskW = mRdp->texture_tile[tile].masks;
    uint32_t maskH = mRdp->texture_tile[tile].maskt;
    if (maskW != 0 && (1u << maskW) >= tile_w && (1u << maskW) < width) {
        width = 1u << maskW;
    }
    if (maskH != 0 && (1u << maskH) >= tile_h && (1u << maskH) < height) {
        height = 1u << maskH;
    }
    if ((pyramidLike || clampS) && tile_w > 0 && tile_w < width) {
        width = tile_w;
    }
    if ((pyramidLike || clampT) && tile_h > 0 && tile_h < height) {
        height = tile_h;
    }

    // A single line of pixels should not equal the entire image (height == 1 non-withstanding)
    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = width * 2;
    }

    uint32_t i = 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t clrIdx = (y * (fullImageLineSizeBytes / 2)) + (x);

            uint16_t col16 = (addr[2 * clrIdx] << 8) | addr[2 * clrIdx + 1];
            uint8_t a = col16 & 1;
            uint8_t r = col16 >> 11;
            uint8_t g = (col16 >> 6) & 0x1f;
            uint8_t b = (col16 >> 1) & 0x1f;
            mTexUploadBuffer[4 * i + 0] = SCALE_5_8(r);
            mTexUploadBuffer[4 * i + 1] = SCALE_5_8(g);
            mTexUploadBuffer[4 * i + 2] = SCALE_5_8(b);
            mTexUploadBuffer[4 * i + 3] = a ? 255 : 0;

            i++;
        }
    }

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
    mRapi->BuildMipmaps();
}

void Interpreter::ImportTextureRgba32(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureRgba32: null texture address for tile {}", tile);
        return;
    }

    uint32_t size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t full_image_line_size_bytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(line_size_bytes, full_image_line_size_bytes, size_bytes,
                                               mRdp->texture_tile[tile].line_size_bytes * 2);
    uint32_t width = widthBytes / 4;
    uint32_t height = widthBytes > 0 ? size_bytes / widthBytes : 0;

    // Clamp to the rendered region only when the loaded buffer is ~1.33x of it (mipmap
    // pyramid signature). Window-scrolling tiles have loaded ≈ rendered or loaded >> rendered;
    // skip both. CLAMP wrap mode always opts in.
    uint32_t tile_w = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].uls, mRdp->texture_tile[tile].lrs);
    uint32_t tile_h = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].ult, mRdp->texture_tile[tile].lrt);
    bool pyramidLike = IsPyramidLike(width, height, tile_w, tile_h, metadata->h_byte_scale, metadata->v_pixel_scale);
    bool clampS = (mRdp->texture_tile[tile].cms & G_TX_CLAMP) != 0;
    bool clampT = (mRdp->texture_tile[tile].cmt & G_TX_CLAMP) != 0;
    // A masked axis wraps every 2^mask texels, so trim an over-loaded texture back to that.
    // Skip a mask smaller than the tile region though - that's stale tile state, not a real load.
    uint32_t maskW = mRdp->texture_tile[tile].masks;
    uint32_t maskH = mRdp->texture_tile[tile].maskt;
    if (maskW != 0 && (1u << maskW) >= tile_w && (1u << maskW) < width) {
        width = 1u << maskW;
    }
    if (maskH != 0 && (1u << maskH) >= tile_h && (1u << maskH) < height) {
        height = 1u << maskH;
    }
    if ((pyramidLike || clampS) && tile_w > 0 && tile_w < width) {
        width = tile_w;
    }
    if ((pyramidLike || clampT) && tile_h > 0 && tile_h < height) {
        height = tile_h;
    }

    if (full_image_line_size_bytes == size_bytes) {
        full_image_line_size_bytes = width * 4;
    }

    // Copy pixel by pixel, respecting full image stride (handles sub-tile loads)
    uint32_t fullImageStridePixels = full_image_line_size_bytes / 4;
    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t srcIdx = y * fullImageStridePixels + x;
            mTexUploadBuffer[4 * i + 0] = addr[4 * srcIdx + 0];
            mTexUploadBuffer[4 * i + 1] = addr[4 * srcIdx + 1];
            mTexUploadBuffer[4 * i + 2] = addr[4 * srcIdx + 2];
            mTexUploadBuffer[4 * i + 3] = addr[4 * srcIdx + 3];
            i++;
        }
    }
    mRapi->UploadTexture(mTexUploadBuffer, width, height);
    mRapi->BuildMipmaps();
}

void Interpreter::ImportTextureIA4(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureIA4: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                               mRdp->texture_tile[tile].line_size_bytes);
    uint32_t width = widthBytes * 2;
    uint32_t height = widthBytes > 0 ? sizeBytes / widthBytes : 0;

    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = widthBytes;
    }

    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t srcPixelIdx = y * (fullImageLineSizeBytes * 2) + x;
            uint8_t byte = addr[srcPixelIdx / 2];
            uint8_t part = (byte >> (4 - (srcPixelIdx % 2) * 4)) & 0xf;
            uint8_t intensity = part >> 1;
            uint8_t alpha = part & 1;
            mTexUploadBuffer[4 * i + 0] = SCALE_3_8(intensity);
            mTexUploadBuffer[4 * i + 1] = SCALE_3_8(intensity);
            mTexUploadBuffer[4 * i + 2] = SCALE_3_8(intensity);
            mTexUploadBuffer[4 * i + 3] = alpha ? 255 : 0;
            i++;
        }
    }

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
    mRapi->BuildMipmaps();
}

void Interpreter::ImportTextureIA8(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureIA8: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t width = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                          mRdp->texture_tile[tile].line_size_bytes);
    uint32_t height = width > 0 ? sizeBytes / width : 0;

    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = width;
    }

    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t srcIdx = y * fullImageLineSizeBytes + x;
            uint8_t intensity = addr[srcIdx] >> 4;
            uint8_t alpha = addr[srcIdx] & 0xf;
            mTexUploadBuffer[4 * i + 0] = SCALE_4_8(intensity);
            mTexUploadBuffer[4 * i + 1] = SCALE_4_8(intensity);
            mTexUploadBuffer[4 * i + 2] = SCALE_4_8(intensity);
            mTexUploadBuffer[4 * i + 3] = SCALE_4_8(alpha);
            i++;
        }
    }

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
    mRapi->BuildMipmaps();
}

void Interpreter::ImportTextureIA16(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureIA16: null texture address for tile {}", tile);
        return;
    }

    uint32_t size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t full_image_line_size_bytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(line_size_bytes, full_image_line_size_bytes, size_bytes,
                                               mRdp->texture_tile[tile].line_size_bytes);
    uint32_t width = widthBytes / 2;
    uint32_t height = widthBytes > 0 ? size_bytes / widthBytes : 0;

    // A single line of pixels should not equal the entire image (height == 1 non-withstanding)
    if (full_image_line_size_bytes == size_bytes) {
        full_image_line_size_bytes = width * 2;
    }

    uint32_t i = 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t clrIdx = (y * (full_image_line_size_bytes / 2)) + (x);

            uint8_t intensity = addr[2 * clrIdx];
            uint8_t alpha = addr[2 * clrIdx + 1];
            uint8_t r = intensity;
            uint8_t g = intensity;
            uint8_t b = intensity;
            mTexUploadBuffer[4 * i + 0] = r;
            mTexUploadBuffer[4 * i + 1] = g;
            mTexUploadBuffer[4 * i + 2] = b;
            mTexUploadBuffer[4 * i + 3] = alpha;

            i++;
        }
    }

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
    mRapi->BuildMipmaps();
}

void Interpreter::ImportTextureI4(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureI4: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                               mRdp->texture_tile[tile].line_size_bytes);
    uint32_t width = widthBytes * 2;
    uint32_t height = widthBytes > 0 ? sizeBytes / widthBytes : 0;

    // A single line of pixels should not equal the entire image (height == 1 non-withstanding)
    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = width / 2;
    }

    uint32_t i = 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t clrIdx = (y * (fullImageLineSizeBytes * 2)) + (x);

            uint8_t byte = addr[clrIdx / 2];
            uint8_t part = (byte >> (4 - (clrIdx % 2) * 4)) & 0xf;
            uint8_t intensity = part;
            uint8_t r = intensity;
            uint8_t g = intensity;
            uint8_t b = intensity;
            uint8_t a = intensity;
            mTexUploadBuffer[4 * i + 0] = SCALE_4_8(r);
            mTexUploadBuffer[4 * i + 1] = SCALE_4_8(g);
            mTexUploadBuffer[4 * i + 2] = SCALE_4_8(b);
            mTexUploadBuffer[4 * i + 3] = SCALE_4_8(a);

            i++;
        }
    }

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
    mRapi->BuildMipmaps();
}

void Interpreter::ImportTextureI8(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureI8: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t width = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                          mRdp->texture_tile[tile].line_size_bytes);
    uint32_t height = width > 0 ? sizeBytes / width : 0;

    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = width;
    }

    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint8_t intensity = addr[y * fullImageLineSizeBytes + x];
            mTexUploadBuffer[4 * i + 0] = intensity;
            mTexUploadBuffer[4 * i + 1] = intensity;
            mTexUploadBuffer[4 * i + 2] = intensity;
            mTexUploadBuffer[4 * i + 3] = intensity;
            i++;
        }
    }

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
    mRapi->BuildMipmaps();
}

void Interpreter::ImportTextureCi4(int tile, bool importReplacement) {
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureCi4: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;
    uint32_t palIdx = mRdp->texture_tile[tile].palette; // 0-15

    const uint8_t* palette;

    if (mRdp->palettes[palIdx / 8] == nullptr) {
        SPDLOG_WARN("CI4: null palette slot {} for palIdx={}", palIdx / 8, palIdx);
        return;
    }
    palette = mRdp->palettes[palIdx / 8] + (palIdx % 8) * 16 * 2;

    uint32_t baseLineSizeBytes = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                                      mRdp->texture_tile[tile].line_size_bytes);
    uint32_t resultLineSizeBytes = baseLineSizeBytes;

    if (metadata->h_byte_scale != 1) {
        resultLineSizeBytes *= metadata->h_byte_scale;
    }

    // CI4: 2 pixels per byte
    uint32_t width = resultLineSizeBytes * 2;
    uint32_t height = resultLineSizeBytes > 0 ? sizeBytes / resultLineSizeBytes : 0;

    // Clamp to the rendered region only when the loaded buffer is ~1.33x of it (mipmap
    // pyramid signature). Window-scrolling tiles have loaded ≈ rendered or loaded >> rendered;
    // skip both. CLAMP wrap mode always opts in.
    uint32_t tile_w = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].uls, mRdp->texture_tile[tile].lrs);
    uint32_t tile_h = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].ult, mRdp->texture_tile[tile].lrt);
    bool pyramidLike = IsPyramidLike(width, height, tile_w, tile_h, metadata->h_byte_scale, metadata->v_pixel_scale);
    bool clampS = (mRdp->texture_tile[tile].cms & G_TX_CLAMP) != 0;
    bool clampT = (mRdp->texture_tile[tile].cmt & G_TX_CLAMP) != 0;
    // A masked axis wraps every 2^mask texels, so trim an over-loaded texture back to that.
    // Skip a mask smaller than the tile region though - that's stale tile state, not a real load.
    uint32_t maskW = mRdp->texture_tile[tile].masks;
    uint32_t maskH = mRdp->texture_tile[tile].maskt;
    if (maskW != 0 && (1u << maskW) >= tile_w && (1u << maskW) < width) {
        width = 1u << maskW;
    }
    if (maskH != 0 && (1u << maskH) >= tile_h && (1u << maskH) < height) {
        height = 1u << maskH;
    }
    if ((pyramidLike || clampS) && tile_w > 0 && tile_w < width) {
        width = tile_w;
    }
    if ((pyramidLike || clampT) && tile_h > 0 && tile_h < height) {
        height = tile_h;
    }

    if (fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = resultLineSizeBytes;
    }

    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t srcPixelIdx = y * (fullImageLineSizeBytes * 2) + x;
            uint8_t byte = addr[srcPixelIdx / 2];
            uint8_t idx = (byte >> (4 - (srcPixelIdx % 2) * 4)) & 0xf;
            uint16_t col16 = (palette[idx * 2] << 8) | palette[idx * 2 + 1]; // Big endian load
            uint8_t a = col16 & 1;
            uint8_t r = col16 >> 11;
            uint8_t g = (col16 >> 6) & 0x1f;
            uint8_t b = (col16 >> 1) & 0x1f;
            mTexUploadBuffer[4 * i + 0] = SCALE_5_8(r);
            mTexUploadBuffer[4 * i + 1] = SCALE_5_8(g);
            mTexUploadBuffer[4 * i + 2] = SCALE_5_8(b);
            mTexUploadBuffer[4 * i + 3] = a ? 255 : 0;
            i++;
        }
    }

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
    mRapi->BuildMipmaps();
}

void Interpreter::ImportTextureCi8(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureCi8: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    if (mRdp->palettes[0] == nullptr || mRdp->palettes[1] == nullptr) {
        SPDLOG_WARN("CI8: null palette (pal0={}, pal1={})", static_cast<const void*>(mRdp->palettes[0]),
                    static_cast<const void*>(mRdp->palettes[1]));
        return;
    }

    for (uint32_t i = 0, j = 0; i < sizeBytes; j += fullImageLineSizeBytes - lineSizeBytes) {
        for (uint32_t k = 0; k < lineSizeBytes; i++, k++, j++) {
            uint8_t idx = addr[j];
            uint16_t col16 = (mRdp->palettes[idx / 128][(idx % 128) * 2] << 8) |
                             mRdp->palettes[idx / 128][(idx % 128) * 2 + 1]; // Big endian load
            uint8_t a = col16 & 1;
            uint8_t r = col16 >> 11;
            uint8_t g = (col16 >> 6) & 0x1f;
            uint8_t b = (col16 >> 1) & 0x1f;
            mTexUploadBuffer[4 * i + 0] = SCALE_5_8(r);
            mTexUploadBuffer[4 * i + 1] = SCALE_5_8(g);
            mTexUploadBuffer[4 * i + 2] = SCALE_5_8(b);
            mTexUploadBuffer[4 * i + 3] = a ? 255 : 0;
        }
    }

    uint32_t baseLineSizeBytes = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                                      mRdp->texture_tile[tile].line_size_bytes);
    uint32_t resultLineSizeBytes = baseLineSizeBytes;
    if (metadata->h_byte_scale != 1) {
        resultLineSizeBytes *= metadata->h_byte_scale;
    }

    uint32_t width = resultLineSizeBytes;
    uint32_t height = resultLineSizeBytes > 0 ? sizeBytes / resultLineSizeBytes : 0;

    // Clamp to the rendered region only when the loaded buffer is ~1.33x of it (mipmap
    // pyramid signature). Window-scrolling tiles have loaded ≈ rendered or loaded >> rendered;
    // skip both. CLAMP wrap mode always opts in.
    uint32_t tile_w = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].uls, mRdp->texture_tile[tile].lrs);
    uint32_t tile_h = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].ult, mRdp->texture_tile[tile].lrt);
    bool pyramidLike = IsPyramidLike(width, height, tile_w, tile_h, metadata->h_byte_scale, metadata->v_pixel_scale);
    bool clampS = (mRdp->texture_tile[tile].cms & G_TX_CLAMP) != 0;
    bool clampT = (mRdp->texture_tile[tile].cmt & G_TX_CLAMP) != 0;
    // A masked axis wraps every 2^mask texels, so trim an over-loaded texture back to that.
    // Skip a mask smaller than the tile region though - that's stale tile state, not a real load.
    uint32_t maskW = mRdp->texture_tile[tile].masks;
    uint32_t maskH = mRdp->texture_tile[tile].maskt;
    if (maskW != 0 && (1u << maskW) >= tile_w && (1u << maskW) < width) {
        width = 1u << maskW;
    }
    if (maskH != 0 && (1u << maskH) >= tile_h && (1u << maskH) < height) {
        height = 1u << maskH;
    }
    if ((pyramidLike || clampS) && tile_w > 0 && tile_w < width) {
        width = tile_w;
    }
    if ((pyramidLike || clampT) && tile_h > 0 && tile_h < height) {
        height = tile_h;
    }

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
    mRapi->BuildMipmaps();
}

void Interpreter::ImportTextureImg(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureImg: null texture address for tile {}", tile);
        return;
    }

    uint16_t width = metadata->width;
    uint16_t height = metadata->height;
    mRapi->UploadTexture(addr, width, height);
    mRapi->BuildMipmaps();
}

void Interpreter::ImportTextureRaw(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureRaw: null texture address for tile {}", tile);
        return;
    }

    uint16_t width = metadata->width;
    uint16_t height = metadata->height;
    Fast::TextureType type = metadata->type;
    std::shared_ptr<Fast::Texture> resource = metadata->resource;

    // if texture type is CI4 or CI8 we need to apply tlut to it
    switch (type) {
        case Fast::TextureType::Palette4bpp:
            ImportTextureCi4(tile, importReplacement);
            return;
        case Fast::TextureType::Palette8bpp:
            ImportTextureCi8(tile, importReplacement);
            return;
        default:
            break;
    }

    uint32_t numLoadedBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t numOriginallyLoadedBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes;

    uint32_t resultOrigLineSize = mRdp->texture_tile[tile].line_size_bytes;
    switch (mRdp->texture_tile[tile].siz) {
        case G_IM_SIZ_32b:
            resultOrigLineSize *= 2;
            break;
    }
    uint32_t resultOrigHeight = numOriginallyLoadedBytes / resultOrigLineSize;
    uint32_t resultNewLineSize = resultOrigLineSize * metadata->h_byte_scale;
    uint32_t resultNewHeight = resultOrigHeight * metadata->v_pixel_scale;

    if (resultNewLineSize == 4 * width && resultNewHeight == height) {
        // Can use the texture directly since it has the correct dimensions
        mRapi->UploadTexture(addr, width, height);
        mRapi->BuildMipmaps();
        return;
    }

    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    // Get the resource's true image size
    uint32_t resourceImageSizeBytes = resource->ImageDataSize;
    uint32_t safeFullImageLineSizeBytes = fullImageLineSizeBytes;
    uint32_t safeLineSizeBytes = line_size_bytes;
    uint32_t safeLoadedBytes = numLoadedBytes;

    // Sometimes the texture load commands will specify a size larger than the authentic texture
    // Normally the OOB info is read as garbage, but will cause a crash on some platforms
    // Restrict the bytes to a safe amount
    if (numLoadedBytes > resourceImageSizeBytes) {
        safeLoadedBytes = resourceImageSizeBytes;
        safeLineSizeBytes = resourceImageSizeBytes;
        safeFullImageLineSizeBytes = resourceImageSizeBytes;
    }

    // Safely only copy the amount of bytes the resource can allow
    for (uint32_t i = 0, j = 0; i < safeLoadedBytes; i += safeLineSizeBytes, j += safeFullImageLineSizeBytes) {
        memcpy(mTexUploadBuffer + i, addr + j, safeLineSizeBytes);
    }

    // Set the remaining bytes to load as 0
    if (numLoadedBytes > resourceImageSizeBytes) {
        memset(mTexUploadBuffer + resourceImageSizeBytes, 0, numLoadedBytes - resourceImageSizeBytes);
    }

    // Describe the buffer by what was actually packed (the loaded HD stride)
    uint32_t uploadWidth = safeLineSizeBytes / 4;
    uint32_t uploadHeight = safeLineSizeBytes > 0 ? safeLoadedBytes / safeLineSizeBytes : 0;
    const bool singleLineLoad = uploadHeight <= 1 && resultNewLineSize != 0 && resultNewLineSize < safeLineSizeBytes;
    if (uploadWidth > (uint32_t)mRapi->GetMaxTextureSize() || singleLineLoad) {
        if (safeLoadedBytes == (uint64_t)width * height * 4) {
            // buffer holds the full image, use its real dimensions
            uploadWidth = width;
            uploadHeight = height;
        } else {
            // partial load, fall back to the tile-derived dimensions
            uploadWidth = resultNewLineSize / 4;
            uploadHeight = resultNewHeight;
        }
    }
    mRapi->UploadTexture(mTexUploadBuffer, uploadWidth, uploadHeight);
    mRapi->BuildMipmaps();
}

void Interpreter::ImportTexture(int i, int tile, bool importReplacement) {
    uint8_t fmt = mRdp->texture_tile[tile].fmt;
    uint8_t siz = mRdp->texture_tile[tile].siz;
    uint32_t texFlags = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].tex_flags;
    uint32_t tmemIdex = mRdp->texture_tile[tile].tmem_index;
    uint8_t paletteIndex = mRdp->texture_tile[tile].palette;
    uint32_t origSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes;

    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* origAddr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[tmemIdex].addr;

    // Check if this texture address is a registered GPU framebuffer mirror.
    // If so, bind the GPU FB directly — full resolution, no CPU readback needed.
    if (origAddr != nullptr && !importReplacement) {
        auto fbIt = mFbTextures.find((uintptr_t)origAddr);
        if (fbIt != mFbTextures.end()) {
            Flush();
            mRapi->SelectTextureFb(fbIt->second);
            mRdp->textures_changed[i] = false;
            return;
        }
    }

    if (origAddr == nullptr) {
        // Try the other TMEM slot -- some multi-tile setups only load one slot
        // and expect both tiles to reference it.
        uint32_t otherTmem = tmemIdex ^ 1;
        origAddr = mRdp->loaded_texture[otherTmem].addr;
        if (origAddr == nullptr) {
            SPDLOG_WARN("ImportTexture: null texture address for tile {} (both TMEM slots empty)", tile);
            return;
        }
        SPDLOG_WARN("ImportTexture: tile {} TMEM slot {} empty, falling back to slot {}", tile, tmemIdex, otherTmem);
        tmemIdex = otherTmem;
        origSizeBytes = mRdp->loaded_texture[otherTmem].orig_size_bytes;
        texFlags = mRdp->loaded_texture[otherTmem].tex_flags;
    }

    // Use palette_dram_addr (the original DRAM source) instead of palettes[]
    // (which always points to the staging buffer) so the same texture drawn
    // with different palettes gets distinct cache entries.
    TextureCacheKey key;
    if (fmt == G_IM_FMT_CI) {
        if (siz == G_IM_SIZ_4b) {
            uint8_t palSlot = paletteIndex / 8;
            key = { origAddr,
                    { palSlot == 0 ? mRdp->palette_dram_addr[0] : nullptr,
                      palSlot == 1 ? mRdp->palette_dram_addr[1] : nullptr },
                    fmt,
                    siz,
                    paletteIndex,
                    origSizeBytes };
        } else {
            // CI8 uses both palette halves
            key = { origAddr,     { mRdp->palette_dram_addr[0], mRdp->palette_dram_addr[1] }, fmt, siz, paletteIndex,
                    origSizeBytes };
        }
    } else {
        key = { origAddr, {}, fmt, siz, paletteIndex, origSizeBytes };
    }

    if (TextureCacheLookup(i, key)) {
        return;
    }

    // Guard against zero-sized textures that would cause divide-by-zero
    // or GPU API errors in UploadTexture.
    if (mRdp->texture_tile[tile].line_size_bytes == 0 || mRdp->loaded_texture[tmemIdex].size_bytes == 0 ||
        origAddr == nullptr) {
        return;
    }

    if ((texFlags & TEX_FLAG_LOAD_AS_IMG) != 0) {
        ImportTextureImg(tile, importReplacement);
        return;
    }

    // if load as raw is set then we load_raw();
    if ((texFlags & TEX_FLAG_LOAD_AS_RAW) != 0) {
        ImportTextureRaw(tile, importReplacement);
        return;
    }

    switch (fmt) {
        case G_IM_FMT_RGBA:
            if (siz == G_IM_SIZ_16b) {
                ImportTextureRgba16(tile, importReplacement);
            } else if (siz == G_IM_SIZ_32b) {
                ImportTextureRgba32(tile, importReplacement);
            } else {
                SPDLOG_ERROR("RGBA Texture that isn't 16 or 32 bit. Size = {}", siz);
                // OTRTODO: Sometimes, seemingly randomly, we end up here. Could be a bad dlist, could be
                // something F3D does not have supported. Further investigation is needed.
            }
            break;
        case G_IM_FMT_IA:
            if (siz == G_IM_SIZ_4b) {
                ImportTextureIA4(tile, importReplacement);
            } else if (siz == G_IM_SIZ_8b) {
                ImportTextureIA8(tile, importReplacement);
            } else if (siz == G_IM_SIZ_16b) {
                ImportTextureIA16(tile, importReplacement);
            } else {
                SPDLOG_ERROR("IA Texture that isn't 4, 8, or 16 bit. Size = {}", siz);
                ;
            }
            break;
        case G_IM_FMT_CI:
            if (siz == G_IM_SIZ_4b) {
                ImportTextureCi4(tile, importReplacement);
            } else if (siz == G_IM_SIZ_8b) {
                ImportTextureCi8(tile, importReplacement);
            } else if (siz == G_IM_SIZ_16b) {
                // CI+16b is hardware-invalid on N64. The tile's fmt is likely
                // stale from a prior draw. Decode as RGBA16 instead.
                ImportTextureRgba16(tile, importReplacement);
            } else if (siz == G_IM_SIZ_32b) {
                ImportTextureRgba32(tile, importReplacement);
            } else {
                SPDLOG_ERROR("CI Texture with unexpected size = {}", siz);
            }
            break;
        case G_IM_FMT_I:
            if (siz == G_IM_SIZ_4b) {
                ImportTextureI4(tile, importReplacement);
            } else if (siz == G_IM_SIZ_8b) {
                ImportTextureI8(tile, importReplacement);
            } else {
                SPDLOG_ERROR("I Texture that isn't 4 or 8 bit. Size = {}", siz);
            }
            break;
        case G_IM_FMT_YUV:
            SPDLOG_ERROR("YUV Textures not supported");
            break;
        default:
            SPDLOG_ERROR("Invalid texture format. Fmt = {}", fmt);
            break;
    }
}

void Interpreter::ImportTextureMask(int i, int tile) {
    uint32_t tmemIndex = mRdp->texture_tile[tile].tmem_index;
    RawTexMetadata metadata = mRdp->loaded_texture[tmemIndex].raw_tex_metadata;

    if (metadata.resource == nullptr) {
        return;
    }

    auto maskIter = mMaskedTextures.find(GetBaseTexturePath(metadata.resource->GetInitData()->Path));
    if (maskIter == mMaskedTextures.end()) {
        return;
    }

    const uint8_t* orig_addr = maskIter->second.mask;

    if (orig_addr == nullptr) {
        return;
    }

    TextureCacheKey key = { orig_addr, {}, 0, 0, 0, 0 };

    if (TextureCacheLookup(i, key)) {
        return;
    }

    uint32_t width = mRdp->texture_tile[tile].line_size_bytes;
    uint32_t height = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes /
                      mRdp->texture_tile[tile].line_size_bytes;
    switch (mRdp->texture_tile[tile].siz) {
        case G_IM_SIZ_4b:
            width *= 2;
            break;
        case G_IM_SIZ_8b:
        default:
            break;
        case G_IM_SIZ_16b:
            width /= 2;
            break;
        case G_IM_SIZ_32b:
            width /= 4;
            break;
    }

    for (uint32_t texIndex = 0; texIndex < width * height; texIndex++) {
        uint8_t masked = orig_addr[texIndex];
        if (masked) {
            mTexUploadBuffer[4 * texIndex + 0] = 0;
            mTexUploadBuffer[4 * texIndex + 1] = 0;
            mTexUploadBuffer[4 * texIndex + 2] = 0;
            mTexUploadBuffer[4 * texIndex + 3] = 0xFF;
        } else {
            mTexUploadBuffer[4 * texIndex + 0] = 0;
            mTexUploadBuffer[4 * texIndex + 1] = 0;
            mTexUploadBuffer[4 * texIndex + 2] = 0;
            mTexUploadBuffer[4 * texIndex + 3] = 0;
        }
    }

    mRapi->UploadTexture(mTexUploadBuffer, width, height);
    mRapi->BuildMipmaps();
}

void Interpreter::NormalizeVector(float v[3]) {
    float s = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= s;
    v[1] /= s;
    v[2] /= s;
}

void Interpreter::TransposedMatrixMul(float res[3], const float a[3], const float b[4][4]) {
    res[0] = a[0] * b[0][0] + a[1] * b[0][1] + a[2] * b[0][2];
    res[1] = a[0] * b[1][0] + a[1] * b[1][1] + a[2] * b[1][2];
    res[2] = a[0] * b[2][0] + a[1] * b[2][1] + a[2] * b[2][2];
}

void Interpreter::MatrixMul(float res[4][4], const float a[4][4], const float b[4][4]) {
    float tmp[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            tmp[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    memcpy(res, tmp, sizeof(tmp));
}

void Interpreter::CalculateNormalDir(const F3DLight_t* light, float coeffs[3]) {
    float light_dir[3] = { light->dir[0] / 127.0f, light->dir[1] / 127.0f, light->dir[2] / 127.0f };

    Interpreter::TransposedMatrixMul(coeffs, light_dir,
                                     mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1]);
    Interpreter::NormalizeVector(coeffs);
}

// Copies the lights exactly as GfxSpVertex consumes them. The shader reproduces that arithmetic, so
// the spaces and scaling here must stay in step with the CPU path in GfxSpVertex.
void Interpreter::UpdatePerPixelLightingState(uint32_t shadeMask) {
    GfxPerPixelLightingState& state = mPerPixelLightState;
    const int ambientIndex = mRsp->current_num_lights > 0 ? mRsp->current_num_lights - 1 : 0;
    const int numLights = std::clamp<int>(mRsp->current_num_lights - 1, 0, MAX_PER_PIXEL_LIGHTS);
    const bool positionalMode = (mRsp->geometry_mode & G_LIGHTING_POSITIONAL) != 0;

    state.numLights = numLights;
    const F3DLight_t& ambient = mRsp->current_lights[ambientIndex].l;
    state.ambient[0] = ambient.col[0];
    state.ambient[1] = ambient.col[1];
    state.ambient[2] = ambient.col[2];

    const float (*mtx)[4] = mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1];
    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            state.viewMatrix[col * 3 + row] = mtx[col][row];
        }
    }

    for (int i = 0; i < numLights; i++) {
        const F3DLight& light = mRsp->current_lights[i];
        GfxPerPixelLight& out = state.lights[i];
        // For directional lights this is the same normalized model-space direction the CPU uses.
        CalculateNormalDir(&light.l, out.dir);
        out.color[0] = light.l.col[0];
        out.color[1] = light.l.col[1];
        out.color[2] = light.l.col[2];
        out.pos[0] = light.p.pos[0];
        out.pos[1] = light.p.pos[1];
        out.pos[2] = light.p.pos[2];
        out.atten[0] = light.p.unk7;
        out.atten[1] = light.p.unkE;
        out.positional = positionalMode && light.p.unk3 != 0;
    }

    state.shadeMask = static_cast<int32_t>(shadeMask);
    state.pointLightQuality = mPerPixelPointLightQuality;
    UpdateCharacterLightingState(state, mtx);
    state.cameraPosition[3] = SolveCameraPosition(state.cameraPosition) ? 1.0f : 0.0f;
    state.serial = ++mPerPixelLightSerial;
}

// The camera position in the world space of the draw's vertices, recovered from the combined view-projection in P:
// the point whose clip x, y and w are all zero. False when P has no such point (the HUD's orthographic projection).
bool Interpreter::SolveCameraPosition(float out[3]) {
    // Solve [e 1] * P = (0, 0, *, 0) for columns 0, 1, 3.
    static const int cols[3] = { 0, 1, 3 };
    float a[3][4];
    for (int i = 0; i < 3; i++) {
        for (int r = 0; r < 3; r++) {
            a[i][r] = mRsp->P_matrix[r][cols[i]];
        }
        a[i][3] = -mRsp->P_matrix[3][cols[i]];
    }
    for (int col = 0; col < 3; col++) {
        int pivot = col;
        for (int r = col + 1; r < 3; r++) {
            if (std::fabs(a[r][col]) > std::fabs(a[pivot][col])) {
                pivot = r;
            }
        }
        if (std::fabs(a[pivot][col]) < 1e-9f) {
            return false;
        }
        for (int k = 0; k < 4; k++) {
            std::swap(a[col][k], a[pivot][k]);
        }
        for (int r = 0; r < 3; r++) {
            if (r == col) {
                continue;
            }
            const float f = a[r][col] / a[col][col];
            for (int k = col; k < 4; k++) {
                a[r][k] -= f * a[col][k];
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        out[i] = a[i][3] / a[i][i];
    }
    return true;
}

// New Dawn character lighting: lit draws under a non-identity model-to-world matrix (actor limbs and objects; room
// meshes are drawn with the identity) get the camera position and a strength; everything else gets strength 0. The
// camera is recovered from the combined view-projection in P: it is the point whose clip x, y and w are all zero
// (the HUD's orthographic projection has no such point, so it stays off).
void Interpreter::UpdateCharacterLightingState(GfxPerPixelLightingState& state, const float (*mtx)[4]) {
    state.characterLight[0] = state.characterLight[1] = state.characterLight[2] = state.characterLight[3] = 0.0f;
    if (!mCharacterLighting || !mRapi->HdrWorldActive()) {
        return;
    }
    bool identity = true;
    for (int r = 0; r < 4 && identity; r++) {
        for (int c = 0; c < 4; c++) {
            if (std::fabs(mtx[r][c] - (r == c ? 1.0f : 0.0f)) > 1e-3f) {
                identity = false;
                break;
            }
        }
    }
    // Actor_Draw binds an actor's lights once, then a skeleton loads one matrix per limb. Props drawn as actors
    // (stalls, towers, doors) load a single matrix after their lights, so they stay on the original shading.
    if (identity || mMatrixLoadsSinceLights < 2) {
        return;
    }
    float eye[3];
    if (!SolveCameraPosition(eye)) {
        return;
    }
    state.characterLight[0] = eye[0];
    state.characterLight[1] = eye[1];
    state.characterLight[2] = eye[2];
    state.characterLight[3] = mCharacterStrength;
}

// ---- New Dawn smooth characters (docs/SMOOTH_CHARACTERS.md) ------------------------------------------------------
//
// Actor triangles are drawn as cubic patches (curved PN triangles) by the tessellation stages. A patch is defined by
// its three corners and, per corner, the offsets of the two edge control points next to it. The surface has no cracks
// when both triangles sharing an edge are given the same offsets for it, and that needs to know the neighbour: the
// vertex stream has no adjacency. It is rebuilt here, once: the first time a triangle command is run its triangle is
// put aside, and when the actor is complete all its new triangles are welded by position and analysed together. The
// result (welded normals, crease tangents) is stored in each vertex's own model space, so it holds for every pose and
// every later frame only has to look it up.

static inline float SmoothDot(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline void SmoothCross(float out[3], const float a[3], const float b[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

// False (and a zero vector) when there is no direction to keep.
static inline bool SmoothNormalize(float v[3]) {
    const float len2 = SmoothDot(v, v);
    if (!(len2 > 1e-20f)) {
        v[0] = v[1] = v[2] = 0.0f;
        return false;
    }
    const float inv = 1.0f / std::sqrt(len2);
    v[0] *= inv;
    v[1] *= inv;
    v[2] *= inv;
    return true;
}

static inline void SmoothQuantize(int16_t out[3], const float v[3]) {
    for (int k = 0; k < 3; k++) {
        out[k] = (int16_t)std::lround(std::clamp(v[k], -1.0f, 1.0f) * 32767.0f);
    }
}

static inline size_t SmoothHash(uint64_t key, const void* v0) {
    const uint64_t h = key * 0x9E3779B97F4A7C15ull ^ ((uint64_t)(uintptr_t)v0 * 0xC2B2AE3D27D4EB4Full);
    return (size_t)(h ^ (h >> 29));
}

void Interpreter::SmoothBeginFrame() {
    if (mSmoothDebug && mSmooth) {
        static uint32_t frames = 0;
        static uint64_t ns = 0, patches = 0;
        ns += mSmoothStats.ns;
        patches += mSmoothStats.patches;
        if (++frames == 300 || mSmoothStats.analysed != 0) {
            SPDLOG_INFO("SMOOTHDBG {} patches/frame, {:.4f} ms CPU/frame over {} frames; cache {} triangles; this "
                        "frame analysed {} new ({} put aside)",
                        patches / frames, double(ns) / 1e6 / frames, frames, mSmoothCount, mSmoothStats.analysed,
                        mSmoothStats.pending);
            frames = 0;
            ns = patches = 0;
        }
    }
    mSmoothStats = {};
    mSmoothMtx.clear();
    mSmoothPending.clear();
    mSmoothNewThisFrame = 0;
    mSmoothCameraSet = false;
    mSmoothBaseDepth = ~size_t(0); // no actor yet
    mSmoothKey = 0;
    for (SmoothSlot& slot : mSmoothSlots) {
        slot.src = nullptr;
    }
}

// The current model-to-world matrix as an entry of this frame's table.
uint32_t Interpreter::SmoothSyncMatrix() {
    const float (*mtx)[4] = mRsp->modelview_matrix_stack[std::max<int>(mRsp->modelview_matrix_stack_size, 1) - 1];
    if (!mSmoothMtx.empty() && memcmp(mSmoothMtx.back().src, mtx, sizeof(float) * 16) == 0) {
        return (uint32_t)mSmoothMtx.size() - 1;
    }
    SmoothMtx& e = mSmoothMtx.emplace_back();
    memcpy(e.src, mtx, sizeof(e.src));
    e.identity = true;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            if (std::fabs(mtx[r][c] - (r == c ? 1.0f : 0.0f)) > 1e-3f) {
                e.identity = false;
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int k = 0; k < 3; k++) {
            e.m[i][k] = mtx[i][k];
        }
    }
    // Vertices are rows: world = v * m. Normals go through the cofactor matrix (rows r1 x r2, r2 x r0, r0 x r1), with
    // the determinant's sign folded in so that they keep pointing outwards on mirrored limbs.
    SmoothCross(e.cof[0], e.m[1], e.m[2]);
    SmoothCross(e.cof[1], e.m[2], e.m[0]);
    SmoothCross(e.cof[2], e.m[0], e.m[1]);
    if (SmoothDot(e.m[0], e.cof[0]) < 0.0f) {
        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 3; k++) {
                e.cof[i][k] = -e.cof[i][k];
            }
        }
    }
    return (uint32_t)mSmoothMtx.size() - 1;
}

Interpreter::SmoothTri* Interpreter::SmoothFind(uint64_t key, const void* v0) {
    if (mSmoothTable.empty()) {
        return nullptr;
    }
    const size_t mask = mSmoothTable.size() - 1;
    for (size_t h = SmoothHash(key, v0) & mask;; h = (h + 1) & mask) {
        SmoothTri& e = mSmoothTable[h];
        if (e.key == 0) {
            return nullptr;
        }
        if (e.key == key && e.v[0] == v0) {
            return &e;
        }
    }
}

// The entry for (key, v0), created empty when new. Growing moves the table: no entry pointer survives a call.
Interpreter::SmoothTri* Interpreter::SmoothInsert(uint64_t key, const void* v0) {
    if (SmoothTri* found = SmoothFind(key, v0)) {
        return found;
    }
    if ((mSmoothCount + 1) * 2 > mSmoothTable.size()) {
        std::vector<SmoothTri> old(std::max<size_t>(mSmoothTable.size() * 2, 4096));
        old.swap(mSmoothTable);
        mSmoothCount = 0;
        for (const SmoothTri& e : old) {
            if (e.key != 0) {
                *SmoothInsert(e.key, e.v[0]) = e;
            }
        }
    }
    const size_t mask = mSmoothTable.size() - 1;
    size_t h = SmoothHash(key, v0) & mask;
    while (mSmoothTable[h].key != 0) {
        h = (h + 1) & mask;
    }
    SmoothTri& e = mSmoothTable[h];
    e = {};
    e.key = key;
    e.v[0] = v0;
    mSmoothCount++;
    return &e;
}

// The cached data of the triangle being drawn when it is a character's; otherwise null. A lit triangle of an actor's
// display list that has not been seen before is put aside for SmoothAnalyze and drawn as it is this once.
const Interpreter::SmoothTri* Interpreter::SmoothLookup(uint64_t key, const uint8_t idx[3]) {
    // Lit geometry (the colour bytes are a normal) inside a display list an actor called: Actor_Draw binds the lights
    // in the frame's own command buffer, so anything deeper than that is a model, not immediate geometry.
    if ((mRsp->geometry_mode & G_LIGHTING) == 0 || mMatrixLoadsSinceLights == 0 ||
        g_exec_stack.cmd_stack.size() <= mSmoothBaseDepth || !(mPerPixelLightingEnabled || mRapi->HdrWorldActive())) {
        return nullptr;
    }
    const SmoothSlot* slot[3];
    for (int i = 0; i < 3; i++) {
        if (idx[i] >= MAX_VERTICES + 4) {
            return nullptr;
        }
        slot[i] = &mSmoothSlots[idx[i]];
        // Room meshes are drawn with the identity; a vertex loaded unlit carries a colour, not a normal.
        if (slot[i]->src == nullptr || !slot[i]->lit || slot[i]->mtx >= mSmoothMtx.size() ||
            mSmoothMtx[slot[i]->mtx].identity) {
            return nullptr;
        }
    }
    SmoothTri* tri = SmoothFind(key, slot[0]->src);
    if (tri != nullptr) {
        if (tri->v[1] == slot[1]->src && tri->v[2] == slot[2]->src) {
            return tri->state == SMOOTH_TRI_CHARACTER ? tri : nullptr;
        }
        // The command now draws other vertices (commands built per frame). A few tries, then it is left alone.
        if (tri->stale >= 4) {
            return nullptr;
        }
        tri->stale++;
    }
    if (mSmoothNewThisFrame >= 60000) {
        return nullptr; // a bound on one frame's analysis; the rest follows next frame
    }
    mSmoothNewThisFrame++;
    SmoothPending& p = mSmoothPending.emplace_back();
    p.key = key;
    p.decal = (mRdp->other_mode_l & ZMODE_DEC) == ZMODE_DEC;
    for (int i = 0; i < 3; i++) {
        p.v[i] = slot[i]->src;
        p.mtx[i] = slot[i]->mtx;
        memcpy(p.n[i], slot[i]->n, sizeof(p.n[i]));
        memcpy(p.pos[i], mRsp->loaded_vertices[idx[i]].view_pos, sizeof(p.pos[i]));
    }
    return nullptr;
}

void Interpreter::SmoothFinishActor() {
    if (mSmoothPending.empty()) {
        return;
    }
    // Actor_Draw binds an actor's lights once and a skeleton then loads a matrix per limb; a prop loads one.
    SmoothAnalyze(mMatrixLoadsSinceLights >= 2 || mSmoothProps);
    mSmoothPending.clear();
}

// Welds one actor's new triangles and stores what each needs to be drawn as a patch.
void Interpreter::SmoothAnalyze(bool character) {
    const auto start = std::chrono::steady_clock::now();
    const std::vector<SmoothPending>& pending = mSmoothPending;
    mSmoothStats.pending += (uint32_t)pending.size();

    // A display list drawn twice by one actor is one mesh: keep the first sight of each command.
    std::vector<uint32_t> tris;
    tris.reserve(pending.size());
    {
        std::unordered_set<uint64_t> seen;
        for (uint32_t i = 0; i < pending.size(); i++) {
            if (seen.insert((uint64_t)SmoothHash(pending[i].key, pending[i].v[0])).second) {
                tris.push_back(i);
            }
        }
    }
    const uint32_t count = (uint32_t)tris.size();
    mSmoothStats.analysed += count;
    const auto store = [&](const SmoothPending& p) {
        SmoothTri* e = SmoothInsert(p.key, p.v[0]);
        const uint8_t stale = e->stale;
        *e = {};
        e->key = p.key;
        e->stale = stale;
        for (int i = 0; i < 3; i++) {
            e->v[i] = p.v[i];
            memcpy(e->own[i], p.n[i], sizeof(e->own[i]));
        }
        e->state = SMOOTH_TRI_SKIP;
        return e;
    };
    if (!character) {
        for (uint32_t t = 0; t < count; t++) {
            store(pending[tris[t]]); // remembered as "not a character", so it is not put aside again
        }
        return;
    }

    // 1. Weld. Corners at the same world position are one point: twins (texture seams, hard edges, the vertices a
    // limb shares with its parent) go through the same matrix, so their positions are equal to the bit.
    struct Corner {
        uint32_t pos, cluster;
        float nw[3]; // world normal
        float weight;
    };
    struct PosKey {
        uint32_t b[3];
        bool operator==(const PosKey& o) const {
            return b[0] == o.b[0] && b[1] == o.b[1] && b[2] == o.b[2];
        }
    };
    struct PosHash {
        size_t operator()(const PosKey& k) const {
            return (size_t)((k.b[0] * 0x9E3779B97F4A7C15ull) ^ (k.b[1] * 0xC2B2AE3D27D4EB4Full) ^
                            (k.b[2] * 0x165667B19E3779F9ull));
        }
    };
    constexpr uint32_t NONE = 0xFFFFFFFFu;
    std::vector<Corner> corners(count * 3);
    std::unordered_map<PosKey, uint32_t, PosHash> posIds;
    posIds.reserve(count * 2);
    for (uint32_t t = 0; t < count; t++) {
        const SmoothPending& p = pending[tris[t]];
        for (int i = 0; i < 3; i++) {
            Corner& c = corners[t * 3 + i];
            PosKey key;
            for (int k = 0; k < 3; k++) {
                const float f = p.pos[i][k] + 0.0f; // -0 and +0 are the same place
                memcpy(&key.b[k], &f, sizeof(f));
            }
            c.pos = posIds.try_emplace(key, (uint32_t)posIds.size()).first->second;
            c.cluster = NONE;
            const SmoothMtx& m = mSmoothMtx[p.mtx[i]];
            for (int k = 0; k < 3; k++) {
                c.nw[k] = p.n[i][0] * m.cof[0][k] + p.n[i][1] * m.cof[1][k] + p.n[i][2] * m.cof[2][k];
            }
            SmoothNormalize(c.nw);
            // The corner's angle weighs its normal in an average.
            const float* a = p.pos[i];
            const float* b = p.pos[(i + 1) % 3];
            const float* d = p.pos[(i + 2) % 3];
            float e1[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, e2[3] = { d[0] - a[0], d[1] - a[1], d[2] - a[2] };
            c.weight = SmoothNormalize(e1) && SmoothNormalize(e2)
                           ? std::acos(std::clamp(SmoothDot(e1, e2), -1.0f, 1.0f)) + 1e-4f : 1e-4f;
        }
    }

    // 2. Normal clusters per point. Normals within the crease angle of each other are one smooth surface and share
    // their average; a point with several clusters lies on a hard edge the artist made by splitting the normals.
    struct Cluster {
        float rep[3], sum[3], g[3];
        bool mixed;
    };
    std::vector<Cluster> clusters;
    std::vector<std::vector<uint32_t>> posClusters(posIds.size());
    for (Corner& c : corners) {
        if (SmoothDot(c.nw, c.nw) < 0.5f) {
            continue; // no normal: this corner stays where it is
        }
        for (uint32_t k : posClusters[c.pos]) {
            const float d = SmoothDot(clusters[k].rep, c.nw);
            if (d >= mSmoothCrease) {
                c.cluster = k;
                if (d < 0.99999f) {
                    clusters[k].mixed = true;
                }
                break;
            }
        }
        if (c.cluster == NONE) {
            c.cluster = (uint32_t)clusters.size();
            posClusters[c.pos].push_back(c.cluster);
            Cluster& k = clusters.emplace_back();
            memcpy(k.rep, c.nw, sizeof(k.rep));
            k.sum[0] = k.sum[1] = k.sum[2] = 0.0f;
            k.mixed = false;
        }
        for (int k = 0; k < 3; k++) {
            clusters[c.cluster].sum[k] += c.nw[k] * c.weight;
        }
    }
    for (Cluster& k : clusters) {
        // Equal normals stay exactly what they were (their average would only add rounding).
        memcpy(k.g, k.mixed ? k.sum : k.rep, sizeof(k.g));
        if (!SmoothNormalize(k.g)) {
            memcpy(k.g, k.rep, sizeof(k.g));
        }
    }

    // 3. Faces and edges. A triangle lying exactly on an earlier one (a decal or a second pass over the same face)
    // takes that one's data, so both stay the same surface; decals are otherwise a layer of their own, and an edge
    // counts the triangles of its layer that use it.
    struct Edge {
        uint32_t tri[2];
        uint32_t count;
        float thin; // the smallest height over this edge of the triangles on it, as a fraction of its length
    };
    struct FaceKey {
        uint32_t p[3];
        bool operator==(const FaceKey& o) const {
            return p[0] == o.p[0] && p[1] == o.p[1] && p[2] == o.p[2];
        }
    };
    struct FaceHash {
        size_t operator()(const FaceKey& k) const {
            return (size_t)((k.p[0] * 0x9E3779B97F4A7C15ull) ^ (k.p[1] * 0xC2B2AE3D27D4EB4Full) ^
                            (k.p[2] * 0x165667B19E3779F9ull));
        }
    };
    const auto edgeKey = [](uint32_t a, uint32_t b, bool decal) {
        if (a > b) {
            std::swap(a, b);
        }
        return ((uint64_t)decal << 63) | ((uint64_t)a << 31) | b;
    };
    std::unordered_map<uint64_t, Edge> edges;
    std::unordered_map<FaceKey, uint32_t, FaceHash> faces;
    edges.reserve(count * 2);
    faces.reserve(count);
    std::vector<uint32_t> baseOf(count, NONE); // the triangle this one lies on; itself for a face of its own
    for (uint32_t t = 0; t < count; t++) {
        const Corner* c = &corners[t * 3];
        if (c[0].pos == c[1].pos || c[1].pos == c[2].pos || c[2].pos == c[0].pos) {
            continue; // degenerate: stays flat and is nobody's neighbour
        }
        FaceKey face = { { c[0].pos, c[1].pos, c[2].pos } };
        std::sort(face.p, face.p + 3);
        const auto [it, added] = faces.try_emplace(face, t);
        baseOf[t] = it->second;
        if (!added) {
            continue;
        }
        const SmoothPending& p = pending[tris[t]];
        const bool decal = p.decal;
        const float e1[3] = { p.pos[1][0] - p.pos[0][0], p.pos[1][1] - p.pos[0][1], p.pos[1][2] - p.pos[0][2] };
        const float e2[3] = { p.pos[2][0] - p.pos[0][0], p.pos[2][1] - p.pos[0][1], p.pos[2][2] - p.pos[0][2] };
        float areaN[3];
        SmoothCross(areaN, e1, e2);
        const float area2 = std::sqrt(SmoothDot(areaN, areaN)); // twice the area
        for (int i = 0; i < 3; i++) {
            Edge& e = edges.try_emplace(edgeKey(c[i].pos, c[(i + 1) % 3].pos, decal), Edge{ { NONE, NONE }, 0, 1e9f })
                          .first->second;
            if (e.count < 2) {
                e.tri[e.count] = t;
            }
            e.count++;
            const float* a = p.pos[i];
            const float* b = p.pos[(i + 1) % 3];
            const float d[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
            e.thin = std::min(e.thin, area2 / std::max(SmoothDot(d, d), 1e-12f));
        }
    }

    // 4. Per triangle: the welded normal of each corner and how each edge end is curved.
    //   NORMAL   the edge leaves the corner in the tangent plane of the corner's own welded normal (PN triangle).
    //            Both triangles on the edge are in the same cluster there, or the edge has one triangle.
    //   CREASE   the two triangles are in different clusters: the edge follows the line both tangent planes share,
    //            so a hard edge curves along itself (a round rim) and stays hard.
    //   STRAIGHT anything else (more than two triangles, opposite normals, no normal): the flat triangle's edge.
    std::vector<SmoothTri> results(count);
    for (uint32_t t = 0; t < count; t++) {
        const SmoothPending& p = pending[tris[t]];
        SmoothTri& r = results[t];
        r = {};
        for (int s = 0; s < 6; s++) {
            r.kind[s] = SMOOTH_END_STRAIGHT;
        }
        if (baseOf[t] != t) {
            continue;
        }
        const Corner* c = &corners[t * 3];
        for (int i = 0; i < 3; i++) {
            if (c[i].cluster == NONE) {
                continue;
            }
            const SmoothMtx& m = mSmoothMtx[p.mtx[i]];
            const Cluster& own = clusters[c[i].cluster];
            // World normal back into this vertex's model space: n = m * N (the inverse of the cofactor transform up
            // to scale).
            float gm[3];
            for (int a = 0; a < 3; a++) {
                gm[a] = m.m[a][0] * own.g[0] + m.m[a][1] * own.g[1] + m.m[a][2] * own.g[2];
            }
            SmoothNormalize(gm);
            SmoothQuantize(r.g[i], gm);
            if (own.mixed) {
                r.mixed |= 1 << i;
            }
            r.thin[i] = (uint16_t)std::min(
                edges.find(edgeKey(c[i].pos, c[(i + 1) % 3].pos, p.decal))->second.thin * 8192.0f, 65535.0f);
            for (int e = 0; e < 2; e++) {
                const int j = e == 0 ? (i + 1) % 3 : (i + 2) % 3;
                const Edge& edge = edges.find(edgeKey(c[i].pos, c[j].pos, p.decal))->second;
                uint8_t& kind = r.kind[i * 2 + e];
                if (edge.count == 1) {
                    kind = SMOOTH_END_NORMAL;
                } else if (edge.count == 2) {
                    const uint32_t other = edge.tri[0] == t ? edge.tri[1] : edge.tri[0];
                    uint32_t theirs = NONE;
                    for (int k = 0; k < 3; k++) {
                        if (corners[other * 3 + k].pos == c[i].pos) {
                            theirs = corners[other * 3 + k].cluster;
                        }
                    }
                    if (theirs == c[i].cluster) {
                        kind = SMOOTH_END_NORMAL;
                    } else if (theirs != NONE) {
                        float tw[3], tm[3];
                        SmoothCross(tw, own.g, clusters[theirs].g);
                        // Nearly parallel or opposite normals (two sides of a sheet) have no shared line.
                        if (SmoothDot(tw, tw) > 0.03f) {
                            SmoothNormalize(tw);
                            // A direction on the surface back into model space: the inverse matrix is the transposed
                            // cofactor over the determinant, and the sign and length do not matter.
                            for (int a = 0; a < 3; a++) {
                                tm[a] = m.cof[a][0] * tw[0] + m.cof[a][1] * tw[1] + m.cof[a][2] * tw[2];
                            }
                            if (SmoothNormalize(tm)) {
                                SmoothQuantize(r.t[i * 2 + e], tm);
                                kind = SMOOTH_END_CREASE;
                            }
                        }
                    }
                }
            }
        }
    }
    for (uint32_t t = 0; t < count; t++) {
        const SmoothPending& p = pending[tris[t]];
        SmoothTri* e = store(p);
        e->state = SMOOTH_TRI_CHARACTER;
        for (int s = 0; s < 6; s++) {
            e->kind[s] = SMOOTH_END_STRAIGHT;
        }
        const uint32_t base = baseOf[t];
        if (base == NONE) {
            continue;
        }
        const SmoothTri& r = results[base];
        // Corner i of this triangle is corner perm[i] of the face it lies on (the same one when it is its own).
        int perm[3] = { 0, 1, 2 };
        if (base != t) {
            for (int i = 0; i < 3; i++) {
                for (int k = 0; k < 3; k++) {
                    if (corners[base * 3 + k].pos == corners[t * 3 + i].pos) {
                        perm[i] = k;
                    }
                }
            }
        }
        for (int i = 0; i < 3; i++) {
            memcpy(e->g[i], r.g[perm[i]], sizeof(e->g[i]));
            if ((r.mixed >> perm[i] & 1) || corners[base * 3 + perm[i]].cluster != corners[t * 3 + i].cluster) {
                e->mixed |= 1 << i;
            }
            for (int s = 0; s < 2; s++) {
                const int j = s == 0 ? (i + 1) % 3 : (i + 2) % 3;
                const int from = perm[i] * 2 + ((perm[i] + 1) % 3 == perm[j] ? 0 : 1);
                e->kind[i * 2 + s] = r.kind[from];
                memcpy(e->t[i * 2 + s], r.t[from], sizeof(e->t[0]));
                if (s == 0) { // the edge from corner i to i+1, whichever way the face it lies on runs along it
                    e->thin[i] = r.thin[(perm[i] + 1) % 3 == perm[j] ? perm[i] : perm[j]];
                }
            }
        }
    }
    mSmoothStats.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
}

// The patch data of one triangle in this frame's pose: per corner the shading normal (in the batch's model space,
// as the shader expects it) and the two control-point offsets (world space).
void Interpreter::SmoothCorners(const SmoothTri* tri, const uint8_t idx[3], const LoadedVertex* const v[3],
                                float normals[3][3], float offsets[3][2][3]) {
    const auto start = mSmoothDebug ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const uint32_t batchMtx = SmoothSyncMatrix();
    const SmoothMtx& bm = mSmoothMtx[batchMtx];
    const float* nw[3];
    bool moved[3];
    for (int i = 0; i < 3; i++) {
        SmoothSlot& s = mSmoothSlots[idx[i]];
        // A CPU-skinned mesh rewrites its normals every frame. Twins with equal normals stay equal, so the vertex's
        // own normal still welds; anything that needed the stored neighbour falls back to a straight edge below.
        moved[i] = memcmp(s.n, tri->own[i], sizeof(s.n)) != 0;
        int16_t g[3];
        if (moved[i]) {
            g[0] = (int16_t)(s.n[0] * 256);
            g[1] = (int16_t)(s.n[1] * 256);
            g[2] = (int16_t)(s.n[2] * 256);
        } else {
            memcpy(g, tri->g[i], sizeof(g));
        }
        if (!s.worldValid || memcmp(s.gKey, g, sizeof(g)) != 0) {
            const SmoothMtx& m = mSmoothMtx[s.mtx];
            for (int k = 0; k < 3; k++) {
                s.world[k] = g[0] * m.cof[0][k] + g[1] * m.cof[1][k] + g[2] * m.cof[2][k];
            }
            SmoothNormalize(s.world);
            memcpy(s.gKey, g, sizeof(g));
            s.worldValid = true;
            s.batchMtx = ~0u;
        }
        if (s.batchMtx != batchMtx) {
            // The shader turns normals with the batch's matrix, whichever matrix the vertex was loaded through (a
            // limb's display list also loads its parent's): hand it the normal that comes out right.
            for (int a = 0; a < 3; a++) {
                s.batch[a] = bm.m[a][0] * s.world[0] + bm.m[a][1] * s.world[1] + bm.m[a][2] * s.world[2];
            }
            if (SmoothNormalize(s.batch)) {
                s.batch[0] *= 127.0f;
                s.batch[1] *= 127.0f;
                s.batch[2] *= 127.0f;
            } else {
                memcpy(s.batch, v[i]->normal, sizeof(s.batch));
            }
            s.batchMtx = batchMtx;
        }
        nw[i] = s.world;
        memcpy(normals[i], s.batch, sizeof(s.batch));
    }
    for (int i = 0; i < 3; i++) {
        for (int e = 0; e < 2; e++) {
            const int j = e == 0 ? (i + 1) % 3 : (i + 2) % 3;
            float* o = offsets[i][e];
            o[0] = o[1] = o[2] = 0.0f;
            uint8_t kind = tri->kind[i * 2 + e];
            if (moved[i] && (kind == SMOOTH_END_CREASE || (tri->mixed >> i & 1))) {
                kind = SMOOTH_END_STRAIGHT;
            }
            if (kind == SMOOTH_END_STRAIGHT) {
                continue;
            }
            // Everything below uses only what both triangles on this edge have in common (the two end points and the
            // stored direction), in the same order, so they get the same offset to the bit.
            const float d[3] = { v[j]->view_pos[0] - v[i]->view_pos[0], v[j]->view_pos[1] - v[i]->view_pos[1],
                                 v[j]->view_pos[2] - v[i]->view_pos[2] };
            const float length = std::sqrt(SmoothDot(d, d));
            // A bulge grows with the length of its edge, so a long edge of a thin part (a spear shaft, a beam, the
            // side of a pointed hat) would swell far beyond the part's own width. The pull is bounded by the height of
            // the narrowest triangle on the edge: round cross-sections are untouched, slivers stay slim.
            const float bound = tri->thin[e == 0 ? i : j] * (1.0f / 8192.0f) * length * mSmoothThin;
            if (kind == SMOOTH_END_NORMAL) {
                // PN triangle: the control point at a third of the edge, dropped onto the corner's tangent plane.
                float w = SmoothDot(d, nw[i]);
                // An edge leaving steeply (a cone's tip, a spike) would swell into a dome: past the limit the pull is
                // mostly held back, so pointed parts keep their shape.
                const float steep = std::fabs(w) / std::max(length, 1e-6f);
                if (steep > mSmoothTip) {
                    w *= (mSmoothTip + (steep - mSmoothTip) * 0.25f) / steep;
                }
                w = std::clamp(w, -bound, bound);
                o[0] = -w * nw[i][0] * (1.0f / 3.0f);
                o[1] = -w * nw[i][1] * (1.0f / 3.0f);
                o[2] = -w * nw[i][2] * (1.0f / 3.0f);
            } else {
                // Crease: the control point goes onto the crease line through the corner.
                const SmoothMtx& m = mSmoothMtx[mSmoothSlots[idx[i]].mtx];
                const int16_t* tm = tri->t[i * 2 + e];
                float tw[3];
                for (int k = 0; k < 3; k++) {
                    tw[k] = tm[0] * m.m[0][k] + tm[1] * m.m[1][k] + tm[2] * m.m[2][k];
                }
                if (!SmoothNormalize(tw)) {
                    continue;
                }
                const float along = SmoothDot(d, tw);
                const float side[3] = { tw[0] * along - d[0], tw[1] * along - d[1], tw[2] * along - d[2] };
                const float pull = std::sqrt(SmoothDot(side, side));
                const float scale = (pull > bound ? bound / pull : 1.0f) * (1.0f / 3.0f);
                o[0] = side[0] * scale;
                o[1] = side[1] * scale;
                o[2] = side[2] * scale;
            }
        }
    }
    mSmoothStats.patches++;
    if (mSmoothDebug) {
        mSmoothStats.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
    }
}

// The tessellation stages turn world-space offsets into clip space with the camera the batch was projected with.
void Interpreter::SmoothSyncCamera() {
    const GfxClipParameters clip = mRapi->GetClipParameters();
    const float xScale = AdjXForAspectRatio(1.0f);
    const float key[4] = { xScale, clip.invertY ? -1.0f : 1.0f, mSmoothStrength, mSmoothTolerance };
    if (mSmoothCameraSet && memcmp(mSmoothP, mRsp->P_matrix, sizeof(mSmoothP)) == 0 &&
        memcmp(mSmoothCameraKey, key, sizeof(key)) == 0) {
        return;
    }
    Flush(); // what is in the buffer was projected with the previous camera
    memcpy(mSmoothP, mRsp->P_matrix, sizeof(mSmoothP));
    memcpy(mSmoothCameraKey, key, sizeof(key));
    mSmoothCameraSet = true;
    float vp[16];
    for (int i = 0; i < 4; i++) {
        vp[i * 4 + 0] = mSmoothP[i][0] * xScale;
        vp[i * 4 + 1] = mSmoothP[i][1] * key[1];
        vp[i * 4 + 2] = clip.z_is_from_0_to_1 ? (mSmoothP[i][2] + mSmoothP[i][3]) * 0.5f : mSmoothP[i][2];
        vp[i * 4 + 3] = mSmoothP[i][3];
    }
    // Pixels a world unit covers at w = 1 (the view rotation is orthonormal: the length of the projection's y column).
    const float fy = std::sqrt(mSmoothP[0][1] * mSmoothP[0][1] + mSmoothP[1][1] * mSmoothP[1][1] +
                               mSmoothP[2][1] * mSmoothP[2][1]);
    const float params[4] = { 0.5f * (float)mCurDimensions.height * fy, mSmoothStrength, (float)mSmoothMaxLevel,
                              mSmoothTolerance };
    mRapi->SetSmoothCharacters(vp, params);
}

void Interpreter::GfxSpMatrix(uint8_t parameters, const int32_t* addr) {
    float matrix[4][4];

    if (auto it = mCurMtxReplacements->find((Mtx*)addr); it != mCurMtxReplacements->end()) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                float v = it->second.mf[i][j];
                int as_int = (int)(v * 65536.0f);
                matrix[i][j] = as_int * (1.0f / 65536.0f);
            }
        }
    } else {
#ifndef GBI_FLOATS
        // Original GBI where fixed point matrices are used
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j += 2) {
                int32_t int_part = addr[i * 2 + j / 2];
                uint32_t frac_part = addr[8 + i * 2 + j / 2];
                matrix[i][j] = (int32_t)((int_part & 0xffff0000) | (frac_part >> 16)) / 65536.0f;
                matrix[i][j + 1] = (int32_t)((int_part << 16) | (frac_part & 0xffff)) / 65536.0f;
            }
        }
#else
        // For a modified GBI where fixed point values are replaced with floats
        memcpy(matrix, addr, sizeof(matrix));
#endif
    }

    const int8_t mtx_projection = get_attr(MTX_PROJECTION);
    const int8_t mtx_load = get_attr(MTX_LOAD);
    const int8_t mtx_push = get_attr(MTX_PUSH);

    if (parameters & mtx_projection) {
        if (parameters & mtx_load) {
            memcpy(mRsp->P_matrix, matrix, sizeof(matrix));
            if (mTaaJitter) {
                JitterProjection();
            }
        } else {
            MatrixMul(mRsp->P_matrix, matrix, mRsp->P_matrix);
        }
        if (mCharacterLighting && mRapi->HdrWorldActive()) {
            mRsp->light_state_revision++; // the camera position is derived from P
        }
    } else { // G_MTX_MODELVIEW
        if ((parameters & mtx_push) && mRsp->modelview_matrix_stack_size < 11) {
            ++mRsp->modelview_matrix_stack_size;
            memcpy(mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1],
                   mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 2], sizeof(matrix));
        }
        if (parameters & mtx_load) {
            if (mRsp->modelview_matrix_stack_size == 0)
                ++mRsp->modelview_matrix_stack_size;
            memcpy(mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1], matrix, sizeof(matrix));
        } else {
            MatrixMul(mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1], matrix,
                      mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1]);
        }
        mRsp->lights_changed = 1;
        mRsp->light_state_revision++;
        mMatrixLoadsSinceLights++;
    }
    MatrixMul(mRsp->MP_matrix, mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1], mRsp->P_matrix);
}

// New Dawn TAA. The game loads its projection (G_MTX_LOAD) and then multiplies the view into it, so a clip-space shift
// applied right after the load ends up as the rightmost factor of the final view-projection. Only perspective
// projections are shifted: orthographic ones (screen fills, HUD-style 3D) stay on the pixel grid.
void Interpreter::JitterProjection() {
    float (*P)[4] = mRsp->P_matrix;
    if (mFbActive || (P[0][3] == 0.0f && P[1][3] == 0.0f && P[2][3] == 0.0f)) {
        return;
    }
    for (int r = 0; r < 4; r++) {
        P[r][0] += mTaaOffset[0] * P[r][3];
        P[r][1] += mTaaOffset[1] * P[r][3];
    }
}

static float Halton(uint32_t index, uint32_t base) {
    float f = 1.0f, r = 0.0f;
    while (index > 0) {
        f /= (float)base;
        r += f * (float)(index % base);
        index /= base;
    }
    return r;
}

// Called right after the renderer opened the HDR world (WORLD_BEGIN). The camera loaded before the marker is still the
// current P here: hand its un-jittered form to the renderer for reprojection, then shift it for everything that
// follows.
void Interpreter::BeginTaa(const NewDawnPostFxParams& params) {
    mTaaJitter = false;
    if (!mRapi->HdrWorldActive()) {
        return;
    }
    auto cvars = Ship::Context::GetRawInstance()->GetConsoleVariables();
    const bool taa = cvars->GetInteger("gNewDawn.TAA.Enabled", 0) != 0;
    // Debug only (measurement fixture): a deterministic constant-speed sideways camera pan in pixels per rendered frame
    // (gNewDawn.TAA.DebugPan). It is a real part of the camera as far as TAA is concerned, and works with TAA off too.
    const float pan = cvars->GetFloat("gNewDawn.TAA.DebugPan", 0.0f);
    if (!taa && pan == 0.0f) {
        return;
    }
    const float (*P)[4] = mRsp->P_matrix;
    if (P[0][3] == 0.0f && P[1][3] == 0.0f && P[2][3] == 0.0f) {
        return; // not a perspective camera (nothing sensible to reproject)
    }
    const float xScale = AdjXForAspectRatio(1.0f);
    const float ndcXScale = xScale > 1e-6f ? 1.0f / xScale : 1.0f;
    const float flipY = mRapi->GetClipParameters().invertY ? -1.0f : 1.0f;
    const float w = (float)std::max<int>(mCurDimensions.width, 1), h = (float)std::max<int>(mCurDimensions.height, 1);
    const uint32_t frame = mTaaIndex++;
    // Pixel offsets become clip-space shifts in the projection's own x/y (the mapping the post passes use to
    // unproject).
    const float panPx = pan * (float)frame;
    const float panX = panPx * 2.0f / w * ndcXScale;
    float jitterX = 0.0f, jitterY = 0.0f, jitterPxX = 0.0f, jitterPxY = 0.0f;
    if (taa) {
        // 16-sample Halton(2,3), centred on the pixel: offsets within (-0.5, 0.5) pixels.
        const uint32_t i = (frame % 16) + 1;
        jitterPxX = Halton(i, 2) - 0.5f;
        jitterPxY = Halton(i, 3) - 0.5f;
        jitterX = jitterPxX * 2.0f / w * ndcXScale;
        jitterY = jitterPxY * 2.0f / h * flipY;
        // The camera the renderer reprojects with: loaded before this marker, so still the current P; the pan is part
        // of it, the jitter is not.
        float vp[16];
        for (int r = 0; r < 4; r++) {
            vp[r * 4 + 0] = P[r][0] + panX * P[r][3];
            vp[r * 4 + 1] = P[r][1];
            vp[r * 4 + 2] = P[r][2];
            vp[r * 4 + 3] = P[r][3];
        }
        mRapi->SetTaaCamera(vp, ndcXScale, flipY, params.zNear, params.zFar, jitterPxX, jitterPxY);
    }
    mTaaOffset[0] = panX + jitterX;
    mTaaOffset[1] = jitterY;
    mTaaJitter = true;
    JitterProjection();
    MatrixMul(mRsp->MP_matrix, mRsp->modelview_matrix_stack[std::max<int>(mRsp->modelview_matrix_stack_size, 1) - 1],
              mRsp->P_matrix);
}

void Interpreter::GfxSpPopMatrix(uint32_t count) {
    while (count--) {
        if (mRsp->modelview_matrix_stack_size > 0) {
            --mRsp->modelview_matrix_stack_size;
            if (mRsp->modelview_matrix_stack_size > 0) {
                MatrixMul(mRsp->MP_matrix, mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1],
                          mRsp->P_matrix);
            }
        }
    }
    mRsp->lights_changed = true;
    mRsp->light_state_revision++;
}

float Interpreter::AdjXForAspectRatio(float x) const {
    // Skip widescreen adjustment for fixed-size off-screen FBs (HUD elements,
    // small capture buffers), or those which specify a fixed aspect ratio.
    if (mFbActive && mActiveFrameBuffer != mFrameBuffers.end() &&
        (!mActiveFrameBuffer->second.resize || mActiveFrameBuffer->second.forceFixedAspect)) {
        return x;
    } else {
        return x * (4.0f / 3.0f) / ((float)mCurDimensions.width / (float)mCurDimensions.height);
    }
}

// Scale the width and height value based on the ratio of the viewport to the native size
void Interpreter::AdjustWidthHeightForScale(uint32_t& width, uint32_t& height, uint32_t nativeWidth,
                                            uint32_t nativeHeight) const {
    width = round(width * (mCurDimensions.width / (2.0f * (nativeWidth / 2))));
    height = round(height * (mCurDimensions.height / (2.0f * (nativeHeight / 2))));

    if (width == 0) {
        width = 1;
    }
    if (height == 0) {
        height = 1;
    }
}

void Interpreter::GfxSpVertex(size_t n_vertices, size_t dest_index, const F3DVtx* vertices) {
    const uint32_t smoothMtx = mSmooth && !mShadowOnly ? SmoothSyncMatrix() : 0;
    for (size_t i = 0; i < n_vertices; i++, dest_index++) {
        const F3DVtx_t* v = &vertices[i].v;
        const F3DVtx_tn* vn = &vertices[i].n;
        struct LoadedVertex* d = &mRsp->loaded_vertices[dest_index];

        if (v == nullptr) {
            return;
        }

        float x = v->ob[0] * mRsp->MP_matrix[0][0] + v->ob[1] * mRsp->MP_matrix[1][0] +
                  v->ob[2] * mRsp->MP_matrix[2][0] + mRsp->MP_matrix[3][0];
        float y = v->ob[0] * mRsp->MP_matrix[0][1] + v->ob[1] * mRsp->MP_matrix[1][1] +
                  v->ob[2] * mRsp->MP_matrix[2][1] + mRsp->MP_matrix[3][1];
        float z = v->ob[0] * mRsp->MP_matrix[0][2] + v->ob[1] * mRsp->MP_matrix[1][2] +
                  v->ob[2] * mRsp->MP_matrix[2][2] + mRsp->MP_matrix[3][2];
        float w = v->ob[0] * mRsp->MP_matrix[0][3] + v->ob[1] * mRsp->MP_matrix[1][3] +
                  v->ob[2] * mRsp->MP_matrix[2][3] + mRsp->MP_matrix[3][3];

        float world_pos[3] = { 0.0 };
        // Per-pixel lighting needs the view-space position for point lights; shadows need world positions for
        // every caster, lit or not.
        const bool per_pixel_lighting =
            (mPerPixelLightingEnabled || mRapi->HdrWorldActive()) && (mRsp->geometry_mode & G_LIGHTING) != 0;
        if ((mRsp->geometry_mode & G_LIGHTING_POSITIONAL) || per_pixel_lighting || mShadowRecording || mWindSway || mVfx ||
            mGrass || mInterior || mGiRecord || mSmooth) {
            float (*mtx)[4] = mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1];
            world_pos[0] = v->ob[0] * mtx[0][0] + v->ob[1] * mtx[1][0] + v->ob[2] * mtx[2][0] + mtx[3][0];
            world_pos[1] = v->ob[0] * mtx[0][1] + v->ob[1] * mtx[1][1] + v->ob[2] * mtx[2][1] + mtx[3][1];
            world_pos[2] = v->ob[0] * mtx[0][2] + v->ob[1] * mtx[1][2] + v->ob[2] * mtx[2][2] + mtx[3][2];
        }

        if (mShadowOnly) {
            // Coverage pass: only the world position (casters) and z/w (a branch-by-depth command may read them).
            if (mGiRecord && (mRsp->geometry_mode & G_LIGHTING) == 0) {
                d->color.r = v->cn[0]; // world GI reads the baked vertex colour of unlit room geometry
                d->color.g = v->cn[1];
                d->color.b = v->cn[2];
            }
            d->view_pos[0] = world_pos[0];
            d->view_pos[1] = world_pos[1];
            d->view_pos[2] = world_pos[2];
            d->z = z;
            d->w = w;
            d->clip_rej = 0;
            continue;
        }

        x = AdjXForAspectRatio(x);

        short U = v->tc[0] * mRsp->texture_scaling_factor.s >> 16;
        short V = v->tc[1] * mRsp->texture_scaling_factor.t >> 16;

        if (mRsp->geometry_mode & G_LIGHTING) {
            if (mRsp->lights_changed) {
                for (int i = 0; i < mRsp->current_num_lights - 1; i++) {
                    CalculateNormalDir(&mRsp->current_lights[i].l, mRsp->current_lights_coeffs[i]);
                }
                /*static const Light_t lookat_x = {{0, 0, 0}, 0, {0, 0, 0}, 0, {127, 0, 0}, 0};
                static const Light_t lookat_y = {{0, 0, 0}, 0, {0, 0, 0}, 0, {0, 127, 0}, 0};*/
                CalculateNormalDir(&mRsp->lookat[0], mRsp->current_lookat_coeffs[0]);
                CalculateNormalDir(&mRsp->lookat[1], mRsp->current_lookat_coeffs[1]);
                mRsp->lights_changed = false;
            }

            int r = mRsp->current_lights[mRsp->current_num_lights - 1].l.col[0];
            int g = mRsp->current_lights[mRsp->current_num_lights - 1].l.col[1];
            int b = mRsp->current_lights[mRsp->current_num_lights - 1].l.col[2];

            for (int i = 0; i < mRsp->current_num_lights - 1; i++) {
                float intensity = 0;
                if ((mRsp->geometry_mode & G_LIGHTING_POSITIONAL) && (mRsp->current_lights[i].p.unk3 != 0)) {
                    // Calculate distance from the light to the vertex
                    float dist_vec[3] = { mRsp->current_lights[i].p.pos[0] - world_pos[0],
                                          mRsp->current_lights[i].p.pos[1] - world_pos[1],
                                          mRsp->current_lights[i].p.pos[2] - world_pos[2] };
                    float dist_sq =
                        dist_vec[0] * dist_vec[0] + dist_vec[1] * dist_vec[1] +
                        dist_vec[2] * dist_vec[2] * 2; // The *2 comes from GLideN64, unsure of why it does it
                    float dist = sqrt(dist_sq);

                    // Transform distance vector (which acts as a direction light vector) into model's space
                    float light_model[3];
                    TransposedMatrixMul(light_model, dist_vec,
                                        mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1]);

                    // Calculate intensity for each axis using standard formula for intensity
                    float light_intensity[3];
                    for (int light_i = 0; light_i < 3; light_i++) {
                        light_intensity[light_i] = 4.0f * light_model[light_i] / dist_sq;
                        light_intensity[light_i] = std::clamp(light_intensity[light_i], -1.0f, 1.0f);
                    }

                    // Adjust intensity based on surface normal and sum up total
                    float total_intensity =
                        light_intensity[0] * vn->n[0] + light_intensity[1] * vn->n[1] + light_intensity[2] * vn->n[2];
                    total_intensity = std::clamp(total_intensity, -1.0f, 1.0f);

                    // Attenuate intensity based on attenuation values.
                    // Example formula found at https://ogldev.org/www/tutorial20/tutorial20.html
                    // Specific coefficients for MM's microcode sourced from GLideN64
                    // https://github.com/gonetz/GLideN64/blob/3b43a13a80dfc2eb6357673440b335e54eaa3896/src/gSP.cpp#L636
                    float distf = floorf(dist);
                    float attenuation = (distf * mRsp->current_lights[i].p.unk7 * 2.0f +
                                         distf * distf * mRsp->current_lights[i].p.unkE / 8.0f) /
                                            (float)0xFFFF +
                                        1.0f;
                    intensity = total_intensity / attenuation;
                } else {
                    intensity += vn->n[0] * mRsp->current_lights_coeffs[i][0];
                    intensity += vn->n[1] * mRsp->current_lights_coeffs[i][1];
                    intensity += vn->n[2] * mRsp->current_lights_coeffs[i][2];
                    intensity /= 127.0f;
                }
                if (intensity > 0.0f) {
                    r += intensity * mRsp->current_lights[i].l.col[0];
                    g += intensity * mRsp->current_lights[i].l.col[1];
                    b += intensity * mRsp->current_lights[i].l.col[2];
                }
            }

            d->color.r = r > 255 ? 255 : r;
            d->color.g = g > 255 ? 255 : g;
            d->color.b = b > 255 ? 255 : b;

            if (mRsp->geometry_mode & G_TEXTURE_GEN) {
                float dotx = 0, doty = 0;
                dotx += vn->n[0] * mRsp->current_lookat_coeffs[0][0];
                dotx += vn->n[1] * mRsp->current_lookat_coeffs[0][1];
                dotx += vn->n[2] * mRsp->current_lookat_coeffs[0][2];
                doty += vn->n[0] * mRsp->current_lookat_coeffs[1][0];
                doty += vn->n[1] * mRsp->current_lookat_coeffs[1][1];
                doty += vn->n[2] * mRsp->current_lookat_coeffs[1][2];

                dotx /= 127.0f;
                doty /= 127.0f;

                dotx = Ship::Math::clamp(dotx, -1.0f, 1.0f);
                doty = Ship::Math::clamp(doty, -1.0f, 1.0f);

                if (mRsp->geometry_mode & G_TEXTURE_GEN_LINEAR) {
                    // Not sure exactly what formula we should use to get accurate values
                    /*dotx = (2.906921f * dotx * dotx + 1.36114f) * dotx;
                    doty = (2.906921f * doty * doty + 1.36114f) * doty;
                    dotx = (dotx + 1.0f) / 4.0f;
                    doty = (doty + 1.0f) / 4.0f;*/
                    dotx = acosf(-dotx) /* M_PI */ * 0.159155f;
                    doty = acosf(-doty) /* M_PI */ * 0.159155f;
                } else {
                    dotx = (dotx + 1.0f) / 4.0f;
                    doty = (doty + 1.0f) / 4.0f;
                }

                U = (int32_t)(dotx * mRsp->texture_scaling_factor.s);
                V = (int32_t)(doty * mRsp->texture_scaling_factor.t);
            }
        } else {
            d->color.r = v->cn[0];
            d->color.g = v->cn[1];
            d->color.b = v->cn[2];
        }

        d->u = U;
        d->v = V;

        if (per_pixel_lighting) {
            d->normal[0] = vn->n[0];
            d->normal[1] = vn->n[1];
            d->normal[2] = vn->n[2];
        }
        if (per_pixel_lighting || mShadowRecording || mWindSway || mGrass || mInterior || mGiRecord || mVfx ||
            mSmooth) { // mVfx: the flame sprites are unlit XLU draws, their world position places the flame light
            d->view_pos[0] = world_pos[0];
            d->view_pos[1] = world_pos[1];
            d->view_pos[2] = world_pos[2];
        }
        if (mSmooth && dest_index < MAX_VERTICES + 4) {
            // Smooth characters: where this vertex came from and the matrix it went through.
            SmoothSlot& slot = mSmoothSlots[dest_index];
            slot.src = &vertices[i];
            slot.mtx = smoothMtx;
            slot.lit = (mRsp->geometry_mode & G_LIGHTING) != 0;
            slot.n[0] = vn->n[0];
            slot.n[1] = vn->n[1];
            slot.n[2] = vn->n[2];
            slot.worldValid = false;
        }

        // trivial clip rejection
        d->clip_rej = 0;
        if (x < -w) {
            d->clip_rej |= 1; // CLIP_LEFT
        }
        if (x > w) {
            d->clip_rej |= 2; // CLIP_RIGHT
        }
        if (y < -w) {
            d->clip_rej |= 4; // CLIP_BOTTOM
        }
        if (y > w) {
            d->clip_rej |= 8; // CLIP_TOP
        }
        // if (z < -w) d->clip_rej |= 16; // CLIP_NEAR
        if (z > w) {
            d->clip_rej |= 32; // CLIP_FAR
        }

        d->x = x;
        d->y = y;
        d->z = z;
        d->w = w;

        if (mRsp->geometry_mode & G_FOG) {
            if (fabsf(w) < 0.001f) {
                // To avoid division by zero
                w = 0.001f;
            }

            float winv = 1.0f / w;
            if (winv < 0.0f) {
                winv = std::numeric_limits<int16_t>::max();
            }

            float fog_z = z * winv * mRsp->fog_mul + mRsp->fog_offset;
            fog_z = Ship::Math::clamp(fog_z, 0.0f, 255.0f);
            d->color.a = fog_z; // Use alpha variable to store fog factor
        } else {
            d->color.a = v->cn[3];
        }
    }
}

void Interpreter::GfxSpModifyVertex(uint16_t vtx_idx, uint8_t where, uint32_t val) {
    SUPPORT_CHECK(where == G_MWO_POINT_ST);

    int16_t s = (int16_t)(val >> 16);
    int16_t t = (int16_t)val;

    LoadedVertex* v = &mRsp->loaded_vertices[vtx_idx];
    v->u = s;
    v->v = t;
}

void Interpreter::SyncViewportScissor() {
    if (mRdp->viewport_or_scissor_changed) {
        if (memcmp(&mRdp->viewport, &mRenderingState.viewport, sizeof(mRdp->viewport)) != 0) {
            Flush();
            mRapi->SetViewport(mRdp->viewport.x, mRdp->viewport.y, mRdp->viewport.width, mRdp->viewport.height);
            mRenderingState.viewport = mRdp->viewport;
        }
        if (memcmp(&mRdp->scissor, &mRenderingState.scissor, sizeof(mRdp->scissor)) != 0) {
            Flush();
            mRapi->SetScissor(mRdp->scissor.x, mRdp->scissor.y, mRdp->scissor.width, mRdp->scissor.height);
            mRenderingState.scissor = mRdp->scissor;
        }
        mRdp->viewport_or_scissor_changed = false;
    }

}

// The opaque, depth-writing draws that cast shadows (render state only).
static inline bool OpaqueCasterMode(uint32_t om) {
    return (om & Z_UPD) == Z_UPD && (om & ZMODE_DEC) != ZMODE_DEC && (om & CVG_X_ALPHA) != CVG_X_ALPHA &&
           (om & (3U << G_MDSFT_ALPHACOMPARE)) != G_AC_THRESHOLD && (om & G_ZS_PRIM) == 0 &&
           !((om & (3 << 24)) == (G_BL_0 << 24) && (om & (3 << 20)) == (G_BL_CLR_MEM << 20));
}

void Interpreter::BeginShadowOnly() {
    mShadowOnlyRsp = *mRsp;
    mShadowOnlyRdp = *mRdp;
    memcpy(mShadowOnlySegments, mSegmentPointers, sizeof(mSegmentPointers));
    mShadowOnlyMatrixLoads = mMatrixLoadsSinceLights;
    mShadowOnly = true;
    mShadowOnlyStart = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void Interpreter::EndShadowOnly() {
    if (!mShadowOnly) {
        return;
    }
    *mRsp = mShadowOnlyRsp;
    *mRdp = mShadowOnlyRdp;
    memcpy(mSegmentPointers, mShadowOnlySegments, sizeof(mSegmentPointers));
    mMatrixLoadsSinceLights = mShadowOnlyMatrixLoads;
    mShadowOnly = false;
    mShadowOnlyGi = false;
    mShadowOnlyNs += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - mShadowOnlyStart;
}

void Interpreter::GfxSpTri1(uint8_t vtx1_idx, uint8_t vtx2_idx, uint8_t vtx3_idx, bool is_rect) {
    const uint64_t smoothKey = mSmoothKey; // the command this triangle belongs to (smooth characters), used once
    mSmoothKey = 0;
    // New Dawn world GI: solid, depth-writing room triangles, taken before any rejection (and before the MM3D swap:
    // the N64 room stays the right shape for bounce light). Render state only; nothing is drawn differently.
    if (mGiRecord && !is_rect) {
        const uint32_t om = mRdp->other_mode_l;
        bool solid = (om & Z_UPD) == Z_UPD && (om & ZMODE_DEC) != ZMODE_DEC && (om & G_ZS_PRIM) == 0 &&
            !((om & (3 << 24)) == (G_BL_0 << 24) && (om & (3 << 20)) == (G_BL_CLR_MEM << 20));
        // Rooms draw many plain floors and walls in the texture-edge modes too. They count when the texture is (nearly)
        // opaque; real cut-outs (fences, leaves, railings) would become solid slabs, so they are left out.
        if (solid && ((om & CVG_X_ALPHA) == CVG_X_ALPHA || (om & (3U << G_MDSFT_ALPHACOMPARE)) == G_AC_THRESHOLD) &&
            AlphaCombinerUsesTexel0(mRdp->combine_mode,
                                    (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE)) == G_CYC_2CYCLE)) {
            float average[4];
            GiTextureAverage(mRdp->first_tile_index, average);
            solid = average[3] >= 0.75f;
        }
        if (solid) {
            RecordGiTriangle(&mRsp->loaded_vertices[vtx1_idx], &mRsp->loaded_vertices[vtx2_idx],
                             &mRsp->loaded_vertices[vtx3_idx]);
        }
        if (mGiDebug) { // gNewDawn.GI.WorldLog=1: every distinct room draw state once, with the albedo it was given
            const auto& lt = mRdp->loaded_texture[mRdp->texture_tile[mRdp->first_tile_index].tmem_index];
            const std::string name = lt.raw_tex_metadata.resource ? lt.raw_tex_metadata.resource->GetInitData()->Path : "-";
            static std::unordered_set<std::string> seen;
            if (seen.insert(name + std::to_string(om) + std::to_string(mRdp->combine_mode)).second) {
                const float* t = solid && mGiTris.size() >= NEWDAWN_GI_TRI_FLOATS
                                     ? &mGiTris[mGiTris.size() - NEWDAWN_GI_TRI_FLOATS] : nullptr;
                SPDLOG_INFO("GIDBG {} tex={} omL={:x} cc={:x} geo={:x} albedo={:.2f},{:.2f},{:.2f}", solid ? "solid" : "SKIP",
                            name, om, mRdp->combine_mode, mRsp->geometry_mode, t ? t[3] : 0, t ? t[4] : 0, t ? t[5] : 0);
            }
        }
    }
    if (mShadowOnly) {
        // Culled room entry (coverage pass): record it as a caster, never draw it.
        if (!is_rect && mShadowRecording && !mShadowOnlyGi && OpaqueCasterMode(mRdp->other_mode_l)) {
            RecordShadowTriangle(&mRsp->loaded_vertices[vtx1_idx], &mRsp->loaded_vertices[vtx2_idx],
                                 &mRsp->loaded_vertices[vtx3_idx]);
            mShadowOnlyTris++;
        }
        return;
    }
    if (mMM3DSuppress && !is_rect) {
        return; // Preserve state/vertices/rectangles, suppress only room triangles and their old casters.
    }
    struct LoadedVertex* v1 = &mRsp->loaded_vertices[vtx1_idx];
    struct LoadedVertex* v2 = &mRsp->loaded_vertices[vtx2_idx];
    struct LoadedVertex* v3 = &mRsp->loaded_vertices[vtx3_idx];
    struct LoadedVertex* v_arr[3] = { v1, v2, v3 };

    // New Dawn smooth characters: the triangle's cached neighbour data, or (first sight) a note to work it out when
    // the actor is complete. Taken before any rejection: culled and off-screen triangles are part of the mesh too.
    const uint8_t smoothIdx[3] = { vtx1_idx, vtx2_idx, vtx3_idx };
    const SmoothTri* smoothTri = mSmooth && smoothKey != 0 && !is_rect ? SmoothLookup(smoothKey, smoothIdx) : nullptr;

    // if (rand()%2) return;

    // Opaque world/actor casters only: skip rects, decals, alpha-tested/cutout foliage and effects. This runs before
    // frustum rejection and culling: a caster outside the view, or facing away from the camera, still shadows the
    // visible ground. All of these conditions are render-state only.
    if (mShadowRecording && !is_rect) {
        if (OpaqueCasterMode(mRdp->other_mode_l)) {
            RecordShadowTriangle(v1, v2, v3);
        }
    }

    // New Dawn: cut-out (texture-edge) geometry also casts, using its texel alpha. Like the opaque casters it counts
    // when off-screen or back-facing, so those triangles continue as far as the texture lookup and stop there.
    bool foliageCaster = false;
    if (mShadowFoliage && mShadowRecording && !is_rect) {
        const uint32_t om = mRdp->other_mode_l;
        const bool blendsMemory = ((om & (3 << 20)) == (G_BL_CLR_MEM << 20) && (om & (3 << 16)) == (G_BL_1MA << 16)) ||
                                  ((om & (3 << 22)) == (G_BL_CLR_MEM << 22) && (om & (3 << 18)) == (G_BL_1MA << 18));
        foliageCaster = (om & Z_UPD) == Z_UPD && (om & ZMODE_DEC) != ZMODE_DEC && (om & CVG_X_ALPHA) == CVG_X_ALPHA &&
                        (om & G_ZS_PRIM) == 0 && !blendsMemory &&
                        !((om & (3 << 24)) == (G_BL_0 << 24) && (om & (3 << 20)) == (G_BL_CLR_MEM << 20));
    }
    bool rejected = false;

    // New Dawn grass: ground triangles are kept even when off-screen (their tufts can reach into view).
    if (mGrass && !is_rect) {
        const auto& lt = mRdp->loaded_texture[mRdp->texture_tile[mRdp->first_tile_index].tmem_index];
        const uint32_t grassOm = mRdp->other_mode_l;
        // Opaque ground only: not cut-out foliage, alpha-blended or decal draws.
        if (lt.grassId != 0 && (grassOm & G_ZS_PRIM) == 0 && (grassOm & CVG_X_ALPHA) == 0 &&
            (grassOm & (3U << G_MDSFT_ALPHACOMPARE)) != G_AC_THRESHOLD && (grassOm & ZMODE_DEC) != ZMODE_DEC &&
            !((grassOm & (3 << 20)) == (G_BL_CLR_MEM << 20) && (grassOm & (3 << 16)) == (G_BL_1MA << 16))) {
            if (mGrassDebug) {
                // gNewDawn.Grass.Debug=1 logs every recorded texture once; gNewDawn.Grass.Only=<substring> isolates one.
                const std::string name = lt.raw_tex_metadata.resource->GetInitData()->Path;
                static std::unordered_set<std::string> seenGrass;
                if (seenGrass.insert(name).second) {
                    SPDLOG_INFO("GRASSDBG texture {}", name);
                }
                const std::string only =
                    Ship::Context::GetRawInstance()->GetConsoleVariables()->GetString("gNewDawn.Grass.Only", "");
                if (!only.empty() && name.find(only) == std::string::npos) {
                    goto skipGrass;
                }
            }
            RecordGrassTriangle(v_arr);
        }
        skipGrass:;
    }

    // New Dawn interior polish: windows and emissive textures. Kept even when off-screen (their light reaches into view).
    mBufMarkNext = 0;
    if (mInterior && !is_rect) {
        const auto& lt = mRdp->loaded_texture[mRdp->texture_tile[mRdp->first_tile_index].tmem_index];
        if (mInteriorDebug && lt.raw_tex_metadata.resource != nullptr) {
            static std::unordered_set<std::string> seenInterior;
            const std::string name = lt.raw_tex_metadata.resource->GetInitData()->Path;
            if (seenInterior.insert(name + "|" + std::to_string(mRdp->other_mode_l)).second) {
                const float* a = v1->view_pos;
                const float* b = v2->view_pos;
                const float* c = v3->view_pos;
                const float e1[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, e2[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
                float n[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
                const float nl = std::max(std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]), 1e-3f);
                SPDLOG_INFO("INTDBG draw {} omL {:x} cc {:x} prim {},{},{},{} env {},{},{},{} at {:.0f},{:.0f},{:.0f} n {:.2f},{:.2f},{:.2f} cullBack {}",
                            name, mRdp->other_mode_l, mRdp->combine_mode, mRdp->prim_color.r, mRdp->prim_color.g,
                            mRdp->prim_color.b, mRdp->prim_color.a, mRdp->env_color.r, mRdp->env_color.g,
                            mRdp->env_color.b, mRdp->env_color.a, (a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3,
                            (a[2] + b[2] + c[2]) / 3, n[0] / nl, n[1] / nl, n[2] / nl,
                            (mRsp->geometry_mode & get_attr(CULL_BACK)) != 0);
            }
        }
        if (mInteriorDebug && lt.raw_tex_metadata.resource != nullptr) {
            // Room extents (tools for placing the per-scene window table): every vertex of scene-bank draws.
            const std::string& path = lt.raw_tex_metadata.resource->GetInitData()->Path;
            if (path.find("scenes/") != std::string::npos) {
                for (int i = 0; i < 3; i++) {
                    for (int a = 0; a < 3; a++) {
                        mInteriorBBox[a] = std::min(mInteriorBBox[a], v_arr[i]->view_pos[a]);
                        mInteriorBBox[3 + a] = std::max(mInteriorBBox[3 + a], v_arr[i]->view_pos[a]);
                    }
                }
            }
        }
        if (lt.interiorIdx != 0) {
            const uint8_t kind = kInteriorTextures[lt.interiorIdx - 1].kind;
            RecordInteriorTriangle(v_arr, lt.interiorIdx);
            if (kind == INTERIOR_GLASS) {
                mBufMarkNext = 4;
            } else if (kind == INTERIOR_LAMP) {
                mBufMarkNext = 8;
            }
        }
    }

    if (v1->clip_rej & v2->clip_rej & v3->clip_rej) {
        // The whole triangle lies outside the visible area
        if (!foliageCaster) {
            return;
        }
        rejected = true;
    }

    mVfxDim = 1.0f;
    if (mVfx && !is_rect) {
        mVfxTriSerial++;
        const uint32_t t0 = mRdp->first_tile_index;
        const auto& lt0 = mRdp->loaded_texture[mRdp->texture_tile[t0].tmem_index];
        const auto& lt1 = mRdp->loaded_texture[mRdp->texture_tile[t0 + 1 < 8 ? t0 + 1 : t0].tmem_index];
        const uint8_t raw = lt0.vfxClass; // tile 1 may hold a stale texture the combiner never samples
        uint8_t cls = raw & 15;
        const float frame = (raw >> 4) / 7.0f;
        if (cls == VFX_GLOW && mRdp->prim_color.a > 80) {
            cls = VFX_NONE; // only Lights_DrawGlow's alpha-50 discs; other users of the texture keep their draw
        }
        if (mVfxDebug) {
            static std::unordered_set<std::string> seenDraws;
            const auto name = [](const auto& lt) {
                return lt.raw_tex_metadata.resource ? lt.raw_tex_metadata.resource->GetInitData()->Path : std::string("-");
            };
            const std::string key = name(lt0) + "|" + std::to_string(mRdp->other_mode_l);
            const std::string n0 = name(lt0);
            if (Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Vfx.DebugAll", 0)
                    ? (n0.find("scenes/") == std::string::npos && n0.find("object_link") == std::string::npos)
                    : (n0.find("gEff") != std::string::npos || n0.find("Fire") != std::string::npos ||
                       n0.find("Flame") != std::string::npos || n0.find("Spark") != std::string::npos ||
                       n0.find("Smoke") != std::string::npos || n0.find("Dust") != std::string::npos ||
                       n0.find("Bomb") != std::string::npos || n0.find("Glow") != std::string::npos)) {
                if (seenDraws.insert(key).second)
                    SPDLOG_INFO("VFXDBG draw tex0={} omL={:x} cc={:x} prim={},{},{},{} env={},{},{},{} w={:.0f}",
                                name(lt0), mRdp->other_mode_l, mRdp->combine_mode, mRdp->prim_color.r,
                                mRdp->prim_color.g, mRdp->prim_color.b, mRdp->prim_color.a, mRdp->env_color.r,
                                mRdp->env_color.g, mRdp->env_color.b, mRdp->env_color.a, v1->w);
            }
        }
        if (cls >= VFX_ARROW_FIRE && cls <= VFX_ARROW_LIGHT) {
            RecordVfxArrow(v_arr, cls);
            if (mRapi->VfxReplacing()) {
                return;
            }
        } else if (cls == VFX_WATER_SPLASH || cls == VFX_WATER_RIPPLE) {
            mVfxDim = cls == VFX_WATER_SPLASH ? 0.5f : 0.8f;
        } else if (cls != VFX_NONE) {
            const bool candle = cls == VFX_CANDLE;
            RecordVfxTriangle(v_arr, candle ? (uint8_t)VFX_FLAME : cls, frame);
            if (candle && mVfxLastTri == mVfxTriSerial) {
                mVfxInst[mVfxInst.size() - NEWDAWN_VFX_FLOATS + 11] = 1.0f;
            }
            if (mRapi->VfxReplacing()) {
                return; // the renderer draws this effect as an HD sprite
            }
        }
    }

    const uint32_t cull_both = get_attr(CULL_BOTH);
    const uint32_t cull_front = get_attr(CULL_FRONT);
    const uint32_t cull_back = get_attr(CULL_BACK);

    if ((mRsp->geometry_mode & cull_both) != 0) {
        float dx1 = v1->x / (v1->w) - v2->x / (v2->w);
        float dy1 = v1->y / (v1->w) - v2->y / (v2->w);
        float dx2 = v3->x / (v3->w) - v2->x / (v2->w);
        float dy2 = v3->y / (v3->w) - v2->y / (v2->w);
        float cross = dx1 * dy2 - dy1 * dx2;

        if ((v1->w < 0) ^ (v2->w < 0) ^ (v3->w < 0)) {
            // If one vertex lies behind the eye, negating cross will give the correct result.
            // If all vertices lie behind the eye, the triangle will be rejected anyway.
            cross = -cross;
        }

        // G_EX_INVERT_CULLING is a LUS extension, not tied to a specific ucode,
        // so apply it regardless of the active microcode handler.
        if ((mRsp->extra_geometry_mode & G_EX_INVERT_CULLING) != 0) {
            cross = -cross;
        }

        auto cull_type = mRsp->geometry_mode & cull_both;

        if (cull_type == cull_front) {
            rejected |= cross <= 0;
        } else if (cull_type == cull_back) {
            rejected |= cross >= 0;
        } else if (cull_type == cull_both) {
            // Why is this even an option?
            rejected = true;
        }
        if (rejected && !foliageCaster) {
            return;
        }
    }

    // depth_test is set when the fragment has a depth value to compare (either from vertex Z via
    // RSP G_ZBUFFER, or from the prim-depth register via G_ZS_PRIM) and Z_CMP is requested.
    bool zbuffer_enabled = (mRsp->geometry_mode & G_ZBUFFER) == G_ZBUFFER;
    bool prim_depth_enabled = (mRdp->other_mode_l & G_ZS_PRIM) != 0;
    bool depth_test = (zbuffer_enabled || prim_depth_enabled) && (mRdp->other_mode_l & Z_CMP) == Z_CMP;
    bool depth_mask = (mRdp->other_mode_l & Z_UPD) == Z_UPD;
    uint8_t depth_test_and_mask = (depth_test ? 1 : 0) | (depth_mask ? 2 : 0);
    if (depth_test_and_mask != mRenderingState.depth_test_and_mask) {
        Flush();
        mRapi->SetDepthTestAndMask(depth_test, depth_mask);
        mRenderingState.depth_test_and_mask = depth_test_and_mask;
    }

    bool zmode_decal = (mRdp->other_mode_l & ZMODE_DEC) == ZMODE_DEC;
    if (zmode_decal != mRenderingState.decal_mode) {
        Flush();
        mRapi->SetZmodeDecal(zmode_decal);
        mRenderingState.decal_mode = zmode_decal;
    }

    SyncViewportScissor();

    uint64_t cc_id = mRdp->combine_mode;
    uint64_t cc_options = 0;
    bool use_alpha = ((mRdp->other_mode_l & (3 << 20)) == (G_BL_CLR_MEM << 20) &&
                      (mRdp->other_mode_l & (3 << 16)) == (G_BL_1MA << 16)) ||
                     ((mRdp->other_mode_l & (3 << 22)) == (G_BL_CLR_MEM << 22) &&
                      (mRdp->other_mode_l & (3 << 18)) == (G_BL_1MA << 18));
    uint8_t blend_src = mRdp->other_mode_l >> 30;
    bool use_blend_color = blend_src == G_BL_CLR_BL;
    bool use_fog = blend_src == G_BL_CLR_FOG || use_blend_color;
    bool texture_edge = (mRdp->other_mode_l & CVG_X_ALPHA) == CVG_X_ALPHA;
    bool use_noise = (mRdp->other_mode_l & (3U << G_MDSFT_ALPHACOMPARE)) == G_AC_DITHER;
    bool use_2cyc = (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE)) == G_CYC_2CYCLE;
    bool alpha_threshold = (mRdp->other_mode_l & (3U << G_MDSFT_ALPHACOMPARE)) == G_AC_THRESHOLD;
    bool invisible =
        (mRdp->other_mode_l & (3 << 24)) == (G_BL_0 << 24) && (mRdp->other_mode_l & (3 << 20)) == (G_BL_CLR_MEM << 20);
    bool use_grayscale = mRdp->grayscale;
    bool use_prim_depth = (mRdp->other_mode_l & G_ZS_PRIM) != 0;

    if (texture_edge) {
        if (use_alpha) {
            alpha_threshold = true;
            texture_edge = false;
        }
        use_alpha = true;
    }

    if (use_alpha) {
        cc_options |= SHADER_OPT(ALPHA);
    }
    if (use_fog) {
        cc_options |= SHADER_OPT(FOG);
    }
    // New Dawn: let the MSAA coverage unit resolve the cutout edge instead of discarding. Same 0.19 threshold, but the
    // alpha ramp across a pixel becomes partial coverage, and blending is off (coverage replaces it).
    const bool alpha_to_coverage = texture_edge && mAlphaToCoverageEnabled && mMsaaLevel > 1 && mRendersToFb &&
                                   !mFbActive;
    if (texture_edge) {
        cc_options |= SHADER_OPT(TEXTURE_EDGE);
    }
    if (alpha_to_coverage) {
        cc_options |= SHADER_OPT(ALPHA_TO_COVERAGE);
    }
    if (use_noise) {
        cc_options |= SHADER_OPT(NOISE);
    }
    if (use_2cyc) {
        cc_options |= SHADER_OPT(_2CYC);
    }
    if (alpha_threshold) {
        cc_options |= SHADER_OPT(ALPHA_THRESHOLD);
    }
    if (invisible) {
        cc_options |= SHADER_OPT(INVISIBLE);
    }
    if (use_grayscale) {
        cc_options |= SHADER_OPT(GRAYSCALE);
    }
    if (use_prim_depth) {
        cc_options |= SHADER_OPT(PRIM_DEPTH);
    }

    // New Dawn: replace the CPU vertex-shaded colour with a per-fragment light calculation.
    const bool per_pixel_lighting =
        (mPerPixelLightingEnabled || mRapi->HdrWorldActive()) && (mRsp->geometry_mode & G_LIGHTING) != 0;
    if (per_pixel_lighting) {
        cc_options |= SHADER_OPT(PER_PIXEL_LIGHTING);
    }
    // Smooth characters draw patches with two more attributes. Swaying cut-outs keep their own (moved) triangles.
    if (smoothTri != nullptr && (!per_pixel_lighting || (mWindSway && (mRdp->other_mode_l & CVG_X_ALPHA) == CVG_X_ALPHA))) {
        smoothTri = nullptr;
    }
    if (smoothTri != nullptr) {
        cc_options |= SHADER_OPT(SMOOTH);
    }

    if (!mShaderStack.empty()) {
        cc_options |= (mShaderStack.top() << SHADER_ID_SHIFT);
    } else {
        cc_options |= -1 << SHADER_ID_SHIFT;
    }

    if (mRdp->loaded_texture[0].masked) {
        cc_options |= SHADER_OPT(TEXEL0_MASK);
    }
    if (mRdp->loaded_texture[1].masked) {
        cc_options |= SHADER_OPT(TEXEL1_MASK);
    }
    if (mRdp->loaded_texture[0].blended) {
        cc_options |= SHADER_OPT(TEXEL0_BLEND);
    }
    if (mRdp->loaded_texture[1].blended) {
        cc_options |= SHADER_OPT(TEXEL1_BLEND);
    }

    ColorCombinerKey key;
    key.combine_mode = mRdp->combine_mode;
    key.options = cc_options;

    ColorCombiner* comb = LookupOrCreateColorCombiner(key);

    const bool materialCandidate =
        mMaterialsEnabled && per_pixel_lighting && !is_rect && !use_alpha && !texture_edge && !zmode_decal &&
        !use_grayscale && !use_prim_depth && (mRsp->geometry_mode & G_TEXTURE_GEN) == 0 && mRendersToFb && !mFbActive &&
        comb->usedTextures[0] && !mRdp->loaded_texture[0].masked && !mRdp->loaded_texture[1].masked &&
        !mRdp->loaded_texture[0].blended && !mRdp->loaded_texture[1].blended && mShaderStack.empty();

    uint32_t tm = 0;
    uint32_t tex_width[2], tex_height[2], tex_width2[2], tex_height2[2];
    uint32_t effective_tile[2];

    for (int i = 0; i < 2; i++) {
        uint32_t tile = mRdp->first_tile_index + i;

        // No LOD support: force both slots to the base mip level.
        if (i == 1 && mRdp->first_tile_index >= 2) {
            tile = mRdp->first_tile_index;
        }
        effective_tile[i] = tile;

        if (comb->usedTextures[i]) {
            if (mRdp->textures_changed[i]) {
                Flush();
                ImportTexture(i, tile, false);
                UpdateTextureCacheSize(i);
                if (mRdp->loaded_texture[i].masked) {
                    ImportTextureMask(SHADER_FIRST_MASK_TEXTURE + i, tile);
                    UpdateTextureCacheSize(SHADER_FIRST_MASK_TEXTURE + i);
                }
                if (mRdp->loaded_texture[i].blended) {
                    ImportTexture(SHADER_FIRST_REPLACEMENT_TEXTURE + i, tile, true);
                    UpdateTextureCacheSize(SHADER_FIRST_REPLACEMENT_TEXTURE + i);
                }
                mRdp->textures_changed[i] = false;
            }

            uint8_t cms = mRdp->texture_tile[tile].cms;
            uint8_t cmt = mRdp->texture_tile[tile].cmt;

            uint32_t loaded_line_size = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;
            uint32_t loaded_size = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
            uint32_t loaded_full_line =
                mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
            uint32_t tex_size_bytes;
            uint32_t line_size;
            if ((loaded_line_size != loaded_size || loaded_full_line != loaded_size) && loaded_line_size > 0) {
                line_size = loaded_line_size;
                tex_size_bytes = loaded_size;
            } else {
                line_size = mRdp->texture_tile[tile].line_size_bytes;
                tex_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes;
                // RGBA32: texture_tile stores TMEM-interleaved stride (half of actual DRAM stride).
                if (mRdp->texture_tile[tile].siz == G_IM_SIZ_32b) {
                    line_size *= 2;
                }
            }

            if (line_size == 0) {
                line_size = 1;
            }

            tex_height[i] = tex_size_bytes / line_size;
            switch (mRdp->texture_tile[tile].siz) {
                case G_IM_SIZ_4b:
                    line_size <<= 1;
                    break;
                case G_IM_SIZ_8b:
                    break;
                case G_IM_SIZ_16b:
                    line_size /= G_IM_SIZ_16b_LINE_BYTES;
                    break;
                case G_IM_SIZ_32b:
                    line_size /= 4; // RGBA32: 4 bytes per pixel (line_size is now actual DRAM stride)
                    break;
            }
            tex_width[i] = line_size;

            tex_width2[i] = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].uls, mRdp->texture_tile[tile].lrs);
            tex_height2[i] = GetTileSizeFromCoordinates(mRdp->texture_tile[tile].ult, mRdp->texture_tile[tile].lrt);

            // Same pyramid-like ratio gate as ImportTexture: only clamp when loaded pixels
            // are close to rendered pixels (mipmap), not when much bigger (window scroll).
            const RawTexMetadata& triMeta = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
            bool pyrLike = IsPyramidLike(tex_width[i], tex_height[i], tex_width2[i], tex_height2[i],
                                         triMeta.h_byte_scale, triMeta.v_pixel_scale);
            // Same wrap-period trim as the import paths. The >= tex_width2 guard skips a stale
            // mask left by an FB blit (the pause background), which would otherwise tile the FB.
            uint32_t maskW = mRdp->texture_tile[tile].masks;
            uint32_t maskH = mRdp->texture_tile[tile].maskt;
            if (maskW != 0 && (1u << maskW) >= tex_width2[i] && (1u << maskW) < tex_width[i]) {
                tex_width[i] = 1u << maskW;
            }
            if (maskH != 0 && (1u << maskH) >= tex_height2[i] && (1u << maskH) < tex_height[i]) {
                tex_height[i] = 1u << maskH;
            }
            if ((pyrLike || (cms & G_TX_CLAMP)) && tex_width2[i] > 0 && tex_width2[i] < tex_width[i]) {
                tex_width[i] = tex_width2[i];
            }
            if ((pyrLike || (cmt & G_TX_CLAMP)) && tex_height2[i] > 0 && tex_height2[i] < tex_height[i]) {
                tex_height[i] = tex_height2[i];
            }

            uint32_t tex_width1 = tex_width[i] << (cms & G_TX_MIRROR);
            uint32_t tex_height1 = tex_height[i] << (cmt & G_TX_MIRROR);

            if ((cms & G_TX_CLAMP) && ((cms & G_TX_MIRROR) || tex_width1 != tex_width2[i])) {
                tm |= 1 << 2 * i;
                cms &= ~G_TX_CLAMP;
            }
            if ((cmt & G_TX_CLAMP) && ((cmt & G_TX_MIRROR) || tex_height1 != tex_height2[i])) {
                tm |= 1 << 2 * i + 1;
                cmt &= ~G_TX_CLAMP;
            }

            if (mRenderingState.mTextures[i] == nullptr) {
                continue;
            }

            bool linear_filter = (mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) != G_TF_POINT;
            if (linear_filter != mRenderingState.mTextures[i]->second.linear_filter ||
                cms != mRenderingState.mTextures[i]->second.cms || cmt != mRenderingState.mTextures[i]->second.cmt) {
                Flush();

                // Set the same sampler params on the blended texture. Needed for opengl.
                if (mRdp->loaded_texture[i].blended) {
                    mRapi->SetSamplerParameters(SHADER_FIRST_REPLACEMENT_TEXTURE + i, linear_filter, cms, cmt);
                }

                mRapi->SetSamplerParameters(i, linear_filter, cms, cmt);
                mRenderingState.mTextures[i]->second.linear_filter = linear_filter;
                mRenderingState.mTextures[i]->second.cms = cms;
                mRenderingState.mTextures[i]->second.cmt = cmt;
            }

            // Lazy mip generation in SetSamplerParameters may have changed the backend footprint.
            UpdateTextureCacheSize(i);
            if (mRdp->loaded_texture[i].masked) {
                UpdateTextureCacheSize(SHADER_FIRST_MASK_TEXTURE + i);
            }
            if (mRdp->loaded_texture[i].blended) {
                UpdateTextureCacheSize(SHADER_FIRST_REPLACEMENT_TEXTURE + i);
            }
        }
    }

    // New Dawn wind: render-only sway of cut-out foliage in scoped room/actor geometry. Drawn vertices and
    // foliage shadows use the same displaced copy. Geometry, animation and collision
    // are untouched: the loaded vertices themselves are never modified.
    LoadedVertex swayed[3];
    if (mWindSway && mWindDisplace && !is_rect && comb->usedTextures[0] && mRenderingState.mTextures[0] != nullptr && tex_height[0] > 0 &&
        (mRdp->other_mode_l & CVG_X_ALPHA) == CVG_X_ALPHA && (mRdp->other_mode_l & G_ZS_PRIM) == 0 &&
        (!mWindRoom || mRdp->loaded_texture[mRdp->texture_tile[effective_tile[0]].tmem_index].swayPlant) &&
        WindTriangleUpright(v_arr, mWindRoom ? 0.75f : 0.95f)) {
        const auto cpuStart = mWindProfile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto& wind = mWindState;
        const float windT = mWindT;
        const float(*P)[4] = mRsp->P_matrix;
        const float minY = std::min({ v_arr[0]->view_pos[1], v_arr[1]->view_pos[1], v_arr[2]->view_pos[1] });
        const float maxY = std::max({ v_arr[0]->view_pos[1], v_arr[1]->view_pos[1], v_arr[2]->view_pos[1] });
        for (int i = 0; i < 3; i++) {
            swayed[i] = *v_arr[i];
            float t = v_arr[i]->v / 32.0f;
            const int shiftt = mRdp->texture_tile[effective_tile[0]].shiftt;
            if (shiftt != 0) {
                t = shiftt <= 10 ? t / (1 << shiftt) : t * (1 << (16 - shiftt));
            }
            t = (t - mRdp->texture_tile[effective_tile[0]].ult / 4.0f) / tex_height[0];
            t = (t > -0.02f && t < 1.02f) ? Ship::Math::clamp(t, 0.0f, 1.0f) : t - floorf(t);
            // Tips (top of the cut-out) move, bases stay planted.
            // ...and only above the triangle's own lowest vertex, so bases shared with neighbours stay welded.
            const float* p = v_arr[i]->view_pos;
            const float weight = (1.0f - t) * (1.0f - t) * std::clamp((p[1] - minY) / (maxY - minY), 0.0f, 1.0f);
            const float phase = (p[0] * wind.dirX + p[2] * wind.dirZ) * 0.015f - windT * 1.7f;
            const float wave = std::sin(phase) * 0.7f + std::sin(phase * 2.9f + p[1] * 0.05f) * 0.3f;
            const float amount = (0.55f + wave * 0.45f) * wind.gust * 3.5f * mWindScale * weight;
            const float flutter = std::sin(windT * 6.0f + p[0] * 0.21f + p[2] * 0.17f) * 0.35f * mWindScale * weight;
            const float d[3] = { wind.dirX * amount - wind.dirZ * flutter, -fabsf(amount) * 0.12f,
                                 wind.dirZ * amount + wind.dirX * flutter };
            swayed[i].view_pos[0] += d[0];
            swayed[i].view_pos[1] += d[1];
            swayed[i].view_pos[2] += d[2];
            float c[4];
            for (int k = 0; k < 4; k++) {
                c[k] = d[0] * P[0][k] + d[1] * P[1][k] + d[2] * P[2][k];
            }
            swayed[i].x += AdjXForAspectRatio(c[0]);
            swayed[i].y += c[1];
            swayed[i].z += c[2];
            swayed[i].w += c[3];
            v_arr[i] = &swayed[i];
        }
        if (mWindProfile) {
            mWindCpuNs += std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - cpuStart).count();
            ++mWindTriangles;
        }
    }

    if (foliageCaster && comb->usedTextures[0] && mRenderingState.mTextures[0] != nullptr &&
        !mRdp->loaded_texture[0].masked && !mRdp->loaded_texture[0].blended &&
        AlphaCombinerUsesTexel0(mRdp->combine_mode, use_2cyc)) {
        RecordShadowFoliageTriangle(v_arr, effective_tile[0], tex_width[0], tex_height[0], mRenderingState.mTextures[0]);
    }
    if (rejected) {
        return;
    }

    bool materialReady = false;
    int materialTile = 0;
    bool materialBlend = false;
    if (materialCandidate) {
        auto a = mRdp->loaded_texture[mRdp->texture_tile[effective_tile[0]].tmem_index].raw_tex_metadata.resource;
        auto b = mRdp->loaded_texture[mRdp->texture_tile[effective_tile[1]].tmem_index].raw_tex_metadata.resource;
        if (comb->materialShadeMask) {
            materialBlend = comb->materialBlend && a != b;
            // For a real two-albedo blend use that tile's UV and blend weight. LOD aliases
            // share the primary map; unrelated tiles never borrow one another's coordinates.
            materialReady = BindMaterialMaps(effective_tile[0], 0);
            if (!materialReady && comb->usedTextures[1] && a != b) {
                materialTile = 1;
                materialReady = BindMaterialMaps(effective_tile[1], 1);
            }
        }
    }
    if (materialReady) {
        key.options |= SHADER_OPT(MATERIAL_MAPS);
        if (materialTile == 1)
            key.options |= SHADER_OPT(MATERIAL_MAPS_TILE1);
        if (materialBlend)
            key.options |= SHADER_OPT(MATERIAL_MAPS_BLEND);
        // Variants have the same input/UV layout as their base. Do not move mPrevCombiner
        // away from that base: bouncing between the two would flush every material triangle.
        auto variant = mColorCombinerPool.find(key);
        if (variant == mColorCombinerPool.end()) {
            variant = mColorCombinerPool.emplace(key, ColorCombiner{}).first;
            GenerateCC(&variant->second, key);
        }
        comb = &variant->second;
    }

    struct ShaderProgram* prg = comb->prg[tm];
    if (prg == NULL) {
        comb->prg[tm] = prg =
            LookupOrCreateShaderProgram(comb->shader_id0, comb->shader_id1 | tm * SHADER_OPT(TEXEL0_CLAMP_S));
    }
    if (prg != mRenderingState.mShaderProgram) {
        Flush();
        mRapi->UnloadShader(mRenderingState.mShaderProgram);
        mRapi->LoadShader(prg);
        mRenderingState.mShaderProgram = prg;
    }
    const bool blend_enable = use_alpha && !alpha_to_coverage;
    if (blend_enable != mRenderingState.alpha_blend) {
        Flush();
        mRapi->SetUseAlpha(blend_enable);
        mRenderingState.alpha_blend = blend_enable;
    }
    if (alpha_to_coverage != mRenderingState.alpha_to_coverage) {
        Flush();
        mRapi->SetAlphaToCoverage(alpha_to_coverage);
        mRenderingState.alpha_to_coverage = alpha_to_coverage;
    }

    // New Dawn foliage light: cut-out leaves of the wind-scoped plants (room plant textures, foliage actors) get light
    // through them in the shader. The renderer holds the actor's origin and the inverse projection for depth unprojection.
    {
        // The cut-out render mode (actor foliage clears texture_edge when the combiner has alpha, room cut-outs keep it).
        const bool cutout = (mRdp->other_mode_l & CVG_X_ALPHA) == CVG_X_ALPHA;
        const bool foliage = mFoliageLight && cutout && !is_rect && mRapi->HdrWorldActive() && comb->usedTextures[0] &&
                             mRenderingState.mTextures[0] != nullptr &&
                             (!mWindRoom ||
                              mRdp->loaded_texture[mRdp->texture_tile[effective_tile[0]].tmem_index].swayPlant);
        float origin[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        if (foliage && !mWindRoom) {
            const float(*m)[4] = mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1];
            origin[0] = m[3][0], origin[1] = m[3][1], origin[2] = m[3][2], origin[3] = 1.0f;
        }
        const bool moved = foliage && (std::fabs(origin[0] - mFoliageOrigin[0]) > 0.25f ||
                                       std::fabs(origin[1] - mFoliageOrigin[1]) > 0.25f ||
                                       std::fabs(origin[2] - mFoliageOrigin[2]) > 0.25f || origin[3] != mFoliageOrigin[3]);
        if (foliage != mFoliageDrawOn || moved) {
            Flush();
            float inv[16];
            if (foliage && Invert4x4(mRsp->P_matrix, inv)) {
                const float xScale = AdjXForAspectRatio(1.0f);
                const int vp[4] = { (int)mRenderingState.viewport.x, (int)mRenderingState.viewport.y,
                                    (int)mRenderingState.viewport.width, (int)mRenderingState.viewport.height };
                mRapi->SetFoliageDraw(true, origin, inv, xScale > 1e-6f ? 1.0f / xScale : 1.0f,
                                      mRapi->GetClipParameters().invertY ? -1.0f : 1.0f, vp);
                mFoliageDrawOn = true;
                memcpy(mFoliageOrigin, origin, sizeof(origin));
            } else {
                mRapi->SetFoliageDraw(false, origin, inv, 1.0f, 1.0f, nullptr);
                mFoliageDrawOn = false;
            }
        }
    }

    // Per-pixel lighting reads the lights from uniforms, so a batch can only be drawn while the
    // light set and the shade input mapping it was built with are still current.
    if (per_pixel_lighting) {
        uint32_t shade_mask = 0;
        for (int j = 0; j < 7; j++) {
            if (comb->shader_input_mapping[0][j] == G_CCMUX_SHADE) {
                shade_mask |= 1u << j;
            }
        }
        if (mBatchLightRevision != mRsp->light_state_revision || mBatchShadeMask != shade_mask ||
            mBatchPointLightQuality != mPerPixelPointLightQuality) {
            Flush();
            UpdatePerPixelLightingState(shade_mask);
            mBatchLightRevision = mRsp->light_state_revision;
            mBatchShadeMask = shade_mask;
            mBatchPointLightQuality = mPerPixelPointLightQuality;
        }
    }

    uint8_t numInputs;
    bool usedTextures[2];

    mRapi->ShaderGetInfo(prg, &numInputs, usedTextures);

    if (mBufMarkNext != mBufMark) {
        Flush();
        mBufMark = mBufMarkNext;
    }

    struct GfxClipParameters clip_parameters = mRapi->GetClipParameters();

    float smoothNormals[3][3], smoothOffsets[3][2][3];
    if (smoothTri != nullptr) {
        SmoothSyncCamera();
        SmoothCorners(smoothTri, smoothIdx, v_arr, smoothNormals, smoothOffsets);
    }

    for (int i = 0; i < 3; i++) {
        float z = v_arr[i]->z, w = v_arr[i]->w;
        if (clip_parameters.z_is_from_0_to_1) {
            z = (z + w) / 2.0f;
        }

        mBufVbo[mBufVboLen++] = v_arr[i]->x;
        mBufVbo[mBufVboLen++] = clip_parameters.invertY ? -v_arr[i]->y : v_arr[i]->y;
        mBufVbo[mBufVboLen++] = z;
        mBufVbo[mBufVboLen++] = w;

        if (per_pixel_lighting) {
            const float* normal = smoothTri != nullptr ? smoothNormals[i] : v_arr[i]->normal;
            mBufVbo[mBufVboLen++] = normal[0];
            mBufVbo[mBufVboLen++] = normal[1];
            mBufVbo[mBufVboLen++] = normal[2];
            mBufVbo[mBufVboLen++] = v_arr[i]->view_pos[0];
            mBufVbo[mBufVboLen++] = v_arr[i]->view_pos[1];
            mBufVbo[mBufVboLen++] = v_arr[i]->view_pos[2];
            if (smoothTri != nullptr) {
                memcpy(&mBufVbo[mBufVboLen], smoothOffsets[i], 6 * sizeof(float));
                mBufVboLen += 6;
            }
        }

        for (int t = 0; t < 2; t++) {
            if (!usedTextures[t]) {
                continue;
            }
            float u = v_arr[i]->u / 32.0f;
            float v = v_arr[i]->v / 32.0f;

            uint32_t uv_tile = effective_tile[t];
            int shifts = mRdp->texture_tile[uv_tile].shifts;
            int shiftt = mRdp->texture_tile[uv_tile].shiftt;
            if (shifts != 0) {
                if (shifts <= 10) {
                    u /= 1 << shifts;
                } else {
                    u *= 1 << (16 - shifts);
                }
            }
            if (shiftt != 0) {
                if (shiftt <= 10) {
                    v /= 1 << shiftt;
                } else {
                    v *= 1 << (16 - shiftt);
                }
            }

            u -= mRdp->texture_tile[uv_tile].uls / 4.0f;
            v -= mRdp->texture_tile[uv_tile].ult / 4.0f;

            if ((mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) != G_TF_POINT) {
                // Linear filter adds 0.5f to the coordinates
                if (!is_rect) {
                    u += 0.5f;
                    v += 0.5f;
                }
            }

            mBufVbo[mBufVboLen++] = u / tex_width[t];
            mBufVbo[mBufVboLen++] = v / tex_height[t];

            bool clampS = tm & (1 << 2 * t);
            bool clampT = tm & (1 << 2 * t + 1);

            if (clampS) {
                mBufVbo[mBufVboLen++] = (tex_width2[t] - 0.5f) / tex_width[t];
            }

            if (clampT) {
                mBufVbo[mBufVboLen++] = (tex_height2[t] - 0.5f) / tex_height[t];
            }
        }

        if (use_fog) {
            if (use_blend_color) {
                // Shroud/blend mode: blend toward blend_color using fog alpha as factor
                mBufVbo[mBufVboLen++] = mRdp->blend_color.r / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->blend_color.g / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->blend_color.b / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->fog_color.a / 255.0f;
            } else {
                mBufVbo[mBufVboLen++] = mRdp->fog_color.r / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->fog_color.g / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->fog_color.b / 255.0f;
                mBufVbo[mBufVboLen++] = v_arr[i]->color.a / 255.0f; // fog factor (not alpha)
            }
        }

        if (use_grayscale) {
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.r / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.g / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.b / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.a / 255.0f; // lerp interpolation factor (not alpha)
        }

        for (int j = 0; j < numInputs; j++) {
            RGBA* color;
            RGBA tmp;
            for (int k = 0; k < 1 + (use_alpha ? 1 : 0); k++) {
                switch (comb->shader_input_mapping[k][j]) {
                        // Note: CCMUX constants and ACMUX constants used here have same value, which is why this works
                        // (except LOD fraction).
                    case G_CCMUX_PRIMITIVE:
                        color = &mRdp->prim_color;
                        break;
                    case G_CCMUX_SHADE:
                        color = &v_arr[i]->color;
                        break;
                    case G_CCMUX_ENVIRONMENT:
                        color = &mRdp->env_color;
                        break;
                    case G_CCMUX_PRIMITIVE_ALPHA: {
                        tmp.r = tmp.g = tmp.b = mRdp->prim_color.a;
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_ENV_ALPHA: {
                        tmp.r = tmp.g = tmp.b = mRdp->env_color.a;
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_PRIM_LOD_FRAC: {
                        tmp.r = tmp.g = tmp.b = mRdp->prim_lod_fraction;
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_LOD_FRACTION: {
                        if (mRdp->other_mode_l & G_TL_LOD) {
                            // "Hack" that works for Bowser - Peach painting
                            float distance_frac = (v1->w - 3000.0f) / 3000.0f;
                            if (distance_frac < 0.0f) {
                                distance_frac = 0.0f;
                            }
                            if (distance_frac > 1.0f) {
                                distance_frac = 1.0f;
                            }
                            tmp.r = tmp.g = tmp.b = tmp.a = distance_frac * 255.0f;
                        } else {
                            tmp.r = tmp.g = tmp.b = tmp.a = 255.0f;
                        }
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_KEY_CENTER:
                        color = &mRdp->key_center;
                        break;
                    case G_CCMUX_KEY_SCALE:
                        color = &mRdp->key_scale;
                        break;
                    case G_CCMUX_CONVERT_K4: {
                        tmp.r = tmp.g = tmp.b = mRdp->convert_k[4];
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_CONVERT_K5: {
                        tmp.r = tmp.g = tmp.b = mRdp->convert_k[5];
                        color = &tmp;
                        break;
                    }
                    case G_ACMUX_PRIM_LOD_FRAC:
                        tmp.a = mRdp->prim_lod_fraction;
                        color = &tmp;
                        break;
                    default:
                        memset(&tmp, 0, sizeof(tmp));
                        color = &tmp;
                        break;
                }
                if (k == 0) {
                    const float dim = color == &mRdp->prim_color ? mVfxDim : 1.0f;
                    mBufVbo[mBufVboLen++] = color->r / 255.0f * dim;
                    mBufVbo[mBufVboLen++] = color->g / 255.0f * dim;
                    mBufVbo[mBufVboLen++] = color->b / 255.0f * dim;
                } else {
                    if (use_fog && !use_blend_color && color == &v_arr[i]->color) {
                        // Shade alpha is 100% for standard fog, blend color mode preserves
                        // it since fog alpha is the blend factor
                        mBufVbo[mBufVboLen++] = 1.0f;
                    } else {
                        mBufVbo[mBufVboLen++] = color->a / 255.0f * (color == &mRdp->prim_color ? mVfxDim : 1.0f);
                    }
                }
            }
        }

        // struct RGBA *color = &v_arr[i]->color;
        // mBufVbo[mBufVboLen++] = color->r / 255.0f;
        // mBufVbo[mBufVboLen++] = color->g / 255.0f;
        // mBufVbo[mBufVboLen++] = color->b / 255.0f;
        // mBufVbo[mBufVboLen++] = color->a / 255.0f;
    }

    if (++mBufVboNumTris == MAX_TRI_BUFFER) {
        // if (++mBufVbo_num_tris == 1) {
        Flush();
    }
}

void Interpreter::GfxSpGeometryMode(uint32_t clear, uint32_t set) {
    mRsp->geometry_mode &= ~clear;
    mRsp->geometry_mode |= set;
}

void Interpreter::GfxSpExtraGeometryMode(uint32_t clear, uint32_t set) {
    mRsp->extra_geometry_mode &= ~clear;
    mRsp->extra_geometry_mode |= set;
}

void Interpreter::AdjustVIewportOrScissor(XYWidthHeight* area) {
    if (!mFbActive) {
        // Adjust the y origin based on the y-inversion for the active framebuffer
        GfxClipParameters clipParameters = mRapi->GetClipParameters();
        if (clipParameters.invertY) {
            area->y -= area->height;
        } else {
            area->y = mNativeDimensions.height - area->y;
        }

        area->width *= RATIO_X(mActiveFrameBuffer, mCurDimensions);
        area->height *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);
        area->x *= RATIO_X(mActiveFrameBuffer, mCurDimensions);
        area->y *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);

        if (!mRendersToFb || GameFbIsWindowSized()) {
            area->x += mGameWindowViewport.x;
            area->y += mGfxCurrentWindowDimensions.height - (mGameWindowViewport.y + mGameWindowViewport.height);
        }
    } else {
        area->y = mActiveFrameBuffer->second.orig_height - area->y;

        if (mActiveFrameBuffer->second.resize) {
            area->width *= RATIO_X(mActiveFrameBuffer, mCurDimensions);
            area->height *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);
            area->x *= RATIO_X(mActiveFrameBuffer, mCurDimensions);
            area->y *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);
        }
    }
}

void Interpreter::CalcAndSetViewport(const F3DVp_t* viewport) {
    // 2 bits fraction
    float width = 2.0f * viewport->vscale[0] / 4.0f;
    float height = 2.0f * viewport->vscale[1] / 4.0f;
    float x = (viewport->vtrans[0] / 4.0f) - width / 2.0f;
    float y = ((viewport->vtrans[1] / 4.0f) + height / 2.0f);

    mRdp->viewport.x = x;
    mRdp->viewport.y = y;
    mRdp->viewport.width = width;
    mRdp->viewport.height = height;

    AdjustVIewportOrScissor(&mRdp->viewport);

    mRdp->viewport_or_scissor_changed = true;
}

void Interpreter::GfxSpMovememF3dex2(uint8_t index, uint8_t offset, const void* data) {
    switch (index) {
        case F3DEX2_G_MV_VIEWPORT:
            CalcAndSetViewport((const F3DVp_t*)data);
            break;
        case F3DEX2_G_MV_LIGHT: {
            int lightidx = offset / 24 - 2;
            if (lightidx >= 0 && lightidx <= MAX_LIGHTS) { // skip lookat
                // NOTE: reads out of bounds if it is an ambient light
                memcpy(mRsp->current_lights + lightidx, data, sizeof(F3DLight));
                mRsp->light_state_revision++;
            } else if (lightidx < 0) {
                memcpy(mRsp->lookat + offset / 24, data, sizeof(F3DLight_t)); // TODO Light?
            }
            break;
        }
    }
}

void Interpreter::GfxSpMovememF3d(uint8_t index, uint8_t offset, const void* data) {
    switch (index) {
        case F3DEX_G_MV_VIEWPORT:
            CalcAndSetViewport((const F3DVp_t*)data);
            break;
        case F3DEX_G_MV_LOOKATY:
        case F3DEX_G_MV_LOOKATX:
            memcpy(mRsp->lookat + (index - F3DEX_G_MV_LOOKATY) / 2, data, sizeof(F3DLight_t));
            break;
        case F3DEX_G_MV_L0:
        case F3DEX_G_MV_L1:
        case F3DEX_G_MV_L2:
        case F3DEX_G_MV_L3:
        case F3DEX_G_MV_L4:
        case F3DEX_G_MV_L5:
        case F3DEX_G_MV_L6:
        case F3DEX_G_MV_L7:
            // NOTE: reads out of bounds if it is an ambient light
            memcpy(mRsp->current_lights + (index - F3DEX_G_MV_L0) / 2, data, sizeof(F3DLight_t));
            mRsp->light_state_revision++;
            break;
    }
}

void Interpreter::GfxSpMovewordF3dex2(uint8_t index, uint16_t offset, uintptr_t data) {
    switch (index) {
        case G_MW_NUMLIGHT:
            mRsp->current_num_lights = data / 24 + 1; // add ambient light
            mRsp->lights_changed = true;
            if (mSmooth && !mShadowOnly) {
                SmoothFinishActor(); // the previous actor is complete: work out its new triangles
                mSmoothBaseDepth = g_exec_stack.cmd_stack.size();
            }
            mMatrixLoadsSinceLights = 0;
            mRsp->light_state_revision++;
            break;
        case G_MW_FOG:
            mRsp->fog_mul = (int16_t)(data >> 16);
            mRsp->fog_offset = (int16_t)data;
            break;
        case G_MW_SEGMENT: {
            int segNumber = offset / 4;
            mSegmentPointers[segNumber] = data;
        } break;
        case G_MW_SEGMENT_INTERP: {
            int segNumber = offset % 16;
            int segIndex = offset / 16;

            if (segIndex == mInterpolationIndex)
                mSegmentPointers[segNumber] = data;
        } break;
    }
}

void Interpreter::GfxSpMovewordF3d(uint8_t index, uint16_t offset, uintptr_t data) {
    switch (index) {
        case G_MW_NUMLIGHT:
            // Ambient light is included
            // The 31st bit is a flag that lights should be recalculated
            mRsp->current_num_lights = (data - 0x80000000U) / 32;
            mRsp->lights_changed = true;
            if (mSmooth && !mShadowOnly) {
                SmoothFinishActor(); // the previous actor is complete: work out its new triangles
                mSmoothBaseDepth = g_exec_stack.cmd_stack.size();
            }
            mMatrixLoadsSinceLights = 0;
            mRsp->light_state_revision++;
            break;
        case G_MW_FOG:
            mRsp->fog_mul = (int16_t)(data >> 16);
            mRsp->fog_offset = (int16_t)data;
            break;
        case G_MW_SEGMENT: {
            int segNumber = offset / 4;
            mSegmentPointers[segNumber] = data;
        } break;
        case G_MW_SEGMENT_INTERP: {
            int segNumber = offset % 16;
            int segIndex = offset / 16;

            if (segIndex == mInterpolationIndex)
                mSegmentPointers[segNumber] = data;
        } break;
    }
}

void Interpreter::GfxSpTexture(uint16_t sc, uint16_t tc, uint8_t level, uint8_t tile, uint8_t on) {
    mRsp->texture_scaling_factor.s = sc;
    mRsp->texture_scaling_factor.t = tc;
    if (mRdp->first_tile_index != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }

    mRdp->first_tile_index = tile;
}

void Interpreter::GfxDpSetScissor(uint32_t mode, uint32_t ulx, uint32_t uly, uint32_t lrx, uint32_t lry) {
    float x = ulx / 4.0f;
    float y = lry / 4.0f;
    float width = (lrx - ulx) / 4.0f;
    float height = (lry - uly) / 4.0f;

    mRdp->scissor.x = x;
    mRdp->scissor.y = y;
    mRdp->scissor.width = width;
    mRdp->scissor.height = height;

    AdjustVIewportOrScissor(&mRdp->scissor);

    mRdp->viewport_or_scissor_changed = true;
}

void Interpreter::GfxDpSetTextureImage(uint32_t format, uint32_t size, uint32_t width, const char* texPath,
                                       uint32_t texFlags, RawTexMetadata rawTexMetdata, const void* addr) {
    // fprintf(stderr, "GfxDpSetTextureImage: %s (width=%d; size=0x%X)\n",
    //         rawTexMetdata.resource ? rawTexMetdata.resource->GetInitData()->Path.c_str() : nullptr, width, size);
    mRdp->texture_to_load.addr = (const uint8_t*)addr;
    mRdp->texture_to_load.siz = size;
    mRdp->texture_to_load.width = width;
    mRdp->texture_to_load.tex_flags = texFlags;
    mRdp->texture_to_load.raw_tex_metadata = rawTexMetdata;
}

void Interpreter::GfxDpSetTile(uint8_t fmt, uint32_t siz, uint32_t line, uint32_t tmem, uint8_t tile, uint32_t palette,
                               uint32_t cmt, uint32_t maskt, uint32_t shiftt, uint32_t cms, uint32_t masks,
                               uint32_t shifts) {
    // OTRTODO:
    // SUPPORT_CHECK(tmem == 0 || tmem == 256);

    if (cms == G_TX_WRAP && masks == G_TX_NOMASK) {
        cms = G_TX_CLAMP;
    }
    if (cmt == G_TX_WRAP && maskt == G_TX_NOMASK) {
        cmt = G_TX_CLAMP;
    }

    mRdp->texture_tile[tile].palette = palette; // palette should set upper 4 bits of color index in 4b mode
    mRdp->texture_tile[tile].fmt = fmt;
    mRdp->texture_tile[tile].siz = siz;
    mRdp->texture_tile[tile].cms = cms;
    mRdp->texture_tile[tile].cmt = cmt;
    mRdp->texture_tile[tile].masks = masks;
    mRdp->texture_tile[tile].maskt = maskt;
    mRdp->texture_tile[tile].shifts = shifts;
    mRdp->texture_tile[tile].shiftt = shiftt;
    mRdp->texture_tile[tile].line_size_bytes = line * 8;

    mRdp->texture_tile[tile].tmem = tmem;
    // mRdp->texture_tile[tile].tmem_index = tmem / 256; // tmem is the 64-bit word offset, so 256 words means 2 kB

    mRdp->texture_tile[tile].tmem_index =
        tmem != 0; // assume one texture is loaded at address 0 and another texture at any other address

    mRdp->textures_changed[0] = true;
    mRdp->textures_changed[1] = true;
}

void Interpreter::GfxDpSetTileSize(uint8_t tile, uint16_t uls, uint16_t ult, uint16_t lrs, uint16_t lrt) {
    mRdp->texture_tile[tile].uls = uls;
    mRdp->texture_tile[tile].ult = ult;
    mRdp->texture_tile[tile].lrs = lrs;
    mRdp->texture_tile[tile].lrt = lrt;
    mRdp->textures_changed[0] = true;
    mRdp->textures_changed[1] = true;
}

void Interpreter::GfxDpLoadTlut(uint8_t tile, uint32_t high_index) {
    SUPPORT_CHECK(mRdp->texture_to_load.siz == G_IM_SIZ_16b);

    uint16_t tmem = mRdp->texture_tile[tile].tmem;
    const uint8_t* src = mRdp->texture_to_load.addr;
    uint32_t entryCount = high_index + 1;
    uint32_t byteCount = entryCount * 2;

    if (tmem >= 256) {
        // N64 TMEM palette area starts at tmem word 256. Each CI4 palette = 16 entries = 16 tmem words.
        uint32_t paletteByteOffset = (tmem - 256) * 2;

        if (high_index == 255 && paletteByteOffset == 0) {
            // CI8: full 256-entry palette spanning both halves
            memcpy(mRdp->palette_staging[0], src, 256);
            memcpy(mRdp->palette_staging[1], src + 256, 256);
            mRdp->palettes[0] = mRdp->palette_staging[0];
            mRdp->palettes[1] = mRdp->palette_staging[1];
            mRdp->palette_dram_addr[0] = src;
            mRdp->palette_dram_addr[1] = src + 256;
        } else if (paletteByteOffset < 256) {
            // Palettes 0-7 range
            uint32_t copyLen = (paletteByteOffset + byteCount <= 256) ? byteCount : (256 - paletteByteOffset);
            memcpy(mRdp->palette_staging[0] + paletteByteOffset, src, copyLen);
            mRdp->palettes[0] = mRdp->palette_staging[0];
            mRdp->palette_dram_addr[0] = src;
        } else {
            // Palettes 8-15 range
            uint32_t offset = paletteByteOffset - 256;
            uint32_t copyLen = (offset + byteCount <= 256) ? byteCount : (256 - offset);
            memcpy(mRdp->palette_staging[1] + offset, src, copyLen);
            mRdp->palettes[1] = mRdp->palette_staging[1];
            mRdp->palette_dram_addr[1] = src;
        }
    } else {
        // tmem < 256: non-standard location, fall back to direct pointer
        mRdp->palettes[1] = src;
        mRdp->palette_dram_addr[1] = src;
    }
}

void Interpreter::GfxDpLoadBlock(uint8_t tile, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t dxt) {
    SUPPORT_CHECK(uls == 0);
    SUPPORT_CHECK(ult == 0);

    // The lrs field rather seems to be number of pixels to load
    uint32_t word_size_shift = 0;
    switch (mRdp->texture_to_load.siz) {
        case G_IM_SIZ_4b:
            word_size_shift = -1;
            break;
        case G_IM_SIZ_8b:
            word_size_shift = 0;
            break;
        case G_IM_SIZ_16b:
            word_size_shift = 1;
            break;
        case G_IM_SIZ_32b:
            word_size_shift = 2;
            break;
    }
    uint32_t orig_size_bytes =
        word_size_shift > 0 ? (lrs + 1) << word_size_shift : (lrs + 1) >> (-(int64_t)word_size_shift);
    uint32_t size_bytes = orig_size_bytes;
    if (mRdp->texture_to_load.raw_tex_metadata.h_byte_scale != 1 ||
        mRdp->texture_to_load.raw_tex_metadata.v_pixel_scale != 1) {
        size_bytes *= mRdp->texture_to_load.raw_tex_metadata.h_byte_scale;
        size_bytes *= mRdp->texture_to_load.raw_tex_metadata.v_pixel_scale;
    }
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes = orig_size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes = size_bytes;
    // Compute actual per-line DRAM stride from SetTextureImage width when available.
    // The standard gDPLoadTextureBlock macro sets width=1, but manually-built DL
    // commands may set the real pixel width.
    uint32_t actual_line_bytes = size_bytes;
    const RawTexMetadata& blkMeta = mRdp->texture_to_load.raw_tex_metadata;
    bool blkHd = blkMeta.h_byte_scale != 1 || blkMeta.v_pixel_scale != 1;
    if (mRdp->texture_to_load.width > 1 && blkHd && blkMeta.height > 0 && size_bytes % blkMeta.height == 0) {
        // HD-upscaled textures report a sentinel SetTextureImage width, so the per-line stride
        // derived from it is bogus. The resource's stored height is authoritative.
        actual_line_bytes = size_bytes / blkMeta.height;
    } else if (mRdp->texture_to_load.width > 1) {
        uint32_t candidate;
        switch (mRdp->texture_to_load.siz) {
            case G_IM_SIZ_4b:
                candidate = (mRdp->texture_to_load.width + 1) / 2;
                break;
            case G_IM_SIZ_8b:
                candidate = mRdp->texture_to_load.width;
                break;
            case G_IM_SIZ_16b:
                candidate = mRdp->texture_to_load.width * 2;
                break;
            case G_IM_SIZ_32b:
                candidate = mRdp->texture_to_load.width * 4;
                break;
            default:
                candidate = mRdp->texture_to_load.width;
                break;
        }
        if (candidate > 0 && candidate < size_bytes && size_bytes % candidate == 0) {
            actual_line_bytes = candidate;
        }
    }
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes = actual_line_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes = actual_line_bytes;
    // assert(size_bytes <= 4096 && "bug: too big texture");
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].tex_flags = mRdp->texture_to_load.tex_flags;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata = mRdp->texture_to_load.raw_tex_metadata;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr = mRdp->texture_to_load.addr;
    // fprintf(stderr, "GfxDpLoadBlock: line_size = 0x%x; orig = 0x%x; bpp=%d; lrs=%d\n", size_bytes,
    // orig_size_bytes,
    //         mRdp->texture_to_load.siz, lrs);

    const std::string_view texPath =
        mRdp->texture_to_load.raw_tex_metadata.resource != nullptr
            ? GetBaseTexturePath(mRdp->texture_to_load.raw_tex_metadata.resource->GetInitData()->Path)
            : std::string_view{};
    auto maskedTextureIter = mMaskedTextures.find(texPath);
    if (maskedTextureIter != mMaskedTextures.end()) {
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].masked = true;
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].blended =
            maskedTextureIter->second.replacementData != nullptr;
    } else {
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].masked = false;
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].blended = false;
    }

    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].vfxClass = mVfx ? VfxClassify(texPath) : 0;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].grassId = mGrass ? GrassClassify(texPath) : 0;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].interiorIdx = mInterior ? InteriorClassify(texPath) : 0;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].swayPlant = mWindSway && mWindRoom && SwayClassify(texPath);
    mRdp->textures_changed[mRdp->texture_tile[tile].tmem_index] = true;
}

void Interpreter::GfxDpLoadTile(uint8_t tile, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t lrt) {
    SUPPORT_CHECK(tile == G_TX_LOADTILE);

    uint32_t word_size_shift = 0;
    switch (mRdp->texture_to_load.siz) {
        case G_IM_SIZ_4b:
            word_size_shift = 0;
            break;
        case G_IM_SIZ_8b:
            word_size_shift = 0;
            break;
        case G_IM_SIZ_16b:
            word_size_shift = 1;
            break;
        case G_IM_SIZ_32b:
            word_size_shift = 2;
            break;
    }

    uint32_t offset_x = uls >> G_TEXTURE_IMAGE_FRAC;
    uint32_t offset_y = ult >> G_TEXTURE_IMAGE_FRAC;
    uint32_t tile_width = ((lrs - uls) >> G_TEXTURE_IMAGE_FRAC) + 1;
    uint32_t tile_height = ((lrt - ult) >> G_TEXTURE_IMAGE_FRAC) + 1;
    uint32_t full_image_width = mRdp->texture_to_load.width;

    uint32_t offset_x_in_bytes = offset_x << word_size_shift;
    uint32_t tile_line_size_bytes = tile_width << word_size_shift;
    uint32_t full_image_line_size_bytes = full_image_width << word_size_shift;

    uint32_t orig_size_bytes = tile_line_size_bytes * tile_height;
    uint32_t size_bytes = orig_size_bytes;
    uint32_t start_offset_bytes = full_image_line_size_bytes * offset_y + offset_x_in_bytes;

    float h_byte_scale = mRdp->texture_to_load.raw_tex_metadata.h_byte_scale;
    float v_pixel_scale = mRdp->texture_to_load.raw_tex_metadata.v_pixel_scale;

    if (h_byte_scale != 1 || v_pixel_scale != 1) {
        start_offset_bytes = h_byte_scale * (v_pixel_scale * offset_y * full_image_line_size_bytes + offset_x_in_bytes);
        size_bytes *= h_byte_scale * v_pixel_scale;
        full_image_line_size_bytes *= h_byte_scale;
        tile_line_size_bytes *= h_byte_scale;
    }

    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes = orig_size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes = size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes = full_image_line_size_bytes;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes = tile_line_size_bytes;

    //    assert(size_bytes <= 4096 && "bug: too big texture");
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].tex_flags = mRdp->texture_to_load.tex_flags;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata = mRdp->texture_to_load.raw_tex_metadata;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr = mRdp->texture_to_load.addr + start_offset_bytes;

    const std::string_view texPath =
        mRdp->texture_to_load.raw_tex_metadata.resource != nullptr
            ? GetBaseTexturePath(mRdp->texture_to_load.raw_tex_metadata.resource->GetInitData()->Path)
            : std::string_view{};
    auto maskedTextureIter = mMaskedTextures.find(texPath);
    if (maskedTextureIter != mMaskedTextures.end()) {
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].masked = true;
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].blended =
            maskedTextureIter->second.replacementData != nullptr;
    } else {
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].masked = false;
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].blended = false;
    }

    mRdp->texture_tile[tile].uls = uls;
    mRdp->texture_tile[tile].ult = ult;
    mRdp->texture_tile[tile].lrs = lrs;
    mRdp->texture_tile[tile].lrt = lrt;

    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].vfxClass = mVfx ? VfxClassify(texPath) : 0;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].grassId = mGrass ? GrassClassify(texPath) : 0;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].interiorIdx = mInterior ? InteriorClassify(texPath) : 0;
    mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].swayPlant = mWindSway && mWindRoom && SwayClassify(texPath);
    mRdp->textures_changed[mRdp->texture_tile[tile].tmem_index] = true;
}

/*static uint8_t color_comb_component(uint32_t v) {
    switch (v) {
        case G_CCMUX_TEXEL0:
            return CC_TEXEL0;
        case G_CCMUX_TEXEL1:
            return CC_TEXEL1;
        case G_CCMUX_PRIMITIVE:
            return CC_PRIM;
        case G_CCMUX_SHADE:
            return CC_SHADE;
        case G_CCMUX_ENVIRONMENT:
            return CC_ENV;
        case G_CCMUX_TEXEL0_ALPHA:
            return CC_TEXEL0A;
        case G_CCMUX_LOD_FRACTION:
            return CC_LOD;
        default:
            return CC_0;
    }
}

static inline uint32_t color_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return color_comb_component(a) |
           (color_comb_component(b) << 3) |
           (color_comb_component(c) << 6) |
           (color_comb_component(d) << 9);
}

static void GfxDpSetCombineMode(uint32_t rgb, uint32_t alpha) {
    mRdp->combine_mode = rgb | (alpha << 12);
}*/

void Interpreter::GfxDpSetCombineMode(uint32_t rgb, uint32_t alpha, uint32_t rgb_cyc2, uint32_t alpha_cyc2) {
    mRdp->combine_mode = rgb | (alpha << 16) | ((uint64_t)rgb_cyc2 << 28) | ((uint64_t)alpha_cyc2 << 44);
}

static inline uint32_t color_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return (a & 0xf) | ((b & 0xf) << 4) | ((c & 0x1f) << 8) | ((d & 7) << 13);
}

static inline uint32_t alpha_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return (a & 7) | ((b & 7) << 3) | ((c & 7) << 6) | ((d & 7) << 9);
}

// Sign-extend a 9-bit value (used for G_SETCONVERT K0..K5).
static inline int16_t sign_extend_9(uint32_t v) {
    return (int16_t)((v & 0x100) ? (int32_t)(v | 0xFFFFFE00u) : (int32_t)v);
}

void Interpreter::GfxDpSetGrayscaleColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->grayscale_color.r = r;
    mRdp->grayscale_color.g = g;
    mRdp->grayscale_color.b = b;
    mRdp->grayscale_color.a = a;
}

void Interpreter::GfxDpSetEnvColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->env_color.r = r;
    mRdp->env_color.g = g;
    mRdp->env_color.b = b;
    mRdp->env_color.a = a;
}

void Interpreter::GfxDpSetPrimColor(uint8_t m, uint8_t l, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->prim_lod_fraction = l;
    mRdp->prim_color.r = r;
    mRdp->prim_color.g = g;
    mRdp->prim_color.b = b;
    mRdp->prim_color.a = a;
}

void Interpreter::GfxDpSetFogColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->fog_color.r = r;
    mRdp->fog_color.g = g;
    mRdp->fog_color.b = b;
    mRdp->fog_color.a = a;
}

void Interpreter::GfxDpSetBlendColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->blend_color.r = r;
    mRdp->blend_color.g = g;
    mRdp->blend_color.b = b;
    mRdp->blend_color.a = a;
}

void Interpreter::GfxDpSetFillColor(uint32_t packed_color) {
    uint16_t col16 = (uint16_t)packed_color;
    uint32_t r = col16 >> 11;
    uint32_t g = (col16 >> 6) & 0x1f;
    uint32_t b = (col16 >> 1) & 0x1f;
    uint32_t a = col16 & 1;
    mRdp->fill_color.r = SCALE_5_8(r);
    mRdp->fill_color.g = SCALE_5_8(g);
    mRdp->fill_color.b = SCALE_5_8(b);
    mRdp->fill_color.a = a * 255;
}

void Interpreter::GfxDrawRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry) {
    uint32_t saved_other_mode_h = mRdp->other_mode_h;
    uint32_t cycle_type = (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE));

    if (cycle_type == G_CYC_COPY) {
        mRdp->other_mode_h = (mRdp->other_mode_h & ~(3U << G_MDSFT_TEXTFILT)) | G_TF_POINT;
    }

    // U10.2 coordinates
    float ulxf = ulx;
    float ulyf = uly;
    float lrxf = lrx;
    float lryf = lry;

    ulxf = ulxf / (4.0f * HALF_SCREEN_WIDTH(mActiveFrameBuffer)) - 1.0f;
    ulyf = -(ulyf / (4.0f * HALF_SCREEN_HEIGHT(mActiveFrameBuffer))) + 1.0f;
    lrxf = lrxf / (4.0f * HALF_SCREEN_WIDTH(mActiveFrameBuffer)) - 1.0f;
    lryf = -(lryf / (4.0f * HALF_SCREEN_HEIGHT(mActiveFrameBuffer))) + 1.0f;

    ulxf = AdjXForAspectRatio(ulxf);
    lrxf = AdjXForAspectRatio(lrxf);

    struct LoadedVertex* ul = &mRsp->loaded_vertices[MAX_VERTICES + 0];
    struct LoadedVertex* ll = &mRsp->loaded_vertices[MAX_VERTICES + 1];
    struct LoadedVertex* lr = &mRsp->loaded_vertices[MAX_VERTICES + 2];
    struct LoadedVertex* ur = &mRsp->loaded_vertices[MAX_VERTICES + 3];

    ul->x = ulxf;
    ul->y = ulyf;
    ul->z = -1.0f;
    ul->w = 1.0f;

    ll->x = ulxf;
    ll->y = lryf;
    ll->z = -1.0f;
    ll->w = 1.0f;

    lr->x = lrxf;
    lr->y = lryf;
    lr->z = -1.0f;
    lr->w = 1.0f;

    ur->x = lrxf;
    ur->y = ulyf;
    ur->z = -1.0f;
    ur->w = 1.0f;

    // The coordinates for texture rectangle shall bypass the viewport setting
    struct XYWidthHeight default_viewport;
    if (!mFbActive) {
        default_viewport = { 0, (int16_t)mNativeDimensions.height, mNativeDimensions.width, mNativeDimensions.height };
    } else {
        default_viewport = { 0, (int16_t)mActiveFrameBuffer->second.orig_height, mActiveFrameBuffer->second.orig_width,
                             mActiveFrameBuffer->second.orig_height };
    }

    struct XYWidthHeight viewport_saved = mRdp->viewport;
    uint32_t geometry_mode_saved = mRsp->geometry_mode;

    AdjustVIewportOrScissor(&default_viewport);

    mRdp->viewport = default_viewport;
    mRdp->viewport_or_scissor_changed = true;
    mRsp->geometry_mode = 0;

    GfxSpTri1(MAX_VERTICES + 0, MAX_VERTICES + 1, MAX_VERTICES + 3, true);
    GfxSpTri1(MAX_VERTICES + 1, MAX_VERTICES + 2, MAX_VERTICES + 3, true);

    mRsp->geometry_mode = geometry_mode_saved;
    mRdp->viewport = viewport_saved;
    mRdp->viewport_or_scissor_changed = true;

    if (cycle_type == G_CYC_COPY) {
        mRdp->other_mode_h = saved_other_mode_h;
    }
}

void Interpreter::GfxDpTextureRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry, uint8_t tile, int16_t uls,
                                        int16_t ult, int16_t dsdx, int16_t dtdy, bool flip) {
    // printf("render %d at %d\n", tile, lrx);
    uint64_t saved_combine_mode = mRdp->combine_mode;
    if ((mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE)) == G_CYC_COPY) {
        // Per RDP Command Summary Set Tile's shift s and this dsdx should be set to 4 texels
        // Divide by 4 to get 1 instead
        dsdx >>= 2;

        // Color combiner is turned off in copy mode
        GfxDpSetCombineMode(color_comb(0, 0, 0, G_CCMUX_TEXEL0), alpha_comb(0, 0, 0, G_ACMUX_TEXEL0), 0, 0);

        // Per documentation one extra pixel is added in this modes to each edge
        lrx += 1 << 2;
        lry += 1 << 2;
    }

    // uls and ult are S10.5
    // dsdx and dtdy are S5.10
    // lrx, lry, ulx, uly are U10.2
    // lrs, lrt are S10.5
    if (flip) {
        dsdx = -dsdx;
        dtdy = -dtdy;
    }
    int16_t width = !flip ? lrx - ulx : lry - uly;
    int16_t height = !flip ? lry - uly : lrx - ulx;
    float lrs = ((uls << 7) + dsdx * width) >> 7;
    float lrt = ((ult << 7) + dtdy * height) >> 7;

    LoadedVertex* ul = &mRsp->loaded_vertices[MAX_VERTICES + 0];
    LoadedVertex* ll = &mRsp->loaded_vertices[MAX_VERTICES + 1];
    LoadedVertex* lr = &mRsp->loaded_vertices[MAX_VERTICES + 2];
    LoadedVertex* ur = &mRsp->loaded_vertices[MAX_VERTICES + 3];
    ul->u = uls;
    ul->v = ult;
    lr->u = lrs;
    lr->v = lrt;
    if (!flip) {
        ll->u = uls;
        ll->v = lrt;
        ur->u = lrs;
        ur->v = ult;
    } else {
        ll->u = lrs;
        ll->v = ult;
        ur->u = uls;
        ur->v = lrt;
    }

    uint8_t saved_tile = mRdp->first_tile_index;
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = tile;

    GfxDrawRectangle(ulx, uly, lrx, lry);
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = saved_tile;
    mRdp->combine_mode = saved_combine_mode;
}

void Interpreter::GfxDpImageRectangle(int32_t tile, int32_t w, int32_t h, int32_t ulx, int32_t uly, int16_t uls,
                                      int16_t ult, int32_t lrx, int32_t lry, int16_t lrs, int16_t lrt) {

    LoadedVertex* ul = &mRsp->loaded_vertices[MAX_VERTICES + 0];
    LoadedVertex* ll = &mRsp->loaded_vertices[MAX_VERTICES + 1];
    LoadedVertex* lr = &mRsp->loaded_vertices[MAX_VERTICES + 2];
    LoadedVertex* ur = &mRsp->loaded_vertices[MAX_VERTICES + 3];
    ul->u = uls * 32;
    ul->v = ult * 32;
    lr->u = lrs * 32;
    lr->v = lrt * 32;
    ll->u = uls * 32;
    ll->v = lrt * 32;
    ur->u = lrs * 32;
    ur->v = ult * 32;

    // ensure we have the correct texture size, format and starting position
    mRdp->texture_tile[tile].siz = G_IM_SIZ_8b;
    mRdp->texture_tile[tile].fmt = G_IM_FMT_RGBA;
    mRdp->texture_tile[tile].cms = 0;
    mRdp->texture_tile[tile].cmt = 0;
    mRdp->texture_tile[tile].shifts = 0;
    mRdp->texture_tile[tile].shiftt = 0;
    mRdp->texture_tile[tile].uls = 0 * 4;
    mRdp->texture_tile[tile].ult = 0 * 4;
    mRdp->texture_tile[tile].lrs = w * 4;
    mRdp->texture_tile[tile].lrt = h * 4;
    mRdp->texture_tile[tile].line_size_bytes = w << (mRdp->texture_tile[tile].siz >> 1);

    auto& loadtex = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index];
    loadtex.full_image_line_size_bytes = loadtex.line_size_bytes = mRdp->texture_tile[tile].line_size_bytes;
    loadtex.size_bytes = loadtex.orig_size_bytes = loadtex.line_size_bytes * h;

    uint8_t saved_tile = mRdp->first_tile_index;
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = tile;

    GfxDrawRectangle(ulx, uly, lrx, lry);
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = saved_tile;
}

void Interpreter::GfxDpFillRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry) {
    if (mRdp->color_image_address == mRdp->z_buf_address) {
        // Fullscreen Z clears are redundant — already done by glClear at frame start.
        bool isFullScreen = (ulx <= 0 && uly <= 0 && lrx >= (int32_t)(mNativeDimensions.width - 1) * 4 &&
                             lry >= (int32_t)(mNativeDimensions.height - 1) * 4);
        if (isFullScreen) {
            return;
        }

        // Partial depth clear (e.g. HUD model regions): clear the actual depth buffer
        // via a scissored depth clear instead of drawing a colored rect to the color buffer.
        Flush();

        // Convert U10.2 coords to pixel coords and add +1 pixel for fill mode
        int32_t expanded_lrx = lrx + (1 << 2);
        int32_t expanded_lry = lry + (1 << 2);
        float x = ulx / 4.0f;
        float y = expanded_lry / 4.0f;
        float w = (expanded_lrx - ulx) / 4.0f;
        float h = (expanded_lry - uly) / 4.0f;
        float halfNativeWidth = (float)HALF_SCREEN_WIDTH(mActiveFrameBuffer);
        x = halfNativeWidth + AdjXForAspectRatio(x - halfNativeWidth);
        w = AdjXForAspectRatio(w);

        struct XYWidthHeight area;
        area.x = (int16_t)x;
        area.y = (int16_t)y;
        area.width = (uint32_t)w;
        area.height = (uint32_t)h;
        AdjustVIewportOrScissor(&area);

        mRapi->ClearDepthRegion(area.x, area.y, area.width, area.height);
        return;
    }
    uint32_t mode = (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE));

    // Expand fullscreen fill rects to cover widescreen viewports.
    // Without this, screen clears and fades only cover the native 4:3 area.
    if (ulx == 0 && uly == 0) {
        bool isFullScreen = (lrx == ((int32_t)(mNativeDimensions.width - 1) * 4) &&
                             lry == ((int32_t)(mNativeDimensions.height - 1) * 4));
        if (isFullScreen) {
            ulx = -1024;
            uly = -1024;
            lrx = 2048;
            lry = 2048;
        }
    }

    if (mode == G_CYC_COPY || mode == G_CYC_FILL) {
        // Per documentation one extra pixel is added in this modes to each edge
        lrx += 1 << 2;
        lry += 1 << 2;
    }

    for (int i = MAX_VERTICES; i < MAX_VERTICES + 4; i++) {
        LoadedVertex* v = &mRsp->loaded_vertices[i];
        v->color = mRdp->fill_color;
    }

    uint64_t saved_combine_mode = mRdp->combine_mode;

    if (mode == G_CYC_FILL) {
        GfxDpSetCombineMode(color_comb(0, 0, 0, G_CCMUX_SHADE), alpha_comb(0, 0, 0, G_ACMUX_SHADE), 0, 0);
    }

    GfxDrawRectangle(ulx, uly, lrx, lry);
    mRdp->combine_mode = saved_combine_mode;
}

void Interpreter::GfxDpSetZImage(void* zBufAddr) {
    mRdp->z_buf_address = zBufAddr;
}

void Interpreter::GfxDpSetColorImage(uint32_t format, uint32_t size, uint32_t width, void* address) {
    mRdp->color_image_address = address;
}

void Interpreter::GfxSpSetOtherMode(uint32_t shift, uint32_t num_bits, uint64_t mode) {
    uint64_t mask = (((uint64_t)1 << num_bits) - 1) << shift;
    uint64_t om = mRdp->other_mode_l | ((uint64_t)mRdp->other_mode_h << 32);
    om = (om & ~mask) | mode;
    mRdp->other_mode_l = (uint32_t)om;
    mRdp->other_mode_h = (uint32_t)(om >> 32);
}

void Interpreter::GfxDpSetOtherMode(uint32_t h, uint32_t l) {
    mRdp->other_mode_h = h;
    mRdp->other_mode_l = l;
}

void Interpreter::Gfxs2dexBgCopy(F3DuObjBg* bg) {
    /*
    bg->b.imageX = 0;
    bg->b.imageW = width * 4;
    bg->b.frameX = frameX * 4;
    bg->b.imageY = 0;
    bg->b.imageH = height * 4;
    bg->b.frameY = frameY * 4;
    bg->b.imagePtr = source;
    bg->b.imageLoad = G_BGLT_LOADTILE;
    bg->b.imageFmt = fmt;
    bg->b.imageSiz = siz;
    bg->b.imagePal = 0;
    bg->b.imageFlip = 0;
    */

    uintptr_t data = (uintptr_t)bg->b.imagePtr;

    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    if ((bool)gfx_check_image_signature((char*)data)) {
        std::shared_ptr<Fast::Texture> tex = std::static_pointer_cast<Fast::Texture>(
            Ship::Context::GetRawInstance()->GetResourceManager()->LoadResourceProcess((char*)data));
        texFlags = tex->Flags;
        rawTexMetadata.width = tex->Width;
        rawTexMetadata.height = tex->Height;
        rawTexMetadata.h_byte_scale = tex->HByteScale;
        rawTexMetadata.v_pixel_scale = tex->VPixelScale;
        rawTexMetadata.type = tex->Type;
        rawTexMetadata.resource = tex;
        data = (uintptr_t)reinterpret_cast<char*>(tex->ImageData);
    }

    s16 dsdx = 4 << 10;
    s16 uls = bg->b.imageX << 3;
    // Flip flag only flips horizontally
    if (bg->b.imageFlip == G_BG_FLAG_FLIPS) {
        dsdx = -dsdx;
        uls = (bg->b.imageW - bg->b.imageX) << 3;
    }

    SUPPORT_CHECK(bg->b.imageSiz == G_IM_SIZ_16b);
    GfxDpSetTextureImage(G_IM_FMT_RGBA, G_IM_SIZ_16b, 0, nullptr, texFlags, rawTexMetadata, (void*)data);
    GfxDpSetTile(G_IM_FMT_RGBA, G_IM_SIZ_16b, 0, 0, G_TX_LOADTILE, 0, 0, 0, 0, 0, 0, 0);
    GfxDpLoadBlock(G_TX_LOADTILE, 0, 0, (bg->b.imageW * bg->b.imageH >> 4) - 1, 0);
    GfxDpSetTile(bg->b.imageFmt, G_IM_SIZ_16b, bg->b.imageW >> 4, 0, G_TX_RENDERTILE, bg->b.imagePal, 0, 0, 0, 0, 0, 0);
    GfxDpSetTileSize(G_TX_RENDERTILE, 0, 0, bg->b.imageW, bg->b.imageH);
    GfxDpTextureRectangle(bg->b.frameX, bg->b.frameY, bg->b.frameX + bg->b.imageW - 4, bg->b.frameY + bg->b.imageH - 4,
                          G_TX_RENDERTILE, uls, bg->b.imageY << 3, dsdx, 1 << 10, false);
}

void Interpreter::Gfxs2dexBg1cyc(F3DuObjBg* bg) {
    uintptr_t data = (uintptr_t)bg->b.imagePtr;

    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    if ((bool)gfx_check_image_signature((char*)data)) {
        std::shared_ptr<Fast::Texture> tex = std::static_pointer_cast<Fast::Texture>(
            Ship::Context::GetRawInstance()->GetResourceManager()->LoadResourceProcess((char*)data));
        texFlags = tex->Flags;
        rawTexMetadata.width = tex->Width;
        rawTexMetadata.height = tex->Height;
        rawTexMetadata.h_byte_scale = tex->HByteScale;
        rawTexMetadata.v_pixel_scale = tex->VPixelScale;
        rawTexMetadata.type = tex->Type;
        rawTexMetadata.resource = tex;
        data = (uintptr_t)reinterpret_cast<char*>(tex->ImageData);
    }

    // TODO: Implement bg scaling correctly
    s16 uls = bg->b.imageX >> 2;
    s16 lrs = bg->b.imageW >> 2;

    s16 dsdxRect = 1 << 10;
    s16 ulsRect = bg->b.imageX << 3;
    // Flip flag only flips horizontally
    if (bg->b.imageFlip == G_BG_FLAG_FLIPS) {
        dsdxRect = -dsdxRect;
        ulsRect = (bg->b.imageW - bg->b.imageX) << 3;
    }

    GfxDpSetTextureImage(bg->b.imageFmt, bg->b.imageSiz, bg->b.imageW >> 2, nullptr, texFlags, rawTexMetadata,
                         (void*)data);
    GfxDpSetTile(bg->b.imageFmt, bg->b.imageSiz, 0, 0, G_TX_LOADTILE, 0, 0, 0, 0, 0, 0, 0);
    GfxDpLoadBlock(G_TX_LOADTILE, 0, 0, (bg->b.imageW * bg->b.imageH >> 4) - 1, 0);
    GfxDpSetTile(bg->b.imageFmt, bg->b.imageSiz, (((lrs - uls) * bg->b.imageSiz) + 7) >> 3, 0, G_TX_RENDERTILE,
                 bg->b.imagePal, 0, 0, 0, 0, 0, 0);
    GfxDpSetTileSize(G_TX_RENDERTILE, 0, 0, bg->b.imageW, bg->b.imageH);

    GfxDpTextureRectangle(bg->b.frameX, bg->b.frameY, bg->b.frameW, bg->b.frameH, G_TX_RENDERTILE, ulsRect,
                          bg->b.imageY << 3, dsdxRect, 1 << 10, false);
}

void Interpreter::Gfxs2dexRecyCopy(F3DuObjSprite* spr) {
    s16 dsdx = 4 << 10;
    [[maybe_unused]] s16 uls = spr->s.objX << 3;
    // Flip flag only flips horizontally
    if (spr->s.imageFlags == G_BG_FLAG_FLIPS) {
        dsdx = -dsdx;
        uls = (spr->s.imageW - spr->s.objX) << 3;
    }

    int realX = spr->s.objX >> 2;
    int realY = spr->s.objY >> 2;
    int realW = (((spr->s.imageW)) >> 5);
    int realH = (((spr->s.imageH)) >> 5);
    float realSW = spr->s.scaleW / 1024.0f;
    float realSH = spr->s.scaleH / 1024.0f;

    int testX = (realX + (realW / realSW));
    int testY = (realY + (realH / realSH));

    GfxDpTextureRectangle(realX << 2, realY << 2, testX << 2, testY << 2, G_TX_RENDERTILE,
                          (s32)mRdp->texture_tile[0].uls << 3, (s32)mRdp->texture_tile[0].ult << 3,
                          (float)(1 << 10) * realSW, (float)(1 << 10) * realSH, false);
}

void* Interpreter::SegAddr(uintptr_t w1) {
    // Segmented?
    if (w1 & 1) {
        uint32_t segNum = (uint32_t)(w1 >> 24);

        uint32_t offset = w1 & 0x00FFFFFE;

        if (mSegmentPointers[segNum] != 0) {
            return (void*)(mSegmentPointers[segNum] + offset);
        } else {
            return (void*)w1;
        }
    } else {
        return (void*)w1;
    }
}

#define C0(pos, width) ((cmd->words.w0 >> (pos)) & ((1U << width) - 1))
#define C1(pos, width) ((cmd->words.w1 >> (pos)) & ((1U << width) - 1))

void GfxExecStack::start(F3DGfx* dlist) {
    while (!cmd_stack.empty())
        cmd_stack.pop();
    gfx_path.clear();
    cmd_stack.push(dlist);
    disp_stack.clear();
}

void GfxExecStack::stop() {
    while (!cmd_stack.empty())
        cmd_stack.pop();
    gfx_path.clear();
}

F3DGfx*& GfxExecStack::currCmd() {
    return cmd_stack.top();
}

void GfxExecStack::openDisp(const char* file, int line) {
    disp_stack.push_back({ file, line });
}
void GfxExecStack::closeDisp() {
    disp_stack.pop_back();
}
const std::vector<GfxExecStack::CodeDisp>& GfxExecStack::getDisp() const {
    return disp_stack;
}

void GfxExecStack::branch(F3DGfx* caller) {
    F3DGfx* old = cmd_stack.top();
    cmd_stack.pop();
    cmd_stack.push(nullptr);
    cmd_stack.push(old);

    gfx_path.push_back(caller);
}

void GfxExecStack::call(F3DGfx* caller, F3DGfx* callee) {
    cmd_stack.push(callee);
    gfx_path.push_back(caller);
}

F3DGfx* GfxExecStack::ret() {
    F3DGfx* cmd = cmd_stack.top();

    cmd_stack.pop();
    if (!gfx_path.empty()) {
        gfx_path.pop_back();
    }

    while (cmd_stack.size() > 0 && cmd_stack.top() == nullptr) {
        cmd_stack.pop();
        if (!gfx_path.empty()) {
            gfx_path.pop_back();
        }
    }
    return cmd;
}

void gfx_set_framebuffer(int fb, float noise_scale);
void gfx_reset_framebuffer();
void gfx_copy_framebuffer(int fb_dst_id, int fb_src_id, bool copyOnce, bool* hasCopiedPtr);

// The main type of the handler function. These function will take a pointer to a pointer to a Gfx. It needs to be a
// double pointer because we sometimes need to increment and decrement the underlying pointer Returns false if the
// current opcode should be incremented after the handler ends.
typedef bool (*GfxOpcodeHandlerFunc)(F3DGfx** gfx);

bool gfx_load_ucode_handler_f3dex2(F3DGfx** cmd) {
    Interpreter* gfx = mInstance.lock().get();
    gfx->mRsp->fog_mul = 0;
    gfx->mRsp->fog_offset = 0;
    return false;
}

bool gfx_cull_dl_handler_f3dex2(F3DGfx** cmd) {
    // TODO:
    return false;
}

bool gfx_marker_handler_otr(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    (*cmd0)++;
    F3DGfx* cmd = (*cmd0);
    gfx->mMarkerOn = true;
    return false;
}

bool gfx_invalidate_tex_cache_handler_f3dex2(F3DGfx** cmd) {
    Interpreter* gfx = mInstance.lock().get();
    const uintptr_t texAddr = (*cmd)->words.w1;

    if (texAddr == 0) {
        gfx->TextureCacheClear();
    } else {
        gfx->TextureCacheDelete((const uint8_t*)texAddr);
    }
    return false;
}

bool gfx_noop_handler_f3dex2(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    const char* filename = (const char*)(cmd)->words.w1;
    uint32_t p = C0(16, 8);
    uint32_t l = C0(0, 16);
    if (p == 7) {
        g_exec_stack.openDisp(filename, l);
    } else if (p == 8) {
        if (g_exec_stack.disp_stack.size() == 0) {
            SPDLOG_WARN("CLOSE_DISPS without matching open {}:{}", p, l);
        } else {
            g_exec_stack.closeDisp();
        }
    }
    return false;
}

bool gfx_mtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    uintptr_t mtxAddr = cmd->words.w1;

    gfx->GfxSpMatrix(C0(0, 8) ^ F3DEX2_G_MTX_PUSH, (const int32_t*)gfx->SegAddr(mtxAddr));
    return false;
}
// Seems to be the same for all other non F3DEX2 microcodes...
bool gfx_mtx_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    uintptr_t mtxAddr = cmd->words.w1;

    gfx->GfxSpMatrix(C0(16, 8), (const int32_t*)gfx->SegAddr(cmd->words.w1));
    return false;
}

bool gfx_mtx_otr_filepath_handler_custom_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    const char* fileName = (const char*)cmd->words.w1;
    const int32_t* mtx = (const int32_t*)Ship::Context::GetRawInstance()->GetResourceManager()->GetResourceRawPointer(
        (const char*)fileName);

    if (mtx != NULL) {
        gfx->GfxSpMatrix(C0(0, 8) ^ F3DEX2_G_MTX_PUSH, mtx);
    }

    return false;
}

bool gfx_mtx_otr_filepath_handler_custom_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    const char* fileName = (const char*)cmd->words.w1;
    const int32_t* mtx = (const int32_t*)Ship::Context::GetRawInstance()->GetResourceManager()->GetResourceRawPointer(
        (const char*)fileName);

    if (mtx != NULL) {
        gfx->GfxSpMatrix(C0(16, 8), mtx);
    }

    return false;
}

bool gfx_mtx_otr_filepath_handler_custom(F3DGfx** cmd0) {
    if (ucode_handler_index == ucode_f3dex2) {
        return gfx_mtx_otr_filepath_handler_custom_f3dex2(cmd0);
    } else {
        return gfx_mtx_otr_filepath_handler_custom_f3d(cmd0);
    }
}

bool gfx_mtx_otr_handler_custom_f3dex2(F3DGfx** cmd0) {
    (*cmd0)++;
    F3DGfx* cmd = *cmd0;

    const uint64_t hash = ((uint64_t)cmd->words.w0 << 32) + cmd->words.w1;
    const int32_t* mtx =
        (const int32_t*)Ship::Context::GetRawInstance()->GetResourceManager()->GetResourceRawPointer(hash);

    if (mtx != NULL) {
        Interpreter* gfx = mInstance.lock().get();
        cmd--;
        gfx->GfxSpMatrix(C0(0, 8) ^ F3DEX2_G_MTX_PUSH, mtx);
        cmd++;
    }

    return false;
}

bool gfx_mtx_otr_handler_custom_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    (*cmd0)++;
    F3DGfx* cmd = *cmd0;

    const uint64_t hash = ((uint64_t)cmd->words.w0 << 32) + cmd->words.w1;
    const int32_t* mtx =
        (const int32_t*)Ship::Context::GetRawInstance()->GetResourceManager()->GetResourceRawPointer(hash);
    if (mtx != nullptr) {
        cmd--;
        gfx->GfxSpMatrix(C0(16, 8), mtx);
        cmd++;
    }
    return false;
}

bool gfx_mtx_otr_handler_custom(F3DGfx** cmd0) {
    if (ucode_handler_index == ucode_f3dex2) {
        return gfx_mtx_otr_handler_custom_f3dex2(cmd0);
    } else {
        return gfx_mtx_otr_handler_custom_f3d(cmd0);
    }
}

bool gfx_pop_mtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpPopMatrix((uint32_t)(cmd->words.w1 / 64));

    return false;
}

bool gfx_pop_mtx_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpPopMatrix(1);

    return false;
}

bool gfx_movemem_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovememF3dex2(C0(0, 8), C0(8, 8) * 8, gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_movemem_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovememF3d(C0(16, 8), 0, gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_movemem_handler_otr(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    const uint8_t index = C1(24, 8);
    const uint8_t offset = C1(16, 8);
    const uint8_t hasOffset = C1(8, 8);

    (*cmd0)++;

    const uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

    if (ucode_handler_index == ucode_f3dex2) {
        gfx->GfxSpMovememF3dex2(index, offset,
                                Ship::Context::GetRawInstance()->GetResourceManager()->GetResourceRawPointer(hash));
    } else {
        auto light =
            (Fast::LightEntry*)Ship::Context::GetRawInstance()->GetResourceManager()->GetResourceRawPointer(hash);
        uintptr_t data = (uintptr_t)&light->Ambient;
        gfx->GfxSpMovememF3d(index, offset, (void*)(data + (hasOffset == 1 ? 0x8 : 0)));
    }
    return false;
}

bool gfx_push_shader(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    const char* path = (const char*)gfx->SegAddr(cmd->words.w1);

    if (!gfx_check_image_signature(path)) {
        SPDLOG_ERROR("G_PUSH_SHADER: Shader is not a valid OTR resource name, unable to register push shader");
        return false;
    }

    path = &path[7];

    size_t shaderId = static_cast<size_t>(-1);
    for (const auto& shader : gfx->mShaders) {
        if (strcmp(shader.second, path) == 0) {
            shaderId = shader.first;
            break;
        }
    }

    if (shaderId == static_cast<size_t>(-1)) {
        shaderId = gfx->mShadersIndex++;
        gfx->mShaders[shaderId] = path;
    }

    gfx->mShaderStack.push(shaderId);

    return false;
}

bool gfx_pop_shader(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    if (!gfx->mShaderStack.empty()) {
        gfx->mShaderStack.pop();
    }

    return false;
}

const char* gfx_get_shader(int16_t id) {
    Interpreter* gfx = mInstance.lock().get();

    for (const std::pair<size_t, const char*>& shader : gfx->mShaders) {
        if (shader.first == id) {
            return shader.second;
        }
    }

    return nullptr; // Use no shader
}

bool gfx_moveword_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovewordF3dex2(C0(16, 8), C0(0, 16), cmd->words.w1);

    return false;
}

bool gfx_moveword_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovewordF3d(C0(0, 8), C0(8, 16), cmd->words.w1);

    return false;
}

bool gfx_texture_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTexture(C1(16, 16), C1(0, 16), C0(11, 3), C0(8, 3), C0(1, 7));

    return false;
}

// Seems to be the same for all other non F3DEX2 microcodes...
bool gfx_texture_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTexture(C1(16, 16), C1(0, 16), C0(11, 3), C0(8, 3), C0(0, 8));

    return false;
}

// Almost all versions of the microcode have their own version of this opcode
bool gfx_vtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpVertex(C0(12, 8), C0(1, 7) - C0(12, 8), (const F3DVtx*)gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_vtx_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    gfx->GfxSpVertex(C0(10, 6), C0(17, 7), (const F3DVtx*)gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_vtx_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpVertex((C0(0, 16)) / sizeof(F3DVtx), C0(16, 4), (const F3DVtx*)gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_vtx_hash_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    // Offset added to the start of the vertices
    const uintptr_t offset = (*cmd0)->words.w1;
    // This is a two-part display list command, so increment the instruction pointer so we can get the CRC64
    // hash from the second
    (*cmd0)++;
    const uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

    // We need to know if the offset is a cached pointer or not. An offset greater than one million is not a
    // real offset, so it must be a real pointer
    if (offset > 0xFFFFF) {
        (*cmd0)--;
        F3DGfx* cmd = *cmd0;
        gfx->GfxSpVertex(C0(12, 8), C0(1, 7) - C0(12, 8), (F3DVtx*)offset);
        (*cmd0)++;
    } else {
        F3DVtx* vtx = (F3DVtx*)Ship::Context::GetRawInstance()->GetResourceManager()->GetResourceRawPointer(hash);

        if (vtx != NULL) {
            vtx = (F3DVtx*)((char*)vtx + offset);

            (*cmd0)--;
            F3DGfx* cmd = *cmd0;

            // TODO: WTF??
            cmd->words.w1 = (uintptr_t)vtx;

            gfx->GfxSpVertex(C0(12, 8), C0(1, 7) - C0(12, 8), vtx);
            (*cmd0)++;
        }
    }
    return false;
}

bool gfx_vtx_otr_filepath_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    char* fileName = (char*)cmd->words.w1;
    (*cmd0)++;
    cmd = *cmd0;
    size_t vtxCnt = cmd->words.w0;
    size_t vtxIdxOff = cmd->words.w1 >> 16;
    size_t vtxDataOff = cmd->words.w1 & 0xFFFF;
    F3DVtx* vtx =
        (F3DVtx*)Ship::Context::GetRawInstance()->GetResourceManager()->GetResourceRawPointer((const char*)fileName);
    vtx += vtxDataOff;

    gfx->GfxSpVertex(vtxCnt, vtxIdxOff, vtx);
    return false;
}

bool gfx_dl_otr_filepath_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    char* fileName = (char*)cmd->words.w1;
    F3DGfx* nDL =
        (F3DGfx*)Ship::Context::GetRawInstance()->GetResourceManager()->GetResourceRawPointer((const char*)fileName);

    if (C0(16, 1) == 0 && nDL != nullptr) {
        g_exec_stack.call(*cmd0, nDL);
    } else {
        if (nDL != nullptr) {
            (*cmd0) = nDL;
            g_exec_stack.branch(cmd);
            return true; // shortcut cmd increment
        } else {
            assert(0 && "???");
            // gfx_path.pop_back();
            // cmd = cmd_stack.top();
            // cmd_stack.pop();
        }
    }
    return false;
}

// The original F3D microcode doesn't seem to have this opcode. Glide handles it as part of moveword
bool gfx_modify_vtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    gfx->GfxSpModifyVertex(C0(1, 15), C0(16, 8), (uint32_t)cmd->words.w1);
    return false;
}

// F3D, F3DEX, and F3DEX2 do the same thing but F3DEX2 has its own opcode number
bool gfx_dl_handler_common(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    F3DGfx* subGFX = (F3DGfx*)gfx->SegAddr(cmd->words.w1);
    if (gfx->mMM3DSuppress && (cmd->words.w1 >> 24) >= 8 && (cmd->words.w1 >> 24) <= 13) {
        gfx->mMM3DSegment = cmd->words.w1 >> 24; // AnimatedMat DL: its tile-size lerp is the live scroll
    }
    if (C0(16, 1) == 0) {
        // Push return address
        if (subGFX != nullptr) {
            g_exec_stack.call(*cmd0, subGFX);
        }
    } else {
        (*cmd0) = subGFX;
        g_exec_stack.branch(cmd);
        return true; // shortcut cmd increment
    }
    return false;
}

bool gfx_dl_otr_hash_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    if (C0(16, 1) == 0) {
        // Push return address
        (*cmd0)++;

        uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

        F3DGfx* gfx = (F3DGfx*)Ship::Context::GetRawInstance()->GetResourceManager()->GetResourceRawPointer(hash);

        if (gfx != 0) {
            g_exec_stack.call(cmd, gfx);
        }
    } else {
        Interpreter* gfx = mInstance.lock().get();
        assert(0 && "????");
        (*cmd0) = (F3DGfx*)gfx->SegAddr((*cmd0)->words.w1);
        return true;
    }
    return false;
}
bool gfx_dl_index_handler(F3DGfx** cmd0) {
    // Compute seg addr by converting an index value to a offset value
    // handling 32 vs 64 bit size differences for Gfx
    // adding 1 to trigger the segaddr flow
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = (*cmd0);
    uint8_t segNum = (uint8_t)(cmd->words.w1 >> 24);
    uint32_t index = (uint32_t)(cmd->words.w1 & 0x00FFFFFF);
    uintptr_t segAddr = (segNum << 24) | (index * sizeof(F3DGfx)) + 1;

    F3DGfx* subGFX = (F3DGfx*)gfx->SegAddr(segAddr);
    if (C0(16, 1) == 0) {
        // Push return address
        if (subGFX != nullptr) {
            g_exec_stack.call((*cmd0), subGFX);
        }
    } else {
        (*cmd0) = subGFX;
        g_exec_stack.branch(cmd);
        return true; // shortcut cmd increment
    }
    return false;
}

// TODO handle special OTR opcodes later...
bool gfx_pushcd_handler_custom(F3DGfx** cmd0) {
    gfx_push_current_dir((char*)(*cmd0)->words.w1);
    return false;
}

// TODO handle special OTR opcodes later...
bool gfx_branch_z_otr_handler_f3dex2(F3DGfx** cmd0) {
    // Push return address
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = (*cmd0);

    uint8_t vbidx = (uint8_t)((*cmd0)->words.w0 & 0x00000FFF);
    uint32_t zval = (uint32_t)((*cmd0)->words.w1);

    (*cmd0)++;

    if (gfx->mRsp->loaded_vertices[vbidx].z <= zval ||
        (gfx->mRsp->extra_geometry_mode & G_EX_ALWAYS_EXECUTE_BRANCH) != 0) {
        uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

        F3DGfx* gfx = (F3DGfx*)Ship::Context::GetRawInstance()->GetResourceManager()->GetResourceRawPointer(hash);

        if (gfx != 0) {
            (*cmd0) = gfx;
            g_exec_stack.branch(cmd);
            return true; // shortcut cmd increment
        }
    }
    return false;
}

// F3D, F3DEX, and F3DEX2 do the same thing but F3DEX2 has its own opcode number
bool gfx_end_dl_handler_common(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    gfx->mMarkerOn = false;
    g_exec_stack.ret();
    return true;
}

bool gfx_set_prim_depth_handler_rdp(F3DGfx** cmd) {
    Interpreter* gfx = mInstance.lock().get();
    uint32_t w1 = (*cmd)->words.w1;
    gfx->mRdp->prim_depth = (uint16_t)((w1 >> 16) & 0x7FFF); // Mask to 15 bits
    return false;
}

// Only on F3DEX2
bool gfx_geometry_mode_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode(~C0(0, 24), (uint32_t)cmd->words.w1);
    return false;
}

// Only on F3DEX and older
bool gfx_set_geometry_mode_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode(0, (uint32_t)cmd->words.w1);
    return false;
}

// Only on F3DEX and older
bool gfx_clear_geometry_mode_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode((uint32_t)cmd->words.w1, 0);
    return false;
}

bool gfx_tri1_otr_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();

    F3DGfx* cmd = *cmd0;
    uint8_t v00 = (uint8_t)(cmd->words.w0 & 0x0000FFFF);
    uint8_t v01 = (uint8_t)(cmd->words.w1 >> 16);
    uint8_t v02 = (uint8_t)(cmd->words.w1 & 0x0000FFFF);
    gfx->mSmoothKey = (uint64_t)(uintptr_t)cmd * 2;
    gfx->GfxSpTri1(v00, v01, v02, false);

    return false;
}

bool gfx_tri1_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mSmoothKey = (uint64_t)(uintptr_t)cmd * 2;
    gfx->GfxSpTri1(C0(16, 8) / 2, C0(8, 8) / 2, C0(0, 8) / 2, false);

    return false;
}

bool gfx_tri1_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mSmoothKey = (uint64_t)(uintptr_t)cmd * 2;
    gfx->GfxSpTri1(C1(17, 7), C1(9, 7), C1(1, 7), false);

    return false;
}

bool gfx_tri1_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mSmoothKey = (uint64_t)(uintptr_t)cmd * 2;
    gfx->GfxSpTri1(C1(16, 8) / 10, C1(8, 8) / 10, C1(0, 8) / 10, false);

    return false;
}

// F3DEX, and F3DEX2 share a tri2 function, however F3DEX has a different quad function.
bool gfx_tri2_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mSmoothKey = (uint64_t)(uintptr_t)cmd * 2;
    gfx->GfxSpTri1(C0(17, 7), C0(9, 7), C0(1, 7), false);
    gfx->mSmoothKey = (uint64_t)(uintptr_t)cmd * 2 + 1;
    gfx->GfxSpTri1(C1(17, 7), C1(9, 7), C1(1, 7), false);
    return false;
}

bool gfx_quad_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mSmoothKey = (uint64_t)(uintptr_t)cmd * 2;
    gfx->GfxSpTri1(C0(16, 8) / 2, C0(8, 8) / 2, C0(0, 8) / 2, false);
    gfx->mSmoothKey = (uint64_t)(uintptr_t)cmd * 2 + 1;
    gfx->GfxSpTri1(C1(16, 8) / 2, C1(8, 8) / 2, C1(0, 8) / 2, false);
    return false;
}

bool gfx_quad_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mSmoothKey = (uint64_t)(uintptr_t)cmd * 2;
    gfx->GfxSpTri1(C1(16, 8) / 2, C1(8, 8) / 2, C1(0, 8) / 2, false);
    gfx->mSmoothKey = (uint64_t)(uintptr_t)cmd * 2 + 1;
    gfx->GfxSpTri1(C1(16, 8) / 2, C1(0, 8) / 2, C1(24, 8) / 2, false);
    return false;
}

bool gfx_othermode_l_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(31 - C0(8, 8) - C0(0, 8), C0(0, 8) + 1, cmd->words.w1);

    return false;
}

bool gfx_othermode_l_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(C0(8, 8), C0(0, 8), cmd->words.w1);

    return false;
}

bool gfx_othermode_h_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(63 - C0(8, 8) - C0(0, 8), C0(0, 8) + 1, (uint64_t)cmd->words.w1 << 32);

    return false;
}

// Only on F3DEX and older
bool gfx_set_geometry_mode_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode(0, (uint32_t)cmd->words.w1);
    return false;
}

// Only on F3DEX and older
bool gfx_clear_geometry_mode_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode((uint32_t)cmd->words.w1, 0);
    return false;
}

bool gfx_othermode_h_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(C0(8, 8) + 32, C0(0, 8), (uint64_t)cmd->words.w1 << 32);

    return false;
}

// A resolved address still in the N64 segmented range (<= 0x0FFFFFFF) usually means SegAddr
// failed to resolve it (segment not set up). Keep it only if it belongs to a loaded module,
// where it's a real low pointer (e.g. a static TLUT) rather than an unresolved segment addr.
static bool IsValidResolvedAddress(uintptr_t addr) {
    if (addr > 0x0FFFFFFF) {
        return true;
    }

    // Still in the N64 segmented range, but might be a false positive (a real low pointer).
#ifdef _WIN32
    // For Windows, check whether the address belongs to a dll.
    HMODULE module = nullptr;
    return GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              reinterpret_cast<LPCSTR>(addr), &module) != 0;
#else
    // For non-Windows platforms, check whether the address belongs to a loaded object.
    Dl_info info;
    return dladdr(reinterpret_cast<void*>(addr), &info) != 0;
#endif
}

bool gfx_set_timg_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    uintptr_t i = (uintptr_t)gfx->SegAddr(cmd->words.w1);

    char* imgData = (char*)i;
    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetdata = {};
    // Default scale factors to 1 for raw N64 textures. OTR textures set these
    // from the resource, but raw textures would leave them at 0.
    rawTexMetdata.h_byte_scale = 1;
    rawTexMetdata.v_pixel_scale = 1;

    if ((i & 1) != 1) {
        if (gfx_check_image_signature(imgData) == 1) {
            bool usedFallback = false;
            std::shared_ptr<Fast::Texture> tex =
                std::static_pointer_cast<Fast::Texture>(gfx->ResolveTextureResource(imgData, usedFallback));

            if (tex == nullptr) {
                (*cmd0)++;
                return false;
            }

            i = (uintptr_t)reinterpret_cast<char*>(tex->ImageData);
            texFlags = tex->Flags;
            rawTexMetdata.width = tex->Width;
            rawTexMetdata.height = tex->Height;
            rawTexMetdata.h_byte_scale = tex->HByteScale;
            rawTexMetdata.v_pixel_scale = tex->VPixelScale;
            rawTexMetdata.type = tex->Type;
            rawTexMetdata.resource = tex;
        }
    }

    if (!IsValidResolvedAddress(i)) {
        return false;
    }

    gfx->GfxDpSetTextureImage(C0(21, 3), C0(19, 2), C0(0, 12) + 1, imgData, texFlags, rawTexMetdata, (void*)i);

    return false;
}

bool gfx_set_timg_otr_hash_handler_custom(F3DGfx** cmd0) {
    uintptr_t addr = (*cmd0)->words.w1;
    (*cmd0)++;
    uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (uint64_t)(*cmd0)->words.w1;

    const char* fileName =
        Ship::Context::GetRawInstance()->GetResourceManager()->GetArchiveManager()->HashToCString(hash);
    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    if (fileName == nullptr) {
        (*cmd0)++;
        return false;
    }

    Interpreter* gfx = mInstance.lock().get();
    bool usedFallback = false;
    std::shared_ptr<Fast::Texture> texture =
        std::static_pointer_cast<Fast::Texture>(gfx->ResolveTextureResource(fileName, usedFallback));
    if (texture != nullptr) {
        texFlags = texture->Flags;
        rawTexMetadata.width = texture->Width;
        rawTexMetadata.height = texture->Height;
        rawTexMetadata.h_byte_scale = texture->HByteScale;
        rawTexMetadata.v_pixel_scale = texture->VPixelScale;
        rawTexMetadata.type = texture->Type;
        rawTexMetadata.resource = texture;

        // OTRTODO: We have disabled caching for now to fix a texture corruption issue with HD texture
        // support. In doing so, there is a potential performance hit since we are not caching lookups. We
        // need to do proper profiling to see whether or not it is worth it to keep the caching system.

        char* tex = reinterpret_cast<char*>(texture->ImageData);

        if (tex != nullptr && !usedFallback) {
            (*cmd0)--;
            uintptr_t oldData = (*cmd0)->words.w1;
            // TODO: wtf??
            (*cmd0)->words.w1 = (uintptr_t)tex;

            // if (ourHash != (uint64_t)-1) {
            //     auto res = ResourceLoad(ourHash);
            // }

            (*cmd0)++;
        }

        (*cmd0)--;
        F3DGfx* cmd = (*cmd0);
        uint32_t fmt = C0(21, 3);
        uint32_t size = C0(19, 2);
        uint32_t width = C0(0, 12) + 1;

        if (tex != NULL) {
            Interpreter* gfx = mInstance.lock().get();
            gfx->GfxDpSetTextureImage(fmt, size, width, fileName, texFlags, rawTexMetadata, tex);
        }
    } else {
        SPDLOG_ERROR("G_SETTIMG_OTR_HASH: Texture is null");
    }

    (*cmd0)++;
    return false;
}

bool gfx_set_timg_otr_filepath_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    const char* fileName = (char*)cmd->words.w1;

    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    Interpreter* gfx = mInstance.lock().get();
    bool usedFallback = false;
    std::shared_ptr<Fast::Texture> texture =
        std::static_pointer_cast<Fast::Texture>(gfx->ResolveTextureResource(fileName, usedFallback));
    if (texture != nullptr) {
        texFlags = texture->Flags;
        rawTexMetadata.width = texture->Width;
        rawTexMetadata.height = texture->Height;
        rawTexMetadata.h_byte_scale = texture->HByteScale;
        rawTexMetadata.v_pixel_scale = texture->VPixelScale;
        rawTexMetadata.type = texture->Type;
        rawTexMetadata.resource = texture;

        uint32_t fmt = C0(21, 3);
        uint32_t size = C0(19, 2);
        uint32_t width = C0(0, 12) + 1;

        gfx->GfxDpSetTextureImage(fmt, size, width, fileName, texFlags, rawTexMetadata,
                                  reinterpret_cast<char*>(texture->ImageData));
    } else {
        SPDLOG_ERROR("G_SETTIMG_OTR_FILEPATH: Texture is null");
    }
    return false;
}

bool gfx_set_fb_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();
    gfx->Flush();

    if (cmd->words.w1) {
        gfx->SetFrameBuffer((int32_t)cmd->words.w1, 1.0f);
        gfx->mActiveFrameBuffer = gfx->mFrameBuffers.find((int32_t)cmd->words.w1);
        gfx->mFbActive = true;
    } else {
        gfx->ResetFrameBuffer();
        gfx->mFbActive = false;
        gfx->mActiveFrameBuffer = gfx->mFrameBuffers.end();
    }
    return false;
}

bool gfx_reset_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    gfx->Flush();
    gfx->mFbActive = false;
    gfx->mActiveFrameBuffer = gfx->mFrameBuffers.end();
    gfx->mRapi->StartDrawToFramebuffer(gfx->mRendersToFb ? gfx->mGameFb : 0,
                                       (float)gfx->mCurDimensions.height / gfx->mNativeDimensions.height);
    // Force viewport and scissor to reapply against the main framebuffer, in case a previous smaller
    // framebuffer truncated the values
    gfx->mRdp->viewport_or_scissor_changed = true;
    gfx->mRenderingState.viewport = {};
    gfx->mRenderingState.scissor = {};
    return false;
}

bool gfx_newdawn_mm3d_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    gfx->Flush();
    if (!cmd->words.w1 || !gfx->mRapi->MM3DAvailable()) {
        return false;
    }
    const auto& params = *reinterpret_cast<const NewDawnMM3DRoomParams*>(cmd->words.w1);
    if (C0(0, 8) == 0) {
        // Upload/compile must succeed BEFORE any native triangle gets suppressed.
        gfx->mMM3DSuppress = gfx->mRapi->PrepareMM3DRoom(params);
        memcpy(gfx->mMM3DProjection, gfx->mRsp->P_matrix, sizeof(gfx->mMM3DProjection));
        gfx->mMM3DAspect = gfx->AdjXForAspectRatio(1.0f);
        gfx->mMM3DFlipY = gfx->mRapi->GetClipParameters().invertY ? -1.0f : 1.0f;
    } else if (gfx->mMM3DSuppress) {
        gfx->mMM3DSuppress = false;
        gfx->mMM3DSegment = 0;
        gfx->SyncViewportScissor();
        gfx->mRapi->DrawMM3DRoom(params, gfx->mMM3DProjection, gfx->mMM3DAspect, gfx->mMM3DFlipY, &gfx->mMM3DScroll);
        if (gfx->mShadowRecording && params.pass == 0) {
            for (uint32_t m = 0; m < params.meshCount; ++m) {
                const auto& mesh = *params.meshes[m];
                for (uint32_t d = 0; d < mesh.drawCount; ++d) {
                    const auto& draw = mesh.draws[d];
                    // Match native policy: solid opaque depth-writing surfaces only.
                    if (draw.material.blend || draw.material.alphaTest || !draw.material.depthWrite) continue;
                    for (uint32_t i = 0; i < draw.indexCount; i += 3) {
                        LoadedVertex vertices[3] = {};
                        for (int k = 0; k < 3; ++k) {
                            const float* v = mesh.vertices + mesh.indices[draw.firstIndex + i + k] * 16;
                            memcpy(vertices[k].view_pos, v, sizeof(float) * 3);
                        }
                        gfx->RecordShadowTriangle(&vertices[0], &vertices[1], &vertices[2]);
                    }
                }
            }
            static int sLogged = 0;
            if (sLogged < 2) {
                ++sLogged;
                SPDLOG_INFO("MM3D: {} shadow-caster triangles recorded after the room pass", gfx->mShadowVertices.size() / 9);
            }
        }
    }
    return false;
}

bool gfx_newdawn_postfx_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    if (cmd->words.w1 == 0) {
        gfx->Flush();
        return false;
    }
    const int stage = C0(0, 8);
    if (stage == NEWDAWN_POSTFX_STAGE_GI_ROOM_BEGIN || stage == NEWDAWN_POSTFX_STAGE_GI_ROOM_END) {
        // World GI room scope: only decides whether later triangles are also recorded; no flush, nothing drawn.
        gfx->mGiRecord = stage == NEWDAWN_POSTFX_STAGE_GI_ROOM_BEGIN && gfx->mPostFxActive && gfx->mRendersToFb &&
                         !gfx->mFbActive && gfx->mRapi->GiWantsGeometry();
        gfx->mGiDebug = gfx->mGiRecord &&
            Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.GI.WorldLog", 0) != 0;
        return false;
    }
    if (stage == NEWDAWN_POSTFX_STAGE_WIND_BEGIN || stage == NEWDAWN_POSTFX_STAGE_WIND_END) {
        // Only changes how later triangles are built: no batch flush needed, so per-actor scopes stay cheap.
        gfx->mWindSway = stage == NEWDAWN_POSTFX_STAGE_WIND_BEGIN && gfx->mPostFxActive;
        gfx->mWindRoom = gfx->mWindSway && ((cmd->words.w0 >> NEWDAWN_POSTFX_WIND_ROOM_SHIFT) & 1);
        gfx->mWindDisplace = gfx->mWindSway && ((cmd->words.w0 >> NEWDAWN_POSTFX_WIND_SWAY_SHIFT) & 1);
        gfx->mFoliageLight = gfx->mWindSway && ((cmd->words.w0 >> NEWDAWN_POSTFX_WIND_FOLIAGE_SHIFT) & 1);
        return false;
    }
    if (stage == NEWDAWN_POSTFX_STAGE_SHADOW_ONLY_BEGIN) {
        // w1 is a display list of room entries the game culled. It only runs while casters are being recorded (never on
        // another backend or with the effect off, where it would draw the entries), and the end marker restores state.
        // World GI hands over its own list (the GI bit): it runs only on the frames room geometry is recorded for the
        // voxel grid, and never records casters.
        gfx->Flush();
        const bool giList = ((cmd->words.w0 >> NEWDAWN_POSTFX_SHADOW_ONLY_GI_SHIFT) & 1) != 0;
        if (gfx->mPostFxActive && (giList ? gfx->mGiRecord : gfx->mShadowRecording) && !gfx->mShadowOnly) {
            gfx->BeginShadowOnly();
            gfx->mShadowOnlyGi = giList;
            g_exec_stack.call(cmd, reinterpret_cast<F3DGfx*>(cmd->words.w1));
        }
        return false;
    }
    if (stage == NEWDAWN_POSTFX_STAGE_SHADOW_ONLY_END) {
        gfx->EndShadowOnly();
        return false;
    }
    gfx->Flush();
    const auto& params = *reinterpret_cast<const NewDawnPostFxParams*>(cmd->words.w1);

    if (stage == NEWDAWN_POSTFX_STAGE_SHADOW_BEGIN) {
        // Water only needs the camera of this frame, not the casters.
        gfx->BeginVfxRecording(gfx->mPostFxActive && params.vfxEnabled != 0);
        gfx->BeginGrassRecording(gfx->mPostFxActive && params.grassEnabled != 0);
        {
            // Sun for the light through the leaves: linear radiance the way the grass lights its blades.
            auto decode = [](float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); };
            const float sunK = std::clamp(Ship::Context::GetRawInstance()->GetConsoleVariables()->GetFloat(
                                              "gNewDawn.HDR.SunIntensity", 1.8f), 0.0f, 16.0f) * 0.5f;
            const float col[3] = { decode(params.foliageSunColor[0]) * sunK, decode(params.foliageSunColor[1]) * sunK,
                                   decode(params.foliageSunColor[2]) * sunK };
            gfx->mRapi->SetFoliageFrame(params.giCameraEye, params.foliageSunDir, col,
                                        gfx->mPostFxActive && params.foliageLight ? params.foliageStrength : 0.0f);
        }
        gfx->BeginInteriorRecording(gfx->mPostFxActive && (params.interiorEnabled != 0 || params.emissiveEnabled != 0),
                                    params.interiorDebug != 0);
        gfx->BeginShadowRecording(gfx->mPostFxActive &&
                                      (params.shadowEnabled != 0 || params.atmosWeather || params.localShadows != 0),
                                  gfx->mPostFxActive && (params.waterEnabled != 0 || params.atmosBiome != 0 ||
                                                         params.fogVolumetric != 0 || params.giEnabled != 0 ||
                                                         params.giWorld != 0 ||
                                                         params.vfxEnabled != 0 || params.grassEnabled != 0));
        return false;
    }

    if (stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END) {
        gfx->EndShadowRecording();
    }

    // Only ever post-process the game framebuffer itself
    if (gfx->mPostFxActive && gfx->mRendersToFb && !gfx->mFbActive) {
        NewDawnPostFxParams effective = params;
        if (stage == NEWDAWN_POSTFX_STAGE_PHYSICAL_SKY || stage == NEWDAWN_POSTFX_STAGE_SUN_FLARE_BEGIN) {
            // P includes the interpolated camera view, independent of the rotating skybox model matrix.
            effective.shadowUseExactProj = Invert4x4(gfx->mRsp->P_matrix, effective.shadowInvViewProj);
            memcpy(effective.shadowViewProj, gfx->mRsp->P_matrix, sizeof(effective.shadowViewProj));
            float xScale = gfx->AdjXForAspectRatio(1.0f);
            effective.shadowNdcXScale = xScale > 1e-6f ? 1.0f / xScale : 1.0f;
            effective.shadowFlipY = gfx->mRapi->GetClipParameters().invertY ? -1.0f : 1.0f;
        } else if (stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END) {
            gfx->FillShadowCameraParams(effective);
            if (effective.shadowUseExactProj == 0) {
                effective.shadowEnabled = 0;
            }
            gfx->SubmitShadowCasters();
            gfx->SubmitGrass();
            gfx->mGrass = false;
            gfx->mRapi->SetGiTriangles(gfx->mGiTris.empty() ? nullptr : gfx->mGiTris.data(),
                                       (uint32_t)(gfx->mGiTris.size() / NEWDAWN_GI_TRI_FLOATS));
            gfx->SubmitInterior(effective);
            gfx->mInterior = false;
        } else if ((stage == NEWDAWN_POSTFX_STAGE_SCENE_END && (effective.waterEnabled != 0 || effective.vfxEnabled != 0)) ||
                   (stage == NEWDAWN_POSTFX_STAGE_WATER_XLU && effective.waterEnabled != 0)) {
            if (stage == NEWDAWN_POSTFX_STAGE_SCENE_END) {
                gfx->SubmitVfx();
            }
            // The camera captured at the start of the world; the water time advances with the interpolation fraction
            // (the display list is the newer tick, the interpolated frames run from the previous one towards it).
            gfx->FillShadowCameraParams(effective);
            effective.waterTime += (gfx->mInterpolationT - 1.0f) * (1.0f / 20.0f);
            if (effective.shadowUseExactProj == 0) {
                effective.waterEnabled = 0;
            }
        }
        gfx->mRapi->RunPostFx(stage, effective, gfx->mGameFb);
        if (stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END) {
            gfx->mGiTris.clear();
        }
        if (stage == NEWDAWN_POSTFX_STAGE_WORLD_BEGIN) {
            gfx->BeginTaa(effective);
        }
    }
    if (stage == NEWDAWN_POSTFX_STAGE_SCENE_END) {
        gfx->mVfx = false; // stop recording until the next frame's begin marker
        gfx->mTaaJitter = false; // the HUD and everything after the world stays on the pixel grid
    }
    return false;
}

bool gfx_copy_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    bool* hasCopiedPtr = (bool*)cmd->words.w1;

    gfx->Flush();
    gfx->CopyFrameBuffer(C0(11, 11), C0(0, 11), (bool)C0(22, 1), hasCopiedPtr);
    return false;
}

bool gfx_read_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    int32_t width, height;
    [[maybe_unused]] int32_t ulx, uly;
    uint16_t* rgba16Buffer = (uint16_t*)cmd->words.w1;
    int fbId = C0(0, 8);
    bool bswap = C0(8, 1);
    ++(*cmd0);
    cmd = *cmd0;
    // Specifying the upper left origin value is unused and unsupported at the renderer level
    ulx = C0(0, 16);
    uly = C0(16, 16);
    width = C1(0, 16);
    height = C1(16, 16);

    gfx->Flush();
    gfx->mRapi->ReadFramebufferToCPU(fbId, width, height, rgba16Buffer);

#ifndef IS_BIGENDIAN
    // byteswap the output to BE
    if (bswap) {
        for (size_t i = 0; i < (size_t)width * height; i++) {
            rgba16Buffer[i] = BE16SWAP(rgba16Buffer[i]);
        }
    }
#endif

    return false;
}

bool gfx_register_blended_texture_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    // Flush incase we are replacing a previous blended texture that hasn't been finialized to the GPU
    gfx->Flush();

    char* timg = (char*)cmd->words.w1;

    ++(*cmd0);
    cmd = *cmd0;

    uint8_t* mask = (uint8_t*)cmd->words.w0;
    uint8_t* replacementTex = (uint8_t*)cmd->words.w1;

    if (!gfx_check_image_signature(timg)) {
        SPDLOG_ERROR(
            "OTR_G_REGBLENDEDTEX: Texture is not a valid OTR resource name, unable to register blended texture");
        return false;
    }

    // With no mask, we should clear the blended texture
    if (mask == nullptr) {
        gfx->UnregisterBlendedTexture(timg);
    } else {
        gfx->RegisterBlendedTexture(timg, mask, replacementTex);
    }

    return false;
}

bool gfx_set_timg_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->Flush();
    gfx->mRapi->SelectTextureFb((uint32_t)cmd->words.w1);
    gfx->mRdp->textures_changed[0] = false;
    gfx->mRdp->textures_changed[1] = false;
    return false;
}

bool gfx_set_grayscale_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mRdp->grayscale = cmd->words.w1;
    return false;
}

bool gfx_load_block_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpLoadBlock(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_load_block_wide_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    uint32_t tile = cmd->words.w0 & 0x7;
    uint32_t lrs = cmd->words.w1;

    (*cmd0)++;
    cmd = *cmd0;

    uint32_t uls = (cmd->words.w0 >> 16) & 0xFFFF;
    uint32_t ult = (cmd->words.w0 >> 0) & 0xFFFF;
    uint32_t dxt = (cmd->words.w1 >> 0) & 0xFFF;

    gfx->GfxDpLoadBlock(tile, uls, ult, lrs, dxt);
    return false;
}

bool gfx_load_tile_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpLoadTile(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_set_tile_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetTile(C0(21, 3), C0(19, 2), C0(9, 9), C0(0, 9), C1(24, 3), C1(20, 4), C1(18, 2), C1(14, 4), C1(10, 4),
                      C1(8, 2), C1(4, 4), C1(0, 4));
    return false;
}

bool gfx_set_tile_size_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetTileSize(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_set_tile_size_interp_handler_rdp(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();

    if (gfx->mInterpolationIndex == gfx->mInterpolationIndexTarget) {
        int tile = C1(24, 3);
        gfx->GfxDpSetTileSize(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
        ++(*cmd0);
        memcpy(&gfx->mRdp->texture_tile[tile].uls, &(*cmd0)->words.w0, sizeof(float));
        memcpy(&gfx->mRdp->texture_tile[tile].ult, &(*cmd0)->words.w1, sizeof(float));
        ++(*cmd0);
        memcpy(&gfx->mRdp->texture_tile[tile].lrs, &(*cmd0)->words.w0, sizeof(float));
        memcpy(&gfx->mRdp->texture_tile[tile].lrt, &(*cmd0)->words.w1, sizeof(float));
    } else {
        ++(*cmd0);
        ++(*cmd0);
    }

    return false;
}

bool gfx_set_tile_size_lerp_handler_rdp(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();

    int tile = C1(24, 3);
    float coords[8];
    for (int i = 0; i < 8; i += 2) {
        ++(*cmd0);
        memcpy(&coords[i], &(*cmd0)->words.w0, sizeof(float));
        memcpy(&coords[i + 1], &(*cmd0)->words.w1, sizeof(float));
    }

    float t = gfx->mInterpolationT;
    gfx->mRdp->texture_tile[tile].uls = coords[0] + t * (coords[4] - coords[0]);
    gfx->mRdp->texture_tile[tile].ult = coords[1] + t * (coords[5] - coords[1]);
    gfx->mRdp->texture_tile[tile].lrs = coords[2] + t * (coords[6] - coords[2]);
    gfx->mRdp->texture_tile[tile].lrt = coords[3] + t * (coords[7] - coords[3]);
    gfx->mRdp->textures_changed[0] = true;
    gfx->mRdp->textures_changed[1] = true;

    if (gfx->mMM3DSuppress && gfx->mMM3DSegment != 0 && tile < 2) {
        // Tile size is 10.2 fixed point: scroll in texels = uls/4; window width = (lrs-uls)/4+1 = texture period.
        const auto& t = gfx->mRdp->texture_tile[tile];
        const float width = (t.lrs - t.uls) / 4.0f + 1.0f, height = (t.lrt - t.ult) / 4.0f + 1.0f;
        float* out = gfx->mMM3DScroll.offset[gfx->mMM3DSegment][tile];
        out[0] = width > 0 ? t.uls / 4.0f / width : 0.0f;
        out[1] = height > 0 ? t.ult / 4.0f / height : 0.0f;
    }

    return false;
}

bool gfx_set_interpolation_index_target(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();

    gfx->mInterpolationIndexTarget = cmd->words.w1;
    return false;
}

bool gfx_load_tlut_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpLoadTlut(C1(24, 3), C1(14, 10));
    return false;
}

bool gfx_set_env_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetEnvColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_prim_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetPrimColor(C0(8, 8), C0(0, 8), C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_fog_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetFogColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

// CENTER/SCALE and K4/K5 are wired as combiner inputs, so the standard (A-B)*C+D
// shader path covers their common uses. TODO: chroma-key width/threshold
// gating from G_SETKEYR/GB (wR/wG/wB ignored) and the YUV->RGB matrix K0..K3
// applied during texture sampling.
// G_SETKEYR: w1 = [wR:12 | cR:8 | sR:8]
bool gfx_set_key_r_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mRdp->key_center.r = C1(8, 8);
    gfx->mRdp->key_scale.r = C1(0, 8);
    return false;
}

// G_SETKEYGB: w0 = [op:8 | wG:12 | _:4 | wB:12], w1 = [cG:8 | sG:8 | cB:8 | sB:8]
bool gfx_set_key_gb_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mRdp->key_center.g = C1(24, 8);
    gfx->mRdp->key_scale.g = C1(16, 8);
    gfx->mRdp->key_center.b = C1(8, 8);
    gfx->mRdp->key_scale.b = C1(0, 8);
    return false;
}

// G_SETCONVERT: w0 = [op:8 | k0:9 | k1:9 | k2_hi:4], w1 = [k2_lo:5 | k3:9 | k4:9 | k5:9]
// K0..K5 are signed 9-bit values; sign-extend after decoding.
bool gfx_set_convert_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mRdp->convert_k[0] = sign_extend_9(C0(13, 9));
    gfx->mRdp->convert_k[1] = sign_extend_9(C0(4, 9));
    // k2 is split across w0 and w1
    gfx->mRdp->convert_k[2] = sign_extend_9((C0(0, 4) << 5) | C1(27, 5));
    gfx->mRdp->convert_k[3] = sign_extend_9(C1(18, 9));
    gfx->mRdp->convert_k[4] = sign_extend_9(C1(9, 9));
    gfx->mRdp->convert_k[5] = sign_extend_9(C1(0, 9));
    return false;
}

bool gfx_set_blend_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetBlendColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_fill_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetFillColor((uint32_t)cmd->words.w1);
    return false;
}

bool gfx_set_intensity_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetGrayscaleColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_combine_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetCombineMode(
        color_comb(C0(20, 4), C1(28, 4), C0(15, 5), C1(15, 3)), alpha_comb(C0(12, 3), C1(12, 3), C0(9, 3), C1(9, 3)),
        color_comb(C0(5, 4), C1(24, 4), C0(0, 5), C1(6, 3)), alpha_comb(C1(21, 3), C1(3, 3), C1(18, 3), C1(0, 3)));
    return false;
}

bool gfx_tex_rect_and_flip_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    int8_t opcode = (int8_t)(cmd->words.w0 >> 24);
    int32_t lrx, lry, tile, ulx, uly;
    uint32_t uls, ult, dsdx, dtdy;

    lrx = C0(12, 12);
    lry = C0(0, 12);
    tile = C1(24, 3);
    ulx = C1(12, 12);
    uly = C1(0, 12);
    // TODO make sure I don't need to increment cmd0
    ++(*cmd0);
    cmd = *cmd0;
    uls = C1(16, 16);
    ult = C1(0, 16);
    ++(*cmd0);
    cmd = *cmd0;
    dsdx = C1(16, 16);
    dtdy = C1(0, 16);

    gfx->GfxDpTextureRectangle(ulx, uly, lrx, lry, tile, uls, ult, dsdx, dtdy, opcode == RDP_G_TEXRECTFLIP);
    return false;
}

bool gfx_tex_rect_wide_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    int8_t opcode = (int8_t)(cmd->words.w0 >> 24);
    int32_t lrx, lry, tile, ulx, uly;
    uint32_t uls, ult, dsdx, dtdy;

    lrx = static_cast<int32_t>((C0(0, 24) << 8)) >> 8;
    lry = static_cast<int32_t>((C1(0, 24) << 8)) >> 8;
    tile = C1(24, 3);
    ++(*cmd0);
    cmd = *cmd0;
    ulx = static_cast<int32_t>((C0(0, 24) << 8)) >> 8;
    uly = static_cast<int32_t>((C1(0, 24) << 8)) >> 8;
    ++(*cmd0);
    cmd = *cmd0;
    uls = C0(16, 16);
    ult = C0(0, 16);
    dsdx = C1(16, 16);
    dtdy = C1(0, 16);
    gfx->GfxDpTextureRectangle(ulx, uly, lrx, lry, tile, uls, ult, dsdx, dtdy, opcode == RDP_G_TEXRECTFLIP);
    return false;
}

bool gfx_image_rect_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    int16_t tile, iw, ih;
    int16_t x0, y0, s0, t0;
    int16_t x1, y1, s1, t1;
    tile = C0(0, 3);
    iw = C1(16, 16);
    ih = C1(0, 16);
    cmd = ++(*cmd0);
    x0 = C0(16, 16);
    y0 = C0(0, 16);
    s0 = C1(16, 16);
    t0 = C1(0, 16);
    cmd = ++(*cmd0);
    x1 = C0(16, 16);
    y1 = C0(0, 16);
    s1 = C1(16, 16);
    t1 = C1(0, 16);
    gfx->GfxDpImageRectangle(tile, iw, ih, x0, y0, s0, t0, x1, y1, s1, t1);

    return false;
}

bool gfx_fill_rect_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpFillRectangle(C1(12, 12), C1(0, 12), C0(12, 12), C0(0, 12));
    return false;
}

bool gfx_fill_wide_rect_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);
    int32_t lrx, lry, ulx, uly;

    lrx = (int32_t)(C0(0, 24) << 8) >> 8;
    lry = (int32_t)(C1(0, 24) << 8) >> 8;
    cmd = ++(*cmd0);
    ulx = (int32_t)(C0(0, 24) << 8) >> 8;
    uly = (int32_t)(C1(0, 24) << 8) >> 8;
    gfx->GfxDpFillRectangle(ulx, uly, lrx, lry);

    return false;
}

bool gfx_SetScissor_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetScissor(C1(24, 2), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_set_z_img_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetZImage(gfx->SegAddr(cmd->words.w1));
    return false;
}

bool gfx_set_c_img_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetColorImage(C0(21, 3), C0(19, 2), C0(0, 11), gfx->SegAddr(cmd->words.w1));
    return false;
}

bool gfx_rdp_set_other_mode_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetOtherMode(C0(0, 24), (uint32_t)cmd->words.w1);
    return false;
}

bool gfx_bg_copy_handler_s2dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    if (!gfx->mMarkerOn) {
        gfx->Gfxs2dexBgCopy((F3DuObjBg*)cmd->words.w1); // not gfx->SegAddr here it seems
    }
    return false;
}

bool gfx_bg_1cyc_handler_s2dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->Gfxs2dexBg1cyc((F3DuObjBg*)cmd->words.w1);
    return false;
}

bool gfx_obj_rectangle_handler_s2dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    if (!gfx->mMarkerOn) {
        gfx->Gfxs2dexRecyCopy((F3DuObjSprite*)cmd->words.w1); // not gfx->SegAddr here it seems
    }
    return false;
}

bool gfx_extra_geometry_mode_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxSpExtraGeometryMode(~C0(0, 24), (uint32_t)cmd->words.w1);
    return false;
}

bool gfx_stubbed_command_handler(F3DGfx** cmd0) {
    return false;
}

bool gfx_spnoop_command_handler_f3dex2(F3DGfx** cmd0) {
    return false;
}

class UcodeHandler {
  public:
    inline constexpr UcodeHandler(
        std::initializer_list<std::pair<int8_t, std::pair<const char*, GfxOpcodeHandlerFunc>>> initializer) {
        std::fill(std::begin(mHandlers), std::end(mHandlers),
                  std::pair<const char*, GfxOpcodeHandlerFunc>(nullptr, nullptr));

        for (const auto& [opcode, handler] : initializer) {
            mHandlers[static_cast<uint8_t>(opcode)] = handler;
        }
    }

    inline bool contains(int8_t opcode) const {
        return mHandlers[static_cast<uint8_t>(opcode)].first != nullptr;
    }

    inline std::pair<const char*, GfxOpcodeHandlerFunc> at(int8_t opcode) const {
        return mHandlers[static_cast<uint8_t>(opcode)];
    }

  private:
    std::pair<const char*, GfxOpcodeHandlerFunc> mHandlers[std::numeric_limits<uint8_t>::max() + 1];
};

static constexpr UcodeHandler rdpHandlers = {
    { RDP_G_SETTARGETINTERPINDEX,
      { "G_SETTARGETINTERPINDEX", gfx_set_interpolation_index_target } }, // G_SETTARGETINTERPINDEX
    { RDP_G_SETTILESIZE_INTERP,
      { "G_SETTILESIZE_INTERP", gfx_set_tile_size_interp_handler_rdp } },                     // G_SETTILESIZE_INTERP
    { RDP_G_SETTILESIZE_LERP, { "G_SETTILESIZE_LERP", gfx_set_tile_size_lerp_handler_rdp } }, // G_SETTILESIZE_LERP
    { RDP_G_TEXRECT, { "G_TEXRECT", gfx_tex_rect_and_flip_handler_rdp } },                    // G_TEXRECT (-28)
    { RDP_G_TEXRECTFLIP, { "G_TEXRECTFLIP", gfx_tex_rect_and_flip_handler_rdp } },            // G_TEXRECTFLIP (-27)
    { RDP_G_RDPLOADSYNC, { "mRdpLOADSYNC", gfx_stubbed_command_handler } },                   // mRdpLOADSYNC (-26)
    { RDP_G_RDPPIPESYNC, { "mRdpPIPESYNC", gfx_stubbed_command_handler } },                   // mRdpPIPESYNC (-25)
    { RDP_G_RDPTILESYNC, { "mRdpTILESYNC", gfx_stubbed_command_handler } },                   // mRdpPIPESYNC (-24)
    { RDP_G_RDPFULLSYNC, { "mRdpFULLSYNC", gfx_stubbed_command_handler } },                   // mRdpFULLSYNC (-23)
    { RDP_G_SETKEYGB, { "G_SETKEYGB", gfx_set_key_gb_handler_rdp } },                         // G_SETKEYGB (-22)
    { RDP_G_SETKEYR, { "G_SETKEYR", gfx_set_key_r_handler_rdp } },                            // G_SETKEYR (-21)
    { RDP_G_SETCONVERT, { "G_SETCONVERT", gfx_set_convert_handler_rdp } },                    // G_SETCONVERT (-20)
    { RDP_G_SETSCISSOR, { "G_SETSCISSOR", gfx_SetScissor_handler_rdp } },                     // G_SETSCISSOR (-19)
    { RDP_G_SETPRIMDEPTH, { "G_SETPRIMDEPTH", gfx_set_prim_depth_handler_rdp } },             // G_SETPRIMDEPTH (-18)
    { RDP_G_RDPSETOTHERMODE, { "mRdpSETOTHERMODE", gfx_rdp_set_other_mode_rdp } },            // mRdpSETOTHERMODE (-17)
    { RDP_G_LOADTLUT, { "G_LOADTLUT", gfx_load_tlut_handler_rdp } },                          // G_LOADTLUT (-16)
    { RDP_G_SETTILESIZE, { "G_SETTILESIZE", gfx_set_tile_size_handler_rdp } },                // G_SETTILESIZE (-14)
    { RDP_G_LOADBLOCK, { "G_LOADBLOCK", gfx_load_block_handler_rdp } },                       // G_LOADBLOCK (-13)
    { RDP_G_LOADTILE, { "G_LOADTILE", gfx_load_tile_handler_rdp } },                          // G_LOADTILE (-12)
    { RDP_G_SETTILE, { "G_SETTILE", gfx_set_tile_handler_rdp } },                             // G_SETTILE (-11)
    { RDP_G_FILLRECT, { "G_FILLRECT", gfx_fill_rect_handler_rdp } },                          // G_FILLRECT (-10)
    { RDP_G_SETFILLCOLOR, { "G_SETFILLCOLOR", gfx_set_fill_color_handler_rdp } },             // G_SETFILLCOLOR (-9)
    { RDP_G_SETFOGCOLOR, { "G_SETFOGCOLOR", gfx_set_fog_color_handler_rdp } },                // G_SETFOGCOLOR (-8)
    { RDP_G_SETBLENDCOLOR, { "G_SETBLENDCOLOR", gfx_set_blend_color_handler_rdp } },          // G_SETBLENDCOLOR (-7)
    { RDP_G_SETPRIMCOLOR, { "G_SETPRIMCOLOR", gfx_set_prim_color_handler_rdp } },             // G_SETPRIMCOLOR (-6)
    { RDP_G_SETENVCOLOR, { "G_SETENVCOLOR", gfx_set_env_color_handler_rdp } },                // G_SETENVCOLOR (-5)
    { RDP_G_SETCOMBINE, { "G_SETCOMBINE", gfx_set_combine_handler_rdp } },                    // G_SETCOMBINE (-4)
    { RDP_G_SETTIMG, { "G_SETTIMG", gfx_set_timg_handler_rdp } },                             // G_SETTIMG (-3)
    { RDP_G_SETZIMG, { "G_SETZIMG", gfx_set_z_img_handler_rdp } },                            // G_SETZIMG (-2)
    { RDP_G_SETCIMG, { "G_SETCIMG", gfx_set_c_img_handler_rdp } },                            // G_SETCIMG (-1)
};

static constexpr UcodeHandler otrHandlers = {
    { OTR_G_SETTIMG_OTR_HASH,
      { "G_SETTIMG_OTR_HASH", gfx_set_timg_otr_hash_handler_custom } },       // G_SETTIMG_OTR_HASH (0x20)
    { OTR_G_SETFB, { "G_SETFB", gfx_set_fb_handler_custom } },                // G_SETFB (0x21)
    { OTR_G_RESETFB, { "G_RESETFB", gfx_reset_fb_handler_custom } },          // G_RESETFB (0x22)
    { OTR_G_SETTIMG_FB, { "G_SETTIMG_FB", gfx_set_timg_fb_handler_custom } }, // G_SETTIMG_FB (0x23)
    { OTR_G_VTX_OTR_FILEPATH,
      { "G_VTX_OTR_FILEPATH", gfx_vtx_otr_filepath_handler_custom } }, // G_VTX_OTR_FILEPATH (0x24)
    { OTR_G_SETTIMG_OTR_FILEPATH,
      { "G_SETTIMG_OTR_FILEPATH", gfx_set_timg_otr_filepath_handler_custom } }, // G_SETTIMG_OTR_FILEPATH (0x25)
    { OTR_G_TRI1_OTR, { "G_TRI1_OTR", gfx_tri1_otr_handler_f3dex2 } },          // G_TRI1_OTR (0x26)
    { OTR_G_DL_OTR_FILEPATH, { "G_DL_OTR_FILEPATH", gfx_dl_otr_filepath_handler_custom } }, // G_DL_OTR_FILEPATH (0x27)
    { OTR_G_PUSHCD, { "G_PUSHCD", gfx_pushcd_handler_custom } },                            // G_PUSHCD (0x28)
    { OTR_G_MTX_OTR_FILEPATH,
      { "G_MTX_OTR_FILEPATH", gfx_mtx_otr_filepath_handler_custom } },          // G_MTX_OTR_FILEPATH (0x29)
    { OTR_G_DL_OTR_HASH, { "G_DL_OTR_HASH", gfx_dl_otr_hash_handler_custom } }, // G_DL_OTR_HASH (0x31)
    { OTR_G_VTX_OTR_HASH, { "G_VTX_OTR_HASH", gfx_vtx_hash_handler_custom } },  // G_VTX_OTR_HASH (0x32)
    { OTR_G_MARKER, { "G_MARKER", gfx_marker_handler_otr } },                   // G_MARKER (0X33)
    { OTR_G_INVALTEXCACHE, { "G_INVALTEXCACHE", gfx_invalidate_tex_cache_handler_f3dex2 } }, // G_INVALTEXCACHE (0X34)
    { OTR_G_BRANCH_Z_OTR, { "G_BRANCH_Z_OTR", gfx_branch_z_otr_handler_f3dex2 } },           // G_BRANCH_Z_OTR (0x35)
    { OTR_G_MTX_OTR, { "G_MTX_OTR", gfx_mtx_otr_handler_custom } },                          // G_MTX_OTR (0x36)
    { OTR_G_TEXRECT_WIDE, { "G_TEXRECT_WIDE", gfx_tex_rect_wide_handler_custom } },          // G_TEXRECT_WIDE (0x37)
    { OTR_G_FILLWIDERECT, { "G_FILLWIDERECT", gfx_fill_wide_rect_handler_custom } },         // G_FILLWIDERECT (0x38)
    { OTR_G_SETGRAYSCALE, { "G_SETGRAYSCALE", gfx_set_grayscale_handler_custom } },          // G_SETGRAYSCALE (0x39)
    { OTR_G_EXTRAGEOMETRYMODE,
      { "G_EXTRAGEOMETRYMODE", gfx_extra_geometry_mode_handler_custom } }, // G_EXTRAGEOMETRYMODE (0x3a)
    { OTR_G_COPYFB, { "G_COPYFB", gfx_copy_fb_handler_custom } },          // G_COPYFB (0x3b)
    { OTR_G_IMAGERECT, { "G_IMAGERECT", gfx_image_rect_handler_custom } }, // G_IMAGERECT (0x3c)
    { OTR_G_DL_INDEX, { "G_DL_INDEX", gfx_dl_index_handler } },            // G_DL_INDEX (0x3d)
    { OTR_G_READFB, { "G_READFB", gfx_read_fb_handler_custom } },          // G_READFB (0x3e)
    { OTR_G_REGBLENDEDTEX,
      { "G_REGBLENDEDTEX", gfx_register_blended_texture_handler_custom } },         // G_REGBLENDEDTEX (0x3f)
    { OTR_G_SETINTENSITY, { "G_SETINTENSITY", gfx_set_intensity_handler_custom } }, // G_SETINTENSITY (0x40)
    { OTR_G_MOVEMEM_HASH, { "OTR_G_MOVEMEM_HASH", gfx_movemem_handler_otr } },      // OTR_G_MOVEMEM_HASH
    { OTR_G_PUSH_SHADER, { "G_PUSH_SHADER", gfx_push_shader } },
    { OTR_G_POP_SHADER, { "G_POP_SHADER", gfx_pop_shader } },
    { OTR_G_NEWDAWN_POSTFX, { "G_NEWDAWN_POSTFX", gfx_newdawn_postfx_handler_custom } },
    { OTR_G_NEWDAWN_MM3D, { "G_NEWDAWN_MM3D", gfx_newdawn_mm3d_handler_custom } },
    { RDP_G_LOADBLOCK_WIDE, { "G_LOADBLOCK_WIDE", gfx_load_block_wide_handler_rdp } }, // RDP_G_LOADBLOCK_WIDE (-15)
    { RDP_G_VTX_WIDE, { "G_VTX_WIDE", gfx_vtx_handler_f3dex2 } },                      // RDP_G_VTX_WIDE (-16)
    { RDP_G_TRI1_WIDE, { "G_TRI1_WIDE", gfx_tri1_handler_f3dex2 } },                   // RDP_G_TRI1_WIDE (-17)
};

static constexpr UcodeHandler f3dex2Handlers = {
    { F3DEX2_G_NOOP, { "G_NOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX2_G_SPNOOP, { "G_SPNOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX2_G_CULLDL, { "G_CULLDL", gfx_cull_dl_handler_f3dex2 } },
    { F3DEX2_G_MTX, { "G_MTX", gfx_mtx_handler_f3dex2 } },
    { F3DEX2_G_POPMTX, { "G_POPMTX", gfx_pop_mtx_handler_f3dex2 } },
    { F3DEX2_G_MOVEMEM, { "G_MOVEMEM", gfx_movemem_handler_f3dex2 } },
    { F3DEX2_G_MOVEWORD, { "G_MOVEWORD", gfx_moveword_handler_f3dex2 } },
    { F3DEX2_G_TEXTURE, { "G_TEXTURE", gfx_texture_handler_f3dex2 } },
    { F3DEX2_G_VTX, { "G_VTX", gfx_vtx_handler_f3dex2 } },
    { F3DEX2_G_MODIFYVTX, { "G_MODIFYVTX", gfx_modify_vtx_handler_f3dex2 } },
    { F3DEX2_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX2_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
    { F3DEX2_G_GEOMETRYMODE, { "G_GEOMETRYMODE", gfx_geometry_mode_handler_f3dex2 } },
    { F3DEX2_G_TRI1, { "G_TRI1", gfx_tri1_handler_f3dex2 } },
    { F3DEX2_G_TRI2, { "G_TRI2", gfx_tri2_handler_f3dex } },
    { F3DEX2_G_QUAD, { "G_QUAD", gfx_quad_handler_f3dex2 } },
    { F3DEX2_G_SETOTHERMODE_L, { "G_SETOTHERMODE_L", gfx_othermode_l_handler_f3dex2 } },
    { F3DEX2_G_SETOTHERMODE_H, { "G_SETOTHERMODE_H", gfx_othermode_h_handler_f3dex2 } },
};

static constexpr UcodeHandler f3dexHandlers = {
    { F3DEX_G_NOOP, { "G_NOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX_G_CULLDL, { "G_CULLDL", gfx_cull_dl_handler_f3dex2 } },
    { F3DEX_G_MTX, { "G_MTX", gfx_mtx_handler_f3d } },
    { F3DEX_G_POPMTX, { "G_POPMTX", gfx_pop_mtx_handler_f3d } },
    { F3DEX_G_MOVEMEM, { "G_POPMEM", gfx_movemem_handler_f3d } },
    { F3DEX_G_MOVEWORD, { "G_MOVEWORD", gfx_moveword_handler_f3d } },
    { F3DEX_G_TEXTURE, { "G_TEXTURE", gfx_texture_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_L, { "G_SETOTHERMODE_L", gfx_othermode_l_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_H, { "G_SETOTHERMODE_H", gfx_othermode_h_handler_f3d } },
    { F3DEX_G_SETGEOMETRYMODE, { "G_SETGEOMETRYMODE", gfx_set_geometry_mode_handler_f3dex } },
    { F3DEX_G_CLEARGEOMETRYMODE, { "G_CLEARGEOMETRYMODE", gfx_clear_geometry_mode_handler_f3dex } },
    { F3DEX_G_VTX, { "G_VTX", gfx_vtx_handler_f3dex } },
    { F3DEX_G_TRI1, { "G_TRI1", gfx_tri1_handler_f3dex } },
    { F3DEX_G_MODIFYVTX, { "G_MODIFYVTX", gfx_modify_vtx_handler_f3dex2 } },
    { F3DEX_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
    { F3DEX_G_TRI2, { "G_TRI2", gfx_tri2_handler_f3dex } },
    { F3DEX_G_SPNOOP, { "G_SPNOOP", gfx_spnoop_command_handler_f3dex2 } },
    { F3DEX_G_RDPHALF_1, { "mRdpHALF_1", gfx_stubbed_command_handler } },
    { F3DEX_G_QUAD, { "G_QUAD", gfx_quad_handler_f3dex } },
};

static constexpr UcodeHandler f3dHandlers = {
    { F3DEX_G_NOOP, { "G_NOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX_G_CULLDL, { "G_CULLDL", gfx_cull_dl_handler_f3dex2 } },
    { F3DEX_G_MTX, { "G_MTX", gfx_mtx_handler_f3d } },
    { F3DEX_G_POPMTX, { "G_POPMTX", gfx_pop_mtx_handler_f3d } },
    { F3DEX_G_MOVEMEM, { "G_POPMEM", gfx_movemem_handler_f3d } },
    { F3DEX_G_MOVEWORD, { "G_MOVEWORD", gfx_moveword_handler_f3d } },
    { F3DEX_G_TEXTURE, { "G_TEXTURE", gfx_texture_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_L, { "G_SETOTHERMODE_L", gfx_othermode_l_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_H, { "G_SETOTHERMODE_H", gfx_othermode_h_handler_f3d } },
    { F3DEX_G_SETGEOMETRYMODE, { "G_SETGEOMETRYMODE", gfx_set_geometry_mode_handler_f3dex } },
    { F3DEX_G_CLEARGEOMETRYMODE, { "G_CLEARGEOMETRYMODE", gfx_clear_geometry_mode_handler_f3dex } },
    { F3DEX_G_VTX, { "G_VTX", gfx_vtx_handler_f3d } },
    { F3DEX_G_TRI1, { "G_TRI1", gfx_tri1_handler_f3d } },
    { F3DEX_G_MODIFYVTX, { "G_MODIFYVTX", gfx_modify_vtx_handler_f3dex2 } },
    { F3DEX_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
    { F3DEX_G_TRI2, { "G_TRI2", gfx_tri2_handler_f3dex } },
    { F3DEX_G_SPNOOP, { "G_SPNOOP", gfx_spnoop_command_handler_f3dex2 } },
    { F3DEX_G_RDPHALF_1, { "mRdpHALF_1", gfx_stubbed_command_handler } },
};

// LUSTODO: These S2DEX commands have different opcode numbers on F3DEX2 vs other ucodes. More research needs to be done
// to see if the implementations are different.
static constexpr UcodeHandler s2dexHandlers = {
    { F3DEX2_G_BG_COPY, { "G_BG_COPY", gfx_bg_copy_handler_s2dex } },
    { F3DEX2_G_BG_1CYC, { "G_BG_1CYC", gfx_bg_1cyc_handler_s2dex } },
    { F3DEX2_G_OBJ_RENDERMODE, { "G_OBJ_RENDERMODE", gfx_stubbed_command_handler } },
    { F3DEX2_G_OBJ_RECTANGLE_R, { "G_OBJ_RECTANGLE_R", gfx_stubbed_command_handler } },
    { F3DEX2_G_OBJ_RECTANGLE, { "G_OBJ_RECTANGLE", gfx_obj_rectangle_handler_s2dex } },
    { F3DEX2_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX2_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
};

static constexpr std::array ucode_handlers = {
    &f3dHandlers,    // ucode_f3db
    &f3dHandlers,    // ucode_f3d
    &f3dexHandlers,  // ucode_f3dex
    &f3dexHandlers,  // ucode_f3dexb
    &f3dex2Handlers, // ucode_f3dex2
    &s2dexHandlers,  // ucode_s2dex
};

const char* GfxGetOpcodeName(int8_t opcode) {
    if (otrHandlers.contains(opcode)) {
        return otrHandlers.at(opcode).first;
    }

    if (rdpHandlers.contains(opcode)) {
        return rdpHandlers.at(opcode).first;
    }

    if (ucode_handler_index < ucode_handlers.size()) {
        if (ucode_handlers[ucode_handler_index]->contains(opcode)) {
            return ucode_handlers[ucode_handler_index]->at(opcode).first;
        } else {
            SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, for loaded ucode: {}", (uint8_t)opcode,
                            (uint32_t)ucode_handler_index);
        }
    } else {
        SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, invalid ucode: {}", (uint8_t)opcode, (uint32_t)ucode_handler_index);
    }

    return nullptr;
}

// TODO, implement a system where we can get the current opcode handler by writing to the GWords. If the powers that be
// are OK with that...
static void gfx_set_ucode_handler(UcodeHandlers ucode) {
    // Loaded ucode must be in range of the supported ucode_handlers
    assert(ucode < ucode_max);
    Interpreter* gfx = mInstance.lock().get();
    ucode_handler_index = ucode;

    // Reset some RSP state values upon ucode load to deal with hardware quirks discovered by emulators
    switch (ucode) {
        case ucode_f3d:
        case ucode_f3db:
        case ucode_f3dex:
        case ucode_f3dexb:
        case ucode_f3dex2:
            gfx->mRsp->fog_mul = 0;
            gfx->mRsp->fog_offset = 0;
            break;
        default:
            break;
    }
}

static void gfx_step() {
    auto& cmd = g_exec_stack.currCmd();
    auto cmd0 = cmd;
    int8_t opcode = (int8_t)(cmd->words.w0 >> 24);

#ifdef USE_GBI_TRACE
    if (cmd->words.trace.valid &&
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gEnableGFXTrace", 0)) {
#define TRACE                                  \
    "\n====================================\n" \
    " - CMD: {:02X}\n"                         \
    " - Path: {}:{}\n"                         \
    " - W0: {:08X}\n"                          \
    " - W1: {:08X}\n"                          \
    "===================================="
        SPDLOG_INFO(TRACE, (uint8_t)opcode, cmd->words.trace.file, cmd->words.trace.idx, cmd->words.w0, cmd->words.w1);
    }
#endif

    if (opcode == F3DEX2_G_LOAD_UCODE) {
        gfx_set_ucode_handler((UcodeHandlers)(cmd->words.w0 & 0xFFFFFF));
        ++cmd;
        return;
        // Instead of having a handler for each ucode for switching ucode, just check for it early and return.
    }

    if (otrHandlers.contains(opcode)) {
        // OTR filepath handlers expect w1 to be a valid string pointer.
        // Guard against null or N64-segment addresses that would crash in strlen/strncmp.
        if (opcode == OTR_G_VTX_OTR_FILEPATH || opcode == OTR_G_SETTIMG_OTR_FILEPATH ||
            opcode == OTR_G_DL_OTR_FILEPATH || opcode == OTR_G_PUSHCD || opcode == OTR_G_MTX_OTR_FILEPATH) {
            uintptr_t w1 = (uintptr_t)cmd->words.w1;
            if (w1 < 0x10000
#if UINTPTR_MAX > 0xFFFFFFFFu
                // On 64-bit: filter kernel/sentinel addresses.
                || w1 > 0x0000FFFFFFFFFFFFull
#endif
            ) {
                ++g_exec_stack.currCmd();
                return;
            }
        }
        if (otrHandlers.at(opcode).second(&cmd)) {
            return;
        }
    } else if (rdpHandlers.contains(opcode)) {
        if (rdpHandlers.at(opcode).second(&cmd)) {
            return;
        }
    } else if (ucode_handler_index < ucode_handlers.size()) {
        if (ucode_handlers[ucode_handler_index]->contains(opcode)) {
            if (ucode_handlers[ucode_handler_index]->at(opcode).second(&cmd)) {
                return;
            }
        } else {
            SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, for loaded ucode: {}", (uint8_t)opcode,
                            (uint32_t)ucode_handler_index);
        }
    } else {
        SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, invalid ucode: {}", (uint8_t)opcode, (uint32_t)ucode_handler_index);
    }

    ++cmd;
}

void Interpreter::SpReset() {
    while (!mShaderStack.empty()) {
        mShaderStack.pop();
    }
    mRsp->modelview_matrix_stack_size = 1;
    mRsp->current_num_lights = 2;
    mRsp->lights_changed = true;
    mRsp->light_state_revision = 1;
    // Force the next per-pixel lighting batch to rebuild its uniforms from the new scene's lights.
    mBatchLightRevision = ~0u;
    mBatchShadeMask = 0xFFFFFFFFu;
    mRsp->lookat[0].dir[0] = 0;
    mRsp->lookat[0].dir[1] = 127;
    mRsp->lookat[0].dir[2] = 0;
    mRsp->lookat[1].dir[0] = 127;
    mRsp->lookat[1].dir[1] = 0;
    mRsp->lookat[1].dir[2] = 0;
    CalculateNormalDir(&mRsp->lookat[0], mRsp->current_lookat_coeffs[0]);
    CalculateNormalDir(&mRsp->lookat[1], mRsp->current_lookat_coeffs[1]);
}

void Interpreter::RegisterFbTexture(const void* cpuAddr, int fbId) {
    mFbTextures[(uintptr_t)cpuAddr] = fbId;
}

void Interpreter::UnregisterFbTexture(const void* cpuAddr) {
    mFbTextures.erase((uintptr_t)cpuAddr);
}

void Interpreter::GetDimensions(uint32_t* width, uint32_t* height, int32_t* posX, int32_t* posY) {
    mWapi->GetDimensions(width, height, posX, posY);
}

void Interpreter::Init(class GfxWindowBackend* wapi, class GfxRenderingAPI* rapi, const char* game_name,
                       bool start_in_fullscreen, uint32_t width, uint32_t height, uint32_t posX, uint32_t posY) {
    mWapi = wapi;
    mRapi = rapi;
    mWapi->Init(game_name, rapi->GetName(), start_in_fullscreen, width, height, posX, posY);
    mRapi->Init();
    mRapi->UpdateFramebufferParameters(0, width, height, 1, false, true, true, true);
    mCurDimensions.internal_mul =
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetFloat(CVAR_INTERNAL_RESOLUTION, 1);
    mMsaaLevel = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger(CVAR_MSAA_VALUE, 1);

    mCurDimensions.width = width;
    mCurDimensions.height = height;

    mGameFb = mRapi->CreateFramebuffer();
    mGameFbMsaaResolved = mRapi->CreateFramebuffer();

    mNativeDimensions.width = SCREEN_WIDTH;
    mNativeDimensions.height = SCREEN_HEIGHT;

    for (int i = 0; i < MAX_SEGMENT_POINTERS; i++) {
        mSegmentPointers[i] = 0;
    }

    if (mTexUploadBuffer == nullptr) {
        // We cap texture max to 8k, because why would you need more?
        int max_tex_size = std::min(8192, mRapi->GetMaxTextureSize());
        mTexUploadBuffer = (uint8_t*)malloc(max_tex_size * max_tex_size * 4);
    }

    ucode_handler_index = UcodeHandlers::ucode_f3dex2;

    // Pre-allocate texture cache buckets to prevent rehash-induced iterator invalidation.
    mTextureCache.map.reserve(TEXTURE_CACHE_MAX_SIZE);
}

void Interpreter::Destroy() {
    // TODO: should also destroy rapi, and any other resources acquired in fast3d
    free(mTexUploadBuffer);
    mWapi->Destroy();

    // Texture cache and loaded textures store references to Resources which need to be unreferenced.
    TextureCacheClear();
    mRdp->texture_to_load.raw_tex_metadata.resource = nullptr;
    mRdp->loaded_texture[0].raw_tex_metadata.resource = nullptr;
    mRdp->loaded_texture[1].raw_tex_metadata.resource = nullptr;
}

GfxRenderingAPI* Interpreter::GetCurrentRenderingAPI() {
    return mRapi;
}

void Interpreter::SetGfxDebugger(std::shared_ptr<GfxDebugger> debugger) {
    mGfxDebugger = std::move(debugger);
}

std::shared_ptr<GfxDebugger> Interpreter::GetGfxDebugger() const {
    return mGfxDebugger;
}

void Interpreter::HandleWindowEvents() {
    mWapi->HandleEvents();
}

bool Interpreter::IsFrameReady() {
    return mWapi->IsFrameReady();
}

bool Interpreter::ViewportMatchesRendererResolution() {
#ifdef __APPLE__
    // Always treat the viewport as not matching the render resolution on mac
    // to avoid issues with retina scaling.
    return false;
#else
    if (mCurDimensions.width == mGameWindowViewport.width && mCurDimensions.height == mGameWindowViewport.height) {
        return true;
    }
    return false;
#endif
}

bool Interpreter::GameFbIsWindowSized() {
    return mMsaaLevel > 1 && !mPostFxActive && mCurDimensions.width == mGameWindowViewport.width &&
           mCurDimensions.height == mGameWindowViewport.height;
}

void Interpreter::StartFrame() {
    mMM3DSuppress = false;
    PumpAltTextureLoads();
    mWapi->GetDimensions(&mGfxCurrentWindowDimensions.width, &mGfxCurrentWindowDimensions.height, &mCurWindowPosX,
                         &mCurWindowPosY);
    if (mCurDimensions.height == 0) {
        // Avoid division by zero
        mCurDimensions.height = 1;
    }
    mCurDimensions.aspect_ratio = (float)mCurDimensions.width / (float)mCurDimensions.height;

    // Update the framebuffer sizes when the viewport or native dimension changes
    if (mCurDimensions.width != mPrvDimensions.width || mCurDimensions.height != mPrvDimensions.height ||
        mNativeDimensions.width != mPrevNativeDimensions.width ||
        mNativeDimensions.height != mPrevNativeDimensions.height) {

        for (auto& fb : mFrameBuffers) {
            uint32_t width = fb.second.orig_width, height = fb.second.orig_height;
            if (fb.second.resize) {
                AdjustWidthHeightForScale(width, height, fb.second.native_width, fb.second.native_height);
            }
            if (width != fb.second.applied_width || height != fb.second.applied_height) {
                mRapi->UpdateFramebufferParameters(fb.first, width, height, 1, true, true, true, true);
                fb.second.applied_width = width;
                fb.second.applied_height = height;
            }
        }
    }

    mPrvDimensions = mCurDimensions;
    mPrevNativeDimensions = mNativeDimensions;
    // Post-processing and shadows need the game rendered offscreen even when the window could be rendered to directly
    const bool postFxEnabled =
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.PostFx.Enabled", 1) != 0;
    const bool shadowsEnabled =
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Shadows.Enabled", 0) != 0;
    const bool waterEnabled =
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Water.Enabled", 0) != 0 ||
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Water.Reflections", 0) != 0;
    mPostFxActive =
        mRapi->PostFxAvailable() &&
        (postFxEnabled || shadowsEnabled || waterEnabled ||
         Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.HDR.Enabled", 0) != 0);
    mShadowRecording = false;
    mGiRecord = false;
    mGiTris.clear();
    mShadowOverflowLogged = false;
    mShadowVertices.clear();
    mShadowFoliage = false;
    mShadowFoliageByTex.clear();
    // New Dawn per-pixel lighting; a backend only gets it when it implements the shader side.
    mPerPixelLightingEnabled =
        mRapi->SupportsPerPixelLighting() &&
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Lighting.PerPixel", 0) != 0;
    mMaterialsEnabled =
        mPerPixelLightingEnabled && mRapi->SupportsMaterialMaps() &&
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Materials.Enabled", 0) != 0 &&
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetFloat("gNewDawn.Materials.Strength", 1.0f) > 0.0f;
    PumpMaterialLoads();
    mAlphaToCoverageEnabled =
        mRapi->SupportsAlphaToCoverage() &&
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Textures.AlphaToCoverage", 1) != 0;
    mCharacterLighting = mPerPixelLightingEnabled &&
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Lighting.Characters", 0) != 0;
    mCharacterStrength = std::clamp(
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetFloat("gNewDawn.Lighting.CharactersStrength", 1.0f),
        0.0f, 2.0f);
    mPerPixelPointLightQuality = std::clamp(
        Ship::Context::GetRawInstance()->GetConsoleVariables()->GetFloat("gNewDawn.Lighting.PointLightQuality", 0.0f),
        0.0f, 1.0f);
    const bool renderSizedFb = !ViewportMatchesRendererResolution() || mPostFxActive;
    if (renderSizedFb || mMsaaLevel > 1) {
        mRendersToFb = true;
        if (renderSizedFb) {
            mRapi->UpdateFramebufferParameters(mGameFb, mCurDimensions.width, mCurDimensions.height, mMsaaLevel, true,
                                               true, true, true);
        } else {
            // MSAA framebuffer needs to be resolved to an equally sized target when complete, which must therefore
            // match the window size
            mRapi->UpdateFramebufferParameters(mGameFb, mGfxCurrentWindowDimensions.width,
                                               mGfxCurrentWindowDimensions.height, mMsaaLevel, false, true, true, true);
        }
        if (mMsaaLevel > 1 && renderSizedFb) {
            mRapi->UpdateFramebufferParameters(mGameFbMsaaResolved, mCurDimensions.width, mCurDimensions.height, 1,
                                               false, false, false, false);
        }
    } else {
        mRendersToFb = false;
    }

    auto cvars = Ship::Context::GetRawInstance()->GetConsoleVariables();
    {
        // New Dawn smooth characters (docs/SMOOTH_CHARACTERS.md). The cached analysis depends on what counts as a
        // character and on the crease angle, so a change of either starts it again.
        const bool props = cvars->GetInteger("gNewDawn.Characters.SmoothProps", 0) != 0;
        const float crease = std::cos(
            std::clamp(cvars->GetFloat("gNewDawn.Characters.SmoothCreaseAngle", 35.0f), 1.0f, 89.0f) * 0.01745329f);
        if (props != mSmoothProps || crease != mSmoothCrease || mSmoothCount > 120000) {
            mSmoothTable.clear();
            mSmoothCount = 0;
        }
        mSmoothProps = props;
        mSmoothCrease = crease;
        mSmooth = mRapi->SupportsSmoothCharacters() && cvars->GetInteger("gNewDawn.Characters.Smooth", 0) != 0;
        mSmoothStrength = std::clamp(cvars->GetFloat("gNewDawn.Characters.SmoothStrength", 1.0f), 0.0f, 1.5f);
        mSmoothTip = std::clamp(cvars->GetFloat("gNewDawn.Characters.SmoothTipLimit", 0.8f), 0.1f, 1.0f);
        mSmoothThin = std::clamp(cvars->GetFloat("gNewDawn.Characters.SmoothThinLimit", 1.0f), 0.0f, 8.0f);
        mSmoothTolerance = std::clamp(cvars->GetFloat("gNewDawn.Characters.SmoothTolerance", 0.35f), 0.05f, 8.0f);
        mSmoothMaxLevel = std::clamp(cvars->GetInteger("gNewDawn.Characters.SmoothMaxLevel", 8), 1, 16);
        mSmoothDebug = cvars->GetInteger("gNewDawn.Characters.SmoothDebug", 0) != 0;
    }
    int cacheBudgetMb = cvars->GetInteger("gNewDawn.Textures.CacheBudgetMB", 2048);
    mTextureCacheBudgetBytes = cacheBudgetMb > 0 ? (uint64_t)cacheBudgetMb * 1024 * 1024 : 0;
    mTextureCacheDebug = cvars->GetInteger("gNewDawn.Textures.CacheDebug", 0) != 0;
    // Apply a lowered budget even when no new texture is uploaded this frame.
    EnforceTextureCacheBudget();

    mFbActive = false;
}

GfxExecStack g_exec_stack = {};

void Interpreter::RunGuiOnly() {
    SpReset();

    mShadowOnly = false;
    mWindSway = false;

    mGetPixelDepthPending.clear();
    mGetPixelDepthCached.clear();

    mRapi->UpdateFramebufferParameters(0, mGfxCurrentWindowDimensions.width, mGfxCurrentWindowDimensions.height, 1,
                                       false, true, true, !mRendersToFb);
    mRapi->StartFrame();
    mRapi->StartDrawToFramebuffer(mRendersToFb ? mGameFb : 0, (float)mCurDimensions.height / mNativeDimensions.height);
    mRapi->ClearFramebuffer(true, true);
    mRdp->viewport_or_scissor_changed = true;
    mRenderingState.viewport = {};
    mRenderingState.scissor = {};

    Flush();
    mGfxFrameBuffer = 0;

    if (mRendersToFb) {
        mRapi->StartDrawToFramebuffer(0, 1);
        mRapi->ClearFramebuffer(true, true);
        if (mMsaaLevel > 1) {
            if (!ViewportMatchesRendererResolution() || mPostFxActive) {
                mRapi->ResolveMSAAColorBuffer(mGameFbMsaaResolved, mGameFb);
                mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFbMsaaResolved);
            } else {
                mRapi->ResolveMSAAColorBuffer(0, mGameFb);
            }
        } else {
            mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFb);
        }
    } else if (mFbActive) {
        // Failsafe reset to main framebuffer to prevent softlocking the renderer
        mFbActive = 0;
        mRapi->StartDrawToFramebuffer(0, 1);

        assert(0 && "active framebuffer was never reset back to original");
    }
}

void Interpreter::Run(Gfx* commands, const std::unordered_map<Mtx*, MtxF>& mtx_replacements) {
    SpReset();

    mWindSway = false;
    mWindProfile = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.PostFx.Debug.Stats", 0) &&
                   Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Atmos.Wind", 0);
    if (!mWindProfile) {
        mWindProfileFrames = 0;
        mWindCpuNs = mWindTriangles = 0;
    }
    if (Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Atmos.Wind", 0)) {
        const double now = NewDawnWindSeconds();
        mWindT = float(now);
        mWindState = NewDawnWindAt(now);
        mWindScale = std::clamp(
            Ship::Context::GetRawInstance()->GetConsoleVariables()->GetFloat("gNewDawn.Atmos.WindScale", 1.0f),
            0.0f, 10.0f);
    }

    mGetPixelDepthPending.clear();
    mGetPixelDepthCached.clear();

    mCurMtxReplacements = &mtx_replacements;

    mRapi->UpdateFramebufferParameters(0, mGfxCurrentWindowDimensions.width, mGfxCurrentWindowDimensions.height, 1,
                                       false, true, true, !mRendersToFb);
    mRapi->StartFrame();
    mRapi->StartDrawToFramebuffer(mRendersToFb ? mGameFb : 0, (float)mCurDimensions.height / mNativeDimensions.height);
    mRapi->ClearFramebuffer(true, true);
    mRdp->viewport_or_scissor_changed = true;
    mRenderingState.viewport = {};
    mRenderingState.scissor = {};

    SmoothBeginFrame();
    auto dbg = mGfxDebugger;
    g_exec_stack.start((F3DGfx*)commands);
    while (!g_exec_stack.cmd_stack.empty()) {
        auto cmd = g_exec_stack.cmd_stack.top();

        if (dbg->IsDebugging()) {
            g_exec_stack.gfx_path.push_back(cmd);
            if (dbg->HasBreakPoint(g_exec_stack.gfx_path)) {
                // On a breakpoint with the active framebuffer still set, we need to reset back to prevent
                // soft locking the renderer
                if (mFbActive) {
                    mFbActive = 0;
                    mRapi->StartDrawToFramebuffer(mRendersToFb ? mGameFb : 0, 1);
                }

                break;
            }
            g_exec_stack.gfx_path.pop_back();
        }
        gfx_step();
    }

    Flush();
    if (mSmooth) {
        SmoothFinishActor();
    }
    mTaaJitter = false;
    mRapi->FinishWorld(); // also closes a partial/debugger display-list walk
    if (mWindProfile && ++mWindProfileFrames == 300) {
        SPDLOG_INFO("NewDawn wind CPU: {:.4f} ms/frame, {:.1f} triangles/frame (300 frames, timer overhead included)",
                    double(mWindCpuNs) / 300000000.0, double(mWindTriangles) / 300.0);
        mWindProfileFrames = 0;
        mWindCpuNs = mWindTriangles = 0;
    }
    if (mRenderingState.alpha_to_coverage) {
        mRapi->SetAlphaToCoverage(false); // never leak into the post passes, the resolve or the UI
        mRenderingState.alpha_to_coverage = false;
    }
    mGfxFrameBuffer = 0;
    currentDir = std::stack<std::string>();

    if (mRendersToFb) {
        mRapi->StartDrawToFramebuffer(0, 1);
        mRapi->ClearFramebuffer(true, true);
        if (mMsaaLevel > 1) {
            if (!ViewportMatchesRendererResolution() || mPostFxActive) {
                mRapi->ResolveMSAAColorBuffer(mGameFbMsaaResolved, mGameFb);
                mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFbMsaaResolved);
            } else {
                mRapi->ResolveMSAAColorBuffer(0, mGameFb);
            }
        } else {
            mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFb);
        }
    } else if (mFbActive) {
        // Failsafe reset to main framebuffer to prevent softlocking the renderer
        mFbActive = 0;
        mRapi->StartDrawToFramebuffer(0, 1);

        assert(0 && "active framebuffer was never reset back to original");
    }
}

void Interpreter::EndFrame() {
    mRapi->EndFrame();
    mWapi->SwapBuffersBegin();
    mRapi->FinishRender();
    mWapi->SwapBuffersEnd();
}

void gfx_set_target_ucode(UcodeHandlers ucode) {
    ucode_handler_index = ucode;
}

int Interpreter::GetTargetFps() {
    return mWapi->GetTargetFps();
}

void Interpreter::SetTargetFps(int fps) {
    mWapi->SetTargetFps(fps);
}

void Interpreter::SetMaxFrameLatency(int latency) {
    mWapi->SetMaxFrameLatency(latency);
}

int Interpreter::CreateFrameBuffer(uint32_t width, uint32_t height, uint32_t native_width, uint32_t native_height,
                                   uint8_t resize, bool forceFixedAspect) {
    uint32_t orig_width = width, orig_height = height;
    if (resize) {
        AdjustWidthHeightForScale(width, height, native_width, native_height);
    }

    int fb = mRapi->CreateFramebuffer();
    mRapi->UpdateFramebufferParameters(fb, width, height, 1, true, true, true, true);

    mFrameBuffers[fb] = {
        orig_width, orig_height, width, height, native_width, native_height, static_cast<bool>(resize), forceFixedAspect
    };
    return fb;
}

void Interpreter::SetFrameBuffer(int fb, float noiseScale) {
    mRapi->StartDrawToFramebuffer(fb, noiseScale);
    mRapi->ClearFramebuffer(false, true);
}

void Interpreter::CopyFrameBuffer(int fb_dst_id, int fb_src_id, bool copyOnce, bool* hasCopiedPtr) {
    // Do not copy again if we have already copied before
    if (copyOnce && hasCopiedPtr != nullptr && *hasCopiedPtr) {
        return;
    }

    if (fb_src_id == 0 && mRendersToFb) {
        // read from the framebuffer we've been rendering to
        fb_src_id = mGameFb;
    }

    int srcX0, srcY0, srcX1, srcY1;
    int dstX0, dstY0, dstX1, dstY1;

    // When rendering to the main window buffer or MSAA is enabled with a buffer size equal to the view port,
    // then the source coordinates must account for any docked ImGui elements
    if (fb_src_id == 0 || GameFbIsWindowSized()) {
        srcX0 = mGameWindowViewport.x;
        srcY0 = mGameWindowViewport.y;
        srcX1 = mGameWindowViewport.x + mGameWindowViewport.width;
        srcY1 = mGameWindowViewport.y + mGameWindowViewport.height;
    } else {
        srcX0 = 0;
        srcY0 = 0;
        srcX1 = mCurDimensions.width;
        srcY1 = mCurDimensions.height;
    }

    dstX0 = 0;
    dstY0 = 0;
    dstX1 = mCurDimensions.width;
    dstY1 = mCurDimensions.height;

    mRapi->CopyFramebuffer(fb_dst_id, fb_src_id, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1);

    // Set the copied pointer if we have one
    if (hasCopiedPtr != nullptr) {
        *hasCopiedPtr = true;
    }
}

void Interpreter::ResetFrameBuffer() {
    mRapi->StartDrawToFramebuffer(0, (float)mCurDimensions.height / mNativeDimensions.height);
}

void Interpreter::AdjustPixelDepthCoordinates(float& x, float& y) {
    x = x * RATIO_X(mActiveFrameBuffer, mCurDimensions) -
        (mNativeDimensions.width * RATIO_X(mActiveFrameBuffer, mCurDimensions) - mCurDimensions.width) / 2;
    y *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);
    if (!mRendersToFb || GameFbIsWindowSized()) {
        x += mGameWindowViewport.x;
        y += mGfxCurrentWindowDimensions.height - (mGameWindowViewport.y + mGameWindowViewport.height);
    }
}

void Interpreter::GetPixelDepthPrepare(float x, float y) {
    AdjustPixelDepthCoordinates(x, y);
    mGetPixelDepthPending.emplace(x, y);
}

uint16_t Interpreter::GetPixelDepth(float x, float y) {
    AdjustPixelDepthCoordinates(x, y);

    if (auto it = mGetPixelDepthCached.find(std::make_pair(x, y)); it != mGetPixelDepthCached.end()) {
        return it->second;
    }

    mGetPixelDepthPending.emplace(x, y);

    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res =
        mRapi->GetPixelDepth(mRendersToFb ? mGameFb : 0, mGetPixelDepthPending);
    mGetPixelDepthCached.merge(res);
    mGetPixelDepthPending.clear();

    return mGetPixelDepthCached.find(std::make_pair(x, y))->second;
}

void gfx_push_current_dir(char* path) {
    if (gfx_check_image_signature(path) == 1)
        path = &path[7];

    currentDir.push(GetPathWithoutFileName(path));
}

int32_t gfx_check_image_signature(const char* imgData) {
    uintptr_t i = (uintptr_t)(imgData);

    if ((i & 1) == 1) {
        return 0;
    }

    // Filter addresses that are obviously not valid string pointers before
    // attempting to dereference for the "__OTR__" check.
    if (i == 0 || i < 0x10000) {
        return 0;
    }
#if UINTPTR_MAX > 0xFFFFFFFFu
    // On 64-bit: filter kernel/sentinel addresses. Upper bound covers all
    // user-space layouts (x86_64 47-bit canonical, ARM64 48-bit VA, etc.).
    if (i > 0x0000FFFFFFFFFFFFull) {
        return 0;
    }
#endif

    return Ship::Context::GetRawInstance()->GetResourceManager()->OtrSignatureCheck(imgData);
}

void Interpreter::RegisterBlendedTexture(const char* name, uint8_t* mask, uint8_t* replacement) {
    if (gfx_check_image_signature(name)) {
        name += 7;
    }

    if (gfx_check_image_signature(reinterpret_cast<char*>(replacement))) {
        Fast::Texture* tex = std::static_pointer_cast<Fast::Texture>(
                                 Ship::Context::GetRawInstance()->GetResourceManager()->LoadResourceProcess(
                                     reinterpret_cast<char*>(replacement)))
                                 .get();

        replacement = tex->ImageData;
    }

    mMaskedTextures[name] = MaskedTextureEntry{ mask, replacement };
}

void Interpreter::UnregisterBlendedTexture(const char* name) {
    if (gfx_check_image_signature(name)) {
        name += 7;
    }

    mMaskedTextures.erase(name);
}

// New getters and setters
void Interpreter::SetNativeDimensions(float width, float height) {
    mNativeDimensions.width = width;
    mNativeDimensions.height = height;
}

void Interpreter::SetResolutionMultiplier(float multiplier) {
    mCurDimensions.internal_mul = multiplier;
}

void Interpreter::SetMsaaLevel(uint32_t level) {
    mMsaaLevel = level;
}

void Interpreter::GetCurDimensions(uint32_t* width, uint32_t* height) {
    *width = mCurDimensions.width;
    *height = mCurDimensions.height;
}

} // namespace Fast

// Only diffuse combiners with a separable shade input are material-compatible. The two-tile
// cases are an albedo blend (including LOD aliases), or albedo multiplied by a detail texture.
uint32_t NewDawnMaterialShadeMask(const CCFeatures& f, bool& twoAlbedoBlend) {
    twoAlbedoBlend = false;
    auto inputBit = [](int x) -> uint32_t { return x >= SHADER_INPUT_1 && x <= SHADER_INPUT_7 ? 1u << (x - 1) : 0; };
    const auto& a = f.c[0][0];
    uint32_t first = 0;
    if (f.do_multiply[0][0]) {
        if (a[0] == SHADER_TEXEL0)
            first = inputBit(a[2]);
        else if (a[2] == SHADER_TEXEL0)
            first = inputBit(a[0]);
    }
    if (!f.opt_2cyc)
        return first;
    const auto& b = f.c[1][0];
    if (f.do_single[1][0] && b[3] == SHADER_COMBINED)
        return first;
    if (!f.do_multiply[1][0])
        return 0;
    uint32_t second = b[0] == SHADER_COMBINED ? inputBit(b[2]) : b[2] == SHADER_COMBINED ? inputBit(b[0]) : 0;
    if (first && second)
        return first ^ second;
    bool twoTextures =
        (a[0] == SHADER_TEXEL0 && a[2] == SHADER_TEXEL1) || (a[0] == SHADER_TEXEL1 && a[2] == SHADER_TEXEL0);
    if (f.do_multiply[0][0] && twoTextures)
        return second;
    twoAlbedoBlend = f.do_mix[0][0] && ((a[0] == SHADER_TEXEL1 && a[1] == SHADER_TEXEL0) ||
                                        (a[0] == SHADER_TEXEL0 && a[1] == SHADER_TEXEL1));
    return twoAlbedoBlend ? second : 0;
}

void gfx_cc_get_features(uint64_t shader_id0, uint64_t shader_id1, struct CCFeatures* cc_features) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 4; k++) {
                cc_features->c[i][j][k] = shader_id0 >> i * 32 + j * 16 + k * 4 & 0xf;
            }
        }
    }

    cc_features->opt_alpha = (shader_id1 & SHADER_OPT(ALPHA)) != 0;
    cc_features->opt_fog = (shader_id1 & SHADER_OPT(FOG)) != 0;
    cc_features->opt_texture_edge = (shader_id1 & SHADER_OPT(TEXTURE_EDGE)) != 0;
    cc_features->opt_alpha_to_coverage = (shader_id1 & SHADER_OPT(ALPHA_TO_COVERAGE)) != 0;
    cc_features->opt_noise = (shader_id1 & SHADER_OPT(NOISE)) != 0;
    cc_features->opt_2cyc = (shader_id1 & SHADER_OPT(_2CYC)) != 0;
    cc_features->opt_alpha_threshold = (shader_id1 & SHADER_OPT(ALPHA_THRESHOLD)) != 0;
    cc_features->opt_invisible = (shader_id1 & SHADER_OPT(INVISIBLE)) != 0;
    cc_features->opt_grayscale = (shader_id1 & SHADER_OPT(GRAYSCALE)) != 0;
    cc_features->opt_prim_depth = (shader_id1 & SHADER_OPT(PRIM_DEPTH)) != 0;
    cc_features->opt_per_pixel_lighting = (shader_id1 & SHADER_OPT(PER_PIXEL_LIGHTING)) != 0;
    cc_features->opt_material_maps = (shader_id1 & SHADER_OPT(MATERIAL_MAPS)) != 0;
    cc_features->opt_material_tile1 = (shader_id1 & SHADER_OPT(MATERIAL_MAPS_TILE1)) != 0;
    cc_features->opt_material_blend = (shader_id1 & SHADER_OPT(MATERIAL_MAPS_BLEND)) != 0;
    cc_features->opt_smooth = (shader_id1 & SHADER_OPT(SMOOTH)) != 0;

    cc_features->clamp[0][0] = shader_id1 & SHADER_OPT(TEXEL0_CLAMP_S);
    cc_features->clamp[0][1] = shader_id1 & SHADER_OPT(TEXEL0_CLAMP_T);
    cc_features->clamp[1][0] = shader_id1 & SHADER_OPT(TEXEL1_CLAMP_S);
    cc_features->clamp[1][1] = shader_id1 & SHADER_OPT(TEXEL1_CLAMP_T);

    cc_features->usedTextures[0] = false;
    cc_features->usedTextures[1] = false;
    cc_features->used_masks[0] = false;
    cc_features->used_masks[1] = false;
    cc_features->used_blend[0] = false;
    cc_features->used_blend[1] = false;
    cc_features->numInputs = 0;

    for (int c = 0; c < 2; c++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                if (cc_features->c[c][i][j] >= SHADER_INPUT_1 && cc_features->c[c][i][j] <= SHADER_INPUT_7) {
                    if (cc_features->c[c][i][j] > cc_features->numInputs) {
                        cc_features->numInputs = cc_features->c[c][i][j];
                    }
                }
                if (cc_features->c[c][i][j] == SHADER_TEXEL0 || cc_features->c[c][i][j] == SHADER_TEXEL0A) {
                    cc_features->usedTextures[0] = true;
                    if (cc_features->opt_2cyc) {
                        cc_features->usedTextures[1] = true;
                    }
                }
                if (cc_features->c[c][i][j] == SHADER_TEXEL1 || cc_features->c[c][i][j] == SHADER_TEXEL1A) {
                    cc_features->usedTextures[1] = true;
                    if (cc_features->opt_2cyc) {
                        cc_features->usedTextures[0] = true;
                    }
                }
            }
        }
    }

    for (int c = 0; c < 2; c++) {
        cc_features->do_single[c][0] = cc_features->c[c][0][2] == SHADER_0;
        cc_features->do_single[c][1] = cc_features->c[c][1][2] == SHADER_0;
        cc_features->do_multiply[c][0] = cc_features->c[c][0][1] == SHADER_0 && cc_features->c[c][0][3] == SHADER_0;
        cc_features->do_multiply[c][1] = cc_features->c[c][1][1] == SHADER_0 && cc_features->c[c][1][3] == SHADER_0;
        cc_features->do_mix[c][0] = cc_features->c[c][0][1] == cc_features->c[c][0][3];
        cc_features->do_mix[c][1] = cc_features->c[c][1][1] == cc_features->c[c][1][3];
        cc_features->color_alpha_same[c] = (shader_id0 >> c * 32 & 0xffff) == (shader_id0 >> c * 32 + 16 & 0xffff);
    }

    if (cc_features->usedTextures[0] && shader_id1 & SHADER_OPT(TEXEL0_MASK)) {
        cc_features->used_masks[0] = true;
    }
    if (cc_features->usedTextures[1] && shader_id1 & SHADER_OPT(TEXEL1_MASK)) {
        cc_features->used_masks[1] = true;
    }

    if (cc_features->usedTextures[0] && shader_id1 & SHADER_OPT(TEXEL0_BLEND)) {
        cc_features->used_blend[0] = true;
    }
    if (cc_features->usedTextures[1] && shader_id1 & SHADER_OPT(TEXEL1_BLEND)) {
        cc_features->used_blend[1] = true;
    }

    cc_features->shader_id = Fast::ShaderIdUnmask(shader_id1);
}

extern "C" int gfx_create_framebuffer(uint32_t width, uint32_t height, uint32_t native_width, uint32_t native_height,
                                      uint8_t resize, bool forceFixedAspect) {
    return Fast::mInstance.lock().get()->CreateFrameBuffer(width, height, native_width, native_height, resize,
                                                           forceFixedAspect);
}

extern "C" void gfx_texture_cache_clear() {
    Fast::mInstance.lock().get()->TextureCacheClear();
}

extern "C" void gfx_shader_cache_clear() {
    auto instance = Fast::mInstance.lock().get();
    instance->mColorCombinerPool.clear();
    instance->mPrevCombiner = Fast::mInstance.lock().get()->mColorCombinerPool.end();
    instance->mRenderingState.mShaderProgram = nullptr;
    instance->mRapi->ClearShaderCache();
}

extern "C" void gfx_register_fb_texture(const void* cpuAddr, int fbId) {
    Fast::mInstance.lock().get()->RegisterFbTexture(cpuAddr, fbId);
}

extern "C" void gfx_unregister_fb_texture(const void* cpuAddr) {
    Fast::mInstance.lock().get()->UnregisterFbTexture(cpuAddr);
}
