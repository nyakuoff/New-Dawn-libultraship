#pragma once

#include <stdint.h>

#include <unordered_map>
#include <set>
#include "imconfig.h"

struct NewDawnPostFxParams;

namespace Fast {
struct ShaderProgram;

#define MAX_PER_PIXEL_LIGHTS 32

// One light exactly as the interpreter's CPU lighting path uses it: directional lights carry a
// model-space direction, MM point lights carry a view-space position plus attenuation terms.
struct GfxPerPixelLight {
    float dir[3];   // model-space unit direction (directional lights)
    float color[3]; // 0..255, same range as the light command colours
    float pos[3];   // view-space position (point lights)
    float atten[2]; // point light attenuation (unk7, unkE)
    int32_t positional;
};

// Everything the fragment shader needs to reproduce the CPU light model. Owned by the
// interpreter; copied by the backend when it changes.
struct GfxPerPixelLightingState {
    int32_t numLights;   // excludes the ambient light
    float ambient[3];    // 0..255
    float viewMatrix[9]; // modelview upper 3x3, column-major
    int32_t shadeMask;   // bit n set: shader input n comes from the shade colour
    float pointLightQuality;
    int32_t serial;
    GfxPerPixelLight lights[MAX_PER_PIXEL_LIGHTS];
};

struct GfxClipParameters {
    bool z_is_from_0_to_1;
    bool invertY;
};

enum FilteringMode { FILTER_THREE_POINT, FILTER_LINEAR, FILTER_NONE };

// A hash function used to hash a: pair<float, float>
struct hash_pair_ff {
    size_t operator()(const std::pair<float, float>& p) const {
        const auto hash1 = std::hash<float>{}(p.first);
        const auto hash2 = std::hash<float>{}(p.second);

        // If hash1 == hash2, their XOR is zero.
        return (hash1 != hash2) ? hash1 ^ hash2 : hash1;
    }
};

class GfxRenderingAPI {
  public:
    virtual ~GfxRenderingAPI() = default;
    virtual const char* GetName() = 0;
    virtual int GetMaxTextureSize() = 0;
    virtual GfxClipParameters GetClipParameters() = 0;
    virtual void UnloadShader(ShaderProgram* oldPrg) = 0;
    virtual void LoadShader(ShaderProgram* newPrg) = 0;
    virtual void ClearShaderCache() = 0;
    virtual ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) = 0;
    virtual ShaderProgram* LookupShader(uint64_t shaderId0, uint64_t shaderId1) = 0;
    virtual void ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) = 0;
    virtual uint32_t NewTexture() = 0;
    virtual void SelectTexture(int tile, uint32_t textureId) = 0;
    virtual void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) = 0;
    // Generate the mip chain for the most recently uploaded texture. Called by the interpreter after
    // game textures are uploaded; backends without mipmap support may leave this as a no-op.
    virtual void BuildMipmaps() {
    }
    // Approximate VRAM footprint of a texture, including its mip chain when present. Backends that do
    // not track texture sizes return 0, which leaves the interpreter's byte budget unenforced.
    virtual uint64_t GetTextureByteSize(uint32_t textureId) {
        return 0;
    }
    virtual void SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) = 0;
    virtual void SetDepthTestAndMask(bool depth_test, bool z_upd) = 0;
    virtual void SetZmodeDecal(bool decal) = 0;
    virtual void SetViewport(int x, int y, int width, int height) = 0;
    virtual void SetScissor(int x, int y, int width, int height) = 0;
    virtual void SetUseAlpha(bool useAlpha) = 0;
    virtual void DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) = 0;
    virtual void Init() = 0;
    virtual void OnResize() = 0;
    virtual void StartFrame() = 0;
    virtual void EndFrame() = 0;
    virtual void FinishRender() = 0;
    virtual int CreateFramebuffer() = 0;
    virtual void UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                             bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                             bool can_extract_depth) = 0;
    virtual void StartDrawToFramebuffer(int fbId, float noiseScale) = 0;
    virtual void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0,
                                 int dstY0, int dstX1, int dstY1) = 0;
    virtual void ClearFramebuffer(bool color, bool depth) = 0;
    virtual void ClearDepthRegion(int x, int y, int w, int h) {
        // Default: full depth clear. Backends that support scissored depth clears
        // (e.g. OpenGL) should override for a more precise partial clear.
        ClearFramebuffer(false, true);
    }
    virtual void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) = 0;
    virtual void ResolveMSAAColorBuffer(int fbIdTarger, int fbIdSrc) = 0;
    virtual std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) = 0;
    virtual void* GetFramebufferTextureId(int fbId) = 0;
    virtual void SelectTextureFb(int fbId) = 0;
    virtual void DeleteTexture(uint32_t texId) = 0;
    virtual void SetTextureFilter(FilteringMode mode) = 0;
    virtual FilteringMode GetTextureFilter() = 0;
    virtual void SetSrgbMode() = 0;
    virtual ImTextureID GetTextureById(int id) = 0;
    virtual void SetCurrentPrimDepth(float depth) = 0;

    // New Dawn scene post-processing (see fast/newdawn_postfx.h). Only implemented for OpenGL.
    virtual bool PostFxAvailable() {
        return false;
    }
    virtual void RunPostFx(int stage, const NewDawnPostFxParams& params, int gameFbId) {
    }

    // New Dawn per-pixel lighting. Only implemented for OpenGL; other backends keep the CPU
    // vertex lighting path. See GfxPerPixelLightingState above.
    virtual bool SupportsPerPixelLighting() const {
        return false;
    }
    virtual void SetPerPixelLighting(const GfxPerPixelLightingState& state) {
    }

  protected:
    int8_t mCurrentDepthTest = 0;
    int8_t mCurrentDepthMask = 0;
    int8_t mCurrentZmodeDecal = 0;
    int8_t mLastDepthTest = -1;
    int8_t mLastDepthMask = -1;
    int8_t mLastZmodeDecal = -1;
    bool mSrgbMode = false;
    float mCurrentPrimDepth = 0.0f;
    bool mPrimDepthDirty = true;
};
} // namespace Fast
