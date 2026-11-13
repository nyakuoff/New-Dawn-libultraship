#ifdef ENABLE_OPENGL
#pragma once

#include "gfx_rendering_api.h"
#include "../interpreter.h"

#ifdef _MSC_VER
#include <SDL2/SDL.h>
// #define GL_GLEXT_PROTOTYPES 1
#include <GL/glew.h>
#elif FOR_WINDOWS
#include <GL/glew.h>
#include "SDL.h"
#define GL_GLEXT_PROTOTYPES 1
#include "SDL_opengl.h"
#elif __APPLE__
#include <SDL2/SDL.h>
#include <GL/glew.h>
#elif USE_OPENGLES
#include <SDL2/SDL.h>
#include <GLES3/gl3.h>
#else
#include <SDL2/SDL.h>
#define GL_GLEXT_PROTOTYPES 1
#include <SDL2/SDL_opengl.h>
#endif
#include <memory>

namespace Fast {
class PostFxGL;
class MM3DGL;

struct ShaderProgram {
    GLuint openglProgramId;
    uint8_t numInputs;
    bool usedTextures[SHADER_MAX_TEXTURES];
    uint8_t numFloats;
    GLint attribLocations[16];
    uint8_t attribSizes[16];
    uint8_t numAttribs;
    GLint hdr_location, hdr_sun_location, hdr_ambient_location, hdr_sky_location, hdr_fog_location;
    GLint frameCountLocation;
    GLint noiseScaleLocation;
    GLint prim_depth_location;
    GLint texture_width_location;
    GLint texture_height_location;
    GLint texture_filtering_location;
    // New Dawn per-pixel lighting uniforms (-1 when the program does not use them).
    GLint per_pixel_light_count_location;
    GLint per_pixel_ambient_location;
    GLint per_pixel_dir_location;
    GLint per_pixel_color_location;
    GLint per_pixel_pos_location;
    GLint per_pixel_atten_location;
    GLint per_pixel_positional_location;
    GLint per_pixel_view_matrix_location;
    GLint per_pixel_shade_mask_location;
    GLint per_pixel_quality_location;
    int materialTile = 0;
    GLint material_strength_location;
    GLint material_debug_location;
    GLint material_mirror_location;
};

struct FramebufferOGL {
    uint32_t width, height;
    bool has_depth_buffer;
    uint32_t msaa_level;
    bool invertY;

    GLuint fbo, clrbuf, clrbufMsaa, rbo;
};

struct TextureInfo {
    uint16_t width;
    uint16_t height;
    uint16_t filtering;
    bool mipmapped;
    uint8_t bytesPerPixel = 4;
    bool mirrorS = false, mirrorT = false;
    bool mipEligible; // uploaded through the interpreter, not by UI code
};

class GfxRenderingAPIOGL final : public GfxRenderingAPI {
  public:
    GfxRenderingAPIOGL();
    ~GfxRenderingAPIOGL() override;
    const char* GetName() override;
    int GetMaxTextureSize() override;
    GfxClipParameters GetClipParameters() override;
    void UnloadShader(ShaderProgram* oldPrg) override;
    void LoadShader(ShaderProgram* newPrg) override;
    ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) override;
    ShaderProgram* LookupShader(uint64_t shaderId0, uint64_t shaderId1) override;
    void ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) override;
    void ClearShaderCache() override;
    uint32_t NewTexture() override;
    void SelectTexture(int tile, uint32_t textureId) override;
    bool SupportsMaterialMaps() override;
    void UploadMaterialTexture(const uint8_t* rg8, uint32_t width, uint32_t height) override;
    void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) override;
    void BuildMipmaps() override;
    uint64_t GetTextureByteSize(uint32_t textureId) override;
    void SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) override;
    void SetDepthTestAndMask(bool depth_test, bool z_upd) override;
    void SetCurrentPrimDepth(float depth) override;
    void SetZmodeDecal(bool decal) override;
    void SetViewport(int x, int y, int width, int height) override;
    void SetScissor(int x, int y, int width, int height) override;
    void SetUseAlpha(bool useAlpha) override;
    void DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) override;
    void Init() override;
    void OnResize() override;
    bool MM3DAvailable() const override;
    bool PrepareMM3DRoom(const NewDawnMM3DRoomParams&) override;
    void DrawMM3DRoom(const NewDawnMM3DRoomParams&, const float*, float, float, const NewDawnMM3DScroll*) override;
    bool HdrWorldActive() const override;
    void FinishWorld() override;
    void StartFrame() override;
    void EndFrame() override;
    void FinishRender() override;
    int CreateFramebuffer() override;
    void UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                     bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                     bool can_extract_depth) override;
    void StartDrawToFramebuffer(int fbId, float noiseScale) override;
    void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0, int dstY0,
                         int dstX1, int dstY1) override;
    void ClearFramebuffer(bool color, bool depth) override;
    void ClearDepthRegion(int x, int y, int w, int h) override;
    void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) override;
    void ResolveMSAAColorBuffer(int fbIdTarger, int fbIdSrc) override;
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) override;
    void* GetFramebufferTextureId(int fbId) override;
    void SelectTextureFb(int fbId) override;
    void DeleteTexture(uint32_t texId) override;
    void SetTextureFilter(FilteringMode mode) override;
    FilteringMode GetTextureFilter() override;
    void SetSrgbMode() override;
    ImTextureID GetTextureById(int id) override;
    bool PostFxAvailable() override;
    void RunPostFx(int stage, const NewDawnPostFxParams& params, int gameFbId) override;
    void SetShadowCasterGeometry(const float* positions, uint32_t vertexCount) override;
    void SetShadowFoliageGeometry(const float* verts, const GfxShadowFoliageBatch* batches,
                                  uint32_t batchCount) override;
    bool SupportsAlphaToCoverage() const override {
        return true;
    }
    void SetAlphaToCoverage(bool enable) override;
    bool SupportsPerPixelLighting() const override {
        return true;
    }
    void SetPerPixelLighting(const GfxPerPixelLightingState& state) override;

  private:
    void SetUniforms(ShaderProgram* prg) const;
    std::string BuildFsShader(const CCFeatures& cc_features);
    void SetPerDrawUniforms();

    std::vector<TextureInfo> textures;
    // Optional nonblocking timers measure all world draw batches equally with materials on/off.
    static constexpr int kDrawTimerCount = 512;
    GLuint mDrawTimers[kDrawTimerCount] = {};
    bool mDrawTimerPending[kDrawTimerCount] = {};
    int mDrawTimerNext = 0;
    int mDrawTimerFrames = 0;
    int mDrawTimerDropped = 0;
    double mDrawTimerMs = 0.0;
    bool mDrawStats = false;
    bool mMaterialUnitsAvailable = false;
    GLuint mCurrentTextureIds[MATERIAL_TEXTURE_SLOTS] = {};
    GLuint mLastBoundTextures[MATERIAL_TEXTURE_SLOTS] = {};
    uint8_t mCurrentTile;
    int8_t mLastActiveTexture = -1;
    int8_t mLastBlendEnabled = -1;
    int8_t mLastAlphaToCoverage = -1;
    int8_t mLastScissorEnabled = -1;

    std::map<std::pair<uint64_t, uint32_t>, ShaderProgram> mShaderProgramPool;
    ShaderProgram* mCurrentShaderProgram;
    ShaderProgram* mLastLoadedShader = nullptr;

    GLuint mOpenglVbo = 0;
#if defined(__APPLE__) || defined(USE_OPENGLES)
    GLuint mOpenglVao;
#endif

    uint32_t mFrameCount = 0;

    // New Dawn per-pixel lighting: the last state the interpreter handed us, and the program it
    // was last uploaded for, so a program switch re-applies the uniforms.
    GfxPerPixelLightingState mPerPixelLightState{};
    int32_t mPerPixelUploadedSerial = -1;
    ShaderProgram* mPerPixelUploadedProgram = nullptr;

    std::vector<FramebufferOGL> mFrameBuffers;
    size_t mCurrentFrameBuffer = 0;
    float mCurrentNoiseScale = 0.0f;
    FilteringMode mCurrentFilterMode = FILTER_THREE_POINT;

    GLint mMaxMsaaLevel = 1;
    float mMaxAnisotropy = 1.0f;
    // Texture settings are cached once per frame; SetSamplerParameters runs for every texture bind.
    bool mMipmapsEnabled = true;
    float mRequestedAnisotropy = 16.0f;
    float mLodBias = 0.0f;
    GLuint mPixelDepthRb = 0;
    GLuint mPixelDepthFb = 0;
    size_t mPixelDepthRbSize = 0;
    const float* mShadowCasterPositions = nullptr;
    uint32_t mShadowCasterVertexCount = 0;
    const float* mShadowFoliageVerts = nullptr;
    const GfxShadowFoliageBatch* mShadowFoliageBatches = nullptr;
    uint32_t mShadowFoliageBatchCount = 0;

#ifndef USE_OPENGLES
    std::unique_ptr<MM3DGL> mMM3D;
    int mHdrGameFb = -1;
    bool mHdrSky = false;
    std::unique_ptr<PostFxGL> mPostFx; // only complete (and only built) where gfx_opengl_postfx.h is available
#endif
};

} // namespace Fast
#endif
