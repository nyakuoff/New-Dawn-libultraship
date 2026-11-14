#if defined(ENABLE_OPENGL) && !defined(USE_OPENGLES)
#pragma once

#include "gfx_opengl.h"
#include <string>
#include <unordered_map>
#include "fast/newdawn_postfx.h"

namespace Fast {

// New Dawn scene post-processing for the OpenGL backend: ambient occlusion on the opaque geometry, then bloom,
// colour grade and dithering of the finished world. It works on private copies of the game framebuffer's depth and
// colour, writes its result back into the game framebuffer and leaves all GL state exactly as it found it, so the
// backend's cached state (and whatever is drawn afterwards) is unaffected. GI retains rejected temporal history.
class PostFxGL {
  public:
    // Checks the GL version and builds the shaders. On failure the feature stays unusable.
    void Init();
    bool Usable() const {
        return mUsable;
    }
    // frameIndex identifies the rendered frame (both stages of a frame share one read of the settings).
    void Run(int stage, const NewDawnPostFxParams& params, const FramebufferOGL& gameFb, uint32_t frameIndex,
             const float* shadowPositions, uint32_t shadowVertexCount);

    // Alpha-tested shadow casters for the next Run (5 floats per vertex, batched by GL texture id).
    void SetFoliageCasters(const float* verts, const GfxShadowFoliageBatch* batches, uint32_t batchCount) {
        mFoliageVerts = verts;
        mFoliageBatches = batches;
        mFoliageBatchCount = batchCount;
    }

    bool BeginHdr(const FramebufferOGL& fb);
    void RunPhysicalSky(const NewDawnPostFxParams& params, const FramebufferOGL& fb, uint32_t frameIndex);
    void RunSunFlare(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void EndHdr(const FramebufferOGL& fb);
    GLuint HdrSnapshot(const FramebufferOGL& fb);
    GLuint HdrDisplayTexture() const {
        return mHdrDisplay.tex;
    }
    bool HdrActive() const {
        return mHdrActive;
    }
    bool WeatherReady() const { return mWeatherParticles.id && mWeatherWet.id; }
    bool WeatherActive() const { return mWeatherFrameReady; }
    bool SkyAmbientActive() const {
        return mHdrActive && mSkyAmbientReady;
    }
    // Dedicated world-shader unit; the interpreter owns 0..7 and ordinary postfx use 8..11.
    static constexpr int kSkyAmbientUnit = 12;
    static constexpr int kSkyReflectionUnit = 13;
    static constexpr int kFogUnit = 14;
    static constexpr int kFlareUnit = 15;
    static constexpr int kFlareSpritesUnit = 16;
    static constexpr int kFlareCloudUnit = 17;
    bool SkyReflectionsActive() const {
        return mHdrActive && mSkyReflectionsReady;
    }
    const float* SkyReflectionEye() const {
        return mSkyReflectionEye;
    }

    struct Settings {
        bool postFxOn, aoOn, bloomOn, gradeOn;
        bool shadowsOn;
        int shadowDebug;
        float shadowFloor; // minimum fraction of the lit colour a fully shadowed pixel keeps
        float aoStrength, aoRadius;
        float bloomStrength, bloomThreshold;
        float contrast, saturation;
        int debugView; // 0 normal, 1 split screen (left half untouched), 2 raw AO buffer
        bool stats;
    };

  private:
    struct Target {
        GLuint tex = 0;
        GLuint fbo = 0;
        int w = 0;
        int h = 0;
        GLint format = 0;
    };
    struct Program {
        GLuint id = 0;
        std::unordered_map<std::string, GLint> uniforms; // cached when linked
    };
    struct CubeTarget {
        GLuint tex = 0;
        GLuint fbo = 0;
        int size = 0, levels = 0;
    };
    bool EnsureCube(CubeTarget& target, int size, int levels = 1);
    void SetSkyUniforms(const NewDawnPostFxParams& params);
    void RunSkyAmbient(const NewDawnPostFxParams& params);
    bool BuildPrograms();
    bool BuildAtmosProgram();
    bool BuildAtmosShader(Program& program, const char* vertex, const char* fragment);
    bool BuildWeatherPrograms();
    void RunWeather(const NewDawnPostFxParams& params, const FramebufferOGL& fb, const float* roofMatrix,
                    const float* positions, uint32_t vertexCount);
    void RunWeatherParticles(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    float mWeatherTime = 0.0f, mWeatherRoofMatrix[16] = {};
    bool mWeatherAttempted = false, mWeatherFrameReady = false;
    Program mWeatherParticles, mWeatherWet;
    Target mWeatherRoof;
    GLuint mWeatherRing = 0, mWeatherNeutralCube = 0;
    float mWeatherWetness = 0.0f;
    int mWeatherScene = -1;
    double mWeatherLastTime = 0.0;
    uint32_t mWeatherLastFrame = UINT32_MAX;
    void RunAtmos(const NewDawnPostFxParams& params, const FramebufferOGL& fb, const float* lightMatrix,
                  bool shadowReady);
    Program mAtmos;
    bool mAtmosAttempted = false;
    GLuint mAtmosSprite = 0, mAtmosLeaf = 0, mAtmosNeutralDepth = 0;
    void EnsureTarget(Target& t, int w, int h, GLint internalFormat, GLenum format, GLenum type, bool depthStencil,
                      bool linear);
    void PrepareDepth(const FramebufferOGL& fb);
    void EnsureShadowTarget(int size);
    void RunShadows(const NewDawnPostFxParams& params, const FramebufferOGL& fb, const float* positions,
                    uint32_t vertexCount, int debug);
    void RunWater(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void RunWaterSurface(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void RunVolumetricFog(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void RunSunShafts(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void RunScreenShafts(const NewDawnPostFxParams& params, const FramebufferOGL& fb, const float sunUv[2]);
    void RunGlobalIllumination(const NewDawnPostFxParams& params, const FramebufferOGL& fb, uint32_t frameIndex);
    void RunAmbientOcclusion(const NewDawnPostFxParams& params, const FramebufferOGL& fb, const Settings& s);
    void RunSceneEnd(const FramebufferOGL& fb, const Settings& s);
    void Draw(const Target& t);
    void Draw(GLuint fbo, int w, int h);
    void Bind(int unit, GLuint tex);
    void Use(const Program& p);
    GLint Loc(const char* name);
    void BeginTimer(int stage);
    void EndTimer(int stage);
    void CheckError(const char* where);

    bool mUsable = false;
    bool mStats = false; // copy of mSettings.stats for the helpers
    uint32_t mSettingsFrame = UINT32_MAX;
    // Art-directed grade of the current frame, copied from the scene-end marker's params.
    struct ArtGrade {
        bool on = false;
        float slope[3] = { 1, 1, 1 }, offset = 0, power = 1, contrast = 0, sat = 1;
        float shadow[3] = {}, high[3] = {}, green[2] = { 0, 1 };
    } mArt;
    Settings mSettings{};
    uint32_t mStatsW = 0, mStatsH = 0, mStatsMsaa = 0;

    bool mHdrActive = false;
    Target mHdr, mHdrDisplay;
    GLuint mHdrMsaa = 0;
    int mHdrSamples = 0, mHdrWidth = 0, mHdrHeight = 0;
    Program mHdrCopy;
    Program mPhysicalSky, mSkyApply;
    Target mSky;
    Program mCloudMarch, mCloudApply;
    Target mClouds;
    GLuint mCloudNoise = 0;
    Program mSkyEnvironment, mCloudEnvironment, mSkyConvolve, mSkyPrefilter;
    CubeTarget mSkyRadiance, mSkyIrradiance, mSkySpecular;
    bool mSkyReflectionsReady = false;
    float mSkyReflectionEye[3] = {};
    bool mSkyAmbientReady = false;
    bool mSkyAmbientUsable = true;
    bool mSkyAmbientCacheValid = false;
    uint32_t mSkyFrame = 0, mSkyLastEligible = UINT32_MAX, mSkyLastRefresh = 0;
    int32_t mSkyCachedScene = -1;
    float mSkyCachedSun[3] = {}, mSkyCachedWeather = 0.0f;
    float mSkyCachedSettings[7] = {};
    void RunClouds(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void RestoreColor(const FramebufferOGL& fb);
    Program mAo, mBlur, mApply, mDown, mUp, mComposite;
    Program mDownHdr, mCompositeHdr, mCompositeNoDither, mShadowApplyHdr;
    Program mShadowDepth, mShadowApply, mShadowFoliage;
    Program mShaftMarch, mShaftBlur, mShaftApply, mWater, mWaterReflection, mWaterReflectionApply, mWaterSurface;
    GLuint mCmpSampler = 0; // linear + depth-compare sampler for the shadow map (bound to its own texture unit)
    const Program* mCurrentProgram = nullptr;
    GLuint mVao = 0;
    GLuint mShadowVao = 0, mShadowVbo = 0;
    GLuint mFoliageVao = 0, mFoliageVbo = 0;
    const float* mFoliageVerts = nullptr;
    const GfxShadowFoliageBatch* mFoliageBatches = nullptr;
    uint32_t mFoliageBatchCount = 0;

    Target mDepth;
    Target mAoRaw, mAoBlur;
    Program mGiGather, mGiFilter, mGiApply;
    Target mGiSource, mGiRaw, mGiHistory[2];
    bool mGiValid = false;
    int mGiRead = 0, mGiScene = -1;
    uint32_t mGiFrame = UINT32_MAX;
    float mGiViewProj[16] = {}, mGiEye[3] = {}, mGiForward[3] = {};
    float mGiNdcXScale = 1.0f, mGiFlipY = 1.0f;
    Target mScene;
    Target mWaterOpaqueColor, mWaterOpaqueDepth; // captured before XLU, only when reflections are enabled
    Target mWaterReflectionRaw; // half-resolution premultiplied colour and reflection coverage
    uint32_t mWaterOpaqueFrame = UINT32_MAX;
    Target mShadowMap;
    Target mShaftA, mShaftB; // half resolution
    bool mRaySkyReady = false;
    Program mRayMask, mRayRadial, mRayApply;
    Target mRayMaskTarget, mRayTarget; // half-resolution sky coverage and integrated radiance
    Program mFogMarch, mFogApply;
    Target mFog;
    GLuint mFogNeutral = 0;
    Program mFlareVisibility, mFlareApply;
    Target mFlareEmitter;
    GLuint mFlareNeutral = 0;
    GLuint mFlareSprites = 0, mFlareSpritesNeutral = 0, mFlareCloudNeutral = 0;
    std::string mFlareAssetPath;
    float mFlareSpriteScale[12] = {};
    float mFlareCloudOpacity = 0.0f;
    bool EnsureFlareAssets();
    static constexpr int kMaxBloomLevels = 6;
    Target mBloom[kMaxBloomLevels];
    float mKernel[16 * 3] = {};

    static constexpr int kQueryRing = 4;
    static constexpr int kQueryStages = 15;
    static constexpr int kGiTimerStage = 14;
    static constexpr int kFlareTimerStage = 10;
    static constexpr int kFogTimerStage = 9;
    static constexpr int kSkyTimerStage = 6;
    static constexpr int kCloudTimerStage = 7;
    static constexpr int kSkyAmbientTimerStage = 8;
    static constexpr int kShadowTimerStage = 2;
    static constexpr int kShaftTimerStage = 3;
    static constexpr int kWaterTimerStage = 4;
    static constexpr int kWaterCaptureTimerStage = 5;
    GLuint mQueries[kQueryStages][kQueryRing] = {};
    bool mQueryPending[kQueryStages][kQueryRing] = {};
    int mQueryNext[kQueryStages] = {};
    double mQueryTotalMs[kQueryStages] = {};
    int mQueryCount[kQueryStages] = {};
};

} // namespace Fast
#endif
