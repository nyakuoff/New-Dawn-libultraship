#if defined(ENABLE_OPENGL) && !defined(USE_OPENGLES)
#pragma once

#include "gfx_opengl.h"
#include <string>
#include <unordered_map>
#include <vector>
#include "fast/newdawn_postfx.h"

namespace Fast {

// Cascaded sun shadows (gNewDawn.Shadows.Cascades > 1): the shadow map texture is an atlas of equal tiles, one per
// cascade, and every cascade matrix maps world space straight to atlas uv, so consumers keep their existing sampler.
// The consumer defines csmTap(): visibility (0..1) of a biased reference depth at an atlas uv, with its own filter.
// csmLit() returns that visibility, or -1 when the position is beyond the last cascade (treat as lit).
inline constexpr const char* kNewDawnCsmLib = R"GLSL(
uniform int uCsmCount;
uniform mat4 uCsmVP[4];
uniform vec4 uCsmRect[4];
uniform vec4 uCsmSplit;
uniform vec4 uCsmBiasScale;
uniform vec4 uCsmTexelScale;
uniform float uCsmBlend;
uniform vec3 uCsmEye;
uniform vec3 uCsmFwd;
float csmViewDepth(vec3 pos) { return dot(pos - uCsmEye, uCsmFwd); }
float csmTap(vec3 uvz);
float csmFetch(int c, vec3 pos, vec3 N, float nBias, float bias) {
    vec4 lc = uCsmVP[c] * vec4(pos + N * (nBias * uCsmTexelScale[c]), 1.0);
    vec3 ndc = lc.xyz / lc.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    float ref = ndc.z * 0.5 + 0.5;
    vec4 r = uCsmRect[c];
    if (any(lessThan(uv, r.xy)) || any(greaterThan(uv, r.zw)) || ref >= 1.0) return -1.0;
    return csmTap(vec3(uv, ref - bias * uCsmBiasScale[c]));
}
int csmIndex(float vz) {
    int c = 0;
    for (int i = 0; i < 3; i++) {
        if (i + 1 < uCsmCount && vz > uCsmSplit[i]) c = i + 1;
    }
    return c;
}
float csmLit(vec3 pos, vec3 N, float vz, float nBias, float bias) {
    float last = uCsmSplit[uCsmCount - 1];
    if (vz > last) return -1.0;
    int c = csmIndex(vz);
    float lit = csmFetch(c, pos, N, nBias, bias);
    if (lit < 0.0 && c + 1 < uCsmCount) {
        c++;
        lit = csmFetch(c, pos, N, nBias, bias);
    }
    if (lit < 0.0) return -1.0;
    if (c + 1 < uCsmCount) {
        float s = uCsmSplit[c];
        float t = smoothstep(s * (1.0 - uCsmBlend), s, vz);
        if (t > 0.0) {
            float next = csmFetch(c + 1, pos, N, nBias, bias);
            if (next >= 0.0) lit = mix(lit, next, t);
        }
    }
    return mix(lit, 1.0, smoothstep(last * 0.9, last, vz));
}
)GLSL";

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

    // New Dawn VFX sprites for the next Run (NEWDAWN_VFX_FLOATS floats each).
    void SetVfxSprites(const float* data, uint32_t count) {
        mVfxData = data;
        mVfxCount = count;
    }

    // New Dawn grass: world-space ground triangles for the next Run (NEWDAWN_GRASS_TRI_FLOATS floats each).
    void SetGrassTriangles(const float* data, uint32_t count) {
        mGrassData = data;
        mGrassCount = count;
    }

    // New Dawn world GI: recorded room triangles for the next Run (NEWDAWN_GI_TRI_FLOATS floats each).
    void SetGiTriangles(const float* data, uint32_t count) {
        mWgiTris = data;
        mWgiTriCount = count;
    }
    bool WorldGiWantsGeometry(uint32_t frameIndex) const;

    bool VfxUsable() const {
        return mUsable && (!mVfxAttempted || mVfx.id != 0);
    }

    // True while this frame's character-classified draws should write stencil 1 (the local-light darkening reads it).
    bool LocalMarking() const {
        return mLocalMarking;
    }
    bool mLocalMarking = false;
    // New Dawn TAA: the un-jittered camera of the frame that just opened (the interpreter jitters P itself).
    void SetTaaCamera(const float* viewProj, float ndcXScale, float flipY, float zNear, float zFar, float jitterPxX,
                      float jitterPxY, uint32_t frameIndex);
    bool BeginHdr(const FramebufferOGL& fb);
    void RunPhysicalSky(const NewDawnPostFxParams& params, const FramebufferOGL& fb, uint32_t frameIndex);
    void RunSunFlare(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void EndHdr(const FramebufferOGL& fb);
    // final: the last read of the finished world (EndHdr); TAA only resolves then, never on mid-frame snapshots.
    GLuint HdrSnapshot(const FramebufferOGL& fb, bool final = false);
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
    static constexpr int kCausticUnit = 28; // atmos 16..19 and 24, vfx 25..27 are taken
    // Water simulation (dedicated lanes, a complete dummy bound at program creation; grass owns 29 and 30).
    static constexpr int kWaveBufUnit = 31;   // per-pixel displaced surface (height, slope, foam)
    static constexpr int kWaveDepthUnit = 32; // eighth-resolution blurred water depth (the wave taper)
    static constexpr int kRippleUnit = 33;    // GPU ripple height field around the player
    static constexpr float kWaveBaseAmp = 13.0f; // swell amplitude (world units) at waterWaves = 1
    static constexpr int kMilkyWayUnit = 21; // night sky panorama (free: atmos/vfx/grass/caustics use others)
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
    bool BuildVfxProgram();
    bool BuildGrassProgram();
    void RunGrass(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void CaptureGrassGround(const FramebufferOGL& fb);
    bool mGrassGroundReady = false; // the ground-only colour copy of this frame exists (taken before actors are drawn)
    Program mGrass;
    bool mGrassAttempted = false;
    GLuint mGrassAtlas = 0, mGrassVao = 0, mGrassVbo = 0;
    Target mGrassGround; // single-sample copy of the finished ground colour, sampled at every tuft root
    const float* mGrassData = nullptr;
    uint32_t mGrassCount = 0;
    std::vector<float> mGrassCells[4]; // per-frame scratch: cell records grouped by slot count
    void RunVfx(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    Program mVfx;
    bool mVfxAttempted = false;
    GLuint mVfxSprites = 0, mVfxFlip = 0, mVfxVao = 0, mVfxVbo = 0;
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
    Program mAtmos, mAtmosCsm;
    const float* mVfxData = nullptr;
    uint32_t mVfxCount = 0;
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
    void RunWaterWaves(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void StepWaterRipples(const NewDawnPostFxParams& params);
    void RunCaustics(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void RunUnderwater(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    bool EnsureCausticAssets();
    bool EnsureNightSkyAssets();
    void RunNightStars(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void SetNightUniforms(const NewDawnPostFxParams& params);
    void RunVolumetricFog(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void RunSunShafts(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void RunScreenShafts(const NewDawnPostFxParams& params, const FramebufferOGL& fb, const float sunUv[2]);
    void RunGlobalIllumination(const NewDawnPostFxParams& params, const FramebufferOGL& fb, uint32_t frameIndex,
                              bool sunShadow);
    void RunAmbientOcclusion(const NewDawnPostFxParams& params, const FramebufferOGL& fb, const Settings& s);
    // World-space GI (gfx_opengl_worldgi.cpp, docs/WORLD_GI.md): a persistent voxel copy of the room geometry, a
    // radiance volume lit from it, an irradiance + distance probe grid traced through it, and the per-pixel apply.
    bool BuildWorldGi();
    void ResetWorldGi(const NewDawnPostFxParams& params, int tier);
    void RunWorldGi(const NewDawnPostFxParams& params, const FramebufferOGL& fb, uint32_t frameIndex,
                    const Target* sceneCopy);
    const float* mWgiTris = nullptr;
    uint32_t mWgiTriCount = 0;
    struct WorldGi {
        bool attempted = false, ready = false;
        Program voxelize, inject, trace, blend, apply;
        GLuint voxAlbedo = 0, voxNormal = 0, voxCoarse = 0, voxRadiance = 0; // 3D
        GLuint probeIrr = 0, probeDist = 0, rays = 0;                         // 2D
        GLuint voxFbo = 0, radFbo = 0, rayFbo = 0, probeFbo = 0, vao = 0, vbo = 0;
        int tier = 0, scene = -1;
        int dim[3] = {}, origin[3] = {}, perProbe = 8, rayCount = 0, rayRows = 0; // window corner in world voxel cells
        float boundsMin[3] = {}, boundsMax[3] = {}, lo[3] = {}, hi[3] = {}, voxSize = 1.0f, probeStep = 1.0f;
        int sliceCursor = 0, rowCursor = 0;
        uint32_t frames = 0, updates = 0, sweeps = 0, lastFrame = UINT32_MAX, want = 0, tris = 0;
        // This frame's smoothed local lights without the designed flicker (filled by RunLocalLights).
        float lightPos[24][4] = {}, lightRgb[24][3] = {};
        int lightCount = 0;
        uint32_t lightFrame = UINT32_MAX;
    } mWgi;
    void RunSceneEnd(const FramebufferOGL& fb, const Settings& s, bool taa = false);
    // Resolves this frame's jittered world against the reprojected history; returns the resolved linear target, or
    // nullptr when TAA is not active (the caller then keeps using mScene).
    const Target* RunTaa(const FramebufferOGL& fb, const Settings& s);
    void Draw(const Target& t);
    void Draw(GLuint fbo, int w, int h);
    void Bind(int unit, GLuint tex);
    void Use(const Program& p);
    GLint Loc(const char* name);
    void BeginTimer(int stage);
    void EndTimer(int stage);
    void CheckError(const char* where);

    bool mUsable = false;
    bool mTimers = false; // stage timers run for the stats log or the performance overlay
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
    int mSkyCachedVariant = 0;
    float mSkyCachedMood = 0.0f;
    float mSkyCachedSettings[7] = {};
    void RunClouds(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void RestoreColor(const FramebufferOGL& fb);
    Program mAo, mBlur, mApply, mDown, mUp, mComposite;
    Program mDownHdr, mCompositeHdr, mCompositeNoDither, mShadowApplyHdr;
    Program mShadowDepth, mShadowApply, mShadowFoliage;
    Program mShadowApplyCsm, mShadowApplyHdrCsm, mShaftMarchCsm, mFogMarchCsm, mGiGatherCsm; // CSM variants
    Program mCaustics, mCausticsCsm, mUnderwater, mUnderwaterShaft, mUnderwaterShaftCsm;
    Target mUwShaft; // half resolution underwater light shafts
    GLuint mCausticTex = 0, mCausticDummy = 0; // 32-layer caustics flipbook on unit kCausticUnit
    std::string mCausticAssetPath;
    // Night sky: real star catalogue (point sprites) and the Milky Way panorama (unit kMilkyWayUnit; a dummy is
    // always bound there so programs that never use it cannot trip sampler-type conflicts).
    std::string mNightAssetPath;
    GLuint mMilkyWayTex = 0, mMilkyWayDummy = 0, mStarVao = 0, mStarVbo = 0;
    int mStarCount = 0;
    float mMilkyWayScale = 1.0f;
    Program mStars;
    uint32_t mCloudsDrawnFrame = UINT32_MAX;
    uint32_t mShadowReadyFrame = UINT32_MAX; // settings frame whose sun shadow map is valid
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
    // HDR local lighting (gfx_opengl_postfx.cpp RunLocalLights)
    void RunLocalLights(const NewDawnPostFxParams& params, const FramebufferOGL& fb, uint32_t frameIndex);
    Program mLocalLight;
    Target mLocalSource;
    std::vector<float> mFlameLights; // previous frame's VFX flames: world xyz + warm rgb
    uint32_t mFlameFrame = 0;
    int mFlameScene = -1;
    Target mGiSource, mGiRaw, mGiHistory[2];
    bool mGiValid = false;
    int mGiRead = 0, mGiScene = -1;
    uint32_t mGiFrame = UINT32_MAX;
    float mGiViewProj[16] = {}, mGiEye[3] = {}, mGiForward[3] = {};
    float mGiNdcXScale = 1.0f, mGiFlipY = 1.0f;
    Target mScene;
    Target mWaterOpaqueColor, mWaterOpaqueDepth; // captured before XLU, only when reflections are enabled
    Target mWaterReflectionRaw; // half-resolution premultiplied colour and reflection coverage
    // Water simulation: ocean swell trace + GPU ripple height field (docs/WATER_SIM.md).
    Program mWaterDepthMap, mWaterWaves, mWaterRipple;
    Target mWaveDepth, mWaveBuf, mRipple[2];
    GLuint mWaveDummy = 0;
    int mRippleRead = 0, mRippleScene = -1;
    bool mRippleValid = false;
    float mRippleMin[2] = {}, mRippleAccum = 0.0f, mRippleLastTime = 0.0f;
    uint32_t mWaveBufFrame = UINT32_MAX; // settings frame whose wave buffer is valid
    uint32_t mWaterOpaqueFrame = UINT32_MAX;
    Target mShadowMap;
    // Cascaded sun shadows, rebuilt by RunShadows each frame (count 1 = the legacy single map, nothing else is used).
    struct Cascades {
        int count = 1;
        int tile = 0;
        float vp[4][16] = {};       // world -> atlas clip
        float rect[4][4] = {};      // atlas uv bounds that are safe to filter in, per cascade
        float split[4] = {};        // far end of each cascade along the view axis
        float biasScale[4] = {};    // depth bias relative to the legacy map
        float texelScale[4] = {};   // texel size relative to the legacy map (scales the normal offset)
        float blend = 0.0f;
        float eye[3] = {}, fwd[3] = {}; // camera origin and view axis, for the view-axis distance
    } mCsm;
    void BuildCascades(const NewDawnPostFxParams& params, int atlasSize);
    void SetCascadeUniforms();
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
    // Temporal anti-aliasing (gNewDawn.TAA.*). History holds tone-compressed colour with the view depth in alpha.
    struct TaaCamera {
        bool valid = false;
        float vp[16] = {}, ndcXScale = 1.0f, flipY = 1.0f, zNear = 10.0f, zFar = 12800.0f, jitter[2] = {};
        uint32_t frame = 0;
    } mTaaCam;
    Program mTaa, mTaaMaskFill, mTaaSharpen;
    Target mTaaHist[2], mTaaOut, mTaaSharp, mTaaDepth, mTaaMask;
    int mTaaRead = 0;
    bool mTaaValid = false;
    uint32_t mTaaLastFrame = UINT32_MAX;
    float mTaaPrevVP[16] = {}, mTaaPrevEye[3] = {}, mTaaPrevFwd[3] = {};
    int mTaaPrevW = 0, mTaaPrevH = 0;
    bool mTaaDidReset = false; // diagnostics: the last resolve started from scratch
    float mKernel[16 * 3] = {};

    static constexpr int kQueryRing = 4;
    static constexpr int kQueryStages = 23;
    static constexpr int kWorldGiTimerStage = 22;
    static constexpr int kNightTimerStage = 20;
    static constexpr int kTaaTimerStage = 21;
    static constexpr int kLocalTimerStage = 16;
    static constexpr int kCausticsTimerStage = 17;
    static constexpr int kUnderwaterTimerStage = 18;
    static constexpr int kGiTimerStage = 14;
    static constexpr int kFlareTimerStage = 10;
    static constexpr int kFogTimerStage = 9;
    static constexpr int kVfxTimerStage = 15;
    static constexpr int kGrassTimerStage = 19;
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
    GLuint mWorldQuery[kQueryRing][2] = {};
    bool mWorldPending[kQueryRing] = {};
    int mWorldNext = 0;
    double mWorldTotalMs = 0.0;
    int mWorldCount = 0;
    int mDumpDone = 0; // gNewDawn.TAA.DumpFrames bookkeeping
};

} // namespace Fast
#endif
