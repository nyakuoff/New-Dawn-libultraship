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
// backend's cached state (and whatever is drawn afterwards) is unaffected. Stateless across calls.
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

    bool BeginHdr(const FramebufferOGL& fb);
    void EndHdr(const FramebufferOGL& fb);
    GLuint HdrSnapshot(const FramebufferOGL& fb);
    bool HdrActive() const {
        return mHdrActive;
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
    bool BuildPrograms();
    void EnsureTarget(Target& t, int w, int h, GLint internalFormat, GLenum format, GLenum type, bool depthStencil,
                      bool linear);
    void PrepareDepth(const FramebufferOGL& fb);
    void EnsureShadowTarget(int size);
    void RunShadows(const NewDawnPostFxParams& params, const FramebufferOGL& fb, const float* positions,
                    uint32_t vertexCount, int debug);
    void RunWater(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
    void RunSunShafts(const NewDawnPostFxParams& params, const FramebufferOGL& fb);
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
    Settings mSettings{};
    uint32_t mStatsW = 0, mStatsH = 0, mStatsMsaa = 0;

    bool mHdrActive = false;
    Target mHdr;
    GLuint mHdrDisplayFbo = 0;
    Program mHdrTransform;
    Program mAo, mBlur, mApply, mDown, mUp, mComposite;
    Program mShadowDepth, mShadowApply;
    Program mShaftMarch, mShaftBlur, mShaftApply, mWater;
    GLuint mCmpSampler = 0; // linear + depth-compare sampler for the shadow map (bound to its own texture unit)
    const Program* mCurrentProgram = nullptr;
    GLuint mVao = 0;
    GLuint mShadowVao = 0, mShadowVbo = 0;

    Target mDepth;
    Target mAoRaw, mAoBlur;
    Target mScene;
    Target mShadowMap;
    Target mShaftA, mShaftB; // half resolution
    static constexpr int kMaxBloomLevels = 6;
    Target mBloom[kMaxBloomLevels];
    float mKernel[16 * 3] = {};

    static constexpr int kQueryRing = 4;
    static constexpr int kQueryStages = 5;
    static constexpr int kShadowTimerStage = 2;
    static constexpr int kShaftTimerStage = 3;
    static constexpr int kWaterTimerStage = 4;
    GLuint mQueries[kQueryStages][kQueryRing] = {};
    bool mQueryPending[kQueryStages][kQueryRing] = {};
    int mQueryNext[kQueryStages] = {};
    double mQueryTotalMs[kQueryStages] = {};
    int mQueryCount[kQueryStages] = {};
};

} // namespace Fast
#endif
