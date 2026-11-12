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
    void Run(int stage, const NewDawnPostFxParams& params, const FramebufferOGL& gameFb, uint32_t frameIndex);

    struct Settings {
        bool aoOn, bloomOn, gradeOn;
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
    };
    struct Program {
        GLuint id = 0;
        std::unordered_map<std::string, GLint> uniforms; // cached when linked
    };
    bool BuildPrograms();
    void EnsureTarget(Target& t, int w, int h, GLint internalFormat, GLenum format, GLenum type, bool depthStencil,
                      bool linear);
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

    Program mAo, mBlur, mApply, mDown, mUp, mComposite;
    const Program* mCurrentProgram = nullptr;
    GLuint mVao = 0;

    Target mDepth;
    Target mAoRaw, mAoBlur;
    Target mScene;
    static constexpr int kMaxBloomLevels = 6;
    Target mBloom[kMaxBloomLevels];
    float mKernel[16 * 3] = {};

    static constexpr int kQueryRing = 4;
    GLuint mQueries[2][kQueryRing] = {};
    bool mQueryPending[2][kQueryRing] = {};
    int mQueryNext[2] = {};
    double mQueryTotalMs[2] = {};
    int mQueryCount[2] = {};
};

} // namespace Fast
#endif
