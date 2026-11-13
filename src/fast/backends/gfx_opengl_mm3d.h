#pragma once
#ifndef USE_OPENGLES
#include "fast/backends/gfx_opengl.h"
#include "fast/newdawn_mm3d.h"
#include <array>
#include <unordered_map>

namespace Fast {
class MM3DGL {
  public:
    ~MM3DGL();
    bool Prepare(const NewDawnMM3DRoomParams& params);
    void Draw(const NewDawnMM3DRoomParams& params, const float* projection, float aspect, float flipY,
              bool hdr, bool mipmaps, float anisotropy, float lodBias);

  private:
    struct Mesh { GLuint vao = 0, vbo = 0, ibo = 0; };
    GLuint mProgram = 0;
    bool mFailed = false;
    std::unordered_map<uint64_t, Mesh> mMeshes;
    std::unordered_map<uint64_t, GLuint> mTextures;
    std::unordered_map<std::string, GLint> mUniforms;
    std::array<GLuint, 8> mQueries{};
    std::array<bool, 8> mQueryPending{};
    unsigned mQuerySlot = 0, mTimerCount = 0;
    double mGpuMs = 0, mCpuMs = 0;
    bool Build();
    GLint U(const std::string& name);
};
}
#endif
