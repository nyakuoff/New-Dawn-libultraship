#pragma once
#include <stdint.h>

/* CPU-owned immutable scene data. Only the interpreter/backend consume this;
 * no game assets or N64 collision are owned by libultraship. */
struct NewDawnMM3DTexture {
    uint64_t id;
    uint32_t width, height;
    const uint8_t* rgba;
};

struct NewDawnMM3DBinding {
    int32_t texture;
    uint32_t wrapS, wrapT;
    uint32_t minFilter, magFilter;
    int32_t source, mapping;
    float transform[5]; /* scaleS/T, translationS/T, rotation in pi radians */
};

struct NewDawnMM3DMaterial {
    NewDawnMM3DBinding bindings[3];
    int32_t cull; /* 0 none, 1 back, 2 front */
    int32_t blend, alphaTest, depthTest, depthWrite, fog, fragmentLighting, vertexLighting;
    uint32_t alphaFunction, depthFunction;
    float alphaReference;
    uint32_t blendRgb[3], blendAlpha[3]; /* src, dst, equation: GL enums */
    float blendConstant[4];
    float constants[6][4];
    float diffuse[4], ambient[4], emission[4];
    int32_t combinerCount;
    int32_t combiners[6][20]; /* original PICA enums */
};

struct NewDawnMM3DDraw {
    uint32_t firstIndex, indexCount;
    NewDawnMM3DMaterial material;
};

struct NewDawnMM3DMesh {
    uint64_t id;
    uint32_t vertexCount, indexCount, drawCount, textureCount;
    /* 16 floats: pos3, normal3, uv0_2, colour4, uv1_2, uv2_2 */
    const float* vertices;
    const uint32_t* indices;
    const NewDawnMM3DDraw* draws;
    const NewDawnMM3DTexture* textures;
};

struct NewDawnMM3DRoomParams {
    int32_t scene, room, pass; /* pass 0 opaque/cutout, 1 blended */
    uint32_t meshCount;
    const NewDawnMM3DMesh* const* meshes;
    float ambient[3], lightDirection[2][3], lightColor[2][3]; /* normalized 0..1 */
    float fogColor[3];
    float fogNear; /* N64 fog position 0..996; far fog position is 1000 */
};
