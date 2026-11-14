#if defined(ENABLE_OPENGL) && !defined(USE_OPENGLES)
// New Dawn VFX: native effect sprites recorded by the interpreter (clip-space quads, see Interpreter::RecordVfxTriangle)
// are redrawn as HD HDR sprites from downloaded CC0 art. Render-side only; the native draw is suppressed per class.
#include "fast/backends/gfx_opengl_postfx.h"
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"
#include <stb_image.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <vector>

namespace Fast {
namespace {
constexpr int kSpriteUnit = 25, kDepthUnit = 26;
constexpr int kDim = 512;
// Layer order of the sprite array; files live in <asset root>/vfx/.
const char* kFiles[] = { "flame_05", "flame_06", "fire_01", "fire_02", "smoke_01", "smoke_04",
                         "smoke_07", "spark_01", "star_01", "light_01", "circle_05", "magic_02" };
constexpr int kLayers = sizeof(kFiles) / sizeof(kFiles[0]);
constexpr int kFlameLayers = 4, kEmberCount = 10;

const char* kVertex = R"GLSL(#version 330 core
layout(location=0) in vec4 aA; // cls, ndc x, ndc y, ndc z
layout(location=1) in vec4 aB; // view depth, half w, half h (ndc), seed
layout(location=2) in vec4 aPrim;
layout(location=3) in vec4 aWorld; // world xyz
layout(location=4) in vec4 aEnv;
uniform mat4 uViewProj;
uniform float uNdcXScale, uFlipY, uTime;
uniform vec2 uProjection, uSize;
uniform vec3 uWind;
uniform float uEmbers;
uniform int uMode; // 0 flame layers, 1 embers
out vec2 vUv;
out float vViewZ, vAlpha, vLayerKind, vSeed, vSize;
flat out int vLayer;
out vec4 vPrim, vEnv;
float hash(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
void main() {
    vec2 corners[6] = vec2[6](vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
    vec2 q = corners[gl_VertexID];
    float seed = aB.w, w = aB.x;
    vPrim = aPrim; vEnv = aEnv; vSeed = seed;
    float H = aB.z * w / uProjection.y; // world half-height of the native sprite
    vec2 u01 = q * 0.5 + 0.5;
    if (uMode == 0) {
        int l = gl_InstanceID % 4;
        vLayerKind = float(l);
        float t = uTime;
        float flutter = sin(t * (7.0 + hash(seed) * 3.0) + float(l) * 1.7 + seed) + 0.5 * sin(t * 13.0 + seed * 3.0 + float(l));
        float hh = aB.z;
        float aspect = uSize.y / uSize.x;
        // Flame glyphs occupy a sub-rectangle of their 512px sprites; the quad maps to that rectangle (v flipped).
        vec2 size, centre;
        vLayer = 0;
        if (l == 3) { // soft halo
            vLayer = 10;
            size = vec2(hh * 9.0 * aspect, hh * 9.0);
            centre = vec2(aA.y, aA.z + hh * 0.1);
            vUv = vec2(u01.x, 1.0 - u01.y);
        } else {
            float h = hh * (l == 2 ? 1.4 : 2.3);
            if (l == 1) { vLayer = 1; h *= 1.0 + 0.08 * sin(t * 6.0 + seed); }
            size = vec2(h * 0.4 * aspect, h);
            centre = vec2(aA.y, aA.z - hh * 0.95 + h * 0.5);
            vUv = (l == 1) ? vec2(0.38 + u01.x * 0.26, 0.14 + (1.0 - u01.y) * 0.62)
                           : vec2(0.40 + u01.x * 0.22, 0.20 + (1.0 - u01.y) * 0.55);
        }
        vec2 offset = vec2(q.x * size.x * 0.5, q.y * size.y * 0.5 * (1.0 + 0.07 * flutter * float(l != 3)));
        if (l != 3) offset.x += flutter * size.x * 0.13 * u01.y * u01.y; // tip sways, base stays put
        vec2 ndc = centre + offset;
        gl_Position = vec4(ndc.x * w, ndc.y * uFlipY * w, 0.0, w);
        vViewZ = w;
        vSize = H;
        vAlpha = aPrim.a;
        return;
    }
    if (uMode == 2) { // light glow: the native quad's footprint, soft halo sprite
        vLayer = 10;
        vLayerKind = 5.0;
        vec2 size = vec2(aB.y, aB.z) * 2.0 * 2.0; // sprite disc fills half the texture
        vUv = vec2(u01.x, 1.0 - u01.y);
        gl_Position = vec4((aA.y + q.x * size.x * 0.5) * w, (aA.z + q.y * size.y * 0.5) * uFlipY * w, 0.0, w);
        vViewZ = w;
        vSize = H;
        vAlpha = aPrim.a;
        return;
    }
    // Embers rise from the flame and cool; deterministic in renderer time and the flame's world seed.
    int e = gl_InstanceID % 10;
    float s = hash(seed * 7.13 + float(e) * 3.77);
    float s2 = hash(seed * 1.91 + float(e) * 9.31);
    float period = 1.6 + s * 2.2;
    float ph = fract(uTime / period + s2);
    vec3 p = aWorld.xyz + vec3(0.0, H * 0.6, 0.0);
    float rise = ph * H * (3.5 + 3.0 * s);
    p.y += rise;
    p.x += (sin(uTime * (1.1 + s) + s2 * 6.28) * 0.55 + (s - 0.5) * 0.8) * H * 0.45 * ph + uWind.x * ph * H;
    p.z += (cos(uTime * (0.9 + s2) + s * 6.28) * 0.55 + (s2 - 0.5) * 0.8) * H * 0.45 * ph + uWind.z * ph * H;
    vec4 clip = uViewProj * vec4(p, 1.0);
    gl_Position = clip;
    gl_Position.x /= uNdcXScale;
    gl_Position.y *= uFlipY;
    float radius = H * (0.10 + 0.10 * s2) * (1.0 - ph * 0.55);
    vUv = vec2(u01.x, 1.0 - u01.y);
    gl_Position.xy += q * radius * uProjection;
    vViewZ = clip.w;
    vLayer = 10;
    vLayerKind = 4.0;
    vSize = H;
    float flick = 0.6 + 0.4 * sin(uTime * (9.0 + s * 8.0) + s2 * 40.0);
    vAlpha = aPrim.a * smoothstep(0.0, 0.07, ph) * pow(1.0 - ph, 1.4) * flick * (float(e) < uEmbers * 10.0 ? 1.0 : 0.0);
    vSeed = ph;
}
)GLSL";

const char* kFragment = R"GLSL(#version 330 core
in vec2 vUv;
in float vViewZ, vAlpha, vLayerKind, vSeed, vSize;
flat in int vLayer;
in vec4 vPrim, vEnv;
out vec4 oColor;
uniform sampler2DArray uSprite;
uniform sampler2D uDepth;
uniform vec2 uSize, uNearFar;
uniform float uStrength;
uniform int uMode, uDebug;
float linearDepth(float d) {
    return 2.0*uNearFar.x*uNearFar.y / max(uNearFar.y+uNearFar.x-(2.0*d-1.0)*(uNearFar.y-uNearFar.x),0.001);
}
void main() {
    if (vViewZ <= 0.0 || vAlpha < 0.002) discard;
    float scene = linearDepth(texture(uDepth, gl_FragCoord.xy / uSize).r);
    // Geometry nearer than the flame hides it; the sprite is allowed to sit slightly inside its own holder.
    float soft = clamp((scene - vViewZ) / max(vSize, 4.0) + 0.6, 0.0, 1.0);
    vec4 t = texture(uSprite, vec3(vUv, float(vLayer)));
    float v = t.r * t.a;
    vec3 rgb;
    float a;
    if (uMode == 2) {
        rgb = vPrim.rgb * vec3(2.2, 1.9, 1.5);
        a = pow(t.a * t.r, 1.8) * vAlpha * 1.1;
    } else if (uMode == 1) {
        rgb = vec3(7.0, 2.4, 0.45) * mix(0.6, 1.4, fract(vSeed * 91.7));
        a = t.a * t.r * vAlpha;
    } else if (vLayerKind > 2.5) {
        // Halo: orange light feel, additive and faint.
        rgb = mix(vEnv.rgb, vPrim.rgb, 0.55) * vec3(1.6, 0.9, 0.35);
        a = pow(t.a * t.r, 1.5) * vAlpha * 0.10;
    } else {
        // Game colours keep per-flame hues (yellow/red torch, blue and green variants); HDR gain is added on top.
        vec3 cool = max(vEnv.rgb, vec3(0.05)) * 1.4;
        vec3 hot = vPrim.rgb * vec3(3.3, 2.5, 3.3);
        rgb = mix(cool, hot, smoothstep(0.12, 0.62, v));
        rgb += mix(vec3(1.0), normalize(hot + 0.05) * 1.7, 0.6) * smoothstep(0.75, 1.0, v) * (vLayerKind > 1.5 ? 1.4 : 0.7);
        a = clamp(t.a * (vLayerKind > 1.5 ? 0.9 : 0.75), 0.0, 1.0) * vAlpha;
    }
    a *= soft * uStrength;
    if (uDebug == 1) { oColor = vec4(t.rgb * t.a * 3.0, 0.0); return; }
    if (uDebug == 2) { oColor = vec4(vec3(soft), 0.0) * 0.5; return; }
    if (uDebug == 3) { oColor = vec4(vec3(vAlpha * uStrength), 0.0) * 0.5; return; }
    oColor = vec4(rgb * a, 0.0); // additive in-scattering
}
)GLSL";

GLuint Compile(GLenum type, const char* text) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &text, nullptr);
    glCompileShader(shader);
    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        SPDLOG_ERROR("NewDawnVfx shader: {}", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}
} // namespace

bool PostFxGL::BuildVfxProgram() {
    if (mVfxAttempted)
        return mVfx.id != 0;
    mVfxAttempted = true;
    GLuint vs = Compile(GL_VERTEX_SHADER, kVertex), fs = Compile(GL_FRAGMENT_SHADER, kFragment);
    if (!vs || !fs) {
        glDeleteShader(vs);
        glDeleteShader(fs);
        return false;
    }
    mVfx.id = glCreateProgram();
    glAttachShader(mVfx.id, vs);
    glAttachShader(mVfx.id, fs);
    glLinkProgram(mVfx.id);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok;
    glGetProgramiv(mVfx.id, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(mVfx.id, sizeof(log), nullptr, log);
        SPDLOG_ERROR("NewDawnVfx link: {}", log);
        glDeleteProgram(mVfx.id);
        mVfx.id = 0;
        return false;
    }
    GLint count;
    glGetProgramiv(mVfx.id, GL_ACTIVE_UNIFORMS, &count);
    for (GLint i = 0; i < count; i++) {
        char name[256];
        GLsizei length;
        GLint size;
        GLenum type;
        glGetActiveUniform(mVfx.id, i, sizeof(name), &length, &size, &type, name);
        mVfx.uniforms[name] = glGetUniformLocation(mVfx.id, name);
    }
    Use(mVfx);
    glUniform1i(Loc("uSprite"), kSpriteUnit);
    glUniform1i(Loc("uDepth"), kDepthUnit);

    // Sprite array from the downloaded CC0 pack; any missing file disables the whole feature (no procedural fallback).
    const std::string root = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetString(
        "gNewDawn.Assets.Path", "assets/newdawn");
    std::vector<uint8_t> pixels((size_t)kDim * kDim * kLayers * 4, 0);
    for (int layer = 0; layer < kLayers; ++layer) {
        std::filesystem::path path = std::filesystem::path(root) / "vfx" / (std::string(kFiles[layer]) + ".png");
        if (!path.is_absolute())
            path = Ship::Context::LocateFileAcrossAppDirs(path.string());
        int w = 0, h = 0, n = 0;
        auto* data = stbi_load(path.string().c_str(), &w, &h, &n, 4);
        if (!data || w != kDim || h != kDim) {
            SPDLOG_WARN("NewDawnVfx: missing/unsupported sprite {}; set gNewDawn.Assets.Path to the asset root",
                        path.string());
            stbi_image_free(data);
            glDeleteProgram(mVfx.id);
            mVfx.id = 0;
            return false;
        }
        std::memcpy(pixels.data() + (size_t)layer * kDim * kDim * 4, data, (size_t)kDim * kDim * 4);
        stbi_image_free(data);
    }
    glActiveTexture(GL_TEXTURE0 + kSpriteUnit);
    glGenTextures(1, &mVfxSprites);
    glBindTexture(GL_TEXTURE_2D_ARRAY, mVfxSprites);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, kDim, kDim, kLayers, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);

    glGenVertexArrays(1, &mVfxVao);
    glGenBuffers(1, &mVfxVbo);
    glBindVertexArray(mVfxVao);
    glBindBuffer(GL_ARRAY_BUFFER, mVfxVbo);
    // Per-instance record (NEWDAWN_VFX_FLOATS): A=[0..3] B=[4..7] prim=[8..11] world=[12..15] env=[16..19].
    for (int i = 0; i < 5; ++i) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, 4, GL_FLOAT, GL_FALSE, NEWDAWN_VFX_FLOATS * sizeof(float),
                              (const void*)(i * 4 * sizeof(float)));
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    SPDLOG_INFO("NewDawnVfx: loaded {} CC0 sprites from {}", kLayers, root);
    return true;
}

void PostFxGL::RunVfx(const NewDawnPostFxParams& p, const FramebufferOGL& fb) {
    if (!mVfxData || mVfxCount == 0 || !BuildVfxProgram())
        return;
    Use(mVfx);
    glActiveTexture(GL_TEXTURE0 + kSpriteUnit);
    glBindTexture(GL_TEXTURE_2D_ARRAY, mVfxSprites);
    glBindSampler(kSpriteUnit, 0);
    glActiveTexture(GL_TEXTURE0 + kDepthUnit);
    glBindTexture(GL_TEXTURE_2D, mDepth.tex);
    glBindSampler(kDepthUnit, 0);
    glUniformMatrix4fv(Loc("uViewProj"), 1, GL_FALSE, p.shadowViewProj);
    glUniform1f(Loc("uNdcXScale"), p.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), p.shadowFlipY);
    const float projection = 1.0f / std::tan(p.fovyDegrees * 0.00872664626f);
    glUniform2f(Loc("uProjection"), projection * float(fb.height) / fb.width, projection);
    glUniform2f(Loc("uSize"), fb.width, fb.height);
    glUniform2f(Loc("uNearFar"), p.zNear, p.zFar);
    glUniform1i(Loc("uDebug"), Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Vfx.DebugView", 0));
    glUniform1f(Loc("uStrength"), std::clamp(p.vfxStrength, 0.0f, 4.0f));
    glUniform3f(Loc("uWind"), 0.0f, 0.0f, 0.0f);
    static const auto epoch = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
    glUniform1f(Loc("uTime"), static_cast<float>(std::fmod(seconds, 3600.0)));

    std::vector<float> flames, glows;
    for (uint32_t i = 0; i < mVfxCount; ++i) {
        const float* f = mVfxData + (size_t)i * NEWDAWN_VFX_FLOATS;
        if (f[0] == 1.0f)
            flames.insert(flames.end(), f, f + NEWDAWN_VFX_FLOATS);
        else if (f[0] == 2.0f)
            glows.insert(glows.end(), f, f + NEWDAWN_VFX_FLOATS);
    }
    if (flames.empty() && glows.empty())
        return;
    const GLsizei n = (GLsizei)(flames.size() / NEWDAWN_VFX_FLOATS);
    const GLsizei ng = (GLsizei)(glows.size() / NEWDAWN_VFX_FLOATS);
    if (Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Vfx.Debug", 0)) {
        static int logged = 0;
        if (flames.size() >= 8 && logged++ < 20) {
            SPDLOG_INFO("VFXDBG draw flames={} first ndc={:.3f},{:.3f} w={:.0f} half={:.4f},{:.4f} fbo={} {}x{}", n,
                        flames[1], flames[2], flames[4], flames[5], flames[6], fb.fbo, fb.width, fb.height);
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
    glViewport(0, 0, fb.width, fb.height);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(mVfxVao);
    glBindBuffer(GL_ARRAY_BUFFER, mVfxVbo);
    if (ng > 0) {
        glBufferData(GL_ARRAY_BUFFER, glows.size() * sizeof(float), glows.data(), GL_STREAM_DRAW);
        glUniform1i(Loc("uMode"), 2);
        for (int i = 0; i < 5; ++i)
            glVertexAttribDivisor(i, 1);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, ng);
    }
    if (n == 0) {
        for (int i = 0; i < 5; ++i)
            glVertexAttribDivisor(i, 0);
        glBindVertexArray(mVao);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        return;
    }
    glBufferData(GL_ARRAY_BUFFER, flames.size() * sizeof(float), flames.data(), GL_STREAM_DRAW);
    glUniform1i(Loc("uMode"), 0);
    for (int i = 0; i < 5; ++i)
        glVertexAttribDivisor(i, kFlameLayers);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, n * kFlameLayers);
    const float embers = std::clamp(p.vfxEmbers, 0.0f, 2.0f);
    if (embers > 0.0f) {
        glUniform1i(Loc("uMode"), 1);
        glUniform1f(Loc("uEmbers"), embers);
        for (int i = 0; i < 5; ++i)
            glVertexAttribDivisor(i, kEmberCount);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, n * kEmberCount);
    }
    for (int i = 0; i < 5; ++i)
        glVertexAttribDivisor(i, 0);
    glBindVertexArray(mVao);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
} // namespace Fast
#endif
