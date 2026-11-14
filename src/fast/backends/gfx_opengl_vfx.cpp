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
constexpr int kSpriteUnit = 25, kDepthUnit = 26, kFlipUnit = 27;
// fire_flipbook_8x8.png: 64 frames in an 8x8 grid of 160x256 cells (authored offline, see tools/newdawn/vfx-flipbook).
constexpr int kFlipCols = 8, kFlipFrames = 64, kFlipW = 160, kFlipH = 256;
constexpr int kDim = 512;
// Layer order of the sprite array; files live in <asset root>/vfx/.
const char* kFiles[] = { "flame_05", "flame_06", "fire_01", "fire_02", "smoke_01", "smoke_04",
                         "smoke_07", "spark_01", "star_01", "light_01", "circle_05", "magic_02" };
constexpr int kLayers = sizeof(kFiles) / sizeof(kFiles[0]);
constexpr int kFlameLayers = 2, kEmberCount = 10;

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
        // Flipbook flame: the native quad's footprint and bottom anchor (not enlarged). Two layers: the main flame
        // and a mirrored, offset, fainter one so the internal structure never repeats in lock-step.
        int l = gl_InstanceID % 2;
        vLayerKind = float(l);
        float hh = aB.z;
        float aspect = uSize.y / uSize.x;
        float hgt = hh * (l == 0 ? 1.3 : 1.15); // the flame body fills ~75% of its cell, so this matches the native size
        float hwid = hh * 0.625 * aspect * (l == 0 ? 0.95 : 0.88);
        vec2 centre = vec2(aA.y, aA.z - hh + hgt);
        vec2 ndc = centre + vec2(q.x * hwid, q.y * hgt);
        vUv = vec2(l == 1 ? 1.0 - u01.x : u01.x, 1.0 - u01.y);
        vSeed = mod(uTime * 22.0 + hash(seed) * 64.0 + (l == 1 ? 29.0 : 0.0), 64.0); // frame position
        gl_Position = vec4(ndc.x * w, ndc.y * uFlipY * w, 0.0, w);
        vViewZ = w;
        vSize = H;
        vAlpha = aPrim.a * (l == 1 ? 0.55 : 1.0);
        return;
    }
    if (uMode == 4) { // magic arrow: a chain of sprites along the model's screen-space axis (24 per arrow)
        int cls = int(aA.x + 0.5);
        int k = gl_InstanceID % 24;
        vec2 axis = vec2(aB.y, aB.z) * uSize * 0.5; // centre -> tip, pixels
        float widthPx = aWorld.w * uSize.y * 0.5;   // half-width of the model, pixels
        float t = (float(k) + 0.5) / 24.0;          // 0 tail .. 1 tip
        float s2 = hash(seed + float(k) * 3.17);
        float tm = uTime * (cls == 9 ? 9.0 : 5.0) + s2 * 40.0;
        float body = 2.0 * t - 1.0;
        int role = k < 20 ? 0 : (k < 22 ? 1 : (k < 23 ? 2 : 3)); // body puff, tip star, halo, long streak
        vec2 dir = normalize(axis + vec2(0.0001));
        vec2 perp = vec2(-dir.y, dir.x);
        float len = length(axis);
        vec2 pxc = vec2(aA.y, aA.z) * uSize * 0.5;
        float jit = (sin(tm * 1.7 + float(k)) * 0.5 + (s2 - 0.5)) * widthPx * (cls == 9 ? 0.9 : 0.5);
        vec2 pos = pxc + axis * body + perp * jit - dir * (cls == 9 ? fract(tm * 0.15 + s2) * widthPx * 2.0 : 0.0);
        float r = widthPx * (cls == 9 ? 2.4 : 1.7) * (0.55 + 0.75 * t) * (0.8 + 0.4 * s2);
        float rot = tm * 0.35 + s2 * 6.28;
        int layer = cls == 9 ? (k % 2 == 0 ? 2 : 3) : (cls == 10 ? (k % 2 == 0 ? 5 : 4) : 10);
        // A near-round model is the arrow's hit burst (the game scales the model up): fill the disc instead of a streak.
        bool burst = len < widthPx * 1.8;
        if (burst) {
            vec2 sc = vec2(hash(seed + float(k) * 1.3), hash(seed + float(k) * 2.9)) * 2.0 - 1.0;
            pos = pxc + sc * widthPx * 0.6; r = widthPx * (0.55 + 0.45 * s2);
            if (cls == 11) { layer = k % 2 == 0 ? 8 : 10; r *= layer == 8 ? 2.2 : 0.9; }
            if (cls == 10) { layer = k % 3 == 0 ? 8 : (k % 3 == 1 ? 7 : 4); if (layer != 4) r *= 1.4; }
        }
        if (cls == 11 && role == 0) { if (k % 2 == 0) { layer = 8; r *= 1.9; } else { layer = 10; r *= 0.8; } }
        if (role == 1) { pos = burst ? pxc : pxc + axis * (0.55 + 0.45 * float(k - 20)); r = burst ? widthPx * 1.4 : max(widthPx * 4.0, len * 0.16); layer = 8; rot = 0.0; }
        else if (role == 2) { pos = burst ? pxc : pxc + axis * 0.9; r = burst ? widthPx * 2.2 : max(widthPx * 9.0, len * 0.55); layer = cls == 11 ? 9 : 10; rot = 0.0; }
        else if (role == 3) { r = 0.0; } // reserved (the native EffectBlure ribbon already draws the streak)
        vec2 qq = q;
        if (role == 3) { qq = vec2(q.x, q.y * 0.07); }
        vec2 local = vec2(cos(rot) * qq.x - sin(rot) * qq.y, sin(rot) * qq.x + cos(rot) * qq.y) * r;
        vec2 ndc = (pos + local) / (uSize * 0.5);
        vUv = vec2(u01.x, 1.0 - u01.y);
        vLayer = layer;
        vLayerKind = float(cls * 10 + role + (burst ? 4 : 0));
        vSeed = t;
        gl_Position = vec4(ndc.x * w, ndc.y * uFlipY * w, 0.0, w);
        vViewZ = w;
        vSize = max(H, 4.0);
        vAlpha = aPrim.a;
        return;
    }
    if (uMode == 3) { // puffs: dust, bomb fireball/smoke, hit sparks, electric sparks (2 layers per recorded quad)
        int cls = int(aA.x + 0.5);
        int l = gl_InstanceID % 2;
        float p = aWorld.w;
        vLayerKind = float(cls * 10 + l);
        vSeed = p;
        float rot = seed * 6.2831 + float(l) * 1.9 + p * (cls == 3 ? 0.5 : 0.25) * (0.5 + hash(seed));
        float scale = cls == 3 ? 0.85 : (cls == 4 ? (l == 0 ? 1.0 : 1.5) : (cls == 6 ? (l == 0 ? 1.7 : 0.8) : 1.0));
        vec2 hp = vec2(aB.y * uSize.x, aB.z * uSize.y) * 0.5; // half size in pixels
        float r = max(hp.x, hp.y) * scale;
        vec2 local = vec2(cos(rot) * q.x - sin(rot) * q.y, sin(rot) * q.x + cos(rot) * q.y) * r;
        vec2 ndc = vec2(aA.y, aA.z) + local / (uSize * 0.5);
        vUv = vec2(u01.x, 1.0 - u01.y);
        vLayer = cls == 3 ? (l == 0 ? 5 : 6) : (cls == 4 ? (hash(seed) < 0.5 ? (l == 0 ? 2 : 4) : (l == 0 ? 3 : 4)) : (cls == 6 ? (l == 0 ? 8 : 10) : (l == 0 ? 7 : 8)));
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
uniform sampler2DArray uSprite, uFlip;
uniform sampler2D uDepth;
uniform vec2 uSize, uNearFar;
uniform float uStrength;
uniform vec3 uLit;
uniform int uMode, uDebug;
float linearDepth(float d) {
    return 2.0*uNearFar.x*uNearFar.y / max(uNearFar.y+uNearFar.x-(2.0*d-1.0)*(uNearFar.y-uNearFar.x),0.001);
}
void main() {
    if (vViewZ <= 0.0 || vAlpha < 0.002) discard;
    float scene = linearDepth(texture(uDepth, gl_FragCoord.xy / uSize).r);
    // Geometry nearer than the flame hides it; the sprite is allowed to sit slightly inside its own holder.
    float soft = clamp((scene - vViewZ) / max(vSize, 4.0) + ((uMode == 3 || uMode == 4) ? 0.35 : 0.6), 0.0, 1.0);
    vec4 t;
    if (uMode == 0) {
        float f0 = floor(vSeed);
        t = mix(texture(uFlip, vec3(vUv, f0)), texture(uFlip, vec3(vUv, mod(f0 + 1.0, 64.0))), fract(vSeed));
    } else {
        t = texture(uSprite, vec3(vUv, float(vLayer)));
    }
    float v = t.r * t.a;
    vec3 rgb;
    float a;
    if (uMode == 4) { // magic arrows: HDR in the arrow's own prim/env colours
        int cls = int(vLayerKind / 10.0 + 0.01);
        int role = int(vLayerKind + 0.5) - cls * 10;
        float burstK = role >= 4 ? (cls == 11 ? (vLayer == 8 ? 0.9 : 0.22) : (cls == 10 ? 0.65 : 0.4)) : 1.0; // burst: many overlapping puffs on a disc, keep the total energy sane
        if (role >= 4) role -= 4;
        float v4 = t.a * t.r * burstK;
        vec3 base = mix(vEnv.rgb, vPrim.rgb, 0.55);
        vec3 c;
        float al;
        float tw = 0.8 + 0.2 * sin(vSeed * 40.0 + vViewZ);
        if (role == 0) {         // body
            c = (cls == 9 ? mix(vec3(1.6, 0.35, 0.05), vec3(7.0, 3.2, 0.7), vSeed) : base * mix(1.2, 3.0, vSeed) + vec3(0.2, 0.35, 0.5) * 0.4) * (0.7 + 0.6 * t.r);
            al = v4 * (cls == 9 ? 0.55 : (cls == 11 ? 0.6 : 0.8)) * vAlpha * (0.4 + 0.6 * vSeed);
        } else if (role == 1) {  // tip star
            c = mix(base, vec3(1.0), 0.35) * 5.0 * tw;
            al = v4 * vAlpha;
        } else if (role == 2) {  // halo
            c = base * (cls == 9 ? vec3(3.2, 1.3, 0.3) : vec3(1.3)) * 1.1;
            al = pow(v4, 1.6) * vAlpha * 0.55;
        } else {                 // light streak along the axis
            c = mix(base, vec3(1.0), 0.45) * 5.0;
            al = v4 * vAlpha * 0.5;
        }
        c *= uStrength;
        oColor = vec4(c * al * soft, 0.0);
        return;
    }
    if (uMode == 3) {
        int cls = int(vLayerKind / 10.0 + 0.01);
        int l = int(vLayerKind + 0.5) - cls * 10;
        float p = vSeed;
        if (cls == 3) { // dust: soft lit smoke, composited over the scene
            vec3 c = mix(vEnv.rgb, vPrim.rgb, 0.6) * uLit * (0.6 + 0.4 * t.r);
            c = mix(vec3(dot(c, vec3(0.30, 0.59, 0.11))), c, 0.55); // the game's sand/earth tints are saturated; keep dust close to the ground tone
            float al = clamp(t.a * vAlpha * 0.7, 0.0, 0.4) * soft * uStrength;
            if (uDebug == 1) { oColor = vec4(t.rgb * t.a * 3.0, 0.0); return; }
            oColor = vec4(c * al, al);
            return;
        }
        if (cls == 4 && l == 1) { // bomb smoke: grows in as the fireball cools
            float al = clamp(t.a * 1.5 * smoothstep(0.2, 0.75, p) * (1.0 - 0.45 * p), 0.0, 0.9) * soft * uStrength;
            oColor = vec4(vec3(0.075, 0.065, 0.058) * uLit * al, al);
            return;
        }
        vec3 c;
        float al;
        if (cls == 4) { // bomb fireball, white-hot to deep orange
            c = mix(vec3(7.0, 4.0, 1.4), vec3(2.2, 0.5, 0.08), smoothstep(0.0, 0.8, p)) * (1.0 - 0.6 * p);
            al = t.a * (0.4 + 0.6 * t.r) * (1.0 - smoothstep(0.55, 1.0, p));
        } else if (cls == 6) { // sparkle: crisp HDR 4-point star with a small halo; the game's env alpha is its twinkle
            float tw = mix(0.3, 1.0, clamp(vEnv.a, 0.0, 1.0));
            c = l == 0 ? mix(vEnv.rgb, vPrim.rgb, 0.45) * 5.5 * tw : vEnv.rgb * 2.5;
            al = l == 0 ? t.a * t.r * vAlpha * tw : pow(t.a * t.r, 1.5) * vAlpha * 0.35 * tw;
        } else if (cls == 5) { // hit spark
            c = max(vPrim.rgb, vec3(0.2)) * 5.0 * (1.0 - 0.6 * p);
            al = t.a * t.r * vAlpha * (l == 0 ? 0.9 : 0.6);
        } else { // electric
            c = vec3(2.2, 3.4, 7.0);
            al = t.a * t.r * vAlpha * (l == 0 ? 0.9 : 0.5);
        }
        oColor = vec4(c * al * soft * uStrength, 0.0);
        return;
    }
    if (uMode == 2) {
        rgb = vPrim.rgb * vec3(2.2, 1.9, 1.5);
        a = pow(t.a * t.r, 1.8) * vAlpha * 0.8;
    } else if (uMode == 1) {
        rgb = vec3(7.0, 2.4, 0.45) * mix(0.6, 1.4, fract(vSeed * 91.7));
        a = t.a * t.r * vAlpha;
    } else {
        // Flipbook flame: R = heat, A = coverage. Edges take the game's env colour, the body its prim colour; bright but
        // not clipped (peak about 2) so the gradient stays readable and bloom carries the glow. Partly transparent.
        vec3 cool = max(vEnv.rgb, vec3(0.05)) * 1.3;
        vec3 hot = mix(vPrim.rgb, vEnv.rgb + vec3(0.15), 0.25) * 2.3;
        float v = t.r;
        rgb = mix(cool, hot, smoothstep(0.05, 0.6, v)) * (0.65 + 1.0 * v);
        a = clamp(t.a * 1.25, 0.0, 1.0) * vAlpha * (vLayerKind > 0.5 ? 0.6 : 0.95);
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
    glUniform1i(Loc("uFlip"), kFlipUnit);

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

    {
        std::filesystem::path path = std::filesystem::path(root) / "vfx" / "fire_flipbook_8x8.png";
        if (!path.is_absolute())
            path = Ship::Context::LocateFileAcrossAppDirs(path.string());
        int w = 0, h = 0, n = 0;
        auto* sheet = stbi_load(path.string().c_str(), &w, &h, &n, 4);
        if (!sheet || w != kFlipCols * kFlipW || h != (kFlipFrames / kFlipCols) * kFlipH) {
            SPDLOG_WARN("NewDawnVfx: missing/unsupported fire flipbook {}", path.string());
            stbi_image_free(sheet);
            glDeleteProgram(mVfx.id);
            mVfx.id = 0;
            return false;
        }
        std::vector<uint8_t> layers((size_t)kFlipW * kFlipH * kFlipFrames * 4);
        for (int f = 0; f < kFlipFrames; ++f) {
            for (int row = 0; row < kFlipH; ++row) {
                const uint8_t* src = sheet + ((size_t)((f / kFlipCols) * kFlipH + row) * w + (f % kFlipCols) * kFlipW) * 4;
                std::memcpy(layers.data() + ((size_t)f * kFlipH + row) * kFlipW * 4, src, (size_t)kFlipW * 4);
            }
        }
        stbi_image_free(sheet);
        glActiveTexture(GL_TEXTURE0 + kFlipUnit);
        glGenTextures(1, &mVfxFlip);
        glBindTexture(GL_TEXTURE_2D_ARRAY, mVfxFlip);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, kFlipW, kFlipH, kFlipFrames, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     layers.data());
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
    }
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
    glActiveTexture(GL_TEXTURE0 + kFlipUnit);
    glBindTexture(GL_TEXTURE_2D_ARRAY, mVfxFlip);
    glBindSampler(kFlipUnit, 0);
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

    std::vector<float> flames, glows, puffs, arrows;
    mFlameLights.clear();
    for (uint32_t i = 0; i < mVfxCount; ++i) {
        const float* f = mVfxData + (size_t)i * NEWDAWN_VFX_FLOATS;
        if (f[0] == 1.0f) {
            flames.insert(flames.end(), f, f + NEWDAWN_VFX_FLOATS);
            mFlameLights.insert(mFlameLights.end(), f, f + NEWDAWN_VFX_FLOATS); // local lights read these next frame
        }
        else if (f[0] == 2.0f)
            glows.insert(glows.end(), f, f + NEWDAWN_VFX_FLOATS);
        else if (f[0] >= 9.0f)
            arrows.insert(arrows.end(), f, f + NEWDAWN_VFX_FLOATS);
        else if (f[0] >= 3.0f)
            puffs.insert(puffs.end(), f, f + NEWDAWN_VFX_FLOATS);
    }
    if (flames.empty() && glows.empty() && puffs.empty() && arrows.empty())
        return;
    if (p.shadowEnabled) {
        float lit[3];
        for (int i = 0; i < 3; ++i)
            lit[i] = 0.55f + 0.45f * std::clamp(p.shadowAmbient[i] + 0.6f * p.shadowDirColor[i], 0.0f, 1.0f); // dust stays pale, as natively
        glUniform3f(Loc("uLit"), lit[0], lit[1], lit[2]);
    } else {
        glUniform3f(Loc("uLit"), 1.0f, 1.0f, 1.0f);
    }
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
    if (!puffs.empty()) {
        glBufferData(GL_ARRAY_BUFFER, puffs.size() * sizeof(float), puffs.data(), GL_STREAM_DRAW);
        glUniform1i(Loc("uMode"), 3);
        for (int i = 0; i < 5; ++i)
            glVertexAttribDivisor(i, 2);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, (GLsizei)(puffs.size() / NEWDAWN_VFX_FLOATS) * 2);
    }
    if (!arrows.empty()) {
        if (Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Vfx.Debug", 0))
            SPDLOG_INFO("VFXDBG arrows={} ndc={:.3f},{:.3f} half={:.3f},{:.3f} wid={:.3f} w={:.0f}", arrows.size() / NEWDAWN_VFX_FLOATS, arrows[1], arrows[2], arrows[5], arrows[6], arrows[15], arrows[4]);
        glBufferData(GL_ARRAY_BUFFER, arrows.size() * sizeof(float), arrows.data(), GL_STREAM_DRAW);
        glUniform1i(Loc("uMode"), 4);
        for (int i = 0; i < 5; ++i)
            glVertexAttribDivisor(i, 24);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, (GLsizei)(arrows.size() / NEWDAWN_VFX_FLOATS) * 24);
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
