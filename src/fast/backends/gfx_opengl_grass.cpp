#if defined(ENABLE_OPENGL) && !defined(USE_OPENGLES)
// New Dawn grass: render-only instanced tufts on ground triangles whose texture is classified "grass".
//
// The interpreter records those triangles in world space (Interpreter::RecordGrassTriangle). Here the world is cut into a
// fixed grid of cells (so blade positions never depend on the camera or on how the ground mesh is split), every cell that
// overlaps a recorded triangle near the camera becomes one small record, and the vertex shader scatters up to 32 tufts per
// cell from a position hash, keeps those that fall inside the triangle, and fades the density with distance. Tufts take
// their colour from the finished ground colour behind their root (so per-pixel lighting, fog, time of day and dusk tint
// all match the ground) and sway with the shared wind. Drawn at the opaque-end marker, before the depth copy.
#include "fast/backends/gfx_opengl_postfx.h"
#include "fast/newdawn_wind.h"
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"
#include <stb_image.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace Fast {
namespace {
constexpr int kGroundUnit = 29, kAtlasUnit = 30;
constexpr float kCell = 20.0f;   // world units
constexpr int kSlots = 32;       // tuft slots per cell at full density
constexpr int kBands[4] = { 32, 16, 8, 4 };
constexpr int kRecordFloats = 12;
constexpr size_t kMaxCells = 70000;
constexpr int kTuftVertices = 24; // two crossed quads, two height segments each

const char* kVertex = R"GLSL(#version 330 core
layout(location=0) in vec4 aR0; // cell min x, cell min z, p0.x, p0.y
layout(location=1) in vec4 aR1; // p0.z, p1.xyz
layout(location=2) in vec4 aR2; // p2.xyz, unused
uniform mat4 uViewProj;
uniform float uNdcXScale, uFlipY;
uniform vec3 uEye, uFocus, uSunDir;
uniform float uCell, uDistance, uDensityNear, uHeight, uWindScale;
uniform int uSlotCount;
uniform vec3 uWind;   // dir x, dir z, gust
uniform vec3 uPhase;  // travelling-wave phases (radians)
uniform vec2 uInvSize;
uniform sampler2D uGround;
uniform int uPushCount, uFlower, uSegs;   // pushers (Link, nearby actors, their recent trail); ground-cover pass; height segments
uniform vec4 uPush[36];                    // x, z, age in seconds, radius
uniform vec4 uFlowerMix;                   // weights of daisy, buttercup, wildflower, clover
uniform float uFlowerDensity, uTrans;      // ground-cover amount; extra sun-through-blades strength (0 = original look)
out vec2 vUv;
flat out int vKind;
out vec3 vGround;
out float vHeight, vTrans, vTipTint, vPatch;

uint ihash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
float rnd(inout uint s) {
    s = ihash(s + 0x9e3779b9u);
    return float(s & 0xFFFFFFu) * (1.0 / 16777216.0);
}
vec3 groundSample(vec2 uv) {
    vec2 o = uInvSize * 5.0;
    vec3 c = textureLod(uGround, uv, 0.0).rgb * 0.4;
    c += textureLod(uGround, uv + vec2(o.x, 0.0), 0.0).rgb * 0.15;
    c += textureLod(uGround, uv - vec2(o.x, 0.0), 0.0).rgb * 0.15;
    c += textureLod(uGround, uv + vec2(0.0, o.y), 0.0).rgb * 0.15;
    c += textureLod(uGround, uv - vec2(0.0, o.y), 0.0).rgb * 0.15;
    return c;
}
const vec2 kCorner[6] = vec2[6](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,0), vec2(1,1), vec2(0,1));

void cull() {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vUv = vec2(0.0); vGround = vec3(0.0); vHeight = 0.0; vTrans = 0.0; vTipTint = 0.0; vPatch = 0.0; vKind = 0;
}

// Sum of the pushes on a point: away from each pusher, strong at its feet, springing back over about a second.
vec3 pushAt(vec2 xz) {
    vec2 sum = vec2(0.0);
    for (int i = 0; i < uPushCount; ++i) {
        vec4 q = uPush[i];
        vec2 d = xz - q.xy;
        float dist = length(d);
        if (dist >= q.w) continue;
        float fall = 1.0 - smoothstep(q.w * 0.30, q.w, dist);
        float age = q.z;
        float life = exp(-3.2 * age) * (1.0 + 0.45 * sin(age * 7.5)) * (1.0 - smoothstep(0.85, 1.2, age));
        sum += (dist > 0.3 ? d / dist : vec2(1.0, 0.0)) * (fall * max(life, 0.0));
    }
    float l = length(sum);
    return vec3(l > 1.0 ? sum / l : sum, min(l, 1.0));
}

void main() {
    int slot = gl_InstanceID % uSlotCount;
    int ix = int(floor(aR0.x / uCell + 0.5)), iz = int(floor(aR0.y / uCell + 0.5));
    uint seed = ihash(uint(ix * 1973 + 9277) ^ ihash(uint(iz * 6271 + 19349) + uint(slot) * 26699u + 1u + (uFlower != 0 ? 7777u : 0u)));
    float r0 = rnd(seed), r1 = rnd(seed);
    vec2 pxz = aR0.xy + vec2(r0, r1) * uCell;
    vec3 p0 = vec3(aR0.zw, aR1.x), p1 = aR1.yzw, p2 = aR2.xyz;
    vec2 e0 = p1.xz - p0.xz, e1 = p2.xz - p0.xz, e2 = pxz - p0.xz;
    float den = e0.x * e1.y - e1.x * e0.y;
    if (abs(den) < 1e-4) { cull(); return; }
    float b1 = (e2.x * e1.y - e1.x * e2.y) / den, b2 = (e0.x * e2.y - e2.x * e0.y) / den, b0 = 1.0 - b1 - b2;
    if (b0 < 0.0 || b1 < 0.0 || b2 < 0.0) { cull(); return; }
    vec3 root = p0 * b0 + p1 * b1 + p2 * b2;

    float u = distance(root, uEye) / uDistance;
    float f = uDensityNear * (1.0 - 0.88 * smoothstep(0.12, 0.95, u));
    float rank = (float(slot) + 0.5) / 32.0;
    float vis = clamp((f - rank) * 8.0, 0.0, 1.0);
    // Everything that changes the look with distance is continuous, so nothing pops while walking.
    vis *= 1.0 - smoothstep(0.80, 1.0, u);
    if (vis <= 0.0) { cull(); return; }

    float r2 = rnd(seed), r3 = rnd(seed), r4 = rnd(seed), r5 = rnd(seed), r6 = rnd(seed);
    float kindF = 0.0;
    if (uFlower != 0) {
        // Ground cover grows in meadow-scale patches (a function of the world position only); a patch is mostly one kind.
        vis *= 1.0 - smoothstep(0.30, 0.45, u);
        float pn = 0.5 + 0.25 * sin(root.x * 0.0155 + 1.3 * sin(root.z * 0.0112)) + 0.25 * sin(root.z * 0.0191 - root.x * 0.0071 + 2.1);
        float pg = smoothstep(0.50, 0.76, pn);
        if (r6 > pg * uFlowerDensity * 0.85 || vis <= 0.0) { cull(); return; }
        uint ps = ihash(uint(int(floor(root.x / 85.0)) * 4099 + 3) ^ ihash(uint(int(floor(root.z / 85.0)) * 6791 + 11)));
        float kr = mix(r5, rnd(ps), 0.82);
        float tot = max(dot(uFlowerMix, vec4(1.0)), 1e-4);
        kr *= tot;
        kindF = kr < uFlowerMix.x ? 0.0 : (kr < uFlowerMix.x + uFlowerMix.y ? 1.0 : (kr < uFlowerMix.x + uFlowerMix.y + uFlowerMix.z ? 2.0 : 3.0));
    }
    float comp = clamp(sqrt(uDensityNear / max(f, 0.05)), 1.0, 2.4); // sparser far field: bigger tufts keep coverage
    float pd = length(root.xz - uFocus.xz);
    float lay = mix(0.45, 1.0, smoothstep(6.0, 28.0, pd)); // bent down right at the player's feet only (render only)
    float h = 11.0 * uHeight * (0.65 + 0.8 * r2) * (1.0 + 0.3 * (comp - 1.0)) * vis * lay;
    float wd = 11.0 * sqrt(uHeight) * (0.95 + 0.4 * r3) * (1.0 + 0.35 * (comp - 1.0)) * vis * mix(0.85, 1.0, lay);
    if (uFlower != 0) { // a cluster of a few stems: small next to the grass, a little taller than the shortest blades
        h = 15.0 * (0.7 + 0.5 * r2) * vis * lay;
        wd = 16.0 * (0.8 + 0.5 * r3) * vis * mix(0.85, 1.0, lay);
    }

    int id = gl_VertexID;
    int q = id / 12, r = id % 12, seg = r / 6, k = r % 6;
    vec2 cn = kCorner[k];
    float v = (float(seg) + cn.y) / float(uSegs);
    float lateral = cn.x - 0.5;
    float yaw = r4 * 3.14159265 + float(q) * 1.5707963;
    vec3 right = vec3(cos(yaw), 0.0, sin(yaw));
    vec3 pos = root + vec3(0.0, -0.4, 0.0) + right * (lateral * wd) + vec3(0.0, v * h, 0.0);

    // Shared wind: a wave travelling along the wind direction (the phase advances with the gust integral) plus a cross
    // ripple and a fast flutter. Tips move, roots stay planted.
    float bendW = v * v;
    float ph = dot(root.xz, uWind.xy) * 0.011 - uPhase.x + r5 * 0.6;
    float rip = dot(root.xz, vec2(-uWind.y, uWind.x)) * 0.031 - uPhase.y;
    float wave = sin(ph) * 0.65 + sin(rip) * 0.35;
    float amp = (0.35 + 0.65 * (0.5 + 0.5 * wave)) * uWind.z;
    vec2 perp = vec2(-uWind.y, uWind.x);
    vec2 d = (uWind.xy * amp * 0.30 + perp * sin(uPhase.z + r6 * 13.0) * 0.05 * uWind.z) * h * bendW * uWindScale;
    pos.xz += d;
    pos.y -= length(d) * 0.35 * bendW;
    if (uPushCount > 0) { // bent away from Link and other actors (render only), then back up
        vec3 pu = pushAt(root.xz);
        pos.xz += pu.xy * (0.85 * h * bendW);
        pos.y -= pu.z * 0.45 * h * bendW;
    }

    vec4 clipRoot = uViewProj * vec4(root, 1.0);
    if (clipRoot.w <= 1.0) { cull(); return; }
    vec2 ndcRoot = vec2(clipRoot.x / uNdcXScale, clipRoot.y * uFlipY) / clipRoot.w;
    vec2 guv = clamp(ndcRoot * 0.5 + 0.5, vec2(0.003), vec2(0.997));
    vGround = groundSample(guv);

    int variant = uFlower != 0 ? int(kindF) : (int(r3 * 4.0) & 3);
    vKind = uFlower;
    vec2 cell = vec2(float(variant & 1), float(variant >> 1));
    float flip = r5 > 0.5 ? 1.0 : 0.0;
    vUv = (cell + vec2(mix(cn.x, 1.0 - cn.x, flip), 1.0 - v)) * 0.5;
    vHeight = v;
    vTipTint = (r1 - 0.5) * 2.0;
    // Slow meadow-scale variation (lush / dry patches); a function of the world position only.
    vPatch = 0.5 + 0.25 * sin(root.x * 0.021 + 1.3 * sin(root.z * 0.017)) + 0.25 * sin(root.z * 0.013 - root.x * 0.009 + 0.7);
    // Sun behind the tuft as seen from the eye: light shines through the blades.
    vec3 viewDir = normalize(root - uEye);
    vTrans = pow(clamp(dot(viewDir, uSunDir), 0.0, 1.0), 3.0) * v;
    // Foliage light: wider lobe, and a sun-above term (thin blades glow when the sun is overhead too), tips most.
    vTrans += uTrans * (pow(clamp(dot(viewDir, uSunDir), 0.0, 1.0), 1.8) * 0.9 * v + 0.28 * clamp(uSunDir.y, 0.0, 1.0) * v * v);

    vec4 clip = uViewProj * vec4(pos, 1.0);
    clip.x /= uNdcXScale;
    clip.y *= uFlipY;
    gl_Position = clip;
}
)GLSL";

const char* kFragment = R"GLSL(#version 330 core
in vec2 vUv;
in vec3 vGround;
in float vHeight, vTrans, vTipTint, vPatch;
flat in int vKind;
uniform sampler2D uAtlas;
uniform int uAlphaToCoverage;
uniform float uLodBias;
uniform vec3 uSunColor;
out vec4 oColor;
void main() {
    vec4 t = texture(uAtlas, vUv, uLodBias);
    float a = t.a;
    if (uAlphaToCoverage == 0) {
        if (a < 0.45) discard;
        a = 1.0;
    } else {
        // Sharpen the ramp so the coverage mask keeps crisp blade edges (as the game's own cut-out draws do).
        a = clamp((a - 0.42) / max(fwidth(a), 0.02) + 0.5, 0.0, 1.0);
        if (a <= 0.0) discard;
    }
    vec3 g = vGround;
    float l = dot(g, vec3(0.2126, 0.7152, 0.0722));
    if (vKind != 0) { // ground cover keeps its own colour; only the light of the ground reaches it
        vec3 lit = mix(vec3(l), g, 0.30) * 2.4 * mix(0.55, 1.0, vHeight);
        vec3 rgb = t.rgb * lit;
        rgb += t.rgb * g * uSunColor * (1.6 * vTrans);
        oColor = vec4(rgb, a);
        return;
    }
    g = mix(vec3(l), g, 1.25);                                   // richer than the flat ground texture
    g *= mix(vec3(1.00, 0.97, 0.80), vec3(0.84, 1.06, 0.84), vPatch); // dry straw <-> lush green meadow patches
    g *= vec3(1.0 + 0.06 * vTipTint, 1.0 + 0.04 * vTipTint * vHeight, 1.0 - 0.10 * vTipTint); // hue drift between tufts
    vec3 rgb = g * t.rgb * mix(0.95, 1.25, vPatch);
    rgb += g * uSunColor * (0.9 * vTrans * t.r);                 // translucent glow when looking towards the sun
    oColor = vec4(rgb, a);
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
        SPDLOG_ERROR("NewDawnGrass shader: {}", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

float CvFloat(const char* name, float def) {
    return Ship::Context::GetRawInstance()->GetConsoleVariables()->GetFloat(name, def);
}

// Where actors recently walked: samples of (x, z, birth time, radius), aged by wall clock so the blades spring back over
// about a second whatever the frame rate. Read-only copies of positions the game side hands over.
struct PushSample {
    float x, z, t, r;
};
struct PushLast {
    uint32_t key;
    float x, z;
    bool used;
};
constexpr int kMaxPush = 36, kLive = NEWDAWN_GRASS_PUSHERS, kTrail = kMaxPush - kLive;
constexpr float kPushLife = 1.25f, kPushSpacing = 6.0f;
PushSample gTrail[kTrail];
int gTrailHead = 0;
PushLast gLast[kLive];

float Decode(float c) {
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}
} // namespace

bool PostFxGL::BuildGrassProgram() {
    if (mGrassAttempted)
        return mGrass.id != 0;
    mGrassAttempted = true;
    GLuint vs = Compile(GL_VERTEX_SHADER, kVertex), fs = Compile(GL_FRAGMENT_SHADER, kFragment);
    if (!vs || !fs) {
        glDeleteShader(vs);
        glDeleteShader(fs);
        return false;
    }
    mGrass.id = glCreateProgram();
    glAttachShader(mGrass.id, vs);
    glAttachShader(mGrass.id, fs);
    glLinkProgram(mGrass.id);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok;
    glGetProgramiv(mGrass.id, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(mGrass.id, sizeof(log), nullptr, log);
        SPDLOG_ERROR("NewDawnGrass link: {}", log);
        glDeleteProgram(mGrass.id);
        mGrass.id = 0;
        return false;
    }
    GLint count;
    glGetProgramiv(mGrass.id, GL_ACTIVE_UNIFORMS, &count);
    for (GLint i = 0; i < count; i++) {
        char name[256];
        GLsizei length;
        GLint size;
        GLenum type;
        glGetActiveUniform(mGrass.id, i, sizeof(name), &length, &size, &type, name);
        mGrass.uniforms[name] = glGetUniformLocation(mGrass.id, name);
        std::string base(name);
        if (base.size() > 3 && base.compare(base.size() - 3, 3, "[0]") == 0)
            mGrass.uniforms[base.substr(0, base.size() - 3)] = mGrass.uniforms[name]; // arrays are looked up by bare name
    }
    Use(mGrass);
    glUniform1i(Loc("uGround"), kGroundUnit);
    glUniform1i(Loc("uAtlas"), kAtlasUnit);

    // The tuft atlas is self-rendered (tools/newdawn/grass-tufts); without it the feature stays off.
    const std::string root = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetString(
        "gNewDawn.Assets.Path", "assets/newdawn");
    std::filesystem::path path = std::filesystem::path(root) / "grass" / "tuft_atlas.png";
    if (!path.is_absolute())
        path = Ship::Context::LocateFileAcrossAppDirs(path.string());
    int w = 0, h = 0, n = 0;
    auto* data = stbi_load(path.string().c_str(), &w, &h, &n, 4);
    if (!data || w != h || w < 256 || w > 4096) { // square 2x2 atlas, any resolution
        SPDLOG_WARN("NewDawnGrass: missing/unsupported tuft atlas {}; set gNewDawn.Assets.Path to the asset root",
                    path.string());
        stbi_image_free(data);
        glDeleteProgram(mGrass.id);
        mGrass.id = 0;
        return false;
    }
    glActiveTexture(GL_TEXTURE0 + kAtlasUnit);
    glGenTextures(1, &mGrassAtlas);
    glBindTexture(GL_TEXTURE_2D, mGrassAtlas);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    GLfloat maxAniso = 1.0f;
    glGetFloatv(0x84FF /* GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT */, &maxAniso);
    glTexParameterf(GL_TEXTURE_2D, 0x84FE /* GL_TEXTURE_MAX_ANISOTROPY_EXT */, std::min(maxAniso, 16.0f));
    glGenerateMipmap(GL_TEXTURE_2D);

    // Optional ground-cover atlas (flowers, clover); without it only the flowers stay off.
    {
        std::filesystem::path fpath = std::filesystem::path(root) / "grass" / "flower_atlas.png";
        if (!fpath.is_absolute())
            fpath = Ship::Context::LocateFileAcrossAppDirs(fpath.string());
        int fw = 0, fh = 0, fn = 0;
        auto* fdata = stbi_load(fpath.string().c_str(), &fw, &fh, &fn, 4);
        if (fdata && fw == fh && fw >= 128 && fw <= 4096) {
            glGenTextures(1, &mFlowerAtlas);
            glBindTexture(GL_TEXTURE_2D, mFlowerAtlas);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, fw, fh, 0, GL_RGBA, GL_UNSIGNED_BYTE, fdata);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameterf(GL_TEXTURE_2D, 0x84FE /* GL_TEXTURE_MAX_ANISOTROPY_EXT */, std::min(maxAniso, 16.0f));
            glGenerateMipmap(GL_TEXTURE_2D);
        } else {
            SPDLOG_WARN("NewDawnGrass: no ground-cover atlas {}; flowers stay off", fpath.string());
        }
        stbi_image_free(fdata);
    }
    glGenVertexArrays(1, &mGrassVao);
    glGenBuffers(1, &mGrassVbo);
    SPDLOG_INFO("NewDawnGrass: loaded tuft atlas from {}", path.string());
    return true;
}

void PostFxGL::CaptureGrassGround(const FramebufferOGL& fb) {
    const int w = fb.width, h = fb.height;
    EnsureTarget(mGrassGround, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
    glDisable(GL_SCISSOR_TEST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mGrassGround.fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST); // resolves MSAA
    mGrassGroundReady = true;
}

void PostFxGL::RunGrass(const NewDawnPostFxParams& p, const FramebufferOGL& fb) {
    if (!mGrassData || mGrassCount == 0 || !BuildGrassProgram())
        return;
    const float density = std::clamp(p.grassDensity, 0.0f, 2.0f);
    const float distance = std::clamp(p.grassDistance, 200.0f, 4000.0f);
    if (density <= 0.0f)
        return;
    const float densityNear = std::min(density * 0.75f, 1.0f);
    const float* eye = p.giCameraEye;
    const float* M = p.shadowViewProj; // M[column * 4 + row]; clip = v * M
    const float xs = p.shadowNdcXScale > 1e-6f ? p.shadowNdcXScale : 1.0f;
    const auto densityAt = [&](float u) {
        const float t = std::clamp((u - 0.12f) / (0.95f - 0.12f), 0.0f, 1.0f);
        return densityNear * (1.0f - 0.88f * t * t * (3.0f - 2.0f * t));
    };

    for (auto& band : mGrassCells)
        band.clear();
    size_t cells = 0;
    const float radius = kCell * 0.7072f;
    for (uint32_t t = 0; t < mGrassCount && cells < kMaxCells; ++t) {
        const float* tri = mGrassData + (size_t)t * NEWDAWN_GRASS_TRI_FLOATS;
        const float minX = std::min({ tri[0], tri[3], tri[6] }), maxX = std::max({ tri[0], tri[3], tri[6] });
        const float minZ = std::min({ tri[2], tri[5], tri[8] }), maxZ = std::max({ tri[2], tri[5], tri[8] });
        const float minY = std::min({ tri[1], tri[4], tri[7] }), maxY = std::max({ tri[1], tri[4], tri[7] });
        if (maxX < eye[0] - distance || minX > eye[0] + distance || maxZ < eye[2] - distance ||
            minZ > eye[2] + distance || minY - eye[1] > distance || eye[1] - maxY > distance)
            continue;
        const float area2 = (tri[3] - tri[0]) * (tri[8] - tri[2]) - (tri[5] - tri[2]) * (tri[6] - tri[0]);
        if (std::fabs(area2) < 1e-3f)
            continue;
        const float sign = area2 > 0.0f ? 1.0f : -1.0f;
        float edgeNx[3], edgeNz[3], edgeC[3]; // inside-positive distance = nx * x + nz * z + c
        for (int i = 0; i < 3; ++i) {
            const float* a = tri + i * 3;
            const float* b = tri + ((i + 1) % 3) * 3;
            const float ex = b[0] - a[0], ez = b[2] - a[2];
            const float len = std::sqrt(ex * ex + ez * ez) + 1e-6f;
            edgeNx[i] = -ez / len * sign;
            edgeNz[i] = ex / len * sign;
            edgeC[i] = -(edgeNx[i] * a[0] + edgeNz[i] * a[2]);
        }
        const int cx0 = (int)std::floor(std::max(minX, eye[0] - distance) / kCell);
        const int cx1 = (int)std::floor(std::min(maxX, eye[0] + distance) / kCell);
        const int cz0 = (int)std::floor(std::max(minZ, eye[2] - distance) / kCell);
        const int cz1 = (int)std::floor(std::min(maxZ, eye[2] + distance) / kCell);
        for (int cz = cz0; cz <= cz1 && cells < kMaxCells; ++cz) {
            for (int cx = cx0; cx <= cx1; ++cx) {
                const float qx = (cx + 0.5f) * kCell, qz = (cz + 0.5f) * kCell;
                if (edgeNx[0] * qx + edgeNz[0] * qz + edgeC[0] < -radius ||
                    edgeNx[1] * qx + edgeNz[1] * qz + edgeC[1] < -radius ||
                    edgeNx[2] * qx + edgeNz[2] * qz + edgeC[2] < -radius)
                    continue;
                // Height of the triangle plane under the cell centre (clamped to the vertex range: edge cells).
                float qy = (minY + maxY) * 0.5f;
                {
                    const float v0x = tri[3] - tri[0], v0z = tri[5] - tri[2], v1x = tri[6] - tri[0], v1z = tri[8] - tri[2];
                    const float v2x = qx - tri[0], v2z = qz - tri[2];
                    const float den = v0x * v1z - v1x * v0z;
                    const float b1 = (v2x * v1z - v1x * v2z) / den, b2 = (v0x * v2z - v2x * v0z) / den;
                    qy = std::clamp(tri[1] + b1 * (tri[4] - tri[1]) + b2 * (tri[7] - tri[1]), minY, maxY);
                }
                const float dx = qx - eye[0], dy = qy - eye[1], dz = qz - eye[2];
                const float dEdge = std::sqrt(dx * dx + dy * dy + dz * dz) - kCell * 0.8f;
                if (dEdge > distance)
                    continue;
                // Frustum cull on the cell's bounding sphere (tuft height and sway included).
                float clip[4];
                for (int r = 0; r < 4; ++r)
                    clip[r] = M[0 * 4 + r] * qx + M[1 * 4 + r] * qy + M[2 * 4 + r] * qz + M[3 * 4 + r];
                const float reach = kCell * 0.75f + 40.0f;
                const float rowX = std::sqrt(M[0] * M[0] + M[4] * M[4] + M[8] * M[8]);
                const float rowY = std::sqrt(M[1] * M[1] + M[5] * M[5] + M[9] * M[9]);
                const float rowW = std::sqrt(M[3] * M[3] + M[7] * M[7] + M[11] * M[11]);
                if (clip[3] < -reach * rowW || std::fabs(clip[0] / xs) > clip[3] + reach * (rowX / xs + rowW) ||
                    std::fabs(clip[1]) > clip[3] + reach * (rowY + rowW))
                    continue;
                const float f = densityAt(std::max(dEdge, 0.0f) / distance);
                const int needed = (int)std::ceil(f * kSlots - 1e-4f);
                if (needed <= 0)
                    continue;
                int band = 0;
                while (band < 3 && needed <= kBands[band + 1])
                    ++band;
                auto& out = mGrassCells[band];
                const float rec[kRecordFloats] = { cx * kCell, cz * kCell, tri[0], tri[1], tri[2], tri[3],
                                                   tri[4],      tri[5],      tri[6], tri[7], tri[8], 0.0f };
                out.insert(out.end(), rec, rec + kRecordFloats);
                ++cells;
            }
        }
    }
    if (cells == 0)
        return;
    if (Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Grass.Debug", 0)) {
        static int frame = 0;
        if (++frame % 300 == 1) {
            size_t instances = 0;
            for (int b = 0; b < 4; ++b)
                instances += mGrassCells[b].size() / kRecordFloats * kBands[b];
            SPDLOG_INFO("GRASSDBG cells {}/{}/{}/{} instances {} ({} vertices)", mGrassCells[0].size() / kRecordFloats,
                        mGrassCells[1].size() / kRecordFloats, mGrassCells[2].size() / kRecordFloats,
                        mGrassCells[3].size() / kRecordFloats, instances, instances * kTuftVertices);
        }
    }

    // ---- GPU
    const int w = fb.width, h = fb.height;
    if (!mGrassGroundReady)
        CaptureGrassGround(fb); // no ground-end marker this frame: fall back to the finished scene
    mGrassGroundReady = false;

    Use(mGrass);
    glActiveTexture(GL_TEXTURE0 + kGroundUnit);
    glBindTexture(GL_TEXTURE_2D, mGrassGround.tex);
    glBindSampler(kGroundUnit, 0);
    glActiveTexture(GL_TEXTURE0 + kAtlasUnit);
    glBindTexture(GL_TEXTURE_2D, mGrassAtlas);
    glBindSampler(kAtlasUnit, 0);

    glUniformMatrix4fv(Loc("uViewProj"), 1, GL_FALSE, p.shadowViewProj);
    glUniform1f(Loc("uNdcXScale"), p.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), p.shadowFlipY);
    glUniform3fv(Loc("uEye"), 1, eye);
    glUniform3fv(Loc("uFocus"), 1, p.grassFocus);
    glUniform3fv(Loc("uSunDir"), 1, p.grassSunDir);
    float sun[3];
    for (int i = 0; i < 3; ++i)
        sun[i] = Decode(p.grassSunColor[i]) * std::clamp(CvFloat("gNewDawn.HDR.SunIntensity", 1.8f), 0.0f, 16.0f) * 0.5f;
    glUniform3fv(Loc("uSunColor"), 1, sun);
    glUniform1f(Loc("uCell"), kCell);
    glUniform1f(Loc("uDistance"), distance);
    glUniform1f(Loc("uDensityNear"), densityNear);
    glUniform1f(Loc("uHeight"), std::clamp(p.grassHeight, 0.2f, 4.0f));
    glUniform1f(Loc("uWindScale"), std::clamp(CvFloat("gNewDawn.Atmos.WindScale", 1.0f), 0.0f, 10.0f));
    glUniform2f(Loc("uInvSize"), 1.0f / w, 1.0f / h);
    const double t = NewDawnWindSeconds();
    const NewDawnWindState wind = NewDawnWindAt(t);
    glUniform3f(Loc("uWind"), wind.dirX, wind.dirZ, wind.gust);
    constexpr double kTwoPi = 6.283185307179586;
    const double travel = NewDawnWindTravel(t);
    glUniform3f(Loc("uPhase"), (float)std::fmod(travel * 1.7, kTwoPi), (float)std::fmod(travel * 1.3, kTwoPi),
                (float)std::fmod(t * 6.0, kTwoPi));
    // Pushers: the live positions plus the recent trail (age in seconds), nothing when the feature is off.
    float push[kMaxPush * 4] = {};
    int pushCount = 0;
    if (p.grassBend) {
        const float now = (float)t;
        for (int i = 0; i < p.grassPushCount && i < kLive; ++i) {
            const float* a = p.grassPush[i];
            PushLast* last = nullptr;
            for (auto& l : gLast)
                if (l.used && l.key == p.grassPushKey[i])
                    last = &l;
            if (!last) {
                for (auto& l : gLast)
                    if (!l.used)
                        last = &l;
                if (!last)
                    last = &gLast[i % kLive];
                *last = { p.grassPushKey[i], a[0], a[2], true };
            }
            const float dx = a[0] - last->x, dz = a[2] - last->z;
            if (dx * dx + dz * dz >= kPushSpacing * kPushSpacing) {
                if (dx * dx + dz * dz < 300.0f * 300.0f) // a teleport leaves no trail
                    gTrail[gTrailHead++ % kTrail] = { last->x, last->z, now, a[3] * 0.8f };
                last->x = a[0];
                last->z = a[2];
            }
            push[pushCount * 4 + 0] = a[0];
            push[pushCount * 4 + 1] = a[2];
            push[pushCount * 4 + 2] = 0.0f;
            push[pushCount * 4 + 3] = a[3];
            ++pushCount;
        }
        for (auto& l : gLast) { // forget actors that are gone
            bool seen = false;
            for (int i = 0; i < p.grassPushCount && i < kLive; ++i)
                seen |= l.key == p.grassPushKey[i];
            if (!seen)
                l.used = false;
        }
        for (const auto& s : gTrail) {
            const float age = now - s.t;
            if (s.r > 0.0f && age >= 0.0f && age < kPushLife && pushCount < kMaxPush) {
                push[pushCount * 4 + 0] = s.x;
                push[pushCount * 4 + 1] = s.z;
                push[pushCount * 4 + 2] = age;
                push[pushCount * 4 + 3] = s.r;
                ++pushCount;
            }
        }
    } else {
        for (auto& s : gTrail)
            s.r = 0.0f;
        for (auto& l : gLast)
            l.used = false;
    }
    glUniform1i(Loc("uPushCount"), pushCount);
    if (pushCount > 0)
        glUniform4fv(Loc("uPush"), pushCount, push);
    glUniform1f(Loc("uTrans"), p.foliageLight ? std::clamp(p.foliageStrength, 0.0f, 3.0f) : 0.0f);
    glUniform4fv(Loc("uFlowerMix"), 1, p.grassFlowerMix);
    glUniform1f(Loc("uFlowerDensity"), p.grassFlowerDensity);
    const bool a2c = fb.msaa_level > 1;
    glUniform1i(Loc("uAlphaToCoverage"), a2c);
    // Alpha-to-coverage hides the shimmer of a sharper mip, so the MSAA path samples finer levels.
    glUniform1f(Loc("uLodBias"), CvFloat("gNewDawn.Grass.LodBias", a2c ? -1.0f : -0.5f));

    glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
    glViewport(0, 0, w, h);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);
    if (a2c)
        glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
    // Tuft fragments write stencil 2 (characters write 1) so TAA can treat swaying blades as their own class. Only
    // when this frame's stencil was cleared at the start of the world (the game never uses it otherwise).
    if (mLocalMarking) {
        glEnable(GL_STENCIL_TEST);
        glStencilMask(0xFF);
        glStencilFunc(GL_ALWAYS, 2, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    }

    glBindVertexArray(mGrassVao);
    glBindBuffer(GL_ARRAY_BUFFER, mGrassVbo);
    size_t total = 0;
    for (const auto& band : mGrassCells)
        total += band.size();
    glBufferData(GL_ARRAY_BUFFER, total * sizeof(float), nullptr, GL_STREAM_DRAW);
    const bool flowers = p.grassFlowers && p.grassFlowerDensity > 0.01f && mFlowerAtlas != 0;
    for (int pass = 0; pass < (flowers ? 2 : 1); ++pass) {
        // Pass 1: ground cover on the nearer cells (a single height segment is enough for small flowers).
        glUniform1i(Loc("uFlower"), pass);
        glUniform1i(Loc("uSegs"), pass ? 1 : 2);
        if (pass) {
            glActiveTexture(GL_TEXTURE0 + kAtlasUnit);
            glBindTexture(GL_TEXTURE_2D, mFlowerAtlas);
        }
        size_t offset = 0;
        for (int band = 0; band < 4; ++band) {
            const auto& records = mGrassCells[band];
            if (records.empty())
                continue;
            if (pass == 0)
                glBufferSubData(GL_ARRAY_BUFFER, offset * sizeof(float), records.size() * sizeof(float), records.data());
            if (pass == 0 || band < 2) {
                for (int i = 0; i < 3; ++i) {
                    glEnableVertexAttribArray(i);
                    glVertexAttribPointer(i, 4, GL_FLOAT, GL_FALSE, kRecordFloats * sizeof(float),
                                          (const void*)((offset + i * 4) * sizeof(float)));
                    glVertexAttribDivisor(i, kBands[band]);
                }
                glUniform1i(Loc("uSlotCount"), kBands[band]);
                const GLsizei cellCount = (GLsizei)(records.size() / kRecordFloats);
                glDrawArraysInstanced(GL_TRIANGLES, 0, pass ? 12 : kTuftVertices, cellCount * kBands[band]);
            }
            offset += records.size();
        }
    }
    if (flowers) {
        glActiveTexture(GL_TEXTURE0 + kAtlasUnit);
        glBindTexture(GL_TEXTURE_2D, mGrassAtlas);
    }
    for (int i = 0; i < 3; ++i)
        glVertexAttribDivisor(i, 0);
    glBindVertexArray(mVao);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    if (mLocalMarking) {
        glDisable(GL_STENCIL_TEST);
        glStencilFunc(GL_ALWAYS, 0, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    }
    CheckError("grass");
}
} // namespace Fast
#endif
