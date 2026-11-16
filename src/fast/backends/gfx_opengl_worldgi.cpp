#if defined(ENABLE_OPENGL) && !defined(USE_OPENGLES)
// New Dawn world-space global illumination (docs/WORLD_GI.md).
//
// The renderer only ever sees display lists, so it builds its own scene: the interpreter records the opaque room
// triangles the game submits (world position, an albedo estimated from the combiner and the average texture colour, and
// the face normal) and they are rasterised into a voxel grid that covers the scene's collision bounds. The grid is
// persistent: a voxel is written once and kept until the scene changes, so geometry the game culls behind the camera
// stays once it has been seen. Actors are not in it.
//
// Lighting, all view independent:
//   1. radiance volume: a few slices per frame, every occupied voxel is lit by the sun or moon (shadow ray through the
//      voxels), the local lights (shadow ray), the occluded sky and the probes themselves (multi-bounce feedback);
//   2. probes: a grid of irradiance + distance probes (octahedral 6x6 tiles); a budgeted block of probes per frame traces
//      rays through the voxels, reads the radiance volume at the hit and blends into the atlases with hysteresis;
//   3. apply: per pixel, from depth and the depth-derived normal, 8 probes weighted by trilinear, wrap and Chebyshev
//      visibility; added to the HDR scene (dual-source blend, MSAA samples intact).
// Nothing here reads or writes game state; with gNewDawn.GI.World=0 none of it runs.
#include "fast/backends/gfx_opengl_postfx.h"
#include "fast/newdawn_wind.h"
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace Fast {
namespace {
// Dedicated lanes; a complete dummy is bound to each when the programs are built and the real textures replace it.
constexpr int kVoxAlbedoUnit = 34, kVoxNormalUnit = 35, kVoxRadianceUnit = 36, kVoxCoarseUnit = 37;
constexpr int kProbeIrrUnit = 38, kProbeDistUnit = 39, kRayUnit = 40;
constexpr int kTile = 8;                    // probe tile: 6x6 octahedral texels + a 1 texel border
constexpr int kProbeDim[3] = { 32, 16, 32 }; // probe window (powers of two: addressing wraps)
constexpr int kAtlasCols = 256, kAtlasRows = 64; // tile = (x + 32 * (z & 7), y + 16 * (z >> 3))
constexpr int kMaxLights = 24;
constexpr int kWarmFrames = 40; // frames after a scene change that run with a 4x budget
constexpr int kWarmBoost = 4;
constexpr int kSnap = 16; // the window moves in steps of this many voxels (a multiple of the probe spacing)

struct Tier {
    int dim[3];     // voxel window (powers of two)
    float maxVoxel; // largest voxel, world units: scenes bigger than dim * maxVoxel get a window that follows the camera
    int rays;       // rays per probe update
    int rows;       // atlas rows (256 probes each) updated per frame
    int slices;     // radiance volume slices relit per frame
    int steps;      // voxel walk iteration cap
};
const Tier kTiers[2] = {
    { { 128, 64, 128 }, 40.0f, 64, 2, 8, 160 },    // High
    { { 256, 128, 256 }, 24.0f, 128, 4, 16, 256 }, // Ultra
};

int CvI(const char* name, int def) {
    return Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger(name, def);
}
float CvF(const char* name, float def) {
    return Ship::Context::GetRawInstance()->GetConsoleVariables()->GetFloat(name, def);
}

const char* kFullscreenVs = R"GLSL(#version 440 core
out vec2 vUv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

const char* kVoxelizeVs = R"GLSL(#version 440 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aAlbedo;
layout(location=2) in vec3 aNormal;
layout(location=3) in float aBaked;
uniform ivec3 uVoxOrigin;
uniform float uVoxSize;
uniform ivec3 uVoxDim;
uniform int uAxis;
out vec3 vVox;
flat out vec3 vAlbedo;
flat out vec4 vNormal;
void main() {
    vVox = aPos / uVoxSize - vec3(uVoxOrigin);
    vec3 n = vVox / vec3(uVoxDim) * 2.0 - 1.0;
    gl_Position = vec4(uAxis == 0 ? n.yz : (uAxis == 1 ? n.xz : n.xy), 0.0, 1.0);
    vAlbedo = aAlbedo;
    vNormal = vec4(aNormal * 0.5 + 0.5, aBaked);
}
)GLSL";

// Rasterised once along each axis at twice the grid resolution: every fragment lies on the triangle, so only voxels
// the surface really crosses are written. First writer wins and nothing is ever overwritten, so the grid is stable
// from frame to frame whatever subset of the room the game submits.
const char* kVoxelizeFs = R"GLSL(#version 440 core
layout(binding=0, rgba8) uniform image3D uImgAlbedo;
layout(binding=1, rgba8) uniform image3D uImgNormal;
layout(binding=2, r8) uniform image3D uImgCoarse;
uniform ivec3 uVoxOrigin;
uniform ivec3 uVoxDim;
in vec3 vVox;
flat in vec3 vAlbedo;
flat in vec4 vNormal;
void main() {
    ivec3 v = ivec3(floor(vVox));
    if (any(lessThan(v, ivec3(0))) || any(greaterThanEqual(v, uVoxDim))) return;
    v = (v + uVoxOrigin) & (uVoxDim - 1);
    if (imageLoad(uImgAlbedo, v).a > 0.0) return;
    imageStore(uImgAlbedo, v, vec4(vAlbedo, 1.0));
    imageStore(uImgNormal, v, vNormal);
    imageStore(uImgCoarse, v >> 2, vec4(1.0));
}
)GLSL";

// Shared by the inject, trace, blend and apply programs. Both grids are windows into the world that wrap: the cell
// with world index c lives at texel c & (size - 1), so moving the window never moves any data.
const char* kLib = R"GLSL(
uniform sampler3D uVoxAlbedo;   // rgb display-space reflectance, a occupancy
uniform sampler3D uVoxNormal;   // rgb face normal * 0.5 + 0.5, a 1 = baked (unlit, vertex-coloured) surface
uniform sampler3D uVoxRadiance; // outgoing radiance of the lit voxel
uniform sampler3D uVoxCoarse;   // occupancy of each 4^3 block
uniform sampler2D uProbeIrr;    // rgb bounced irradiance / pi, a sky visibility
uniform sampler2D uProbeDist;   // r mean hit distance, g mean squared distance
uniform ivec3 uVoxOrigin;   // window corner, world voxel index
uniform float uVoxSize;
uniform ivec3 uProbeOrigin; // window corner, world probe index (probe i sits at (i + 0.5) * uProbeStep)
uniform float uProbeStep;
uniform float uProbeBias;
uniform int uSteps;
const ivec3 WGI_PROBES = ivec3(32, 16, 32);
const float WGI_TILE = 8.0;

vec2 wgiSign(vec2 v) { return vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0); }
vec2 wgiOctEncode(vec3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    return n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * wgiSign(n.xy);
}
vec3 wgiOctDecode(vec2 e) {
    vec3 v = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) v.xy = (1.0 - abs(v.yx)) * wgiSign(v.xy);
    return normalize(v);
}
vec3 wgiFibonacci(int i, int n) {
    float phi = 6.2831853 * fract(float(i) * 0.6180339887);
    float c = 1.0 - (2.0 * float(i) + 1.0) / float(n);
    float s = sqrt(max(0.0, 1.0 - c * c));
    return vec3(cos(phi) * s, sin(phi) * s, c);
}
// Atlas tile of a probe (any world index) and the inverse for a tile.
ivec2 wgiProbeTile(ivec3 c) {
    c &= WGI_PROBES - 1;
    return ivec2(c.x + 32 * (c.z & 7), c.y + 16 * (c.z >> 3));
}
ivec3 wgiTileProbe(ivec2 tile) {
    ivec3 w = ivec3(tile.x & 31, tile.y & 15, (tile.x >> 5) + 8 * (tile.y >> 4));
    return uProbeOrigin + ((w - uProbeOrigin) & (WGI_PROBES - 1));
}
vec2 wgiProbeUv(ivec2 tile, vec3 dir, vec2 atlas) {
    return (vec2(tile) * WGI_TILE + 1.0 + (wgiOctEncode(dir) * 0.5 + 0.5) * (WGI_TILE - 2.0)) / atlas;
}

// Voxel walk from o (voxel units from the window corner) along the unit vector d for at most tMax voxels. The start
// cell never counts (callers start on or beside a surface). Empty 4^3 blocks are crossed in one step. hit is the
// texel of the voxel that stopped the ray.
bool wgiTrace(vec3 o, vec3 d, float tMax, out ivec3 hit, out float tHit) {
    ivec3 dim = textureSize(uVoxAlbedo, 0);
    vec3 dd = mix(d, vec3(1e-5), lessThan(abs(d), vec3(1e-5)));
    vec3 inv = 1.0 / dd, ahead = step(vec3(0.0), dd), sgn = sign(dd), dimf = vec3(dim);
    ivec3 start = ivec3(floor(o));
    vec3 nudge = vec3(0.0);
    float t = 0.0;
    hit = ivec3(0);
    tHit = 0.0;
    for (int i = 0; i < uSteps; ++i) {
        vec3 p = o + dd * t + nudge;
        if (t > tMax || any(lessThan(p, vec3(0.0))) || any(greaterThanEqual(p, dimf))) return false;
        ivec3 c = ivec3(p);
        ivec3 w = (c + uVoxOrigin) & (dim - 1);
        vec3 tt;
        if (texelFetch(uVoxCoarse, w >> 2, 0).r == 0.0) {
            tt = ((vec3(c >> 2) + ahead) * 4.0 - o) * inv;
        } else {
            if (c != start && texelFetch(uVoxAlbedo, w, 0).a > 0.0) {
                hit = w;
                tHit = t;
                return true;
            }
            tt = (vec3(c) + ahead - o) * inv;
        }
        t = min(tt.x, min(tt.y, tt.z));
        nudge = sgn * vec3(lessThanEqual(tt, vec3(t))) * 2e-3;
    }
    return false;
}

// Probe lookup for a surface point P with normal N: rgb bounced irradiance / pi, a sky visibility. Probes behind the
// surface, and probes whose stored depth says something stands between them and P, are weighted out.
vec4 wgiIrradiance(vec3 P, vec3 N) {
    vec3 Pb = P + N * uProbeBias;
    vec3 g = Pb / uProbeStep - 0.5;
    ivec3 base = ivec3(floor(g));
    vec3 f = g - vec3(base);
    vec2 irrSize = vec2(textureSize(uProbeIrr, 0)), distSize = vec2(textureSize(uProbeDist, 0));
    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    for (int i = 0; i < 8; ++i) {
        ivec3 o = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        ivec3 c = base + o, rel = c - uProbeOrigin;
        if (any(lessThan(rel, ivec3(0))) || any(greaterThanEqual(rel, WGI_PROBES))) continue;
        ivec2 tile = wgiProbeTile(c);
        vec3 probe = (vec3(c) + 0.5) * uProbeStep;
        vec3 toProbe = normalize(probe - P);
        float wrap = (dot(toProbe, N) + 1.0) * 0.5;
        float w = wrap * wrap + 0.2;
        vec3 fromProbe = Pb - probe;
        float dist = length(fromProbe);
        vec2 m = textureLod(uProbeDist, wgiProbeUv(tile, fromProbe / max(dist, 1e-3), distSize), 0.0).rg;
        if (dist > m.x) {
            float variance = abs(m.x * m.x - m.y), dd = dist - m.x;
            float cheb = variance / (variance + dd * dd);
            w *= max(cheb * cheb * cheb, 0.0);
        }
        w = max(w, 1e-6);
        if (w < 0.2) w *= w * w / 0.04;
        vec3 tri = mix(1.0 - f, f, vec3(o));
        w *= tri.x * tri.y * tri.z + 1e-4;
        sum += textureLod(uProbeIrr, wgiProbeUv(tile, N, irrSize), 0.0) * w;
        wsum += w;
    }
    return sum / max(wsum, 1e-8);
}
)GLSL";

// One slice of the radiance volume: direct light on every occupied voxel plus the light the probes say arrives there.
// Lit geometry (outdoors) takes the sun or moon, the occluded sky, the local lights and the bounce. Baked geometry
// (interiors and dungeons are unlit and vertex-coloured: the game never lights them) emits what the game displays, its
// painted colour, plus the local lights and the bounce it receives.
const char* kInjectFs = R"GLSL(
out vec4 oRadiance;
uniform int uLayer;
uniform vec3 uSunDir;
uniform vec3 uSunRgb;
uniform vec3 uSkyRgb;
uniform int uSkyCube;
uniform samplerCube uSkyIrradiance;
uniform int uLightCount;
uniform vec4 uLight[24];
uniform vec3 uLightRgb[24];
uniform float uBounce;
uniform float uBaked;
uniform vec2 uGain; // how much of the sun/sky (x) and of the local lights (y) is sent on as bounce
void main() {
    ivec3 dim = textureSize(uVoxAlbedo, 0);
    ivec3 w = ivec3(ivec2(gl_FragCoord.xy), uLayer);
    vec4 a = texelFetch(uVoxAlbedo, w, 0);
    oRadiance = vec4(0.0);
    if (a.a == 0.0) return;
    vec4 nb = texelFetch(uVoxNormal, w, 0);
    vec3 N = normalize(nb.xyz * 2.0 - 1.0);
    vec3 vc = vec3((w - uVoxOrigin) & (dim - 1)) + 0.5; // voxel units from the window corner
    vec3 P = (vec3(uVoxOrigin) + vc) * uVoxSize;
    vec3 from = vc + N * 1.5;
    ivec3 hit;
    float t;
    vec3 local = vec3(0.0);
    for (int i = 0; i < uLightCount; ++i) {
        vec3 V = uLight[i].xyz - P;
        float R = uLight[i].w, d2 = dot(V, V);
        if (d2 >= R * R) continue;
        float dist = sqrt(d2);
        vec3 L = V / max(dist, 0.001);
        // Same falloff as the screen-space local-light pass.
        float x = dist / R, window = clamp(1.0 - x * x * x * x, 0.0, 1.0), core = R * 0.2;
        float lit = window * window * core * core / (d2 + core * core) * clamp((dot(N, L) + 0.3) / 1.3, 0.0, 1.0);
        if (lit < 0.002) continue;
        if (wgiTrace(from, L, dist / uVoxSize - 2.5, hit, t)) continue;
        local += uLightRgb[i] * lit;
    }
    vec4 probes = wgiIrradiance(P + N * uVoxSize, N);
    vec3 E = 2.4 * local / (2.4 + local) * uGain.y + probes.rgb * uBounce;
    if (nb.a > 0.5) {
        oRadiance = vec4(pow(a.rgb, vec3(2.2)) * uBaked + a.rgb * E, 1.0);
        return;
    }
    float nl = dot(N, uSunDir);
    vec3 day = vec3(0.0);
    if (nl > 0.0 && uSunRgb.r + uSunRgb.g + uSunRgb.b > 0.0 && !wgiTrace(from, uSunDir, 1e9, hit, t)) day = uSunRgb * nl;
    vec3 sky = uSkyRgb;
    if (uSkyCube != 0) sky = texture(uSkyIrradiance, N).rgb * dot(uSkyRgb, vec3(0.2126, 0.7152, 0.0722));
    oRadiance = vec4(a.rgb * (E + (day + sky * probes.a) * uGain.x), 1.0);
}
)GLSL";

// One texel per ray: x = ray, y = probe (tile index) - uBase. rgb radiance at the hit, a = distance (negative: a back
// face, 10000: nothing hit, the ray reached the sky or left the window).
const char* kTraceFs = R"GLSL(
out vec4 oRay;
uniform int uBase;
uniform int uRayCount;
uniform mat3 uRot;
void main() {
    int index = uBase + int(gl_FragCoord.y);
    oRay = vec4(0.0, 0.0, 0.0, 10000.0);
    vec3 origin = (vec3(wgiTileProbe(ivec2(index & 255, index >> 8))) + 0.5) * uProbeStep;
    vec3 dir = uRot * wgiFibonacci(int(gl_FragCoord.x), uRayCount);
    ivec3 hit;
    float t;
    if (!wgiTrace(origin / uVoxSize - vec3(uVoxOrigin), dir, 1e9, hit, t)) return;
    vec3 n = texelFetch(uVoxNormal, hit, 0).xyz * 2.0 - 1.0;
    float dist = max(t * uVoxSize, 0.01);
    oRay = dot(n, dir) > 0.25 ? vec4(0.0, 0.0, 0.0, -dist) : vec4(texelFetch(uVoxRadiance, hit, 0).rgb, dist);
}
)GLSL";

// Folds this frame's rays into the irradiance and distance tiles of the updated probes. Border texels compute the
// value of the interior texel they mirror, so bilinear lookups never cross into a neighbouring probe.
const char* kBlendFs = R"GLSL(
layout(location=0) out vec4 oIrr;
layout(location=1) out vec4 oDist;
uniform sampler2D uRays;
uniform int uBase;
uniform int uRayCount;
uniform mat3 uRot;
uniform float uMaxDist;
void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    ivec2 tile = px / 8;
    int row = tile.y * 256 + tile.x - uBase;
    vec2 e = (vec2(px - tile * 8) - 0.5) / 6.0 * 2.0 - 1.0;
    if (abs(e.x) > 1.0) e = vec2(sign(e.x) * (2.0 - abs(e.x)), -e.y);
    if (abs(e.y) > 1.0) e = vec2(-e.x, sign(e.y) * (2.0 - abs(e.y)));
    vec3 dir = wgiOctDecode(e);
    vec4 irr = vec4(0.0);
    vec2 moments = vec2(0.0);
    float wi = 0.0, wd = 0.0;
    for (int r = 0; r < uRayCount; ++r) {
        vec4 ray = texelFetch(uRays, ivec2(r, row), 0);
        float c = dot(dir, uRot * wgiFibonacci(r, uRayCount));
        if (c <= 0.0) continue;
        bool miss = ray.a >= 9000.0;
        irr += vec4(ray.rgb, miss ? 1.0 : 0.0) * c;
        wi += c;
        // A back face means the probe sits inside or behind geometry: a short distance keeps it from lighting anything.
        float d = miss ? uMaxDist : min(abs(ray.a) * (ray.a < 0.0 ? 0.2 : 1.0), uMaxDist);
        float lobe = pow(c, 12.0);
        moments += vec2(d, d * d) * lobe;
        wd += lobe;
    }
    if (wd < 1e-3) discard;
    oIrr = irr / max(wi, 1e-4);
    oDist = vec4(moments / wd, 0.0, 1.0);
}
)GLSL";

const char* kApplyFs = R"GLSL(
in vec2 vUv;
// Dual-source blend (GL_ONE, GL_SRC1_COLOR): result = oColor + framebuffer * oFactor, per MSAA sample.
layout(location = 0, index = 0) out vec4 oColor;
layout(location = 0, index = 1) out vec4 oFactor;
uniform sampler2D uScene;
uniform sampler2D uDepth;
uniform ivec2 uSize;
uniform mat4 uInvViewProj;
uniform float uNdcXScale;
uniform float uFlipY;
uniform float uStrength;
uniform float uOcclusion;
uniform float uReceive;
uniform float uDay; // 0 no sun (night, interiors) .. 1 full daylight
uniform float uFogMul;
uniform float uFogOff;
uniform int uDebug;
uniform int uGrass;
const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);
float fetchDepth(ivec2 p) { return texelFetch(uDepth, clamp(p, ivec2(0), uSize - 1), 0).r; }
vec3 position(vec2 uv, float d) {
    vec4 p = uInvViewProj * vec4((uv * 2.0 - 1.0) * vec2(uNdcXScale, uFlipY), d * 2.0 - 1.0, 1.0);
    return p.xyz / p.w;
}
vec3 positionAt(ivec2 p) { return position((vec2(p) + 0.5) / vec2(uSize), fetchDepth(p)); }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    oColor = vec4(0.0);
    oFactor = vec4(1.0);
    float d = fetchDepth(p);
    if (d >= 0.99999) return;
    vec3 P = position((vec2(p) + 0.5) / vec2(uSize), d);
    vec3 L = positionAt(p - ivec2(1, 0)), R = positionAt(p + ivec2(1, 0));
    vec3 B = positionAt(p - ivec2(0, 1)), T = positionAt(p + ivec2(0, 1));
    vec3 dx = dot(R - P, R - P) < dot(P - L, P - L) ? R - P : P - L;
    vec3 dy = dot(T - P, T - P) < dot(P - B, P - B) ? T - P : P - B;
    vec3 N = cross(dx, dy) * uFlipY;
    N /= max(length(N), 1e-6);
    // Grass tufts (their own stencil class, second draw): thin blades have no usable depth normal, and they stand on
    // the ground, so they take the light the ground gets.
    if (uGrass != 0) N = vec3(0.0, 1.0, 0.0);
    vec4 e = wgiIrradiance(P, N);
    // Albedo: the voxel copy of the surface where there is one (static world), otherwise (actors) the hue of the
    // lit pixel at a neutral reflectance. The voxels hold the display-space reflectance the bounce is emitted with;
    // the receiving surface is drawn with its (darker) linear albedo, so the exponent sits between the two.
    ivec3 dim = textureSize(uVoxAlbedo, 0);
    vec3 q = P / uVoxSize - N * 0.1; // world voxel coordinates; the sampler wraps like the window does
    vec4 va = textureLod(uVoxAlbedo, q / vec3(dim), 0.0);
    vec3 scene = max(texelFetch(uScene, p, 0).rgb, vec3(0.0));
    float Y = dot(scene, kLuma);
    // The pixel's own estimate (the square root of the lit colour, the local-light pass's convention) carries the
    // texture detail and keeps painted darks dark; the voxel average carries the true hue and level. World
    // surfaces use the geometric mean of the two.
    vec3 proxy = sqrt(scene) * 0.9;
    vec3 voxel = pow(va.rgb / max(va.a, 1e-3), vec3(uReceive));
    vec3 albedo = mix(proxy, sqrt(proxy * voxel), smoothstep(0.05, 0.35, va.a));
    // Towards the faces of the window the probes know less and less: fade out over the last two probe cells.
    vec3 rel = P / uVoxSize - vec3(uVoxOrigin);
    vec3 edge = min(rel, vec3(dim) - rel) * uVoxSize / (2.0 * uProbeStep);
    float inside = clamp(min(edge.x, min(edge.y, edge.z)), 0.0, 1.0);
    // Native fog already hides distant surfaces; light added on top of the fog colour would read as haze.
    float fog = clamp((d * 2.0 - 1.0) * uFogMul + uFogOff, 0.0, 255.0) / 255.0;
    float k = inside * (1.0 - fog);
    vec3 bounce = e.rgb * albedo * (uStrength * k);
    // Already bright (sunlit) surfaces need the fill less: keeps highlights from clipping.
    bounce *= 1.0 - 0.6 * smoothstep(0.6, 1.6, Y);
    // Deep shade stays deep: in daylight the bounce may raise a pixel by at most about 60% of its own level, so dark
    // rock and painted blacks are tinted rather than greyed. By lamplight (night, interiors) it may fill freely.
    float ratio = dot(bounce, kLuma) / max(Y, 1e-4);
    bounce /= 1.0 + ratio / mix(3.0, 0.6, uDay);
    oColor.rgb = bounce;
    // Optional: enclosed places see less sky, so the flat ambient they were given is pulled down.
    oFactor.rgb = vec3(mix(1.0, mix(1.0 - uOcclusion, 1.0, smoothstep(0.0, 0.7, e.a)), k));
    if (uDebug != 0) {
        oFactor = vec4(0.0);
        // The surface's own voxel: the one holding P, or its neighbour along the normal when P sits near a face.
        vec3 qv = P / uVoxSize;
        ivec3 v = ivec3(floor(qv)) & (dim - 1);
        for (int i = 0; i < 2 && texelFetch(uVoxAlbedo, v, 0).a == 0.0; ++i)
            v = ivec3(floor(qv + N * (i == 0 ? -0.5 : 0.5))) & (dim - 1);
        if (uDebug == 1) oColor.rgb = bounce * 4.0;
        else if (uDebug == 2) oColor.rgb = texelFetch(uVoxAlbedo, v, 0).rgb;
        else if (uDebug == 3) oColor.rgb = texelFetch(uVoxRadiance, v, 0).rgb;
        else if (uDebug == 4) oColor.rgb = vec3(e.a);
        else if (uDebug == 5) oColor.rgb = texelFetch(uVoxNormal, v, 0).rgb * texelFetch(uVoxAlbedo, v, 0).a;
        else if (uDebug == 6) oColor.rgb = e.rgb;
        else if (uDebug == 8) oColor.rgb = vec3(va.a, texelFetch(uVoxNormal, v, 0).a, inside);
        else oColor.rgb = N * 0.5 + 0.5;
    }
}
)GLSL";

void Texture3D(GLuint& tex, int unit, GLint internal, int w, int h, int d, GLenum format, GLenum type, GLint filter) {
    if (!tex) glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_3D, tex);
    glTexImage3D(GL_TEXTURE_3D, 0, internal, w, h, d, 0, format, type, nullptr);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, filter);
    for (GLenum wrap : { GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R })
        glTexParameteri(GL_TEXTURE_3D, wrap, GL_REPEAT);
}

void Texture2D(GLuint& tex, int unit, GLint internal, int w, int h, GLenum format, GLint filter) {
    if (!tex) glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, format, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

int FloorDiv(int a, int b) {
    return (a >= 0 ? a : a - b + 1) / b;
}
} // namespace

bool PostFxGL::WorldGiWantsGeometry(uint32_t frameIndex) const {
    // Every frame while a scene settles or the window has just moved (rooms and their textures arrive over several
    // frames), then four times a second: enough to pick up meshes the game starts submitting when the camera turns.
    // Frozen-frame fixture: the restart on landing needs this frame's geometry, and the cadence counts from the landing.
    const uint32_t tick = NewDawnFreezeTick().load(std::memory_order_relaxed);
    const bool restart = mWgi.freezeEpoch != NewDawnFreezeEpoch().load(std::memory_order_relaxed);
    return mWgi.ready && mHdrActive && (mWgi.want > 0 || restart || (tick != 0 ? tick : frameIndex) % 15u == 0u);
}

bool PostFxGL::BuildWorldGi() {
    mWgi.attempted = true;
    GLint units = 0, major = 0, minor = 0;
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &units);
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    if (major < 4 || (major == 4 && minor < 4) || units <= kRayUnit) {
        SPDLOG_WARN("NewDawnWorldGI: unavailable, needs OpenGL 4.4 and {} texture units", kRayUnit + 1);
        return false;
    }
    // Complete dummies on every lane before any program that declares the samplers exists.
    GLint previousActive;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActive);
    Texture3D(mWgi.voxAlbedo, kVoxAlbedoUnit, GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR);
    Texture3D(mWgi.voxNormal, kVoxNormalUnit, GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, GL_NEAREST);
    Texture3D(mWgi.voxRadiance, kVoxRadianceUnit, GL_R11F_G11F_B10F, 1, 1, 1, GL_RGB, GL_FLOAT, GL_NEAREST);
    Texture3D(mWgi.voxCoarse, kVoxCoarseUnit, GL_R8, 1, 1, 1, GL_RED, GL_UNSIGNED_BYTE, GL_NEAREST);
    Texture2D(mWgi.probeIrr, kProbeIrrUnit, GL_RGBA16F, 1, 1, GL_RGBA, GL_LINEAR);
    Texture2D(mWgi.probeDist, kProbeDistUnit, GL_RG16F, 1, 1, GL_RG, GL_LINEAR);
    Texture2D(mWgi.rays, kRayUnit, GL_RGBA16F, 1, 1, GL_RGBA, GL_NEAREST);
    glActiveTexture(previousActive);

    static const std::string version = "#version 440 core\n";
    const std::string inject = version + kLib + kInjectFs, trace = version + kLib + kTraceFs;
    const std::string blend = version + kLib + kBlendFs, apply = version + kLib + kApplyFs;
    const bool ok = BuildAtmosShader(mWgi.voxelize, kVoxelizeVs, kVoxelizeFs) &&
                    BuildAtmosShader(mWgi.inject, kFullscreenVs, inject.c_str()) &&
                    BuildAtmosShader(mWgi.trace, kFullscreenVs, trace.c_str()) &&
                    BuildAtmosShader(mWgi.blend, kFullscreenVs, blend.c_str()) &&
                    BuildAtmosShader(mWgi.apply, kFullscreenVs, apply.c_str());
    if (!ok) {
        SPDLOG_ERROR("NewDawnWorldGI: shader build failed, disabled");
        return false;
    }
    for (Program* program : { &mWgi.inject, &mWgi.trace, &mWgi.blend, &mWgi.apply }) {
        Use(*program);
        glUniform1i(Loc("uVoxAlbedo"), kVoxAlbedoUnit);
        glUniform1i(Loc("uVoxNormal"), kVoxNormalUnit);
        glUniform1i(Loc("uVoxRadiance"), kVoxRadianceUnit);
        glUniform1i(Loc("uVoxCoarse"), kVoxCoarseUnit);
        glUniform1i(Loc("uProbeIrr"), kProbeIrrUnit);
        glUniform1i(Loc("uProbeDist"), kProbeDistUnit);
        glUniform1i(Loc("uRays"), kRayUnit);
        glUniform1i(Loc("uSkyIrradiance"), kSkyAmbientUnit);
        glUniform1i(Loc("uScene"), 8);
        glUniform1i(Loc("uDepth"), 9);
    }
    glGenFramebuffers(1, &mWgi.voxFbo);
    glGenFramebuffers(1, &mWgi.radFbo);
    glGenFramebuffers(1, &mWgi.rayFbo);
    glGenFramebuffers(1, &mWgi.probeFbo);
    glGenVertexArrays(1, &mWgi.vao);
    glGenBuffers(1, &mWgi.vbo);
    glBindVertexArray(mWgi.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mWgi.vbo);
    const GLsizei stride = (NEWDAWN_GI_TRI_FLOATS / 3) * sizeof(float);
    for (int i = 0; i < 4; ++i) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, i < 3 ? 3 : 1, GL_FLOAT, GL_FALSE, stride, (const void*)(i * 3 * sizeof(float)));
    }
    glBindVertexArray(mVao);
    mWgi.ready = true;
    return true;
}

// Where the window should sit for this camera: the whole scene when it fits (it then never moves), otherwise centred
// on the camera and kept inside the scene's bounds. In voxel cells, snapped to kSnap.
static void WorldGiDesiredOrigin(const float lo[3], const float hi[3], const float eye[3], const int dim[3], float voxel,
                                 int out[3]) {
    for (int i = 0; i < 3; ++i) {
        const float window = (float)dim[i] * voxel, extent = hi[i] - lo[i];
        float corner = lo[i] - 0.5f * (window - extent);
        if (extent > window)
            corner = std::clamp(eye[i] - 0.5f * window, lo[i], hi[i] - window);
        out[i] = FloorDiv((int)std::floor(corner / voxel) + kSnap / 2, kSnap) * kSnap;
    }
}

// New scene (or quality tier): size the window to the scene and start from empty.
void PostFxGL::ResetWorldGi(const NewDawnPostFxParams& params, int tier) {
    const Tier& t = kTiers[tier - 1];
    auto& g = mWgi;
    const bool resize = g.tier != tier;
    g.tier = tier;
    g.scene = params.giSceneId;
    std::copy_n(params.giWorldBoundsMin, 3, g.boundsMin);
    std::copy_n(params.giWorldBoundsMax, 3, g.boundsMax);
    // Walls and roofs usually rise above the walkable collision: pad the top generously, the sides a little.
    g.voxSize = 4.0f;
    for (int i = 0; i < 3; ++i) {
        const float size = std::max(g.boundsMax[i] - g.boundsMin[i], 100.0f);
        g.lo[i] = g.boundsMin[i] - size * 0.04f - 40.0f;
        g.hi[i] = g.boundsMax[i] + size * (i == 1 ? 0.35f : 0.04f) + (i == 1 ? 200.0f : 40.0f);
        g.voxSize = std::max(g.voxSize, (g.hi[i] - g.lo[i]) / (float)t.dim[i]);
        g.dim[i] = t.dim[i];
    }
    g.voxSize = std::min(g.voxSize, t.maxVoxel);
    g.perProbe = t.dim[0] / kProbeDim[0];
    g.probeStep = g.voxSize * (float)g.perProbe;
    WorldGiDesiredOrigin(g.lo, g.hi, params.giCameraEye, g.dim, g.voxSize, g.origin);
    g.rayCount = t.rays;
    g.rayRows = t.rows * kWarmBoost;
    GLint previousActive;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActive);
    if (resize) {
        Texture3D(g.voxAlbedo, kVoxAlbedoUnit, GL_RGBA8, t.dim[0], t.dim[1], t.dim[2], GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR);
        Texture3D(g.voxNormal, kVoxNormalUnit, GL_RGBA8, t.dim[0], t.dim[1], t.dim[2], GL_RGBA, GL_UNSIGNED_BYTE, GL_NEAREST);
        Texture3D(g.voxRadiance, kVoxRadianceUnit, GL_R11F_G11F_B10F, t.dim[0], t.dim[1], t.dim[2], GL_RGB, GL_FLOAT,
                  GL_NEAREST);
        Texture3D(g.voxCoarse, kVoxCoarseUnit, GL_R8, t.dim[0] / 4, t.dim[1] / 4, t.dim[2] / 4, GL_RED, GL_UNSIGNED_BYTE,
                  GL_NEAREST);
        Texture2D(g.probeIrr, kProbeIrrUnit, GL_RGBA16F, kAtlasCols * kTile, kAtlasRows * kTile, GL_RGBA, GL_LINEAR);
        Texture2D(g.probeDist, kProbeDistUnit, GL_RG16F, kAtlasCols * kTile, kAtlasRows * kTile, GL_RG, GL_LINEAR);
        Texture2D(g.rays, kRayUnit, GL_RGBA16F, t.rays, g.rayRows * kAtlasCols, GL_RGBA, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, g.voxFbo);
        const int side = 2 * std::max(t.dim[0], std::max(t.dim[1], t.dim[2]));
        glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, side);
        glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_HEIGHT, side);
        glBindFramebuffer(GL_FRAMEBUFFER, g.rayFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g.rays, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, g.probeFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g.probeIrr, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, g.probeDist, 0);
        const GLenum both[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(2, both);
    }
    glActiveTexture(previousActive);
    glClearTexImage(g.voxAlbedo, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glClearTexImage(g.voxNormal, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glClearTexImage(g.voxCoarse, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    glClearTexImage(g.voxRadiance, 0, GL_RGB, GL_FLOAT, nullptr);
    glClearTexImage(g.probeIrr, 0, GL_RGBA, GL_FLOAT, nullptr);
    glClearTexImage(g.probeDist, 0, GL_RG, GL_FLOAT, nullptr);
    g.sliceCursor = g.rowCursor = 0;
    g.frames = g.updates = g.sweeps = 0;
    g.tris = 0;
    g.want = kWarmFrames;
    SPDLOG_INFO("NewDawnWorldGI: scene {} tier {}: {}x{}x{} voxels of {:.1f} units ({}), probes every {:.0f} units",
                g.scene, tier, g.dim[0], g.dim[1], g.dim[2], g.voxSize,
                g.voxSize >= t.maxVoxel ? "window follows the camera" : "whole scene", g.probeStep);
}

// The window moved by whole cells: the cells that left it and the cells that entered share texels, so those slabs
// (voxels, radiance and probe tiles) are emptied. Everything else keeps its data.
static void WorldGiClearSlabs(const int oldOrigin[3], const int newOrigin[3], const int dim[3], int perProbe,
                              const GLuint vox[4], GLuint probeIrr, GLuint probeDist) {
    const GLenum formats[4] = { GL_RGBA, GL_RGBA, GL_RGB, GL_RED };
    const GLenum types[4] = { GL_UNSIGNED_BYTE, GL_UNSIGNED_BYTE, GL_FLOAT, GL_UNSIGNED_BYTE };
    for (int axis = 0; axis < 3; ++axis) {
        const int delta = newOrigin[axis] - oldOrigin[axis];
        if (delta == 0) continue;
        // Entering cells, as a wrapped start and a length.
        const int first = delta > 0 ? oldOrigin[axis] + dim[axis] : newOrigin[axis];
        const int count = std::abs(delta);
        for (int done = 0; done < count;) {
            const int start = (first + done) & (dim[axis] - 1);
            const int run = std::min(count - done, dim[axis] - start);
            for (int k = 0; k < 4; ++k) {
                const int scale = k == 3 ? 4 : 1; // the coarse occupancy grid
                int offset[3] = { 0, 0, 0 }, size[3] = { dim[0] / scale, dim[1] / scale, dim[2] / scale };
                offset[axis] = start / scale;
                size[axis] = run / scale;
                glClearTexSubImage(vox[k], 0, offset[0], offset[1], offset[2], size[0], size[1], size[2], formats[k],
                                   types[k], nullptr);
            }
            for (int p = start / perProbe; p < (start + run) / perProbe; ++p) {
                auto clear = [&](int tx, int ty, int tw, int th) {
                    glClearTexSubImage(probeIrr, 0, tx * kTile, ty * kTile, 0, tw * kTile, th * kTile, 1, GL_RGBA, GL_FLOAT,
                                       nullptr);
                    glClearTexSubImage(probeDist, 0, tx * kTile, ty * kTile, 0, tw * kTile, th * kTile, 1, GL_RG, GL_FLOAT,
                                       nullptr);
                };
                if (axis == 0) {
                    for (int j = 0; j < 8; ++j) clear(p + 32 * j, 0, 1, kAtlasRows);
                } else if (axis == 1) {
                    for (int j = 0; j < 4; ++j) clear(0, p + 16 * j, kAtlasCols, 1);
                } else {
                    clear(32 * (p & 7), 16 * (p >> 3), 32, 16);
                }
            }
            done += run;
        }
    }
}

void PostFxGL::RunWorldGi(const NewDawnPostFxParams& params, const FramebufferOGL& fb, uint32_t frameIndex,
                          const Target* sceneCopy) {
    auto& g = mWgi;
    if (!g.attempted) BuildWorldGi();
    if (!g.ready) return;
    const int tier = std::clamp(params.giWorld, 1, 2);
    const Tier& t = kTiers[tier - 1];
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    const uint32_t freezeEpoch = NewDawnFreezeEpoch().load(std::memory_order_relaxed);
    if (g.tier != tier || g.scene != params.giSceneId || g.freezeEpoch != freezeEpoch ||
        memcmp(g.boundsMin, params.giWorldBoundsMin, sizeof(g.boundsMin)) != 0 ||
        memcmp(g.boundsMax, params.giWorldBoundsMax, sizeof(g.boundsMax)) != 0) {
        g.freezeEpoch = freezeEpoch;
        ResetWorldGi(params, tier);
    } else {
        // Follow the camera in big scenes. The window only moves once the camera is most of a snap step off-centre,
        // so pacing around one spot never moves it back and forth.
        int desired[3], exact[3];
        WorldGiDesiredOrigin(g.lo, g.hi, params.giCameraEye, g.dim, g.voxSize, desired);
        bool move = false, far = false;
        for (int i = 0; i < 3; ++i) {
            const float window = (float)g.dim[i] * g.voxSize;
            exact[i] = g.hi[i] - g.lo[i] > window
                           ? (int)std::floor(std::clamp(params.giCameraEye[i] - 0.5f * window, g.lo[i], g.hi[i] - window) /
                                             g.voxSize)
                           : g.origin[i];
            move = move || std::abs(exact[i] - g.origin[i]) > kSnap * 3 / 4;
            far = far || std::abs(desired[i] - g.origin[i]) >= g.dim[i];
        }
        if (far) {
            ResetWorldGi(params, tier); // a warp within the scene: nothing overlaps
        } else if (move && memcmp(desired, g.origin, sizeof(desired)) != 0) {
            const GLuint vox[4] = { g.voxAlbedo, g.voxNormal, g.voxRadiance, g.voxCoarse };
            WorldGiClearSlabs(g.origin, desired, g.dim, g.perProbe, vox, g.probeIrr, g.probeDist);
            std::copy_n(desired, 3, g.origin);
            g.want = std::max<uint32_t>(g.want, 3);
            if (mStats)
                SPDLOG_INFO("NewDawnWorldGI: window moved to {},{},{} (frame {})", desired[0], desired[1], desired[2],
                            g.frames);
        }
    }
    const bool warm = g.frames < (uint32_t)kWarmFrames;
    const int boost = warm ? kWarmBoost : 1;
    if (g.want > 0) --g.want;

    const int probeOrigin[3] = { g.origin[0] / g.perProbe, g.origin[1] / g.perProbe, g.origin[2] / g.perProbe };
    auto grid = [&]() {
        glUniform3iv(Loc("uVoxOrigin"), 1, g.origin);
        glUniform1f(Loc("uVoxSize"), g.voxSize);
        glUniform3iv(Loc("uVoxDim"), 1, g.dim);
        glUniform3iv(Loc("uProbeOrigin"), 1, probeOrigin);
        glUniform1f(Loc("uProbeStep"), g.probeStep);
        glUniform1f(Loc("uProbeBias"), 0.2f * g.probeStep);
        glUniform1i(Loc("uSteps"), t.steps);
    };

    // gNewDawn.GI.WorldFreeze=1 (debug, timing): keep the grids as they are and only run the apply pass.
    const bool frozen = CvI("gNewDawn.GI.WorldFreeze", 0) != 0 && g.sweeps > 8;
    // 1. Voxelise whatever room geometry was recorded this frame.
    // Stats only: wall-clock cost of a build (first voxelisation of a scene), bracketed by glFinish.
    const bool timeBuild = mStats && g.frames < 2 && mWgiTriCount > 0;
    std::chrono::steady_clock::time_point buildStart;
    if (timeBuild) {
        glFinish();
        buildStart = std::chrono::steady_clock::now();
    }
    if (!frozen && mWgiTris != nullptr && mWgiTriCount > 0) {
        glBindVertexArray(g.vao);
        glBindBuffer(GL_ARRAY_BUFFER, g.vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)mWgiTriCount * NEWDAWN_GI_TRI_FLOATS * sizeof(float), mWgiTris,
                     GL_STREAM_DRAW);
        Use(g.voxelize);
        grid();
        glBindImageTexture(0, g.voxAlbedo, 0, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA8);
        glBindImageTexture(1, g.voxNormal, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA8);
        glBindImageTexture(2, g.voxCoarse, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);
        glBindFramebuffer(GL_FRAMEBUFFER, g.voxFbo);
        for (int axis = 0; axis < 3; ++axis) {
            const int a = axis == 0 ? 1 : 0, b = axis == 2 ? 1 : 2;
            glViewport(0, 0, 2 * g.dim[a], 2 * g.dim[b]);
            glUniform1i(Loc("uAxis"), axis);
            if (NewDawnTimeFrozen().load(std::memory_order_relaxed)) {
                // Frozen-frame fixture: the shader keeps the first triangle to write a voxel, which is a race between
                // fragments. One triangle per draw with a barrier after each fixes the winner (the draw order).
                for (uint32_t tri = 0; tri < mWgiTriCount; ++tri) {
                    glDrawArrays(GL_TRIANGLES, (GLint)tri * 3, 3);
                    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
                }
            } else {
                glDrawArrays(GL_TRIANGLES, 0, (GLsizei)mWgiTriCount * 3);
            }
        }
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        glBindVertexArray(mVao);
        if (mWgiTriCount != g.tris && mStats)
            SPDLOG_INFO("NewDawnWorldGI: {} room triangles voxelised (frame {})", mWgiTriCount, g.frames);
        g.tris = mWgiTriCount;
    }
    if (timeBuild) {
        glFinish();
        SPDLOG_INFO("NewDawnWorldGI: voxelising {} triangles took {:.2f} ms (frame {}, with glFinish)", mWgiTriCount,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count(),
                    g.frames);
    }

    // 2. Relight a block of radiance slices.
    const float sun = std::clamp(CvF("gNewDawn.HDR.SunIntensity", 1.8f), 0.0f, 16.0f);
    const float fill = std::clamp(CvF("gNewDawn.HDR.AmbientFill", 0.4f), 0.0f, 4.0f);
    const float skyScale = std::clamp(CvF("gNewDawn.HDR.SkyIntensity", 1.0f), 0.0f, 4.0f);
    float sunRgb[3], skyRgb[3];
    for (int i = 0; i < 3; ++i) {
        sunRgb[i] = params.giWorldSunColor[i] * sun * skyScale;
        skyRgb[i] = params.giWorldSkyColor[i] * fill * skyScale;
    }
    const bool lights = g.lightFrame == frameIndex;
    float hysteresis = 0.0f;
    // The radiance and probe update runs at most gNewDawn.GI.WorldUpdateHz times a second of wall-clock time (default
    // 30, 0 = every displayed frame), so its cost no longer grows with the interpolated frame rate: the probes follow
    // the sun and the lamps over seconds, a 60 or 144 Hz update bought nothing visible. The first frames of a scene and
    // of a window move always update.
    const double nowT = NewDawnClockSeconds(); // stands still under gNewDawn.Debug.FreezeTime: only the warm-up sweeps run
    const float updateHz = std::clamp(CvF("gNewDawn.GI.WorldUpdateHz", 30.0f), 0.0f, 1000.0f);
    const bool step = !frozen && !NewDawnFreezeHold() && (warm || g.want > 0 || updateHz <= 0.0f || nowT - g.lastUpdateT >= 0.9 / updateHz);
    if (step) {
    g.lastUpdateT = nowT;
    Use(g.inject);
    grid();
    glUniform3fv(Loc("uSunDir"), 1, params.giWorldSunDir);
    glUniform3fv(Loc("uSunRgb"), 1, sunRgb);
    glUniform3fv(Loc("uSkyRgb"), 1, skyRgb);
    glUniform1i(Loc("uSkyCube"), mSkyAmbientReady ? 1 : 0);
    glUniform1i(Loc("uLightCount"), lights ? g.lightCount : 0);
    glUniform4fv(Loc("uLight[0]"), kMaxLights, &g.lightPos[0][0]);
    glUniform3fv(Loc("uLightRgb[0]"), kMaxLights, &g.lightRgb[0][0]);
    glUniform1f(Loc("uBounce"), std::clamp(CvF("gNewDawn.GI.WorldBounce", 1.0f), 0.0f, 1.0f));
    // Art direction: daylight is plentiful, so only part of it is sent on (the game's ambient already stands for
    // some of it); a lamp's pool is small and capped, so its bounce is pushed to carry across a room.
    glUniform2f(Loc("uGain"), std::clamp(CvF("gNewDawn.GI.WorldSunGain", 0.5f), 0.0f, 4.0f),
                std::clamp(CvF("gNewDawn.GI.WorldLocalGain", 3.0f), 0.0f, 16.0f));
    // Baked surfaces are displayed as their painted colour, pulled down by the local-light pass's darkening.
    // Only part of it is sent on: the paint already contains the artists' own bounce.
    glUniform1f(Loc("uBaked"), std::clamp(CvF("gNewDawn.GI.WorldBakedGain", 0.6f), 0.0f, 4.0f) * skyScale *
                                   (params.localEnabled ? 1.0f - std::clamp(params.localDarken, 0.0f, 0.9f) : 1.0f));
    glBindFramebuffer(GL_FRAMEBUFFER, g.radFbo);
    glViewport(0, 0, g.dim[0], g.dim[1]);
    for (int i = 0, n = std::min(t.slices * boost, g.dim[2]); i < n; ++i) {
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, g.voxRadiance, 0, g.sliceCursor);
        glUniform1i(Loc("uLayer"), g.sliceCursor);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        g.sliceCursor = (g.sliceCursor + 1) % g.dim[2];
    }

    // 3. Trace and blend a block of probes. A fresh rotation of the ray set every update; the first sweeps after a
    // scene change average (hysteresis 0, 1/2, 2/3 ...) so the probes converge fast, then settle at the long blend.
    const int rows = std::min(std::min(t.rows * boost, g.rayRows), kAtlasRows - g.rowCursor);
    float rot[9];
    {
        uint32_t seed = g.updates * 747796405u + 2891336453u;
        auto next = [&seed]() {
            seed = seed * 1664525u + 1013904223u;
            return (float)(seed >> 8) / 16777216.0f;
        };
        const float u1 = next(), u2 = next() * 6.2831853f, u3 = next() * 6.2831853f;
        const float a = std::sqrt(1.0f - u1), b = std::sqrt(u1);
        const float x = a * std::sin(u2), y = a * std::cos(u2), z = b * std::sin(u3), w = b * std::cos(u3);
        const float m[9] = { 1 - 2 * (y * y + z * z), 2 * (x * y + z * w),     2 * (x * z - y * w),
                             2 * (x * y - z * w),     1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
                             2 * (x * z + y * w),     2 * (y * z - x * w),     1 - 2 * (x * x + y * y) };
        std::copy_n(m, 9, rot);
    }
    const int base = g.rowCursor * kAtlasCols;
    Use(g.trace);
    grid();
    glUniform1i(Loc("uBase"), base);
    glUniform1i(Loc("uRayCount"), g.rayCount);
    glUniformMatrix3fv(Loc("uRot"), 1, GL_FALSE, rot);
    glBindFramebuffer(GL_FRAMEBUFFER, g.rayFbo);
    glViewport(0, 0, g.rayCount, rows * kAtlasCols);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    const float longBlend = std::clamp(CvF("gNewDawn.GI.WorldHysteresis", 0.88f), 0.0f, 0.98f);
    hysteresis = std::min(longBlend, 1.0f - 1.0f / (float)(g.sweeps + 1));
    Use(g.blend);
    grid();
    glUniform1i(Loc("uBase"), base);
    glUniform1i(Loc("uRayCount"), g.rayCount);
    glUniformMatrix3fv(Loc("uRot"), 1, GL_FALSE, rot);
    glUniform1f(Loc("uMaxDist"), 1.5f * 1.7320508f * g.probeStep);
    glBindFramebuffer(GL_FRAMEBUFFER, g.probeFbo);
    glViewport(0, g.rowCursor * kTile, kAtlasCols * kTile, rows * kTile);
    GLfloat previousBlend[4];
    glGetFloatv(GL_BLEND_COLOR, previousBlend);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendColor(0.0f, 0.0f, 0.0f, hysteresis);
    glBlendFunc(GL_ONE_MINUS_CONSTANT_ALPHA, GL_CONSTANT_ALPHA);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_BLEND);
    glBlendColor(previousBlend[0], previousBlend[1], previousBlend[2], previousBlend[3]);
    g.rowCursor += rows;
    if (g.rowCursor >= kAtlasRows) {
        g.rowCursor = 0;
        ++g.sweeps;
    }
    ++g.updates;
    } // step
    ++g.frames;
    g.lastFrame = frameIndex;
    if (mStats && g.frames % 300 == 0)
        SPDLOG_INFO("NewDawnWorldGI: frame {} sweeps {} hysteresis {:.2f} lights {} sun {:.2f},{:.2f},{:.2f} sky "
                    "{:.2f},{:.2f},{:.2f} cube {} origin {},{},{}",
                    g.frames, g.sweeps, hysteresis, lights ? g.lightCount : -1, sunRgb[0], sunRgb[1], sunRgb[2], skyRgb[0],
                    skyRgb[1], skyRgb[2], mSkyAmbientReady, g.origin[0], g.origin[1], g.origin[2]);

    if (CvI("gNewDawn.GI.WorldLog", 0) != 0 && g.frames % 30 == 0) {
        // Debug evidence: the mean of the whole probe field. It must not depend on where the camera looks.
        std::vector<float> px((size_t)kAtlasCols * kTile * kAtlasRows * kTile * 4);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, g.probeFbo);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, kAtlasCols * kTile, kAtlasRows * kTile, GL_RGBA, GL_FLOAT, px.data());
        double sum[4] = {};
        for (size_t i = 0; i < px.size(); ++i)
            sum[i & 3] += px[i];
        const double n = (double)(px.size() / 4);
        SPDLOG_INFO("NewDawnWorldGI: probe field mean {:.5f},{:.5f},{:.5f} sky {:.4f} frame {} tris {}", sum[0] / n,
                    sum[1] / n, sum[2] / n, sum[3] / n, g.frames, g.tris);
    }

    // 4. Apply to the scene.
    const int debug = CvI("gNewDawn.GI.WorldDebug", 0);
    Bind(0, sceneCopy->tex);
    Bind(1, mDepth.tex);
    Use(g.apply);
    grid();
    glUniform2i(Loc("uSize"), fb.width, fb.height);
    glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, params.shadowInvViewProj);
    glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), params.shadowFlipY);
    glUniform1f(Loc("uStrength"), params.giWorldStrength);
    glUniform1f(Loc("uOcclusion"), std::clamp(CvF("gNewDawn.GI.WorldOcclusion", 0.0f), 0.0f, 1.0f));
    glUniform1i(Loc("uDebug"), debug);
    glUniform1f(Loc("uReceive"), std::clamp(CvF("gNewDawn.GI.WorldReceive", 1.0f), 1.0f, 2.2f));
    glUniform1f(Loc("uDay"), std::clamp((0.2126f * sunRgb[0] + 0.7152f * sunRgb[1] + 0.0722f * sunRgb[2]) / 0.5f, 0.0f, 1.0f));
    // The native fog factor, as the local-light pass derives it from the depth buffer.
    float fogMul = 0.0f, fogOff = 0.0f;
    if (params.fogNear >= 997 && params.fogNear < 1000) {
        fogMul = 32767.0f;
        fogOff = -32512.0f;
    } else if (params.fogNear < 0) {
        fogOff = 255.0f;
    } else if (params.fogNear < 997) {
        const int range = std::max(params.fogFar - params.fogNear, 1);
        fogMul = static_cast<float>(static_cast<int16_t>(128000 / range));
        fogOff = static_cast<float>(static_cast<int16_t>((500 - params.fogNear) * 256 / range));
    }
    glUniform1f(Loc("uFogMul"), fogMul);
    glUniform1f(Loc("uFogOff"), fogOff);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_SRC1_COLOR);
    // Everything but the grass tufts (stencil 2, written by the grass pass), then the tufts with the ground's normal.
    glEnable(GL_STENCIL_TEST);
    glStencilMask(0);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glStencilFunc(GL_NOTEQUAL, 2, 0xFF);
    glUniform1i(Loc("uGrass"), 0);
    Draw(fb.fbo, fb.width, fb.height);
    glStencilFunc(GL_EQUAL, 2, 0xFF);
    glUniform1i(Loc("uGrass"), 1);
    Draw(fb.fbo, fb.width, fb.height);
    glStencilMask(0xFF);
    glStencilFunc(GL_ALWAYS, 0, 0xFF);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_BLEND);
    CheckError("world GI");
}

} // namespace Fast
#endif
