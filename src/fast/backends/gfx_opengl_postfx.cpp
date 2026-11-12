#if defined(ENABLE_OPENGL) && !defined(USE_OPENGLES)

#include "fast/backends/gfx_opengl_postfx.h"

#include <algorithm>
#include <cmath>
#include <string>

#include <spdlog/spdlog.h>

#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"

namespace Fast {

// Defaults for the gNewDawn.PostFx.* settings. The menu in the game uses the same numbers.
namespace Defaults {
constexpr float kAoStrength = 1.4f;
constexpr float kAoRadius = 40.0f;
constexpr float kBloomStrength = 0.25f;
constexpr float kBloomThreshold = 0.90f;
constexpr float kContrast = 1.06f;
constexpr float kSaturation = 1.08f;
} // namespace Defaults

static int CvInt(const char* name, int def) {
    return Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger(name, def);
}

static float CvFloat(const char* name, float def) {
    return Ship::Context::GetRawInstance()->GetConsoleVariables()->GetFloat(name, def);
}

static PostFxGL::Settings ReadSettings() {
    PostFxGL::Settings s;
    s.postFxOn = CvInt("gNewDawn.PostFx.Enabled", 1) != 0;
    s.shadowsOn = CvInt("gNewDawn.Shadows.Enabled", 0) != 0;
    s.shadowDebug = CvInt("gNewDawn.Shadows.Debug", 0);
    s.aoOn = CvInt("gNewDawn.PostFx.AO.Enabled", 1) != 0;
    s.aoStrength = std::clamp(CvFloat("gNewDawn.PostFx.AO.Strength", Defaults::kAoStrength), 0.0f, 4.0f);
    s.aoRadius = std::clamp(CvFloat("gNewDawn.PostFx.AO.Radius", Defaults::kAoRadius), 1.0f, 500.0f);
    s.bloomOn = CvInt("gNewDawn.PostFx.Bloom.Enabled", 1) != 0;
    s.bloomStrength = std::clamp(CvFloat("gNewDawn.PostFx.Bloom.Strength", Defaults::kBloomStrength), 0.0f, 2.0f);
    s.bloomThreshold = std::clamp(CvFloat("gNewDawn.PostFx.Bloom.Threshold", Defaults::kBloomThreshold), 0.0f, 0.99f);
    s.gradeOn = CvInt("gNewDawn.PostFx.Grade.Enabled", 1) != 0;
    s.contrast = std::clamp(CvFloat("gNewDawn.PostFx.Grade.Contrast", Defaults::kContrast), 0.5f, 1.5f);
    s.saturation = std::clamp(CvFloat("gNewDawn.PostFx.Grade.Saturation", Defaults::kSaturation), 0.0f, 2.0f);
    s.debugView = CvInt("gNewDawn.PostFx.Debug.View", 0);
    s.stats = CvInt("gNewDawn.PostFx.Debug.Stats", 0) != 0;
    return s;
}

static void Normalize3(const float in[3], float out[3]) {
    float len = std::sqrt(in[0] * in[0] + in[1] * in[1] + in[2] * in[2]);
    if (len < 1e-6f) {
        out[0] = out[1] = out[2] = 0.0f;
        return;
    }
    out[0] = in[0] / len;
    out[1] = in[1] / len;
    out[2] = in[2] / len;
}

static void Cross3(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static void Mat4Multiply(const float a[16], const float b[16], float out[16]) {
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
}

// Builds world -> light clip (column-major for GLSL). The light looks from the sun at the focus.
static void BuildLightViewProj(const NewDawnPostFxParams& params, float out[16]) {
    float dir[3] = { params.shadowSunDir[0], params.shadowSunDir[1], params.shadowSunDir[2] };
    float focus[3] = { params.shadowFocus[0], params.shadowFocus[1], params.shadowFocus[2] };
    float up[3] = { 0.0f, 1.0f, 0.0f };
    if (std::fabs(dir[1]) > 0.98f) {
        up[0] = 0.0f;
        up[1] = 0.0f;
        up[2] = 1.0f;
    }
    float forward[3] = { -dir[0], -dir[1], -dir[2] };
    Normalize3(forward, forward);
    float right[3];
    Cross3(forward, up, right);
    Normalize3(right, right);
    float realUp[3];
    Cross3(right, forward, realUp);

    // Snap the centre to the shadow-map texel grid so the map does not swim as Link moves.
    float texel = 2.0f * params.shadowRadius / (float)params.shadowMapSize;
    float focusRight = focus[0] * right[0] + focus[1] * right[1] + focus[2] * right[2];
    float focusUp = focus[0] * realUp[0] + focus[1] * realUp[1] + focus[2] * realUp[2];
    float snapRight = std::round(focusRight / texel) * texel - focusRight;
    float snapUp = std::round(focusUp / texel) * texel - focusUp;
    for (int i = 0; i < 3; i++) {
        focus[i] += right[i] * snapRight + realUp[i] * snapUp;
    }

    float distance = params.shadowRadius + 1000.0f;
    float eye[3] = { focus[0] + dir[0] * distance, focus[1] + dir[1] * distance, focus[2] + dir[2] * distance };

    float view[16] = {
        right[0], realUp[0], -forward[0], 0.0f, right[1], realUp[1], -forward[1], 0.0f,
        right[2], realUp[2], -forward[2], 0.0f, -right[0] * eye[0] - right[1] * eye[1] - right[2] * eye[2],
        -realUp[0] * eye[0] - realUp[1] * eye[1] - realUp[2] * eye[2],
        forward[0] * eye[0] + forward[1] * eye[1] + forward[2] * eye[2], 1.0f,
    };

    float radius = params.shadowRadius;
    float nearZ = 1.0f;
    float farZ = distance + radius + 1000.0f;
    float ortho[16] = {
        1.0f / radius, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f / radius, 0.0f, 0.0f, 0.0f, 0.0f,
        -2.0f / (farZ - nearZ), 0.0f, 0.0f, 0.0f, -(farZ + nearZ) / (farZ - nearZ), 1.0f,
    };
    Mat4Multiply(ortho, view, out);
}

// ---------------------------------------------------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------------------------------------------------

static const char* kVersion = "#version 330 core\n";

// One oversized triangle covering the screen, no vertex attributes needed.
static const char* kVertexShader = R"GLSL(
out vec2 vUv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

// Depth helpers shared by the AO passes. The projection is guPerspective with OpenGL conventions, so the depth buffer
// holds (ndc.z * 0.5 + 0.5) and distance along the view axis is recovered with the usual inverse.
static const char* kDepthLib = R"GLSL(
uniform sampler2D uDepth;
uniform ivec2 uSize;
uniform float uNear;
uniform float uFar;

float linDepth(float d) {
    float z = d * 2.0 - 1.0;
    return 2.0 * uNear * uFar / (uFar + uNear - z * (uFar - uNear));
}

float fetchDepth(ivec2 p) {
    return texelFetch(uDepth, clamp(p, ivec2(0), uSize - 1), 0).r;
}
)GLSL";

static const char* kAoShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform vec2 uTan;     // tan of the half field of view, horizontal and vertical
uniform float uFlipY;  // -1 when the framebuffer is stored vertically flipped
uniform float uRadius;
uniform float uMaxPx;
uniform vec3 uKernel[16];

const float kBayer[16] = float[16](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);

vec3 viewPos(ivec2 p, float d) {
    vec2 ndc = (vec2(p) + 0.5) / vec2(uSize) * 2.0 - 1.0;
    ndc.y *= uFlipY;
    float dist = linDepth(d);
    return vec3(ndc.x * uTan.x * dist, ndc.y * uTan.y * dist, -dist);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float d0 = fetchDepth(p);
    if (d0 >= 0.99999) {
        oColor = vec4(1.0);
        return;
    }
    vec3 P = viewPos(p, d0);

    // Normal from the neighbours on the side with the smaller depth step, which keeps silhouettes clean.
    vec3 Pl = viewPos(p + ivec2(-1, 0), fetchDepth(p + ivec2(-1, 0)));
    vec3 Pr = viewPos(p + ivec2(1, 0), fetchDepth(p + ivec2(1, 0)));
    vec3 Pd = viewPos(p + ivec2(0, -1), fetchDepth(p + ivec2(0, -1)));
    vec3 Pu = viewPos(p + ivec2(0, 1), fetchDepth(p + ivec2(0, 1)));
    bool useR = (p.x == 0) || (p.x < uSize.x - 1 && abs(Pr.z - P.z) < abs(P.z - Pl.z));
    bool useU = (p.y == 0) || (p.y < uSize.y - 1 && abs(Pu.z - P.z) < abs(P.z - Pd.z));
    vec3 dx = useR ? (Pr - P) : (P - Pl);
    vec3 dy = useU ? (Pu - P) : (P - Pd);
    vec3 n = cross(dx, dy);
    float nl = length(n);
    if (nl < 1e-8) {
        oColor = vec4(1.0);
        return;
    }
    vec3 N = n / nl;
    if (dot(N, P) > 0.0) {
        N = -N; // face the camera, which sits at the origin
    }

    float dist = -P.z;
    float rad = uRadius;
    float pxR = rad / (dist * uTan.y) * 0.5 * float(uSize.y); // projected radius in pixels
    if (pxR > uMaxPx) {
        rad *= uMaxPx / pxR;
        pxR = uMaxPx;
    }
    float fade = smoothstep(1.5, 4.0, pxR); // far away the kernel shrinks below a pixel: fade out instead of popping
    if (fade <= 0.0) {
        oColor = vec4(1.0);
        return;
    }

    // Rotate the kernel around the normal by a per-pixel angle from a 4x4 pattern; the blur averages it away.
    float ang = (kBayer[(p.y & 3) * 4 + (p.x & 3)] + 0.5) / 16.0 * 6.2831853;
    vec3 rv = vec3(cos(ang), sin(ang), 0.0);
    vec3 T = rv - N * dot(rv, N);
    float tl = length(T);
    T = (tl > 1e-3) ? T / tl : normalize(cross(N, vec3(0.0, 0.0, 1.0)));
    vec3 B = cross(N, T);

    float bias = 0.05 * rad;
    float occ = 0.0;
    for (int i = 0; i < 16; i++) {
        vec3 k = uKernel[i];
        vec3 S = P + (T * k.x + B * k.y + N * k.z) * rad;
        if (S.z > -0.01) {
            continue;
        }
        vec2 ndc = vec2(S.x / (-S.z) / uTan.x, S.y / (-S.z) / uTan.y);
        ndc.y *= uFlipY;
        vec2 uv = ndc * 0.5 + 0.5;
        if (uv.x < 0.0 || uv.y < 0.0 || uv.x >= 1.0 || uv.y >= 1.0) {
            continue; // off screen: assume open
        }
        float sd = linDepth(fetchDepth(ivec2(uv * vec2(uSize))));
        float range = smoothstep(0.0, 1.0, rad / max(abs(dist - sd), 1e-3));
        occ += ((-S.z) - sd > bias) ? range : 0.0;
    }
    float ao = 1.0 - occ / 16.0;
    oColor = vec4(vec3(mix(1.0, ao, fade)), 1.0);
}
)GLSL";

// Depth-aware blur of the AO buffer. Edge-stopping is relative to the AO radius (and a little to the distance, so
// slanted floors still blur), which keeps the result from bleeding across silhouettes.
static const char* kBlurLib = R"GLSL(
uniform sampler2D uAo;
uniform ivec2 uDir;
uniform float uRadius;

float blurAo(ivec2 p) {
    float dc = linDepth(fetchDepth(p));
    float sigma = max(0.5 * uRadius, 0.04 * dc);
    float sum = 0.0;
    float wsum = 0.0;
    for (int i = -4; i <= 4; i++) {
        ivec2 q = clamp(p + uDir * i, ivec2(0), uSize - 1);
        float dz = (linDepth(fetchDepth(q)) - dc) / sigma;
        float w = exp(-dz * dz);
        sum += texelFetch(uAo, q, 0).r * w;
        wsum += w;
    }
    return sum / wsum;
}
)GLSL";

static const char* kBlurShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    oColor = vec4(vec3(blurAo(p)), 1.0);
}
)GLSL";

// Second blur pass, fused with applying the result to the scene: multiplies the framebuffer (via blending) or, for
// the debug view, replaces it.
static const char* kApplyShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform float uStrength;
uniform float uFogMul;
uniform float uFogOff;
uniform int uMode; // 0 apply, 1 show the AO buffer
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float d = fetchDepth(p);
    if (d >= 0.99999) {
        oColor = vec4(1.0); // sky
        return;
    }
    float a = pow(clamp(blurAo(p), 0.0, 1.0), uStrength);
    if (uMode == 1) {
        oColor = vec4(vec3(a), 1.0);
        return;
    }
    // The N64 fog factor is linear in NDC z (see GfxSpVertex): fade the AO out as the scene fades into fog.
    float fog = clamp((d * 2.0 - 1.0) * uFogMul + uFogOff, 0.0, 255.0) / 255.0;
    oColor = vec4(vec3(mix(1.0, a, 1.0 - fog)), 1.0);
}
)GLSL";

// Bloom downsample: 13 taps. The first level also keeps only what is above the threshold, normalised so that pure
// white contributes 1.0.
static const char* kDownShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform int uPrefilter;
uniform float uThreshold;

vec3 pre(vec3 c) {
    if (uPrefilter == 0) {
        return c;
    }
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float k = max(0.25 * (1.0 - uThreshold), 1e-3);
    float soft = clamp(l - uThreshold + k, 0.0, 2.0 * k);
    soft = soft * soft / (4.0 * k);
    float ex = max(soft, l - uThreshold);
    return c * (ex / (1.0 - uThreshold)) / max(l, 1e-4);
}

vec3 tap(vec2 o) {
    return pre(texture(uSrc, vUv + o * uTexel).rgb);
}

void main() {
    vec3 a = tap(vec2(-2.0, -2.0)), b = tap(vec2(0.0, -2.0)), c = tap(vec2(2.0, -2.0));
    vec3 d = tap(vec2(-2.0, 0.0)), e = tap(vec2(0.0, 0.0)), f = tap(vec2(2.0, 0.0));
    vec3 g = tap(vec2(-2.0, 2.0)), h = tap(vec2(0.0, 2.0)), i = tap(vec2(2.0, 2.0));
    vec3 j = tap(vec2(-1.0, -1.0)), k = tap(vec2(1.0, -1.0));
    vec3 l = tap(vec2(-1.0, 1.0)), m = tap(vec2(1.0, 1.0));
    vec3 r = e * 0.125 + (j + k + l + m) * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625;
    oColor = vec4(r, 1.0);
}
)GLSL";

// Bloom upsample: 3x3 tent, added onto the next larger level by blending.
static const char* kUpShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform float uWeight;
void main() {
    vec3 s = texture(uSrc, vUv + vec2(-1.0, -1.0) * uTexel).rgb + 2.0 * texture(uSrc, vUv + vec2(0.0, -1.0) * uTexel).rgb +
             texture(uSrc, vUv + vec2(1.0, -1.0) * uTexel).rgb + 2.0 * texture(uSrc, vUv + vec2(-1.0, 0.0) * uTexel).rgb +
             4.0 * texture(uSrc, vUv).rgb + 2.0 * texture(uSrc, vUv + vec2(1.0, 0.0) * uTexel).rgb +
             texture(uSrc, vUv + vec2(-1.0, 1.0) * uTexel).rgb + 2.0 * texture(uSrc, vUv + vec2(0.0, 1.0) * uTexel).rgb +
             texture(uSrc, vUv + vec2(1.0, 1.0) * uTexel).rgb;
    oColor = vec4(s * (uWeight / 16.0), 1.0);
}
)GLSL";

static const char* kCompositeShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform int uUseBloom;
uniform float uBloomStrength;
uniform int uUseGrade;
uniform float uContrast;
uniform float uSaturation;

float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

void main() {
    vec3 c = texelFetch(uScene, ivec2(gl_FragCoord.xy), 0).rgb;
    if (uUseBloom != 0) {
        vec3 b = clamp(texture(uBloom, vUv).rgb * uBloomStrength, 0.0, 1.0);
        c = 1.0 - (1.0 - c) * (1.0 - b); // screen: glows without ever clipping
    }
    if (uUseGrade != 0) {
        // Contrast as a blend towards a smoothstep S-curve (fixes black and white, mid slope equals uContrast).
        vec3 s = c * c * (3.0 - 2.0 * c);
        c = c + (uContrast - 1.0) * 2.0 * (s - c);
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c = mix(vec3(l), c, uSaturation);
    }
    // Dither before the 8 bit quantisation. +-half a step leaves unchanged pixels (and pure black) exactly as they were.
    c += (ign(gl_FragCoord.xy) - 0.5) / 255.0;
    oColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
)GLSL";

static const char* kShadowDepthVertex = R"GLSL(
layout(location = 0) in vec3 aPos;
uniform mat4 uLightViewProj;
void main() {
    gl_Position = uLightViewProj * vec4(aPos, 1.0);
}
)GLSL";

static const char* kShadowDepthFragment = R"GLSL(
void main() {
}
)GLSL";

static const char* kShadowApplyShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uColor;
uniform sampler2D uShadow;
uniform vec2 uShadowTexel;
uniform mat4 uInvViewProj;
uniform mat4 uLightViewProj;
uniform vec3 uSunDir;
uniform vec3 uAmbient;
uniform vec3 uDirColor;
uniform vec3 uCameraEye;
uniform float uStrength;
uniform float uMinFactor;
uniform float uNormalBias;
uniform float uNdcXScale;
uniform float uFlipY;
uniform int uDebug;

vec3 worldPos(ivec2 p, float d) {
    vec2 ndc = (vec2(p) + 0.5) / vec2(uSize) * 2.0 - 1.0;
    ndc.y *= uFlipY;
    ndc.x *= uNdcXScale;
    vec4 w = uInvViewProj * vec4(ndc.x, ndc.y, d * 2.0 - 1.0, 1.0);
    return w.xyz / w.w;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float d0 = fetchDepth(p);
    if (d0 >= 0.99999) {
        oColor = vec4(texture(uColor, vUv).rgb, 1.0);
        return;
    }
    vec3 P = worldPos(p, d0);
    if (uDebug == 1) {
        oColor = vec4(fract(P / 100.0), 1.0);
        return;
    }

    float dL = fetchDepth(p + ivec2(-1, 0));
    float dR = fetchDepth(p + ivec2(1, 0));
    float dD = fetchDepth(p + ivec2(0, -1));
    float dU = fetchDepth(p + ivec2(0, 1));
    vec3 Pl = worldPos(p + ivec2(-1, 0), dL);
    vec3 Pr = worldPos(p + ivec2(1, 0), dR);
    vec3 Pd = worldPos(p + ivec2(0, -1), dD);
    vec3 Pu = worldPos(p + ivec2(0, 1), dU);
    bool useR = (p.x == 0) || (p.x < uSize.x - 1 && abs(dR - d0) < abs(d0 - dL));
    bool useU = (p.y == 0) || (p.y < uSize.y - 1 && abs(dU - d0) < abs(d0 - dD));
    vec3 dx = useR ? (Pr - P) : (P - Pl);
    vec3 dy = useU ? (Pu - P) : (P - Pd);
    vec3 n = cross(dx, dy);
    float nl = length(n);
    vec3 N = (nl > 1e-8) ? (n / nl) : vec3(0.0, 1.0, 0.0);
    if (dot(N, uCameraEye - P) < 0.0) {
        N = -N;
    }
    float ndl = max(dot(N, uSunDir), 0.0);

    float lit = 1.0;
    vec4 lc = uLightViewProj * vec4(P + N * uNormalBias, 1.0);
    vec3 ndc = lc.xyz / lc.w;
    vec2 suv = ndc.xy * 0.5 + 0.5;
    float ref = ndc.z * 0.5 + 0.5;
    if (ndl > 0.0 && ref < 1.0 && all(greaterThanEqual(suv, vec2(0.0))) &&
        all(lessThanEqual(suv, vec2(1.0)))) {
        float bias = 0.003 + 0.010 * (1.0 - ndl);
        float sum = 0.0;
        for (int y = -2; y <= 2; y++) {
            for (int x = -2; x <= 2; x++) {
                float sd = texture(uShadow, suv + vec2(x, y) * uShadowTexel).r;
                sum += (ref - bias <= sd) ? 1.0 : 0.0;
            }
        }
        lit = sum / 25.0;
    }
    if (uDebug == 2) {
        oColor = vec4(vec3(lit), 1.0);
        return;
    }

    vec3 direct = uDirColor * ndl;
    vec3 total = max(uAmbient + direct, vec3(1e-4));
    vec3 shade = uAmbient + direct * lit;
    vec3 factor = clamp(shade / total, vec3(uMinFactor), vec3(1.0));
    factor = mix(vec3(1.0), factor, uStrength);
    oColor = vec4(texture(uColor, vUv).rgb * factor, 1.0);
}
)GLSL";

// ---------------------------------------------------------------------------------------------------------------------
// GL helpers
// ---------------------------------------------------------------------------------------------------------------------

static GLuint CompileStage(GLenum type, const char* const* sources, int count, const char* name) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, count, sources, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        SPDLOG_ERROR("NewDawnPostFx: failed to compile {}: {}", name, log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

static GLuint BuildProgramWithVertex(const char* name, const char* const* vsParts, int vsCount,
                                     const char* const* fsParts, int fsCount,
                                     std::unordered_map<std::string, GLint>& uniforms) {
    GLuint vs = CompileStage(GL_VERTEX_SHADER, vsParts, vsCount, "vertex shader");
    GLuint fs = CompileStage(GL_FRAGMENT_SHADER, fsParts, fsCount, name);
    if (!vs || !fs) {
        glDeleteShader(vs);
        glDeleteShader(fs);
        return 0;
    }
    GLuint prg = glCreateProgram();
    glAttachShader(prg, vs);
    glAttachShader(prg, fs);
    glLinkProgram(prg);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(prg, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(prg, sizeof(log), nullptr, log);
        SPDLOG_ERROR("NewDawnPostFx: failed to link {}: {}", name, log);
        glDeleteProgram(prg);
        return 0;
    }
    // Look every uniform up once, here, instead of on each draw
    GLint count = 0;
    glGetProgramiv(prg, GL_ACTIVE_UNIFORMS, &count);
    for (GLint i = 0; i < count; i++) {
        char uname[128];
        GLint size;
        GLenum type;
        glGetActiveUniform(prg, i, sizeof(uname), nullptr, &size, &type, uname);
        std::string key = uname;
        if (key.size() > 3 && key.compare(key.size() - 3, 3, "[0]") == 0) {
            key.resize(key.size() - 3);
        }
        uniforms[key] = glGetUniformLocation(prg, uname);
    }
    return prg;
}

static GLuint BuildProgram(const char* name, const char* const* fsParts, int fsCount,
                           std::unordered_map<std::string, GLint>& uniforms) {
    const char* vsParts[] = { kVersion, kVertexShader };
    return BuildProgramWithVertex(name, vsParts, 2, fsParts, fsCount, uniforms);
}

// Saves every piece of GL state the passes touch and puts it back on destruction. The backend caches a good part of
// this state (blend, scissor, depth, program, bound framebuffer...), so the restore has to be exact.
namespace {
struct GlStateGuard {
    GLint drawFbo, readFbo, program, vao, activeTexture, arrayBuffer;
    GLint viewport[4], scissor[4];
    GLint depthFunc, blendSrcRgb, blendDstRgb, blendSrcA, blendDstA, blendEqRgb, blendEqA;
    GLboolean depthMask, colorMask[4];
    GLboolean blend, depthTest, scissorTest, cullFace, polyOffset, stencilTest, dither, srgb;

    GlStateGuard() {
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFbo);
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFbo);
        glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &arrayBuffer);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
        glGetIntegerv(GL_VIEWPORT, viewport);
        glGetIntegerv(GL_SCISSOR_BOX, scissor);
        glGetIntegerv(GL_DEPTH_FUNC, &depthFunc);
        glGetIntegerv(GL_BLEND_SRC_RGB, &blendSrcRgb);
        glGetIntegerv(GL_BLEND_DST_RGB, &blendDstRgb);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &blendSrcA);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &blendDstA);
        glGetIntegerv(GL_BLEND_EQUATION_RGB, &blendEqRgb);
        glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &blendEqA);
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);
        glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);
        blend = glIsEnabled(GL_BLEND);
        depthTest = glIsEnabled(GL_DEPTH_TEST);
        scissorTest = glIsEnabled(GL_SCISSOR_TEST);
        cullFace = glIsEnabled(GL_CULL_FACE);
        polyOffset = glIsEnabled(GL_POLYGON_OFFSET_FILL);
        stencilTest = glIsEnabled(GL_STENCIL_TEST);
        dither = glIsEnabled(GL_DITHER);
        srgb = glIsEnabled(GL_FRAMEBUFFER_SRGB);
    }

    static void Set(GLenum cap, GLboolean on) {
        if (on) {
            glEnable(cap);
        } else {
            glDisable(cap);
        }
    }

    // Everything the passes rely on: no tests, no blending, plain colour writes, no depth writes.
    void Neutralise() const {
        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_CULL_FACE);
        glDisable(GL_POLYGON_OFFSET_FILL);
        glDisable(GL_STENCIL_TEST);
        glDisable(GL_DITHER);
        glDisable(GL_FRAMEBUFFER_SRGB);
        glDepthMask(GL_FALSE);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    }

    ~GlStateGuard() {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFbo);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, readFbo);
        glUseProgram(program);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, arrayBuffer);
        glActiveTexture(activeTexture);
        glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
        glScissor(scissor[0], scissor[1], scissor[2], scissor[3]);
        glDepthFunc(depthFunc);
        glBlendFuncSeparate(blendSrcRgb, blendDstRgb, blendSrcA, blendDstA);
        glBlendEquationSeparate(blendEqRgb, blendEqA);
        glDepthMask(depthMask);
        glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
        Set(GL_BLEND, blend);
        Set(GL_DEPTH_TEST, depthTest);
        Set(GL_SCISSOR_TEST, scissorTest);
        Set(GL_CULL_FACE, cullFace);
        Set(GL_POLYGON_OFFSET_FILL, polyOffset);
        Set(GL_STENCIL_TEST, stencilTest);
        Set(GL_DITHER, dither);
        Set(GL_FRAMEBUFFER_SRGB, srgb);
    }
};

// Texture units 8 and up are never used by the interpreter's own draws, so binding there cannot disturb its cache.
constexpr int kFirstUnit = 8;
} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// PostFxGL
// ---------------------------------------------------------------------------------------------------------------------

void PostFxGL::Init() {
    GLint major = 0, minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    if (major < 3 || (major == 3 && minor < 3)) {
        SPDLOG_WARN("NewDawnPostFx: disabled, OpenGL {}.{} is below the required 3.3", major, minor);
        return;
    }
    // Missing units would make the passes silently wrong
    GLint units = 0;
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &units);
    if (units < kFirstUnit + 4) {
        SPDLOG_WARN("NewDawnPostFx: disabled, only {} texture units available", units);
        return;
    }

    if (!BuildPrograms()) {
        SPDLOG_WARN("NewDawnPostFx: disabled, shader build failed");
        return;
    }

    // Cosine weighted hemisphere kernel (Hammersley points), samples bunched towards the centre. Fixed, so every
    // interpolated frame produces the same occlusion.
    for (int i = 0; i < 16; i++) {
        float u1 = (i + 0.5f) / 16.0f;
        uint32_t bits = static_cast<uint32_t>(i);
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        float u2 = static_cast<float>(bits) * 2.3283064365386963e-10f;
        float r = std::sqrt(u1);
        float phi = 6.2831853f * u2;
        float z = std::max(std::sqrt(1.0f - u1), 0.12f); // keep samples off the surface plane
        float t = static_cast<float>(i) / 15.0f;
        float scale = 0.15f + 0.85f * t * t;
        mKernel[i * 3 + 0] = r * std::cos(phi) * scale;
        mKernel[i * 3 + 1] = r * std::sin(phi) * scale;
        mKernel[i * 3 + 2] = z * scale;
    }

    glGenVertexArrays(1, &mVao);
    glGenVertexArrays(1, &mShadowVao);
    glGenBuffers(1, &mShadowVbo);
    glBindVertexArray(mShadowVao);
    glBindBuffer(GL_ARRAY_BUFFER, mShadowVbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);
    glGenQueries(kQueryStages * kQueryRing, &mQueries[0][0]);
    mUsable = true;
    SPDLOG_INFO("NewDawnPostFx: enabled (OpenGL {}.{}, {})", major, minor,
                reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
}

bool PostFxGL::BuildPrograms() {
    const char* ao[] = { kVersion, kDepthLib, kAoShader };
    const char* blur[] = { kVersion, kDepthLib, kBlurLib, kBlurShader };
    const char* apply[] = { kVersion, kDepthLib, kBlurLib, kApplyShader };
    const char* down[] = { kVersion, kDownShader };
    const char* up[] = { kVersion, kUpShader };
    const char* comp[] = { kVersion, kCompositeShader };
    const char* shadowApply[] = { kVersion, kDepthLib, kShadowApplyShader };
    const char* shadowDepthVs[] = { kVersion, kShadowDepthVertex };
    const char* shadowDepthFs[] = { kVersion, kShadowDepthFragment };
    mAo.id = BuildProgram("AO shader", ao, 3, mAo.uniforms);
    mBlur.id = BuildProgram("AO blur shader", blur, 4, mBlur.uniforms);
    mApply.id = BuildProgram("AO apply shader", apply, 4, mApply.uniforms);
    mDown.id = BuildProgram("bloom downsample shader", down, 2, mDown.uniforms);
    mUp.id = BuildProgram("bloom upsample shader", up, 2, mUp.uniforms);
    mComposite.id = BuildProgram("composite shader", comp, 2, mComposite.uniforms);
    mShadowApply.id = BuildProgram("shadow apply shader", shadowApply, 3, mShadowApply.uniforms);
    mShadowDepth.id = BuildProgramWithVertex("shadow depth shader", shadowDepthVs, 2, shadowDepthFs, 2,
                                             mShadowDepth.uniforms);
    return mAo.id && mBlur.id && mApply.id && mDown.id && mUp.id && mComposite.id && mShadowApply.id &&
           mShadowDepth.id;
}

void PostFxGL::EnsureTarget(Target& t, int w, int h, GLint internalFormat, GLenum format, GLenum type,
                               bool depthStencil, bool linear) {
    if (t.tex != 0 && t.w == w && t.h == h) {
        return;
    }
    const bool first = t.tex == 0;
    if (first) {
        glGenTextures(1, &t.tex);
        glGenFramebuffers(1, &t.fbo);
    }
    glActiveTexture(GL_TEXTURE0 + kFirstUnit);
    glBindTexture(GL_TEXTURE_2D, t.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, nullptr);
    if (first) {
        const GLint filter = linear ? GL_LINEAR : GL_NEAREST;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, depthStencil ? GL_DEPTH_STENCIL_ATTACHMENT : GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, t.tex, 0);
    if (depthStencil) {
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
    }
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        SPDLOG_ERROR("NewDawnPostFx: incomplete {}x{} framebuffer (format 0x{:X}), disabling", w, h, internalFormat);
        mUsable = false;
    }
    t.w = w;
    t.h = h;
}

void PostFxGL::Bind(int unit, GLuint tex) {
    glActiveTexture(GL_TEXTURE0 + kFirstUnit + unit);
    glBindTexture(GL_TEXTURE_2D, tex);
}

void PostFxGL::Use(const Program& p) {
    glUseProgram(p.id);
    mCurrentProgram = &p;
}

GLint PostFxGL::Loc(const char* name) {
    auto it = mCurrentProgram->uniforms.find(name);
    return it != mCurrentProgram->uniforms.end() ? it->second : -1; // -1 is ignored by glUniform*
}

void PostFxGL::Draw(GLuint fbo, int w, int h) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, w, h);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

void PostFxGL::Draw(const Target& t) {
    Draw(t.fbo, t.w, t.h);
}

void PostFxGL::CheckError(const char* where) {
    if (!mStats) {
        return;
    }
    for (GLenum e = glGetError(); e != GL_NO_ERROR; e = glGetError()) {
        SPDLOG_ERROR("NewDawnPostFx: GL error 0x{:X} after {}", e, where);
    }
}

void PostFxGL::BeginTimer(int stage) {
    if (!mStats) {
        return;
    }
    int slot = mQueryNext[stage];
    if (mQueryPending[stage][slot]) {
        GLuint64 ns = 0;
        glGetQueryObjectui64v(mQueries[stage][slot], GL_QUERY_RESULT, &ns);
        mQueryPending[stage][slot] = false;
        mQueryTotalMs[stage] += ns / 1e6;
        if (++mQueryCount[stage] == 300) {
            const char* label = (stage == 0) ? "opaque end / AO"
                                : (stage == 1) ? "scene end / bloom+grade"
                                               : "shadows";
            SPDLOG_INFO("NewDawnPostFx: stage {} ({}) averages {:.3f} ms GPU over {} runs", stage, label,
                        mQueryTotalMs[stage] / 300.0, 300);
            mQueryTotalMs[stage] = 0.0;
            mQueryCount[stage] = 0;
        }
    }
    glBeginQuery(GL_TIME_ELAPSED, mQueries[stage][slot]);
}

void PostFxGL::EndTimer(int stage) {
    if (!mStats) {
        return;
    }
    glEndQuery(GL_TIME_ELAPSED);
    mQueryPending[stage][mQueryNext[stage]] = true;
    mQueryNext[stage] = (mQueryNext[stage] + 1) % kQueryRing;
}

void PostFxGL::Run(int stage, const NewDawnPostFxParams& params, const FramebufferOGL& fb, uint32_t frameIndex,
                   const float* shadowPositions, uint32_t shadowVertexCount) {
    if (!mUsable || fb.width < 8 || fb.height < 8) {
        return;
    }
    // Both stages of a frame (and every walk of one game tick's display list is a new frame) share one read
    if (frameIndex != mSettingsFrame) {
        mSettings = ReadSettings();
        mSettingsFrame = frameIndex;
    }
    const Settings& s = mSettings;
    const bool doShadow = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && s.shadowsOn && params.shadowEnabled != 0 &&
                          shadowPositions != nullptr && shadowVertexCount >= 3;
    const bool doAo = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && s.postFxOn && s.aoOn;
    // The AO debug view needs the finished world left alone, otherwise the grade would tint the AO buffer
    const bool doScene = stage == NEWDAWN_POSTFX_STAGE_SCENE_END && s.postFxOn && s.debugView != 2 &&
                         (s.bloomOn || s.gradeOn);
    if (!doShadow && !doAo && !doScene) {
        return;
    }

    mStats = s.stats;
    if (mStats) {
        if (fb.width != mStatsW || fb.height != mStatsH || fb.msaa_level != mStatsMsaa) {
            mStatsW = fb.width;
            mStatsH = fb.height;
            mStatsMsaa = fb.msaa_level;
            SPDLOG_INFO("NewDawnPostFx: game framebuffer {}x{}, msaa {}, invertY {}", fb.width, fb.height, fb.msaa_level,
                        fb.invertY);
        }
        // Drop errors raised by the rest of the renderer so only ours get reported
        while (glGetError() != GL_NO_ERROR) {}
    }

    GlStateGuard guard;
    guard.Neutralise();
    glBindVertexArray(mVao);

    if (doShadow || doAo) {
        PrepareDepth(fb);
    }
    if (doShadow) {
        BeginTimer(kShadowTimerStage);
        RunShadows(params, fb, shadowPositions, shadowVertexCount, s.shadowDebug);
        EndTimer(kShadowTimerStage);
    }
    if (doAo) {
        BeginTimer(0);
        RunAmbientOcclusion(params, fb, s);
        EndTimer(0);
    } else if (doScene) {
        BeginTimer(1);
        RunSceneEnd(fb, s);
        EndTimer(1);
    }

    CheckError(doShadow ? "shadow stage" : (stage == 0 ? "opaque end stage" : "scene end stage"));
}

void PostFxGL::PrepareDepth(const FramebufferOGL& fb) {
    const int w = fb.width, h = fb.height;
    EnsureTarget(mDepth, w, h, GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, true, false);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mDepth.fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    CheckError("depth blit");
}

void PostFxGL::EnsureShadowTarget(int size) {
    if (mShadowMap.tex != 0 && mShadowMap.w == size) {
        return;
    }
    const bool first = mShadowMap.tex == 0;
    if (first) {
        glGenTextures(1, &mShadowMap.tex);
        glGenFramebuffers(1, &mShadowMap.fbo);
    }
    glBindTexture(GL_TEXTURE_2D, mShadowMap.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, size, size, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    const GLfloat border[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);
    glBindFramebuffer(GL_FRAMEBUFFER, mShadowMap.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, mShadowMap.tex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        SPDLOG_ERROR("NewDawnPostFx: incomplete {}x{} shadow framebuffer, disabling", size, size);
        mUsable = false;
    }
    mShadowMap.w = size;
    mShadowMap.h = size;
}

void PostFxGL::RunShadows(const NewDawnPostFxParams& params, const FramebufferOGL& fb, const float* positions,
                          uint32_t vertexCount, int debug) {
    float lightViewProj[16];
    BuildLightViewProj(params, lightViewProj);
    const int size = params.shadowMapSize;
    const int w = fb.width, h = fb.height;
    EnsureShadowTarget(size);
    if (!mUsable) {
        return;
    }

    if (debug != 0) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            const float* P = params.shadowViewProj;
            const float* invP = params.shadowInvViewProj;
            float f[4] = { params.shadowFocus[0], params.shadowFocus[1], params.shadowFocus[2], 1.0f };
            float clip[4];
            for (int j = 0; j < 4; j++) {
                clip[j] = f[0] * P[0 * 4 + j] + f[1] * P[1 * 4 + j] + f[2] * P[2 * 4 + j] + P[3 * 4 + j];
            }
            float ndcX = clip[0] / clip[3];
            float ndcY = clip[1] / clip[3] * params.shadowFlipY;
            int px = std::clamp((int)((ndcX * 0.5f + 0.5f) * w), 0, w - 1);
            int pyA = std::clamp((int)((ndcY * 0.5f + 0.5f) * h), 0, h - 1);
            int pyB = std::clamp(h - 1 - pyA, 0, h - 1);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, mDepth.fbo);
            for (int py : { pyA, pyB }) {
                float depth = 0.0f;
                glReadPixels(px, py, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
                float ndc[4] = { ndcX * params.shadowNdcXScale, (float)py / (float)h * 2.0f - 1.0f,
                                  depth * 2.0f - 1.0f, 1.0f };
                ndc[1] *= params.shadowFlipY;
                float world[4] = {};
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < 4; i++) {
                        world[j] += ndc[i] * invP[i * 4 + j];
                    }
                }
                if (world[3] != 0.0f) {
                    world[0] /= world[3];
                    world[1] /= world[3];
                    world[2] /= world[3];
                }
                float dx = world[0] - f[0];
                float dy = world[1] - f[1];
                float dz = world[2] - f[2];
                SPDLOG_INFO("NewDawn shadows: focus ({:.1f},{:.1f},{:.1f}) -> pixel ({},{}), reconstructed "
                            "({:.1f},{:.1f},{:.1f}), distance {:.1f}",
                            f[0], f[1], f[2], px, py, world[0], world[1], world[2],
                            std::sqrt(dx * dx + dy * dy + dz * dz));
            }
        }
    }

    // Depth-only render of the recorded world-space triangles from the sun.
    glBindFramebuffer(GL_FRAMEBUFFER, mShadowMap.fbo);
    glViewport(0, 0, size, size);
    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    Use(mShadowDepth);
    glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, lightViewProj);
    glBindVertexArray(mShadowVao);
    glBindBuffer(GL_ARRAY_BUFFER, mShadowVbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)vertexCount * 3 * sizeof(float), positions, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)vertexCount);
    CheckError("shadow map render");
    glBindVertexArray(mVao);
    // The shadow map pass enabled depth writes; the full-screen apply must not touch the game depth.
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    // Screen-space shadow apply. Work on a copy of the colour so the game framebuffer is never sampled while bound.
    EnsureTarget(mScene, w, h, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE, false, true);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mScene.fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    CheckError("shadow colour blit");


    Bind(0, mScene.tex);
    Bind(1, mDepth.tex);
    Bind(2, mShadowMap.tex);
    Use(mShadowApply);
    glUniform1i(Loc("uColor"), kFirstUnit + 0);
    glUniform1i(Loc("uDepth"), kFirstUnit + 1);
    glUniform1i(Loc("uShadow"), kFirstUnit + 2);
    glUniform2i(Loc("uSize"), w, h);
    glUniform1f(Loc("uNear"), params.zNear);
    glUniform1f(Loc("uFar"), params.zFar);
    glUniform1f(Loc("uFlipY"), params.shadowFlipY);
    glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
    glUniform2f(Loc("uShadowTexel"), 1.0f / size, 1.0f / size);
    glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_TRUE, params.shadowInvViewProj);
    glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, lightViewProj);
    glUniform3fv(Loc("uSunDir"), 1, params.shadowSunDir);
    glUniform3fv(Loc("uAmbient"), 1, params.shadowAmbient);
    glUniform3fv(Loc("uDirColor"), 1, params.shadowDirColor);
    glUniform3fv(Loc("uCameraEye"), 1, params.shadowCameraEye);
    glUniform1f(Loc("uStrength"), params.shadowStrength);
    glUniform1f(Loc("uMinFactor"), 0.35f);
    glUniform1f(Loc("uNormalBias"), 2.0f * params.shadowRadius / (float)size * 4.0f);
    glUniform1i(Loc("uDebug"), debug);
    Draw(fb.fbo, w, h);
    CheckError("shadow apply");
}

void PostFxGL::RunAmbientOcclusion(const NewDawnPostFxParams& params, const FramebufferOGL& fb, const Settings& s) {
    const int w = fb.width, h = fb.height;
    EnsureTarget(mAoRaw, w, h, GL_R16F, GL_RED, GL_FLOAT, false, false);
    EnsureTarget(mAoBlur, w, h, GL_R16F, GL_RED, GL_FLOAT, false, false);
    if (!mUsable) {
        return;
    }

    const float tanY = std::tan(params.fovyDegrees * 3.14159265f / 360.0f);
    // The game projects at 4:3 and the interpreter rescales clip x for the real aspect ratio, so the horizontal
    // extent follows the target's own aspect.
    const float tanX = tanY * static_cast<float>(w) / static_cast<float>(h);
    const float flipY = fb.invertY ? -1.0f : 1.0f;
    const float maxPx = 0.14f * h;

    // Fog as the interpreter computes it: factor = clamp(ndcZ * mul + offset, 0, 255) / 255. The special cases are
    // those of Gfx_SetFog in the game.
    float fogMul, fogOff;
    if (params.fogNear >= 1000) {
        fogMul = 0.0f;
        fogOff = 0.0f;
    } else if (params.fogNear >= 997) {
        fogMul = 32767.0f;
        fogOff = -32512.0f;
    } else if (params.fogNear < 0) {
        fogMul = 0.0f;
        fogOff = 255.0f;
    } else {
        const int range = std::max(params.fogFar - params.fogNear, 1);
        fogMul = static_cast<float>(static_cast<int16_t>(128000 / range));
        fogOff = static_cast<float>(static_cast<int16_t>((500 - params.fogNear) * 256 / range));
    }

    Bind(0, mDepth.tex);

    // 1. Raw occlusion
    Use(mAo);
    glUniform1i(Loc("uDepth"), kFirstUnit + 0);
    glUniform2i(Loc("uSize"), w, h);
    glUniform1f(Loc("uNear"), params.zNear);
    glUniform1f(Loc("uFar"), params.zFar);
    glUniform2f(Loc("uTan"), tanX, tanY);
    glUniform1f(Loc("uFlipY"), flipY);
    glUniform1f(Loc("uRadius"), s.aoRadius);
    glUniform1f(Loc("uMaxPx"), maxPx);
    glUniform3fv(Loc("uKernel"), 16, mKernel);
    Draw(mAoRaw);
    CheckError("AO pass");

    // 2. Horizontal blur
    Bind(1, mAoRaw.tex);
    Use(mBlur);
    glUniform1i(Loc("uDepth"), kFirstUnit + 0);
    glUniform1i(Loc("uAo"), kFirstUnit + 1);
    glUniform2i(Loc("uSize"), w, h);
    glUniform1f(Loc("uNear"), params.zNear);
    glUniform1f(Loc("uFar"), params.zFar);
    glUniform1f(Loc("uRadius"), s.aoRadius);
    glUniform2i(Loc("uDir"), 1, 0);
    Draw(mAoBlur);
    CheckError("AO blur pass");

    // 3. Vertical blur straight into the game framebuffer
    Bind(1, mAoBlur.tex);
    Use(mApply);
    glUniform1i(Loc("uDepth"), kFirstUnit + 0);
    glUniform1i(Loc("uAo"), kFirstUnit + 1);
    glUniform2i(Loc("uSize"), w, h);
    glUniform1f(Loc("uNear"), params.zNear);
    glUniform1f(Loc("uFar"), params.zFar);
    glUniform1f(Loc("uRadius"), s.aoRadius);
    glUniform2i(Loc("uDir"), 0, 1);
    glUniform1f(Loc("uStrength"), s.aoStrength);
    glUniform1f(Loc("uFogMul"), fogMul);
    glUniform1f(Loc("uFogOff"), fogOff);
    glUniform1i(Loc("uMode"), s.debugView == 2 ? 1 : 0);
    if (s.debugView == 2) {
        // Replace the picture with the AO buffer
    } else {
        glEnable(GL_BLEND);
        glBlendFunc(GL_ZERO, GL_SRC_COLOR); // framebuffer *= AO
        glBlendEquation(GL_FUNC_ADD);
    }
    if (s.debugView == 1) {
        glEnable(GL_SCISSOR_TEST);
        glScissor(w / 2, 0, w - w / 2, h);
    }
    Draw(fb.fbo, w, h);
    CheckError("AO apply pass");
}

void PostFxGL::RunSceneEnd(const FramebufferOGL& fb, const Settings& s) {
    const int w = fb.width, h = fb.height;
    // Linear: the first bloom downsample is a 13-tap kernel that relies on bilinear taps
    EnsureTarget(mScene, w, h, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE, false, true);

    int levels = 0;
    if (s.bloomOn) {
        for (int sz = h / 2; levels < kMaxBloomLevels && sz >= 8; sz /= 2) {
            levels++;
        }
        for (int l = 0; l < levels; l++) {
            EnsureTarget(mBloom[l], std::max(w >> (l + 1), 1), std::max(h >> (l + 1), 1), GL_R11F_G11F_B10F, GL_RGB,
                         GL_FLOAT, false, true);
        }
    }
    if (!mUsable) {
        return;
    }

    // Colour out of the game framebuffer (resolving it when multisampled)
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mScene.fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    CheckError("colour blit");

    const bool useBloom = s.bloomOn && levels > 0;
    if (useBloom) {
        Bind(0, mScene.tex);
        Use(mDown);
        glUniform1i(Loc("uSrc"), kFirstUnit + 0);
        glUniform1f(Loc("uThreshold"), s.bloomThreshold);
        for (int l = 0; l < levels; l++) {
            const Target& src = (l == 0) ? mScene : mBloom[l - 1];
            Bind(0, src.tex);
            glUniform1i(Loc("uPrefilter"), l == 0 ? 1 : 0);
            glUniform2f(Loc("uTexel"), 1.0f / src.w, 1.0f / src.h);
            Draw(mBloom[l]);
        }
        CheckError("bloom downsample");

        Use(mUp);
        glUniform1i(Loc("uSrc"), kFirstUnit + 0);
        glUniform1f(Loc("uWeight"), 0.6f);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glBlendEquation(GL_FUNC_ADD);
        for (int l = levels - 2; l >= 0; l--) {
            Bind(0, mBloom[l + 1].tex);
            glUniform2f(Loc("uTexel"), 1.0f / mBloom[l + 1].w, 1.0f / mBloom[l + 1].h);
            Draw(mBloom[l]);
        }
        glDisable(GL_BLEND);
        CheckError("bloom upsample");
    }

    Bind(0, mScene.tex);
    Bind(1, useBloom ? mBloom[0].tex : mScene.tex);
    Use(mComposite);
    glUniform1i(Loc("uScene"), kFirstUnit + 0);
    glUniform1i(Loc("uBloom"), kFirstUnit + 1);
    glUniform1i(Loc("uUseBloom"), useBloom ? 1 : 0);
    glUniform1f(Loc("uBloomStrength"), s.bloomStrength);
    glUniform1i(Loc("uUseGrade"), s.gradeOn ? 1 : 0);
    glUniform1f(Loc("uContrast"), s.contrast);
    glUniform1f(Loc("uSaturation"), s.saturation);
    if (s.debugView == 1) {
        glEnable(GL_SCISSOR_TEST);
        glScissor(w / 2, 0, w - w / 2, h);
    }
    Draw(fb.fbo, w, h);
    CheckError("composite pass");
}

} // namespace Fast
#endif
