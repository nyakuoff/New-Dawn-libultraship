#if defined(ENABLE_OPENGL) && !defined(USE_OPENGLES)

#include "fast/backends/gfx_opengl_postfx.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <string>
#include <vector>

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
    s.shadowFloor = std::clamp(CvFloat("gNewDawn.Shadows.Floor", 0.55f), 0.0f, 1.0f);
    s.aoOn = CvInt("gNewDawn.PostFx.AO.Enabled", 1) != 0;
    s.aoStrength = std::clamp(CvFloat("gNewDawn.PostFx.AO.Strength", Defaults::kAoStrength), 0.0f, 4.0f);
    s.aoRadius = std::clamp(CvFloat("gNewDawn.PostFx.AO.Radius", Defaults::kAoRadius), 1.0f, 500.0f);
    s.bloomOn = CvInt("gNewDawn.PostFx.Bloom.Enabled", 1) != 0;
    s.bloomStrength = std::clamp(CvFloat("gNewDawn.PostFx.Bloom.Strength", Defaults::kBloomStrength), 0.0f, 2.0f);
    s.bloomThreshold = std::clamp(CvFloat("gNewDawn.PostFx.Bloom.Threshold", Defaults::kBloomThreshold), 0.0f, 16.0f);
    s.gradeOn = CvInt("gNewDawn.PostFx.Grade.Enabled", 1) != 0;
    s.contrast = std::clamp(CvFloat("gNewDawn.PostFx.Grade.Contrast", Defaults::kContrast), 0.5f, 1.5f);
    s.saturation = std::clamp(CvFloat("gNewDawn.PostFx.Grade.Saturation", Defaults::kSaturation), 0.0f, 2.0f);
    s.debugView = CvInt("gNewDawn.PostFx.Debug.View", 0);
    s.stats = CvInt("gNewDawn.PostFx.Debug.Stats", 0) != 0;
    return s;
}

static float DecodeColor(float c) {
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// Single scattering in a spherical atmosphere. Distances in km; Rayleigh coefficients at sea level,
// exponential 8 km / 1.2 km scale heights and a Cornette-Shanks aerosol phase. Original implementation.
// No sun disc here: Environment_Draw retains the game's sun, stars and Moon in their original order.
static const char* kSkyPaletteLib = R"GLSL(
uniform vec3 uPaletteColor;
vec3 skyPaletteTint(vec3 radiance, float sunHeight) {
    const vec3 luma = vec3(0.2126, 0.7152, 0.0722);
    float twilight = (1.0-smoothstep(0.12, 0.42, abs(sunHeight))) * smoothstep(-0.15, 0.025, sunHeight);
    float y = dot(radiance, luma);
    vec3 tint = uPaletteColor / max(dot(uPaletteColor, luma), 0.001);
    // Retain scattering luminance/gradient and the bright Mie aureole; restore Termina's dusk/dawn mood.
    float weight = 0.82 * twilight * (1.0-smoothstep(0.8, 3.0, y));
    return mix(radiance, tint * y, weight);
}
)GLSL";

static const char* kPhysicalSkyShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform mat4 uInvViewProj;
uniform float uNdcXScale;
uniform float uFlipY;
uniform vec3 uSunDir;
uniform vec3 uFog;
uniform float uWeather;
uniform float uOvercast;
uniform float uIntensity;
const float PI = 3.14159265;
const float Rg = 6360.0;
const float Ra = 6440.0;
const vec3 betaR = vec3(0.005802, 0.013558, 0.033100);
const float betaM = 0.004440;
float exitDistance(vec3 p, vec3 d) {
    float b = dot(p, d);
    return -b + sqrt(max(b*b - dot(p,p) + Ra*Ra, 0.0));
}
vec2 density(vec3 p) {
    float h = max(length(p) - Rg, 0.0);
    return exp(-h / vec2(8.0, 1.2));
}
vec3 sky(vec3 ray) {
    vec3 origin = vec3(0.0, Rg + 0.2, 0.0);
    float distance = exitDistance(origin, ray);
    vec2 optical = vec2(0.0);
    vec3 sumR = vec3(0.0), sumM = vec3(0.0);
    float aerosol = mix(mix(1.0, 3.0, uWeather), 10.0, uOvercast);
    for (int i = 0; i < 12; ++i) {
        float t0 = pow(float(i) / 12.0, 2.0) * distance;
        float t1 = pow(float(i+1) / 12.0, 2.0) * distance;
        float ds = t1-t0;
        vec3 p = origin + ray * (t0+t1) * 0.5;
        vec2 local = density(p) * ds;
        optical += local * 0.5;
        float sunDistance = exitDistance(p, uSunDir);
        vec2 sunOptical = vec2(0.0);
        // Planet shadow, smoothed over the solar disc at sunset.
        float mu = dot(normalize(p), uSunDir);
        float horizon = -sqrt(max(1.0 - Rg*Rg / dot(p,p), 0.0));
        float visible = smoothstep(horizon - 0.005, horizon + 0.005, mu);
        for (int j = 0; j < 4; ++j) {
            // Concentrate quadrature near the sample: uniform steps miss dense air at low sun angles.
            float s0 = pow(float(j)/4.0, 2.0) * sunDistance;
            float s1 = pow(float(j+1)/4.0, 2.0) * sunDistance;
            sunOptical += density(p + uSunDir * (s0+s1)*0.5) * (s1-s0);
        }
        vec2 total = optical + sunOptical;
        vec3 transmission = exp(-betaR * total.x - betaM * aerosol * total.y);
        sumR += transmission * local.x * visible;
        sumM += transmission * local.y * visible;
        optical += local * 0.5;
    }
    float c = dot(ray, uSunDir);
    float phaseR = 3.0 / (16.0 * PI) * (1.0 + c*c);
    float g = 0.76;
    float phaseM = 3.0 / (8.0 * PI) * ((1.0-g*g)*(1.0+c*c)) /
                   ((2.0+g*g)*pow(max(1.0+g*g-2.0*g*c, 0.001), 1.5));
    vec3 radiance = 22.0 * (sumR * betaR * phaseR + sumM * betaM * 0.9 * aerosol * phaseM);
    float daylight = smoothstep(-0.12, 0.12, uSunDir.y);
    // Bounded multiple-scattering fill and starlight; the game still draws the visible stars afterwards.
    radiance += vec3(0.018, 0.035, 0.065) * daylight + vec3(0.0015, 0.0025, 0.006) * (1.0-daylight);
    // Overcast (winter snow sky): low contrast, mostly the fog's own hue, so the deck meets the snow fog.
    float haze = max(uWeather * 0.65, uOvercast * 0.8);
    radiance = mix(radiance, mix(uFog, vec3(dot(radiance, vec3(0.2126,0.7152,0.0722))), 0.65), haze);
    return skyPaletteTint(radiance, uSunDir.y) * uIntensity;
}
void main() {
#ifdef SKY_CAPTURE
    vec3 ray = cubeDirection(vUv);
#else
    vec2 ndc = (vUv * 2.0 - 1.0) * vec2(uNdcXScale, uFlipY);
    vec4 a = uInvViewProj * vec4(ndc, -1.0, 1.0);
    vec4 b = uInvViewProj * vec4(ndc, 0.999, 1.0);
    vec3 ray = normalize(b.xyz/b.w - a.xyz/a.w);
#endif
    vec3 rayAbove = normalize(vec3(ray.x, max(ray.y, 0.005), ray.z));
    // Match the game's fog radiance exactly at/below the horizon, preserving distant silhouettes.
    vec3 color = mix(uFog, sky(rayAbove), smoothstep(0.0, mix(0.05, 0.45, uOvercast), ray.y));
#ifdef SKY_CAPTURE
    color = cubeFinite(color);
#endif
    oColor = vec4(color, 1.0);
}
)GLSL";

static const char* kSkyApplyShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uSky;
uniform float uStrength;
void main() { oColor = vec4(texture(uSky, vUv).rgb, uStrength); }
)GLSL";

// A world-anchored cumulus layer, with Beer-Lambert view/light transmittance and a low-frequency
// coverage map. Fixed samples + filtered noise + spatial upscale avoid a history buffer and its cut/occlusion ghosts.
static const char* kCloudMarchShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler3D uNoise;
uniform mat4 uInvViewProj;
uniform float uNdcXScale;
uniform float uFlipY;
uniform vec3 uSunDir;
uniform float uWeather;
uniform float uOvercast;
uniform vec3 uFog;
uniform float uCoverage;
uniform vec2 uWind;
uniform float uIntensity;
#ifdef SKY_CAPTURE
uniform float uCloudStrength;
#endif
float noise3(vec3 p, float lod) {
    return textureLod(uNoise, p / 64.0, lod).r;
}
float density(vec3 p, float lod) {
    float h = (p.y - 1.6) / 1.8;
    if (h <= 0.0 || h >= 1.0) return 0.0;
    p.xz += uWind;
    // Weather cells span tens of km; the density octaves span km down to hundreds of metres.
    float weather = noise3(vec3(p.x / 16.0, 3.0, p.z / 16.0), 0.0);
    float cover = clamp(uCoverage + (weather - 0.5) * 0.65 + uWeather * 0.25, 0.0, 1.0);
    float shape = noise3(p, lod) * 0.55 + noise3(p * 2.0, lod+1.0) * 0.25 +
                  noise3(p * 4.0, lod+2.0) * 0.14 + noise3(p * 8.0, lod+3.0) * 0.06;
    float profile = smoothstep(0.0, 0.15, h) * (1.0 - smoothstep(0.45, 1.0, h));
    float threshold = 0.72 - cover * 0.48 + (1.0 - profile) * 0.22;
    return smoothstep(threshold, threshold + 0.10, shape) * profile;
}
float lightDensity(vec3 p) {
    float h = (p.y - 1.6) / 1.8;
    if (h <= 0.0 || h >= 1.0) return 0.0;
    p.xz += uWind;
    float weather = noise3(vec3(p.x / 16.0, 3.0, p.z / 16.0), 0.0);
    float cover = clamp(uCoverage + (weather - 0.5) * 0.65 + uWeather * 0.25, 0.0, 1.0);
    float profile = smoothstep(0.0, 0.15, h) * (1.0 - smoothstep(0.45, 1.0, h));
    float shape = noise3(p, 0.0)*0.70 + noise3(p*2.0, 0.0)*0.30;
    return smoothstep(0.72 - cover * 0.48 + (1.0-profile)*0.22, 0.82 - cover * 0.48 + (1.0-profile)*0.22,
                      shape) * profile;
}
void main() {
    oColor = vec4(0.0);
    if (uCoverage <= 0.0) return;
#ifdef SKY_CAPTURE
    vec3 ray = cubeDirection(vUv);
    // Cube captures include the lower hemisphere; cloud slabs exist only above the horizon.
    // A negative march length there can overflow extinction and contaminate rough specular mips with NaNs.
    if (ray.y <= 0.025) return;
    vec3 origin = uCubeOrigin * 0.00002;
#else
    vec2 ndc = (vUv * 2.0 - 1.0) * vec2(uNdcXScale, uFlipY);
    vec4 a = uInvViewProj * vec4(ndc, -1.0, 1.0);
    vec4 b = uInvViewProj * vec4(ndc, 0.999, 1.0);
    vec3 ray = normalize(b.xyz/b.w - a.xyz/a.w);
    if (ray.y <= 0.025) return;
    // Game units -> km. Using the near point also includes exact interpolated camera translation.
    vec3 origin = a.xyz/a.w * 0.00002;
#endif
    origin.y = clamp(origin.y, -0.5, 1.2);
    float start = max((1.6-origin.y)/ray.y, 0.0);
    float end = min((3.4-origin.y)/ray.y, start + 45.0);
    float ds = (end-start)/40.0;
    float lod = clamp(log2(max(ds * 0.75, 0.001)), -4.0, 3.0);
    // A spatial, world-direction-anchored offset breaks up marching bands without frame-varying noise.
    float jitter = noise3(ray * 512.0 + origin * 0.125, 0.0);
    float day = smoothstep(-0.10, 0.06, uSunDir.y);
    vec3 light = day > 0.01 ? uSunDir : -uSunDir;
    light.y = max(light.y, 0.04);
    light = normalize(light);
    vec3 sunColor = exp(-vec3(0.005802,0.013558,0.033100) * 8.0 / max(uSunDir.y, 0.035));
    sunColor = mix(vec3(0.018,0.026,0.045), sunColor * 1.7, day);
    float c = max(dot(ray, light), 0.0);
    float phase = 0.35 + 0.65 * pow(c, 8.0); // broad forward lobe, bounded for HDR silver linings
    float trans = 1.0;
    vec3 color = vec3(0.0);
    for (int i = 0; i < 40; ++i) {
        vec3 p = origin + ray * (start + (float(i)+jitter)*ds);
        float d = density(p, lod);
        if (d > 0.002) {
            float tau = lightDensity(p + light*0.25)*0.45 + lightDensity(p + light*0.85)*0.85 +
                        lightDensity(p + light*1.8)*1.2;
            float sunTrans = exp(-tau * 2.2);
            float h = clamp((p.y-1.6)/1.8, 0.0, 1.0);
            vec3 ambient = mix(vec3(0.085,0.11,0.16), vec3(0.20,0.25,0.33), h) * day +
                           vec3(0.0015,0.0025,0.005) * (1.0-day);
            vec3 lighting = ambient + sunColor * phase * sunTrans * (0.8 + 0.2*(1.0-exp(-d*2.0)));
            // Overcast snow deck: a bright, low-contrast layer lit by the diffuse fog colour. Its base is a
            // little darker, and thick cores shade softly instead of blackening.
            vec3 deck = uFog * (0.68 + 0.30*h) * (0.74 + 0.26*sunTrans) * (1.0 - 0.22*smoothstep(0.25, 1.0, d));
            lighting = mix(lighting, deck, uOvercast); // fog colour already carries the time of day
            float opacity = 1.0-exp(-d*ds*2.8);
            color += trans * opacity * lighting;
            trans *= 1.0-opacity;
            if (trans < 0.015) break;
        }
    }
    float horizon = smoothstep(0.025, mix(0.12, 0.45, uOvercast), ray.y);
    oColor = vec4(skyPaletteTint(color, uSunDir.y) * uIntensity, 1.0-trans) * horizon;
#ifdef SKY_CAPTURE
    oColor *= uCloudStrength;
    if (any(isnan(oColor)) || any(isinf(oColor))) oColor = vec4(0.0);
    oColor = max(oColor, vec4(0.0));
#endif
}
)GLSL";

static const char* kCloudApplyShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uClouds;
uniform float uStrength;
void main() {
    vec2 texel = 0.75 / vec2(textureSize(uClouds, 0));
    vec4 c = texture(uClouds, vUv) * 0.5;
    c += texture(uClouds, vUv + texel * vec2(1.0,1.0)) * 0.125;
    c += texture(uClouds, vUv + texel * vec2(-1.0,1.0)) * 0.125;
    c += texture(uClouds, vUv + texel * vec2(1.0,-1.0)) * 0.125;
    c += texture(uClouds, vUv - texel) * 0.125;
    oColor = c * uStrength;
}
)GLSL";

static const char* kSkyCubeDirectionLib = R"GLSL(
uniform int uCubeFace;
uniform vec3 uCubeOrigin;
// A single invalid cached texel would otherwise contaminate every surface for eight frames.
vec3 cubeFinite(vec3 c) {
    if (any(isnan(c)) || any(isinf(c))) c = vec3(0.0);
    return max(c,vec3(0.0));
}
vec3 cubeDirection(vec2 uv) {
    vec2 p = uv*2.0-1.0;
    if (uCubeFace == 0) return normalize(vec3(1.0,-p.y,-p.x));
    if (uCubeFace == 1) return normalize(vec3(-1.0,-p.y,p.x));
    if (uCubeFace == 2) return normalize(vec3(p.x,1.0,p.y));
    if (uCubeFace == 3) return normalize(vec3(p.x,-1.0,-p.y));
    if (uCubeFace == 4) return normalize(vec3(p.x,-p.y,1.0));
    return normalize(vec3(-p.x,-p.y,-1.0));
}
)GLSL";

static const char* kSkyConvolveShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform samplerCube uEnvironment;
uniform vec3 uGroundAlbedo;
void main() {
    const vec3 luma = vec3(0.2126,0.7152,0.0722);
    vec3 N = cubeDirection(vUv);
    vec3 irradiance = vec3(0.0);
    float upperReference = 0.0;
    // Fixed equal-area sphere quadrature, E/pi. Lower directions reflect the actual cloudy sky
    // through representative diffuse ground albedo; no artificial light is introduced at night.
    for (int i=0; i<64; ++i) {
        float y = 1.0-2.0*(float(i)+0.5)/64.0;
        float r = sqrt(max(1.0-y*y,0.0));
        float phi = float(i)*2.39996323;
        vec3 d = vec3(r*cos(phi),y,r*sin(phi));
        vec3 radiance = textureLod(uEnvironment, vec3(d.x,abs(d.y),d.z),0.0).rgb;
        upperReference += dot(radiance,luma)*max(d.y,0.0)*(4.0/64.0);
        if (d.y<0.0) radiance *= uGroundAlbedo;
        irradiance += radiance * max(dot(N,d),0.0) * (4.0/64.0);
    }
    float y = dot(irradiance,luma);
    vec3 chroma = y>1e-6 ? irradiance/y : vec3(1.0);
    // Calibrate to the existing ambient energy rather than changing exposure/night visibility.
    // Preserve at least 95% of baseline ambient luminance in every direction, allow 20% overhead fill.
    float gain = clamp(y/max(upperReference*0.65,1e-6),0.95,1.20);
    oColor = vec4(cubeFinite(mix(vec3(1.0),chroma,0.65)*gain),1.0);
}
)GLSL";

// Deterministic GGX prefilter. The base mip retains cloud shape; the other four soften with roughness.
static const char* kSkyPrefilterShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform samplerCube uEnvironment;
uniform float uRoughness;
void main() {
    vec3 N = cubeDirection(vUv);
    if (uRoughness < 0.001) { oColor = vec4(cubeFinite(textureLod(uEnvironment,N,0.0).rgb),1.0); return; }
    vec3 up = abs(N.y)<0.99 ? vec3(0,1,0) : vec3(1,0,0);
    vec3 T = normalize(cross(up,N)), B = cross(N,T);
    vec3 sum = vec3(0.0); float weight = 0.0;
    float a = uRoughness*uRoughness;
    for (int i=0; i<32; ++i) {
        float x = float(i)/32.0;
        // Radical inverse for this fixed 32-sample Hammersley sequence, compatible with GLSL 1.30.
        float y=0.0, f=0.5; int bits=i;
        for (int b=0; b<5; ++b) { y+=float(bits%2)*f; bits/=2; f*=0.5; }
        float phi=6.2831853*x;
        float ct=sqrt((1.0-y)/(1.0+(a*a-1.0)*y));
        float st=sqrt(max(1.0-ct*ct,0.0));
        vec3 H=T*(cos(phi)*st)+B*(sin(phi)*st)+N*ct;
        vec3 L=reflect(-N,H);
        float nl=max(dot(N,L),0.0);
        // Sky/ground seam stays tied to the game's horizon, rather than reflecting a bright lower dome.
        sum+=textureLod(uEnvironment,L,0.0).rgb*nl; weight+=nl;
    }
    oColor=vec4(cubeFinite(sum/max(weight,1e-5)),1.0);
}
)GLSL";

// Keep the ambient transfer in step with newDawnIrradiance in both packaged default shaders.
static const char* kSkyAmbientLightingLib = R"GLSL(
uniform bool uSkyAmbientEnabled;
uniform samplerCube uSkyIrradiance;
vec3 skyAmbientColor(vec3 baseline, vec3 N) {
    if (!uSkyAmbientEnabled) return baseline;
    float y = dot(baseline,vec3(0.2126,0.7152,0.0722));
    return texture(uSkyIrradiance,N).rgb*y;
}
)GLSL";

static void FogCoefficients(const NewDawnPostFxParams& params, float& fogMul, float& fogOff) {
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
#ifdef HDR_SCENE
const int uHdr = 1;
#else
const int uHdr = 0;
#endif

vec3 pre(vec3 c) {
    if (uPrefilter == 0) {
        return c;
    }
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float k = uHdr != 0 ? 0.25 * max(uThreshold, 0.01) : max(0.25 * (1.0 - uThreshold), 1e-3);
    float soft = clamp(l - uThreshold + k, 0.0, 2.0 * k);
    soft = soft * soft / (4.0 * k);
    float ex = max(soft, l - uThreshold);
    return c * (ex / (uHdr != 0 ? 1.0 : 1.0 - uThreshold)) / max(l, 1e-4);
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
#ifdef HDR_SCENE
const int uHdr = 1;
#else
const int uHdr = 0;
#endif
uniform float uExposure;
uniform int uDither;

float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

void main() {
    vec3 c = texelFetch(uScene, ivec2(gl_FragCoord.xy), 0).rgb;
    if (uUseBloom != 0) {
        vec3 b = texture(uBloom, vUv).rgb * uBloomStrength;
        if (uHdr != 0) c += b; // scene-referred radiance, before the display transform
        else {
            b = clamp(b, 0.0, 1.0);
            c = 1.0 - (1.0 - c) * (1.0 - b);
        }
    }
    if (uHdr != 0) {
        vec3 x = max(c, vec3(0.0)) * exp2(uExposure);
        // Hue-preserving highlight roll-off: identity below 0.76 (base colours and shadows keep their saturation and
        // level), above it the peak channel is compressed and the colour eases towards white. This is the compression
        // half of the Khronos PBR Neutral operator; its dark-end offset is omitted because it strips the minimum
        // channel and over-saturates shadows.
        float pk = max(x.r, max(x.g, x.b));
        if (pk >= 0.76) {
            const float d = 0.24;
            float np = 1.0 - d * d / (pk + d - 0.76);
            x *= np / pk;
            x = mix(x, vec3(np), 1.0 - 1.0 / (0.15 * (pk - np) + 1.0));
        }
        c = clamp(x, 0.0, 1.0);
        c = mix(1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, 12.92 * c, lessThanEqual(c, vec3(0.0031308)));
    }
    if (uUseGrade != 0) {
        // HDR path only: gentle split tone (cool shadows, warm highlights) for depth. Multiplicative and luminance
        // keyed, so black stays black and the +-3% tint cannot shift hues noticeably.
        if (uHdr != 0) {
            float t = smoothstep(0.2, 0.8, dot(c, vec3(0.2126, 0.7152, 0.0722)));
            c *= mix(vec3(0.97, 1.0, 1.03), vec3(1.03, 1.0, 0.97), t);
        }
        // Contrast as a blend towards a smoothstep S-curve (fixes black and white, mid slope equals uContrast).
        vec3 s = c * c * (3.0 - 2.0 * c);
        c = c + (uContrast - 1.0) * 2.0 * (s - c);
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c = mix(vec3(l), c, uSaturation);
    }
    // Dither before the 8 bit quantisation. +-half a step leaves unchanged pixels (and pure black) exactly as they were.
    #ifndef NO_DITHER
#ifdef HDR_SCENE
    if (uDither != 0)
#endif
        c += (ign(gl_FragCoord.xy) - 0.5) / 255.0;
#endif
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

// Alpha-tested casters (tree leaves, fences): same 0.19 cutout threshold as the main draw's texture-edge path.
static const char* kShadowFoliageVertex = R"GLSL(
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUv;
uniform mat4 uLightViewProj;
out vec2 vUv;
void main() {
    vUv = aUv;
    gl_Position = uLightViewProj * vec4(aPos, 1.0);
}
)GLSL";

static const char* kShadowFoliageFragment = R"GLSL(
in vec2 vUv;
uniform sampler2D uTex;
void main() {
    if (texture(uTex, vUv).a <= 0.19) discard;
}
)GLSL";

static const char* kShadowApplyShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uColor;
uniform sampler2D uShadow;       // raw depth, for the debug views
uniform sampler2DShadow uShadowCmp; // same texture, bilinear-filtered depth compare
uniform vec2 uShadowTexel;
uniform mat4 uInvViewProj;
uniform mat4 uLightViewProj;
uniform vec3 uSunDir;
uniform vec3 uAmbient;
uniform vec3 uDirColor;
uniform vec3 uCameraEye;
uniform float uStrength;
uniform float uMinFactor;
uniform float uRadius;
#ifdef HDR_SCENE
const int uHdr = 1;
#else
const int uHdr = 0;
#endif
uniform float uFogMul;
uniform float uFogOff;
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

float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

const vec2 kPoisson[12] = vec2[12](
    vec2(-0.326212, -0.405810), vec2(-0.840144, -0.073580), vec2(-0.695914, 0.457137), vec2(-0.203345, 0.620716),
    vec2(0.962340, -0.194983), vec2(0.473434, -0.480026), vec2(0.519456, 0.767022), vec2(0.185461, -0.893124),
    vec2(0.507431, 0.064425), vec2(0.896420, 0.412458), vec2(-0.321940, -0.932615), vec2(-0.791559, -0.597710));

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (uDebug == 3) {
        vec3 color = texture(uColor, vUv).rgb;
        const vec2 mapMin = vec2(0.015, 0.68);
        const vec2 mapSize = vec2(0.30, 0.30);
        if (all(greaterThanEqual(vUv, mapMin)) && all(lessThan(vUv, mapMin + mapSize))) {
            vec2 mapUv = (vUv - mapMin) / mapSize;
            float depth = texture(uShadow, mapUv).r;
            // Show caster coverage as a high-contrast silhouette; clear texels stay white.
            color = (depth < 0.999999) ? vec3(0.04, 0.05, 0.06) : vec3(0.96);
        }
        oColor = vec4(color, 1.0);
        return;
    }
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
    bool inMap = ref < 1.0 && all(greaterThanEqual(suv, vec2(0.0))) && all(lessThanEqual(suv, vec2(1.0)));
    float sampledLit = 1.0;
    if (inMap) {
        float bias = 0.003 + 0.010 * (1.0 - ndl);
        // Rotated Poisson disk (angle from interleaved gradient noise, fixed per screen pixel so it never crawls);
        // every tap is itself a bilinear 2x2 compare, which removes the stair steps of the map texels.
        float a = ign(gl_FragCoord.xy) * 6.2831853;
        vec2 rot = vec2(cos(a), sin(a));
        float sum = 0.0;
        for (int i = 0; i < 12; i++) {
            vec2 o = kPoisson[i];
            o = vec2(o.x * rot.x - o.y * rot.y, o.x * rot.y + o.y * rot.x) * 2.2;
            sum += texture(uShadowCmp, vec3(suv + o * uShadowTexel, ref - bias));
        }
        sampledLit = sum / 12.0;
    }
    if (uDebug == 4) {
        oColor = vec4(ndl, 1.0 - sampledLit, inMap ? 1.0 : 0.0, 1.0);
        return;
    }
    if (ndl > 0.0 && inMap) {
        lit = sampledLit;
    }
    if (uDebug == 2) {
        oColor = vec4(vec3(lit), 1.0);
        return;
    }

    vec3 direct = uDirColor * ndl;
    vec3 ambient = skyAmbientColor(uAmbient, N);
    vec3 total = max(ambient + direct, vec3(1e-4));
    vec3 shade = ambient + direct * lit;
    vec3 factor = clamp(shade / total, vec3(uMinFactor), vec3(1.0));
    factor = mix(vec3(1.0), factor, uStrength);
    if (uHdr != 0) {
        // Fog has already been composited into scene colour. Do not shadow its in-scatter:
        // otherwise geometry hidden by full fog becomes a dark silhouette against the sky.
        float fog = clamp((d0 * 2.0 - 1.0) * uFogMul + uFogOff, 0.0, 255.0) / 255.0;
        float coverage = 1.0 - smoothstep(uRadius * 0.8, uRadius, length(P - uCameraEye));
        factor = mix(vec3(1.0), factor, coverage * (1.0 - fog));
    }
    oColor = vec4(texture(uColor, vUv).rgb * factor, 1.0);
}
)GLSL";

// ---------------------------------------------------------------------------------------------------------------------
// Sun shafts. Half resolution raymarch from the camera to each pixel's world position through the shadow map (the
// sample count is fixed, the start of every ray is jittered with interleaved gradient noise), blurred depth-aware and
// applied with a screen blend. R = in-scatter amount (phase included), G = linear depth of the pixel it was taken at.
// ---------------------------------------------------------------------------------------------------------------------
static const char* kShaftMarchShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2DShadow uShadowCmp;
uniform mat4 uInvViewProj;
uniform mat4 uLightViewProj;
uniform vec3 uSunDir;
uniform vec3 uCameraEye;
uniform float uNdcXScale;
uniform float uFlipY;
uniform float uMaxDist;
uniform float uDensity;
uniform float uAnisotropy;

float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

vec3 worldPos(ivec2 p, float d) {
    vec2 ndc = (vec2(p) + 0.5) / vec2(uSize) * 2.0 - 1.0;
    ndc.y *= uFlipY;
    ndc.x *= uNdcXScale;
    vec4 w = uInvViewProj * vec4(ndc.x, ndc.y, d * 2.0 - 1.0, 1.0);
    return w.xyz / w.w;
}

const int kSteps = 20;

void main() {
    ivec2 hp = ivec2(gl_FragCoord.xy);
    ivec2 fp = min(hp * 2, uSize - 1);
    float d = fetchDepth(fp);
    bool sky = d >= 0.99999;
    vec3 P = worldPos(fp, sky ? 0.9999 : d);
    vec3 ray = P - uCameraEye;
    float len = length(ray);
    vec3 dir = ray / max(len, 1e-3);
    float L = sky ? uMaxDist : min(len, uMaxDist);

    // Henyey-Greenstein, normalised so that looking straight at the sun gives 1; a little isotropic floor so the
    // shafts do not vanish completely when the sun is to the side.
    float g = uAnisotropy;
    float cosT = dot(dir, uSunDir);
    float hg = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosT, 1.5);
    float hgMax = (1.0 - g * g) / pow((1.0 - g) * (1.0 - g), 1.5);
    float phase = 0.04 + 0.96 * (hg / hgMax);

    float jitter = ign(gl_FragCoord.xy);
    float stepLen = L / float(kSteps);
    float acc = 0.0;
    float wsum = 0.0;
    for (int i = 0; i < kSteps; i++) {
        float t = (float(i) + jitter) * stepLen;
        vec3 pos = uCameraEye + dir * t;
        vec4 lc = uLightViewProj * vec4(pos, 1.0);
        vec3 ndc = lc.xyz / lc.w;
        vec2 suv = ndc.xy * 0.5 + 0.5;
        float ref = ndc.z * 0.5 + 0.5;
        float lit = 1.0;
        if (ref < 1.0 && all(greaterThanEqual(suv, vec2(0.0))) && all(lessThanEqual(suv, vec2(1.0)))) {
            lit = texture(uShadowCmp, vec3(suv, ref - 0.004));
        }
        float w = exp(-uDensity * t);
        acc += lit * w;
        wsum += w;
    }
    float opacity = 1.0 - exp(-uDensity * L);
    // Only the well-lit part of the ray counts: partly shadowed rays fall off quickly, which is what gives the shafts shape.
    float lit = clamp((acc / max(wsum, 1e-4) - 0.3) / 0.7, 0.0, 1.0);
    float v = lit * opacity * phase;
    oColor = vec4(v, sky ? uFar : linDepth(d), 0.0, 1.0);
}
)GLSL";

// Depth-aware separable blur at half resolution; the jitter noise averages away.
static const char* kShaftBlurShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uSrc;
uniform ivec2 uDir;
uniform ivec2 uHalfSize;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 c = texelFetch(uSrc, p, 0).rg;
    float sigma = max(0.06 * c.y, 20.0);
    float sum = 0.0, wsum = 0.0;
    for (int i = -3; i <= 3; i++) {
        vec2 s = texelFetch(uSrc, clamp(p + uDir * i, ivec2(0), uHalfSize - 1), 0).rg;
        float dz = (s.y - c.y) / sigma;
        float w = exp(-dz * dz) * exp(-0.08 * float(i * i));
        sum += s.x * w;
        wsum += w;
    }
    oColor = vec4(sum / wsum, c.y, 0.0, 1.0);
}
)GLSL";

// Depth-aware bilinear upsample straight into the game framebuffer. Screen blend: 1 - (1 - c)(1 - s).
static const char* kShaftApplyShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uSrc;
uniform ivec2 uHalfSize;
uniform vec3 uColor;
uniform float uStrength;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float d = fetchDepth(p);
    float lz = (d >= 0.99999) ? uFar : linDepth(d);
    vec2 hp = (vec2(p) + 0.5) * 0.5 - 0.5;
    ivec2 b = ivec2(floor(hp));
    vec2 f = hp - vec2(b);
    float sum = 0.0, wsum = 0.0;
    float sigma = max(0.06 * lz, 20.0);
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            ivec2 q = clamp(b + ivec2(i, j), ivec2(0), uHalfSize - 1);
            vec2 s = texelFetch(uSrc, q, 0).rg;
            float dz = (s.y - lz) / sigma;
            float w = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y) * (exp(-dz * dz) + 1e-3);
            sum += s.x * w;
            wsum += w;
        }
    }
    float v = sum / wsum;
    oColor = vec4(uColor * (v * uStrength * 4.0), 1.0); // 4.0: so that a strength of 0.5 is "subtle but visible" 
}
)GLSL";


// Height fog: single-scattering Beer-Lambert integration. Half resolution, fixed spatial jitter,
// bilateral reconstruction, no temporal history. Native fog is retained and excludes our overlay at full extinction.
static const char* kFogMarchShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2DShadow uShadowCmp;
uniform mat4 uInvViewProj, uLightViewProj;
uniform vec3 uEye, uSunDir, uSunColor, uAmbient, uFogColor;
uniform float uNdcXScale, uFlipY, uDensity, uBaseHeight, uHeight, uMaxDist;
vec3 fogWorld(ivec2 p, float d) {
    vec2 ndc = (vec2(p) + 0.5) / vec2(uSize) * 2.0 - 1.0;
    vec4 w = uInvViewProj * vec4(ndc * vec2(uNdcXScale, uFlipY), d * 2.0 - 1.0, 1.0);
    return w.xyz / w.w;
}
void main() {
    ivec2 p = min(ivec2(gl_FragCoord.xy) * 2, uSize - 1);
    float d = fetchDepth(p);
    if (d >= 0.99999) { oColor = vec4(0.0); return; }
    vec3 ray = fogWorld(p,d) - uEye;
    float len = length(ray);
    vec3 dir = ray / max(len, 1e-3);
    float L = min(len,uMaxDist);
    const int steps = 24;
    float ds = L / float(steps);
    // Interleaved gradient noise is fixed across frames: no animation-dependent grain or history on scene cuts.
    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy,vec2(0.06711056,0.00583715))));
    const float g = 0.6;
    float phase = (1.0-g*g)/(12.566371*pow(max(1.0+g*g-2.0*g*dot(dir,uSunDir),0.01),1.5));
    vec3 ambient = mix(uFogColor, skyAmbientColor(uAmbient,normalize(vec3(-dir.x,0.6,-dir.z))),0.25);
    vec3 scatter = vec3(0.0);
    float T = 1.0;
    for (int i=0; i<steps; ++i) {
        vec3 pos = uEye + dir * ((float(i)+jitter)*ds);
        float density = uDensity * exp(clamp(-(pos.y-uBaseHeight)/uHeight,-8.0,0.7));
        float opacity = 1.0-exp(-density*ds);
        vec4 lc = uLightViewProj * vec4(pos,1.0);
        vec3 sc = lc.xyz/lc.w*0.5+0.5;
        float lit = 1.0;
        if (all(greaterThan(sc,vec3(0.0))) && all(lessThan(sc,vec3(1.0)))) {
            float edge = smoothstep(0.0,0.04,min(min(sc.x,1.0-sc.x),min(sc.y,1.0-sc.y)));
            lit = mix(1.0,texture(uShadowCmp,vec3(sc.xy,sc.z-0.002)),edge);
        }
        scatter += T*opacity*(ambient+uSunColor*(lit*phase*1.5));
        T *= 1.0-opacity;
    }
    oColor = vec4(scatter,1.0-T);
}
)GLSL";

static const char* kFogApplyShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uFogVolume;
uniform ivec2 uHalfSize;
uniform float uFogMul, uFogOff;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float d = fetchDepth(p);
    // Never haze the sky or replace the game's distance fade. The same projection/fog coefficients as native draws.
    if (d >= 0.99999) { oColor = vec4(0.0); return; }
    float nativeFog = clamp(((d*2.0-1.0)*uFogMul+uFogOff)/255.0,0.0,1.0);
    float z = linDepth(d);
    vec2 hp = (vec2(p)+0.5)*0.5-0.5;
    ivec2 base = ivec2(floor(hp));
    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    float sigma = max(z*0.04,12.0);
    for (int y=-1; y<=2; ++y) {
        for (int x=-1; x<=2; ++x) {
            ivec2 q = clamp(base+ivec2(x,y),ivec2(0),uHalfSize-1);
            float qd = fetchDepth(min(q*2,uSize-1));
            if (qd >= 0.99999) continue;
            float dz = (linDepth(qd)-z)/sigma;
            vec2 delta = vec2(q)-hp;
            float w = exp(-dz*dz-dot(delta,delta)*0.65);
            sum += texelFetch(uFogVolume,q,0)*w;
            wsum += w;
        }
    }
    oColor = sum/max(wsum,1e-6)*(1.0-nativeFog);
}
)GLSL";

// ---------------------------------------------------------------------------------------------------------------------
// Water. At the scene-end marker (after the game's own translucent water): for every pixel whose view ray crosses a
// water plane (from the scene collision) before it reaches the depth-buffer surface, add a Fresnel-weighted sky tint and
// GGX sun glints from a scrolling procedural normal. Opt-in SSR adds a filtered premultiplied reflection overlay.
// ---------------------------------------------------------------------------------------------------------------------
static const char* kWaterPlaneLib = R"GLSL(
uniform mat4 uInvViewProj;
uniform float uNdcXScale;
uniform float uFlipY;
uniform vec3 uEye;
uniform int uBoxCount;
uniform vec4 uBoxRect[16]; // minX, minZ, maxX, maxZ
uniform float uBoxY[16];
vec3 worldPos(ivec2 p, float d) {
    vec2 ndc = (vec2(p) + 0.5) / vec2(uSize) * 2.0 - 1.0;
    ndc.y *= uFlipY;
    ndc.x *= uNdcXScale;
    vec4 w = uInvViewProj * vec4(ndc.x, ndc.y, d * 2.0 - 1.0, 1.0);
    return w.xyz / w.w;
}

int waterPlane(vec3 P, bool sky, out float bestT) {
    vec3 ray = P - uEye;
    bestT = 2.0;
    int bi = -1;
    for (int i = 0; i < uBoxCount; i++) {
        float y = uBoxY[i];
        if (uEye.y <= y + 0.5 || ray.y >= -1e-4) {
            continue; // camera under (or at) the surface, or looking away from it
        }
        float t = (y - uEye.y) / ray.y;
        // Some of the game's water writes depth (its own surface is then what the depth buffer holds): a surface within
        // a couple of units of the plane counts as the water itself, not as something in front of it.
        if (!sky && abs(P.y - y) < 1.5) {
            t = 1.0;
        } else if (t >= 1.0) {
            continue; // the surface (bed, wall, actor) is nearer than the plane
        }
        if (t <= 0.0 || t >= bestT) {
            continue; // behind the camera
        }
        vec2 h = uEye.xz + ray.xz * t;
        if (h.x < uBoxRect[i].x || h.x > uBoxRect[i].z || h.y < uBoxRect[i].y || h.y > uBoxRect[i].w) {
            continue;
        }
        bestT = t;
        bi = i;
    }
    return bi;
}
)GLSL";

static const char* kWaterShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform int uDebug;
uniform float uTime;
uniform float uStrength;
uniform float uPixelAngle;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uSky;
uniform bool uSkyReflectionsEnabled;
uniform samplerCube uSkySpecular;
#ifdef WATER_REFLECTIONS
uniform mat4 uViewProj;
uniform sampler2D uOpaqueColor;
uniform sampler2D uOpaqueDepth;
uniform float uReflectionStrength;

// March uniformly in screen space, interpolating reciprocal clip W for perspective-correct axial depth.
// Stable deterministic steps: no temporal history, random jitter or feedback from the translucent world.
vec4 traceReflection(vec3 origin, vec3 direction, float planeY, out int status) {
    status = 0; // 0: sky/miss, 1: accepted hit, 2: rejected candidate
    if (direction.y <= 0.002) {
        status = 2;
        return vec4(0.0);
    }
    vec4 c0 = uViewProj * vec4(origin, 1.0);
    vec4 c1 = uViewProj * vec4(origin + direction * min(uFar, 6000.0), 1.0);
    if (c0.w <= uNear) {
        return vec4(0.0);
    }
    if (c1.w < uNear) {
        c1 = mix(c0, c1, (c0.w - uNear) / (c0.w - c1.w));
    }
    vec2 uv0 = vec2(c0.x / (c0.w * uNdcXScale), c0.y / c0.w * uFlipY) * 0.5 + 0.5;
    vec2 uv1 = vec2(c1.x / (c1.w * uNdcXScale), c1.y / c1.w * uFlipY) * 0.5 + 0.5;
    vec2 delta = uv1 - uv0;
    float end = 1.0;
    for (int axis = 0; axis < 2; axis++) {
        if (abs(delta[axis]) > 1e-6) {
            float edge = delta[axis] > 0.0 ? 0.999 : 0.001;
            end = min(end, (edge - uv0[axis]) / delta[axis]);
        }
    }
    end = clamp(end, 0.0, 1.0);
    float pixels = length(delta * end * vec2(uSize));
    if (pixels < 2.0) {
        return vec4(0.0);
    }
    int steps = int(clamp(ceil(pixels / 3.0), 8.0, 64.0));
    float k0 = 1.0 / c0.w;
    float k1 = 1.0 / c1.w;
    float previous = 0.0;
    bool inFront = false;
    for (int i = 1; i <= 64; i++) {
        if (i > steps) {
            break;
        }
        float t = end * float(i) / float(steps);
        vec2 uv = uv0 + delta * t;
        float d = texture(uOpaqueDepth, uv).r;
        float gap = 1.0 / mix(k0, k1, t) - linDepth(d);
        if (d < 0.99999 && gap >= 0.0 && inFront) {
            float lo = previous, hi = t;
            for (int j = 0; j < 7; j++) {
                float mid = (lo + hi) * 0.5;
                float md = texture(uOpaqueDepth, uv0 + delta * mid).r;
                if (1.0 / mix(k0, k1, mid) > linDepth(md)) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            uv = uv0 + delta * hi;
            d = texture(uOpaqueDepth, uv).r;
            float depth = linDepth(d);
            float mismatch = abs(1.0 / mix(k0, k1, hi) - depth);
            float thickness = max(8.0, depth * 0.015);
            vec3 hit = worldPos(ivec2(uv * vec2(uSize)), d);
            // Never reflect the bed, or a water mesh that happened to write opaque depth.
            if (d >= 0.99999 || hit.y <= planeY + 2.0 || mismatch > thickness) {
                status = 2;
                inFront = false;
                previous = t;
                continue;
            }
            vec2 edges = min(uv, 1.0 - uv);
            float confidence = smoothstep(0.0, 0.06, min(edges.x, edges.y));
            confidence *= 1.0 - smoothstep(thickness * 0.25, thickness, mismatch);
            confidence *= smoothstep(0.002, 0.06, direction.y);
            confidence *= 1.0 - smoothstep(0.75, 1.0, hi);
            // Small depth-aware cross gather softens the reflection without bleeding across silhouettes.
            vec3 color = texture(uOpaqueColor, uv).rgb;
            float weight = 1.0;
            const vec2 taps[4] = vec2[4](vec2(-1, 0), vec2(1, 0), vec2(0, -1), vec2(0, 1));
            for (int j = 0; j < 4; j++) {
                vec2 tap = uv + taps[j] * 2.5 / vec2(uSize);
                float td = linDepth(texture(uOpaqueDepth, tap).r);
                float w = exp(-abs(td - depth) / thickness);
                color += texture(uOpaqueColor, tap).rgb * w;
                weight += w;
            }
            status = confidence > 0.02 ? 1 : 2;
            return vec4(color / weight, confidence);
        }
        inFront = gap < 0.0;
        previous = t;
    }
    return vec4(0.0);
}
#endif

// Gradient noise with analytic derivatives: (value, d/dx, d/dy).
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}
vec3 noised(vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    vec2 ga = hash2(i + vec2(0.0, 0.0));
    vec2 gb = hash2(i + vec2(1.0, 0.0));
    vec2 gc = hash2(i + vec2(0.0, 1.0));
    vec2 gd = hash2(i + vec2(1.0, 1.0));
    float va = dot(ga, f - vec2(0.0, 0.0));
    float vb = dot(gb, f - vec2(1.0, 0.0));
    float vc = dot(gc, f - vec2(0.0, 1.0));
    float vd = dot(gd, f - vec2(1.0, 1.0));
    return vec3(va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd),
                ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) +
                    du * (u.yx * (va - vb - vc + vd) + vec2(vb, vc) - va));
}

// Surface slope: two scrolling octaves, each faded out as the pixel footprint approaches its wavelength so the
// glints cannot alias.
vec2 waterSlope(vec2 pos, float footprint) {
    float wl1 = 110.0;
    float wl2 = 41.0;
    float f1 = 1.0 - smoothstep(0.12, 0.5, footprint / wl1);
    float f2 = 1.0 - smoothstep(0.12, 0.5, footprint / wl2);
    vec2 s = vec2(0.0);
    s += 0.9 * f1 * noised(pos / wl1 + vec2(0.071, 0.023) * uTime).yz / wl1 * 14.0;
    s += 0.55 * f2 * noised(pos / wl2 + vec2(-0.043, 0.081) * uTime + 7.3).yz / wl2 * 9.0;
    return s;
}

void main() {
#ifdef WATER_REFLECTIONS
    ivec2 p = clamp(ivec2(vUv * vec2(uSize)), ivec2(0), uSize - 1);
#else
    ivec2 p = ivec2(gl_FragCoord.xy);
#endif
    float d = fetchDepth(p);
    bool sky = d >= 0.99999;
    vec3 P = worldPos(p, sky ? 0.9999 : d);
    vec3 ray = P - uEye;
    float bestT;
    int bi = waterPlane(P, sky, bestT);
    if (bi < 0) {
#ifdef WATER_REFLECTIONS
        oColor = vec4(0.0); // premultiplied overlay: leave non-water untouched
#else
        oColor = vec4(0.0, 0.0, 0.0, 1.0);
#endif
        return;
    }
    if (uDebug == 1) {
        oColor = vec4(0.45, 0.0, 0.35, 1.0) * (0.55 + 0.45 * float(bi % 2)); // screen-blended magenta, alternating tone per box
        return;
    }

    vec3 H = uEye + ray * bestT;
    float dist = bestT * length(ray);
    vec3 V = normalize(uEye - H);
    float footprint = dist * uPixelAngle;
    // Vertical depth of the water above the bed: the glints fade out over the shallows.
    float depthY = sky ? 1.0e4 : (abs(P.y - uBoxY[bi]) < 1.5 ? 1.0e4 : max(uBoxY[bi] - P.y, 0.0));
    float shore = smoothstep(2.0, 40.0, depthY);

    vec2 slope = waterSlope(H.xz, footprint);
    vec3 N = normalize(vec3(-slope.x, 1.0, -slope.y));
    float ndv = max(dot(N, V), 1e-3);
    float fres = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    vec3 refl = uSky * fres;
    if (uSkyReflectionsEnabled) {
        vec3 R = reflect(-V, normalize(vec3(-slope.x * 0.25, 1.0, -slope.y * 0.25)));
        refl = textureLod(uSkySpecular, R, 0.4 + clamp(footprint * 0.02, 0.0, 1.1)).rgb * fres;
    }

    vec3 L = uLightDir;
    vec3 Hv = normalize(L + V);
    float ndh = max(dot(N, Hv), 0.0);
    float ndl = max(dot(N, L), 0.0);
    float a = sqrt(0.06 * 0.06 + (footprint * 0.012) * (footprint * 0.012));
    float a2 = a * a;
    float den = ndh * ndh * (a2 - 1.0) + 1.0;
    float D = a2 / (3.14159265 * den * den);
    float vdh = max(dot(V, Hv), 0.0);
    float fl = 0.02 + 0.98 * pow(1.0 - vdh, 5.0);
    float spec = D * fl * ndl * 0.25 / max(ndv, 0.2);
    spec = min(spec, 12.0);

    vec3 add = (refl * 0.6 + uLightColor * spec) * uStrength * shore;
#ifdef WATER_REFLECTIONS
    // A filtered, gently distorted reflection normal; retain the original glint slope separately.
    vec3 reflectionNormal = normalize(vec3(-slope.x * 0.25, 1.0, -slope.y * 0.25));
    int hitStatus;
    // Use the exact plane even when a water mesh's own depth was accepted within the mask's 1.5-unit tolerance.
    vec3 reflectionH = uEye + ray * ((uBoxY[bi] - uEye.y) / ray.y);
    vec4 hit = traceReflection(reflectionH + vec3(0.0, 2.0, 0.0), reflect(-V, reflectionNormal), uBoxY[bi], hitStatus);
    if (uDebug == 2) {
        oColor = vec4(hitStatus == 1 ? vec3(0.0, 1.0, 0.0) :
                      (hitStatus == 2 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0)), 1.0);
        return;
    }
    // A premultiplied reflection can darken the painted water as well as brighten it. Screen-only addition
    // erased dark reflected objects. Lift the Fresnel response perceptually while retaining its view-angle falloff.
    if (uSkyReflectionsEnabled) {
        vec3 R = reflect(-V, reflectionNormal);
        vec3 environment = textureLod(uSkySpecular, R, 0.4 + clamp(footprint * 0.02, 0.0, 1.1)).rgb;
        // Retain the established perceptual visibility for confident geometry hits. The cube uses
        // physical water Schlick (F0=0.02), so a miss cannot increase reflection strength or wash out
        // mid-angle turquoise. Confidence crossfades the two premultiplied contributions.
        float ssrWeight = uReflectionStrength * sqrt(fres) * shore * hit.a;
        float skyWeight = 0.4 * uReflectionStrength * fres * shore * (1.0-hit.a);
        float weight = ssrWeight + skyWeight;
        vec3 glints = max(add - refl * 0.6 * uStrength * shore, vec3(0.0));
        oColor = vec4(environment * skyWeight + hit.rgb * ssrWeight + glints * (1.0-weight), weight);
        return;
    }
    float reflectionWeight = uReflectionStrength * sqrt(fres) * shore * hit.a;
    add += uSky * fres * 0.6 * uReflectionStrength * shore * (1.0 - hit.a);
    oColor = vec4(hit.rgb * reflectionWeight + max(add, 0.0) * (1.0 - reflectionWeight), reflectionWeight);
#else
    oColor = vec4(max(add, 0.0), 1.0);
#endif
}
)GLSL";

static const char* kWaterReflectionApplyShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uReflection;
uniform int uDebug;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float d = fetchDepth(p);
    bool sky = d >= 0.99999;
    vec3 P = worldPos(p, sky ? 0.9999 : d);
    float t;
    if (waterPlane(P, sky, t) < 0) {
        oColor = vec4(0.0);
        return;
    }
    ivec2 halfSize = textureSize(uReflection, 0);
    if (uDebug == 2) {
        // Classify the same half-resolution rays used by the visible effect, without filtering their labels.
        ivec2 q = clamp(ivec2(vUv * vec2(halfSize)), ivec2(0), halfSize - 1);
        vec3 mask = texelFetch(uReflection, q, 0).rgb;
        oColor = vec4(dot(mask, mask) > 0.5 ? mask : vec3(0.0, 0.0, 1.0), 1.0);
        return;
    }
    vec2 samplePos = vUv * vec2(halfSize) - 0.5;
    ivec2 base = ivec2(floor(samplePos));
    float depth = linDepth(d);
    float depthScale = max(4.0, depth * 0.025);
    vec4 color = vec4(0.0);
    float weight = 0.0;
    for (int y = -1; y <= 2; y++) {
        for (int x = -1; x <= 2; x++) {
            ivec2 q = clamp(base + ivec2(x, y), ivec2(0), halfSize - 1);
            vec2 delta = vec2(q) - samplePos;
            ivec2 fullQ = clamp(ivec2((vec2(q) + 0.5) / vec2(halfSize) * vec2(uSize)), ivec2(0), uSize - 1);
            float tapDepth = linDepth(fetchDepth(fullQ));
            float w = exp(-dot(delta, delta) / 3.0 - abs(tapDepth - depth) / depthScale);
            color += texelFetch(uReflection, q, 0) * w;
            weight += w;
        }
    }
    oColor = color / max(weight, 1e-6);
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
    // Postfx shadow programs also declare the ambient cube. Assign its distinct unit at link time,
    // including legacy/HDR-off programs, before any validation or draw can observe the default unit 0.
    GLint previous;
    glGetIntegerv(GL_CURRENT_PROGRAM, &previous);
    glUseProgram(prg);
    for (auto sampler : { std::pair<const char*, int>{ "uSkyIrradiance", PostFxGL::kSkyAmbientUnit },
                          { "uSkySpecular", PostFxGL::kSkyReflectionUnit },
                          { "uFogVolume", PostFxGL::kFogUnit }, { "uDepth", 9 }, { "uShadowCmp", 11 } }) {
        if (auto location = uniforms.find(sampler.first); location != uniforms.end())
            glUniform1i(location->second, sampler.second);
    }
    glUseProgram(previous);
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
    GLboolean blend, depthTest, scissorTest, cullFace, polyOffset, stencilTest, dither, srgb, alphaToCoverage;

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
        alphaToCoverage = glIsEnabled(GL_SAMPLE_ALPHA_TO_COVERAGE);
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
        glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
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
        Set(GL_SAMPLE_ALPHA_TO_COVERAGE, alphaToCoverage);
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
    if (units <= kFogUnit) {
        SPDLOG_WARN("NewDawnPostFx: disabled, only {} texture units available", units);
        return;
    }

    // Unit 14 belongs only to fog. Keep a complete transparent dummy bound even with the feature off.
    GLint previousActive;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActive);
    glActiveTexture(GL_TEXTURE0 + kFogUnit);
    glGenTextures(1, &mFogNeutral);
    glBindTexture(GL_TEXTURE_2D, mFogNeutral);
    const float neutral[4] = {};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 1, 1, 0, GL_RGBA, GL_FLOAT, neutral);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glActiveTexture(previousActive);

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
    glGenVertexArrays(1, &mFoliageVao);
    glGenBuffers(1, &mFoliageVbo);
    glBindVertexArray(mFoliageVao);
    glBindBuffer(GL_ARRAY_BUFFER, mFoliageVbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (const void*)(3 * sizeof(float)));
    glBindVertexArray(0);
    glGenSamplers(1, &mCmpSampler);
    glSamplerParameteri(mCmpSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(mCmpSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(mCmpSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(mCmpSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    const GLfloat cmpBorder[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glSamplerParameterfv(mCmpSampler, GL_TEXTURE_BORDER_COLOR, cmpBorder);
    glSamplerParameteri(mCmpSampler, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glSamplerParameteri(mCmpSampler, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glGenQueries(kQueryStages * kQueryRing, &mQueries[0][0]);
    mUsable = true;
    SPDLOG_INFO("NewDawnPostFx: enabled (OpenGL {}.{}, {})", major, minor,
                reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
}

bool PostFxGL::BuildPrograms() {
    const char* physicalSky[] = { kVersion, kSkyPaletteLib, kPhysicalSkyShader };
    const char* skyApply[] = { kVersion, kSkyApplyShader };
    // Sky shaders fail independently; an unavailable optional sky never disables the established HDR path.
    mPhysicalSky.id = BuildProgram("physical sky", physicalSky, 3, mPhysicalSky.uniforms);
    mSkyApply.id = BuildProgram("sky upscale", skyApply, 2, mSkyApply.uniforms);
    const char* cloudMarch[] = { kVersion, kSkyPaletteLib, kCloudMarchShader };
    const char* cloudApply[] = { kVersion, kCloudApplyShader };
    mCloudMarch.id = BuildProgram("volumetric clouds", cloudMarch, 3, mCloudMarch.uniforms);
    mCloudApply.id = BuildProgram("cloud upscale", cloudApply, 2, mCloudApply.uniforms);
    const char* skyEnvironment[] = { kVersion, "#define SKY_CAPTURE\n", kSkyPaletteLib, kSkyCubeDirectionLib,
                                     kPhysicalSkyShader };
    const char* cloudEnvironment[] = { kVersion, "#define SKY_CAPTURE\n", kSkyPaletteLib, kSkyCubeDirectionLib,
                                       kCloudMarchShader };
    const char* convolve[] = { kVersion, kSkyCubeDirectionLib, kSkyConvolveShader };
    const char* prefilter[] = { kVersion, kSkyCubeDirectionLib, kSkyPrefilterShader };
    mSkyPrefilter.id = BuildProgram("sky GGX prefilter", prefilter, 3, mSkyPrefilter.uniforms);
    mSkyEnvironment.id = BuildProgram("sky radiance cube", skyEnvironment, 5, mSkyEnvironment.uniforms);
    mCloudEnvironment.id = BuildProgram("cloud radiance cube", cloudEnvironment, 5, mCloudEnvironment.uniforms);
    mSkyConvolve.id = BuildProgram("sky diffuse convolution", convolve, 3, mSkyConvolve.uniforms);
    const char* ao[] = { kVersion, kDepthLib, kAoShader };
    const char* blur[] = { kVersion, kDepthLib, kBlurLib, kBlurShader };
    const char* apply[] = { kVersion, kDepthLib, kBlurLib, kApplyShader };
    const char* down[] = { kVersion, kDownShader };
    const char* downHdr[] = { kVersion, "#define HDR_SCENE\n", kDownShader };
    const char* up[] = { kVersion, kUpShader };
    const char* copy[] = { kVersion, R"GLSL(
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uScene;
void main() { fragColor = texture(uScene, vUv); }
)GLSL" };
    mHdrCopy.id = BuildProgram("HDR presentation copy", copy, 2, mHdrCopy.uniforms);
    const char* comp[] = { kVersion, kCompositeShader };
    const char* compHdr[] = { kVersion, "#define HDR_SCENE\n", kCompositeShader };
    const char* compNoDither[] = { kVersion, "#define NO_DITHER\n", kCompositeShader };
    const char* shadowApply[] = { kVersion, kDepthLib, kSkyAmbientLightingLib, kShadowApplyShader };
    const char* shadowApplyHdr[] = { kVersion, "#define HDR_SCENE\n", kDepthLib, kSkyAmbientLightingLib,
                                     kShadowApplyShader };
    const char* shadowDepthVs[] = { kVersion, kShadowDepthVertex };
    const char* shadowDepthFs[] = { kVersion, kShadowDepthFragment };
    const char* foliageVs[] = { kVersion, kShadowFoliageVertex };
    const char* foliageFs[] = { kVersion, kShadowFoliageFragment };
    const char* water[] = { kVersion, kDepthLib, kWaterPlaneLib, kWaterShader };
    mWater.id = BuildProgram("water shader", water, 4, mWater.uniforms);
    const char* waterReflection[] = { kVersion, "#define WATER_REFLECTIONS\n", kDepthLib, kWaterPlaneLib, kWaterShader };
    mWaterReflection.id = BuildProgram("water reflection shader", waterReflection, 5, mWaterReflection.uniforms);
    const char* waterApply[] = { kVersion, kDepthLib, kWaterPlaneLib, kWaterReflectionApplyShader };
    mWaterReflectionApply.id = BuildProgram("water reflection upsample", waterApply, 4, mWaterReflectionApply.uniforms);
    const char* fogMarch[] = { kVersion, kDepthLib, kSkyAmbientLightingLib, kFogMarchShader };
    const char* fogApply[] = { kVersion, kDepthLib, kFogApplyShader };
    // Optional: fog shader failure must not disable the existing renderer.
    mFogMarch.id = BuildProgram("volumetric fog march", fogMarch, 4, mFogMarch.uniforms);
    mFogApply.id = BuildProgram("volumetric fog apply", fogApply, 3, mFogApply.uniforms);
    const char* shaftMarch[] = { kVersion, kDepthLib, kShaftMarchShader };
    const char* shaftBlur[] = { kVersion, kShaftBlurShader };
    const char* shaftApply[] = { kVersion, kDepthLib, kShaftApplyShader };
    mShaftMarch.id = BuildProgram("sun shaft march shader", shaftMarch, 3, mShaftMarch.uniforms);
    mShaftBlur.id = BuildProgram("sun shaft blur shader", shaftBlur, 2, mShaftBlur.uniforms);
    mShaftApply.id = BuildProgram("sun shaft apply shader", shaftApply, 3, mShaftApply.uniforms);
    mAo.id = BuildProgram("AO shader", ao, 3, mAo.uniforms);
    mBlur.id = BuildProgram("AO blur shader", blur, 4, mBlur.uniforms);
    mApply.id = BuildProgram("AO apply shader", apply, 4, mApply.uniforms);
    mDownHdr.id = BuildProgram("HDR bloom downsample", downHdr, 3, mDownHdr.uniforms);
    mCompositeHdr.id = BuildProgram("HDR bloom/display/grade", compHdr, 3, mCompositeHdr.uniforms);
    mCompositeNoDither.id = BuildProgram("LDR composite without dither", compNoDither, 3, mCompositeNoDither.uniforms);
    mShadowApplyHdr.id = BuildProgram("HDR shadow apply", shadowApplyHdr, 5, mShadowApplyHdr.uniforms);
    mDown.id = BuildProgram("bloom downsample shader", down, 2, mDown.uniforms);
    mUp.id = BuildProgram("bloom upsample shader", up, 2, mUp.uniforms);
    mComposite.id = BuildProgram("composite shader", comp, 2, mComposite.uniforms);
    mShadowApply.id = BuildProgram("shadow apply shader", shadowApply, 4, mShadowApply.uniforms);
    mShadowDepth.id = BuildProgramWithVertex("shadow depth shader", shadowDepthVs, 2, shadowDepthFs, 2,
                                             mShadowDepth.uniforms);
    mShadowFoliage.id = BuildProgramWithVertex("shadow foliage shader", foliageVs, 2, foliageFs, 2,
                                               mShadowFoliage.uniforms);
    return mShadowFoliage.id && mDownHdr.id && mCompositeHdr.id && mCompositeNoDither.id && mShadowApplyHdr.id && mAo.id && mBlur.id &&
           mApply.id && mDown.id && mUp.id && mComposite.id && mShadowApply.id && mShadowDepth.id && mShaftMarch.id &&
           mShaftBlur.id && mShaftApply.id && mWater.id && mWaterReflection.id && mWaterReflectionApply.id;
}

void PostFxGL::EnsureTarget(Target& t, int w, int h, GLint internalFormat, GLenum format, GLenum type,
                               bool depthStencil, bool linear) {
    if (t.tex != 0 && t.w == w && t.h == h && t.format == internalFormat) {
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
    t.format = internalFormat;
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
            const char* label = (stage == 0)   ? "opaque end / AO"
                                : (stage == 1) ? "scene end / bloom+grade"
                                : (stage == 2) ? "shadows"
                                : (stage == 3) ? "sun shafts"
                                : (stage == 4) ? "water"
                                : (stage == 5) ? "water opaque copy"
                                : (stage == 6) ? "physical sky"
                                : (stage == 7) ? "volumetric clouds"
                                : (stage == 9) ? "volumetric fog"
                                               : "sky ambient cube";
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

void PostFxGL::RestoreColor(const FramebufferOGL& fb) {
    glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
    if (fb.msaa_level > 1)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, fb.clrbufMsaa);
    else
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fb.clrbuf, 0);
}

bool PostFxGL::BeginHdr(const FramebufferOGL& fb) {
    // Eligibility must be re-established by this frame's physical-sky marker. Never reuse a previous
    // outdoor frame's illumination in interiors, scripted/final-day skies, pause backgrounds or HDR fallbacks.
    mSkyAmbientReady = false;
    mSkyReflectionsReady = false;
    if (!mUsable || !mHdrCopy.id || !CvInt("gNewDawn.HDR.Enabled", 0))
        return false;
    GlStateGuard guard;
    guard.Neutralise();
    EnsureTarget(mHdr, fb.width, fb.height, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
    EnsureTarget(mHdrDisplay, fb.width, fb.height, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE, false, true);
    if (!mUsable)
        return false;
    if (fb.msaa_level > 1) {
        // Query the format-specific support when available (GL 4.2 / ARB_internalformat_query).
        // On GL 3.3 framebuffer completeness below remains the authoritative fallback.
        GLint major = 0, minor = 0;
        glGetIntegerv(GL_MAJOR_VERSION, &major);
        glGetIntegerv(GL_MINOR_VERSION, &minor);
        if (major > 4 || (major == 4 && minor >= 2)) {
            GLint count = 0;
            glGetInternalformativ(GL_RENDERBUFFER, GL_RGBA16F, GL_NUM_SAMPLE_COUNTS, 1, &count);
            std::vector<GLint> samples(count);
            if (count > 0)
                glGetInternalformativ(GL_RENDERBUFFER, GL_RGBA16F, GL_SAMPLES, count, samples.data());
            if (std::find(samples.begin(), samples.end(), fb.msaa_level) == samples.end()) {
                SPDLOG_WARN("NewDawnHDR: RGBA16F does not support {} samples; using LDR", fb.msaa_level);
                return false;
            }
        }
        if (!mHdrMsaa)
            glGenRenderbuffers(1, &mHdrMsaa);
        if (mHdrSamples != (int)fb.msaa_level || mHdrWidth != (int)fb.width || mHdrHeight != (int)fb.height) {
            GLint previous;
            glGetIntegerv(GL_RENDERBUFFER_BINDING, &previous);
            glBindRenderbuffer(GL_RENDERBUFFER, mHdrMsaa);
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, fb.msaa_level, GL_RGBA16F, fb.width, fb.height);
            glBindRenderbuffer(GL_RENDERBUFFER, previous);
            mHdrSamples = fb.msaa_level;
            mHdrWidth = fb.width;
            mHdrHeight = fb.height;
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
    if (fb.msaa_level > 1)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, mHdrMsaa);
    else
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mHdr.tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        RestoreColor(fb);
        SPDLOG_ERROR("NewDawnHDR: shared-depth framebuffer incomplete; using LDR");
        return false;
    }
    // Only swap colour; game depth, stencil, coverage and query scheduling stay intact.
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    mHdrActive = true;
    return true;
}

void PostFxGL::SetSkyUniforms(const NewDawnPostFxParams& params) {
    glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, params.shadowInvViewProj);
    glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), params.shadowFlipY);
    glUniform3fv(Loc("uSunDir"), 1, params.skySunDir);
    float fog[3];
    for (int i = 0; i < 3; ++i) {
        fog[i] =
            DecodeColor(params.skyFogColor[i]) * std::clamp(CvFloat("gNewDawn.HDR.FogIntensity", 1.0f), 0.0f, 4.0f);
    }
    glUniform3fv(Loc("uFog"), 1, fog);
    float palette[3];
    for (int i = 0; i < 3; ++i) {
        palette[i] = DecodeColor(params.skyPaletteColor[i]) * 0.8f + DecodeColor(params.skyFogColor[i]) * 0.2f;
    }
    glUniform3fv(Loc("uPaletteColor"), 1, palette);
    glUniform1f(Loc("uWeather"), params.skyWeather);
    glUniform1f(Loc("uOvercast"), params.skyOvercast);
    glUniform1f(Loc("uIntensity"), std::clamp(CvFloat("gNewDawn.HDR.SkyIntensity", 1.0f), 0.0f, 4.0f));
    glUniform1f(Loc("uCoverage"),
                std::lerp(std::clamp(CvFloat("gNewDawn.Sky.Coverage", 0.55f), 0.0f, 1.0f), 0.8f, params.skyOvercast));
    // Fixed 18/9 m/s wind, driven by monotonic renderer time, independent of the game clock, pause,
    // frame rate and time-speed enhancements. 1024 km is a common period of all noise octaves/maps.
    static const auto epoch = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
    glUniform2f(Loc("uWind"), static_cast<float>(std::fmod(seconds * 0.018, 1024.0)),
                static_cast<float>(std::fmod(seconds * 0.009, 1024.0)));
}

void PostFxGL::RunPhysicalSky(const NewDawnPostFxParams& params, const FramebufferOGL& fb, uint32_t frameIndex) {
    if (!mUsable || !mHdrActive || !mPhysicalSky.id || !mSkyApply.id || !params.shadowUseExactProj ||
        !CvInt("gNewDawn.Sky.Physical", 0)) {
        return;
    }
    mSkyFrame = frameIndex;
    GlStateGuard guard;
    guard.Neutralise();
    glBindVertexArray(mVao);
    mStats = CvInt("gNewDawn.PostFx.Debug.Stats", 0) != 0;
    BeginTimer(kSkyTimerStage);
    // Smooth atmospheric radiance needs no full-resolution raymarch or temporal history. Bilinear
    // reconstruction is stable under camera motion and cannot ghost at scene cuts or the Moon's edge.
    EnsureTarget(mSky, (fb.width + 1) / 2, (fb.height + 1) / 2, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
    if (!mUsable) {
        EndTimer(kSkyTimerStage);
        return;
    }
    Use(mPhysicalSky);
    SetSkyUniforms(params);
    Draw(mSky);
    Use(mSkyApply);
    Bind(0, mSky.tex);
    glUniform1i(Loc("uSky"), kFirstUnit);
    glUniform1f(Loc("uStrength"), std::clamp(CvFloat("gNewDawn.Sky.Strength", 1.0f), 0.0f, 1.0f));
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    Draw(fb.fbo, fb.width, fb.height);
    EndTimer(kSkyTimerStage);
    CheckError("physical sky");
    RunClouds(params, fb);
    RunSkyAmbient(params);
}

void PostFxGL::RunClouds(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    const float strength = std::clamp(CvFloat("gNewDawn.Sky.CloudStrength", 1.0f), 0.0f, 1.0f) *
                           std::clamp(CvFloat("gNewDawn.Sky.Strength", 1.0f), 0.0f, 1.0f);
    const float coverage = std::clamp(CvFloat("gNewDawn.Sky.Coverage", 0.55f), 0.0f, 1.0f);
    if (!CvInt("gNewDawn.Sky.Clouds", 0) || strength <= 0.0f || coverage <= 0.0f || !mCloudMarch.id ||
        !mCloudApply.id) {
        return;
    }
    GlStateGuard guard;
    guard.Neutralise();
    // Volume binding is separate from the backend's 2D texture cache. Restore it even on target failure;
    // the outer guard restores the active texture unit after this local guard has restored its binding.
    struct VolumeBindingGuard {
        GLint previous;
        VolumeBindingGuard() {
            glActiveTexture(GL_TEXTURE0 + kFirstUnit + 1);
            glGetIntegerv(GL_TEXTURE_BINDING_3D, &previous);
        }
        ~VolumeBindingGuard() {
            glActiveTexture(GL_TEXTURE0 + kFirstUnit + 1);
            glBindTexture(GL_TEXTURE_3D, previous);
        }
    } volumeBinding;
    glBindVertexArray(mVao);
    BeginTimer(kCloudTimerStage);
    if (!mCloudNoise) {
        // A small, seamless, original noise volume. No asset IO, licensed textures or random game state.
        constexpr int size = 64;
        std::vector<uint8_t> noise(size * size * size);
        uint32_t seed = 0x91E10DA5;
        for (auto& value : noise) {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            value = static_cast<uint8_t>(seed >> 24);
        }
        glGenTextures(1, &mCloudNoise);
        glActiveTexture(GL_TEXTURE0 + kFirstUnit + 1);
        glBindTexture(GL_TEXTURE_3D, mCloudNoise);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, size, size, size, 0, GL_RED, GL_UNSIGNED_BYTE, noise.data());
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
        glGenerateMipmap(GL_TEXTURE_3D);
    }
    EnsureTarget(mClouds, (fb.width + 1) / 2, (fb.height + 1) / 2, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
    if (!mUsable) {
        EndTimer(kCloudTimerStage);
        return;
    }
    glActiveTexture(GL_TEXTURE0 + kFirstUnit + 1);
    glBindTexture(GL_TEXTURE_3D, mCloudNoise);
    Use(mCloudMarch);
    glUniform1i(Loc("uNoise"), kFirstUnit + 1);
    SetSkyUniforms(params);
    Draw(mClouds);
    Use(mCloudApply);
    Bind(0, mClouds.tex);
    glUniform1i(Loc("uClouds"), kFirstUnit);
    glUniform1f(Loc("uStrength"), strength);
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    Draw(fb.fbo, fb.width, fb.height);
    EndTimer(kCloudTimerStage);
    CheckError("volumetric clouds");
}

bool PostFxGL::EnsureCube(CubeTarget& target, int size, int levels) {
    if (target.size == size && target.levels == levels)
        return true;
    glActiveTexture(GL_TEXTURE0 + kFirstUnit);
    if (!target.tex)
        glGenTextures(1, &target.tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, target.tex);
    for (int level = 0; level < levels; ++level)
        for (int face = 0; face < 6; ++face) {
            const int dim = std::max(size >> level, 1);
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, level, GL_RGB16F, dim, dim, 0, GL_RGB, GL_FLOAT,
                         nullptr);
        }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, levels > 1 ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, levels - 1);
    if (!target.fbo)
        glGenFramebuffers(1, &target.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, target.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X, target.tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        SPDLOG_ERROR("NewDawnSky: incomplete ambient cube; keeping legacy ambient");
        mSkyAmbientUsable = false;
        return false;
    }
    target.size = size;
    target.levels = levels;
    return true;
}

void PostFxGL::RunSkyAmbient(const NewDawnPostFxParams& params) {
    const bool ambient = CvInt("gNewDawn.Sky.Ambient", 0) != 0;
    const bool reflections = CvInt("gNewDawn.Sky.Reflections", 0) != 0 && mSkyPrefilter.id;
    if ((!ambient && !reflections) || !mSkyAmbientUsable || !mSkyEnvironment.id || !mCloudEnvironment.id ||
        !mSkyConvolve.id) {
        return;
    }
    const float settings[] = { CvFloat("gNewDawn.HDR.SkyIntensity", 1.0f),
                               CvFloat("gNewDawn.HDR.FogIntensity", 1.0f),
                               CvFloat("gNewDawn.Sky.Strength", 1.0f),
                               CvFloat("gNewDawn.Sky.Coverage", 0.55f),
                               CvFloat("gNewDawn.Sky.CloudStrength", 1.0f),
                               (float)CvInt("gNewDawn.Sky.Clouds", 0),
                               (float)reflections };
    float sunDot = 0.0f;
    for (int i = 0; i < 3; ++i)
        sunDot += params.skySunDir[i] * mSkyCachedSun[i];
    bool settingsChanged = false;
    for (int i = 0; i < 7; ++i)
        settingsChanged |= settings[i] != mSkyCachedSettings[i];
    // Eligibility is reset each HDR frame, while the lighting cache persists only across consecutive
    // eligible frames. Refresh slowly moving cloud irradiance every eight rendered frames; force an
    // immediate refresh after scene/fallback gaps, control/weather changes or about one degree of sun motion.
    const bool refresh = !mSkyAmbientCacheValid || mSkyFrame - mSkyLastEligible != 1 ||
                         mSkyCachedScene != params.skySceneId || mSkyFrame - mSkyLastRefresh >= 8 ||
                         sunDot < 0.9998477f || std::abs(params.skyWeather - mSkyCachedWeather) > 0.01f ||
                         settingsChanged;
    mSkyLastEligible = mSkyFrame;
    // Exact interpolated near-centre point; its near-plane offset is negligible at sky distances.
    const float* inv = params.shadowInvViewProj;
    const float w = inv[15] - inv[11];
    for (int i = 0; i < 3; ++i)
        mSkyReflectionEye[i] = (inv[12 + i] - inv[8 + i]) / (std::abs(w) > 1e-6f ? w : 1.0f);
    // This timer includes reuse frames, so its average is the actual amortised GPU cost per eligible frame.
    BeginTimer(kSkyAmbientTimerStage);
    if (!refresh) {
        mSkyAmbientReady = ambient;
        mSkyReflectionsReady = reflections;
        EndTimer(kSkyAmbientTimerStage);
        CheckError("sky ambient reuse");
        return;
    }
    GlStateGuard guard;
    guard.Neutralise();
    struct BindingGuard {
        GLint cube, volume;
        BindingGuard() {
            glActiveTexture(GL_TEXTURE0 + kFirstUnit);
            glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &cube);
            glActiveTexture(GL_TEXTURE0 + kFirstUnit + 1);
            glGetIntegerv(GL_TEXTURE_BINDING_3D, &volume);
        }
        ~BindingGuard() {
            glActiveTexture(GL_TEXTURE0 + kFirstUnit);
            glBindTexture(GL_TEXTURE_CUBE_MAP, cube);
            glActiveTexture(GL_TEXTURE0 + kFirstUnit + 1);
            glBindTexture(GL_TEXTURE_3D, volume);
        }
    } bindings;
    glBindVertexArray(mVao);
    if (!EnsureCube(mSkyRadiance, reflections ? 32 : 16) || !EnsureCube(mSkyIrradiance, 8) ||
        (reflections && !EnsureCube(mSkySpecular, 32, 5))) {
        EndTimer(kSkyAmbientTimerStage);
        return;
    }
    const bool clouds = CvInt("gNewDawn.Sky.Clouds", 0) && mCloudNoise &&
                        CvFloat("gNewDawn.Sky.Coverage", 0.55f) > 0.0f &&
                        CvFloat("gNewDawn.Sky.CloudStrength", 1.0f) > 0.0f;
    const float cloudStrength = std::clamp(CvFloat("gNewDawn.Sky.CloudStrength", 1.0f), 0.0f, 1.0f) *
                                std::clamp(CvFloat("gNewDawn.Sky.Strength", 1.0f), 0.0f, 1.0f);
    if (clouds) {
        glActiveTexture(GL_TEXTURE0 + kFirstUnit + 1);
        glBindTexture(GL_TEXTURE_3D, mCloudNoise);
    }
    // Group by program so shared uniforms/CVars are uploaded once, not twelve times between tiny draws.
    // This also avoids GPU idle bubbles while the CPU changes program state for each cube face.
    Use(mSkyEnvironment);
    SetSkyUniforms(params);
    glDisable(GL_BLEND);
    for (int face = 0; face < 6; ++face) {
        glBindFramebuffer(GL_FRAMEBUFFER, mSkyRadiance.fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                               mSkyRadiance.tex, 0);
        glUniform1i(Loc("uCubeFace"), face);
        Draw(mSkyRadiance.fbo, mSkyRadiance.size, mSkyRadiance.size);
    }
    if (clouds) {
        Use(mCloudEnvironment);
        SetSkyUniforms(params);
        glUniform1i(Loc("uNoise"), kFirstUnit + 1);
        glUniform3fv(Loc("uCubeOrigin"), 1, mSkyReflectionEye);
        glUniform1f(Loc("uCloudStrength"), cloudStrength);
        glEnable(GL_BLEND);
        glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
        for (int face = 0; face < 6; ++face) {
            glBindFramebuffer(GL_FRAMEBUFFER, mSkyRadiance.fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                                   mSkyRadiance.tex, 0);
            glUniform1i(Loc("uCubeFace"), face);
            Draw(mSkyRadiance.fbo, mSkyRadiance.size, mSkyRadiance.size);
        }
    }
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0 + kFirstUnit);
    glBindTexture(GL_TEXTURE_CUBE_MAP, mSkyRadiance.tex);
    Use(mSkyConvolve);
    glUniform1i(Loc("uEnvironment"), kFirstUnit);
    glUniform3fv(Loc("uGroundAlbedo"), 1, params.skyGroundAlbedo);
    for (int face = 0; face < 6; ++face) {
        glBindFramebuffer(GL_FRAMEBUFFER, mSkyIrradiance.fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                               mSkyIrradiance.tex, 0);
        glUniform1i(Loc("uCubeFace"), face);
        Draw(mSkyIrradiance.fbo, mSkyIrradiance.size, mSkyIrradiance.size);
    }
    glActiveTexture(GL_TEXTURE0 + kSkyAmbientUnit);
    glBindTexture(GL_TEXTURE_CUBE_MAP, mSkyIrradiance.tex);
    mSkyAmbientReady = ambient;
    mSkyReflectionsReady = reflections;
    if (reflections) {
        Use(mSkyPrefilter);
        glUniform1i(Loc("uEnvironment"), kFirstUnit);
        for (int level = 0; level < 5; ++level) {
            glUniform1f(Loc("uRoughness"), level / 4.0f);
            for (int face = 0; face < 6; ++face) {
                glBindFramebuffer(GL_FRAMEBUFFER, mSkySpecular.fbo);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                                       mSkySpecular.tex, level);
                glUniform1i(Loc("uCubeFace"), face);
                Draw(mSkySpecular.fbo, std::max(mSkySpecular.size >> level, 1),
                     std::max(mSkySpecular.size >> level, 1));
            }
        }
        glActiveTexture(GL_TEXTURE0 + kSkyReflectionUnit);
        glBindTexture(GL_TEXTURE_CUBE_MAP, mSkySpecular.tex);
    }
    mSkyAmbientCacheValid = true;
    mSkyLastRefresh = mSkyFrame;
    mSkyCachedScene = params.skySceneId;
    mSkyCachedWeather = params.skyWeather;
    std::copy(params.skySunDir, params.skySunDir + 3, mSkyCachedSun);
    std::copy(settings, settings + 7, mSkyCachedSettings);
    EndTimer(kSkyAmbientTimerStage);
    CheckError("sky ambient cube");
}

GLuint PostFxGL::HdrSnapshot(const FramebufferOGL& fb) {
    GlStateGuard guard;
    guard.Neutralise();
    glBindVertexArray(mVao);
    mStats = CvInt("gNewDawn.PostFx.Debug.Stats", 0) != 0;
    Settings s = ReadSettings();
    if (!s.postFxOn)
        s.bloomOn = s.gradeOn = false;
    BeginTimer(1);
    RunSceneEnd(fb, s); // resolves RGBA16F before bloom, then tonemaps once into display colour
    EndTimer(1);
    CheckError("HDR display transform");
    return mHdrDisplay.fbo;
}

void PostFxGL::EndHdr(const FramebufferOGL& fb) {
    HdrSnapshot(fb);
    GlStateGuard guard;
    guard.Neutralise();
    glBindVertexArray(mVao);
    RestoreColor(fb);
    // A shader copy supports both single-sample and multisample compatibility attachments.
    Bind(0, mHdrDisplay.tex);
    Use(mHdrCopy);
    glUniform1i(Loc("uScene"), kFirstUnit);
    Draw(fb.fbo, fb.width, fb.height);
    CheckError("HDR presentation copy");
    mHdrActive = false;
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
    const bool captureWater = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && params.waterReflections != 0 &&
                              params.waterBoxCount > 0 && params.shadowUseExactProj != 0;
    const bool doWater = stage == NEWDAWN_POSTFX_STAGE_SCENE_END && params.waterEnabled != 0 && params.waterBoxCount > 0 &&
                         params.shadowUseExactProj != 0;
    const bool doFog = doShadow && mHdrActive && params.fogVolumetric && mFogMarch.id && mFogApply.id &&
                       params.shadowUseExactProj && s.debugView == 0 && s.shadowDebug == 0;
    const bool doShafts = !doFog && doShadow && params.shaftEnabled != 0 && params.shaftStrength > 0.0f && s.debugView != 2 &&
                          s.shadowDebug == 0;
    // The AO debug view needs the finished world left alone, otherwise the grade would tint the AO buffer
    const bool doScene = !mHdrActive && stage == NEWDAWN_POSTFX_STAGE_SCENE_END && s.postFxOn && s.debugView != 2 &&
                         (s.bloomOn || s.gradeOn);
    if (!doShadow && !doAo && !doScene && !doWater && !captureWater) {
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

    if (doShadow || doAo || doWater) {
        PrepareDepth(fb);
    }
    if (doWater) {
        BeginTimer(kWaterTimerStage);
        RunWater(params, fb);
        EndTimer(kWaterTimerStage);
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
    if (doShafts) {
        BeginTimer(kShaftTimerStage);
        RunSunShafts(params, fb);
        EndTimer(kShaftTimerStage);
    }

    if (doFog) {
        BeginTimer(kFogTimerStage);
        RunVolumetricFog(params, fb);
        EndTimer(kFogTimerStage);
    }
    if (captureWater) {
        BeginTimer(kWaterCaptureTimerStage);
        // AO split view leaves its apply scissor enabled; copies must cover the whole source.
        glDisable(GL_SCISSOR_TEST);
        const int w = fb.width, h = fb.height;
        EnsureTarget(mWaterOpaqueColor, w, h, mHdrActive ? GL_RGBA16F : GL_RGB8, GL_RGBA,
                     mHdrActive ? GL_FLOAT : GL_UNSIGNED_BYTE, false, true);
        EnsureTarget(mWaterOpaqueDepth, w, h, GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, true, false);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mWaterOpaqueColor.fbo);
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mWaterOpaqueDepth.fbo);
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
        mWaterOpaqueFrame = frameIndex;
        EndTimer(kWaterCaptureTimerStage);
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

    static bool sampleLogsDone = false;
    if (debug == 4 && !sampleLogsDone) {
        const int sampleX[] = { (int)(w * 0.29f), (int)(w * 0.78f) };
        const int sampleYTop[] = { (int)(h * 0.23f), (int)(h * 0.46f) };
        glBindFramebuffer(GL_READ_FRAMEBUFFER, mDepth.fbo);
        for (int sample = 0; sample < 2; sample++) {
            int px = std::clamp(sampleX[sample], 0, w - 1);
            int py = std::clamp(sampleYTop[sample], 0, h - 1);
            float depth = 1.0f;
            glReadPixels(px, py, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
            if (depth >= 0.99999f) {
                SPDLOG_INFO("NewDawn shadows: sample {} at ({},{}) is background", sample, px, py);
                continue;
            }

            float ndc[4] = { ((float)px + 0.5f) / w * 2.0f - 1.0f,
                             ((float)py + 0.5f) / h * 2.0f - 1.0f, depth * 2.0f - 1.0f, 1.0f };
            ndc[0] *= params.shadowNdcXScale;
            ndc[1] *= params.shadowFlipY;
            float world[4] = {};
            for (int j = 0; j < 4; j++) {
                for (int i = 0; i < 4; i++) {
                    world[j] += ndc[i] * params.shadowInvViewProj[i * 4 + j];
                }
            }
            if (std::fabs(world[3]) > 1.0e-8f) {
                world[0] /= world[3];
                world[1] /= world[3];
                world[2] /= world[3];
            }

            float bestDistSq = std::numeric_limits<float>::max();
            float bestVertex[3] = {};
            for (uint32_t i = 0; i < vertexCount; i++) {
                float dx = positions[i * 3 + 0] - world[0];
                float dy = positions[i * 3 + 1] - world[1];
                float dz = positions[i * 3 + 2] - world[2];
                float distSq = dx * dx + dy * dy + dz * dz;
                if (distSq < bestDistSq) {
                    bestDistSq = distSq;
                    bestVertex[0] = positions[i * 3 + 0];
                    bestVertex[1] = positions[i * 3 + 1];
                    bestVertex[2] = positions[i * 3 + 2];
                }
            }
            SPDLOG_INFO("NewDawn shadows: sample {} pixel ({},{}) depth {:.5f} world ({:.1f},{:.1f},{:.1f}); "
                        "nearest caster vertex ({:.1f},{:.1f},{:.1f}), distance {:.1f}",
                        sample, px, py, depth, world[0], world[1], world[2], bestVertex[0], bestVertex[1],
                        bestVertex[2], std::sqrt(bestDistSq));
        }
        sampleLogsDone = true;
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

    if (mFoliageBatchCount > 0 && mFoliageVerts != nullptr) {
        // Cut-out casters: same light matrix and depth state, fragment-discarded by the texel alpha.
        Use(mShadowFoliage);
        glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, lightViewProj);
        glUniform1i(Loc("uTex"), kFirstUnit);
        glBindVertexArray(mFoliageVao);
        glBindBuffer(GL_ARRAY_BUFFER, mFoliageVbo);
        const GLsizeiptr floats = (GLsizeiptr)(mFoliageBatches[mFoliageBatchCount - 1].firstVertex +
                                               mFoliageBatches[mFoliageBatchCount - 1].vertexCount) * 5;
        glBufferData(GL_ARRAY_BUFFER, floats * sizeof(float), mFoliageVerts, GL_STREAM_DRAW);
        for (uint32_t i = 0; i < mFoliageBatchCount; i++) {
            Bind(0, mFoliageBatches[i].textureId);
            glDrawArrays(GL_TRIANGLES, (GLint)mFoliageBatches[i].firstVertex, (GLsizei)mFoliageBatches[i].vertexCount);
        }
        CheckError("shadow foliage render");
    }

    if (debug == 4) {
        static bool mapSampleLogsDone = false;
        if (!mapSampleLogsDone) {
            const int sampleX[] = { (int)(w * 0.29f), (int)(w * 0.78f) };
            const int sampleYTop[] = { (int)(h * 0.23f), (int)(h * 0.46f) };
            float lightViewProj[16];
            BuildLightViewProj(params, lightViewProj);
            for (int sample = 0; sample < 2; sample++) {
                int px = std::clamp(sampleX[sample], 0, w - 1);
                int py = std::clamp(sampleYTop[sample], 0, h - 1);
                float depth = 1.0f;
                glBindFramebuffer(GL_READ_FRAMEBUFFER, mDepth.fbo);
                glReadPixels(px, py, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
                if (depth >= 0.99999f) {
                    continue;
                }
                float ndc[4] = { ((float)px + 0.5f) / w * 2.0f - 1.0f,
                                 ((float)py + 0.5f) / h * 2.0f - 1.0f, depth * 2.0f - 1.0f, 1.0f };
                ndc[0] *= params.shadowNdcXScale;
                ndc[1] *= params.shadowFlipY;
                float world[4] = {};
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < 4; i++) {
                        world[j] += ndc[i] * params.shadowInvViewProj[i * 4 + j];
                    }
                }
                if (std::fabs(world[3]) > 1.0e-8f) {
                    world[0] /= world[3];
                    world[1] /= world[3];
                    world[2] /= world[3];
                }

                float clip[4] = {};
                for (int row = 0; row < 4; row++) {
                    clip[row] = lightViewProj[row] * world[0] + lightViewProj[4 + row] * world[1] +
                                lightViewProj[8 + row] * world[2] + lightViewProj[12 + row];
                }
                float suv[2] = { clip[0] / clip[3] * 0.5f + 0.5f, clip[1] / clip[3] * 0.5f + 0.5f };
                float receiverRef = clip[2] / clip[3] * 0.5f + 0.5f;
                int mapX = std::clamp((int)(suv[0] * size), 0, size - 1);
                int mapY = std::clamp((int)(suv[1] * size), 0, size - 1);
                float mapDepth = 1.0f;
                glBindFramebuffer(GL_READ_FRAMEBUFFER, mShadowMap.fbo);
                glReadPixels(mapX, mapY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &mapDepth);
                SPDLOG_INFO("NewDawn shadows: receiver sample {} world ({:.1f},{:.1f},{:.1f}) light ({:.3f},{:.3f}) "
                            "ref {:.5f}, map depth {:.5f}",
                            sample, world[0], world[1], world[2], suv[0], suv[1], receiverRef, mapDepth);
            }
            mapSampleLogsDone = true;
        }
    }

    glBindVertexArray(mVao);
    // The shadow map pass enabled depth writes; the full-screen apply must not touch the game depth.
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    // Screen-space shadow apply. Work on a copy of the colour so the game framebuffer is never sampled while bound.
    EnsureTarget(mScene, w, h, mHdrActive ? GL_RGBA16F : GL_RGB8, GL_RGBA, mHdrActive ? GL_FLOAT : GL_UNSIGNED_BYTE,
                 false, true);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mScene.fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    CheckError("shadow colour blit");


    Bind(0, mScene.tex);
    Bind(1, mDepth.tex);
    Bind(2, mShadowMap.tex);
    Bind(3, mShadowMap.tex);
    glBindSampler(kFirstUnit + 3, mCmpSampler);
    Use(mHdrActive ? mShadowApplyHdr : mShadowApply);
    glUniform1i(Loc("uColor"), kFirstUnit + 0);
    glUniform1i(Loc("uDepth"), kFirstUnit + 1);
    glUniform1i(Loc("uShadow"), kFirstUnit + 2);
    glUniform1i(Loc("uShadowCmp"), kFirstUnit + 3);
    glUniform2i(Loc("uSize"), w, h);
    glUniform1f(Loc("uNear"), params.zNear);
    glUniform1f(Loc("uFar"), params.zFar);
    glUniform1f(Loc("uFlipY"), params.shadowFlipY);
    glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
    glUniform2f(Loc("uShadowTexel"), 1.0f / size, 1.0f / size);
    glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, params.shadowInvViewProj);
    glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, lightViewProj);
    glUniform3fv(Loc("uSunDir"), 1, params.shadowSunDir);
    float ambient[3], direct[3];
    for (int i = 0; i < 3; ++i) {
        ambient[i] = params.shadowAmbient[i] *
                     (mHdrActive ? std::clamp(CvFloat("gNewDawn.HDR.AmbientFill", 0.4f), 0.0f, 4.0f) : 1.0f);
        direct[i] = params.shadowDirColor[i] *
                    (mHdrActive ? std::clamp(CvFloat("gNewDawn.HDR.SunIntensity", 1.8f), 0.0f, 16.0f) : 1.0f);
    }
    glUniform3fv(Loc("uAmbient"), 1, ambient);
    glUniform1i(Loc("uSkyAmbientEnabled"), SkyAmbientActive());
    glUniform1i(Loc("uSkyIrradiance"), kSkyAmbientUnit);
    glUniform3fv(Loc("uDirColor"), 1, direct);
    glUniform3fv(Loc("uCameraEye"), 1, params.shadowCameraEye);
    glUniform1f(Loc("uStrength"), params.shadowStrength);
    glUniform1f(Loc("uMinFactor"), mSettings.shadowFloor);
    glUniform1f(Loc("uRadius"), params.shadowRadius);
    float fogMul, fogOff;
    FogCoefficients(params, fogMul, fogOff);
    glUniform1i(Loc("uHdr"), mHdrActive);
    glUniform1f(Loc("uFogMul"), fogMul);
    glUniform1f(Loc("uFogOff"), fogOff);
    glUniform1f(Loc("uNormalBias"), 2.0f * params.shadowRadius / (float)size * 4.0f);
    glUniform1i(Loc("uDebug"), debug);
    Draw(fb.fbo, w, h);
    glBindSampler(kFirstUnit + 3, 0);
    CheckError("shadow apply");
}

void PostFxGL::RunWater(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    const int w = fb.width, h = fb.height;
    const int count = std::clamp(params.waterBoxCount, 0, NEWDAWN_MAX_WATER_BOXES);
    float rect[NEWDAWN_MAX_WATER_BOXES * 4] = {};
    float surfaceY[NEWDAWN_MAX_WATER_BOXES] = {};
    for (int i = 0; i < count; i++) {
        rect[i * 4 + 0] = params.waterBoxes[i][0];
        rect[i * 4 + 1] = params.waterBoxes[i][1];
        rect[i * 4 + 2] = params.waterBoxes[i][2];
        rect[i * 4 + 3] = params.waterBoxes[i][3];
        surfaceY[i] = params.waterBoxes[i][4];
    }
    const bool reflections = params.waterReflections != 0 && mWaterOpaqueFrame == mSettingsFrame &&
                             mWaterOpaqueColor.w == w && mWaterOpaqueColor.h == h;
    const bool filter = reflections && params.waterDebug != 1;
    const int hw = std::max(w / 2, 1), hh = std::max(h / 2, 1);
    if (filter) {
        EnsureTarget(mWaterReflectionRaw, hw, hh, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, false);
    }
    const auto setPlaneUniforms = [&]() {
        glUniform1i(Loc("uDepth"), kFirstUnit + 1);
        glUniform2i(Loc("uSize"), w, h);
        glUniform1f(Loc("uNear"), params.zNear);
        glUniform1f(Loc("uFar"), params.zFar);
        glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
        glUniform1f(Loc("uFlipY"), params.shadowFlipY);
        glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, params.shadowInvViewProj);
        glUniform3fv(Loc("uEye"), 1, params.waterCameraEye);
        glUniform1i(Loc("uBoxCount"), count);
        glUniform4fv(Loc("uBoxRect"), count, rect);
        glUniform1fv(Loc("uBoxY"), count, surfaceY);
    };
    Bind(1, mDepth.tex);
    Use(reflections ? mWaterReflection : mWater);
    if (reflections) {
        Bind(0, mWaterOpaqueColor.tex);
        Bind(2, mWaterOpaqueDepth.tex);
        glUniform1i(Loc("uOpaqueColor"), kFirstUnit + 0);
        glUniform1i(Loc("uOpaqueDepth"), kFirstUnit + 2);
        glUniformMatrix4fv(Loc("uViewProj"), 1, GL_FALSE, params.shadowViewProj);
        glUniform1f(Loc("uReflectionStrength"), params.waterReflectionStrength);
    }
    setPlaneUniforms();
    glUniform1i(Loc("uDebug"), params.waterDebug);
    glUniform1f(Loc("uTime"), params.waterTime);
    glUniform1f(Loc("uStrength"), params.waterStrength);
    glUniform1f(Loc("uPixelAngle"),
                2.0f * std::tan(params.fovyDegrees * 3.14159265f / 360.0f) / (float)(filter ? hh : h));
    glUniform3fv(Loc("uLightDir"), 1, params.waterLightDir);
    float light[3], sky[3];
    for (int i = 0; i < 3; ++i) {
        light[i] = mHdrActive ? DecodeColor(params.waterLightColor[i]) *
                                    std::clamp(CvFloat("gNewDawn.HDR.SunIntensity", 1.8f), 0.0f, 16.0f)
                              : params.waterLightColor[i];
        sky[i] = mHdrActive ? DecodeColor(params.waterSkyColor[i]) *
                                  std::clamp(CvFloat("gNewDawn.HDR.SkyIntensity", 1.0f), 0.0f, 4.0f)
                            : params.waterSkyColor[i];
    }
    glUniform3fv(Loc("uLightColor"), 1, light);
    glUniform3fv(Loc("uSky"), 1, sky);
    glUniform1i(Loc("uSkyReflectionsEnabled"), SkyReflectionsActive());
    glUniform1i(Loc("uSkySpecular"), kSkyReflectionUnit);
    if (filter) {
        glDisable(GL_BLEND);
        Draw(mWaterReflectionRaw);
        Bind(0, mWaterReflectionRaw.tex);
        Bind(1, mDepth.tex);
        Use(mWaterReflectionApply);
        setPlaneUniforms();
        glUniform1i(Loc("uReflection"), kFirstUnit + 0);
        glUniform1i(Loc("uDebug"), params.waterDebug);
    }
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    // Original water remains screen blended. SSR is a premultiplied overlay; its hit mask uses exact colours.
    glBlendFunc(GL_ONE, reflections && params.waterDebug == 2
                            ? GL_ZERO
                            : (reflections && params.waterDebug != 1
                                   ? GL_ONE_MINUS_SRC_ALPHA
                                   : (mHdrActive && params.waterDebug == 0 ? GL_ONE : GL_ONE_MINUS_SRC_COLOR)));
    Draw(fb.fbo, w, h);
    glDisable(GL_BLEND);
    CheckError("water");
}

void PostFxGL::RunVolumetricFog(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    const int w = fb.width, h = fb.height;
    const int hw = (w + 1) / 2, hh = (h + 1) / 2;
    EnsureTarget(mFog, hw, hh, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, false);
    if (!mUsable)
        return;
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    float lightViewProj[16];
    BuildLightViewProj(params, lightViewProj);
    float fogMul, fogOff;
    FogCoefficients(params, fogMul, fogOff);
    // Native start distance supplies the aerosol scale. Bound the artist input so special/near-infinite
    // fog distances cannot make the volume opaque or numerically unstable.
    float ndcStart = fogMul > 0.0f ? std::clamp(-fogOff / fogMul, -1.0f, 1.0f) : 1.0f;
    float startDistance = 2.0f * params.zNear * params.zFar /
                          (params.zFar + params.zNear - ndcStart * (params.zFar - params.zNear));
    const float sunHeight = std::clamp(params.shadowSunDir[1], 0.0f, 1.0f);
    const float density = 0.16f / std::clamp(startDistance, 800.0f, 2400.0f) * params.fogDensityScale *
                          (1.4f - 0.7f * sunHeight);
    const float sunFade = std::clamp((sunHeight - 0.03f) / 0.25f, 0.0f, 1.0f);
    float fog[3], sun[3], ambient[3];
    for (int i = 0; i < 3; ++i) {
        fog[i] = DecodeColor(params.shaftFogColor[i]) *
                 std::clamp(CvFloat("gNewDawn.HDR.FogIntensity", 1.0f), 0.0f, 4.0f);
        ambient[i] = DecodeColor(params.shadowAmbient[i]);
        sun[i] = DecodeColor(params.shadowDirColor[i]) * sunFade *
                 std::clamp(CvFloat("gNewDawn.HDR.SunIntensity", 1.8f), 0.0f, 16.0f);
    }
    Bind(1, mDepth.tex);
    Bind(3, mShadowMap.tex);
    glBindSampler(kFirstUnit + 3, mCmpSampler);
    Use(mFogMarch);
    glUniform2i(Loc("uSize"), w, h);
    glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, params.shadowInvViewProj);
    glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, lightViewProj);
    glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), params.shadowFlipY);
    glUniform3fv(Loc("uEye"), 1, params.shadowCameraEye);
    glUniform3fv(Loc("uSunDir"), 1, params.shadowSunDir);
    glUniform3fv(Loc("uSunColor"), 1, sun);
    glUniform3fv(Loc("uAmbient"), 1, ambient);
    glUniform3fv(Loc("uFogColor"), 1, fog);
    glUniform1i(Loc("uSkyAmbientEnabled"), SkyAmbientActive());
    glUniform1f(Loc("uDensity"), density);
    glUniform1f(Loc("uBaseHeight"), params.fogBaseHeight);
    glUniform1f(Loc("uHeight"), params.fogHeight);
    glUniform1f(Loc("uMaxDist"), std::min(params.zFar, 6000.0f));
    Draw(mFog);
    glBindSampler(kFirstUnit + 3, 0);

    glActiveTexture(GL_TEXTURE0 + kFogUnit);
    glBindTexture(GL_TEXTURE_2D, mFog.tex);
    Use(mFogApply);
    glUniform2i(Loc("uSize"), w, h);
    glUniform2i(Loc("uHalfSize"), hw, hh);
    glUniform1f(Loc("uNear"), params.zNear);
    glUniform1f(Loc("uFar"), params.zFar);
    glUniform1f(Loc("uFogMul"), fogMul);
    glUniform1f(Loc("uFogOff"), fogOff);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    Draw(fb.fbo, w, h);
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0 + kFogUnit);
    glBindTexture(GL_TEXTURE_2D, mFogNeutral);
    CheckError("volumetric fog");
}

void PostFxGL::RunSunShafts(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    const int w = fb.width, h = fb.height;
    const int hw = std::max(w / 2, 1), hh = std::max(h / 2, 1);
    EnsureTarget(mShaftA, hw, hh, GL_RG16F, GL_RG, GL_FLOAT, false, false);
    EnsureTarget(mShaftB, hw, hh, GL_RG16F, GL_RG, GL_FLOAT, false, false);
    if (!mUsable) {
        return;
    }
    float lightViewProj[16];
    BuildLightViewProj(params, lightViewProj);
    // The AO apply pass leaves multiplicative blending (and possibly a scissor) enabled.
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);

    // 1. March: how much of the ray towards each pixel is lit, weighted by the phase function.
    Bind(1, mDepth.tex);
    Bind(3, mShadowMap.tex);
    glBindSampler(kFirstUnit + 3, mCmpSampler);
    Use(mShaftMarch);
    glUniform1i(Loc("uDepth"), kFirstUnit + 1);
    glUniform1i(Loc("uShadowCmp"), kFirstUnit + 3);
    glUniform2i(Loc("uSize"), w, h);
    glUniform1f(Loc("uNear"), params.zNear);
    glUniform1f(Loc("uFar"), params.zFar);
    glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), params.shadowFlipY);
    glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, params.shadowInvViewProj);
    glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, lightViewProj);
    glUniform3fv(Loc("uSunDir"), 1, params.shadowSunDir);
    glUniform3fv(Loc("uCameraEye"), 1, params.shadowCameraEye);
    glUniform1f(Loc("uMaxDist"), 2.0f * params.shadowRadius);
    glUniform1f(Loc("uDensity"), 0.6f / params.shadowRadius);
    glUniform1f(Loc("uAnisotropy"), 0.65f);
    Draw(mShaftA);
    glBindSampler(kFirstUnit + 3, 0);
    CheckError("sun shaft march");

    // 2. Blur, horizontal then vertical.
    Use(mShaftBlur);
    glUniform1i(Loc("uSrc"), kFirstUnit + 0);
    glUniform2i(Loc("uHalfSize"), hw, hh);
    Bind(0, mShaftA.tex);
    glUniform2i(Loc("uDir"), 1, 0);
    Draw(mShaftB);
    Bind(0, mShaftB.tex);
    glUniform2i(Loc("uDir"), 0, 1);
    Draw(mShaftA);
    CheckError("sun shaft blur");

    // 3. Upsample into the game framebuffer with a screen blend.
    Bind(0, mShaftA.tex);
    Use(mShaftApply);
    glUniform1i(Loc("uSrc"), kFirstUnit + 0);
    glUniform1i(Loc("uDepth"), kFirstUnit + 1);
    glUniform2i(Loc("uSize"), w, h);
    glUniform2i(Loc("uHalfSize"), hw, hh);
    glUniform1f(Loc("uNear"), params.zNear);
    glUniform1f(Loc("uFar"), params.zFar);
    // The in-scatter colour: the game's fog colour pulled towards the sun's light, so the shafts read as part of the
    // existing atmosphere rather than as a separate overlay.
    float color[3];
    for (int i = 0; i < 3; i++) {
        color[i] = mHdrActive ? 0.6f * DecodeColor(params.shaftFogColor[i]) *
                                        std::clamp(CvFloat("gNewDawn.HDR.FogIntensity", 1.0f), 0.0f, 4.0f) +
                                    0.4f * DecodeColor(params.shadowDirColor[i])
                              : 0.6f * params.shaftFogColor[i] + 0.4f * params.shadowDirColor[i];
    }
    glUniform3fv(Loc("uColor"), 1, color);
    glUniform1f(Loc("uStrength"), params.shaftStrength);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, mHdrActive ? GL_ONE : GL_ONE_MINUS_SRC_COLOR);
    Draw(fb.fbo, w, h);
    glDisable(GL_BLEND);
    CheckError("sun shaft apply");
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
    FogCoefficients(params, fogMul, fogOff);

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
    EnsureTarget(mScene, w, h, mHdrActive ? GL_RGBA16F : GL_RGB8, GL_RGBA, mHdrActive ? GL_FLOAT : GL_UNSIGNED_BYTE,
                 false, true);

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
        Use(mHdrActive ? mDownHdr : mDown);
        glUniform1i(Loc("uSrc"), kFirstUnit + 0);
        glUniform1f(Loc("uThreshold"),
                    mHdrActive ? std::max(s.bloomThreshold, 1.0f) : std::min(s.bloomThreshold, 0.99f));
        glUniform1i(Loc("uHdr"), mHdrActive);
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
    Use(mHdrActive ? mCompositeHdr : (CvInt("gNewDawn.PostFx.Dither.Enabled", 1) ? mComposite : mCompositeNoDither));
    glUniform1i(Loc("uScene"), kFirstUnit + 0);
    glUniform1i(Loc("uBloom"), kFirstUnit + 1);
    glUniform1i(Loc("uUseBloom"), useBloom ? 1 : 0);
    glUniform1f(Loc("uBloomStrength"), s.bloomStrength);
    glUniform1i(Loc("uHdr"), mHdrActive);
    glUniform1i(Loc("uDither"), CvInt("gNewDawn.PostFx.Dither.Enabled", 1) != 0);
    glUniform1f(Loc("uExposure"), std::clamp(CvFloat("gNewDawn.HDR.Exposure", -0.2f), -8.0f, 8.0f));
    glUniform1i(Loc("uUseGrade"), s.gradeOn ? 1 : 0);
    glUniform1f(Loc("uContrast"), s.contrast);
    glUniform1f(Loc("uSaturation"), s.saturation);
    if (s.debugView == 1 && !mHdrActive) {
        glEnable(GL_SCISSOR_TEST);
        glScissor(w / 2, 0, w - w / 2, h);
    }
    Draw(mHdrActive ? mHdrDisplay.fbo : fb.fbo, w, h);
    CheckError("composite pass");
}

} // namespace Fast
#endif
