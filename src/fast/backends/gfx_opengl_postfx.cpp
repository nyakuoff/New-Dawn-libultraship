#if defined(ENABLE_OPENGL) && !defined(USE_OPENGLES)

#include "fast/backends/gfx_opengl_postfx.h"
#include "fast/newdawn_perf.h"
#include "fast/newdawn_wind.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <limits>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <stb_image.h>
#include <string>
#include <unordered_map>
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

static std::string CvString(const char* name, const char* def) {
    return Ship::Context::GetRawInstance()->GetConsoleVariables()->GetString(name, def);
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
// Special-sky look (defaults reproduce the normal sky): haze target, haze/grey mix, and a warm horizon glow.
uniform vec3 uHaze;
uniform float uHazeGrey;
uniform vec3 uGlow;
uniform float uGlowAmt;
uniform float uGlowFall;
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
// Night sky (gNewDawn.Sky.Night): the Milky Way from a real all-sky panorama (NASA SVS Deep Star Maps, equatorial
// coordinates) rotated by the game clock, plus a cool glow around the Moon. Stars are separate sharp point sprites.
uniform mat3 uCelestial;   // world -> J2000 equatorial
uniform sampler2D uMilkyWay;
uniform float uMilkyScale;
uniform float uMilkyGain;  // 0 = off, includes night/variant/weather visibility
uniform vec3 uMoonDir;
uniform float uMoonGlowAmt;
uniform float uMoonRadius;
float starExtinction(float e) {
    float airmass = 1.0 / (max(e, 0.0) * 0.95 + 0.06);
    return exp(-0.20 * (airmass - 1.0)) * smoothstep(0.0, 0.05, e);
}
vec3 nightSky(vec3 ray) {
    vec3 col = vec3(0.0);
    if (uMilkyGain > 0.0) {
        vec3 v = uCelestial * ray;
        float lat = 0.5 - asin(clamp(v.z, -1.0, 1.0)) / PI;
        float ra0 = atan(v.y, v.x) / (2.0 * PI);
        float ra1 = atan(-v.y, -v.x) / (2.0 * PI) + 0.5;
#ifdef SKY_CAPTURE
        vec3 tex = textureLod(uMilkyWay, vec2(ra0, lat), 5.0).rgb;
#else
        // Two phase-shifted longitudes: derivatives stay continuous across the atan seam.
        bool first = abs(ra0) < 0.25;
        vec3 tex = textureGrad(uMilkyWay, vec2(first ? ra0 : ra1, lat), vec2(first ? dFdx(ra0) : dFdx(ra1), dFdx(lat)),
                               vec2(first ? dFdy(ra0) : dFdy(ra1), dFdy(lat))).rgb;
#endif
        col += tex * uMilkyScale * uMilkyGain * vec3(0.97, 0.98, 1.02) * starExtinction(ray.y);
    }
    if (uMoonGlowAmt > 0.0) {
        // Soft cool glow outside the native disc (a world draw that stays), falling off over a few disc radii.
        float ang = acos(clamp(dot(ray, uMoonDir), -1.0, 1.0));
        float x = max(ang - uMoonRadius, 0.0) / uMoonRadius;
        float halo = 0.22 * exp(-x * 3.0) + 0.05 * exp(-x * 0.9) + 0.006 * exp(-x * 0.3);
        col += vec3(0.50, 0.64, 1.0) * halo * uMoonGlowAmt * smoothstep(-0.05, 0.12, uMoonDir.y);
    }
    return col;
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
    radiance = mix(radiance, mix(uHaze, vec3(dot(radiance, vec3(0.2126,0.7152,0.0722))), uHazeGrey), haze);
    radiance += uGlow * uGlowAmt * exp(-ray.y * uGlowFall); // apocalyptic/toxic horizon band, independent of the sun
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
    color += nightSky(ray);
#ifdef SKY_CAPTURE
    color = cubeFinite(color);
#endif
    oColor = vec4(color, 1.0);
}
)GLSL";

// Real star catalogue (HYG v4.1): each star is a point sprite with a Gaussian PSF, flux 10^(-0.4 mag), its B-V colour,
// atmospheric extinction/reddening near the horizon, scintillation that grows towards the horizon, and occlusion by the
// volumetric clouds drawn before it. Full resolution, so the field stays crisp under MSAA and camera motion.
static const char* kStarVertexShader = R"GLSL(
layout(location = 0) in vec3 aDir;
layout(location = 1) in float aMag;
layout(location = 2) in vec3 aCol;
uniform mat4 uViewProj;
uniform float uNdcXScale;
uniform float uFlipY;
uniform mat3 uCelestial;
uniform float uFlux;
uniform float uPix;
uniform float uTime;
uniform float uCloudsOn;
uniform vec2 uRes;
uniform sampler2D uClouds;
out vec3 vColor;
out float vSigma;
out vec2 vCenter;
void main() {
    vec3 d = transpose(uCelestial) * aDir;
    vec4 clip = uViewProj * vec4(d, 0.0);
    gl_PointSize = 1.0;
    vColor = vec3(0.0);
    vSigma = 1.0;
    vCenter = vec2(0.0);
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    if (clip.w <= 0.0 || d.y <= 0.0) return;
    vec2 ndc = clip.xy / clip.w / vec2(uNdcXScale, uFlipY);
    if (abs(ndc.x) > 1.15 || abs(ndc.y) > 1.15) return;
    float e = d.y;
    float sigmaN = 0.62 + 0.20 * clamp(2.0 - aMag, 0.0, 3.5);
    float peak = uFlux * pow(10.0, -0.4 * aMag) / (6.2831853 * sigmaN * sigmaN);
    float phase = fract(sin(dot(aDir, vec3(12.9898, 78.233, 37.719))) * 43758.5453) * 6.2831853;
    float amp = 0.04 + 0.42 * pow(clamp(1.0 - e * 2.5, 0.0, 1.0), 2.0);
    float tw = 1.0 + amp * (0.6 * sin(uTime * 6.5 + phase) + 0.4 * sin(uTime * 11.3 + phase * 1.7));
    float airmass = 1.0 / (e * 0.95 + 0.06);
    float ext = exp(-0.20 * (airmass - 1.0)) * smoothstep(0.0, 0.05, e);
    vec3 red = mix(vec3(1.0, 0.72, 0.50), vec3(1.0), clamp(e * 6.0, 0.0, 1.0));
    float vis = 1.0;
    if (uCloudsOn > 0.0) vis = 1.0 - textureLod(uClouds, ndc * 0.5 + 0.5, 1.5).a;
    vColor = aCol * red * peak * max(tw, 0.0) * ext * vis;
    float sigma = sigmaN * uPix;
    vSigma = sigma;
    vCenter = (ndc * 0.5 + 0.5) * uRes;
    gl_PointSize = clamp(ceil(sigma * 6.0) + 2.0, 3.0, 63.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)GLSL";

static const char* kStarFragmentShader = R"GLSL(
in vec3 vColor;
in float vSigma;
in vec2 vCenter;
out vec4 oColor;
void main() {
    vec2 p = gl_FragCoord.xy - vCenter;
    oColor = vec4(vColor * exp(-dot(p, p) / (2.0 * vSigma * vSigma)), 0.0);
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
// Special-sky look (defaults reproduce the normal cumulus): slab base/thickness in km, wind shear, domain
// turbulence, deck colour, and the warm light from below.
uniform vec2 uLayer;
uniform float uShear;
uniform float uTurb;
uniform vec3 uDeck;
uniform vec3 uGlow;
uniform float uGlowAmt;
#ifdef SKY_CAPTURE
uniform float uCloudStrength;
#endif
float noise3(vec3 p, float lod) {
    return textureLod(uNoise, p / 64.0, lod).r;
}
float density(vec3 p, float lod) {
    float h = (p.y - uLayer.x) / uLayer.y;
    if (h <= 0.0 || h >= 1.0) return 0.0;
    p.xz += uWind * (1.0 + uShear * h);
    if (uTurb > 0.0) p.xz += uTurb * (noise3(vec3(p.x * 0.7, p.y * 3.0 + 7.0, p.z * 0.7), lod + 1.0) - 0.5);
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
    float h = (p.y - uLayer.x) / uLayer.y;
    if (h <= 0.0 || h >= 1.0) return 0.0;
    p.xz += uWind * (1.0 + uShear * h);
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
    float start = max((uLayer.x-origin.y)/ray.y, 0.0);
    float end = min((uLayer.x+uLayer.y-origin.y)/ray.y, start + 45.0);
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
            float h = clamp((p.y-uLayer.x)/uLayer.y, 0.0, 1.0);
            vec3 ambient = mix(vec3(0.085,0.11,0.16), vec3(0.20,0.25,0.33), h) * day +
                           vec3(0.0015,0.0025,0.005) * (1.0-day);
            vec3 lighting = ambient + sunColor * phase * sunTrans * (0.8 + 0.2*(1.0-exp(-d*2.0)));
            // Overcast snow deck: a bright, low-contrast layer lit by the diffuse fog colour. Its base is a
            // little darker, and thick cores shade softly instead of blackening.
            vec3 deck = uDeck * (0.68 + 0.30*h) * (0.74 + 0.26*sunTrans) * (1.0 - 0.22*smoothstep(0.25, 1.0, d));
            // Light from below: the horizon glow reaches cloud bases and thin edges, thick cores stay dark.
            // Thin edges and gaps glow, thick cores stay dark, so the deck reads as a silhouette against the glow.
            deck += uGlow * uGlowAmt * (0.9 * (1.0 - smoothstep(0.05, 0.55, d)) * (0.45 + 0.55*(1.0-h)) +
                                        0.25 * pow(1.0-h, 2.0) * exp(-d*2.0));
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
// Cascaded sun shadows
// ---------------------------------------------------------------------------------------------------------------------

// Camera-space point at NDC (x, y, z) through the exact inverse projection (same layout the shaders read).
static void UnprojectNdc(const NewDawnPostFxParams& p, float x, float y, float z, float out[3]) {
    const float ndc[4] = { x * p.shadowNdcXScale, y * p.shadowFlipY, z, 1.0f };
    float w[4] = {};
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            w[j] += ndc[i] * p.shadowInvViewProj[i * 4 + j];
        }
    }
    const float inv = std::fabs(w[3]) > 1e-12f ? 1.0f / w[3] : 1.0f;
    for (int j = 0; j < 3; j++) {
        out[j] = w[j] * inv;
    }
}

// One tile of the atlas per cascade, fitted to a bounding sphere of the slice of the view frustum it covers. The sphere
// does not change size when the camera turns and the centre snaps to the texel grid, so shadow edges do not shimmer.
// Slices overlap by `blend` of the nearer cascade, which is where the two are cross-faded.
void PostFxGL::BuildCascades(const NewDawnPostFxParams& p, int atlasSize) {
    mCsm = Cascades{};
    const int n = std::clamp(p.shadowCascades, 1, 4);
    if (n <= 1 || !p.shadowUseExactProj || atlasSize < 512 || !mShadowApplyCsm.id || !mShadowApplyHdrCsm.id ||
        !mShaftMarchCsm.id || !mFogMarchCsm.id || !mGiGatherCsm.id) {
        return;
    }
    constexpr float kGuardTexels = 5.0f;     // PCF radius 2.2 + bilinear + margin, never filtered across a tile edge
    constexpr float kLegacyTexel = 2.0f * 1500.0f / 4096.0f; // the single map's texel; bias constants are tuned for it
    constexpr float kLegacyRange = 5000.0f;  // and its depth range
    constexpr float kLambda = 0.85f;         // practical split scheme: mostly logarithmic
    mCsm.count = n;
    mCsm.tile = atlasSize / 2;
    mCsm.blend = 0.12f;

    float nearC[4][3], farC[4][3], nc[3], fc[3];
    const float sx[4] = { -1.0f, 1.0f, 1.0f, -1.0f }, sy[4] = { -1.0f, -1.0f, 1.0f, 1.0f };
    for (int k = 0; k < 4; k++) {
        UnprojectNdc(p, sx[k], sy[k], -1.0f, nearC[k]);
        UnprojectNdc(p, sx[k], sy[k], 1.0f, farC[k]);
    }
    UnprojectNdc(p, 0.0f, 0.0f, -1.0f, nc);
    UnprojectNdc(p, 0.0f, 0.0f, 1.0f, fc);
    float fwd[3] = { fc[0] - nc[0], fc[1] - nc[1], fc[2] - nc[2] };
    Normalize3(fwd, fwd);
    const float zNear = std::max(p.zNear, 1.0f), zFar = std::max(p.zFar, zNear + 1.0f);
    for (int i = 0; i < 3; i++) {
        mCsm.fwd[i] = fwd[i];
        mCsm.eye[i] = nc[i] - fwd[i] * zNear;
    }
    float off[3] = { farC[0][0] - fc[0], farC[0][1] - fc[1], farC[0][2] - fc[2] };
    const float h2 = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]) / (zFar * zFar); // tan^2 of the half diagonal

    const float distance = std::clamp(p.shadowDistance, zNear + 100.0f, zFar);
    const float start = std::max(zNear, 20.0f);
    for (int i = 0; i < n; i++) {
        const float f = (float)(i + 1) / (float)n;
        const float logSplit = start * std::pow(distance / start, f);
        const float linSplit = start + (distance - start) * f;
        mCsm.split[i] = (i == n - 1) ? distance : kLambda * logSplit + (1.0f - kLambda) * linSplit;
    }

    float dir[3] = { p.shadowSunDir[0], p.shadowSunDir[1], p.shadowSunDir[2] };
    float up[3] = { 0.0f, 1.0f, 0.0f };
    if (std::fabs(dir[1]) > 0.98f) {
        up[1] = 0.0f;
        up[2] = 1.0f;
    }
    float forward[3] = { -dir[0], -dir[1], -dir[2] };
    Normalize3(forward, forward);
    float right[3], realUp[3];
    Cross3(forward, up, right);
    Normalize3(right, right);
    Cross3(right, forward, realUp);

    const float atlas = (float)atlasSize, tile = (float)mCsm.tile;
    for (int i = 0; i < n; i++) {
        const float a = (i == 0) ? zNear : mCsm.split[i - 1] * (1.0f - mCsm.blend);
        const float b = mCsm.split[i];
        float pts[8][3];
        for (int k = 0; k < 4; k++) {
            const float ta = (a - zNear) / (zFar - zNear), tb = (b - zNear) / (zFar - zNear);
            for (int j = 0; j < 3; j++) {
                pts[k][j] = nearC[k][j] + (farC[k][j] - nearC[k][j]) * ta;
                pts[4 + k][j] = nearC[k][j] + (farC[k][j] - nearC[k][j]) * tb;
            }
        }
        const float z = std::clamp((a + b) * (1.0f + h2) * 0.5f, a, b);
        float center[3];
        for (int j = 0; j < 3; j++) {
            center[j] = mCsm.eye[j] + fwd[j] * z;
        }
        float radiusSq = 0.0f;
        for (auto& q : pts) {
            const float d[3] = { q[0] - center[0], q[1] - center[1], q[2] - center[2] };
            radiusSq = std::max(radiusSq, d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        }
        const float half = std::sqrt(radiusSq) * tile / (tile - 2.0f * kGuardTexels);
        const float texel = 2.0f * half / tile;
        const float cr = center[0] * right[0] + center[1] * right[1] + center[2] * right[2];
        const float cu = center[0] * realUp[0] + center[1] * realUp[1] + center[2] * realUp[2];
        const float sr = std::round(cr / texel) * texel - cr, su = std::round(cu / texel) * texel - cu;
        for (int j = 0; j < 3; j++) {
            center[j] += right[j] * sr + realUp[j] * su;
        }

        // Depth range hugs the tile: casters in front of the near plane are kept by depth clamping when drawn.
        const float eyeDist = half + 100.0f;
        const float eye[3] = { center[0] + dir[0] * eyeDist, center[1] + dir[1] * eyeDist, center[2] + dir[2] * eyeDist };
        const float nearZ = 1.0f, farZ = eyeDist + half + 100.0f;
        const float view[16] = {
            right[0], realUp[0], -forward[0], 0.0f, right[1], realUp[1], -forward[1], 0.0f,
            right[2], realUp[2], -forward[2], 0.0f, -right[0] * eye[0] - right[1] * eye[1] - right[2] * eye[2],
            -realUp[0] * eye[0] - realUp[1] * eye[1] - realUp[2] * eye[2],
            forward[0] * eye[0] + forward[1] * eye[1] + forward[2] * eye[2], 1.0f,
        };
        const float ortho[16] = {
            1.0f / half, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f / half, 0.0f, 0.0f, 0.0f, 0.0f,
            -2.0f / (farZ - nearZ), 0.0f, 0.0f, 0.0f, -(farZ + nearZ) / (farZ - nearZ), 1.0f,
        };
        float lightVP[16];
        Mat4Multiply(ortho, view, lightVP);
        // Clip space of the whole light volume -> this tile of the atlas.
        const float ox = (float)(i & 1) * tile, oy = (float)(i >> 1) * tile;
        const float s = tile / atlas;
        const float atlasXf[16] = { s, 0.0f, 0.0f, 0.0f, 0.0f, s, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                    2.0f * ox / atlas + s - 1.0f, 2.0f * oy / atlas + s - 1.0f, 0.0f, 1.0f };
        Mat4Multiply(atlasXf, lightVP, mCsm.vp[i]);
        mCsm.rect[i][0] = (ox + kGuardTexels) / atlas;
        mCsm.rect[i][1] = (oy + kGuardTexels) / atlas;
        mCsm.rect[i][2] = (ox + tile - kGuardTexels) / atlas;
        mCsm.rect[i][3] = (oy + tile - kGuardTexels) / atlas;
        mCsm.texelScale[i] = texel / kLegacyTexel;
        mCsm.biasScale[i] = kLegacyRange * mCsm.texelScale[i] / (farZ - nearZ);
    }
}

void PostFxGL::SetCascadeUniforms() {
    auto loc = [this](const char* name) {
        GLint l = Loc(name);
        return l >= 0 ? l : Loc((std::string(name) + "[0]").c_str()); // atmos keeps the array suffix
    };
    glUniform1i(Loc("uCsmCount"), mCsm.count);
    if (mCsm.count <= 1) {
        return;
    }
    glUniformMatrix4fv(loc("uCsmVP"), 4, GL_FALSE, &mCsm.vp[0][0]);
    glUniform4fv(loc("uCsmRect"), 4, &mCsm.rect[0][0]);
    glUniform4fv(Loc("uCsmSplit"), 1, mCsm.split);
    glUniform4fv(Loc("uCsmBiasScale"), 1, mCsm.biasScale);
    glUniform4fv(Loc("uCsmTexelScale"), 1, mCsm.texelScale);
    glUniform1f(Loc("uCsmBlend"), mCsm.blend);
    glUniform3fv(Loc("uCsmEye"), 1, mCsm.eye);
    glUniform3fv(Loc("uCsmFwd"), 1, mCsm.fwd);
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

// GI uses the opaque lit colour once, before fog/particles/water. There is no recursive colour feedback.
static const char* kGiGeometryLib = R"GLSL(
uniform mat4 uInvViewProj;
uniform float uNdcXScale;
uniform float uFlipY;
vec3 giPosition(vec2 uv, float d) {
    vec4 p = uInvViewProj * vec4((uv*2.0-1.0)*vec2(uNdcXScale,uFlipY), d*2.0-1.0, 1.0);
    return p.xyz/p.w;
}
vec3 giNormal(ivec2 p, vec3 P) {
    ivec2 l = max(p-ivec2(1,0),ivec2(0)), r = min(p+ivec2(1,0),uSize-1);
    ivec2 b = max(p-ivec2(0,1),ivec2(0)), t = min(p+ivec2(0,1),uSize-1);
    vec3 L = giPosition((vec2(l)+0.5)/vec2(uSize),fetchDepth(l));
    vec3 R = giPosition((vec2(r)+0.5)/vec2(uSize),fetchDepth(r));
    vec3 B = giPosition((vec2(b)+0.5)/vec2(uSize),fetchDepth(b));
    vec3 T = giPosition((vec2(t)+0.5)/vec2(uSize),fetchDepth(t));
    vec3 dx = dot(R-P,R-P)<dot(P-L,P-L) ? R-P : P-L;
    vec3 dy = dot(T-P,T-P)<dot(P-B,P-B) ? T-P : P-B;
    vec3 n = cross(dx,dy)*uFlipY;
    return n/max(length(n),1e-6);
}
// Same, over a footprint of k pixels each way (smooths faceted low-poly shapes); the smaller one-sided step still
// keeps silhouettes from blending with the background.
vec3 giNormalWide(ivec2 p, vec3 P, int k) {
    ivec2 l = max(p-ivec2(k,0),ivec2(0)), r = min(p+ivec2(k,0),uSize-1);
    ivec2 b = max(p-ivec2(0,k),ivec2(0)), t = min(p+ivec2(0,k),uSize-1);
    vec3 L = giPosition((vec2(l)+0.5)/vec2(uSize),fetchDepth(l));
    vec3 R = giPosition((vec2(r)+0.5)/vec2(uSize),fetchDepth(r));
    vec3 B = giPosition((vec2(b)+0.5)/vec2(uSize),fetchDepth(b));
    vec3 T = giPosition((vec2(t)+0.5)/vec2(uSize),fetchDepth(t));
    vec3 dx = dot(R-P,R-P)<dot(P-L,P-L) ? R-P : P-L;
    vec3 dy = dot(T-P,T-P)<dot(P-B,P-B) ? T-P : P-B;
    vec3 n = cross(dx,dy)*uFlipY;
    return n/max(length(n),1e-6);
}
// One-sided normal for emitters: the hemisphere test tolerates silhouette error, and it costs half as much.
vec3 giNormalFast(ivec2 p, vec3 P) {
    vec3 R = giPosition((vec2(p+ivec2(1,0))+0.5)/vec2(uSize),fetchDepth(p+ivec2(1,0)));
    vec3 T = giPosition((vec2(p+ivec2(0,1))+0.5)/vec2(uSize),fetchDepth(p+ivec2(0,1)));
    vec3 n = cross(R-P,T-P)*uFlipY;
    return n/max(length(n),1e-6);
}
const vec3 giLuma = vec3(0.2126,0.7152,0.0722);
)GLSL";

static const char* kGiGatherShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uScene;
uniform sampler2DShadow uGiShadow;
uniform mat4 uLightViewProj;
uniform vec3 uSunDir;
uniform float uNormalBias;
uniform int uSunShadow;
uniform int uFrame;
float giIgn(vec2 p) { return fract(52.9829189*fract(dot(p,vec2(0.06711056,0.00583715)))); }
#ifdef CSM
float csmTap(vec3 uvz) { return texture(uGiShadow,uvz); }
#endif
void main() {
    ivec2 p = min(ivec2(vUv*vec2(uSize)),uSize-1);
    float d = fetchDepth(p);
    oColor = vec4(0.0,0.0,0.0,d);
    if (d>=0.99999) return;
    vec2 uv = (vec2(p)+0.5)/vec2(uSize);
    vec3 P = giPosition(uv,d), N = giNormal(p,P);
    float z = linDepth(d);
    // 140 units is about two Link heights. Project through the exact interpolated camera.
    vec3 side = giPosition(uv+vec2(1.0/float(uSize.x),0.0),d)-P;
    float radius = min(160.0,140.0/max(length(side),0.001));
    // Direct sun visibility from the sun shadow map: bounce is mostly a shade effect.
    float shade = 1.0;
    if (uSunShadow != 0) {
        shade = dot(N,uSunDir) <= 0.0 ? 1.0 : 0.0;
        if (shade == 0.0) {
#ifdef CSM
            float s = csmLit(P,N,z,uNormalBias,0.004);
            if (s >= 0.0) shade = 1.0-s;
#else
            vec4 lc = uLightViewProj*vec4(P+N*uNormalBias,1.0);
            vec3 sc = lc.xyz/lc.w*0.5+0.5;
            if (all(greaterThan(sc.xy,vec2(0.0))) && all(lessThan(sc.xy,vec2(1.0))) && sc.z<1.0)
                shade = 1.0-texture(uGiShadow,vec3(sc.xy,sc.z-0.004));
#endif
        }
    }
    vec3 sum = vec3(0.0);
    // Interleaved gradient noise, re-seeded every frame: the temporal filter integrates the pattern away.
    vec2 jp = gl_FragCoord.xy+5.588238*float(uFrame&63);
    float rotation = giIgn(jp)*6.2831853, jr = giIgn(jp.yx+17.0);
    for (int i=0;i<16;++i) {
        float a = float(i)*2.39996323+rotation;
        float r = sqrt((float(i)+jr)/16.0);
        vec2 qUv = uv+vec2(cos(a),sin(a))*radius*r/vec2(uSize);
        if (any(lessThan(qUv,vec2(0.0))) || any(greaterThanEqual(qUv,vec2(1.0)))) continue;
        ivec2 q = ivec2(qUv*vec2(uSize));
        float qd = fetchDepth(q);
        if (qd>=0.99999) continue;
        qUv = (vec2(q)+0.5)/vec2(uSize);
        vec3 Q = giPosition(qUv,qd), V = Q-P;
        float dist = length(V);
        if (dist<2.0 || dist>140.0) continue;
        vec3 dir = V/dist;
        // Both surfaces must face the transfer. Coplanar floor samples contribute zero.
        float receive = max(dot(N,dir)-0.035,0.0);
        if (receive<=0.0) continue;
        vec3 M = giNormalFast(q,Q);
        float emit = max(dot(M,-dir),0.0);
        // Reject an occluding surface along the segment rather than leaking through foreground edges.
        vec2 midUv = mix(uv,qUv,0.5);
        float midZ = linDepth(texture(uDepth,midUv).r);
        float segmentZ = 2.0/(1.0/z+1.0/linDepth(qd));
        if (midZ < segmentZ-max(3.0,dist*0.08)) continue;
        float edge = smoothstep(0.0,0.035,min(min(qUv.x,qUv.y),min(1.0-qUv.x,1.0-qUv.y)));
        float weight = receive*emit*pow(1.0-dist/140.0,2.0)*edge;
        vec3 c = max(texelFetch(uScene,q,0).rgb,vec3(0.0));
        // Luminance-preserving firefly limit, before gathering: retain coloured wall/grass hue.
        c *= min(1.0,0.6/max(dot(c,giLuma),0.0001));
        sum += c*weight;
    }
    oColor.rgb = sum*(40.0/16.0)*smoothstep(1.5,5.0,radius)*mix(0.12,1.0,shade);
}
)GLSL";

static const char* kGiFilterShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uGiRaw;
uniform sampler2D uGiHistory;
uniform mat4 uPreviousViewProj;
uniform vec2 uPreviousScale;
uniform int uHistoryValid;
void main() {
    ivec2 hs = textureSize(uGiRaw,0);
    ivec2 p = min(ivec2(vUv*vec2(uSize)),uSize-1);
    float d = fetchDepth(p);
    oColor = vec4(0.0,0.0,0.0,d);
    if (d>=0.99999) return;
    float z = linDepth(d);
    vec3 sum=vec3(0.0), lo=vec3(1e6), hi=vec3(0.0);
    float ws=0.0;
    ivec2 center = ivec2(gl_FragCoord.xy);
    for (int y=-2;y<=2;++y) for (int x=-2;x<=2;++x) {
        ivec2 q=clamp(center+ivec2(x,y),ivec2(0),hs-1);
        vec4 g=texelFetch(uGiRaw,q,0);
        float dz=(linDepth(g.a)-z)/max(2.0,z*0.015);
        float w=exp(-dz*dz-float(x*x+y*y)*0.22);
        sum+=g.rgb*w;ws+=w;
        if (w>0.05) { lo=min(lo,g.rgb);hi=max(hi,g.rgb); }
    }
    vec3 current=sum/max(ws,0.0001);
    vec3 P=giPosition((vec2(p)+0.5)/vec2(uSize),d);
    vec4 clip=uPreviousViewProj*vec4(P,1.0);
    vec3 ndc=clip.xyz/max(clip.w,0.0001);
    vec2 prevUv=ndc.xy/uPreviousScale*0.5+0.5;
    if (uHistoryValid!=0 && clip.w>0.0 && all(greaterThan(prevUv,vec2(0.0))) && all(lessThan(prevUv,vec2(1.0)))) {
        // Bilinear reprojection, each tap rejected by depth error; history is clamped to the (widened) current
        // neighbourhood so moving actors and changing lights cannot leave trails.
        vec2 hp=prevUv*vec2(hs)-0.5;
        ivec2 b=ivec2(floor(hp));vec2 f=fract(hp);
        float expected=linDepth(ndc.z*0.5+0.5);
        vec3 acc=vec3(0.0);float wsum=0.0;
        for (int y=0;y<2;++y) for (int x=0;x<2;++x) {
            vec4 old=texelFetch(uGiHistory,clamp(b+ivec2(x,y),ivec2(0),hs-1),0);
            float bw=mix(1.0-f.x,f.x,float(x))*mix(1.0-f.y,f.y,float(y));
            float error=abs(linDepth(old.a)-expected);
            bw*=1.0-smoothstep(max(1.0,expected*0.002),max(3.0,expected*0.008),error);
            acc+=old.rgb*bw;wsum+=bw;
        }
        if (wsum>0.0001) {
            vec3 hist=clamp(acc/wsum,lo*0.75,hi*1.3+0.002);
            current=mix(current,hist,0.9*min(wsum,1.0));
        }
    }
    oColor.rgb=current;
}
)GLSL";

static const char* kGiApplyShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uScene;
uniform sampler2D uGiHistory;
uniform float uStrength;
uniform float uDebug;
void main() {
    ivec2 p=ivec2(gl_FragCoord.xy);
    vec4 scene=texelFetch(uScene,p,0);
    float d=fetchDepth(p),z=linDepth(d);
    // Additive pass (GL_ONE, GL_ONE): keeps MSAA samples intact and writes nothing on sky.
    oColor=vec4(0.0);
    if (d>=0.99999) return;
    ivec2 hs=textureSize(uGiHistory,0);
    vec2 hpos=vUv*vec2(hs)-0.5;
    ivec2 base=ivec2(floor(hpos));vec2 f=fract(hpos);
    vec3 sum=vec3(0.0);float ws=0.0;
    for (int y=0;y<2;++y) for (int x=0;x<2;++x) {
        ivec2 q=clamp(base+ivec2(x,y),ivec2(0),hs-1);
        vec4 g=texelFetch(uGiHistory,q,0);
        float dz=(linDepth(g.a)-z)/max(1.0,z*0.01);
        float w=exp(-dz*dz)*mix(1.0-f.x,f.x,float(x))*mix(1.0-f.y,f.y,float(y));
        sum+=g.rgb*w;ws+=w;
    }
    float Y=max(dot(scene.rgb,giLuma),0.0);
    // Albedo is unavailable: bounded lit-colour proxy preserves receiver hue and extinguishes crushed blacks.
    vec3 receiver=clamp(scene.rgb/max(Y,0.01),vec3(0.0),vec3(2.0));
    vec3 bounce=sum/max(ws,0.0001)*mix(vec3(1.0),receiver,0.5)*uStrength;
    bounce*=smoothstep(0.002,0.025,Y)*(1.0-smoothstep(0.5,1.4,Y));
    bounce*=min(1.0,0.9*Y/max(dot(bounce,giLuma),0.00001));
    oColor.rgb=bounce*uDebug;
}
)GLSL";


// Point-light shadows: a distance cube map. Each fragment writes its distance to the light, depth-tested against the
// nearest one, so the lookup in the local-light pass is a plain compare in world units (no projection maths).
static const char* kPointShadowVertex = R"GLSL(
layout(location = 0) in vec3 aPos;
uniform mat4 uVP;
out vec3 vWorld;
void main() {
    vWorld = aPos;
    gl_Position = uVP * vec4(aPos, 1.0);
}
)GLSL";

static const char* kPointShadowFragment = R"GLSL(
in vec3 vWorld;
uniform vec3 uLight;
out float oDist;
void main() {
    oDist = length(vWorld - uLight);
}
)GLSL";

static const char* kLocalLightShader = R"GLSL(
in vec2 vUv;
// Dual-source blend (GL_ONE, GL_SRC1_COLOR): result = oColor + framebuffer * oFactor, per MSAA sample.
layout(location = 0, index = 0) out vec4 oColor;
layout(location = 0, index = 1) out vec4 oFactor;
uniform sampler2D uScene;
uniform mat4 uViewProj;
uniform int uLightCount;
uniform vec4 uLight[24];    // world position, range
uniform vec3 uLightRgb[24]; // HDR radiance (strength and flicker already applied)
uniform float uLightFlame[24]; // 1 for fire lights (torches, braziers, candles): they light dark surfaces better, see uFlameExp
uniform float uFlameExp, uFlameScale; // fire light: albedo' = max(albedo, albedo^exp * scale)
uniform float uDebug;
uniform float uDarken;
uniform float uFogMul;
uniform float uFogOff;
uniform float uSoft; // 1 for the stencil-marked characters: their low-poly depth normals are smoothed
// Point-light shadow cubes (one per shadow-casting light; -1 = unused). Each holds the distance to the nearest caster.
uniform samplerCube uPsCube0;
uniform samplerCube uPsCube1;
uniform int uPsIdx0;
uniform int uPsIdx1;
uniform vec4 uPsLight0; // the cube's light: world position, far range
uniform vec4 uPsLight1;
uniform vec3 uPsParams; // light radius (penumbra size), cube texel angle, bias scale
uniform float uPsDebug;
float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
// Screen-space contact shadow along the segment to the light, from the opaque depth buffer: walls and actors between
// the surface and the flame occlude it. Samples outside the screen count as visible.
float localVisibility(vec3 P, vec3 L, float jitter) {
    // 12 steps with a small per-pixel offset; each hit only dims by a fraction, so edges become a soft penumbra
    // rather than a stippled binary mask.
    float hits = 0.0;
    for (int s = 0; s < 12; ++s) {
        vec3 sp = P + (L - P) * ((float(s) + 0.5 * jitter) * (0.92 / 12.0) + 0.02);
        vec4 clip = uViewProj * vec4(sp, 1.0);
        if (clip.w <= 0.0) continue;
        vec2 uv = (clip.xy / clip.w / vec2(uNdcXScale, uFlipY)) * 0.5 + 0.5;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) continue;
        float sceneZ = linDepth(fetchDepth(ivec2(uv * vec2(uSize))));
        float diff = clip.w - sceneZ;
        hits += smoothstep(max(1.0, clip.w * 0.006), max(4.0, clip.w * 0.02), diff) * (1.0 - step(70.0, diff));
    }
    return 1.0 - min(1.0, hits * 0.22);
}
// Contact-hardening soft shadow from a distance cube: a few taps search for blockers, whose distance sets the
// penumbra (wide far behind the occluder, sharp where it touches), then a rotated Vogel-disc PCF filters it.
float pointShadowVis(samplerCube cube, vec4 Lr, vec3 P, vec3 N, float rnd) {
    vec3 v = P + N * 1.5 - Lr.xyz;
    float dist = length(v);
    if (dist < 3.0 || dist >= Lr.w) return 1.0;
    vec3 dir = v / dist;
    float texel = uPsParams.y * dist;
    float cosN = clamp(dot(N, -dir), 0.0, 1.0);
    float bias = texel * (1.2 + 2.5 * (1.0 - cosN)) + 0.8 * uPsParams.z;
    vec3 up = abs(dir.y) < 0.95 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, dir));
    vec3 B = cross(dir, T);
    float rot = rnd * 6.2831853;
    float lightAng = uPsParams.x / dist;
    float searchAng = clamp(lightAng * 3.0, 0.03, 0.22);
    float blocker = 0.0, nb = 0.0;
    for (int i = 0; i < 6; ++i) {
        float r = sqrt((float(i) + 0.5) / 6.0), a = float(i) * 2.399963 + rot;
        float ds = texture(cube, normalize(dir + (T * cos(a) + B * sin(a)) * (r * searchAng))).r;
        if (ds < dist - bias) { blocker += ds; nb += 1.0; }
    }
    if (nb < 0.5) return 1.0;
    float avg = blocker / nb;
    float pen = uPsParams.x * (dist - avg) / max(avg, 1.0);
    float ang = clamp(pen / dist, uPsParams.y * 1.5, 0.25);
    float soft = texel * 2.0 + 1.0;
    float lit = 0.0;
    for (int i = 0; i < 12; ++i) {
        float r = sqrt((float(i) + 0.5) / 12.0), a = float(i) * 2.399963 + rot;
        float ds = texture(cube, normalize(dir + (T * cos(a) + B * sin(a)) * (r * ang))).r;
        lit += 1.0 - smoothstep(0.0, soft, dist - bias - ds);
    }
    return lit / 12.0;
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 scene = texelFetch(uScene, p, 0);
    oColor = vec4(0.0);
    oFactor = vec4(1.0);
    float d = fetchDepth(p);
    if (d >= 0.99999) return;
    vec3 P = giPosition((vec2(p) + 0.5) / vec2(uSize), d);
    vec3 N = giNormal(p, P);
    if (uSoft > 0.0) {
        // Characters are low-poly: the 1-pixel depth normal is faceted. Average it with wide-footprint normals so the
        // light rolls over the body instead of snapping between triangles.
        vec3 wide = N;
        for (int k = 1; k <= 2; ++k) wide += giNormalWide(p, P, 3 * k);
        N = normalize(wide);
    }
    float jitter = ign(gl_FragCoord.xy);
    if (uPsDebug > 0.5) {
        // gNewDawn.Shadows.PointDebug=1: the first cube's visibility on its own (white lit, black shadowed, blue outside
        // its range, nothing else drawn); 2 shows the cube's stored distance.
        vec3 dv = P - uPsLight0.xyz;
        float dd = length(dv);
        vec3 c = vec3(0.0, 0.0, 0.4);
        if (uPsIdx0 >= 0 && dd < uPsLight0.w) {
            float vis = pointShadowVis(uPsCube0, uPsLight0, P, N, jitter);
            c = uPsDebug < 1.5 ? vec3(vis) : vec3(texture(uPsCube0, dv / max(dd, 0.001)).r / uPsLight0.w);
        }
        oColor = vec4(c, 0.0);
        oFactor = vec4(0.0);
        return;
    }
    vec3 E = vec3(0.0), Ef = vec3(0.0);
    for (int i = 0; i < uLightCount; ++i) {
        vec3 V = uLight[i].xyz - P;
        float R = uLight[i].w, d2 = dot(V, V);
        if (d2 >= R * R) continue;
        float dist = sqrt(d2);
        vec3 L = V / max(dist, 0.001);
        // Smooth range window (Karis) over a softened inverse-square core; ~1 at the flame, 0 at the range.
        float x = dist / R, window = clamp(1.0 - x * x * x * x, 0.0, 1.0);
        // Fires: a steeper window, (1 - x^2)^3 after squaring, so the pool is gone well before its range and never ends in a rim.
        if (uLightFlame[i] > 0.5) window = pow(clamp(1.0 - x * x, 0.0, 1.0), 1.5);
        float core = R * (uLightFlame[i] > 0.5 ? 0.13 : 0.2); // fires fall off faster: a pool, not a wash
        float atten = window * window * core * core / (d2 + core * core);
        // Wrapped Lambert: the depth-derived normal is faceted, so the terminator is softened.
        float wrap = mix(0.3, 0.7, uSoft);
        float diffuse = clamp((dot(N, L) + wrap) / (1.0 + wrap), 0.0, 1.0);
        float lit = atten * diffuse;
        // Fires carry a high intensity, so the early-out below would leave a ring where it cuts in: subtract the
        // threshold instead of clipping at it, which makes the pool end continuously.
        if (uLightFlame[i] > 0.5) lit = max(lit - 0.0018, 0.0);
        if (lit < 0.002 && uLightFlame[i] < 0.5) continue;
        if (lit <= 0.0) continue;
        float vis = localVisibility(P, uLight[i].xyz, jitter);
        if (i == uPsIdx0) vis *= pointShadowVis(uPsCube0, uPsLight0, P, N, jitter);
        else if (i == uPsIdx1) vis *= pointShadowVis(uPsCube1, uPsLight1, P, N, jitter);
        if (uLightFlame[i] > 0.5) Ef += uLightRgb[i] * (lit * vis);
        else E += uLightRgb[i] * (lit * vis);
    }
    // Bounded: walls beside a torch glow without clipping, and the cap is per channel so the hue survives.
    const float cap = 2.4;
    {
        vec3 T = E + Ef;
        vec3 kc = cap / (cap + T);
        E *= kc;
        Ef *= kc;
    }
    vec3 c = max(scene.rgb, vec3(0.0));
    float Y = dot(c, giLuma);
    // The painted scene is the albedo: its square root lifts crushed blacks while keeping hue. A bright, already-lit
    // surface (daylight) needs the extra light less.
    vec3 albedo = sqrt(c) * 0.9 + 0.03;
    float need = 1.0 - 0.75 * smoothstep(0.35, 1.0, Y);
    // Contrast: pull the unlit scene down (night / interior) so the lights carry it. Surfaces the lights reach keep
    // their authored brightness; the player gets a soft exemption and nothing drops below a small floor.
    // Native fog is a light of its own: surfaces it already hides keep the fog colour instead of being pulled into a
    // dark band against the (undarkened) sky. Characters (stencil-marked draws) get a reduced uDarken from the second draw.
    float fog = clamp((d * 2.0 - 1.0) * uFogMul + uFogOff, 0.0, 255.0) / 255.0;
    float D = uDarken * (1.0 - fog);
    float keep = smoothstep(0.0, 0.45, dot(E + Ef, giLuma));
    vec3 moon = mix(vec3(1.0), vec3(0.90, 0.97, 1.10), min(D * 1.3, 1.0)); // cool, dim fill
    vec3 base = c * mix(1.0 - D, 1.0, keep) * mix(moon, vec3(1.0), keep);
    base = max(base, min(c, vec3(0.0055)));
    if (uDebug > 0.0) {
        oColor = vec4((E + Ef) * uDebug, 0.0);
        oFactor = vec4(0.0);
        return;
    }
    // Fire light on a dark surface: the dark authored colour would swallow it, so fires see a gently lifted albedo (only
    // where their light reaches; the unlit scene between the lights keeps its darkness).
    // The lift keeps the surface's own contrast: a gamma lift (exponent < 1) of the texture albedo, never a constant colour.
    vec3 albedoF = max(albedo, pow(albedo, vec3(uFlameExp)) * uFlameScale);
    oColor = vec4((albedo * E + albedoF * Ef) * need, 0.0);
    oFactor = vec4(clamp(base / max(c, vec3(0.0001)), 0.0, 1.0), 1.0);
}
)GLSL";


// Interior polish: light through window and door openings. Each opening is an oriented rectangle (centre, inward normal,
// horizontal axis, half extents) recorded from the draw stream; the beam leaves it along a direction set by the time of
// day. Dual-source blend: result = shaft in-scatter + framebuffer * pool boost, per MSAA sample.
//  - pool: where the beam meets a surface (depth-reconstructed), lit by a soft-edged, pane-barred projection of the
//    opening, shadowed by a short screen-space march toward it;
//  - shaft: the view ray's path through the beam volume, forward-scattered (Henyey-Greenstein) and clamped at the
//    first surface.
static const char* kInteriorLightShader = R"GLSL(
in vec2 vUv;
layout(location = 0, index = 0) out vec4 oColor;
layout(location = 0, index = 1) out vec4 oFactor;
uniform sampler2D uScene;
uniform mat4 uViewProj;
uniform vec3 uEye;
uniform int uCount;
uniform vec4 uWin0[12]; // centre xyz, half width
uniform vec4 uWin1[12]; // inward normal xyz, half height
uniform vec4 uWin2[12]; // horizontal axis xyz, panes (x + 16 y)
uniform vec4 uWinL[12]; // beam direction xyz (into the room), unused
uniform vec3 uColor;    // linear light colour including intensity and user strength
uniform float uPool, uShaft;
float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
float hg(float c) {
    const float g = 0.5;
    return 0.25 * (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * c, 1.5);
}
// Soft rectangular membership with pane bars; q = (u, v) from the opening centre, soft = penumbra half width.
float openMask(vec2 q, vec2 h, float soft, float panes) {
    vec2 a = h - abs(q);
    float m = smoothstep(-soft, soft, a.x) * smoothstep(-soft, soft, a.y);
    float nu = mod(panes, 16.0), nv = floor(panes / 16.0);
    if (nu > 1.5) {
        float x = (q.x / h.x * 0.5 + 0.5) * nu;
        float d = abs(x - floor(x + 0.5)) * 2.0 * h.x / nu;
        m *= (x > 0.5 && x < nu - 0.5) ? smoothstep(1.2, 1.2 + soft, d) : 1.0;
    }
    if (nv > 1.5) {
        float y = (q.y / h.y * 0.5 + 0.5) * nv;
        float d = abs(y - floor(y + 0.5)) * 2.0 * h.y / nv;
        m *= (y > 0.5 && y < nv - 0.5) ? smoothstep(1.2, 1.2 + soft, d) : 1.0;
    }
    return m;
}
// Visibility of the segment from surface point P back to its source point on the opening (screen-space depth march).
float openVisibility(vec3 P, vec3 S, float jitter) {
    float hits = 0.0;
    for (int k = 0; k < 10; ++k) {
        vec3 sp = P + (S - P) * ((float(k) + jitter) * 0.085 + 0.05);
        vec4 clip = uViewProj * vec4(sp, 1.0);
        if (clip.w <= 0.0) continue;
        vec2 uv = (clip.xy / clip.w / vec2(uNdcXScale, uFlipY)) * 0.5 + 0.5;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) continue;
        float sceneZ = linDepth(fetchDepth(ivec2(uv * vec2(uSize))));
        float diff = clip.w - sceneZ;
        hits += smoothstep(max(2.0, clip.w * 0.01), max(8.0, clip.w * 0.03), diff) * (1.0 - step(160.0, diff));
    }
    return 1.0 - min(1.0, hits * 0.3);
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 scene = texelFetch(uScene, p, 0);
    oColor = vec4(0.0);
    oFactor = vec4(1.0);
    float d = fetchDepth(p);
    bool sky = d >= 0.99999;
    vec2 uv = (vec2(p) + 0.5) / vec2(uSize);
    vec3 P = giPosition(uv, sky ? 0.99999 : d);
    vec3 rd = P - uEye;
    float rl = length(rd);
    rd /= max(rl, 1e-3);
    float tEnd = min(rl, 1600.0);
    float jitter = ign(gl_FragCoord.xy);
    vec3 N = sky ? vec3(0.0, 1.0, 0.0) : giNormal(p, P);
    vec3 pool = vec3(0.0), shaft = vec3(0.0);
    for (int i = 0; i < uCount; ++i) {
        vec3 C = uWin0[i].xyz, n = uWin1[i].xyz, U = uWin2[i].xyz, L = uWinL[i].xyz;
        vec2 h = vec2(uWin0[i].w, uWin1[i].w);
        vec3 V = cross(n, U);
        float ln = dot(L, n);
        if (ln < 0.12) continue;
        float panes = uWin2[i].w;
        // ---- pool on the surface
        if (!sky && uPool > 0.0) {
            vec3 a = P - C;
            float s = dot(a, n) / ln;
            if (s > 0.5 && s < 1400.0) {
                vec3 S = P - L * s; // where the light leaves the opening
                vec3 q3 = S - C;
                vec2 q = vec2(dot(q3, U), dot(q3, V));
                float soft = 2.5 + 0.035 * s;
                float m = openMask(q, h, soft, panes);
                if (m > 0.002) {
                    float facing = clamp((dot(N, -L) + 0.15) / 1.15, 0.0, 1.0);
                    float vis = facing > 0.0 ? openVisibility(P, S, jitter) : 0.0;
                    pool += m * facing * vis * exp(-s / 1100.0);
                }
            }
        }
        // ---- shaft in the air: the ray against the beam prism in (u, v, s) coordinates
        if (uShaft > 0.0) {
            vec3 Lc = cross(V, L), Lv = cross(L, U), Ls = cross(U, V);
            float det = dot(U, Lc);
            if (abs(det) > 1e-4) {
                vec3 a = uEye - C;
                vec3 o = vec3(dot(Lc, a), dot(Lv, a), dot(Ls, a)) / det;       // u, v, s coordinates of the eye
                vec3 dv = vec3(dot(Lc, rd), dot(Lv, rd), dot(Ls, rd)) / det;   // and of the ray direction
                const float smax = 900.0;
                vec3 lo = vec3(-h.x - 10.0, -h.y - 10.0, 0.0), hi = vec3(h.x + 10.0, h.y + 10.0, smax);
                vec3 inv = 1.0 / (dv + vec3(1e-6) * sign(dv + vec3(1e-9)));
                vec3 t0 = (lo - o) * inv, t1 = (hi - o) * inv;
                vec3 tn = min(t0, t1), tf = max(t0, t1);
                float tA = max(max(tn.x, tn.y), max(tn.z, 0.0)), tB = min(min(tf.x, tf.y), min(tf.z, tEnd));
                if (tB > tA) {
                    float len = tB - tA;
                    float acc = 0.0;
                    for (int k = 0; k < 6; ++k) {
                        float t = tA + len * ((float(k) + jitter) / 6.0);
                        vec3 c = o + dv * t;
                        float soft = 4.0 + 0.04 * c.z;
                        float m = openMask(c.xy, h, soft, panes);
                        acc += m * exp(-c.z / 520.0) * smoothstep(0.0, 24.0, c.z);
                    }
                    shaft += acc * (len / 6.0) * hg(dot(L, -rd));
                }
            }
        }
    }
    vec3 c = max(scene.rgb, vec3(0.0));
    vec3 albedo = sqrt(c) * 0.9 + 0.03;
    float Y = dot(c, giLuma);
    float need = 1.0 - 0.65 * smoothstep(0.30, 0.9, Y);
    vec3 E = uColor * pool * uPool;
    E = 3.0 * E / (3.0 + E); // bounded per channel so a bright wall never clips
    oColor = vec4(albedo * E * (2.4 * need) + uColor * shaft * (uShaft * 0.00035), 0.0);
    oFactor = vec4(1.0 + E * 1.6 * need, 1.0);
}
)GLSL";

// Emissive surfaces: draws of classified textures (night-lit glass, lamps and signs) wrote stencil 4 / 8 at the opaque
// stage. The HDR emission is added there, from the painted colour, so the texture's own pattern survives.
static const char* kEmissiveShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uScene;
uniform float uGain;
uniform vec3 uTint;
uniform int uClass;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 c = max(texelFetch(uScene, p, 0).rgb, vec3(0.0));
    float Y = dot(c, vec3(0.2126, 0.7152, 0.0722));
    vec3 e;
    if (uClass == 4) {
        // Night glass: a warm interior light seen through the pane; the painted pattern (bars, curtains) modulates it.
        float pattern = clamp(sqrt(Y) * 3.2, 0.0, 1.0);
        e = uTint * (0.2 + 1.0 * pattern) * uGain;
    } else {
        // Lamps, signs, clock faces: lift the painted colour into HDR, keeping its hue.
        e = (sqrt(c) * 0.8 + c) * uTint * uGain;
    }
    oColor = vec4(e, 0.0);
}
)GLSL";


// Interior polish: the pane itself for the render-only windows (kind 2). A flat quad on the wall plane, a dark wood frame
// around a sky-lit pane with mullions, depth-tested against the room so furniture and characters stay in front.
static const char* kInteriorGlassVertex = R"GLSL(
uniform mat4 uViewProj;
uniform float uNdcXScale, uFlipY;
uniform vec4 uWin0[12], uWin1[12], uWin2[12];
out vec2 vLocal;
flat out int vWin;
void main() {
    int i = gl_InstanceID;
    vec3 C = uWin0[i].xyz, n = uWin1[i].xyz, U = uWin2[i].xyz;
    vec2 h = vec2(uWin0[i].w, uWin1[i].w) + vec2(8.0);
    vec3 V = cross(n, U);
    vec2 corners[6] = vec2[6](vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
    vec2 q = corners[gl_VertexID];
    vLocal = q * h;
    vWin = i;
    vec3 P = C + n * 0.9 + U * (q.x * h.x) + V * (q.y * h.y);
    gl_Position = uViewProj * vec4(P, 1.0);
    gl_Position.x /= uNdcXScale;
    gl_Position.y *= uFlipY;
}
)GLSL";
static const char* kInteriorGlassFragment = R"GLSL(
in vec2 vLocal;
flat in int vWin;
layout(location = 0) out vec4 oColor;
uniform vec4 uWin0[12], uWin1[12], uWin2[12];
uniform vec3 uGlass;     // linear radiance of the sky seen through the pane (zero level = night)
uniform vec3 uFrame;     // linear albedo of the frame under the room's darkness
float h21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y);
}
void main() {
    vec2 h = vec2(uWin0[vWin].w, uWin1[vWin].w);
    vec2 a = h - abs(vLocal);
    float panes = uWin2[vWin].w;
    float nu = mod(panes, 16.0), nv = floor(panes / 16.0);
    float bar = 0.0;
    if (nu > 1.5) {
        float x = (vLocal.x / h.x * 0.5 + 0.5) * nu;
        bar = max(bar, 1.0 - smoothstep(1.4, 2.4, abs(x - floor(x + 0.5)) * 2.0 * h.x / nu));
        bar *= (x > 0.5 && x < nu - 0.5) ? 1.0 : 0.0;
    }
    if (nv > 1.5) {
        float y = (vLocal.y / h.y * 0.5 + 0.5) * nv;
        float b = (1.0 - smoothstep(1.4, 2.4, abs(y - floor(y + 0.5)) * 2.0 * h.y / nv)) * ((y > 0.5 && y < nv - 0.5) ? 1.0 : 0.0);
        bar = max(bar, b);
    }
    float edge = min(a.x, a.y);              // distance inside the opening
    bool glass = edge > 0.0 && bar < 0.5;
    if (glass) {
        // Sky behind frosted glass: horizon glow below, bluer zenith above, soft variation and darker toward the frame.
        float t = clamp(vLocal.y / h.y * 0.5 + 0.5, 0.0, 1.0);
        vec3 sky = mix(vec3(1.0, 0.93, 0.80), vec3(0.50, 0.68, 1.0), smoothstep(0.0, 1.0, t));
        float frost = vnoise(vLocal * 0.22) * 0.65 + vnoise(vLocal * 0.9 + 7.0) * 0.35;
        sky *= (0.78 + 0.34 * frost) * mix(0.62, 1.0, smoothstep(0.0, 16.0, edge));
        oColor = vec4(uGlass * sky, 1.0);
    } else {
        // Frame and mullions: warm painted wood with grain along the bar, and a lit inner lip.
        float grain = vnoise(vec2(vLocal.x * 0.05, vLocal.y * 0.9)) * 0.5 + vnoise(vec2(vLocal.x * 0.9, vLocal.y * 0.05)) * 0.5;
        float lip = smoothstep(2.5, 0.0, abs(edge)) * 0.5;
        oColor = vec4(uFrame * (0.55 + 0.9 * grain) * (0.8 + 0.6 * lip) + uGlass * 0.03 * lip, 1.0);
    }
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
uniform int uStep; // 1: one AO texel per pixel; 2: half resolution (every sample position is a full-resolution pixel)

const float kBayer[16] = float[16](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);

vec3 viewPos(ivec2 p, float d) {
    vec2 ndc = (vec2(p) + 0.5) / vec2(uSize) * 2.0 - 1.0;
    ndc.y *= uFlipY;
    float dist = linDepth(d);
    return vec3(ndc.x * uTan.x * dist, ndc.y * uTan.y * dist, -dist);
}

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    ivec2 p = q * uStep; // full-resolution pixel this texel stands for
    float d0 = fetchDepth(p);
    // The linear depth of the sample rides in green so the blur can weight by it without fetching the depth buffer.
    float ld = linDepth(d0);
    if (d0 >= 0.99999) {
        oColor = vec4(1.0, ld, 0.0, 1.0);
        return;
    }
    vec3 P = viewPos(p, d0);

    // Normal from the neighbours on the side with the smaller depth step, which keeps silhouettes clean.
    ivec2 sx = ivec2(uStep, 0), sy = ivec2(0, uStep);
    vec3 Pl = viewPos(p - sx, fetchDepth(p - sx));
    vec3 Pr = viewPos(p + sx, fetchDepth(p + sx));
    vec3 Pd = viewPos(p - sy, fetchDepth(p - sy));
    vec3 Pu = viewPos(p + sy, fetchDepth(p + sy));
    bool useR = (p.x == 0) || (p.x < uSize.x - 1 && abs(Pr.z - P.z) < abs(P.z - Pl.z));
    bool useU = (p.y == 0) || (p.y < uSize.y - 1 && abs(Pu.z - P.z) < abs(P.z - Pd.z));
    vec3 dx = useR ? (Pr - P) : (P - Pl);
    vec3 dy = useU ? (Pu - P) : (P - Pd);
    vec3 n = cross(dx, dy);
    float nl = length(n);
    if (nl < 1e-8) {
        oColor = vec4(1.0, ld, 0.0, 1.0);
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
        oColor = vec4(1.0, ld, 0.0, 1.0);
        return;
    }

    // Rotate the kernel around the normal by a per-texel angle from a 4x4 pattern (2x2 at half resolution, so the
    // pattern spans the same four full-resolution pixels); the blur averages it away.
    const float kBayer2[4] = float[4](0.0, 2.0, 3.0, 1.0);
    float ang = uStep == 1 ? (kBayer[(q.y & 3) * 4 + (q.x & 3)] + 0.5) / 16.0 * 6.2831853
                           : (kBayer2[(q.y & 1) * 2 + (q.x & 1)] + 0.5) / 4.0 * 6.2831853;
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
    oColor = vec4(mix(1.0, ao, fade), ld, 0.0, 1.0);
}
)GLSL";

// Depth-aware blur of the AO buffer. Edge-stopping is relative to the AO radius (and a little to the distance, so
// slanted floors still blur), which keeps the result from bleeding across silhouettes.
static const char* kBlurLib = R"GLSL(
uniform sampler2D uAo;  // red: occlusion, green: linear depth of the sample
uniform ivec2 uDir;
uniform float uRadius;
uniform int uAoShift;   // log2 of the full-resolution pixels per AO texel side (1 for the half-resolution raw buffer, else 0)

float blurAo(ivec2 p) {
    float dc = linDepth(fetchDepth(p));
    float sigma = max(0.5 * uRadius, 0.04 * dc);
    float sum = 0.0;
    float wsum = 0.0;
    for (int i = -4; i <= 4; i++) {
        ivec2 q = clamp(p + uDir * i, ivec2(0), uSize - 1);
        vec2 t = texelFetch(uAo, q >> uAoShift, 0).rg;
        float dz = (t.g - dc) / sigma;
        float w = exp(-dz * dz);
        sum += t.r * w;
        wsum += w;
    }
    // A half-resolution texel can belong to another surface than the pixel (a silhouette): when no tap matches the
    // pixel's depth, take the nearest texel's occlusion instead of dividing zero by zero.
    return wsum > 1e-4 ? sum / wsum : texelFetch(uAo, p >> uAoShift, 0).r;
}
)GLSL";

static const char* kBlurShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    oColor = vec4(blurAo(p), linDepth(fetchDepth(p)), 0.0, 1.0);
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
// Art-directed grade (game-side look, already interpolated over time of day and scene changes); skipped when off.
uniform int uArtOn;
uniform vec3 uArtSlope;
uniform float uArtOffset;
uniform float uArtPower;
uniform float uArtContrast;
uniform float uArtSat;
uniform vec3 uArtShadow;
uniform vec3 uArtHigh;
uniform vec2 uArtGreen;

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
    if (uArtOn != 0) {
        const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
        // Green hue range: foliage greens drift towards yellow (+) or teal (-) and get their own saturation.
        float gw = smoothstep(0.0, 0.15, c.g - max(c.r, c.b));
        if (uArtGreen.x >= 0.0) c.r = mix(c.r, c.g, gw * uArtGreen.x);
        else c.b = mix(c.b, c.g, -gw * uArtGreen.x);
        float gl = dot(c, lw);
        c = mix(vec3(gl), c, 1.0 + gw * (uArtGreen.y - 1.0));
        // Slope / offset / power, then an S-curve, split tone and global saturation (all in display space).
        // Deep shadows stay black and neutral: the lift and the gamma fade in above them (a fixed lift or a gamma below 1
        // would otherwise turn true black into a tinted grey).
        float dk = smoothstep(0.0, 0.12, dot(c, lw));
        c = pow(max(c * uArtSlope + uArtOffset * dk, vec3(0.0)), vec3(mix(1.0, uArtPower, dk)));
        // Keep the contrast curve monotonic when slope/saturation put a channel above display white.
        vec3 cc = clamp(c, 0.0, 1.0);
        vec3 sc = cc * cc * (3.0 - 2.0 * cc);
        c += 2.0 * uArtContrast * (sc - cc);
        float al = dot(c, lw);
        c += uArtShadow * ((1.0 - smoothstep(0.0, 0.5, al)) * smoothstep(0.0, 0.2, al)) + uArtHigh * smoothstep(0.45, 1.0, al);
        al = dot(c, lw);
        c = mix(vec3(al), c, uArtSat);
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

// New Dawn temporal anti-aliasing. The interpreter jitters the 3D world's projection by a sub-pixel Halton offset, this
// pass reprojects last frame's resolved image with the (un-jittered) camera matrices and blends. There are no
// per-object motion vectors: only the static scene is reprojected exactly, everything that moves on its own (actors,
// wind, water, effects) is controlled by the neighbourhood variance clip, the depth test and the actor mask. Colour is
// held in a tone-compressed YCoCg space so bright specular glints cannot dominate the blend.
static const char* kTaaShader = R"GLSL(
in vec2 vUv;
layout(location = 0) out vec4 oHistory; // compressed colour, view depth
layout(location = 1) out vec4 oScene;   // linear HDR colour for bloom and the display transform
uniform sampler2D uScene;
uniform sampler2D uDepth;
uniform sampler2D uTaaHistory;
uniform sampler2D uTaaMask;
uniform ivec2 uSize;
uniform mat4 uInvViewProj;
uniform mat4 uPrevViewProj;
uniform vec2 uNdc;
uniform vec2 uJitter; // this frame's sub-pixel shift of the whole image, in pixels (GL window axes)
uniform float uNear;
uniform float uFar;
uniform float uFeedback;
uniform int uReset;
uniform int uUseMask;
uniform int uDebug;
uniform vec4 uGrassTune; // blend cap, clip width, filter sharpness (Gaussian exponent), unused

const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);
vec3 compressC(vec3 c) { c = max(c, vec3(0.0)); return c / (1.0 + dot(c, kLuma)); }
vec3 expandC(vec3 c) { return c / max(1.0 - dot(c, kLuma), 1e-3); }
vec3 toYCoCg(vec3 c) { return vec3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b); }
vec3 fromYCoCg(vec3 c) { return vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }
float linDepth(float d) {
    float z = d * 2.0 - 1.0;
    return 2.0 * uNear * uFar / (uFar + uNear - z * (uFar - uNear));
}
ivec2 clampP(ivec2 p) { return clamp(p, ivec2(0), uSize - 1); }

// Catmull-Rom history filter in 5 bilinear taps (the corners are dropped and the rest renormalised): sharper than plain
// bilinear, which would soften every moving frame a little more.
vec3 sampleHistory(vec2 uv) {
    vec2 size = vec2(uSize);
    vec2 pos = uv * size;
    vec2 c = floor(pos - 0.5) + 0.5;
    vec2 f = pos - c;
    vec2 f2 = f * f, f3 = f2 * f;
    vec2 w0 = -0.5 * f3 + f2 - 0.5 * f;
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
    vec2 w3 = 0.5 * f3 - 0.5 * f2;
    vec2 w12 = w1 + w2;
    vec2 t12 = (c + w2 / w12) / size;
    vec2 t0 = (c - 1.0) / size;
    vec2 t3 = (c + 2.0) / size;
    float k0 = w12.x * w0.y, k1 = w0.x * w12.y, k2 = w12.x * w12.y, k3 = w3.x * w12.y, k4 = w12.x * w3.y;
    vec3 r = texture(uTaaHistory, vec2(t12.x, t0.y)).rgb * k0 + texture(uTaaHistory, vec2(t0.x, t12.y)).rgb * k1 +
             texture(uTaaHistory, t12).rgb * k2 + texture(uTaaHistory, vec2(t3.x, t12.y)).rgb * k3 +
             texture(uTaaHistory, vec2(t12.x, t3.y)).rgb * k4;
    return r / (k0 + k1 + k2 + k3 + k4);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 uv = (vec2(p) + 0.5) / vec2(uSize);
    vec3 sum = vec3(0.0), sum2 = vec3(0.0), lo = vec3(1e9), hi = vec3(-1e9);
    float actor = 0.0, grass = 0.0;
    vec3 cur = vec3(0.0);
    float curW = 0.0;
    // Tuft pixels (stencil 2, written by the grass pass): thin blades that sway, so no history can follow them.
    float grassHere = uUseMask != 0 ? texelFetch(uTaaMask, p, 0).g : 0.0;
    float kFilter = mix(2.29, uGrassTune.z, grassHere);
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            ivec2 q = clampP(p + ivec2(i, j));
            vec3 s = toYCoCg(compressC(texelFetch(uScene, q, 0).rgb));
            // The sample at q was taken uJitter pixels away from its centre, so it describes the scene at q - uJitter:
            // weighting the 3x3 samples by their distance to this pixel's centre (a Gaussian close to Blackman-Harris)
            // gives a current-frame estimate that no longer depends on the jitter phase, which is what makes a
            // converged still image pulse once per jitter cycle.
            vec2 dd = vec2(i, j) - uJitter;
            float wt = exp(-kFilter * dot(dd, dd));
            cur += s * wt;
            curW += wt;
            sum += s;
            sum2 += s * s;
            lo = min(lo, s);
            hi = max(hi, s);
            if (uUseMask != 0) {
                vec2 mk = texelFetch(uTaaMask, q, 0).rg;
                actor = max(actor, mk.r);
                grass = max(grass, mk.g);
            }
        }
    }
    cur /= curW;
    vec3 mean = sum / 9.0;
    vec3 sigma = sqrt(max(sum2 / 9.0 - mean * mean, vec3(0.0)));

    float d = texelFetch(uDepth, p, 0).r;
    float w = linDepth(d);
    vec3 outC = cur;
    float blend = 0.0;
    float tint = 0.0;
    if (uReset == 0) {
        vec4 wp = uInvViewProj * vec4((uv * 2.0 - 1.0) * uNdc, d * 2.0 - 1.0, 1.0);
        vec3 world = wp.xyz / wp.w;
        vec4 pc = uPrevViewProj * vec4(world, 1.0);
        vec2 hUv = (pc.xy / pc.w / uNdc) * 0.5 + 0.5;
        bool ok = pc.w > 0.0 && all(greaterThan(hUv, vec2(0.0))) && all(lessThan(hUv, vec2(1.0)));
        if (ok) {
            // Depth agreement. The expected view depth must fall within the range stored around the history texel (plus
            // room for the slope of the surface): at a silhouette the depth under a pixel flips between the near and
            // the far surface with the jitter phase, so comparing against one texel would throw the history away exactly
            // on the edges that need it. A wide disocclusion has no matching depth anywhere in that window.
            ivec2 hp = clampP(ivec2(hUv * vec2(uSize)));
            float hMin = 1e9, hMax = -1e9;
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    float hw = texelFetch(uTaaHistory, clampP(hp + ivec2(i, j)), 0).a;
                    hMin = min(hMin, hw);
                    hMax = max(hMax, hw);
                }
            }
            float gw = max(max(abs(linDepth(texelFetch(uDepth, clampP(p + ivec2(1, 0)), 0).r) - w),
                               abs(linDepth(texelFetch(uDepth, clampP(p - ivec2(1, 0)), 0).r) - w)),
                           max(abs(linDepth(texelFetch(uDepth, clampP(p + ivec2(0, 1)), 0).r) - w),
                               abs(linDepth(texelFetch(uDepth, clampP(p - ivec2(0, 1)), 0).r) - w)));
            float tol = 0.012 * pc.w + 1.5 * gw + 0.5;
            ok = pc.w >= hMin - tol && pc.w <= hMax + tol;
        }
        if (ok) {
            vec3 hist = toYCoCg(sampleHistory(hUv));
            // Clip the history towards the neighbourhood: what the current frame cannot explain is not kept.
            float gamma = mix(mix(1.1, uGrassTune.y, grass), 0.6, actor);
            vec3 ext = gamma * sigma + 0.002;
            vec3 bl = max(lo, mean - ext), bh = min(hi, mean + ext);
            vec3 ctr = 0.5 * (bl + bh), half_ = 0.5 * (bh - bl) + 1e-5;
            vec3 v = hist - ctr;
            vec3 a = abs(v / half_);
            float m = max(a.x, max(a.y, a.z));
            vec3 clipped = m > 1.0 ? ctr + v / m : hist;
            float clipAmt = clamp(length(clipped - hist) / (length(hist) + 0.02), 0.0, 1.0);
            float speed = length((hUv - uv) * vec2(uSize));
            blend = uFeedback;
            blend -= 0.12 * smoothstep(0.0, 2.0, speed);  // resampling under motion softens, so trust history less
            blend *= 1.0 - 0.6 * clipAmt;                 // history that had to be clipped is suspect
            blend = mix(blend, min(blend, uGrassTune.x), grass); // tufts sway without motion vectors: short memory
            blend = mix(blend, min(blend, 0.5), actor);   // actors move without motion vectors: short memory
            outC = mix(cur, clipped, blend);
            tint = clipAmt;
        }
    }
    vec3 rgb = fromYCoCg(outC);
    oHistory = vec4(rgb, w);
    oScene = vec4(expandC(rgb), 1.0);
    if (uDebug == 1) oScene = vec4(vec3(blend), 1.0);
    else if (uDebug == 2) oScene = vec4(tint, actor, grass, 1.0);
}
)GLSL";

// Light contrast-adaptive sharpen (after AMD FidelityFX CAS) for the resolved TAA image. TAA integrates sub-pixel
// samples, which is a one-pixel box filter: it removes aliasing but softens fine detail a little. This works in a
// perceptual (square-root) space on tone-compressed colour so HDR highlights are not over-sharpened, backs off where
// the local contrast is already high, and fades out in the darks: dark areas get no extra edge emphasis.
static const char* kTaaSharpenShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uScene;
uniform ivec2 uSize;
uniform float uSharpen;
uniform sampler2D uTaaMask;
uniform int uUseMask;
uniform vec4 uSharpTune; // grass sharpness, dark fade start, dark fade end, unused
const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);
vec3 percept(ivec2 q) {
    vec3 c = max(texelFetch(uScene, clamp(q, ivec2(0), uSize - 1), 0).rgb, vec3(0.0));
    return sqrt(c / (1.0 + dot(c, kLuma)));
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 E = percept(p), B = percept(p + ivec2(0, -1)), D = percept(p + ivec2(-1, 0)), F = percept(p + ivec2(1, 0)),
         H = percept(p + ivec2(0, 1));
    vec3 mn = min(min(min(B, D), min(F, H)), E);
    vec3 mx = max(max(max(B, D), max(F, H)), E);
    vec3 amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, vec3(1e-4)), 0.0, 1.0));
    float g = 0.0;
    if (uUseMask != 0) {
        g = max(max(texelFetch(uTaaMask, p, 0).g, texelFetch(uTaaMask, clamp(p + ivec2(1, 0), ivec2(0), uSize - 1), 0).g),
                max(texelFetch(uTaaMask, clamp(p - ivec2(1, 0), ivec2(0), uSize - 1), 0).g,
                    max(texelFetch(uTaaMask, clamp(p + ivec2(0, 1), ivec2(0), uSize - 1), 0).g,
                        texelFetch(uTaaMask, clamp(p - ivec2(0, 1), ivec2(0), uSize - 1), 0).g)));
    }
    vec3 w = amp * (-1.0 / mix(8.0, 5.0, mix(uSharpen, uSharpTune.x, g)));
    vec3 S = clamp(((B + D + F + H) * w + E) / (1.0 + 4.0 * w), 0.0, 1.0);
    S = mix(E, S, smoothstep(uSharpTune.y, uSharpTune.z, dot(E, kLuma))); // nothing in the darks
    vec3 c = S * S;
    oColor = vec4(c / max(1.0 - dot(c, kLuma), 1e-3), 1.0);
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
uniform float uMoonLift;
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

#ifdef CSM
vec2 gRot;
float csmTap(vec3 uvz) {
    float sum = 0.0;
    for (int i = 0; i < 12; i++) {
        vec2 o = kPoisson[i];
        o = vec2(o.x * gRot.x - o.y * gRot.y, o.x * gRot.y + o.y * gRot.x) * 2.2;
        sum += texture(uShadowCmp, vec3(uvz.xy + o * uShadowTexel, uvz.z));
    }
    return sum / 12.0;
}
#endif

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
#ifdef CSM
    // Cascaded: the same rotated Poisson filter, sampled in the atlas tile that covers this pixel's distance.
    float a = ign(gl_FragCoord.xy) * 6.2831853;
    gRot = vec2(cos(a), sin(a));
    float csm = csmLit(P, N, linDepth(d0), uNormalBias, 0.003 + 0.010 * (1.0 - ndl));
    bool inMap = csm >= 0.0;
    float sampledLit = inMap ? csm : 1.0;
#else
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
#endif
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
    // Moonlight (night): a faint cool lift where the Moon reaches, so lit surfaces and silhouettes read against the
    // native dark. uMoonLift is 0 for the sun, which keeps the daytime result bit-identical.
    vec3 moonLift = vec3(1.0) + uMoonLift * ndl * lit * vec3(0.55, 0.75, 1.0);
    if (uHdr != 0) {
        // Fog has already been composited into scene colour. Do not shadow its in-scatter:
        // otherwise geometry hidden by full fog becomes a dark silhouette against the sky.
        float fog = clamp((d0 * 2.0 - 1.0) * uFogMul + uFogOff, 0.0, 255.0) / 255.0;
        float coverage = 1.0 - smoothstep(uRadius * 0.8, uRadius, length(P - uCameraEye));
        factor = mix(vec3(1.0), factor, coverage * (1.0 - fog));
        moonLift = mix(vec3(1.0), moonLift, coverage * (1.0 - fog));
    }
    factor *= moonLift;
    vec3 result = texture(uColor, vUv).rgb * factor;
#ifdef CSM
    if (uDebug == 5) {
        // Debug view: tint by cascade (red, green, blue, yellow = nearest to furthest).
        const vec3 tint[4] = vec3[4](vec3(1.0, 0.35, 0.35), vec3(0.35, 1.0, 0.35), vec3(0.4, 0.5, 1.0), vec3(1.0, 1.0, 0.3));
        result *= mix(vec3(1.0), tint[csmIndex(linDepth(d0))], 0.55);
    }
#endif
    oColor = vec4(result, 1.0);
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

#ifdef CSM
float csmTap(vec3 uvz) {
    return texture(uShadowCmp, uvz);
}
#endif

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
#ifdef CSM
        float s = csmLit(pos, vec3(0.0), csmViewDepth(pos), 0.0, 0.004);
        if (s >= 0.0) lit = s;
#else
        if (ref < 1.0 && all(greaterThanEqual(suv, vec2(0.0))) && all(lessThanEqual(suv, vec2(1.0)))) {
            lit = texture(uShadowCmp, vec3(suv, ref - 0.004));
        }
#endif
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


// Downloaded lens-flare VFX textures. A 1x1 disc integration supplies depth/cloud-occluded HDR energy;
// soft glow and coloured iris ghosts scatter it along the optical axis. No generated spike pattern.
static const char* kFlareVisibilityShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uScene;
uniform sampler2D uFlareClouds;
uniform float uCloudOpacity;
uniform vec2 uSunUv;
uniform float uSunDepth;
void main() {
    vec3 radiance = vec3(0.0);
    float visible = 0.0, total = 0.0;
    for (int i=0; i<48; ++i) {
        float r = sqrt((float(i)+0.5)/48.0);
        float a = float(i)*2.39996323;
        vec2 uv = uSunUv + vec2(cos(a),sin(a))*r*(0.012*float(uSize.y))/vec2(uSize);
        float weight = 1.0-r*r*0.65;
        total += weight;
        if (any(lessThan(uv,vec2(0.0))) || any(greaterThan(uv,vec2(1.0)))) continue;
        float d = texture(uDepth,uv).r;
        float lit = smoothstep(uSunDepth-0.00002,uSunDepth-0.000002,d);
        lit *= 1.0-clamp(texture(uFlareClouds,uv).a*uCloudOpacity,0.0,1.0);
        vec3 c = clamp(texture(uScene,uv).rgb,vec3(0.0),vec3(8.0));
        radiance += c*lit*weight;
        visible += lit*weight;
    }
    float visibility = visible/max(total,1e-6);
    vec3 c = radiance/max(visible,1e-6);
    float energy = dot(c,vec3(0.2126,0.7152,0.0722));
    // Whiten the optical scatter; sunset colour stays local without a saturated screen veil.
    vec3 tint = mix(vec3(1.0),c/max(energy,1e-4),0.25);
    float border = min(min(uSunUv.x,1.0-uSunUv.x),min(uSunUv.y,1.0-uSunUv.y));
    float strength = clamp(sqrt(max(energy,0.0)),0.0,2.0)*smoothstep(0.0,1.0,visibility)*
                     smoothstep(0.0,0.08,border);
    oColor = vec4(tint*strength,visibility);
}
)GLSL";

// GPU Gems 3, chapter 13 style radial light transport. The mask contains only sky coverage;
// geometry never emits. All noise is spatially fixed, with no history or frame-dependent jitter.
// Depth alone misses the game's depthless foliage. Compare its HDR colour against this frame's
// sky/cloud radiance as well; dark leaves block rays even when they did not write Z.
static const char* kRaySkyMaskLib = R"GLSL(
uniform sampler2D uScene;
uniform sampler2D uRaySky;
uniform sampler2D uFlareClouds;
uniform float uCloudOpacity;
float raySkyCoverage(ivec2 p) {
    if (fetchDepth(p) < 0.99999) return 0.0;
    vec2 uv = (vec2(p)+0.5)/vec2(uSize);
    vec3 sky = texture(uRaySky,uv).rgb;
    if (uCloudOpacity > 0.0) {
        vec2 texel = 0.75/vec2(textureSize(uFlareClouds,0));
        vec4 c = texture(uFlareClouds,uv)*0.5;
        c += texture(uFlareClouds,uv+texel)*0.125;
        c += texture(uFlareClouds,uv-texel)*0.125;
        c += texture(uFlareClouds,uv+texel*vec2(-1,1))*0.125;
        c += texture(uFlareClouds,uv+texel*vec2(1,-1))*0.125;
        c *= uCloudOpacity;
        sky = sky*(1.0-c.a)+c.rgb;
    }
    const vec3 luma = vec3(0.2126,0.7152,0.0722);
    float ratio = dot(texelFetch(uScene,p,0).rgb,luma)/max(dot(sky,luma),0.01);
    return smoothstep(0.65,0.95,ratio);
}
)GLSL";

static const char* kRayMaskShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform vec2 uSunUv;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy)*2;
    float sky = 0.0;
    for (int y=0; y<2; ++y)
        for (int x=0; x<2; ++x)
            sky += raySkyCoverage(min(p+ivec2(x,y),uSize-1))*0.25;
    vec2 delta = (vUv-uSunUv)*vec2(float(uSize.x)/float(uSize.y),1.0);
    float source = exp(-dot(delta,delta)/0.10);
    oColor = vec4(sky*source,0.0,0.0,1.0);
}
)GLSL";

static const char* kRayRadialShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uRayMask;
uniform vec2 uSunUv;
void main() {
    const int steps = 48;
    float jitter = fract(52.9829189*fract(dot(gl_FragCoord.xy,vec2(0.06711056,0.00583715))));
    vec2 delta = uSunUv-vUv;
    float sum = 0.0, weights = 0.0;
    for (int i=0; i<steps; ++i) {
        float t = (float(i)+jitter)/float(steps);
        float weight = exp(-0.5*t);
        sum += texture(uRayMask,vUv+delta*t).r*weight;
        weights += weight;
    }
    oColor = vec4(sum/weights,0.0,0.0,1.0);
}
)GLSL";

static const char* kRayApplyShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uRays;
uniform sampler2D uFlareEmitter;
uniform vec2 uSunUv;
uniform vec3 uColor;
uniform float uStrength;
void main() {
    float visible = texelFetch(uFlareEmitter,ivec2(0),0).a;
    float border = min(min(uSunUv.x,1.0-uSunUv.x),min(uSunUv.y,1.0-uSunUv.y));
    float fade = smoothstep(0.0,0.10,border)*smoothstep(0.0,0.6,visible);
    // Keep foreground silhouettes crisp; a screen-space ray cannot locate participating fog in front of them.
    float coverage = raySkyCoverage(ivec2(gl_FragCoord.xy));
    float ray = texture(uRays,vUv).r;
    oColor = vec4(uColor*(ray*ray*uStrength*6.0*fade*coverage),0.0);
}
)GLSL";

static const char* kFlareApplyShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uFlareEmitter;
uniform sampler2DArray uFlareSprites;
uniform vec2 uSpriteScale[6];
uniform vec2 uSunUv;
uniform vec2 uScreenSize;
vec3 sprite(vec2 center,float radius,int layer,vec3 tint) {
    vec2 uv=(vUv-center)*vec2(uScreenSize.x/uScreenSize.y,1.0)/(2.0*radius)+0.5;
    if (any(lessThan(uv,vec2(0.0))) || any(greaterThan(uv,vec2(1.0)))) return vec3(0.0);
    vec4 mask=texture(uFlareSprites,vec3((uv-0.5)*uSpriteScale[layer]+0.5,float(layer)));
    return tint*mask.r*mask.a;
}
void main() {
    vec3 energy=texelFetch(uFlareEmitter,ivec2(0),0).rgb;
    vec2 axis=vec2(0.5)-uSunUv;
    // All shapes come from downloaded PNGs, including the soft bloom and iris/ring structure.
    vec3 glow=sprite(uSunUv,0.115,0,vec3(1.0,0.93,0.82)*0.30);
    glow+=sprite(uSunUv,0.15,1,vec3(1.0,0.85,0.63)*0.035);
    vec3 ghosts=sprite(uSunUv+axis*0.45,0.025,3,vec3(0.45,0.70,1.0)*0.075);
    ghosts+=sprite(uSunUv+axis*1.1,0.028,2,vec3(0.40,0.90,0.72)*0.055);
    ghosts+=sprite(uSunUv+axis*1.65,0.050,2,vec3(1.0,0.67,0.35)*0.045);
    ghosts+=sprite(uSunUv+axis*2.2,0.080,3,vec3(0.62,0.45,1.0)*0.040);
    ghosts+=sprite(uSunUv+axis*2.6,0.115,5,vec3(0.45,0.90,0.66)*0.030);
    float separation=length(axis*vec2(uScreenSize.x/uScreenSize.y,1.0));
    oColor=vec4(energy*(glow+ghosts*smoothstep(0.025,0.25,separation)),0.0);
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
uniform bool uFogSun;
uniform int uPointCount;
uniform vec4 uPointPosRadius[4];
uniform vec3 uPointColor[4];
#ifdef CSM
float csmTap(vec3 uvz) {
    return texture(uShadowCmp, uvz);
}
#endif
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
    const int steps = 32;
    float ds = L / float(steps);
    // Interleaved gradient noise is fixed across frames: no animation-dependent grain or history on scene cuts.
    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy,vec2(0.06711056,0.00583715))));
    const float g = 0.68;
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
#ifdef CSM
        if (uFogSun) {
            float s = csmLit(pos, vec3(0.0), csmViewDepth(pos), 0.0, 0.002);
            if (s >= 0.0) lit = s;
        }
#else
        if (uFogSun && all(greaterThan(sc,vec3(0.0))) && all(lessThan(sc,vec3(1.0)))) {
            float edge = smoothstep(0.0,0.04,min(min(sc.x,1.0-sc.x),min(sc.y,1.0-sc.y)));
            lit = mix(1.0,texture(uShadowCmp,vec3(sc.xy,sc.z-0.002)),edge);
        }
#endif
        vec3 local = vec3(0.0);
        for (int j=0; j<uPointCount; ++j) {
            float d = length(pos-uPointPosRadius[j].xyz)/max(uPointPosRadius[j].w,1.0);
            float attenuation = pow(max(1.0-d,0.0),2.0);
            local += uPointColor[j]*attenuation*2.5;
        }
        scatter += T*opacity*(ambient+uSunColor*(lit*phase*1.5)+local);
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
uniform sampler2D uWaveBuf; // displaced surface from the wave trace pass (height, slope, 1 + foam); dummy when off
uniform int uWaves;
#ifdef WATER_REFLECTIONS
uniform mat4 uViewProj;
uniform sampler2D uOpaqueColor;
uniform sampler2D uOpaqueDepth;
uniform float uReflectionStrength;
uniform float uSurfaceDetail;

// March uniformly in screen space, interpolating reciprocal clip W for perspective-correct axial depth.
// Stable deterministic steps: no temporal history, random jitter or feedback from the translucent world.
vec4 traceReflection(vec3 origin, vec3 direction, float planeY, float roughness, out int status, out float hitDist) {
    status = 0; // 0: sky/miss, 1: accepted hit, 2: rejected candidate
    hitDist = 0.0;
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
            // Real water is not a mirror: the reflected image softens with the distance between the reflected point and
            // the waterline (a rough surface integrates a cone of directions) and smears along the vertical, since wave
            // slopes tilt the reflection mostly up and down the screen. Radius is in full-resolution pixels.
            hitDist = length(hit - origin);
            float angular = hitDist * roughness / max(depth, 1.0);
            float radius = clamp(angular / max(uPixelAngle * 0.5, 1e-5), 0.65, 9.0);
            // Restyled water uses a subpixel gather. The legacy 2.5px gather is retained when Restyle is off.
            radius = max(radius, mix(2.5, 0.65, uSurfaceDetail));
            vec3 color = texture(uOpaqueColor, uv).rgb;
            float weight = 1.0;
            const float golden = 2.39996323;
            for (int j = 0; j < 12; j++) {
                float a = float(j) * golden;
                float r = sqrt((float(j) + 0.5) / 12.0);
                vec2 tap = uv + vec2(cos(a), sin(a) * 1.8) * r * radius / vec2(uSize);
                float td = linDepth(texture(uOpaqueDepth, tap).r);
                float w = exp(-abs(td - depth) / (thickness * 2.0)) * (1.0 - 0.5 * r);
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

#ifdef WATER_REFLECTIONS
// Normal that bends the reflection: the glint slope plus two finer travelling octaves, each faded out as the pixel footprint
// reaches its wavelength. This is what makes a reflected tower wobble and shear instead of staying a perfect copy.
vec2 reflectionSlope(vec2 pos, float footprint) {
    float f3 = 1.0 - smoothstep(0.12, 0.5, footprint / 17.0);
    float f4 = 1.0 - smoothstep(0.12, 0.5, footprint / 6.5);
    vec2 s = waterSlope(pos, footprint) * 0.6;
    s += 0.9 * f3 * noised(pos / 17.0 + vec2(0.17, -0.11) * uTime + 3.7).yz / 17.0 * 2.6;
    s += 0.6 * f4 * noised(pos / 6.5 + vec2(-0.31, 0.23) * uTime + 11.9).yz / 6.5 * 1.1;
    return s;
}
#endif

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

    // Ocean swell: the hit is on the displaced surface, and its large-scale slope tilts every normal below.
    vec4 wv = uWaves != 0 ? texelFetch(uWaveBuf, p, 0) : vec4(0.0);
    bool wave = wv.a > 0.5;
    vec2 big = wave ? wv.gb : vec2(0.0);
    float surfY = uBoxY[bi] + (wave ? wv.r : 0.0);
    if (wave) {
        bestT = (surfY - uEye.y) / ray.y;
    }
    vec3 H = uEye + ray * bestT;
    float dist = bestT * length(ray);
    vec3 V = normalize(uEye - H);
    float footprint = dist * uPixelAngle;
    // Vertical depth of the water above the bed: the glints fade out over the shallows.
    float depthY = sky ? 1.0e4 : (abs(P.y - uBoxY[bi]) < 1.5 ? 1.0e4 : max(uBoxY[bi] - P.y, 0.0));
    float shore = smoothstep(2.0, 40.0, depthY);

    float footprintS = footprint / max(V.y, 0.06); // see the surface shader: pixel footprint along the view direction
    vec2 slope = waterSlope(H.xz, footprintS) + big;
    vec3 N = normalize(vec3(-slope.x, 1.0, -slope.y));
    float ndv = max(dot(N, V), 1e-3);
    float fres = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    vec3 refl = uSky * fres;
    if (uSkyReflectionsEnabled) {
        vec3 R = reflect(-V, normalize(vec3(-(slope.x - big.x) * 0.25 - big.x, 1.0, -(slope.y - big.y) * 0.25 - big.y)));
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
    vec2 reflSlope = reflectionSlope(H.xz, footprintS) + big;
    vec3 reflectionNormal = normalize(vec3(-reflSlope.x, 1.0, -reflSlope.y));
    int hitStatus;
    float hitDist;
    // Wave steepness roughens the reflection further.
    float roughness = 0.012 + 0.12 * length(reflSlope);
    // Use the exact plane even when a water mesh's own depth was accepted within the mask's 1.5-unit tolerance.
    vec3 reflectionH = uEye + ray * ((surfY - uEye.y) / ray.y);
    vec4 hit = traceReflection(reflectionH + vec3(0.0, 2.0, 0.0), reflect(-V, reflectionNormal), uBoxY[bi], roughness,
                               hitStatus, hitDist);
    if (uDebug == 2) {
        oColor = vec4(hitStatus == 1 ? vec3(0.0, 1.0, 0.0) :
                      (hitStatus == 2 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0)), 1.0);
        return;
    }
    // A premultiplied reflection can darken the painted water as well as brighten it. Screen-only addition
    // erased dark reflected objects. Lift the Fresnel response perceptually while retaining its view-angle falloff.
    if (uSkyReflectionsEnabled) {
        vec3 R = reflect(-V, reflectionNormal);
        R.y = max(R.y, 0.02);
        vec3 environment = textureLod(uSkySpecular, normalize(R), 0.8 + clamp(footprint * 0.02, 0.0, 1.1) + roughness * 6.0).rgb;
        // Retain the established perceptual visibility for confident geometry hits. The cube uses
        // physical water Schlick (F0=0.02), so a miss cannot increase reflection strength or wash out
        // mid-angle turquoise. Confidence crossfades the two premultiplied contributions.
        // Imperfect on purpose (a mirror reads as fake): Fresnel is the physical F raised to 0.62 rather than the old square
        // root, so steep views reflect clearly less; the reflection never reaches full opacity, thins with the distance the
        // light travelled, and breaks up slightly where the surface is rougher.
        float breakup = 1.0 - 0.3 * smoothstep(0.35, 0.85, 0.5 + 0.5 * noised(H.xz / 23.0 + vec2(0.05, 0.04) * uTime).x * 1.4);
        float travel = 1.0 - 0.45 * smoothstep(200.0, 3500.0, hitDist);
        float ssrWeight = uReflectionStrength * min(pow(fres, 0.62), 0.85) * shore * hit.a * breakup * travel;
        float skyWeight = 0.4 * uReflectionStrength * fres * shore * (1.0-hit.a) * breakup;
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

// ---------------------------------------------------------------------------------------------------------------------
// Water surface: shoreline/contact foam, refraction and depth absorption. Runs at the scene-end marker before the glint
// and reflection pass, on a copy of the scene colour. Thickness is the vertical distance between the water plane and
// the OPAQUE depth buffer (so a water mesh that wrote depth cannot hide the bed). Everything is a premultiplied
// overlay: out = src + dst * (1 - foam), with src = foam + (refracted/absorbed - original) * (1 - foam).
// ---------------------------------------------------------------------------------------------------------------------
#include "gfx_opengl_rivers_shaders.inc"

static const char* kWaterSurfaceShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uScene;
uniform sampler2D uOpaqueColor;
uniform sampler2D uVisibleDepth;
uniform mat4 uViewProj;
uniform float uFoam;
uniform float uRefr;
uniform float uPixelAngle;
uniform float uRestyle;
uniform float uSurfaceVisibility;
uniform vec3 uScatter;
uniform vec3 uAbsorb;
uniform float uFogMul;
uniform float uFogOff;
uniform vec3 uFogColor;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uSky;
uniform sampler2D uWaveBuf; // displaced surface from the wave trace pass (height, slope, 1 + foam); dummy when off
uniform int uWaves;
uniform float uWaveAmp;
uniform float uSwash;       // ocean shoreline swash strength, 0 = off
uniform int uDebug;         // 3: displaced height (red above / blue below the plane, grey = no wave), 4: slope
uniform sampler2D uFlowBuf; // river flow (world xz velocity of the native texture scroll, 1 = covered); dummy when off
uniform float uRiver;       // river streak strength, 0 = off
uniform sampler2D uFoamTex; // tileable foam (R lace, G clumps, B bubbles, A density), committed CC0

float hash1(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash1(i), hash1(i + vec2(1.0, 0.0)), f.x), mix(hash1(i + vec2(0.0, 1.0)), hash1(i + vec2(1.0, 1.0)), f.x), f.y);
}

// Continuity of the waterline foam along the shore: 1 = foam, 0 = a gap. Two scales of noise so the line thickens, thins and
// breaks in places (it must never read as a painted stripe).
float shoreBreak(vec2 xz) {
    float n = 0.6 * vnoise(xz / 9.0 + 17.3) + 0.4 * vnoise(xz / 2.7);
    return smoothstep(0.28, 0.62, n);
}

// Slope of a few warped travelling waves; each is faded out as its wavelength approaches the pixel footprint.
vec2 refractSlope(vec2 x, float footprint) {
    const vec2 dirs[6] = vec2[6](vec2(0.8, 0.6), vec2(-0.5, 0.866), vec2(0.2, -0.98), vec2(-0.9, -0.436), vec2(0.95, -0.312), vec2(-0.64, 0.768));
    const float wl[6] = float[6](70.0, 43.0, 27.0, 17.0, 7.5, 3.4);
    const float amp[6] = float[6](0.07, 0.06, 0.05, 0.04, 0.025, 0.018);
    const float spd[6] = float[6](0.8, 1.0, 1.3, 1.7, 2.1, 2.7);
    // A second, slower warp keeps the six directions from lining up into a regular lattice.
    float warp = 2.5 * vnoise(x / 60.0) + 6.0 * vnoise(x / 190.0 + 3.1);
    vec2 s = vec2(0.0);
    for (int i = 0; i < 6; i++) {
        float k = 6.2831853 / wl[i];
        float fade = 1.0 - smoothstep(0.15, 0.6, footprint / wl[i]);
        s += dirs[i] * amp[i] * fade * cos(k * dot(dirs[i], x) + spd[i] * uTime + warp * float(i + 1));
    }
    return s;
}

vec2 projectUv(vec3 w) {
    vec4 c = uViewProj * vec4(w, 1.0);
    return vec2(c.x / (c.w * uNdcXScale), c.y / c.w * uFlipY) * 0.5 + 0.5;
}

// Foam colour: sky and sunlight, then the scene fog at the foam's depth.
vec3 foamTint(vec3 H) {
    vec3 col = min(0.9 * (uSky * 0.85 + uLightColor * (0.25 + 0.75 * max(uLightDir.y, 0.0))), vec3(2.0));
    vec4 cf = uViewProj * vec4(H, 1.0);
    float fogF = clamp(clamp((cf.z / cf.w), -1.0, 1.0) * uFogMul + uFogOff, 0.0, 255.0) / 255.0;
    return mix(col, uFogColor, fogF);
}

// Shoreline swash (ocean): the wash climbing the opaque ground beside a plane. Distance to the waterline along the ground is
// the height above the plane over the local slope (from the world-space normal), so a near-flat beach gets a wide band and a
// steep bank a narrow one, and level decks (slope 0) or walls get none. A front runs up the slope and drains back (fast
// uprush, slow backwash, staggered along the shore by noise); ground it has reached stays damp, and the front carries a lacy
// foam rim with a thin foam sheet behind it.
const float SWASH_REACH = 48.0;
void swashOverlay(ivec2 p, float d, vec3 P, float lvl, float gslope) {
    int best = -1;
    float bestDy = 1.0e9;
    for (int i = 0; i < uBoxCount; i++) {
        float dy = P.y - uBoxY[i];
        if (dy < 0.0 || dy > 14.0 || dy >= bestDy || uEye.y <= uBoxY[i]) {
            continue;
        }
        if (P.x < uBoxRect[i].x || P.x > uBoxRect[i].z || P.z < uBoxRect[i].y || P.z > uBoxRect[i].w) {
            continue;
        }
        best = i;
        bestDy = dy;
    }
    if (best < 0) {
        return;
    }
    float vd = texelFetch(uVisibleDepth, p, 0).r;
    if (vd < d - 1.0e-6 || lvl <= 0.0) {
        return; // an actor or effect is drawn in front of the ground, or this is a wall (cliff, pier post), not a beach
    }
    float dist = bestDy / clamp(gslope, 0.02, 0.6); // ground distance to the waterline, world units
    if (uDebug == 5) {
        // distance to the waterline: red < 10, green < 25, blue < 48, white < 77 (dim: flatness)
        oColor = vec4((dist < 10.0 ? vec3(1.0, 0.0, 0.0) : (dist < 25.0 ? vec3(0.0, 1.0, 0.0) : (dist < 48.0 ? vec3(0.0, 0.0, 1.0) : vec3(0.8)))) * (0.4 + 0.6 * lvl), 1.0);
        return;
    }
    if (dist > SWASH_REACH * 1.6 || gslope < 0.015) {
        return; // too far from the water, or perfectly level (a deck or float, not a beach)
    }
    float fp = length(P - uEye) * uPixelAngle;
    float edgeW = max(2.0, 2.2 * fp / max(abs(normalize(P - uEye).y), 0.15));
    float nA = vnoise(P.xz / 160.0);
    float nB = vnoise(P.xz / 38.0 + vec2(uTime * 0.05, 0.0));
    float nC = vnoise(P.xz / 7.0 - vec2(0.0, uTime * 0.15));
    float cyc = fract(uTime * 0.115 + nA * 1.7);
    float front = SWASH_REACH * smoothstep(0.0, 0.3, cyc) * (1.0 - smoothstep(0.3, 1.0, cyc)) * (0.55 + 0.45 * nB);
    float damp = SWASH_REACH * (0.22 + 0.2 * nA);
    float wetLine = max(front, damp);
    float wet = 1.0 - smoothstep(wetLine - edgeW, wetLine + edgeW * 0.5, dist);
    float swashOn = smoothstep(4.0, 12.0, front);
    float rim = exp(-pow((dist - front) / (edgeW * 1.4), 2.0)) * swashOn;
    float sheet = (1.0 - smoothstep(0.0, max(front, 1.0), dist)) * swashOn;
    float lace = smoothstep(0.32, 0.68, 0.6 * nB + 0.4 * nC);
    float foam = clamp(rim * (0.55 + 0.45 * lace) + sheet * 0.3 * lace, 0.0, 1.0) * uSwash * lvl;
    float dark = wet * 0.4 * uSwash * lvl;
    float a = 1.0 - (1.0 - dark) * (1.0 - foam);
    oColor = vec4(foamTint(P) * foam, a);
}

// Horizontal distance from a water point to the nearest bank, found by probing the opaque depth around its pixel at three world
// radii in eight directions. A probe counts as bank when it is a surface above the plane that is also close to the point on the plane (so
// a wading actor or the foreground in front of far water is not mistaken for the shore). 1e4 = none within about 16 units.
float shoreDist(ivec2 p, vec3 H, float planeY, float footprint) {
    const vec2 dirs[8] = vec2[8](vec2(1, 0), vec2(-1, 0), vec2(0, 1), vec2(0, -1), vec2(0.7, 0.7), vec2(-0.7, 0.7), vec2(0.7, -0.7), vec2(-0.7, -0.7));
    const float rw[3] = float[3](3.0, 8.0, 16.0);
    float best = 1.0e4;
    for (int k = 0; k < 3; k++) {
        float rpx = clamp(rw[k] / max(footprint, 1e-3), 1.5, 28.0);
        for (int i = 0; i < 8; i++) {
            ivec2 q = clamp(p + ivec2(round(dirs[i] * rpx)), ivec2(0), uSize - 1);
            float dq = fetchDepth(q);
            if (dq >= 0.99999) {
                continue;
            }
            vec3 Q = worldPos(q, dq);
            float dh = length(Q.xz - H.xz); // horizontal only: a tall wall seen at a grazing angle is far away in 3D but right at the bank
            if (Q.y >= planeY - 0.3 && dh < rw[k] * 2.5 + 4.0) {
                best = min(best, dh);
            }
        }
        if (best < 1.0e3) {
            break;
        }
    }
    return best;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    oColor = vec4(0.0);
    float d = fetchDepth(p); // opaque depth
    bool sky = d >= 0.99999;
    vec3 P = worldPos(p, sky ? 0.9999 : d);
    // How level the ground is (1 beach / flat, 0 wall), from the world-space surface normal.
    vec3 Ng = normalize(cross(dFdx(P), dFdy(P)));
    float lvl = smoothstep(0.45, 0.8, abs(Ng.y));
    float gny = max(abs(Ng.y), 0.2);
    float gslope = sqrt(max(1.0 - gny * gny, 0.0)) / gny;
    vec3 ray = P - uEye;
    float bestT;
    int bi = waterPlane(P, sky, bestT);
    if (bi < 0) {
        if (uSwash > 0.0 && !sky) {
            swashOverlay(p, d, P, lvl, gslope);
        }
        return;
    }
    // Opaque depth supplies the bed; scene-end depth still protects foreground XLU that writes depth.
    float planeY = uBoxY[bi];
    if (!sky && P.y >= planeY) {
        if (uSwash > 0.0) {
            swashOverlay(p, d, P, lvl, gslope); // ground just above the plane (a near-flat beach is within the depth tolerance)
        } else if (uFoam > 0.0 && P.y < planeY + 1.4 && lvl < 1.0) {
            // Meniscus: bank pixels a hair above the plane are the waterline itself; without foam on them the line is peppered
            // with the dark bank showing through (depth quantisation).
            float fm = (1.0 - smoothstep(0.2, 1.4, P.y - planeY)) * (1.0 - smoothstep(2.5, 8.0, length(ray) * uPixelAngle)) * uFoam * uSurfaceVisibility;
            float brk = shoreBreak(P.xz);
            fm *= 0.2 + 0.8 * brk;
            vec3 tintM = mix(vec3(1.0), uScatter / max(max(uScatter.r, uScatter.g), max(uScatter.b, 1e-3)), 0.2);
            oColor = vec4(foamTint(P) * (0.55 + 0.5 * brk) * tintM * fm * 0.6, fm * 0.6);
        }
        return;
    }
    float visibleDepth = texelFetch(uVisibleDepth, p, 0).r;
    if (visibleDepth < d && worldPos(p, visibleDepth).y > planeY + 1.5) {
        return;
    }
    // Ocean swell: the surface point is the traced displaced hit, not the plane. Thickness, path length and foam follow it.
    vec4 wv = uWaves != 0 ? texelFetch(uWaveBuf, p, 0) : vec4(0.0);
    bool wave = wv.a > 0.5;
    if (uDebug == 3 || uDebug == 4) {
        float hh = wv.r / max(uWaveAmp, 1.0);
        oColor = !wave ? vec4(0.2, 0.2, 0.2, 1.0) :
                 (uDebug == 3 ? vec4(max(hh, 0.0) * 1.6, 0.25, max(-hh, 0.0) * 3.2, 1.0) : vec4(0.5 + wv.g * 3.0, 0.5 + wv.b * 3.0, wv.a - 1.0, 1.0));
        return;
    }
    vec2 big = wave ? wv.gb : vec2(0.0);
    float surfY = planeY + (wave ? wv.r : 0.0);
    bestT = (surfY - uEye.y) / ray.y;
    vec3 H = uEye + ray * bestT;
    float dist = bestT * length(ray);
    vec3 V = normalize(uEye - H);
    float footprint = dist * uPixelAngle;
    // A pixel covers footprint / sin(elevation) along the view direction: patterns that are fine in that direction have to
    // fade out, or a tiled or sinusoidal pattern collapses into horizontal streaks on the open sea.
    float footprintS = footprint / max(V.y, 0.06);
    float graze = smoothstep(0.15, 0.5, V.y);
    float depthY = sky ? 1.0e4 : max(surfY - P.y, 0.0);

    // Rivers: the flow buffer holds the native texture scroll as a world velocity; streak normals and foam are advected along it.
    vec3 rs = vec3(0.0);
    bool river = false;
    vec2 flowV = vec2(0.0);
    float rfade = 1.0 - smoothstep(8.0, 34.0, footprintS);
    {
        vec2 gH = dFdx(H.xz), gH2 = dFdy(H.xz);
        if (uRiver > 0.0 && !sky) {
            vec4 fl = texelFetch(uFlowBuf, p, 0);
            if (fl.z > 0.5) {
                river = true;
                flowV = fl.xy;
                rs = flowSurface(H.xz, flowV, gH, gH2);
            }
        }
    }
    if (uDebug == 6) {
        oColor = vec4(river ? vec3(rs.z, 0.5 + flowV * 0.01) : vec3(0.0), 1.0); // flow mask and direction
        return;
    }
    vec2 riverSlope = rs.xy * (0.7 * uRiver * rfade);

    vec3 C = texelFetch(uScene, p, 0).rgb;
    vec3 D = C;
    float fogH = 0.0;
    if (uRefr > 0.0 || uRestyle > 0.0) {
        float s = sky ? 900.0 : min(length(P - H), 900.0);
        vec3 Bp = texelFetch(uOpaqueColor, p, 0).rgb;
        vec3 B = Bp; // the bed as seen through the (wobbling) surface
        if (!sky && uRefr > 0.0) {
            vec2 slope = refractSlope(H.xz, footprintS) * 1.6 + big + riverSlope * 1.6;
            vec3 N = normalize(vec3(-slope.x, 1.0, -slope.y));
            vec3 Rw = refract(-V, N, 1.0 / 1.33);
            vec3 Rf = refract(-V, vec3(0.0, 1.0, 0.0), 1.0 / 1.33);
            vec2 off = (Rw.xz / max(-Rw.y, 0.25) - Rf.xz / max(-Rf.y, 0.25)) * min(depthY, 220.0);
            vec2 uv = (vec2(p) + 0.5) / vec2(uSize);
            vec2 duv = projectUv(P + vec3(off.x, 0.0, off.y)) - projectUv(P);
            float len = length(duv);
            duv *= min(1.0, 0.012 / max(len, 1e-6));
            vec2 uv2 = uv + duv;
            // Validate every bilinear footprint texel, including the visible foreground depth. Invalid weight
            // returns to the unshifted bed instead of pulling a bank, actor or pole into its silhouette.
            vec2 texel = clamp(uv2 * vec2(uSize) - 0.5, vec2(0.0), vec2(uSize - 1));
            ivec2 base = ivec2(floor(texel));
            vec2 frac = fract(texel);
            vec3 shifted = vec3(0.0);
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 2; i++) {
                    ivec2 q = min(base + ivec2(i, j), uSize - 1);
                    float ds = fetchDepth(q);
                    vec3 Ps = worldPos(q, ds);
                    float dv = texelFetch(uVisibleDepth, q, 0).r;
                    float delta = abs(linDepth(d) - linDepth(ds));
                    float ok = (ds < 0.99999 && Ps.y < planeY - 0.5 &&
                                (dv >= ds || worldPos(q, dv).y < planeY)) ?
                               1.0 - smoothstep(4.0, 4.0 + 0.04 * linDepth(d), delta) : 0.0;
                    float weight = (i == 0 ? 1.0 - frac.x : frac.x) * (j == 0 ? 1.0 - frac.y : frac.y);
                    shifted += mix(Bp, texelFetch(uOpaqueColor, q, 0).rgb, ok) * weight;
                }
            }
            float wr = uRefr * smoothstep(3.0, 28.0, depthY) * (1.0 - smoothstep(150.0, 450.0, depthY));
            B = mix(Bp, shifted, wr);
        }
        if (uRestyle > 0.0) {
            // Physical model: the lit bed is transmitted with per-channel absorption, the rest is in-scatter of the
            // water body toward the biome's deep colour, plus a sunlit subsurface glow looking toward the light.
            vec3 T = exp(-s * uAbsorb);
            vec3 lit = uSky * 0.55 + uLightColor * (0.25 + 0.75 * max(uLightDir.y, 0.0)) * 0.45;
            float lumLit = dot(lit, vec3(0.2126, 0.7152, 0.0722));
            float E = clamp(lumLit, 0.45, 1.4);
            // The water body takes the colour of the light (golden at dusk, cool by moonlight).
            vec3 tint = mix(vec3(1.0), lit / max(lumLit, 1e-3), 0.3);
            float sss = pow(max(dot(-V, uLightDir), 0.0), 3.0) * (1.0 - uLightDir.y * 0.5);
            vec3 body = uScatter * tint * E + uLightColor * uScatter * (0.35 * sss);
            // The game fogs its own water by the depth of the surface; the replacement colour has to follow it.
            vec4 ch = uViewProj * vec4(H, 1.0);
            fogH = clamp(((ch.z / ch.w) * 0.5 + 0.5) * 2.0 - 1.0, -1.0, 1.0) * uFogMul + uFogOff;
            fogH = clamp(fogH, 0.0, 255.0) / 255.0;
            vec3 R = B * T + body * (1.0 - T);
            if (!sky) {
                // Wave facets facing the light are brighter (a stand-in for the water's own ripple lighting).
                vec2 ws = refractSlope(H.xz, footprintS) + riverSlope;
                float facet = clamp(dot(ws, normalize(uLightDir.xz + vec2(1e-3))) * 3.2, -0.3, 0.3);
                facet += clamp(dot(riverSlope, normalize(uLightDir.xz + vec2(1e-3))) * 3.0, -0.45, 0.45);
                if (wave) {
                    // Swell faces towards the light are brighter, faces away darker; a thin crest in front of the sun glows
                    // with the water's own scatter colour.
                    vec3 Nw = normalize(vec3(-big.x, 1.0, -big.y));
                    facet += clamp((dot(Nw, uLightDir) - uLightDir.y) * 3.0, -0.6, 0.6) +
                             clamp(wv.r / max(uWaveAmp, 1.0), -0.5, 1.0) * 0.2;
                    float crestN = clamp(wv.r / max(uWaveAmp * 0.617, 1.0), 0.0, 1.0);
                    R += uScatter * uLightColor * (1.3 * crestN * crestN * (0.3 + 0.7 * sss));
                }
                R *= 1.0 + facet;
            }
            // Keep the game's animated texture as detail: its high-pass luminance (ripples, caustic lines, splashes).
            float lc = dot(C, vec3(0.2126, 0.7152, 0.0722));
            vec3 average = vec3(0.0);
            const ivec2 taps[4] = ivec2[4](ivec2(-3, 0), ivec2(3, 0), ivec2(0, -3), ivec2(0, 3));
            for (int i = 0; i < 4; i++) {
                ivec2 q = clamp(p + taps[i], ivec2(0), uSize - 1);
                float dq = fetchDepth(q);
                float dv = texelFetch(uVisibleDepth, q, 0).r;
                bool sameWater = dq < 0.99999 && worldPos(q, dq).y < planeY &&
                                 (dv >= dq || worldPos(q, dv).y < planeY);
                average += sameWater ? texelFetch(uScene, q, 0).rgb : C;
            }
            average *= 0.25;
            float la = dot(average, vec3(0.2126, 0.7152, 0.0722));
            float hp = lc - la;
            float fade = 1.0 - smoothstep(4.0, 12.0, footprint);
            // Preserve bright, chromatic fine XLU details (splashes/particles). The painted ripple texture is only kept
            // where it is not stretched (looking down); at a low angle it is the source of the horizontal streaks.
            R = R * (1.0 + clamp(hp / max(la, 0.05) * 1.5, -0.35, 0.35) * fade * graze) + max(C - average * 1.3, vec3(0.0)) * fade;
            // Fog the entire replacement, after wave/detail modulation, including the transmitted bed.
            R = mix(R, uFogColor, fogH);
            D = mix(D + (B - Bp) * 0.6, R, uRestyle);
        } else {
            D += (B - Bp) * 0.6;
            // Beer-Lambert absorption of the seen path toward a deeper version of the game's own water colour.
            float luma = dot(C, vec3(0.2126, 0.7152, 0.0722));
            vec3 deep = max(mix(vec3(luma), C, 1.3) * 0.72, vec3(0.0));
            vec3 T = exp(-s * vec3(1.0 / 260.0, 1.0 / 420.0, 1.0 / 520.0));
            D = mix(D, deep, clamp(uRefr * (1.0 - T), 0.0, 1.0));
        }
    }

    float f = 0.0;
    vec3 foamCol = vec3(0.0);
    if (uFoam > 0.0 && !sky) {
        // Shoreline foam: a thin, crisp bright line on the waterline plus broken lacy foam that thins out over a short
        // distance, both read from the committed foam texture (R lace, A density) so they stay lace and not noise. Distance
        // to the waterline comes from the thickness and its screen gradient (a gentle shore gets the same line as a steep one).
        float n1 = vnoise(H.xz / 38.0 + vec2(0.05, -0.03) * uTime);
        float grad = length(vec2(dFdx(depthY), dFdy(depthY))) / max(footprint, 1e-3);
        float edge = depthY / clamp(grad, 0.12, 1.0);
        if (uSwash <= 0.0 && footprint < 3.0 && edge > 1.0) {
            edge = min(edge, shoreDist(p, H, planeY, footprint) * 0.9); // a wall bank has no depth ramp: measure to the bank itself
        }
        // The waterline surges in and out slowly (staggered along the shore), and the foam texture drifts with the river flow
        // or a slow wind drift; two layers half a cycle apart hide the loop.
        float surge = 0.5 + 0.5 * sin(uTime * 0.8 + n1 * 6.2831853 + dot(H.xz, vec2(0.013, 0.009)));
        edge = max(edge - surge * 0.9, 0.0);
        vec2 drift = river ? flowV * 0.5 : vec2(1.6, -1.0);
        vec2 gF = dFdx(H.xz) / 90.0, gF2 = dFdy(H.xz) / 90.0;
        float ph0 = fract(uTime / 3.0), ph1 = fract(uTime / 3.0 + 0.5);
        vec4 ft0 = textureGrad(uFoamTex, H.xz / 90.0 - drift * ph0 * 3.0 / 90.0, gF, gF2);
        vec4 ft1 = textureGrad(uFoamTex, H.xz / 90.0 + vec2(0.37, 0.61) - drift * ph1 * 3.0 / 90.0, gF, gF2);
        float w0 = 1.0 - abs(2.0 * ph0 - 1.0);
        vec4 ft = mix(ft1, ft0, w0);
        // Waterline: thin, variable thickness, broken by the shore noise and the lace pattern, soft on the water side.
        float brk = shoreBreak(H.xz);
        float lineW = max(0.8, footprint * 1.2) * (0.45 + 1.1 * brk);
        float line = pow(1.0 - smoothstep(0.0, lineW, edge), 1.5) * smoothstep(0.2, 0.6, 0.6 * brk + 0.52 * ft.r);
        // Lace beyond the line: a fading net, clearly visible, thinning out over 9-20 units.
        float reach = 10.0 + 8.0 * n1;
        float fall = 1.0 - smoothstep(0.0, reach, edge);
        float dens = clamp(ft.a * 1.7 + ft.r * 0.8 + ft.b * 0.6, 0.0, 1.0);
        float thr = 0.9 - 0.55 * fall; // dense near the line, only the thickest lace far out
        float lacy = smoothstep(thr, thr + 0.3, dens) * smoothstep(0.0, 0.15, fall);
        float core = max(line, lacy * 0.9);
        float foam = core * (1.0 - smoothstep(3.0, 9.0, footprint));
        f = min(foam, 0.9) * uFoam;
        // Brightness follows the lace pattern: most of the foam sits at 55-70 %, only the dense peaks reach HDR white. The
        // colour is the scene's own sun/sky/fog light (foamTint), so it is dim in shade and at night. The low-alpha fringe picks
        // up the water's scatter colour as a cheap subsurface tint.
        float peak = smoothstep(0.35, 0.95, ft.r * 0.6 + ft.a * 0.6 + lacy * 0.3);
        vec3 tint = mix(vec3(1.0), uScatter / max(max(uScatter.r, uScatter.g), max(uScatter.b, 1e-3)), 0.45);
        foamCol = foamTint(H) * mix(0.4, 1.25, peak) * mix(tint, vec3(1.0), smoothstep(0.35, 0.9, core));
    }
    if (river && uFoam > 0.0 && !sky) {
        // Streak foam: the thick ridges of the flowing mask, strongest where the water is shallow (banks, rocks in the
        // stream) and where it runs fast. The depth-based contact foam above still marks the shoreline itself.
        float sp = clamp(length(flowV) / 45.0, 0.0, 1.0);
        float shallow = 1.0 - smoothstep(14.0, 110.0, depthY);
        float rf = smoothstep(0.3, 0.8, rs.z) * (0.3 + 0.7 * shallow) * (0.35 + 0.65 * sp) *
                   (1.0 - smoothstep(4.0, 14.0, footprintS)) * uFoam * min(uRiver, 1.5) * 0.9;
        if (rf > f) {
            f = rf;
            foamCol = foamTint(H);
        }
    }
    if (wave && uFoam > 0.0) {
        // Crest whitecaps and the wake foam of the ripple field (both are in the trace pass's alpha). Foam is only opaque where
        // two fine noises agree, so it reads as thin, broken streaks and lace, never as a solid sheet.
        float cf = clamp(wv.a - 1.0, 0.0, 1.0);
        if (cf > 0.0) {
            float n1 = vnoise(H.xz / 2.6 + vec2(0.3, -0.2) * uTime);
            float n2 = vnoise(H.xz / 7.5 - vec2(0.15, 0.1) * uTime + 4.0);
            float broken = smoothstep(0.3, 0.85, cf * (0.3 + 1.0 * (0.6 * n1 + 0.4 * n2)));
            float fw = broken * (1.0 - smoothstep(5.0, 16.0, footprintS)) * uFoam * 0.65;
            if (fw > f) {
                f = fw;
                foamCol = foamTint(H);
            }
        }
    }
    f *= uSurfaceVisibility;
    D = mix(C, D, uSurfaceVisibility);
    oColor = vec4(foamCol * f + (D - C) * (1.0 - f), f);
}
)GLSL";

static const char* kWaterReflectionApplyShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uReflection;
uniform int uDebug;
uniform float uSurfaceDetail;
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
    bool sharp = uSurfaceDetail > 0.5;
    // Four depth-aware bilinear taps keep half-res SSR stable without the old 4x4 Gaussian smear.
    int first = sharp ? 0 : -1;
    int last = sharp ? 1 : 2;
    for (int y = first; y <= last; y++) {
        for (int x = first; x <= last; x++) {
            ivec2 q = clamp(base + ivec2(x, y), ivec2(0), halfSize - 1);
            vec2 delta = vec2(q) - samplePos;
            ivec2 fullQ = clamp(ivec2((vec2(q) + 0.5) / vec2(halfSize) * vec2(uSize)), ivec2(0), uSize - 1);
            float tapDepth = linDepth(fetchDepth(fullQ));
            float spatial = sharp ? max(1.0 - abs(delta.x), 0.0) * max(1.0 - abs(delta.y), 0.0) :
                                    exp(-dot(delta, delta) / 3.0);
            float w = spatial * exp(-abs(tapDepth - depth) / depthScale);
            color += texelFetch(uReflection, q, 0) * w;
            weight += w;
        }
    }
    oColor = color / max(weight, 1e-6);
}
)GLSL";

// ---------------------------------------------------------------------------------------------------------------------
// Water simulation (docs/WATER_SIM.md). Render only: the WaterBox plane is the mean sea level and stays the game's
// authority; swimming, buoyancy and collision never see the displaced surface.
//  1. Depth map: eighth-resolution blurred water depth (plane minus the opaque bed; land counts as zero). It tapers the
//     swell so waves die away over shallows, beaches, piers and rocks without a hard edge.
//  2. Wave trace: per water pixel, a ray march of a height field (three sharpened-sine, trochoid-like swells travelling
//     with the shared wind, plus the GPU ripple height field) between plane +/- the amplitude bound. It stores the real
//     displaced hit as (height above plane, slope x, slope z, 1 + foam) so the surface, glint and SSR passes shade the
//     displaced surface with a consistent normal. A = 0 means "no wave": those passes keep the flat-plane behaviour.
//  3. Ripples: a wave equation (height + velocity + foam) on a 256^2 window around the player, fed by read-only sources.
// ---------------------------------------------------------------------------------------------------------------------
static const char* kWaterDepthMapShader = R"GLSL(
out vec4 oColor;
void main() {
    ivec2 q = ivec2(gl_FragCoord.xy) * 8;
    float sum = 0.0;
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            ivec2 p = min(q + ivec2(i, j) * 2 + 1, uSize - 1); // 16 taps spread over the 8x8 block
            float d = fetchDepth(p);
            bool sky = d >= 0.99999;
            vec3 P = worldPos(p, sky ? 0.9999 : d);
            float t;
            int bi = waterPlane(P, sky, t);
            if (bi >= 0) {
                sum += sky ? 400.0 : clamp(uBoxY[bi] - P.y, 0.0, 400.0);
            }
        }
    }
    oColor = vec4(sum * (1.0 / 16.0), 0.0, 0.0, 1.0);
}
)GLSL";

static const char* kWaterWaveLib = R"GLSL(
uniform float uTime;
uniform vec2 uWaveDirs[6]; // direction each swell travels: the wind, fanned out by fixed angles
uniform float uWaveAmp;    // swell amplitude (world units)
uniform vec4 uRippleWin;   // window min X, min Z, 1 / size, active
uniform sampler2D uRipple; // R height, G velocity, B foam

// Six swell trains. The first three carry the displacement the march sees; the finer three only shape the normal.
const float WV_LEN[6] = float[6](470.0, 270.0, 160.0, 105.0, 68.0, 44.0);
const float WV_FRAC[6] = float[6](1.0, 0.55, 0.3, 0.16, 0.09, 0.05);
const float WV_PH[6] = float[6](0.0, 1.9, 4.1, 0.7, 2.6, 5.3);
const float WV_SHARP = 1.4;
const float WV_MEAN = 0.3831; // mean of exp(1.4 (sin - 1)): the WaterBox plane stays the mean sea level
const float WV_ENV_MAX = 1.6; // upper bound of waveEnvelope

float wvHash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float wvNoise(vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(wvHash(i), wvHash(i + vec2(1.0, 0.0)), f.x), mix(wvHash(i + vec2(0.0, 1.0)), wvHash(i + vec2(1.0, 1.0)), f.x), f.y);
}

// Every train has its own slowly drifting envelope (a product of two long sines, 0.2 .. 1.6): trains come and go, so the
// crests break up into irregular patches instead of one set of parallel lines.
float waveEnvelope(int i, vec2 x) {
    float fi = float(i);
    float a = dot(x, vec2(0.0043 + 0.0011 * fi, -0.0021 - 0.0007 * fi)) + 1.7 * fi + 0.021 * uTime * (1.0 + 0.3 * fi);
    float b = dot(x, vec2(-0.0019 - 0.0005 * fi, 0.0039 + 0.0009 * fi)) + 3.1 * fi - 0.017 * uTime;
    return 0.2 + 0.7 * (1.0 + sin(a) * sin(b));
}

// detail = false: the three displacing trains only (used by the march); true: all six (normals, foam).
float oceanHeight(vec2 x, float amp, vec3 fadeA, vec3 fadeB, bool detail) {
    // A slow phase warp bends the crest lines so the swell never reads as a perfect set of straight stripes.
    float warp = 1.7 * sin(0.0011 * x.x - 0.0017 * x.y + 0.05 * uTime) + 1.1 * sin(0.0023 * x.x + 0.0009 * x.y - 0.03 * uTime) +
                 3.5 * (wvNoise(x * (1.0 / 520.0) + 11.7) - 0.5);
    float h = 0.0;
    for (int i = 0; i < 6; i++) {
        if (i >= 3 && !detail) {
            break;
        }
        float fade = i < 3 ? fadeA[i] : fadeB[i - 3];
        if (fade <= 0.0) {
            continue;
        }
        float k = 6.2831853 / WV_LEN[i];
        // Deep-water dispersion: phase speed grows with the square root of the wavelength.
        float th = k * (dot(uWaveDirs[i], x) - 3.455 * sqrt(WV_LEN[i]) * uTime) + WV_PH[i] + warp * (0.5 + 0.35 * float(i));
        h += amp * WV_FRAC[i] * fade * waveEnvelope(i, x) * (exp(WV_SHARP * (sin(th) - 1.0)) - WV_MEAN);
    }
    return h;
}

// Crest height of the same sum (the foam's reference): 0.617 of every train's local amplitude.
float oceanCrest(vec2 x, float amp, vec3 fadeA, vec3 fadeB) {
    float c = 0.0;
    for (int i = 0; i < 6; i++) {
        c += amp * WV_FRAC[i] * (i < 3 ? fadeA[i] : fadeB[i - 3]) * waveEnvelope(i, x);
    }
    return 0.617 * c;
}

vec3 rippleAt(vec2 x) {
    return texture(uRipple, (x - uRippleWin.xy) * uRippleWin.z).rgb;
}

float surfaceHeight(vec2 x, float amp, vec3 fadeA, vec3 fadeB, bool detail, float simW) {
    float h = oceanHeight(x, amp, fadeA, fadeB, detail);
    if (simW > 0.0) {
        h += simW * rippleAt(x).r;
    }
    return h;
}


)GLSL";

static const char* kWaterWaveShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uWaveDepth;
uniform float uPixelAngle;
uniform float uWaveGust;

void main() {
    oColor = vec4(0.0);
    ivec2 p = ivec2(gl_FragCoord.xy);
    float d = fetchDepth(p);
    bool sky = d >= 0.99999;
    vec3 P = worldPos(p, sky ? 0.9999 : d);
    vec3 ray = P - uEye;
    float bestT;
    int bi = waterPlane(P, sky, bestT);
    if (bi < 0) {
        return;
    }
    float planeY = uBoxY[bi];
    if (!sky && P.y >= planeY) {
        return;
    }
    float rayLen = length(ray);
    vec3 dir = ray / rayLen;
    float tPlane = (planeY - uEye.y) / dir.y;
    vec3 H0 = uEye + dir * tPlane;
    float footprint = tPlane * uPixelAngle;
    // Taper: the swell needs deep water under it, and a camera well above it.
    float bed = texture(uWaveDepth, (vec2(p) + 0.5) / vec2(uSize)).r;
    float taper = smoothstep(10.0, 70.0, bed);
    float camFade = smoothstep(uWaveAmp * 1.5, uWaveAmp * 5.0 + 1.0, uEye.y - planeY);
    float amp = uWaveAmp * taper * camFade;
    // A train fades out once a pixel covers a good fraction of its wavelength along the view direction (footprint stretched by
    // 1 / sin(elevation)): distant water converges to a smooth, glossy surface instead of banding, and a far crest can neither
    // alias nor be missed by the march.
    float footprintS = footprint / max(-dir.y, 0.06);
    vec3 fadeA = 1.0 - smoothstep(0.03, 0.3, vec3(footprintS) / vec3(WV_LEN[0], WV_LEN[1], WV_LEN[2]));
    vec3 fadeB = 1.0 - smoothstep(0.03, 0.3, vec3(footprintS) / vec3(WV_LEN[3], WV_LEN[4], WV_LEN[5]));
    float simW = 0.0;
    if (uRippleWin.w > 0.5) {
        vec2 win = (H0.xz - uRippleWin.xy) * uRippleWin.z;
        if (min(min(win.x, win.y), min(1.0 - win.x, 1.0 - win.y)) > -0.25) {
            simW = smoothstep(2.0, 16.0, bed); // no displacement on the beach; the normal still ripples
        }
    }
    float sumF = WV_FRAC[0] * fadeA.x + WV_FRAC[1] * fadeA.y + WV_FRAC[2] * fadeA.z;
    float simBound = simW > 0.0 ? 4.0 : 0.0;
    float hi = amp * WV_ENV_MAX * 0.617 * sumF + simBound;
    float lo = amp * WV_ENV_MAX * 0.32 * sumF + simBound;
    if (hi + lo < 0.05) {
        return;
    }
    float tIn = max((planeY + hi - uEye.y) / dir.y, 0.0);
    float tOut = min((planeY - lo - uEye.y) / dir.y, rayLen);
    if (tIn >= tOut) {
        return;
    }
    const int N = 14;
    float dt = (tOut - tIn) / float(N);
    float ta = tIn;
    vec3 pa = uEye + dir * ta;
    float fa = pa.y - planeY - surfaceHeight(pa.xz, amp, fadeA, fadeB, false, simW);
    if (fa <= 0.0) {
        return; // the camera is inside the swell
    }
    float tHit = -1.0;
    for (int i = 1; i <= N; i++) {
        float tb = tIn + dt * float(i);
        vec3 pb = uEye + dir * tb;
        float fb = pb.y - planeY - surfaceHeight(pb.xz, amp, fadeA, fadeB, false, simW);
        if (fb <= 0.0) {
            float a = ta, b = tb, fA = fa, fB = fb;
            for (int j = 0; j < 4; j++) {
                float m = a + (b - a) * fA / (fA - fB);
                vec3 pm = uEye + dir * m;
                float fm = pm.y - planeY - surfaceHeight(pm.xz, amp, fadeA, fadeB, false, simW);
                if (fm > 0.0) {
                    a = m;
                    fA = fm;
                } else {
                    b = m;
                    fB = fm;
                }
            }
            tHit = 0.5 * (a + b);
            break;
        }
        ta = tb;
        fa = fb;
    }
    if (tHit < 0.0) {
        return; // the bed is nearer than the surface: leave the flat plane
    }
    vec3 H = uEye + dir * tHit;
    float e = 3.0 + footprint * 0.6;
    float h0 = surfaceHeight(H.xz, amp, fadeA, fadeB, true, simW);
    float hx = surfaceHeight(H.xz + vec2(e, 0.0), amp, fadeA, fadeB, true, simW);
    float hz = surfaceHeight(H.xz + vec2(0.0, e), amp, fadeA, fadeB, true, simW);
    vec2 slope = vec2(hx - h0, hz - h0) / e;
    float hOff = H.y - planeY;
    // Whitecaps ride the crest tops of the strongest trains, broken up by noise and gated into patches; stronger wind, more
    // caps. They only exist where the pixel still resolves them (no stretched white smears at a grazing angle).
    float crestH = oceanCrest(H.xz, amp, fadeA, fadeB);
    float hn = crestH > 0.01 ? oceanHeight(H.xz, amp, fadeA, fadeB, true) / crestH : 0.0;
    float breakup = 0.6 * wvNoise(H.xz / 11.0 + uWaveDirs[0] * uTime * 4.0) + 0.4 * wvNoise(H.xz / 3.7 - uWaveDirs[1] * uTime * 6.0);
    float patch = wvNoise(H.xz / 85.0 + uWaveDirs[0] * uTime * 2.0 + 17.0);
    float capFade = 1.0 - smoothstep(6.0, 18.0, footprintS);
    float cap = smoothstep(0.68, 1.0, hn + (breakup - 0.5) * 0.5 + (patch - 0.5) * 1.1) * (0.25 + 0.75 * uWaveGust) * taper * capFade;
    float wake = uRippleWin.w > 0.5 ? rippleAt(H.xz).b : 0.0;
    float foam = clamp(max(cap, wake), 0.0, 1.0);
    if (any(isnan(vec4(hOff, slope, foam))) || any(isinf(vec4(hOff, slope, foam)))) {
        return;
    }
    oColor = vec4(hOff, slope, 1.0 + foam);
}
)GLSL";

// One step of the ripple wave equation, semi-implicit Euler (v first, then h with the new v). State: R height, G velocity,
// B foam. The window scrolls with the player in whole texels (uShift), the border is absorbing (no reflections).
static const char* kWaterRippleShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uState;
uniform ivec2 uShift;
uniform float uDt;
uniform float uC2;
uniform float uDamp;
uniform vec3 uWin; // new window min X, min Z, world units per texel
uniform int uSrcCount;
uniform vec4 uSrc[16]; // x, z, radius, strength

vec4 st(ivec2 p) {
    ivec2 s = textureSize(uState, 0);
    p += uShift;
    if (p.x < 0 || p.y < 0 || p.x >= s.x || p.y >= s.y) {
        return vec4(0.0);
    }
    return texelFetch(uState, p, 0);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 s = textureSize(uState, 0);
    vec4 c = st(p);
    float lap = st(p + ivec2(-1, 0)).r + st(p + ivec2(1, 0)).r + st(p + ivec2(0, -1)).r + st(p + ivec2(0, 1)).r - 4.0 * c.r;
    float v = c.g + (uC2 * lap - uDamp * c.g) * uDt;
    // Wake foam is thin and short lived: it is raised where the surface is being agitated hard (the ring crests) plus a narrow
    // line right under a moving source, and it fades in about a second.
    float foam = c.b * exp(-uDt / 1.1);
    foam += smoothstep(18.0, 70.0, abs(c.g)) * 1.6 * uDt;
    vec2 world = uWin.xy + (vec2(p) + 0.5) * uWin.z;
    for (int i = 0; i < 16; i++) {
        if (i >= uSrcCount) {
            break;
        }
        vec2 dl = world - uSrc[i].xy;
        float g = exp(-dot(dl, dl) / (uSrc[i].z * uSrc[i].z));
        v += uSrc[i].w * g * uDt;
        float gn = exp(-dot(dl, dl) / (0.2 * uSrc[i].z * uSrc[i].z));
        foam += gn * clamp(abs(uSrc[i].w) / 450.0, 0.0, 1.0) * 1.6 * uDt;
    }
    float h = c.r + v * uDt;
    int edge = min(min(p.x, p.y), min(s.x - 1 - p.x, s.y - 1 - p.y));
    float a = smoothstep(0.0, 14.0, float(edge));
    oColor = vec4(clamp(h, -4.0, 4.0) * a, clamp(v, -400.0, 400.0) * a, min(foam, 0.8) * a, 0.0);
}
)GLSL";

// ---------------------------------------------------------------------------------------------------------------------
// Water caustics and underwater rendering. Both read the WaterBox planes (kWaterPlaneLib), the sun shadow map and one
// self-rendered 32-frame seamless caustics flipbook (a texture array on kCausticUnit, see tools/newdawn/caustics).
// ---------------------------------------------------------------------------------------------------------------------
static const char* kCausticLib = R"GLSL(
uniform sampler2DArray uCausticTex;
uniform float uCausticFrame; // playback position, in flipbook frames
uniform sampler2DShadow uShadowCmp;
uniform bool uShadowReady;
uniform mat4 uLightViewProj;
const float kCausticTile = 320.0;
#ifdef CSM
float csmTap(vec3 uvz) {
    return texture(uShadowCmp, uvz);
}
#endif
// Photon density (mean 1) at a world xz position, two flipbook layers at different scale/rotation/phase so the
// repetition is not readable. lod blurs the pattern for the soft god rays.
float causticSample(vec2 uv, float frame, float lod) {
    float f0 = floor(frame);
    float a = mod(f0, 32.0);
    float b = mod(f0 + 1.0, 32.0);
    return mix(textureLod(uCausticTex, vec3(uv, a), lod).r, textureLod(uCausticTex, vec3(uv, b), lod).r, frame - f0) * 4.0;
}
float causticPattern(vec2 xz, float lod) {
    float a = causticSample(xz / kCausticTile, uCausticFrame, lod);
    vec2 r = vec2(xz.x * 0.866 - xz.y * 0.5, xz.x * 0.5 + xz.y * 0.866);
    float b = causticSample(r / (kCausticTile * 0.62) + 0.37, uCausticFrame * 0.8 + 11.0, lod);
    return a * (0.35 + 0.65 * b);
}
float sunVisibility(vec3 pos) {
    if (!uShadowReady) {
        return 1.0;
    }
#ifdef CSM
    float s = csmLit(pos, vec3(0.0), csmViewDepth(pos), 0.0, 0.004);
    return s < 0.0 ? 1.0 : s;
#else
    vec4 lc = uLightViewProj * vec4(pos, 1.0);
    vec3 sc = lc.xyz / lc.w * 0.5 + 0.5;
    if (any(lessThan(sc, vec3(0.0))) || any(greaterThan(sc, vec3(1.0)))) {
        return 1.0;
    }
    float edge = smoothstep(0.0, 0.04, min(min(sc.x, 1.0 - sc.x), min(sc.y, 1.0 - sc.y)));
    return mix(1.0, texture(uShadowCmp, vec3(sc.xy, sc.z - 0.003)), edge);
#endif
}
float rectMask(vec2 w, vec4 rect, float soft) {
    vec2 m = min(w - rect.xy, rect.zw - w);
    return smoothstep(0.0, soft, min(m.x, m.y));
}
)GLSL";

// Caustics: runs at the opaque end, before the opaque colour capture, so refraction/absorption of the bed includes them.
// Output is a gain g, blended as dst * (1 + g). Below a plane the sun enters the surface at W and reaches the surface
// along the refracted direction; just above a plane the surface reflects the light onto it (reflected direction).
static const char* kCausticsShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform float uStrength;
uniform float uPixelAngle;
uniform float uFogMul;
uniform float uFogOff;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAbsorb;

vec3 normalFromDepth(ivec2 p, vec3 P) {
    vec3 L = worldPos(p - ivec2(1, 0), fetchDepth(p - ivec2(1, 0)));
    vec3 R = worldPos(p + ivec2(1, 0), fetchDepth(p + ivec2(1, 0)));
    vec3 B = worldPos(p - ivec2(0, 1), fetchDepth(p - ivec2(0, 1)));
    vec3 T = worldPos(p + ivec2(0, 1), fetchDepth(p + ivec2(0, 1)));
    vec3 dx = length(R - P) < length(P - L) ? R - P : P - L;
    vec3 dy = length(T - P) < length(P - B) ? T - P : P - B;
    vec3 N = normalize(cross(dx, dy) + vec3(0.0, 1e-6, 0.0));
    return dot(N, uEye - P) < 0.0 ? -N : N;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    oColor = vec4(0.0);
    float d = fetchDepth(p);
    if (d >= 0.99999) {
        return;
    }
    vec3 P = worldPos(p, d);
    vec3 Lt = normalize(uLightDir);
    if (Lt.y < 0.12) {
        return;
    }
    vec3 up = normalize(-refract(-Lt, vec3(0.0, 1.0, 0.0), 1.0 / 1.33)); // refracted light, towards the surface
    vec3 Rf = vec3(-Lt.x, Lt.y, -Lt.z);                                    // mirror-reflected light, away from it
    float fres = 0.02 + 0.98 * pow(1.0 - Lt.y, 5.0);
    float dist = length(P - uEye);
    vec3 N = normalFromDepth(p, P);
    float gain = 0.0;
    vec3 tint = vec3(1.0);
    for (int i = 0; i < uBoxCount; i++) {
        float y = uBoxY[i];
        float below = y - P.y;
        if (below > 2.0 && below < 700.0) {
            float t = below / up.y;
            vec2 W = P.xz + up.xz * t;
            float inside = rectMask(W, uBoxRect[i], 14.0);
            if (inside <= 0.0) {
                continue;
            }
            float facing = clamp(dot(N, up) / max(up.y, 0.4), 0.0, 1.0);
            float vis = min(sunVisibility(vec3(W.x, y + 1.0, W.y)), sunVisibility(P + N * 2.0));
            float lod = clamp(log2(max(dist * uPixelAngle / kCausticTile * 256.0, 1.0)), 0.0, 5.0);
            float c = causticPattern(W, lod);
            float fade = exp(-below / 130.0);
            gain = c * facing * vis * fade * inside * 1.1;
            tint = exp(-t * uAbsorb * 0.8);
            break;
        } else if (below < -2.0 && below > -90.0) {
            float h = P.y - y;
            float t = h / Rf.y;
            vec2 W = P.xz - Rf.xz * t;
            float inside = rectMask(W, uBoxRect[i], 8.0);
            if (inside <= 0.0) {
                continue;
            }
            float facing = clamp(dot(N, -Rf) * 1.2, 0.0, 1.0);
            float vis = sunVisibility(vec3(W.x, y + 1.0, W.y));
            float c = causticPattern(W * 0.8, clamp(log2(max(dist * uPixelAngle / kCausticTile * 256.0, 1.0)), 0.0, 5.0));
            gain = c * facing * vis * exp(-h / 40.0) * inside * (0.12 + 3.0 * fres);
            break;
        }
    }
    if (gain <= 0.0) {
        return;
    }
    // Light colour normalised to a unit luminance; its brightness (the sun fade) lives in uLightColor's length.
    float lum = max(dot(uLightColor, vec3(0.2126, 0.7152, 0.0722)), 1e-4);
    vec3 col = uLightColor / lum * min(lum, 1.0);
    float fog = clamp(clamp(d * 2.0 - 1.0, -1.0, 1.0) * uFogMul + uFogOff, 0.0, 255.0) / 255.0;
    oColor = vec4(col * tint * gain * uStrength * (1.0 - fog), 0.0);
}
)GLSL";

// Underwater: camera below a plane. One ray march per pixel through the water between the camera and the geometry
// (or the surface above), accumulating depth-dependent ambient in-scatter and sun shafts that follow the surface
// caustic pattern and the sun shadow map, and attenuating the scene with the palette's absorption. A small
// time-varying screen displacement gives the wavy refraction. uUnder cross-fades the whole thing.
static const char* kUnderwaterShader = R"GLSL(
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uScene;
uniform sampler2D uUwShafts;
uniform float uTime;
uniform float uUnder;
uniform float uPlaneY;
uniform vec4 uEyeRect;
uniform vec3 uScatter;
uniform vec3 uAbsorb;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uSky;
uniform float uShaftStrength;

float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// Water interval [t0, t1] (distance from the eye) along the view ray through pixel q.
vec3 rayDir(ivec2 q, out float s, out float t0, out float t1) {
    float dq = fetchDepth(q);
    bool sky = dq >= 0.99999;
    vec3 ray = worldPos(q, sky ? 0.9999 : dq) - uEye;
    vec3 dir = ray / max(length(ray), 1e-3);
    s = sky ? 2200.0 : length(ray);
    t0 = 0.0;
    t1 = s;
    float tp = abs(dir.y) > 1e-4 ? (uPlaneY - uEye.y) / dir.y : 1e9;
    if (uEye.y < uPlaneY) {
        if (dir.y > 0.0) {
            t1 = min(t1, tp);
        }
    } else if (dir.y < 0.0) {
        t0 = min(max(tp, 0.0), s);
    } else {
        t1 = 0.0;
    }
    return dir;
}

vec3 sigmaCoefficient() {
    return uAbsorb * 0.5 + vec3(1.0 / 2600.0);
}

#ifdef SHAFT_PASS
// Half resolution: the sunlit light shafts only (the expensive, texture and shadow bound part).
void main() {
    ivec2 q = min(ivec2(gl_FragCoord.xy) * 2, uSize - 1);
    float s, t0, t1;
    vec3 dir = rayDir(q, s, t0, t1);
    vec3 Lt = normalize(uLightDir);
    oColor = vec4(0.0);
    if (Lt.y <= 0.12 || dot(uLightColor, vec3(1.0)) <= 1e-3 || uShaftStrength <= 0.0) {
        return;
    }
    vec3 up = normalize(-refract(-Lt, vec3(0.0, 1.0, 0.0), 1.0 / 1.33));
    float g = 0.55;
    float cosT = dot(-up, -dir);
    float phase = (1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * cosT, 0.05), 1.5);
    float lumSun = dot(uLightColor, vec3(0.2126, 0.7152, 0.0722));
    vec3 sunCol = uLightColor / max(lumSun, 1e-4) * min(lumSun, 1.0);
    vec3 sigma = sigmaCoefficient();
    const int kSteps = 16;
    float ds = max(t1 - t0, 0.0) / float(kSteps);
    float jitter = ign(gl_FragCoord.xy);
    float lod = clamp(log2(max(ds * 0.35, 1.0)), 2.0, 6.0);
    vec3 T = vec3(1.0);
    vec3 scat = vec3(0.0);
    for (int i = 0; i < kSteps; ++i) {
        vec3 X = uEye + dir * (t0 + (float(i) + jitter) * ds);
        float depthX = max(uPlaneY - X.y, 0.0);
        vec2 W = X.xz + up.xz * (depthX / up.y);
        float vis = rectMask(W, uEyeRect, 16.0);
        vec3 stepT = exp(-sigma * ds);
        if (vis > 0.0) {
            vis *= sunVisibility(vec3(W.x, uPlaneY + 1.0, W.y));
            float pat = causticPattern(W, lod);
            vec3 sunAtt = exp(-depthX / up.y * uAbsorb);
            scat += T * (1.0 - stepT) * sunCol * sunAtt * (pat * pat * vis * phase * 0.9 * uShaftStrength);
        }
        T *= stepT;
    }
    oColor = vec4(scat, 1.0);
}
#else
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 sz = vec2(uSize);
    vec2 uv = (vec2(p) + 0.5) / sz;
    vec3 C0 = texelFetch(uScene, p, 0).rgb;
    // Wavy refraction: slow broad swell plus finer ripple, about 3 px.
    vec2 w = vec2(sin(uv.y * 17.0 + uTime * 1.3) + 0.6 * sin(uv.y * 43.0 - uTime * 2.1 + uv.x * 9.0),
                  sin(uv.x * 21.0 + uTime * 1.1) + 0.6 * sin(uv.x * 51.0 + uTime * 1.9 - uv.y * 7.0));
    vec2 suv = clamp(uv + w * 0.0016 * uUnder * vec2(1.0, sz.x / sz.y), 0.5 / sz, 1.0 - 0.5 / sz);
    ivec2 q = ivec2(suv * sz);
    vec3 C = texture(uScene, suv).rgb;
    float s, t0, t1;
    vec3 dir = rayDir(q, s, t0, t1);
    vec3 Lt = normalize(uLightDir);
    vec3 sigma = sigmaCoefficient();
    vec3 lit = uSky * 0.55 + uLightColor * (0.25 + 0.75 * max(Lt.y, 0.0)) * 0.45;
    float lumLit = dot(lit, vec3(0.2126, 0.7152, 0.0722));
    float E = clamp(lumLit, 0.45, 1.4);
    vec3 tintLight = mix(vec3(1.0), lit / max(lumLit, 1e-3), 0.3);
    const int kSteps = 8;
    float ds = max(t1 - t0, 0.0) / float(kSteps);
    float jitter = ign(gl_FragCoord.xy);
    vec3 T = vec3(1.0);
    vec3 scat = vec3(0.0);
    vec3 stepT = exp(-sigma * ds);
    for (int i = 0; i < kSteps; ++i) {
        float depthX = max(uPlaneY - (uEye.y + dir.y * (t0 + (float(i) + jitter) * ds)), 0.0);
        vec3 S = uScatter * tintLight * E * max(exp(-depthX / 260.0), 0.18);
        scat += T * (1.0 - stepT) * S;
        T *= stepT;
    }
    scat += texture(uUwShafts, suv).rgb;
    vec3 result = C * T + scat;
    oColor = vec4(mix(C0, result, clamp(uUnder, 0.0, 1.0)), 1.0);
}
#endif
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
    // Assign sampler units before any validation/draw, including legacy and feature-off programs.
    // Cubes and volume/emitter own 12..15; shared colour, depth and shadow compare use 8, 9 and 11.
    GLint previous;
    glGetIntegerv(GL_CURRENT_PROGRAM, &previous);
    glUseProgram(prg);
    for (auto sampler : { std::pair<const char*, int>{ "uSkyIrradiance", PostFxGL::kSkyAmbientUnit },
                          { "uSkySpecular", PostFxGL::kSkyReflectionUnit },
                          { "uFogVolume", PostFxGL::kFogUnit },
                          { "uFlareEmitter", PostFxGL::kFlareUnit },
                          { "uFlareSprites", PostFxGL::kFlareSpritesUnit },
                          { "uFlareClouds", PostFxGL::kFlareCloudUnit },
                          { "uScene", 8 },
                          { "uDepth", 9 },
                          { "uGiRaw", 10 },
                          { "uGiShadow", 10 },
                          { "uGiHistory", 11 },
                          { "uTaaHistory", 10 },
                          { "uTaaMask", 11 },
                          { "uRayMask", 8 },
                          { "uRays", 11 },
                          { "uRaySky", 10 },
                          { "uShadowCmp", 11 },
                          { "uCausticTex", PostFxGL::kCausticUnit },
                          { "uMilkyWay", PostFxGL::kMilkyWayUnit },
                          { "uClouds", 8 },
                          { "uUwShafts", 10 },
                          { "uWaveBuf", PostFxGL::kWaveBufUnit },
                          { "uWaveDepth", PostFxGL::kWaveDepthUnit },
                          { "uRipple", PostFxGL::kRippleUnit },
                          { "uFlowBuf", PostFxGL::kFlowBufUnit },
                          { "uFlowNoise", PostFxGL::kFlowNoiseUnit },
                          { "uFoamTex", PostFxGL::kFoamTexUnit },
                          { "uPsCube0", PostFxGL::kPointShadowUnit0 },
                          { "uPsCube1", PostFxGL::kPointShadowUnit1 } }) {
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
    if (units <= kFoamTexUnit) {
        SPDLOG_WARN("NewDawnPostFx: disabled, only {} texture units available", units);
        return;
    }

    // Fog/emitter own units 14/15. Keep complete transparent dummies bound even with the features off.
    GLint previousActive;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActive);
    glActiveTexture(GL_TEXTURE0 + kFogUnit);
    glGenTextures(1, &mFogNeutral);
    glBindTexture(GL_TEXTURE_2D, mFogNeutral);
    const float neutral[4] = {};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 1, 1, 0, GL_RGBA, GL_FLOAT, neutral);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glActiveTexture(GL_TEXTURE0 + kFlareUnit);
    glGenTextures(1, &mFlareNeutral);
    glBindTexture(GL_TEXTURE_2D, mFlareNeutral);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 1, 1, 0, GL_RGBA, GL_FLOAT, neutral);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glActiveTexture(GL_TEXTURE0 + kFlareSpritesUnit);
    glGenTextures(1, &mFlareSpritesNeutral);
    glBindTexture(GL_TEXTURE_2D_ARRAY, mFlareSpritesNeutral);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 1, 1, 6, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 std::vector<uint8_t>(6 * 4, 0).data());
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glActiveTexture(GL_TEXTURE0 + kFlareCloudUnit);
    glGenTextures(1, &mFlareCloudNeutral);
    glBindTexture(GL_TEXTURE_2D, mFlareCloudNeutral);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 1, 1, 0, GL_RGBA, GL_FLOAT, neutral);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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
    glGenQueries(kQueryRing * 2, &mWorldQuery[0][0]);
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
    {
        // Milky Way panorama lane: a complete 2D dummy is bound before any program that declares uMilkyWay is built.
        GLint previousActive;
        glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActive);
        glActiveTexture(GL_TEXTURE0 + kMilkyWayUnit);
        glGenTextures(1, &mMilkyWayDummy);
        glBindTexture(GL_TEXTURE_2D, mMilkyWayDummy);
        const uint16_t zero[3] = {};
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 1, 1, 0, GL_RGB, GL_HALF_FLOAT, zero);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glActiveTexture(previousActive);
    }
    const char* starVs[] = { kVersion, kStarVertexShader };
    const char* starFs[] = { kVersion, kStarFragmentShader };
    mStars.id = BuildProgramWithVertex("night stars", starVs, 2, starFs, 2, mStars.uniforms);
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
    // Cascaded shadows are separate programs built from the same sources with CSM defined; the originals compile
    // from the unchanged text, so one cascade is bit-identical to the single shadow map.
    const char* kCsm = "#define CSM 1\n";
    const char* giGather[] = { kVersion, kDepthLib, kGiGeometryLib, kGiGatherShader };
    const char* giGatherCsm[] = { kVersion, kCsm, kDepthLib, kGiGeometryLib, kNewDawnCsmLib, kGiGatherShader };
    const char* giFilter[] = { kVersion, kDepthLib, kGiGeometryLib, kGiFilterShader };
    const char* giApply[] = { kVersion, kDepthLib, kGiGeometryLib, kGiApplyShader };
    // Optional GI failures leave the existing rendering path available. Reuse dedicated 2D lanes 8..11;
    // every sampler is assigned at link time and each lane starts with a complete transparent dummy.
    for (int unit = 0; unit < 4; ++unit) Bind(unit, mFogNeutral);
    mGiGather.id = BuildProgram("GI gather", giGather, 4, mGiGather.uniforms);
    mGiGatherCsm.id = BuildProgram("GI gather (cascades)", giGatherCsm, 6, mGiGatherCsm.uniforms);
    mGiFilter.id = BuildProgram("GI spatial temporal filter", giFilter, 4, mGiFilter.uniforms);
    mGiApply.id = BuildProgram("GI bilateral composite", giApply, 4, mGiApply.uniforms);
    {
        // Point-light shadow lanes: a complete one-texel cube (distance "far away") on both units, bound before the
        // local-light program that declares the samplers is built and put back after every pass that binds real cubes.
        GLint previousActivePs;
        glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActivePs);
        glGenTextures(1, &mPointShadowDummy);
        glBindTexture(GL_TEXTURE_CUBE_MAP, mPointShadowDummy);
        const float far = 1.0e6f;
        for (int face = 0; face < 6; ++face)
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_R32F, 1, 1, 0, GL_RED, GL_FLOAT, &far);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        for (int unit : { kPointShadowUnit0, kPointShadowUnit1 }) {
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_CUBE_MAP, mPointShadowDummy);
        }
        glActiveTexture(previousActivePs);
        const char* psVs[] = { kVersion, kPointShadowVertex };
        const char* psFs[] = { kVersion, kPointShadowFragment };
        mPointShadowDepth.id = BuildProgramWithVertex("point shadow depth", psVs, 2, psFs, 2, mPointShadowDepth.uniforms);
        glGenVertexArrays(1, &mPointShadowVao);
        glGenBuffers(1, &mPointShadowVbo);
        glBindVertexArray(mPointShadowVao);
        glBindBuffer(GL_ARRAY_BUFFER, mPointShadowVbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glBindVertexArray(0);
    }
    // Local lights reuse the colour (8) and depth (9) lanes bound to dummies above; the shadow cubes use their own.
    const char* localLight[] = { kVersion, kDepthLib, kGiGeometryLib, kLocalLightShader };
    mLocalLight.id = BuildProgram("local lights", localLight, 4, mLocalLight.uniforms);
    // Interior polish reuses the same colour (8) and depth (9) lanes.
    const char* interiorLight[] = { kVersion, kDepthLib, kGiGeometryLib, kInteriorLightShader };
    mInteriorLight.id = BuildProgram("interior window light", interiorLight, 4, mInteriorLight.uniforms);
    const char* emissive[] = { kVersion, kEmissiveShader };
    mEmissive.id = BuildProgram("emissive surfaces", emissive, 2, mEmissive.uniforms);
    {
        const char* vs[] = { kVersion, kInteriorGlassVertex };
        const char* fs[] = { kVersion, kInteriorGlassFragment };
        mInteriorGlass.id = BuildProgramWithVertex("interior window glass", vs, 2, fs, 2, mInteriorGlass.uniforms);
    }
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
    // Optional: a failure only leaves TAA unavailable. All four samplers sit on the dedicated postfx lanes 8..11
    // (see the unit table in BuildProgramWithVertex) and every lane holds a complete dummy until a pass binds real
    // data.
    const char* taaFill[] = { kVersion, R"GLSL(
out vec4 oColor;
void main() { oColor = vec4(1.0); }
)GLSL" };
    const char* taa[] = { kVersion, kTaaShader };
    for (int unit = 0; unit < 4; ++unit)
        Bind(unit, mFogNeutral);
    mTaa.id = BuildProgram("temporal anti-aliasing", taa, 2, mTaa.uniforms);
    mTaaMaskFill.id = BuildProgram("TAA actor mask", taaFill, 2, mTaaMaskFill.uniforms);
    const char* taaSharpen[] = { kVersion, kTaaSharpenShader };
    mTaaSharpen.id = BuildProgram("TAA sharpen", taaSharpen, 2, mTaaSharpen.uniforms);
    const char* comp[] = { kVersion, kCompositeShader };
    const char* compHdr[] = { kVersion, "#define HDR_SCENE\n", kCompositeShader };
    const char* compNoDither[] = { kVersion, "#define NO_DITHER\n", kCompositeShader };
    const char* shadowApply[] = { kVersion, kDepthLib, kSkyAmbientLightingLib, kShadowApplyShader };
    const char* shadowApplyCsm[] = { kVersion, kCsm, kDepthLib, kSkyAmbientLightingLib, kNewDawnCsmLib,
                                     kShadowApplyShader };
    const char* shadowApplyHdrCsm[] = { kVersion, kCsm, "#define HDR_SCENE\n", kDepthLib, kSkyAmbientLightingLib,
                                        kNewDawnCsmLib, kShadowApplyShader };
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
    const char* waterSurface[] = { kVersion, kDepthLib, kWaterPlaneLib, kRiverFlowLib, kWaterSurfaceShader };
    mWaterSurface.id = BuildProgram("water surface shader", waterSurface, 5, mWaterSurface.uniforms);
    {
        // Caustics/underwater: optional. Their texture array lives on its own unit (kCausticUnit) with a complete dummy
        // bound from creation on, so no program ever observes unit 0 for that sampler.
        GLint previousActiveCaustic;
        glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActiveCaustic);
        glActiveTexture(GL_TEXTURE0 + kCausticUnit);
        glGenTextures(1, &mCausticDummy);
        glBindTexture(GL_TEXTURE_2D_ARRAY, mCausticDummy);
        const uint8_t zero = 0;
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R8, 1, 1, 1, 0, GL_RED, GL_UNSIGNED_BYTE, &zero);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glActiveTexture(previousActiveCaustic);
        const char* kCsmDefine = "#define CSM 1\n";
        const char* caustics[] = { kVersion, kDepthLib, kWaterPlaneLib, kCausticLib, kCausticsShader };
        const char* causticsCsm[] = { kVersion, kCsmDefine, kDepthLib, kWaterPlaneLib, kNewDawnCsmLib, kCausticLib,
                                      kCausticsShader };
        const char* underwater[] = { kVersion, kDepthLib, kWaterPlaneLib, kUnderwaterShader };
        const char* kShaftDefine = "#define SHAFT_PASS 1\n";
        const char* underwaterShaft[] = { kVersion, kShaftDefine, kDepthLib, kWaterPlaneLib, kCausticLib,
                                          kUnderwaterShader };
        const char* kShaftCsmDefine = "#define SHAFT_PASS 1\n#define CSM 1\n";
        const char* underwaterShaftCsm[] = { kVersion, kShaftCsmDefine, kDepthLib, kWaterPlaneLib, kNewDawnCsmLib,
                                             kCausticLib, kUnderwaterShader };
        mCaustics.id = BuildProgram("water caustics", caustics, 5, mCaustics.uniforms);
        mCausticsCsm.id = BuildProgram("water caustics (cascades)", causticsCsm, 7, mCausticsCsm.uniforms);
        mUnderwater.id = BuildProgram("underwater", underwater, 4, mUnderwater.uniforms);
        mUnderwaterShaft.id = BuildProgram("underwater shafts", underwaterShaft, 6, mUnderwaterShaft.uniforms);
        mUnderwaterShaftCsm.id =
            BuildProgram("underwater shafts (cascades)", underwaterShaftCsm, 7, mUnderwaterShaftCsm.uniforms);
    }
    {
        // Water simulation (optional: a failure only leaves the flat-plane water). The three lanes get a complete
        // zero dummy before any program can observe them; passes bind the real targets while they run.
        GLint previousActiveWave;
        glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActiveWave);
        glGenTextures(1, &mWaveDummy);
        const float zero4[4] = {};
        for (int unit : { kWaveBufUnit, kWaveDepthUnit, kRippleUnit, kFlowBufUnit }) {
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_2D, mWaveDummy);
            if (unit == kWaveBufUnit) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 1, 1, 0, GL_RGBA, GL_FLOAT, zero4);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            }
        }
        // Rivers and waterfalls: the flow noise and spray sprite lanes get a complete 1x1 texture now; the asset loader
        // later replaces their contents in place (same texture names), so the lanes are never incomplete.
        {
            const uint8_t grey[4] = { 128, 128, 128, 0 };
            GLuint* names[3] = { &mFlowNoise, &mRiverSprite, &mFoamTex };
            const int lanes[3] = { kFlowNoiseUnit, kRiverSpriteUnit, kFoamTexUnit };
            for (int i = 0; i < 3; i++) {
                glGenTextures(1, names[i]);
                glActiveTexture(GL_TEXTURE0 + lanes[i]);
                glBindTexture(GL_TEXTURE_2D, *names[i]);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, grey);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            }
        }
        glActiveTexture(previousActiveWave);
        const char* waterDepthMap[] = { kVersion, kDepthLib, kWaterPlaneLib, kWaterDepthMapShader };
        const char* waterWaves[] = { kVersion, kDepthLib, kWaterPlaneLib, kWaterWaveLib, kWaterWaveShader };
        const char* waterRipple[] = { kVersion, kWaterRippleShader };
        mWaterDepthMap.id = BuildProgram("water depth map", waterDepthMap, 4, mWaterDepthMap.uniforms);
        mWaterWaves.id = BuildProgram("water waves", waterWaves, 5, mWaterWaves.uniforms);
        mWaterRipple.id = BuildProgram("water ripples", waterRipple, 2, mWaterRipple.uniforms);
    }
    if (mWaterSurface.id) {
        // Reuse the four dedicated postfx 2D lanes (8..11). Every sampler is distinct at creation, and each
        // lane has a complete dummy until a pass supplies its target. Never leave a new sampler on unit 0.
        GLint previousProgram, previousActive;
        glGetIntegerv(GL_CURRENT_PROGRAM, &previousProgram);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActive);
        glUseProgram(mWaterSurface.id);
        GLuint dummy;
        glGenTextures(1, &dummy);
        const uint8_t neutral[4] = { 255, 255, 255, 255 };
        const char* samplers[] = { "uScene", "uOpaqueColor", "uDepth", "uVisibleDepth" };
        for (int i = 0; i < 4; ++i) {
            glActiveTexture(GL_TEXTURE0 + kFirstUnit + i);
            glBindTexture(GL_TEXTURE_2D, dummy);
            if (i == 0) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, neutral);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            }
            glUniform1i(glGetUniformLocation(mWaterSurface.id, samplers[i]), kFirstUnit + i);
        }
        glUseProgram(previousProgram);
        glActiveTexture(previousActive);
    }
    const char* flareVisibility[] = { kVersion, kDepthLib, kFlareVisibilityShader };
    const char* flareApply[] = { kVersion, kFlareApplyShader };
    mFlareVisibility.id = BuildProgram("sun disc visibility", flareVisibility, 3, mFlareVisibility.uniforms);
    mFlareApply.id = BuildProgram("textured sun flare", flareApply, 2, mFlareApply.uniforms);
    const char* rayMask[] = { kVersion, kDepthLib, kRaySkyMaskLib, kRayMaskShader };
    const char* rayRadial[] = { kVersion, kRayRadialShader };
    const char* rayApply[] = { kVersion, kDepthLib, kRaySkyMaskLib, kRayApplyShader };
    mRayMask.id = BuildProgram("sun ray sky mask", rayMask, 4, mRayMask.uniforms);
    mRayRadial.id = BuildProgram("sun ray radial integration", rayRadial, 2, mRayRadial.uniforms);
    mRayApply.id = BuildProgram("sun ray HDR apply", rayApply, 4, mRayApply.uniforms);
    const char* fogMarch[] = { kVersion, kDepthLib, kSkyAmbientLightingLib, kFogMarchShader };
    const char* fogMarchCsm[] = { kVersion, kCsm, kDepthLib, kSkyAmbientLightingLib, kNewDawnCsmLib, kFogMarchShader };
    const char* fogApply[] = { kVersion, kDepthLib, kFogApplyShader };
    // Optional: fog shader failure must not disable the existing renderer.
    mFogMarch.id = BuildProgram("volumetric fog march", fogMarch, 4, mFogMarch.uniforms);
    mFogMarchCsm.id = BuildProgram("volumetric fog march (cascades)", fogMarchCsm, 6, mFogMarchCsm.uniforms);
    mFogApply.id = BuildProgram("volumetric fog apply", fogApply, 3, mFogApply.uniforms);
    const char* shaftMarch[] = { kVersion, kDepthLib, kShaftMarchShader };
    const char* shaftMarchCsm[] = { kVersion, kCsm, kDepthLib, kNewDawnCsmLib, kShaftMarchShader };
    const char* shaftBlur[] = { kVersion, kShaftBlurShader };
    const char* shaftApply[] = { kVersion, kDepthLib, kShaftApplyShader };
    mShaftMarch.id = BuildProgram("sun shaft march shader", shaftMarch, 3, mShaftMarch.uniforms);
    mShaftMarchCsm.id = BuildProgram("sun shaft march (cascades)", shaftMarchCsm, 5, mShaftMarchCsm.uniforms);
    mShaftBlur.id = BuildProgram("sun shaft blur shader", shaftBlur, 2, mShaftBlur.uniforms);
    mShaftApply.id = BuildProgram("sun shaft apply shader", shaftApply, 3, mShaftApply.uniforms);
    mAo.id = BuildProgram("AO shader", ao, 3, mAo.uniforms);
    mBlur.id = BuildProgram("AO blur shader", blur, 4, mBlur.uniforms);
    mApply.id = BuildProgram("AO apply shader", apply, 4, mApply.uniforms);
    mDownHdr.id = BuildProgram("HDR bloom downsample", downHdr, 3, mDownHdr.uniforms);
    mCompositeHdr.id = BuildProgram("HDR bloom/display/grade", compHdr, 3, mCompositeHdr.uniforms);
    mCompositeNoDither.id = BuildProgram("LDR composite without dither", compNoDither, 3, mCompositeNoDither.uniforms);
    mShadowApplyHdr.id = BuildProgram("HDR shadow apply", shadowApplyHdr, 5, mShadowApplyHdr.uniforms);
    mShadowApplyHdrCsm.id = BuildProgram("HDR shadow apply (cascades)", shadowApplyHdrCsm, 7, mShadowApplyHdrCsm.uniforms);
    mDown.id = BuildProgram("bloom downsample shader", down, 2, mDown.uniforms);
    mUp.id = BuildProgram("bloom upsample shader", up, 2, mUp.uniforms);
    mComposite.id = BuildProgram("composite shader", comp, 2, mComposite.uniforms);
    mShadowApply.id = BuildProgram("shadow apply shader", shadowApply, 4, mShadowApply.uniforms);
    mShadowApplyCsm.id = BuildProgram("shadow apply (cascades)", shadowApplyCsm, 6, mShadowApplyCsm.uniforms);
    mShadowDepth.id = BuildProgramWithVertex("shadow depth shader", shadowDepthVs, 2, shadowDepthFs, 2,
                                             mShadowDepth.uniforms);
    mShadowFoliage.id = BuildProgramWithVertex("shadow foliage shader", foliageVs, 2, foliageFs, 2,
                                               mShadowFoliage.uniforms);
    return mShadowFoliage.id && mDownHdr.id && mCompositeHdr.id && mCompositeNoDither.id && mShadowApplyHdr.id &&
           mAo.id && mBlur.id && mApply.id && mDown.id && mUp.id && mComposite.id && mShadowApply.id &&
           mShadowDepth.id && mShaftMarch.id && mShaftBlur.id && mShaftApply.id && mWater.id && mWaterReflection.id &&
           mWaterReflectionApply.id && mWaterSurface.id;
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

static const char* const kStageNames[NEWDAWN_PERF_STAGES] = {
    "opaque end / AO", "scene end / bloom+grade", "shadows", "sun shafts", "water", "water opaque copy",
    "physical sky", "volumetric clouds", "sky ambient cube", "volumetric fog", "textured sun flare",
    "ambient particles", "wet surfaces and rain shelter", "rain and snow", "screen-space GI", "VFX sprites",
    "local lights", "water caustics", "underwater", "grass", "night stars", "TAA", "point-light shadows",
    "interior window light", "emissive surfaces", "world GI" };
static float sStageMs[NEWDAWN_PERF_STAGES] = {};
float gNewDawnGpuFrameMs = 0.0f; // written by GfxRenderingAPIOGL

const char* NewDawnPerfStageName(int stage) {
    return stage >= 0 && stage < NEWDAWN_PERF_STAGES ? kStageNames[stage] : "";
}

void NewDawnPerfGet(NewDawnPerfInfo* out) {
    std::copy(sStageMs, sStageMs + NEWDAWN_PERF_STAGES, out->stageMs);
    out->gpuFrameMs = gNewDawnGpuFrameMs;
    out->vramTotalMB = out->vramFreeMB = -1;
#ifndef USE_OPENGLES
    static int mode = -1; // 0 none, 1 NVX, 2 ATI
    if (mode < 0) {
        mode = 0;
        GLint n = 0;
        glGetIntegerv(GL_NUM_EXTENSIONS, &n);
        for (GLint i = 0; i < n; i++) {
            const char* e = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i));
            if (e && !strcmp(e, "GL_NVX_gpu_memory_info")) {
                mode = 1;
            } else if (e && !strcmp(e, "GL_ATI_meminfo") && mode == 0) {
                mode = 2;
            }
        }
    }
    GLint v[4] = {};
    if (mode == 1) {
        glGetIntegerv(0x9048, &v[0]); // GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX (KB)
        glGetIntegerv(0x9049, &v[1]); // GPU_MEM_INFO_CURRENT_AVAILABLE_VIDMEM_NVX (KB)
        out->vramTotalMB = v[0] / 1024;
        out->vramFreeMB = v[1] / 1024;
    } else if (mode == 2) {
        glGetIntegerv(0x87FB, v); // VBO_FREE_MEMORY_ATI: [0] = total free KB
        out->vramFreeMB = v[0] / 1024;
    }
#endif
}

void PostFxGL::BeginTimer(int stage) {
    static_assert(kQueryStages == NEWDAWN_PERF_STAGES, "add new timer stages to kStageNames and NEWDAWN_PERF_STAGES");
    if (!mTimers) {
        return;
    }
    int slot = mQueryNext[stage];
    if (mQueryPending[stage][slot]) {
        GLuint64 ns = 0;
        glGetQueryObjectui64v(mQueries[stage][slot], GL_QUERY_RESULT, &ns);
        mQueryPending[stage][slot] = false;
        sStageMs[stage] += (static_cast<float>(ns / 1e6) - sStageMs[stage]) * 0.1f;
        if (mStats) {
            mQueryTotalMs[stage] += ns / 1e6;
            if (++mQueryCount[stage] == 300) {
                SPDLOG_INFO("NewDawnPostFx: stage {} ({}) averages {:.3f} ms GPU over {} runs", stage,
                            kStageNames[stage], mQueryTotalMs[stage] / 300.0, 300);
                mQueryTotalMs[stage] = 0.0;
                mQueryCount[stage] = 0;
            }
        }
    }
    glBeginQuery(GL_TIME_ELAPSED, mQueries[stage][slot]);
}

void PostFxGL::EndTimer(int stage) {
    if (!mTimers) {
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
    mWeatherFrameReady = false;
    // Eligibility must be re-established by this frame's physical-sky marker. Never reuse a previous
    // outdoor frame's illumination in interiors, scripted/final-day skies, pause backgrounds or HDR fallbacks.
    mSkyAmbientReady = false;
    mSkyReflectionsReady = false;
    mFlareCloudOpacity = 0.0f;
    mRaySkyReady = false;
    mLocalMarking = false;
    mTaaCam.valid = false; // the interpreter sets it again, after this call, when TAA is on for the frame
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
    if (CvInt("gNewDawn.Lighting.Local", 0) || CvInt("gNewDawn.TAA.Enabled", 0) || CvInt("gNewDawn.Interior.Emissive", 0)) {
        // Fresh actor mask for this frame (TAA reads it too, and the grass pass marks its tufts in it); the game never
        // uses stencil.
        glStencilMask(0xFF);
        glClearStencil(0);
        glClear(GL_STENCIL_BUFFER_BIT);
        mLocalMarking = true;
    }
    mHdrActive = true;
    mStats = CvInt("gNewDawn.PostFx.Debug.Stats", 0) != 0;
    if (mStats) {
        const int slot = mWorldNext;
        if (mWorldPending[slot]) {
            GLuint64 t0 = 0, t1 = 0;
            glGetQueryObjectui64v(mWorldQuery[slot][0], GL_QUERY_RESULT, &t0);
            glGetQueryObjectui64v(mWorldQuery[slot][1], GL_QUERY_RESULT, &t1);
            mWorldPending[slot] = false;
            mWorldTotalMs += (t1 - t0) / 1e6;
            if (++mWorldCount == 300) {
                SPDLOG_INFO("NewDawnPostFx: world total (HDR open to presentation) averages {:.3f} ms GPU over {} runs",
                            mWorldTotalMs / 300.0, 300);
                mWorldTotalMs = 0.0;
                mWorldCount = 0;
            }
        }
        glQueryCounter(mWorldQuery[slot][0], GL_TIMESTAMP);
    }
    return true;
}

// Parameter sets of the gNewDawn.Sky.Special variants (skyVariant). Pure function of read-only game state; the
// normal and winter rows reproduce the previous behaviour exactly. Colours are linear.
struct SkyLook {
    float overcast, coverage, hazeGrey, glowAmt, glowFall, layerBase, layerThick, windMul, shear, turb;
    float haze[3], deck[3], glow[3];
};

static void ComputeSkyLook(const NewDawnPostFxParams& params, const float fog[3], SkyLook& l) {
    const float luma = fog[0] * 0.2126f + fog[1] * 0.7152f + fog[2] * 0.0722f;
    const float m = std::clamp(params.skyMood, 0.0f, 1.0f);
    l = {};
    l.overcast = params.skyOvercast;
    l.coverage = std::lerp(std::clamp(CvFloat("gNewDawn.Sky.Coverage", 0.55f), 0.0f, 1.0f), 0.8f, params.skyOvercast);
    l.hazeGrey = 0.65f;
    l.layerBase = 1.6f;
    l.layerThick = 1.8f;
    l.windMul = 1.0f;
    l.glowFall = 3.5f;
    for (int i = 0; i < 3; ++i)
        l.haze[i] = l.deck[i] = fog[i];
    auto tinted = [&](const float t[3], float greyMix, float out[3]) {
        for (int i = 0; i < 3; ++i)
            out[i] = (fog[i] + (luma - fog[i]) * greyMix) * t[i];
    };
    switch (params.skyVariant) {
        case 2: { // final day: low, fast storm deck lit from below; the native sickly yellow-green turns red-orange
            l.overcast = 0.55f + 0.40f * m;
            l.coverage = 0.74f + 0.10f * m;
            l.hazeGrey = 0.0f;
            l.layerBase = 0.9f;
            l.layerThick = 1.1f;
            l.windMul = 2.0f + 3.0f * m;
            l.shear = 0.6f;
            l.turb = 0.35f + 0.5f * m;
            l.glowFall = 2.2f;
            const float dark = 1.0f - 0.45f * m;
            const float red = std::clamp((m - 0.45f) / 0.4f, 0.0f, 1.0f);
            const float peak = std::max({ fog[0], fog[1], fog[2], 0.001f });
            const float apoc[3] = { 1.0f, 0.26f, 0.05f };
            const float rs = red * red * (3.0f - 2.0f * red);
            for (int i = 0; i < 3; ++i) {
                const float native = std::pow(fog[i] / peak, 1.6f); // the game's own final-day hue, kept saturated
                const float hue = native + (apoc[i] - native) * rs;
                l.glow[i] = hue;
                l.haze[i] = hue * 0.085f * dark;
                l.deck[i] = hue * 0.030f * dark;
            }
            l.glowAmt = 0.75f + 0.55f * m;
            break;
        }
        case 3: { // cursed Ikana: desaturated violet-grey, thin, nearly still
            const float hz[3] = { 1.10f, 0.98f, 1.32f }, dk[3] = { 0.76f, 0.66f, 0.92f };
            l.overcast = 1.0f;
            l.coverage = 0.72f;
            l.hazeGrey = 0.8f;
            l.layerBase = 1.5f;
            l.layerThick = 1.1f;
            l.windMul = 0.12f;
            tinted(hz, 0.55f, l.haze);
            tinted(dk, 0.55f, l.deck);
            for (int i = 0; i < 3; ++i) { // a faint violet sky even when the night fog is nearly black
                l.haze[i] = std::max(l.haze[i], hz[i] * 0.060f);
                l.deck[i] = std::max(l.deck[i], dk[i] * 0.035f);
            }
            break;
        }
        case 4: { // poisoned swamp: sickly green haze under a purple deck, matching the poison water palette
            const float hz[3] = { 0.62f, 0.95f, 0.30f }, dk[3] = { 0.78f, 0.50f, 0.98f };
            l.overcast = 0.95f;
            l.coverage = 0.80f;
            l.hazeGrey = 0.15f;
            l.layerBase = 1.3f;
            l.layerThick = 1.2f;
            l.windMul = 0.5f;
            l.glowFall = 2.5f;
            tinted(hz, 0.5f, l.haze);
            tinted(dk, 0.5f, l.deck);
            for (int i = 0; i < 3; ++i) { // keep a toxic minimum brightness even when the game's fog is dark
                l.haze[i] = std::max(l.haze[i], hz[i] * 0.10f);
                l.deck[i] = std::max(l.deck[i], dk[i] * 0.07f);
            }
            l.glow[0] = 0.35f; l.glow[1] = 0.70f; l.glow[2] = 0.10f;
            l.glowAmt = 0.22f;
            break;
        }
        default:
            break;
    }
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
    SkyLook look;
    ComputeSkyLook(params, fog, look);
    glUniform3fv(Loc("uHaze"), 1, look.haze);
    glUniform1f(Loc("uHazeGrey"), look.hazeGrey);
    glUniform3fv(Loc("uGlow"), 1, look.glow);
    glUniform1f(Loc("uGlowAmt"), look.glowAmt);
    glUniform1f(Loc("uGlowFall"), look.glowFall);
    glUniform3fv(Loc("uDeck"), 1, look.deck);
    glUniform2f(Loc("uLayer"), look.layerBase, look.layerThick);
    glUniform1f(Loc("uShear"), look.shear);
    glUniform1f(Loc("uTurb"), look.turb);
    float palette[3];
    for (int i = 0; i < 3; ++i) {
        palette[i] = DecodeColor(params.skyPaletteColor[i]) * 0.8f + DecodeColor(params.skyFogColor[i]) * 0.2f;
    }
    glUniform3fv(Loc("uPaletteColor"), 1, palette);
    glUniform1f(Loc("uWeather"), params.skyWeather);
    glUniform1f(Loc("uOvercast"), look.overcast);
    glUniform1f(Loc("uIntensity"), std::clamp(CvFloat("gNewDawn.HDR.SkyIntensity", 1.0f), 0.0f, 4.0f));
    glUniform1f(Loc("uCoverage"), look.coverage);
    SetNightUniforms(params);
    // 18/9 m/s drift, gusting with the shared wind when enabled, independent of the game clock, pause,
    // frame rate and time-speed enhancements. 1024 km is a common period of all noise octaves/maps.
    static const auto epoch = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
    // Density samples world + offset, so negative travel advects clouds along the positive foliage/particle wind.
    const double drift = CvInt("gNewDawn.Atmos.Wind", 0) ? -NewDawnWindTravel(NewDawnWindSeconds()) : seconds;
    glUniform2f(Loc("uWind"), static_cast<float>(std::fmod(drift * 0.018 * look.windMul, 1024.0)),
                static_cast<float>(std::fmod(drift * 0.009 * look.windMul, 1024.0)));
}

// Night sky assets (committed under mm/assets/newdawn/nightsky, CC BY-SA): `<gNewDawn.Assets.Path>/nightsky/{stars.bin,milkyway_eq.jpg,milkyway_eq.txt}`, baked by
// tools/newdawn/nightsky/prep_nightsky.py from the HYG v4.1 catalogue and NASA SVS Deep Star Maps 2020.
bool PostFxGL::EnsureNightSkyAssets() {
    const std::string root = CvString("gNewDawn.Assets.Path", "assets/newdawn");
    if (root == mNightAssetPath)
        return mMilkyWayTex != 0 && mStarCount > 0;
    mNightAssetPath = root;
    if (mMilkyWayTex) {
        glDeleteTextures(1, &mMilkyWayTex);
        mMilkyWayTex = 0;
    }
    mStarCount = 0;
    const auto dir = std::filesystem::path(root) / "nightsky";
    GLint previousActive;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActive);
    {
        // 8-bit JPEG, pixel = 255 * (linear / scale)^(1/gamma); the sidecar carries gamma and scale
        // (tools/newdawn/nightsky/prep_nightsky.py). Decoded to half-float strip by strip to keep the temporary small.
        float gamma = 3.0f, scale = 0.0f;
        if (std::ifstream meta(dir / "milkyway_eq.txt"); meta) {
            std::string line;
            std::getline(meta, line);
            std::sscanf(line.c_str(), "gamma=%f scale=%f", &gamma, &scale);
        }
        int w = 0, h = 0, n = 0;
        auto* pixels = scale > 0.0f && gamma > 0.0f ? stbi_load((dir / "milkyway_eq.jpg").string().c_str(), &w, &h, &n, 3)
                                                    : nullptr;
        if (pixels && w > 0 && h > 0 && w <= 8192 && h <= 4096) {
            float lut[256];
            for (int i = 0; i < 256; ++i)
                lut[i] = std::pow(i / 255.0f, gamma);
            glActiveTexture(GL_TEXTURE0 + kMilkyWayUnit);
            glGenTextures(1, &mMilkyWayTex);
            glBindTexture(GL_TEXTURE_2D, mMilkyWayTex);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_FLOAT, nullptr);
            constexpr int strip = 128;
            std::vector<float> linear((size_t)w * strip * 3);
            for (int y = 0; y < h; y += strip) {
                const int rows = std::min(strip, h - y);
                const uint8_t* src = pixels + (size_t)y * w * 3;
                for (size_t i = 0; i < (size_t)w * rows * 3; ++i)
                    linear[i] = lut[src[i]];
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, y, w, rows, GL_RGB, GL_FLOAT, linear.data());
            }
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glGenerateMipmap(GL_TEXTURE_2D);
            mMilkyWayScale = scale;
        }
        stbi_image_free(pixels);
    }
    {
        std::ifstream f(dir / "stars.bin", std::ios::binary);
        char magic[4];
        uint32_t count = 0;
        if (f.read(magic, 4) && std::memcmp(magic, "NDST", 4) == 0 && f.read((char*)&count, 4) && count > 0 &&
            count < 200000) {
            std::vector<float> data((size_t)count * 7);
            if (f.read((char*)data.data(), data.size() * 4)) {
                if (!mStarVao) {
                    glGenVertexArrays(1, &mStarVao);
                    glGenBuffers(1, &mStarVbo);
                }
                glBindVertexArray(mStarVao);
                glBindBuffer(GL_ARRAY_BUFFER, mStarVbo);
                glBufferData(GL_ARRAY_BUFFER, data.size() * 4, data.data(), GL_STATIC_DRAW);
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 28, (void*)0);
                glEnableVertexAttribArray(1);
                glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 28, (void*)12);
                glEnableVertexAttribArray(2);
                glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 28, (void*)16);
                mStarCount = (int)count;
            }
        }
    }
    glActiveTexture(previousActive);
    if (mMilkyWayTex && mStarCount > 0) {
        SPDLOG_INFO("NewDawnNightSky: loaded {} stars and the Milky Way panorama from {}", mStarCount, dir.string());
    } else {
        SPDLOG_WARN("NewDawnNightSky: missing/unsupported assets in {} (stars.bin, milkyway_eq.jpg/.txt); the native stars "
                    "stay. Set gNewDawn.Assets.Path to the asset root.",
                    dir.string());
    }
    return mMilkyWayTex != 0 && mStarCount > 0;
}

// Night visibility of stars and the Milky Way: astronomical twilight (sun about 17 degrees down), the variant's own
// multiplier (dimmed under the final-day storm, 0 in cursed Ikana and the poisoned swamp), rain and overcast.
static float NightVisibility(const NewDawnPostFxParams& params) {
    if (!CvInt("gNewDawn.Sky.Night", 0) || params.skyStarVis <= 0.0f)
        return 0.0f;
    const float t = std::clamp((-params.skySunDir[1] - 0.03f) / 0.27f, 0.0f, 1.0f);
    const float dark = t * t * (3.0f - 2.0f * t);
    const float weather = 1.0f - 0.9f * std::clamp(params.skyWeather, 0.0f, 1.0f);
    const float overcast = 1.0f - 0.8f * std::clamp(params.skyOvercast, 0.0f, 1.0f);
    return dark * params.skyStarVis * weather * overcast * std::clamp(CvFloat("gNewDawn.Sky.Strength", 1.0f), 0.0f, 1.0f);
}

// World -> J2000 equatorial. The sun moves in a plane through the origin (the game's sunPos is
// (-sin a, cos a, cos a / 6)), so the whole sky turns about that plane's normal by the same hour angle: the north
// celestial pole is the plane normal (about 9.5 degrees above the northern -Z horizon). The sun's J2000 right
// ascension is fixed at 6h (June), so galactic centre culminates in the south near midnight.
static void CelestialMatrix(float clock, float out[9]) {
    const double pi = 3.14159265358979323846;
    const double a = (clock - 0.5) * 2.0 * pi;
    double zc[3] = { 0.0, 0.16440, -0.98639 };
    double b[3] = { 0.0, 0.98639, 0.16440 }; // the sun at noon
    // t = zc x b
    double t[3] = { zc[1] * b[2] - zc[2] * b[1], zc[2] * b[0] - zc[0] * b[2], zc[0] * b[1] - zc[1] * b[0] };
    const double alpha = pi * 0.5;
    const double ca = std::cos(alpha), sa = std::sin(alpha);
    // M = [b t zc] * [s t0 z]^T, with s = (ca, sa, 0), t0 = (-sa, ca, 0), z = (0, 0, 1)
    double M[3][3];
    for (int i = 0; i < 3; ++i) {
        M[i][0] = b[i] * ca + t[i] * (-sa);
        M[i][1] = b[i] * sa + t[i] * ca;
        M[i][2] = zc[i];
    }
    // R_Q(-a): rotation by -a about Q = -zc, i.e. by +a about zc. Rodrigues with axis zc, angle a.
    const double c = std::cos(a), s = std::sin(a);
    double R[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R[i][j] = c * (i == j) + (1.0 - c) * zc[i] * zc[j];
    R[0][1] += -s * zc[2]; R[0][2] += s * zc[1];
    R[1][0] += s * zc[2];  R[1][2] += -s * zc[0];
    R[2][0] += -s * zc[1]; R[2][1] += s * zc[0];
    // eq = M^T * R * world; store row-major, uploaded transposed.
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double v = 0.0;
            for (int k = 0; k < 3; ++k)
                v += M[k][i] * R[k][j];
            out[i * 3 + j] = (float)v;
        }
    }
}

void PostFxGL::SetNightUniforms(const NewDawnPostFxParams& params) {
    float vis = NightVisibility(params);
    float moon = 0.0f;
    if (CvInt("gNewDawn.Sky.Night", 0)) {
        const float t = std::clamp((-params.skySunDir[1] - 0.02f) / 0.2f, 0.0f, 1.0f);
        moon = t * t * (3.0f - 2.0f * t) * params.skyMoonGlow * std::clamp(CvFloat("gNewDawn.Sky.MoonGlow", 1.0f), 0.0f, 4.0f);
    }
    if ((vis <= 0.0f && moon <= 0.0f) || !EnsureNightSkyAssets()) {
        vis = 0.0f;
        moon = 0.0f;
    }
    float m[9];
    CelestialMatrix(params.skyClock, m);
    glUniformMatrix3fv(Loc("uCelestial"), 1, GL_TRUE, m);
    glUniform1f(Loc("uMilkyScale"), mMilkyWayScale);
    glUniform1f(Loc("uMilkyGain"), vis * std::clamp(CvFloat("gNewDawn.Sky.MilkyWay", 1.0f), 0.0f, 4.0f) * 0.75f);
    glUniform3fv(Loc("uMoonDir"), 1, params.skyMoonDir);
    glUniform1f(Loc("uMoonRadius"), std::max(params.skyMoonRadius, 0.004f));
    glUniform1f(Loc("uMoonGlowAmt"), moon);
    glUniform1i(Loc("uMilkyWay"), kMilkyWayUnit);
    if (mMilkyWayTex) {
        // Keep the panorama (or the dummy when assets are missing) on its lane for every program that declares it.
        GLint previousActive;
        glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActive);
        glActiveTexture(GL_TEXTURE0 + kMilkyWayUnit);
        glBindTexture(GL_TEXTURE_2D, mMilkyWayTex);
        glActiveTexture(previousActive);
    }
}

void PostFxGL::RunNightStars(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    const float vis = NightVisibility(params);
    if (vis <= 0.0f || !mStars.id || !EnsureNightSkyAssets())
        return;
    GlStateGuard guard;
    guard.Neutralise();
    BeginTimer(kNightTimerStage);
    glBindVertexArray(mStarVao);
    Use(mStars);
    glUniformMatrix4fv(Loc("uViewProj"), 1, GL_FALSE, params.shadowViewProj);
    glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), params.shadowFlipY);
    float m[9];
    CelestialMatrix(params.skyClock, m);
    glUniformMatrix3fv(Loc("uCelestial"), 1, GL_TRUE, m);
    // HDR units: the display exposure is about 0.05, so a mag 6 star peaks near 0.5, Vega near 50.
    glUniform1f(Loc("uFlux"), vis * std::clamp(CvFloat("gNewDawn.Sky.StarBrightness", 1.0f), 0.0f, 8.0f) * 300.0f);
    glUniform1f(Loc("uPix"), std::max((float)fb.height, 1.0f) / 1080.0f);
    static const auto epoch = std::chrono::steady_clock::now();
    glUniform1f(Loc("uTime"), (float)std::fmod(std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count(), 3600.0));
    const bool clouds = mCloudsDrawnFrame == mSkyFrame && mClouds.tex != 0;
    glUniform1f(Loc("uCloudsOn"), clouds ? mFlareCloudOpacity : 0.0f);
    glUniform2f(Loc("uRes"), (float)fb.width, (float)fb.height);
    Bind(0, clouds ? mClouds.tex : mFogNeutral);
    glUniform1i(Loc("uClouds"), kFirstUnit);
    glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
    glViewport(0, 0, fb.width, fb.height);
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ZERO, GL_ONE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, mStarCount);
    glDisable(GL_PROGRAM_POINT_SIZE);
    EndTimer(kNightTimerStage);
    CheckError("night stars");
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
    mTimers = mStats || CvInt("gNewDawn.Perf.Overlay", 0) != 0;
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
    RunNightStars(params, fb);
    mRaySkyReady = CvFloat("gNewDawn.Sky.Strength", 1.0f) >= 0.999f;
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
    mFlareCloudOpacity = strength;
    mCloudsDrawnFrame = mSkyFrame;
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
                         params.skyVariant != mSkyCachedVariant || std::abs(params.skyMood - mSkyCachedMood) > 0.01f ||
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
    mSkyCachedVariant = params.skyVariant;
    mSkyCachedMood = params.skyMood;
    std::copy(params.skySunDir, params.skySunDir + 3, mSkyCachedSun);
    std::copy(settings, settings + 7, mSkyCachedSettings);
    EndTimer(kSkyAmbientTimerStage);
    CheckError("sky ambient cube");
}

GLuint PostFxGL::HdrSnapshot(const FramebufferOGL& fb, bool final) {
    GlStateGuard guard;
    guard.Neutralise();
    glBindVertexArray(mVao);
    mStats = CvInt("gNewDawn.PostFx.Debug.Stats", 0) != 0;
    mTimers = mStats || CvInt("gNewDawn.Perf.Overlay", 0) != 0;
    Settings s = ReadSettings();
    if (!s.postFxOn)
        s.bloomOn = s.gradeOn = false;
    BeginTimer(1);
    RunSceneEnd(fb, s, final); // resolves RGBA16F before bloom, then tonemaps once into display colour
    EndTimer(1);
    CheckError("HDR display transform");
    return mHdrDisplay.fbo;
}

void PostFxGL::EndHdr(const FramebufferOGL& fb) {
    HdrSnapshot(fb, true);
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
    if (mStats) {
        // Whole 3D world on the GPU: HDR world open to the presentation copy (timestamps, so the per-pass timers nest).
        glQueryCounter(mWorldQuery[mWorldNext][1], GL_TIMESTAMP);
        mWorldPending[mWorldNext] = true;
        mWorldNext = (mWorldNext + 1) % kQueryRing;
    }
    // Debug-only measurement fixture: while the file DumpDir/go exists, dump up to DumpFrames consecutive final world
    // frames as raw RGB8, bottom row first, to DumpDir/fNNNN_WxH.rgb. Nothing happens unless the cvar is set.
    if (const int dumpCount = CvInt("gNewDawn.TAA.DumpFrames", 0); dumpCount > 0) {
        const char* dir = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetString("gNewDawn.TAA.DumpDir", "");
        std::error_code ec;
        if (dir != nullptr && *dir != '\0' && mDumpDone < dumpCount && mHdrDisplay.tex != 0 &&
            std::filesystem::exists(std::string(dir) + "/go", ec)) {
            std::vector<uint8_t> pixels((size_t)mHdrDisplay.w * mHdrDisplay.h * 3);
            GLint packAlign = 4;
            glGetIntegerv(GL_PACK_ALIGNMENT, &packAlign);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            Bind(0, mHdrDisplay.tex);
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
            glPixelStorei(GL_PACK_ALIGNMENT, packAlign);
            char name[512];
            snprintf(name, sizeof(name), "%s/f%04d_%dx%d.rgb", dir, mDumpDone, mHdrDisplay.w, mHdrDisplay.h);
            if (FILE* f = fopen(name, "wb")) {
                fwrite(pixels.data(), 1, pixels.size(), f);
                fclose(f);
            }
            if (CvInt("gNewDawn.Shadows.CoverageDebug", 0) >= 2) {
                SPDLOG_INFO("NewDawn dump f{:04d}", mDumpDone); // pairs the frame with the preceding caster line
            }
            mDumpDone++;
        }
    } else {
        mDumpDone = 0;
    }
    mHdrActive = false;
}

// True when at least one water box's surface (its rectangle plus a 300 unit margin for waves and shore foam, and from
// just below the plane to 300 above it for wave crests) touches the camera's view pyramid. A bed, caustics or a
// refracted shore are only ever seen through the surface's own screen footprint, so the surface is what to test.
// The scene's nearest 16 boxes are always submitted, so a field with a pond far behind the camera used to run the water
// passes and the opaque colour/depth copy for nothing. Only the four side planes are tested (conservative).
static bool WaterBoxInView(const NewDawnPostFxParams& params) {
    if (!params.shadowUseExactProj || params.waterUnderwaterState > 0.0f) { // submerged: the bed is seen from below the surface
        return params.waterBoxCount > 0;
    }
    const float* M = params.shadowViewProj;
    const float invScale = params.shadowNdcXScale > 1e-6f ? 1.0f / params.shadowNdcXScale : 1.0f;
    float planes[4][4];
    // clip = M * (x, y, z, 1) with M[4 * column + row]; row r of the matrix is (M[r], M[4 + r], M[8 + r], M[12 + r]).
    auto row = [&](int r, float s, float* out) {
        for (int c = 0; c < 4; c++) {
            out[c] = M[4 * c + r] * s;
        }
    };
    float r0[4], r1[4], r3[4];
    row(0, invScale, r0);
    row(1, 1.0f, r1);
    row(3, 1.0f, r3);
    for (int c = 0; c < 4; c++) {
        planes[0][c] = r3[c] + r0[c];
        planes[1][c] = r3[c] - r0[c];
        planes[2][c] = r3[c] + r1[c];
        planes[3][c] = r3[c] - r1[c];
    }
    const int count = std::clamp(params.waterBoxCount, 0, NEWDAWN_MAX_WATER_BOXES);
    for (int i = 0; i < count; i++) {
        const float* b = params.waterBoxes[i];
        const float lo[3] = { b[0] - 300.0f, b[4] - 60.0f, b[1] - 300.0f };
        const float hi[3] = { b[2] + 300.0f, b[4] + 300.0f, b[3] + 300.0f };
        bool outside = false;
        for (const auto& p : planes) {
            const float x = p[0] >= 0.0f ? hi[0] : lo[0];
            const float y = p[1] >= 0.0f ? hi[1] : lo[1];
            const float z = p[2] >= 0.0f ? hi[2] : lo[2];
            if (p[0] * x + p[1] * y + p[2] * z + p[3] < 0.0f) {
                outside = true;
                break;
            }
        }
        if (!outside) {
            return true;
        }
    }
    return false;
}

void PostFxGL::Run(int stage, const NewDawnPostFxParams& params, const FramebufferOGL& fb, uint32_t frameIndex,
                   const float* shadowPositions, uint32_t shadowVertexCount) {
    if (!mUsable || fb.width < 8 || fb.height < 8) {
        return;
    }
    if (stage == NEWDAWN_POSTFX_STAGE_SCENE_END) {
        mArt.on = params.artEnabled != 0;
        std::copy(params.artSlope, params.artSlope + 3, mArt.slope);
        mArt.offset = params.artOffset;
        mArt.power = params.artPower;
        mArt.contrast = params.artContrast;
        mArt.sat = params.artSat;
        std::copy(params.artShadowTint, params.artShadowTint + 3, mArt.shadow);
        std::copy(params.artHighTint, params.artHighTint + 3, mArt.high);
        std::copy(params.artGreen, params.artGreen + 2, mArt.green);
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
    const bool doGi = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && mHdrActive && s.postFxOn &&
                      params.giEnabled && params.giStrength > 0.0f && params.shadowUseExactProj &&
                      mGiGather.id && mGiFilter.id && mGiApply.id && s.debugView == 0 && s.shadowDebug == 0;
    if (stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && !doGi) mGiValid = false;
    // No water box in view: skip the water passes and the opaque copy (gNewDawn.Perf.WaterCull=0 turns this off).
    const bool waterInView = params.waterBoxCount > 0 && (CvInt("gNewDawn.Perf.WaterCull", 1) == 0 || WaterBoxInView(params));
    const bool captureWater = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END &&
                              (params.waterReflections != 0 || params.waterFoam > 0.0f ||
                               params.waterRefraction > 0.0f || params.waterRestyle > 0.0f) &&
                              waterInView && params.shadowUseExactProj != 0;
    // The water surface runs before the soft-sprite effects (NEWDAWN_POSTFX_STAGE_WATER_XLU) so splashes and ripples draw above it
    const bool doWater = stage == NEWDAWN_POSTFX_STAGE_WATER_XLU && params.waterEnabled != 0 && waterInView &&
                         params.shadowUseExactProj != 0;
    // Waterfalls do not need a water plane: a fall can have no WaterBox near it (docs/WATER_SIM.md).
    const bool doFalls = stage == NEWDAWN_POSTFX_STAGE_WATER_XLU && params.waterEnabled != 0 &&
                         (params.waterFalls > 0.0f || params.waterRivers > 0.0f) &&
                         mFlowCount > 0 && params.shadowUseExactProj != 0 && mHdrActive;
    const bool doWeather = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && mHdrActive && params.atmosWeather &&
                           params.shadowUseExactProj && s.debugView == 0 && s.shadowDebug == 0;
    const bool doAtmos = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && mHdrActive && params.atmosParticles &&
                         params.shadowUseExactProj && s.debugView == 0 && s.shadowDebug == 0;
    const bool doFog = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && (doShadow || params.fogVolumetric == 2) &&
                       mHdrActive && params.fogVolumetric && mFogMarch.id && mFogApply.id &&
                       params.shadowUseExactProj && s.debugView == 0 && s.shadowDebug == 0;
    const bool doVfx = stage == NEWDAWN_POSTFX_STAGE_SCENE_END && mHdrActive && params.vfxEnabled && (mVfxCount > 0 || params.vfxFlash[3] > 0.0f) &&
                       params.shadowUseExactProj && s.debugView == 0 && s.shadowDebug == 0;
    const bool doLocal = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && mHdrActive && params.localEnabled && mLocalLight.id &&
                         params.shadowUseExactProj && s.debugView == 0 && s.shadowDebug == 0;
    const bool doInterior = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && mHdrActive && params.interiorEnabled &&
                            params.interiorWindowCount > 0 && mInteriorLight.id && params.shadowUseExactProj &&
                            s.debugView == 0 && s.shadowDebug == 0;
    const bool doProbe = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && mHdrActive && params.interiorEnabled &&
                         params.shadowUseExactProj && CvInt("gNewDawn.Interior.Probe", 0) != 0;
    const bool doEmissive = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && mHdrActive && params.emissiveEnabled && (params.flags & 1u) &&
                            mLocalMarking && mEmissive.id && params.shadowUseExactProj && s.debugView == 0 &&
                            s.shadowDebug == 0;
    const bool doWorldGi = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && mHdrActive && s.postFxOn && params.giWorld &&
                           params.giWorldStrength > 0.0f && params.shadowUseExactProj && s.debugView == 0 &&
                           s.shadowDebug == 0;
    if (stage == NEWDAWN_POSTFX_STAGE_GROUND_END) {
        // Ground-only colour for the grass: room geometry is in, actors (Link, NPCs, props) are not, so tufts rooted behind
        // an actor never take its colours.
        if (mHdrActive && params.grassEnabled && s.debugView == 0 && s.shadowDebug == 0) {
            GlStateGuard groundGuard;
            groundGuard.Neutralise();
            CaptureGrassGround(fb);
        }
        return;
    }
    const bool doGrass = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && mHdrActive && params.grassEnabled && mGrassCount > 0 &&
                         params.shadowUseExactProj && s.debugView == 0 && s.shadowDebug == 0;
    if (stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && !doGrass)
        mGrassGroundReady = false;
    const bool doCaustics = stage == NEWDAWN_POSTFX_STAGE_OPAQUE_END && params.waterCaustics > 0.0f &&
                            waterInView && params.shadowUseExactProj && mCaustics.id && s.debugView == 0 &&
                            s.shadowDebug == 0;
    const bool doUnderwater = stage == NEWDAWN_POSTFX_STAGE_SCENE_END && params.waterUnderwater > 0.0f &&
                              params.waterUnderwaterState > 0.0f && params.shadowUseExactProj && mUnderwater.id &&
                              s.debugView == 0 && s.shadowDebug == 0;
    const bool doShafts = CvInt("gNewDawn.SunShafts.Mode", 0) == 0 && !doFog && doShadow &&
                          params.shaftEnabled != 0 && params.shaftStrength > 0.0f && s.debugView != 2 &&
                          s.shadowDebug == 0;
    // The AO debug view needs the finished world left alone, otherwise the grade would tint the AO buffer
    const bool doScene = !mHdrActive && stage == NEWDAWN_POSTFX_STAGE_SCENE_END && s.postFxOn && s.debugView != 2 &&
                         (s.bloomOn || s.gradeOn || mArt.on);
    if (!doShadow && !doAo && !doScene && !doWater && !captureWater && !doFog && !doAtmos && !doWeather && !doGi &&
        !doVfx && !doLocal && !doGrass && !doCaustics && !doUnderwater && !doWorldGi && !doFalls) {
        return;
    }

    mStats = s.stats;
    mTimers = mStats || CvInt("gNewDawn.Perf.Overlay", 0) != 0;
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

    if (doGrass) {
        // Drawn into the world before depth is copied: the AO, shadow, local-light and GI passes then treat tufts as
        // ordinary opaque geometry.
        BeginTimer(kGrassTimerStage);
        RunGrass(params, fb);
        mGrassGroundReady = false;
        EndTimer(kGrassTimerStage);
    }
    if (doShadow || doAo || doWater || doFog || doAtmos || doWeather || doGi || doVfx || doLocal || doCaustics ||
        doUnderwater || doInterior || doProbe || doWorldGi || doFalls) {
        PrepareDepth(fb);
    }
    if (doProbe) {
        ProbeInterior(params, fb);
    }
    if (doUnderwater) {
        BeginTimer(kUnderwaterTimerStage);
        RunUnderwater(params, fb);
        EndTimer(kUnderwaterTimerStage);
    }
    // Fully submerged cameras use the underwater pass (scene end) instead of the surface
    if (doWater && !(params.waterUnderwater > 0.0f && params.waterUnderwaterState >= 1.0f && mUnderwater.id)) {
        BeginTimer(kWaterTimerStage);
        RunWaterWaves(params, fb);
        RunRiverFlow(params, fb);
        RunWaterSurface(params, fb);
        if (params.waterStrength > 0.0f || params.waterReflections != 0) {
            RunWater(params, fb);
        }
        RunWaterfalls(params, fb);
        // The lanes go back to their complete dummy: no program ever observes a stale or unit-0 texture there.
        for (int unit : { kWaveBufUnit, kWaveDepthUnit, kRippleUnit, kFlowBufUnit }) {
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_2D, mWaveDummy);
        }
        mWaveBufFrame = UINT32_MAX;
        mFlowBufFrame = UINT32_MAX;
        EndTimer(kWaterTimerStage);
    } else if (doFalls) {
        BeginTimer(kWaterTimerStage);
        RunWaterfalls(params, fb);
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
    if (doGi) {
        BeginTimer(kGiTimerStage);
        RunGlobalIllumination(params, fb, frameIndex, doShadow);
        EndTimer(kGiTimerStage);
    }
    if (doLocal) {
        PrepareLocalLights(params, frameIndex);
        mCasterPositions = shadowPositions;
        mCasterVertexCount = shadowVertexCount;
        if (params.localShadows > 0 && mPointShadowDepth.id) {
            BeginTimer(kPointTimerStage);
            RunPointShadows(params);
            EndTimer(kPointTimerStage);
        } else {
            for (auto& ps : mPointShadow)
                ps.light = -1;
        }
        BeginTimer(kLocalTimerStage);
        RunLocalLights(params, fb, frameIndex);
        EndTimer(kLocalTimerStage);
    }
    if (doInterior) {
        BeginTimer(kInteriorTimerStage);
        RunInteriorLight(params, fb);
        EndTimer(kInteriorTimerStage);
    }
    if (doEmissive) {
        BeginTimer(kEmissiveTimerStage);
        RunEmissive(params, fb);
        EndTimer(kEmissiveTimerStage);
    }
    if (doWorldGi) {
        // After the local lights: their darkening must not swallow the bounce they cause. The albedo proxy for
        // actors only needs the pixel's hue, so the screen-space GI's copy of this frame is reused when it exists.
        BeginTimer(kWorldGiTimerStage);
        const Target* copy = doGi ? &mGiSource : nullptr;
        if (copy == nullptr) {
            EnsureTarget(mGiSource, fb.width, fb.height, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, false);
            glDisable(GL_BLEND);
            glDisable(GL_SCISSOR_TEST);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mGiSource.fbo);
            glBlitFramebuffer(0, 0, fb.width, fb.height, 0, 0, fb.width, fb.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
            copy = &mGiSource;
        }
        RunWorldGi(params, fb, frameIndex, copy);
        EndTimer(kWorldGiTimerStage);
    }
    if (doCaustics) {
        BeginTimer(kCausticsTimerStage);
        RunCaustics(params, fb);
        EndTimer(kCausticsTimerStage);
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
    if (doWeather) {
        auto roofParams = params;
        roofParams.shadowSunDir[0] = roofParams.shadowSunDir[2] = 0.0f;
        roofParams.shadowSunDir[1] = 1.0f;
        roofParams.shadowRadius = 900.0f;
        roofParams.shadowMapSize = 1024;
        std::copy_n(params.atmosEye, 3, roofParams.shadowFocus);
        float roofMatrix[16];
        BuildLightViewProj(roofParams, roofMatrix);
        BeginTimer(12);
        RunWeather(params, fb, roofMatrix, shadowPositions, shadowVertexCount);
        EndTimer(12);
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

    if (doWeather) {
        BeginTimer(13);
        RunWeatherParticles(params, fb);
        EndTimer(13);
    }
    if (doAtmos) {
        float lightMatrix[16] = {};
        if (doShadow) BuildLightViewProj(params, lightMatrix);
        BeginTimer(11);
        RunAtmos(params, fb, lightMatrix, doShadow);
        EndTimer(11);
    }

    if (doVfx) {
        mFlameFrame = frameIndex;
        mFlameScene = params.giSceneId;
        BeginTimer(kVfxTimerStage);
        RunVfx(params, fb);
        EndTimer(kVfxTimerStage);
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
    BuildCascades(params, size);
    const bool csm = mCsm.count > 1;

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
    if (csm) {
        // Every cascade draws all casters into its own tile of the atlas. Depth clamp keeps casters that lie in front
        // of a tile's tight near plane (hills and buildings towards the sun) as depth 0 instead of clipping them.
        glEnable(GL_DEPTH_CLAMP);
        glEnable(GL_SCISSOR_TEST);
        for (int c = 0; c < mCsm.count; c++) {
            glScissor((c & 1) * mCsm.tile, (c >> 1) * mCsm.tile, mCsm.tile, mCsm.tile);
            glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, mCsm.vp[c]);
            glDrawArrays(GL_TRIANGLES, 0, (GLsizei)vertexCount);
        }
    } else {
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)vertexCount);
    }
    CheckError("shadow map render");

    if (mFoliageBatchCount > 0 && mFoliageVerts != nullptr) {
        // Cut-out casters: same light matrix and depth state, fragment-discarded by the texel alpha.
        Use(mShadowFoliage);
        glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, csm ? mCsm.vp[0] : lightViewProj);
        glUniform1i(Loc("uTex"), kFirstUnit);
        glBindVertexArray(mFoliageVao);
        glBindBuffer(GL_ARRAY_BUFFER, mFoliageVbo);
        const GLsizeiptr floats = (GLsizeiptr)(mFoliageBatches[mFoliageBatchCount - 1].firstVertex +
                                               mFoliageBatches[mFoliageBatchCount - 1].vertexCount) * 5;
        glBufferData(GL_ARRAY_BUFFER, floats * sizeof(float), mFoliageVerts, GL_STREAM_DRAW);
        for (uint32_t i = 0; i < mFoliageBatchCount; i++) {
            Bind(0, mFoliageBatches[i].textureId);
            if (csm) {
                for (int c = 0; c < mCsm.count; c++) {
                    glScissor((c & 1) * mCsm.tile, (c >> 1) * mCsm.tile, mCsm.tile, mCsm.tile);
                    glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, mCsm.vp[c]);
                    glDrawArrays(GL_TRIANGLES, (GLint)mFoliageBatches[i].firstVertex,
                                 (GLsizei)mFoliageBatches[i].vertexCount);
                }
            } else {
                glDrawArrays(GL_TRIANGLES, (GLint)mFoliageBatches[i].firstVertex,
                             (GLsizei)mFoliageBatches[i].vertexCount);
            }
        }
        CheckError("shadow foliage render");
    }
    if (csm) {
        glDisable(GL_DEPTH_CLAMP);
        glDisable(GL_SCISSOR_TEST);
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
    Use(csm ? (mHdrActive ? mShadowApplyHdrCsm : mShadowApplyCsm) : (mHdrActive ? mShadowApplyHdr : mShadowApply));
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
    glUniform1f(Loc("uMoonLift"), params.shadowIsMoon ? 0.7f * std::clamp(CvFloat("gNewDawn.Sky.MoonlightLift", 1.0f), 0.0f, 3.0f) : 0.0f);
    glUniform1f(Loc("uMinFactor"), mSettings.shadowFloor);
    glUniform1f(Loc("uRadius"), csm ? mCsm.split[mCsm.count - 1] : params.shadowRadius);
    SetCascadeUniforms();
    float fogMul, fogOff;
    FogCoefficients(params, fogMul, fogOff);
    glUniform1i(Loc("uHdr"), mHdrActive);
    glUniform1f(Loc("uFogMul"), fogMul);
    glUniform1f(Loc("uFogOff"), fogOff);
    // Cascades scale this per tile by its texel size relative to the legacy map (radius 1500 over 4096).
    glUniform1f(Loc("uNormalBias"), csm ? 2.0f * 1500.0f / 4096.0f * 4.0f : 2.0f * params.shadowRadius / (float)size * 4.0f);
    glUniform1i(Loc("uDebug"), debug);
    Draw(fb.fbo, w, h);
    glBindSampler(kFirstUnit + 3, 0);
    mShadowReadyFrame = mSettingsFrame;
    CheckError("shadow apply");
}

// Steps the ripple height field. State persists across frames; the window follows the player in whole texels.
void PostFxGL::StepWaterRipples(const NewDawnPostFxParams& params) {
    constexpr int kSize = 256;
    constexpr float kTexel = 4.0f, kStep = 1.0f / 60.0f;
    const bool fresh = mRipple[0].tex == 0 || mRipple[1].tex == 0;
    EnsureTarget(mRipple[0], kSize, kSize, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
    EnsureTarget(mRipple[1], kSize, kSize, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
    const float now = params.waterTime;
    // A new scene, a rewound clock or a long gap (loading, pause menu) restarts the field from rest.
    if (fresh || mRippleScene != params.giSceneId || now < mRippleLastTime - 0.2f || now - mRippleLastTime > 1.0f) {
        mRippleValid = false;
        mRippleAccum = 0.0f;
        mRippleScene = params.giSceneId;
        mRippleLastTime = now;
    }
    const float dt = std::clamp(now - mRippleLastTime, 0.0f, 0.1f);
    mRippleLastTime = now;
    mRippleAccum += dt;
    int steps = std::min((int)(mRippleAccum / kStep), 4);
    mRippleAccum = std::min(mRippleAccum - steps * kStep, kStep);
    if (!mRippleValid) {
        for (auto& t : mRipple) {
            glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);
        }
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        mRippleRead = 0;
        steps = std::max(steps, 1); // establish the window around the player right away
    }
    if (steps == 0) {
        return;
    }
    const float half = kSize * kTexel * 0.5f;
    const float newMin[2] = { std::floor((params.waterRippleFocus[0] - half) / kTexel) * kTexel,
                              std::floor((params.waterRippleFocus[1] - half) / kTexel) * kTexel };
    int shift[2] = {};
    if (mRippleValid) {
        for (int i = 0; i < 2; i++) {
            shift[i] = (int)std::lround((newMin[i] - mRippleMin[i]) / kTexel);
        }
    }
    float src[NEWDAWN_MAX_WATER_RIPPLES * 4] = {};
    const int count = std::clamp(params.waterRippleCount, 0, NEWDAWN_MAX_WATER_RIPPLES);
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < 4; j++) {
            src[i * 4 + j] = params.waterRippleSrc[i][j];
        }
        src[i * 4 + 3] *= params.waterRipples;
    }
    glDisable(GL_BLEND);
    Use(mWaterRipple);
    glUniform1i(Loc("uState"), kFirstUnit);
    glUniform1f(Loc("uDt"), kStep);
    glUniform1f(Loc("uC2"), (110.0f / kTexel) * (110.0f / kTexel));
    glUniform1f(Loc("uDamp"), 0.9f);
    glUniform3f(Loc("uWin"), newMin[0], newMin[1], kTexel);
    glUniform1i(Loc("uSrcCount"), count);
    glUniform4fv(Loc("uSrc"), count, src);
    for (int i = 0; i < steps; i++) {
        const int write = 1 - mRippleRead;
        Bind(0, mRipple[mRippleRead].tex);
        glUniform2i(Loc("uShift"), i == 0 ? shift[0] : 0, i == 0 ? shift[1] : 0);
        Draw(mRipple[write]);
        mRippleRead = write;
    }
    mRippleMin[0] = newMin[0];
    mRippleMin[1] = newMin[1];
    mRippleValid = true;
    CheckError("water ripples");
}

// Blurred depth map, then the wave trace into mWaveBuf (consumed by RunWaterSurface and RunWater through uWaves).
void PostFxGL::RunWaterWaves(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    mWaveBufFrame = UINT32_MAX;
    const int w = fb.width, h = fb.height;
    const bool ripples = params.waterRipples > 0.0f;
    if (!ripples) {
        mRippleValid = false; // starts from rest when switched back on
    }
    if (!mHdrActive || params.waterRestyle <= 0.0f || (params.waterWaves <= 0.0f && !ripples) || !mWaterDepthMap.id ||
        !mWaterWaves.id || !mWaterRipple.id || params.waterDebug == 1 || mWaterOpaqueFrame != mSettingsFrame ||
        mWaterOpaqueColor.w != w || mWaterOpaqueColor.h != h) {
        return;
    }
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
    if (ripples) {
        StepWaterRipples(params);
    }
    const bool rippleLive = ripples && mRippleValid;
    if (params.waterWaves <= 0.0f && !rippleLive) {
        return;
    }
    const int count = std::clamp(params.waterBoxCount, 0, NEWDAWN_MAX_WATER_BOXES);
    float rect[NEWDAWN_MAX_WATER_BOXES * 4] = {};
    float surfaceY[NEWDAWN_MAX_WATER_BOXES] = {};
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < 4; j++) {
            rect[i * 4 + j] = params.waterBoxes[i][j];
        }
        surfaceY[i] = params.waterBoxes[i][4];
    }
    EnsureTarget(mWaveDepth, (w + 7) / 8, (h + 7) / 8, GL_R16F, GL_RED, GL_FLOAT, false, true);
    EnsureTarget(mWaveBuf, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, false);
    const auto setPlane = [&]() {
        glUniform1i(Loc("uDepth"), kFirstUnit + 2);
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
    Bind(2, mWaterOpaqueDepth.tex); // the bed: opaque depth, exactly what the surface pass reads
    Use(mWaterDepthMap);
    setPlane();
    Draw(mWaveDepth);

    glActiveTexture(GL_TEXTURE0 + kWaveDepthUnit);
    glBindTexture(GL_TEXTURE_2D, mWaveDepth.tex);
    glActiveTexture(GL_TEXTURE0 + kRippleUnit);
    glBindTexture(GL_TEXTURE_2D, rippleLive ? mRipple[mRippleRead].tex : mWaveDummy);
    Use(mWaterWaves);
    setPlane();
    float dirs[12];
    // Each swell train is the wind rotated by a fixed angle; the angles are wide and uneven so crests never run parallel.
    const float angles[6] = { 0.0f, 0.55f, -0.85f, 1.15f, -1.4f, 0.28f };
    const float wl = std::sqrt(params.waterWaveDir[0] * params.waterWaveDir[0] + params.waterWaveDir[1] * params.waterWaveDir[1]);
    const float bx = wl > 1e-4f ? params.waterWaveDir[0] / wl : 1.0f, bz = wl > 1e-4f ? params.waterWaveDir[1] / wl : 0.0f;
    for (int i = 0; i < 6; i++) {
        dirs[i * 2 + 0] = bx * std::cos(angles[i]) - bz * std::sin(angles[i]);
        dirs[i * 2 + 1] = bx * std::sin(angles[i]) + bz * std::cos(angles[i]);
    }
    glUniform2fv(Loc("uWaveDirs"), 6, dirs);
    glUniform1f(Loc("uTime"), params.waterTime);
    glUniform1f(Loc("uWaveAmp"), kWaveBaseAmp * params.waterWaves);
    glUniform1f(Loc("uWaveGust"), params.waterWaveGust);
    glUniform1f(Loc("uPixelAngle"), 2.0f * std::tan(params.fovyDegrees * 3.14159265f / 360.0f) / (float)h);
    glUniform4f(Loc("uRippleWin"), mRippleMin[0], mRippleMin[1], 1.0f / (256.0f * 4.0f), rippleLive ? 1.0f : 0.0f);
    Draw(mWaveBuf);
    mWaveBufFrame = mSettingsFrame;
    CheckError("water waves");
}

// ---------------------------------------------------------------------------------------------------------------------
// Rivers and waterfalls (docs/WATER_SIM.md). Everything here reads the triangles the interpreter recorded from the native
// water draws (with the velocity of their texture scroll); nothing is written back to the game.
// ---------------------------------------------------------------------------------------------------------------------
bool PostFxGL::BuildRiverPrograms() {
    if (mRiverAttempted) {
        return mRiverFlow.id != 0;
    }
    mRiverAttempted = true;
    const std::string version = "#version 330 core\n";
    const std::string flowV = version + kRiverFlowVertex, flowF = version + kRiverFlowFragment;
    const std::string fallV = version + kFallVertex, fallF = version + kRiverFlowLib + kFallFragment;
    const std::string mistV = version + kMistVertex, mistF = version + kMistFragment;
    GLint previousProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM, &previousProgram);
    BuildAtmosShader(mRiverFlow, flowV.c_str(), flowF.c_str());
    if (BuildAtmosShader(mFallDraw, fallV.c_str(), fallF.c_str())) {
        Use(mFallDraw);
        glUniform1i(Loc("uVisibleDepth"), kFirstUnit + 3);
        glUniform1i(Loc("uFlowNoise"), kFlowNoiseUnit);
    }
    if (BuildAtmosShader(mMistDraw, mistV.c_str(), mistF.c_str())) {
        Use(mMistDraw);
        glUniform1i(Loc("uSprite"), kRiverSpriteUnit);
        glUniform1i(Loc("uVisibleDepth"), kFirstUnit + 3);
        glUniform1i(Loc("uFlowNoise"), kFlowNoiseUnit);
    }
    glUseProgram(previousProgram);
    glGenVertexArrays(1, &mFlowVao);
    glGenBuffers(1, &mFlowVbo);
    GLint previousVao = 0, previousVbo = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &previousVao);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &previousVbo);
    glBindVertexArray(mFlowVao);
    glBindBuffer(GL_ARRAY_BUFFER, mFlowVbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glBindVertexArray(previousVao);
    glBindBuffer(GL_ARRAY_BUFFER, previousVbo);
    return mRiverFlow.id != 0;
}

// Loads the flow noise and the mist sprite into their lanes (the textures exist from Init as 1x1 dummies).
bool PostFxGL::EnsureRiverAssets() {
    const std::string root = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetString("gNewDawn.Assets.Path", "assets/newdawn");
    if (root == mRiverAssetPath) {
        return mRiverAssetsOk;
    }
    mRiverAssetPath = root;
    const auto load = [&](GLuint tex, int unit, const std::filesystem::path& path, bool repeat) {
        int w = 0, h = 0, n = 0;
        auto* data = stbi_load(path.string().c_str(), &w, &h, &n, 4);
        if (!data) {
            SPDLOG_WARN("NewDawnRivers: missing {} (set gNewDawn.Assets.Path to the asset root); rivers and waterfalls are skipped",
                        path.string());
            return false;
        }
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        stbi_image_free(data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
        glGenerateMipmap(GL_TEXTURE_2D);
        return true;
    };
    const bool ok = load(mFlowNoise, kFlowNoiseUnit, std::filesystem::path(root) / "water" / "flow_noise_256.png", true) &&
                    load(mRiverSprite, kRiverSpriteUnit, std::filesystem::path(root) / "atmos" / "dust_64.png", false);
    mRiverAssetsOk = ok; // not retried (and not warned about) every frame
    return ok;
}

// Loads the shoreline foam texture into its lane (a 1x1 dummy until then; loaded once per asset root).
void PostFxGL::EnsureFoamTexture() {
    const std::string root = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetString("gNewDawn.Assets.Path", "assets/newdawn");
    if (root == mFoamAssetPath) {
        return;
    }
    mFoamAssetPath = root;
    const auto path = std::filesystem::path(root) / "water" / "foam_lace_512.png";
    int w = 0, h = 0, n = 0;
    auto* data = stbi_load(path.string().c_str(), &w, &h, &n, 4);
    if (!data) {
        SPDLOG_WARN("NewDawnWater: missing {} (set gNewDawn.Assets.Path to the asset root); shoreline foam is skipped", path.string());
        return;
    }
    GLint previousActive;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActive);
    glActiveTexture(GL_TEXTURE0 + kFoamTexUnit);
    glBindTexture(GL_TEXTURE_2D, mFoamTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_2D);
    glActiveTexture(previousActive);
}

// Re-rasterises the recorded river surfaces from the camera into the flow buffer (RG = world xz velocity, B = covered).
void PostFxGL::RunRiverFlow(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    mFlowBufFrame = UINT32_MAX;
    if (params.waterRivers <= 0.0f || mFlowCount == 0 || mFlowData == nullptr || !BuildRiverPrograms() ||
        !EnsureRiverAssets()) {
        return;
    }
    mFlowScratch.clear();
    for (uint32_t t = 0; t < mFlowCount; t++) {
        const float* tri = mFlowData + (size_t)t * NEWDAWN_FLOW_TRI_FLOATS;
        if (tri[12] != 0.0f) {
            continue; // a waterfall face
        }
        for (int i = 0; i < 3; i++) {
            mFlowScratch.insert(mFlowScratch.end(), tri + i * 3, tri + i * 3 + 3);
            mFlowScratch.insert(mFlowScratch.end(), tri + 9, tri + 12);
        }
    }
    if (mFlowScratch.empty()) {
        return;
    }
    const int w = fb.width, h = fb.height;
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    EnsureTarget(mFlowBuf, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, false);
    glBindFramebuffer(GL_FRAMEBUFFER, mFlowBuf.fbo);
    glViewport(0, 0, w, h);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    Use(mRiverFlow);
    glUniformMatrix4fv(Loc("uViewProj"), 1, GL_FALSE, params.shadowViewProj);
    glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), params.shadowFlipY);
    glBindVertexArray(mFlowVao);
    glBindBuffer(GL_ARRAY_BUFFER, mFlowVbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(mFlowScratch.size() * sizeof(float)), mFlowScratch.data(), GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(mFlowScratch.size() / 6));
    glBindVertexArray(mVao);
    mFlowBufFrame = mSettingsFrame;
    CheckError("river flow");
}

// Waterfalls: a flowing layer over the native fall faces, then mist, spray and foam at their feet.
void PostFxGL::RunWaterfalls(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    if ((params.waterFalls <= 0.0f && params.waterRivers <= 0.0f) || mFlowCount == 0 || mFlowData == nullptr ||
        !BuildRiverPrograms() || !EnsureRiverAssets() || !mFallDraw.id) {
        return;
    }
    mFlowScratch.clear();
    // Foot of each fall: per 48-unit cell the lowest vertex, the top, and the face normal (a fall is vertical, so one cell
    // holds one column of its mesh).
    struct Cell {
        float low[3] = { 0, 1e9f, 0 }, high = -1e9f, nx = 0, nz = 0;
    };
    std::unordered_map<int64_t, Cell> cells;
    // Painted streams (kind 2) go through the same overlay shader in their own mode, drawn after the falls.
    std::vector<float> streams;
    for (uint32_t t = 0; t < mFlowCount; t++) {
        const float* tri = mFlowData + (size_t)t * NEWDAWN_FLOW_TRI_FLOATS;
        if (tri[12] == 2.0f && params.waterRivers > 0.0f) {
            for (int i = 0; i < 3; i++) {
                streams.insert(streams.end(), tri + i * 3, tri + i * 3 + 3);
                streams.insert(streams.end(), tri + 9, tri + 12);
            }
        }
    }
    for (uint32_t t = 0; t < mFlowCount; t++) {
        const float* tri = mFlowData + (size_t)t * NEWDAWN_FLOW_TRI_FLOATS;
        if (tri[12] != 1.0f || params.waterFalls <= 0.0f) {
            continue;
        }
        for (int i = 0; i < 3; i++) {
            mFlowScratch.insert(mFlowScratch.end(), tri + i * 3, tri + i * 3 + 3);
            mFlowScratch.insert(mFlowScratch.end(), tri + 9, tri + 12);
        }
        const float e1[3] = { tri[3] - tri[0], tri[4] - tri[1], tri[5] - tri[2] };
        const float e2[3] = { tri[6] - tri[0], tri[7] - tri[1], tri[8] - tri[2] };
        float nx = e1[1] * e2[2] - e1[2] * e2[1], nz = e1[0] * e2[1] - e1[1] * e2[0];
        const float nl = std::sqrt(nx * nx + nz * nz);
        if (nl > 1e-3f) {
            nx /= nl;
            nz /= nl;
        }
        for (int i = 0; i < 3; i++) {
            const float* p = tri + i * 3;
            const int64_t key = ((int64_t)std::floor(p[0] / 48.0f) << 32) ^ (int64_t)(uint32_t)(int32_t)std::floor(p[2] / 48.0f);
            Cell& cell = cells[key];
            if (p[1] < cell.low[1]) {
                cell.low[0] = p[0], cell.low[1] = p[1], cell.low[2] = p[2];
            }
            cell.high = std::max(cell.high, p[1]);
            // face the camera so the mist leaves towards the viewer's side of the fall
            const float sign = ((params.waterCameraEye[0] - p[0]) * nx + (params.waterCameraEye[2] - p[2]) * nz) < 0.0f ? -1.0f : 1.0f;
            cell.nx += nx * sign;
            cell.nz += nz * sign;
        }
    }
    if (mFlowScratch.empty() && streams.empty()) {
        return;
    }
    const int w = fb.width, h = fb.height;
    float light[3], sky[3], fogColor[3];
    for (int i = 0; i < 3; ++i) {
        light[i] = mHdrActive ? DecodeColor(params.waterLightColor[i]) * std::clamp(CvFloat("gNewDawn.HDR.SunIntensity", 1.8f), 0.0f, 16.0f)
                              : params.waterLightColor[i];
        sky[i] = mHdrActive ? DecodeColor(params.waterSkyColor[i]) * std::clamp(CvFloat("gNewDawn.HDR.SkyIntensity", 1.0f), 0.0f, 4.0f)
                            : params.waterSkyColor[i];
        fogColor[i] = mHdrActive ? DecodeColor(params.waterSkyColor[i]) * std::clamp(CvFloat("gNewDawn.HDR.FogIntensity", 1.0f), 0.0f, 4.0f)
                                 : params.waterSkyColor[i];
    }
    float fogMul, fogOff;
    FogCoefficients(params, fogMul, fogOff);
    const float strength = std::min(params.waterFalls, 2.0f);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    Bind(3, mDepth.tex); // visible depth
    glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
    glViewport(0, 0, w, h);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    const float pixelAngle = 2.0f * std::tan(params.fovyDegrees * 3.14159265f / 360.0f) / (float)h;
    const auto common = [&]() {
        glUniformMatrix4fv(Loc("uViewProj"), 1, GL_FALSE, params.shadowViewProj);
        glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
        glUniform1f(Loc("uFlipY"), params.shadowFlipY);
        glUniform1f(Loc("uTime"), params.waterTime);
        glUniform1f(Loc("uNear"), params.zNear);
        glUniform1f(Loc("uFar"), params.zFar);
        glUniform1f(Loc("uStrength"), strength);
        glUniform1f(Loc("uPixelAngle"), pixelAngle);
        glUniform1i(Loc("uDebug"), CvInt("gNewDawn.Water.FlowDebug", 0));
        glUniform1f(Loc("uFogMul"), fogMul);
        glUniform1f(Loc("uFogOff"), fogOff);
        glUniform3fv(Loc("uFogColor"), 1, fogColor);
        glUniform3fv(Loc("uEye"), 1, params.waterCameraEye);
        glUniform3fv(Loc("uLightDir"), 1, params.waterLightDir);
        glUniform3fv(Loc("uLightColor"), 1, light);
        glUniform3fv(Loc("uSky"), 1, sky);
    };
    Use(mFallDraw);
    common();
    glBindVertexArray(mFlowVao);
    glBindBuffer(GL_ARRAY_BUFFER, mFlowVbo);
    if (!mFlowScratch.empty()) {
        glUniform1i(Loc("uMode"), 0);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(mFlowScratch.size() * sizeof(float)), mFlowScratch.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(mFlowScratch.size() / 6));
    }
    if (!streams.empty()) {
        glUniform1i(Loc("uMode"), 1);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(streams.size() * sizeof(float)), streams.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(streams.size() / 6));
    }
    // Mist, spray and foam at the foot of each fall: nearest first, at most 48, none beyond 4000 units.
    struct Emitter {
        float dist, pos[4], dir[4];
    };
    std::vector<Emitter> emitters;
    for (const auto& [key, cell] : cells) {
        const float height = cell.high - cell.low[1];
        if (height < 36.0f) {
            continue;
        }
        const float dx = cell.low[0] - params.waterCameraEye[0], dz = cell.low[2] - params.waterCameraEye[2];
        const float dist = std::sqrt(dx * dx + dz * dz);
        if (dist > 4000.0f) {
            continue;
        }
        float nl = std::sqrt(cell.nx * cell.nx + cell.nz * cell.nz);
        Emitter e;
        e.dist = dist;
        e.pos[0] = cell.low[0], e.pos[1] = cell.low[1], e.pos[2] = cell.low[2];
        e.pos[3] = std::clamp(height / 180.0f, 0.45f, 1.5f);
        e.dir[0] = nl > 1e-3f ? cell.nx / nl : 0.0f, e.dir[1] = 0.0f, e.dir[2] = nl > 1e-3f ? cell.nz / nl : 1.0f;
        e.dir[3] = 34.0f;
        emitters.push_back(e);
    }
    std::sort(emitters.begin(), emitters.end(), [](const Emitter& a, const Emitter& b) { return a.dist < b.dist; });
    if (emitters.size() > 48) {
        emitters.resize(48);
    }
    if (!emitters.empty() && mMistDraw.id) {
        float emit[48][4] = {}, emitB[48][4] = {};
        for (size_t i = 0; i < emitters.size(); i++) {
            std::copy(emitters[i].pos, emitters[i].pos + 4, emit[i]);
            std::copy(emitters[i].dir, emitters[i].dir + 4, emitB[i]);
        }
        Use(mMistDraw);
        common();
        const float projection = 1.0f / std::tan(params.fovyDegrees * 0.00872664626f);
        glUniform2f(Loc("uProjection"), projection * float(h) / w, projection);
        glUniform2f(Loc("uSize"), (float)w, (float)h);
        glUniform1i(Loc("uCount"), (int)emitters.size());
        glUniform4fv(Loc("uEmit[0]"), 48, &emit[0][0]);
        glUniform4fv(Loc("uEmitB[0]"), 48, &emitB[0][0]);
        glBindVertexArray(mVao);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, (GLsizei)emitters.size() * 30);
    }
    glBindVertexArray(mVao);
    glDisable(GL_BLEND);
    CheckError("waterfalls");
}

void PostFxGL::RunWaterSurface(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    const int w = fb.width, h = fb.height;
    const float foam = params.waterFoam;
    const float refraction = mHdrActive ? params.waterRefraction : 0.0f; // the overlay darkens: needs a float target
    const float restyle = mHdrActive ? params.waterRestyle : 0.0f;
    if ((foam <= 0.0f && refraction <= 0.0f && restyle <= 0.0f) || params.waterDebug == 1 ||
        mWaterOpaqueFrame != mSettingsFrame || mWaterOpaqueColor.w != w || mWaterOpaqueColor.h != h) {
        return;
    }
    const int count = std::clamp(params.waterBoxCount, 0, NEWDAWN_MAX_WATER_BOXES);
    float rect[NEWDAWN_MAX_WATER_BOXES * 4] = {};
    float surfaceY[NEWDAWN_MAX_WATER_BOXES] = {};
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < 4; j++) {
            rect[i * 4 + j] = params.waterBoxes[i][j];
        }
        surfaceY[i] = params.waterBoxes[i][4];
    }
    if (foam > 0.0f) {
        EnsureFoamTexture();
    }
    // Work on a copy of the colour so the game framebuffer is never sampled while bound.
    glDisable(GL_SCISSOR_TEST);
    EnsureTarget(mScene, w, h, mHdrActive ? GL_RGBA16F : GL_RGB8, GL_RGBA, mHdrActive ? GL_FLOAT : GL_UNSIGNED_BYTE,
                 false, true);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mScene.fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    Bind(0, mScene.tex);
    Bind(1, mWaterOpaqueColor.tex);
    Bind(2, mWaterOpaqueDepth.tex);
    Bind(3, mDepth.tex);
    Use(mWaterSurface);
    glUniform1i(Loc("uScene"), kFirstUnit + 0);
    glUniform1i(Loc("uOpaqueColor"), kFirstUnit + 1);
    glUniform1i(Loc("uDepth"), kFirstUnit + 2);
    glUniform1i(Loc("uVisibleDepth"), kFirstUnit + 3);
    glUniform2i(Loc("uSize"), w, h);
    glUniform1f(Loc("uNear"), params.zNear);
    glUniform1f(Loc("uFar"), params.zFar);
    glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), params.shadowFlipY);
    glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, params.shadowInvViewProj);
    glUniformMatrix4fv(Loc("uViewProj"), 1, GL_FALSE, params.shadowViewProj);
    glUniform3fv(Loc("uEye"), 1, params.waterCameraEye);
    glUniform1i(Loc("uBoxCount"), count);
    glUniform4fv(Loc("uBoxRect"), count, rect);
    glUniform1fv(Loc("uBoxY"), count, surfaceY);
    glUniform1f(Loc("uTime"), params.waterTime);
    glUniform1f(Loc("uFoam"), foam);
    glUniform1f(Loc("uRefr"), refraction);
    glUniform1f(Loc("uRestyle"), restyle);
    glUniform1f(Loc("uSurfaceVisibility"), params.waterSurfaceVisibility);
    glUniform3fv(Loc("uScatter"), 1, params.waterScatter);
    glUniform3fv(Loc("uAbsorb"), 1, params.waterAbsorb);
    glUniform1f(Loc("uPixelAngle"), 2.0f * std::tan(params.fovyDegrees * 3.14159265f / 360.0f) / (float)h);
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
    float fogMul, fogOff;
    FogCoefficients(params, fogMul, fogOff);
    float fogColor[3];
    for (int i = 0; i < 3; ++i) {
        fogColor[i] = mHdrActive ? DecodeColor(params.waterSkyColor[i]) *
                                       std::clamp(CvFloat("gNewDawn.HDR.FogIntensity", 1.0f), 0.0f, 4.0f)
                                 : params.waterSkyColor[i];
    }
    glUniform1f(Loc("uFogMul"), fogMul);
    glUniform1f(Loc("uFogOff"), fogOff);
    glUniform3fv(Loc("uFogColor"), 1, fogColor);
    const bool waves = mWaveBufFrame == mSettingsFrame;
    if (waves) {
        glActiveTexture(GL_TEXTURE0 + kWaveBufUnit);
        glBindTexture(GL_TEXTURE_2D, mWaveBuf.tex);
    }
    glUniform1i(Loc("uWaves"), waves);
    const bool rivers = mFlowBufFrame == mSettingsFrame && params.waterRivers > 0.0f;
    if (rivers) {
        glActiveTexture(GL_TEXTURE0 + kFlowBufUnit);
        glBindTexture(GL_TEXTURE_2D, mFlowBuf.tex);
    }
    glUniform1f(Loc("uRiver"), rivers ? std::min(params.waterRivers, 2.0f) : 0.0f);
    glUniform1i(Loc("uDebug"), params.waterDebug);
    glUniform1f(Loc("uWaveAmp"), kWaveBaseAmp * params.waterWaves);
    // Swash is an ocean feature: it follows the swell setting, and the foam toggle/strength.
    glUniform1f(Loc("uSwash"), params.waterWaves > 0.0f && restyle > 0.0f ? std::min(foam, 1.0f) : 0.0f);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    Draw(fb.fbo, w, h);
    glDisable(GL_BLEND);
    CheckError("water surface");
}


bool PostFxGL::EnsureCausticAssets() {
    const std::string root = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetString(
        "gNewDawn.Assets.Path", "assets/newdawn");
    if (root == mCausticAssetPath)
        return mCausticTex != 0;
    mCausticAssetPath = root;
    if (mCausticTex) {
        glDeleteTextures(1, &mCausticTex);
        mCausticTex = 0;
    }
    constexpr int dim = 256, cols = 8, rows = 4;
    const auto path = std::filesystem::path(root) / "caustics" / "caustics_8x4.png";
    int w = 0, h = 0, components = 0;
    auto* data = stbi_load(path.string().c_str(), &w, &h, &components, 1);
    if (!data || w != dim * cols || h != dim * rows) {
        stbi_image_free(data);
        SPDLOG_WARN("NewDawnCaustics: missing/unsupported flipbook {} (expected {}x{} grey); caustics and underwater "
                    "shafts are skipped. Set gNewDawn.Assets.Path to the asset root.",
                    path.string(), dim * cols, dim * rows);
        return false;
    }
    std::vector<uint8_t> layers((size_t)dim * dim * cols * rows);
    for (int f = 0; f < cols * rows; ++f) {
        for (int row = 0; row < dim; ++row) {
            std::memcpy(layers.data() + ((size_t)f * dim + row) * dim,
                        data + ((size_t)(f / cols) * dim + row) * w + (size_t)(f % cols) * dim, dim);
        }
    }
    stbi_image_free(data);
    glActiveTexture(GL_TEXTURE0 + kCausticUnit);
    glGenTextures(1, &mCausticTex);
    glBindTexture(GL_TEXTURE_2D_ARRAY, mCausticTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R8, dim, dim, cols * rows, 0, GL_RED, GL_UNSIGNED_BYTE, layers.data());
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
    SPDLOG_INFO("NewDawnCaustics: loaded the {} caustics frames from {}", cols * rows, path.string());
    return true;
}

// Shared by the caustics and underwater passes: plane/camera uniforms, the shadow sampler and the flipbook clock.
void PostFxGL::RunCaustics(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    if (!EnsureCausticAssets()) {
        return;
    }
    const int w = fb.width, h = fb.height;
    const int count = std::clamp(params.waterBoxCount, 0, NEWDAWN_MAX_WATER_BOXES);
    float rect[NEWDAWN_MAX_WATER_BOXES * 4] = {};
    float surfaceY[NEWDAWN_MAX_WATER_BOXES] = {};
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < 4; j++) {
            rect[i * 4 + j] = params.waterBoxes[i][j];
        }
        surfaceY[i] = params.waterBoxes[i][4];
    }
    const bool shadowReady = params.waterCausticSun != 0 && mShadowReadyFrame == mSettingsFrame && mShadowMap.tex != 0;
    const bool csm = shadowReady && mCsm.count > 1;
    glDisable(GL_SCISSOR_TEST);
    Bind(1, mDepth.tex);
    Bind(3, shadowReady ? mShadowMap.tex : mDepth.tex);
    glBindSampler(kFirstUnit + 3, mCmpSampler);
    Use(csm ? mCausticsCsm : mCaustics);
    SetCascadeUniforms();
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
    glUniform1i(Loc("uShadowReady"), shadowReady);
    if (shadowReady && !csm) {
        float lightViewProj[16];
        BuildLightViewProj(params, lightViewProj);
        glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, lightViewProj);
    }
    glUniform1f(Loc("uCausticFrame"), params.waterTime * 12.0f);
    glUniform1f(Loc("uStrength"), params.waterCaustics);
    glUniform1f(Loc("uPixelAngle"), 2.0f * std::tan(params.fovyDegrees * 3.14159265f / 360.0f) / (float)h);
    glUniform3fv(Loc("uLightDir"), 1, params.waterCausticDir);
    glUniform3fv(Loc("uLightColor"), 1, params.waterCausticColor);
    glUniform3fv(Loc("uAbsorb"), 1, params.waterAbsorb);
    float fogMul, fogOff;
    FogCoefficients(params, fogMul, fogOff);
    glUniform1f(Loc("uFogMul"), fogMul);
    glUniform1f(Loc("uFogOff"), fogOff);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_DST_COLOR, GL_ONE); // dst * (1 + gain)
    Draw(fb.fbo, w, h);
    glDisable(GL_BLEND);
    glBindSampler(kFirstUnit + 3, 0);
    CheckError("water caustics");
}

void PostFxGL::RunUnderwater(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    if (!EnsureCausticAssets() || !mUnderwaterShaft.id) {
        return;
    }
    const int w = fb.width, h = fb.height;
    const int hw = std::max(w / 2, 1), hh = std::max(h / 2, 1);
    const bool shadowReady = params.waterCausticSun != 0 && mShadowReadyFrame == mSettingsFrame && mShadowMap.tex != 0;
    const bool csm = shadowReady && mCsm.count > 1;
    glDisable(GL_SCISSOR_TEST);
    EnsureTarget(mScene, w, h, mHdrActive ? GL_RGBA16F : GL_RGB8, GL_RGBA, mHdrActive ? GL_FLOAT : GL_UNSIGNED_BYTE,
                 false, true);
    EnsureTarget(mUwShaft, hw, hh, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mScene.fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    float light[3], sky[3];
    for (int i = 0; i < 3; ++i) {
        light[i] = mHdrActive ? DecodeColor(params.waterCausticColor[i]) *
                                    std::clamp(CvFloat("gNewDawn.HDR.SunIntensity", 1.8f), 0.0f, 16.0f)
                              : params.waterCausticColor[i];
        sky[i] = mHdrActive ? DecodeColor(params.waterSkyColor[i]) *
                                  std::clamp(CvFloat("gNewDawn.HDR.SkyIntensity", 1.0f), 0.0f, 4.0f)
                            : params.waterSkyColor[i];
    }
    const auto setCommon = [&]() {
        glUniform2i(Loc("uSize"), w, h);
        glUniform1f(Loc("uNear"), params.zNear);
        glUniform1f(Loc("uFar"), params.zFar);
        glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
        glUniform1f(Loc("uFlipY"), params.shadowFlipY);
        glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, params.shadowInvViewProj);
        glUniform3fv(Loc("uEye"), 1, params.waterCameraEye);
        glUniform1f(Loc("uCausticFrame"), params.waterTime * 12.0f);
        glUniform1f(Loc("uTime"), params.waterTime);
        glUniform1f(Loc("uUnder"), params.waterUnderwater * params.waterUnderwaterState);
        glUniform1f(Loc("uPlaneY"), params.waterEyeSurfaceY);
        glUniform4fv(Loc("uEyeRect"), 1, params.waterEyeRect);
        glUniform3fv(Loc("uScatter"), 1, params.waterScatter);
        glUniform3fv(Loc("uAbsorb"), 1, params.waterAbsorb);
        glUniform3fv(Loc("uLightDir"), 1, params.waterCausticDir);
        glUniform3fv(Loc("uLightColor"), 1, light);
        glUniform3fv(Loc("uSky"), 1, sky);
        glUniform1f(Loc("uShaftStrength"), 0.4f);
    };
    // 1. Half resolution sunlit shafts.
    Bind(1, mDepth.tex);
    Bind(3, shadowReady ? mShadowMap.tex : mDepth.tex);
    glBindSampler(kFirstUnit + 3, mCmpSampler);
    Use(csm ? mUnderwaterShaftCsm : mUnderwaterShaft);
    SetCascadeUniforms();
    setCommon();
    glUniform1i(Loc("uShadowReady"), shadowReady);
    if (shadowReady && !csm) {
        float lightViewProj[16];
        BuildLightViewProj(params, lightViewProj);
        glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, lightViewProj);
    }
    Draw(mUwShaft);
    glBindSampler(kFirstUnit + 3, 0);
    // 2. Full resolution absorption, ambient in-scatter, refraction, plus the upsampled shafts.
    Bind(0, mScene.tex);
    Bind(2, mUwShaft.tex);
    Use(mUnderwater);
    setCommon();
    Draw(fb.fbo, w, h);
    CheckError("underwater");
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
        glUniform1f(Loc("uSurfaceDetail"), mHdrActive ? params.waterRestyle : 0.0f);
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
    const bool waves = mWaveBufFrame == mSettingsFrame;
    if (waves) {
        glActiveTexture(GL_TEXTURE0 + kWaveBufUnit);
        glBindTexture(GL_TEXTURE_2D, mWaveBuf.tex);
    }
    glUniform1i(Loc("uWaves"), waves);
    if (filter) {
        glDisable(GL_BLEND);
        Draw(mWaterReflectionRaw);
        Bind(0, mWaterReflectionRaw.tex);
        Bind(1, mDepth.tex);
        Use(mWaterReflectionApply);
        setPlaneUniforms();
        glUniform1i(Loc("uReflection"), kFirstUnit + 0);
        glUniform1f(Loc("uSurfaceDetail"), mHdrActive ? params.waterRestyle : 0.0f);
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

bool PostFxGL::EnsureFlareAssets() {
    const std::string root = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetString(
        "gNewDawn.Assets.Path", "assets/newdawn");
    if (root == mFlareAssetPath)
        return mFlareSprites != 0;
    mFlareAssetPath = root;
    if (mFlareSprites) {
        glDeleteTextures(1, &mFlareSprites);
        mFlareSprites = 0;
    }
    const char* files[] = { "star.png", "extend.png", "hexangle.png", "iris.png", "sun.png", "ring.png" };
    constexpr int dim = 256, layers = 6;
    std::vector<uint8_t> pixels(dim * dim * layers * 4, 0);
    for (int layer = 0; layer < layers; ++layer) {
        const auto path = std::filesystem::path(root) / "oga-lens-flares" / "flare" / files[layer];
        int w = 0, h = 0, components = 0;
        auto* data = stbi_load(path.string().c_str(), &w, &h, &components, 4);
        if (!data || w < 1 || h < 1 || w > dim || h > dim) {
            stbi_image_free(data);
            SPDLOG_WARN("NewDawnFlare: missing/unsupported asset {}; set gNewDawn.Assets.Path to the asset root", path.string());
            return false;
        }
        const int x = (dim - w) / 2, y = (dim - h) / 2;
        for (int row = 0; row < h; ++row)
            std::memcpy(pixels.data() + ((layer * dim + y + row) * dim + x) * 4, data + row * w * 4, w * 4);
        mFlareSpriteScale[layer * 2] = (float)w / dim;
        mFlareSpriteScale[layer * 2 + 1] = (float)h / dim;
        stbi_image_free(data);
    }
    glActiveTexture(GL_TEXTURE0 + kFlareSpritesUnit);
    glGenTextures(1, &mFlareSprites);
    glBindTexture(GL_TEXTURE_2D_ARRAY, mFlareSprites);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, dim, dim, layers, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
    SPDLOG_INFO("NewDawnFlare: loaded six downloaded VFX sprites from {}", root);
    return true;
}

void PostFxGL::RunSunFlare(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    if (!mUsable || !mHdrActive || !mFlareVisibility.id || !mFlareApply.id || !params.shadowUseExactProj)
        return;
    float clip[4] = {};
    for (int j = 0; j < 4; ++j) {
        clip[j] = params.flareSunPos[0] * params.shadowViewProj[j] +
                  params.flareSunPos[1] * params.shadowViewProj[4 + j] +
                  params.flareSunPos[2] * params.shadowViewProj[8 + j] + params.shadowViewProj[12 + j];
    }
    if (clip[3] <= 0.0f)
        return;
    float uv[2] = { (clip[0] / clip[3] / params.shadowNdcXScale) * 0.5f + 0.5f,
                    (clip[1] / clip[3] * params.shadowFlipY) * 0.5f + 0.5f };
    if (uv[0] < -0.02f || uv[0] > 1.02f || uv[1] < -0.02f || uv[1] > 1.02f)
        return;
    mStats = CvInt("gNewDawn.PostFx.Debug.Stats", 0) != 0;
    mTimers = mStats || CvInt("gNewDawn.Perf.Overlay", 0) != 0;
    GlStateGuard guard;
    guard.Neutralise();
    glBindVertexArray(mVao);
    const bool flare = CvInt("gNewDawn.Sky.Flare", 0) && EnsureFlareAssets();
    const bool rays = mRaySkyReady && params.screenShaftStrength > 0.0f && mRayMask.id && mRayRadial.id && mRayApply.id;
    if (!flare && !rays)
        return;
    BeginTimer(kFlareTimerStage);
    const int w = fb.width, h = fb.height;
    PrepareDepth(fb);
    EnsureTarget(mScene, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
    EnsureTarget(mFlareEmitter, 1, 1, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, false);
    if (!mUsable) {
        EndTimer(kFlareTimerStage);
        return;
    }
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mScene.fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    Bind(0, mScene.tex);
    Bind(1, mDepth.tex);
    Use(mFlareVisibility);
    glUniform2i(Loc("uSize"), w, h);
    glUniform2fv(Loc("uSunUv"), 1, uv);
    glUniform1f(Loc("uSunDepth"), std::min(clip[2] / clip[3] * 0.5f + 0.5f, 1.0f));
    glActiveTexture(GL_TEXTURE0 + kFlareCloudUnit);
    glBindTexture(GL_TEXTURE_2D, mFlareCloudOpacity > 0.0f ? mClouds.tex : mFlareCloudNeutral);
    glUniform1f(Loc("uCloudOpacity"), mFlareCloudOpacity);
    Draw(mFlareEmitter);
    glBindTexture(GL_TEXTURE_2D, mFlareCloudNeutral);
    glActiveTexture(GL_TEXTURE0 + kFlareUnit);
    glBindTexture(GL_TEXTURE_2D, mFlareEmitter.tex);
    if (flare) {
        glActiveTexture(GL_TEXTURE0 + kFlareSpritesUnit);
        glBindTexture(GL_TEXTURE_2D_ARRAY, mFlareSprites);
        Use(mFlareApply);
        glUniform2fv(Loc("uSpriteScale"), 6, mFlareSpriteScale);
        glUniform2fv(Loc("uSunUv"), 1, uv);
        glUniform2f(Loc("uScreenSize"), (float)w, (float)h);
        glEnable(GL_BLEND);
        glBlendEquation(GL_FUNC_ADD);
        glBlendFunc(GL_ONE, GL_ONE);
        Draw(fb.fbo, w, h);
        glDisable(GL_BLEND);
        glActiveTexture(GL_TEXTURE0 + kFlareSpritesUnit);
        glBindTexture(GL_TEXTURE_2D_ARRAY, mFlareSpritesNeutral);
    }
    EndTimer(kFlareTimerStage);
    if (rays) {
        BeginTimer(kShaftTimerStage);
        RunScreenShafts(params, fb, uv);
        EndTimer(kShaftTimerStage);
    }
    glActiveTexture(GL_TEXTURE0 + kFlareUnit);
    glBindTexture(GL_TEXTURE_2D, mFlareNeutral);
    CheckError("textured sun flare");
}

void PostFxGL::RunScreenShafts(const NewDawnPostFxParams& params, const FramebufferOGL& fb,
                                const float sunUv[2]) {
    const int w = fb.width, h = fb.height;
    const int hw = (w + 1) / 2, hh = (h + 1) / 2;
    EnsureTarget(mRayMaskTarget, hw, hh, GL_R16F, GL_RED, GL_FLOAT, false, true);
    EnsureTarget(mRayTarget, hw, hh, GL_R16F, GL_RED, GL_FLOAT, false, true);
    if (!mUsable)
        return;
    glDisable(GL_BLEND);
    Bind(0, mScene.tex);
    Bind(1, mDepth.tex);
    Bind(2, mSky.tex);
    glActiveTexture(GL_TEXTURE0 + kFlareCloudUnit);
    glBindTexture(GL_TEXTURE_2D, mFlareCloudOpacity > 0.0f ? mClouds.tex : mFlareCloudNeutral);
    Use(mRayMask);
    glUniform1f(Loc("uCloudOpacity"), mFlareCloudOpacity);
    glUniform2i(Loc("uSize"), w, h);
    glUniform2fv(Loc("uSunUv"), 1, sunUv);
    Draw(mRayMaskTarget);
    Bind(0, mRayMaskTarget.tex);
    Use(mRayRadial);
    glUniform2fv(Loc("uSunUv"), 1, sunUv);
    Draw(mRayTarget);
    Bind(0, mScene.tex);
    Bind(3, mRayTarget.tex);
    Use(mRayApply);
    glUniform1f(Loc("uCloudOpacity"), mFlareCloudOpacity);
    float color[3];
    for (int i = 0; i < 3; ++i)
        color[i] = DecodeColor(params.screenShaftColor[i]);
    const float luminance = std::max(0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2], 0.01f);
    for (float& channel : color)
        channel = 0.65f + 0.35f * channel / luminance;
    glUniform2i(Loc("uSize"), w, h);
    glUniform3fv(Loc("uColor"), 1, color);
    glUniform2fv(Loc("uSunUv"), 1, sunUv);
    glUniform1f(Loc("uStrength"), params.screenShaftStrength);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE);
    Draw(fb.fbo, w, h);
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0 + kFlareCloudUnit);
    glBindTexture(GL_TEXTURE_2D, mFlareCloudNeutral);
    CheckError("screen-space sun rays");
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
    if (params.fogVolumetric == 1) {
        BuildLightViewProj(params, lightViewProj);
    } else {
        std::fill(lightViewProj, lightViewProj + 16, 0.0f);
        lightViewProj[0] = lightViewProj[5] = lightViewProj[10] = lightViewProj[15] = 1.0f;
    }
    float fogMul, fogOff;
    FogCoefficients(params, fogMul, fogOff);
    // Native start distance supplies the aerosol scale. Bound the artist input so special/near-infinite
    // fog distances cannot make the volume opaque or numerically unstable.
    float ndcStart = fogMul > 0.0f ? std::clamp(-fogOff / fogMul, -1.0f, 1.0f) : 1.0f;
    float startDistance = 2.0f * params.zNear * params.zFar /
                          (params.zFar + params.zNear - ndcStart * (params.zFar - params.zNear));
    const float sunHeight = std::clamp(params.shadowSunDir[1], 0.0f, 1.0f);
    const float density = 0.32f / std::clamp(startDistance, 800.0f, 2400.0f) * params.fogDensityScale *
                          (1.4f - 0.7f * sunHeight);
    const float sunFade = params.fogVolumetric == 1 && !params.shadowIsMoon ? std::clamp((sunHeight + 0.02f) / 0.22f, 0.0f, 1.0f) : 0.0f;
    float fog[3], sun[3], ambient[3];
    for (int i = 0; i < 3; ++i) {
        fog[i] = DecodeColor(params.fogColor[i]) *
                 std::clamp(CvFloat("gNewDawn.HDR.FogIntensity", 1.0f), 0.0f, 4.0f);
        ambient[i] = DecodeColor(params.fogAmbient[i]);
        sun[i] = DecodeColor(params.shadowDirColor[i]) * sunFade *
                 std::clamp(CvFloat("gNewDawn.HDR.SunIntensity", 1.8f), 0.0f, 16.0f);
    }
    Bind(1, mDepth.tex);
    Bind(3, params.fogVolumetric == 2 ? mDepth.tex : mShadowMap.tex);
    glBindSampler(kFirstUnit + 3, mCmpSampler);
    Use(mCsm.count > 1 && params.fogVolumetric == 1 ? mFogMarchCsm : mFogMarch);
    SetCascadeUniforms();
    glUniform2i(Loc("uSize"), w, h);
    glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, params.shadowInvViewProj);
    glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, lightViewProj);
    glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), params.shadowFlipY);
    glUniform3fv(Loc("uEye"), 1, params.shadowCameraEye);
    glUniform3fv(Loc("uSunDir"), 1, params.shadowSunDir);
    glUniform3fv(Loc("uSunColor"), 1, sun);
    glUniform1i(Loc("uFogSun"), params.fogVolumetric == 1);
    glUniform1i(Loc("uPointCount"), params.fogPointCount);
    glUniform4fv(Loc("uPointPosRadius"), NEWDAWN_FOG_POINT_LIGHTS, params.fogPointPosRadius[0]);
    float pointColors[NEWDAWN_FOG_POINT_LIGHTS][3];
    for (int j = 0; j < NEWDAWN_FOG_POINT_LIGHTS; ++j)
        for (int i = 0; i < 3; ++i)
            pointColors[j][i] = DecodeColor(params.fogPointColor[j][i]);
    glUniform3fv(Loc("uPointColor"), NEWDAWN_FOG_POINT_LIGHTS, pointColors[0]);
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
    Use(mCsm.count > 1 ? mShaftMarchCsm : mShaftMarch);
    SetCascadeUniforms();
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

void PostFxGL::RunGlobalIllumination(const NewDawnPostFxParams& params, const FramebufferOGL& fb,
                                    uint32_t frameIndex, bool sunShadow) {
    const int w = fb.width, h = fb.height;
    const int hw = (w + 3) / 4, hh = (h + 3) / 4; // quarter resolution: bounce light is low-frequency
    bool valid = mGiValid && mGiFrame + 1 == frameIndex && mGiScene == params.giSceneId &&
                 mGiSource.w == w && mGiSource.h == h && mGiNdcXScale == params.shadowNdcXScale &&
                 mGiFlipY == params.shadowFlipY;
    const float* inv = params.shadowInvViewProj;
    float forward[3];
    float length = 0.0f, movement = 0.0f, angle = 0.0f;
    // Centre ray from the exact inverse projection, independent of the shadow/sun eligibility.
    for (int i = 0; i < 3; ++i) {
        forward[i] = (inv[8 + i] * 0.5f + inv[12 + i]) / (inv[11] * 0.5f + inv[15]) -
                     inv[12 + i] / inv[15];
        length += forward[i] * forward[i];
        float delta = params.giCameraEye[i] - mGiEye[i];
        movement += delta * delta;
    }
    length = std::sqrt(std::max(length, 1e-12f));
    for (int i = 0; i < 3; ++i) {
        forward[i] /= length;
        angle += forward[i] * mGiForward[i];
    }
    valid = valid && movement < 60.0f * 60.0f && angle > 0.85f;
    EnsureTarget(mGiSource, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, false);
    EnsureTarget(mGiRaw, hw, hh, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, false);
    for (auto& history : mGiHistory) {
        EnsureTarget(history, hw, hh, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, false);
    }
    if (!mUsable) {
        mGiValid = false;
        return;
    }
    // The AO apply pass leaves a multiplicative blend (and possibly a scissor) enabled.
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mGiSource.fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    Bind(0, mGiSource.tex);
    Bind(1, mDepth.tex);
    // The gather samples the sun shadow map on unit 10 (depth-compare); the filter samples the raw buffer there.
    float lightViewProj[16] = {};
    if (sunShadow) BuildLightViewProj(params, lightViewProj);
    Bind(2, sunShadow ? mShadowMap.tex : mDepth.tex);
    glBindSampler(kFirstUnit + 2, mCmpSampler);
    Bind(3, mGiHistory[mGiRead].tex);
    auto geometry = [&]() {
        glUniform2i(Loc("uSize"), w, h);
        glUniform1f(Loc("uNear"), params.zNear);
        glUniform1f(Loc("uFar"), params.zFar);
        glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, params.shadowInvViewProj);
        glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
        glUniform1f(Loc("uFlipY"), params.shadowFlipY);
    };
    Use(sunShadow && mCsm.count > 1 ? mGiGatherCsm : mGiGather);
    geometry();
    glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, lightViewProj);
    glUniform3fv(Loc("uSunDir"), 1, params.shadowSunDir);
    SetCascadeUniforms();
    glUniform1f(Loc("uNormalBias"), sunShadow && mCsm.count > 1
                                        ? 2.0f * 1500.0f / 4096.0f * 4.0f
                                        : 2.0f * params.shadowRadius / (float)std::max(params.shadowMapSize, 1) * 4.0f);
    glUniform1i(Loc("uSunShadow"), sunShadow ? 1 : 0);
    glUniform1i(Loc("uFrame"), (int)(frameIndex & 63u));
    Draw(mGiRaw);
    glBindSampler(kFirstUnit + 2, 0);
    Bind(2, mGiRaw.tex);
    const int write = 1 - mGiRead;
    Use(mGiFilter);
    geometry();
    glUniformMatrix4fv(Loc("uPreviousViewProj"), 1, GL_FALSE, mGiViewProj);
    glUniform2f(Loc("uPreviousScale"), mGiNdcXScale, mGiFlipY);
    glUniform1i(Loc("uHistoryValid"), valid);
    Draw(mGiHistory[write]);
    Bind(3, mGiHistory[write].tex);
    Use(mGiApply);
    geometry();
    // With world-space GI on, the screen-space term stays as near-field detail (contact colour, actors) at half weight.
    glUniform1f(Loc("uStrength"), params.giStrength * (params.giWorld ? 0.5f : 1.0f));
    const float debug = static_cast<float>(CvInt("gNewDawn.GI.Debug", 0)); // 0 off, N: show only bounce * N
    glUniform1f(Loc("uDebug"), debug > 0.0f ? debug : 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, debug > 0.0f ? GL_ZERO : GL_ONE);
    glBlendEquation(GL_FUNC_ADD);
    Draw(fb.fbo, w, h);
    glDisable(GL_BLEND);
    mGiRead = write;
    mGiFrame = frameIndex;
    mGiScene = params.giSceneId;
    mGiNdcXScale = params.shadowNdcXScale;
    mGiFlipY = params.shadowFlipY;
    std::copy_n(params.shadowViewProj, 16, mGiViewProj);
    std::copy_n(params.giCameraEye, 3, mGiEye);
    std::copy_n(forward, 3, mGiForward);
    mGiValid = true;
    CheckError("screen-space GI");
}

// The CPU half of the local lights: the game's lights plus the VFX flames, smoothed and flickered, into mLocalPos and
// mLocalRgb. No GL calls, so the point-shadow pass can pick its lights from the result before the lights are drawn.
void PostFxGL::PrepareLocalLights(const NewDawnPostFxParams& params, uint32_t frameIndex) {
    constexpr int kMax = 24;
    float (&pos)[kMax][4] = mLocalPos;
    float (&rgb)[kMax][3] = mLocalRgb;
    bool (&fairy)[kMax] = mLocalFairy;
    int& count = mLocalCount;
    std::fill(&pos[0][0], &pos[0][0] + kMax * 4, 0.0f);
    std::fill(&rgb[0][0], &rgb[0][0] + kMax * 3, 0.0f);
    std::fill(fairy, fairy + kMax, false);
    std::fill(mLocalFlame, mLocalFlame + kMax, false);
    count = 0;
    static const auto epoch = std::chrono::steady_clock::now();
    const float seconds =
        static_cast<float>(std::fmod(std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count(), 3600.0));
    // Render-only flicker: a slow drift plus two faster, weaker terms (all under ~5 Hz), keyed on the light's quantised
    // position and the wall clock. Warm (fire) lights flicker up to ~18%, neutral lamps barely. No game data changes.
    // The game re-rolls some lights every tick (a torch sets its intensity to 128 + rand * 127 at 20 Hz): white noise of
    // up to +-33% that read as strobing on every surface and character the light reaches. Low-pass each light's
    // colour and radius over ~0.6 s of wall-clock time, keyed on its position, before the designed flicker below.
    static std::unordered_map<uint64_t, std::array<float, 5>> smoothed; // r, g, b, radius, last seen
    static double lastNow = 0.0;
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
    const bool smoothLights = CvInt("gNewDawn.Lighting.LocalSmooth", 1) != 0; // 0: the unfiltered game values (A/B)
    const float blend =
        smoothLights ? 1.0f - std::exp(-static_cast<float>(std::clamp(now - lastNow, 0.0, 0.5)) / 0.6f) : 1.0f;
    lastNow = now;
    // Flames (gNewDawn.Lighting.FlameLight): the VFX flames recorded last frame, with their world height. A light that sits
    // on a flame (the game's own torch light, or the one tied to a flame the game gives none) is a flame light: it gets
    // the flame's reach, a warm colour and more intensity, so torches and braziers throw real light pools. The global
    // darkening, the floor and every other light are untouched.
    struct FlameInfo {
        float pos[3];
        float height; // world half-height of the flame sprite
    };
    std::vector<FlameInfo> flameList;
    const bool flameLight = CvInt("gNewDawn.Lighting.FlameLight", 0) != 0;
    const float flameGain = std::clamp(CvFloat("gNewDawn.Lighting.FlameStrength", 1.0f), 0.0f, 4.0f);
    const float flameReach = std::clamp(CvFloat("gNewDawn.Lighting.FlameRange", 1.0f), 0.25f, 3.0f);
    const bool flamesFresh = params.vfxEnabled && mFlameScene == params.giSceneId && frameIndex - mFlameFrame <= 2u;
    if (flameLight && flamesFresh) {
        const float tanHalf = std::tan(params.fovyDegrees * 0.00872664626f);
        for (size_t i = 0; i + NEWDAWN_VFX_FLOATS <= mFlameLights.size(); i += NEWDAWN_VFX_FLOATS) {
            const float* f = &mFlameLights[i];
            flameList.push_back({ { f[12], f[13], f[14] }, f[6] * f[4] * tanHalf });
        }
    }
    if (mStats && !flameList.empty()) {
        static int logged = 0;
        if (logged++ % 300 == 0) {
            for (const FlameInfo& fl : flameList)
                SPDLOG_INFO("NewDawnFlameLight: flame ({:.0f},{:.0f},{:.0f}) half-height {:.1f}", fl.pos[0], fl.pos[1], fl.pos[2], fl.height);
            for (int i = 0; i < params.localCount; ++i)
                SPDLOG_INFO("NewDawnFlameLight: game light {} ({:.0f},{:.0f},{:.0f}) radius {:.0f} rgb {:.2f},{:.2f},{:.2f}", i,
                            params.localLights[i][0], params.localLights[i][1], params.localLights[i][2], params.localLights[i][3],
                            params.localColors[i][0], params.localColors[i][1], params.localColors[i][2]);
        }
    }
    auto add = [&](const float* p, float radiusIn, const float* colIn, float boost, float flameH = 0.0f) {
        if (count >= kMax)
            return;
        const uint64_t key = (static_cast<uint64_t>(static_cast<int32_t>(std::floor(p[0] / 8.0f))) & 0xFFFFF) |
                             ((static_cast<uint64_t>(static_cast<int32_t>(std::floor(p[1] / 8.0f))) & 0xFFFFF) << 20) |
                             ((static_cast<uint64_t>(static_cast<int32_t>(std::floor(p[2] / 8.0f))) & 0xFFFFF) << 40);
        auto& st = smoothed[key];
        if (now - st[4] > 1.0) { // new, or gone for a while: start at the current value
            st = { colIn[0], colIn[1], colIn[2], radiusIn, 0.0f };
        } else {
            for (int i = 0; i < 3; ++i)
                st[i] += (colIn[i] - st[i]) * blend;
            st[3] += (radiusIn - st[3]) * blend;
        }
        st[4] = static_cast<float>(now);
        const float radius = st[3];
        const float col[3] = { st[0], st[1], st[2] };
        const float m = std::max(col[0], std::max(col[1], col[2]));
        if (m <= 0.0f)
            return;
        // Keep the light's own hue; compress its brightness so dim game lights still contribute.
        const float k = (0.5f + 0.5f * m) / m;
        // Warm the yellow game colours a little, then pull 25% of the saturation toward white so a big flame never
        // washes a floor mustard.
        float lightRgb[3] = { col[0], col[1] * (1.0f - 0.3f * std::clamp((col[0] - col[2]) / std::max(col[0], 0.01f), 0.0f, 1.0f)), col[2] };
        const float lightLum = 0.2126f * lightRgb[0] + 0.7152f * lightRgb[1] + 0.0722f * lightRgb[2];
        for (float& c : lightRgb)
            c = c + (lightLum - c) * 0.25f;
        const float warm = std::clamp((col[0] - col[2]) / std::max(col[0], 0.01f), 0.0f, 1.0f);
        const float seed = std::fmod(std::floor(p[0] / 4.0f) * 12.9898f + std::floor(p[1] / 4.0f) * 78.233f +
                                         std::floor(p[2] / 4.0f) * 37.719f,
                                     1000.0f);
        const float amp = std::clamp(params.localFlicker, 0.0f, 1.0f) * (smoothLights ? 0.02f + 0.10f * warm : 0.03f + 0.15f * warm);
        const float f = 1.0f + amp * (0.60f * std::sin(seconds * 8.2f + seed) +
                                      0.30f * std::sin(seconds * 18.2f + seed * 1.7f + 1.0f) +
                                      0.10f * std::sin(seconds * 32.0f + seed * 2.3f + 2.0f));
        fairy[count] = boost < 0.99f; // Tatl and other fairies: moving glows, never shadow-casting lights
        mLocalFlame[count] = flameH > 0.0f && boost >= 0.99f;
        pos[count][0] = p[0], pos[count][1] = p[1], pos[count][2] = p[2];
        pos[count][3] = std::clamp(radius * 1.3f, 90.0f, 700.0f);
        float gain = 1.0f;
        if (flameH > 0.0f && boost >= 0.99f) {
            // Reach follows the flame's size (a candle a few steps, a brazier a few body lengths); the colour is the
            // game's own pulled toward a warm fire colour; intensity is raised for fires only.
            pos[count][3] = std::clamp(std::clamp(flameH * 9.0f, 130.0f, 260.0f) * flameReach, 90.0f, 700.0f); // the flame's own size, not the game's (large, soft) light radius
            const float warmFire[3] = { 1.0f, 0.52f, 0.20f };
            const float lum = 0.2126f * lightRgb[0] + 0.7152f * lightRgb[1] + 0.0722f * lightRgb[2];
            for (int i = 0; i < 3; ++i)
                lightRgb[i] = lightRgb[i] * 0.2f + warmFire[i] * (lum / 0.60f) * 0.8f; // 0.60: the luminance of warmFire
            gain = 7.5f * flameGain;
        }
        for (int i = 0; i < 3; ++i) {
            rgb[count][i] = lightRgb[i] * k * f * boost * gain * params.localStrength * params.localGain * 4.0f;
            mWgi.lightRgb[count][i] = rgb[count][i] / f; // world GI injects the light without the designed flicker
        }
        std::copy_n(pos[count], 4, mWgi.lightPos[count]);
        ++count;
    };
    for (int i = 0; i < params.localCount && i < NEWDAWN_LOCAL_LIGHTS; ++i) {
        float flameH = 0.0f;
        for (const FlameInfo& fl : flameList) {
            float d2 = 0.0f;
            for (int a = 0; a < 3; ++a)
                d2 += (params.localLights[i][a] - fl.pos[a]) * (params.localLights[i][a] - fl.pos[a]);
            if (d2 < 90.0f * 90.0f)
                flameH = std::max(flameH, fl.height);
        }
        add(params.localLights[i], params.localLights[i][3], params.localColors[i], params.localBoost[i], flameH);
    }
    // VFX flames the game gives no light to (recorded last frame): tie a warm light to each, nearest first.
    if (params.vfxEnabled && mFlameScene == params.giSceneId && frameIndex - mFlameFrame <= 2u) {
        struct Flame {
            const float* f;
            float score;
        };
        std::vector<Flame> flames;
        for (size_t i = 0; i + NEWDAWN_VFX_FLOATS <= mFlameLights.size(); i += NEWDAWN_VFX_FLOATS) {
            const float* f = &mFlameLights[i];
            bool covered = false;
            for (int j = 0; j < params.localCount && !covered; ++j) {
                float d2 = 0.0f;
                for (int a = 0; a < 3; ++a)
                    d2 += (params.localLights[j][a] - f[12 + a]) * (params.localLights[j][a] - f[12 + a]);
                covered = d2 < 90.0f * 90.0f;
            }
            if (covered)
                continue;
            float d2 = 0.0f;
            for (int a = 0; a < 3; ++a)
                d2 += (params.giCameraEye[a] - f[12 + a]) * (params.giCameraEye[a] - f[12 + a]);
            flames.push_back({ f, d2 });
        }
        std::sort(flames.begin(), flames.end(), [](const Flame& a, const Flame& b) { return a.score < b.score; });
        for (const Flame& fl : flames) {
            if (count >= kMax)
                break;
            bool dup = false; // a flame drawn as several quads is one light
            for (int j = params.localCount; j < count && !dup; ++j) {
                float d2 = 0.0f;
                for (int a = 0; a < 3; ++a)
                    d2 += (pos[j][a] - fl.f[12 + a]) * (pos[j][a] - fl.f[12 + a]);
                dup = d2 < 40.0f * 40.0f;
            }
            if (dup)
                continue;
            const float warmRgb[3] = { 1.0f, 0.62f, 0.32f };
            float col[3];
            for (int a = 0; a < 3; ++a)
                col[a] = warmRgb[a];
            float h = 0.0f;
            for (const FlameInfo& f2 : flameList) {
                float d2 = 0.0f;
                for (int a = 0; a < 3; ++a)
                    d2 += (fl.f[12 + a] - f2.pos[a]) * (fl.f[12 + a] - f2.pos[a]);
                if (d2 < 4.0f * 4.0f)
                    h = f2.height;
            }
            add(fl.f + 12, 150.0f, col, 1.0f, h);
        }
    }
    // Emissive fixtures (lamps, signs, lit windows) the interpreter recorded: a matching light each. Negative intensity
    // marks night-only glass.
    for (int i = 0; i < params.emissiveLightCount && i < NEWDAWN_EMISSIVE_LIGHTS; ++i) {
        const float* l = params.emissiveLights[i];
        const float intensity = l[7] < 0.0f ? -l[7] * params.emissiveNight : l[7] * (0.35f + 0.65f * params.emissiveNight);
        if (!params.emissiveEnabled || intensity < 0.02f)
            continue;
        const float col[3] = { l[4], l[5], l[6] };
        add(l, l[3], col, intensity * params.emissiveStrength);
    }
    for (auto it = smoothed.begin(); it != smoothed.end();)
        it = (now - it->second[4] > 2.0) ? smoothed.erase(it) : std::next(it);
    mWgi.lightCount = count;
    mWgi.lightFrame = frameIndex;
}

void PostFxGL::RunLocalLights(const NewDawnPostFxParams& params, const FramebufferOGL& fb, uint32_t frameIndex) {
    constexpr int kMax = 24;
    const float (&pos)[kMax][4] = mLocalPos;
    const float (&rgb)[kMax][3] = mLocalRgb;
    const int count = mLocalCount;
    const bool smoothLights = CvInt("gNewDawn.Lighting.LocalSmooth", 1) != 0; // 0: the unfiltered game values (A/B)
    if (count == 0 && params.localDarken <= 0.0f)
        return;
    if (!mUsable)
        return;
    const int w = fb.width, h = fb.height;
    EnsureTarget(mLocalSource, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, false);
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mLocalSource.fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    Bind(0, mLocalSource.tex);
    Bind(1, mDepth.tex);
    Use(mLocalLight);
    glUniform2i(Loc("uSize"), w, h);
    glUniform1f(Loc("uNear"), params.zNear);
    glUniform1f(Loc("uFar"), params.zFar);
    glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, params.shadowInvViewProj);
    glUniformMatrix4fv(Loc("uViewProj"), 1, GL_FALSE, params.shadowViewProj);
    glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), params.shadowFlipY);
    glUniform1i(Loc("uLightCount"), count);
    {
        const int units[2] = { kPointShadowUnit0, kPointShadowUnit1 };
        for (int k = 0; k < 2; ++k) {
            const PointShadow& ps = mPointShadow[k];
            const bool on = ps.light >= 0 && ps.valid;
            glActiveTexture(GL_TEXTURE0 + units[k]);
            glBindTexture(GL_TEXTURE_CUBE_MAP, on ? ps.tex : mPointShadowDummy);
            glUniform1i(Loc(k == 0 ? "uPsIdx0" : "uPsIdx1"), on ? ps.light : -1);
            glUniform4f(Loc(k == 0 ? "uPsLight0" : "uPsLight1"), ps.pos[0], ps.pos[1], ps.pos[2], on ? ps.range : 0.0f);
        }
        const float soft = std::clamp(CvFloat("gNewDawn.Shadows.PointSoftness", 1.0f), 0.0f, 3.0f);
        const int size = mPointShadow[0].size > 0 ? mPointShadow[0].size : 256;
        // light radius, cube texel angle (a 90 degree face), bias scale
        glUniform3f(Loc("uPsParams"), 11.0f * soft, 2.0f / (float)size, 1.0f);
        glUniform1f(Loc("uPsDebug"), (float)CvInt("gNewDawn.Shadows.PointDebug", 0));
    }
    glUniform4fv(Loc("uLight"), kMax, &pos[0][0]);
    glUniform3fv(Loc("uLightRgb"), kMax, &rgb[0][0]);
    {
        float flame[kMax];
        for (int i = 0; i < kMax; ++i)
            flame[i] = mLocalFlame[i] ? 1.0f : 0.0f;
        glUniform1fv(Loc("uLightFlame"), kMax, flame);
        const float lift = std::clamp(CvFloat("gNewDawn.Lighting.FlameAlbedo", 0.5f), 0.0f, 1.0f); // 0: none
        glUniform1f(Loc("uFlameExp"), 1.0f - 0.4f * lift);
        glUniform1f(Loc("uFlameScale"), 1.0f + 0.3f * lift);
    }
    const float debug = static_cast<float>(CvInt("gNewDawn.Lighting.LocalDebug", 0)); // N: show only the light * N
    glUniform1f(Loc("uDebug"), debug);
    float fogMul, fogOff;
    FogCoefficients(params, fogMul, fogOff);
    // result = light + framebuffer * factor, evaluated per MSAA sample, so edges keep their anti-aliasing.
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_SRC1_COLOR);
    glBlendEquation(GL_FUNC_ADD);
    // World first, then the stencil-marked characters (enemies, NPCs, Link, props) with only 40% of the darkening, so
    // they stay readable in a dark room or street.
    const float darken = std::clamp(params.localDarken, 0.0f, 0.9f);
    glEnable(GL_STENCIL_TEST);
    glStencilMask(0);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glStencilFunc(GL_EQUAL, 0, 0x0D); // not a character (1), not emissive glass (4) or a lamp or sign (8)
    glUniform1f(Loc("uDarken"), darken);
    glUniform1f(Loc("uFogMul"), fogMul);
    glUniform1f(Loc("uFogOff"), fogOff);
    glUniform1f(Loc("uSoft"), 0.0f);
    Draw(fb.fbo, w, h);
    glStencilFunc(GL_EQUAL, 1, 0xFF);
    glUniform1f(Loc("uDarken"), darken * 0.4f);
    glUniform1f(Loc("uSoft"), smoothLights ? 1.0f : 0.0f);
    Draw(fb.fbo, w, h);
    glStencilMask(0xFF);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_BLEND);
    for (int unit : { kPointShadowUnit0, kPointShadowUnit1 }) {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_CUBE_MAP, mPointShadowDummy);
    }
    if (debug > 0.0f) {
        static int logged = 0;
        if (logged++ % 400 < 1) {
            SPDLOG_INFO("NewDawnLocalLights: {} lights ({} game, rest flames)", count, params.localCount);
            for (int i = 0; i < count; ++i)
                SPDLOG_INFO("  light {} pos {:.0f},{:.0f},{:.0f} range {:.0f} rgb {:.2f},{:.2f},{:.2f}", i, pos[i][0],
                            pos[i][1], pos[i][2], pos[i][3], rgb[i][0], rgb[i][1], rgb[i][2]);
        }
    }
    CheckError("local lights");
}

// Shader-ready opening data: centre + half width, inward normal + half height, horizontal axis + pane counts, beam direction.
void PackInteriorWindows(const NewDawnPostFxParams& params, float (*win0)[4], float (*win1)[4], float (*win2)[4],
                         float (*beam)[4]) {
    const int count = std::min<int>(params.interiorWindowCount, NEWDAWN_INTERIOR_WINDOWS);
    for (int i = 0; i < NEWDAWN_INTERIOR_WINDOWS; ++i)
        win0[i][0] = win0[i][1] = win0[i][2] = win0[i][3] = win1[i][0] = win1[i][1] = win1[i][2] = win1[i][3] =
            win2[i][0] = win2[i][1] = win2[i][2] = win2[i][3] = beam[i][0] = beam[i][1] = beam[i][2] = beam[i][3] = 0.0f;
    const float sinE = std::clamp(params.interiorSinElevation, 0.1f, 1.0f);
    const float cosE = std::sqrt(1.0f - sinE * sinE);
    for (int i = 0; i < count; ++i) {
        const float* w = params.interiorWindows[i];
        for (int a = 0; a < 3; ++a) {
            win0[i][a] = w[a];
            win1[i][a] = w[3 + a];
            win2[i][a] = w[6 + a];
        }
        win0[i][3] = w[9];
        win1[i][3] = w[10];
        win2[i][3] = w[11] + 16.0f * w[12];
        // The beam keeps the opening's horizontal heading and takes the sun's elevation; a skylight (normal pointing
        // down) just drops straight in.
        const float hx = w[3], hz = w[5];
        const float hl = std::sqrt(hx * hx + hz * hz);
        float L[3];
        if (hl < 0.3f) {
            L[0] = w[3] * 0.2f, L[1] = -1.0f, L[2] = w[5] * 0.2f;
        } else {
            L[0] = hx / hl * cosE, L[1] = -sinE, L[2] = hz / hl * cosE;
        }
        const float ll = std::sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
        for (int a = 0; a < 3; ++a)
            beam[i][a] = L[a] / ll;
    }
}

// Developer aid for placing the per-scene window table: logs the world position under a 9 x 5 grid of screen points.
void PostFxGL::ProbeInterior(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    static uint32_t frame = 0;
    if (frame++ % 150 != 5)
        return;
    const int w = fb.width, h = fb.height;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, mDepth.fbo);
    for (int gy = 0; gy < 5; ++gy) {
        std::string row;
        for (int gx = 0; gx < 9; ++gx) {
            const int px = (gx * 2 + 1) * w / 18, py = (gy * 2 + 1) * h / 10;
            float depth = 1.0f;
            glReadPixels(px, py, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
            const float uvx = (px + 0.5f) / w, uvy = (py + 0.5f) / h;
            const float in[4] = { (uvx * 2 - 1) * params.shadowNdcXScale, (uvy * 2 - 1) * params.shadowFlipY, depth * 2 - 1, 1.0f };
            float out[4] = {};
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    out[i] += params.shadowInvViewProj[j * 4 + i] * in[j];
            char buf[64];
            snprintf(buf, sizeof(buf), " (%.0f,%.0f,%.0f)", out[0] / out[3], out[1] / out[3], out[2] / out[3]);
            row += buf;
        }
        SPDLOG_INFO("INTPROBE row {} (screen top first, y flipped {}){}", gy, params.shadowFlipY, row);
    }
}

void PostFxGL::RunInteriorLight(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    constexpr int kMax = NEWDAWN_INTERIOR_WINDOWS;
    const int count = std::min<int>(params.interiorWindowCount, kMax);
    if (count <= 0 || !mUsable)
        return;
    float win0[kMax][4], win1[kMax][4], win2[kMax][4], beam[kMax][4];
    PackInteriorWindows(params, win0, win1, win2, beam);
    const int w = fb.width, h = fb.height;
    EnsureTarget(mLocalSource, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, false);
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mLocalSource.fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    Bind(0, mLocalSource.tex);
    Bind(1, mDepth.tex);
    Use(mInteriorLight);
    glUniform2i(Loc("uSize"), w, h);
    glUniform1f(Loc("uNear"), params.zNear);
    glUniform1f(Loc("uFar"), params.zFar);
    glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, params.shadowInvViewProj);
    glUniformMatrix4fv(Loc("uViewProj"), 1, GL_FALSE, params.shadowViewProj);
    glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), params.shadowFlipY);
    glUniform3fv(Loc("uEye"), 1, params.giCameraEye);
    glUniform1i(Loc("uCount"), count);
    glUniform4fv(Loc("uWin0"), kMax, &win0[0][0]);
    glUniform4fv(Loc("uWin1"), kMax, &win1[0][0]);
    glUniform4fv(Loc("uWin2"), kMax, &win2[0][0]);
    glUniform4fv(Loc("uWinL"), kMax, &beam[0][0]);
    const float strength = params.interiorStrength;
    const float rgb[3] = { params.interiorColor[0] * strength, params.interiorColor[1] * strength,
                           params.interiorColor[2] * strength };
    glUniform3fv(Loc("uColor"), 1, rgb);
    glUniform1f(Loc("uPool"), static_cast<float>(CvInt("gNewDawn.Interior.Pools", 1)));
    glUniform1f(Loc("uShaft"), static_cast<float>(CvInt("gNewDawn.Interior.Shafts", 1)));
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_SRC1_COLOR);
    glBlendEquation(GL_FUNC_ADD);
    Draw(fb.fbo, w, h);
    glDisable(GL_BLEND);
    CheckError("interior window light");

    // The panes of the render-only windows (kind 2), after the light so they are not lit twice.
    if (mInteriorGlass.id && CvInt("gNewDawn.Interior.Panes", 1)) {
        bool any = false;
        for (int i = 0; i < count; ++i)
            any |= params.interiorWindows[i][13] == 2.0f;
        if (any) {
            Use(mInteriorGlass);
            glUniformMatrix4fv(Loc("uViewProj"), 1, GL_FALSE, params.shadowViewProj);
            glUniform1f(Loc("uNdcXScale"), params.shadowNdcXScale);
            glUniform1f(Loc("uFlipY"), params.shadowFlipY);
            // Only the virtual windows are drawn: pack them to the front.
            float v0[kMax][4] = {}, v1[kMax][4] = {}, v2[kMax][4] = {};
            int drawn = 0;
            for (int i = 0; i < count; ++i) {
                if (params.interiorWindows[i][13] != 2.0f)
                    continue;
                memcpy(v0[drawn], win0[i], sizeof(v0[drawn]));
                memcpy(v1[drawn], win1[i], sizeof(v1[drawn]));
                memcpy(v2[drawn], win2[i], sizeof(v2[drawn]));
                ++drawn;
            }
            glUniform4fv(Loc("uWin0"), kMax, &v0[0][0]);
            glUniform4fv(Loc("uWin1"), kMax, &v1[0][0]);
            glUniform4fv(Loc("uWin2"), kMax, &v2[0][0]);
            // Sky brightness behind the glass: the light colour scaled up (a window is far brighter than the room it lights).
            const float sky = 2.4f;
            const float night[3] = { 0.010f, 0.016f, 0.034f };
            const float glass[3] = { night[0] + params.interiorColor[0] * sky, night[1] + params.interiorColor[1] * sky,
                                     night[2] + params.interiorColor[2] * sky };
            glUniform3fv(Loc("uGlass"), 1, glass);
            const float frame[3] = { 0.075f, 0.043f, 0.020f };
            glUniform3fv(Loc("uFrame"), 1, frame);
            glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
            glViewport(0, 0, w, h);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LEQUAL);
            glDepthMask(GL_FALSE);
            glDisable(GL_CULL_FACE);
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, drawn);
            glDepthMask(GL_TRUE);
            glDisable(GL_DEPTH_TEST);
            CheckError("interior window glass");
        }
    }
}

void PostFxGL::RunEmissive(const NewDawnPostFxParams& params, const FramebufferOGL& fb) {
    if (!mUsable)
        return;
    const int w = fb.width, h = fb.height;
    EnsureTarget(mLocalSource, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, false);
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mLocalSource.fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    Bind(0, mLocalSource.tex);
    Use(mEmissive);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);
    glEnable(GL_STENCIL_TEST);
    glStencilMask(0);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    const float night = std::clamp(params.emissiveNight, 0.0f, 1.0f);
    const float strength = params.emissiveStrength;
    // Night glass (stencil 4): a warm interior light behind the pane, only after dusk.
    if (night > 0.01f) {
        const float tint[3] = { 1.0f, 0.62f, 0.28f };
        glUniform3fv(Loc("uTint"), 1, tint);
        glUniform1f(Loc("uGain"), 1.1f * night * strength);
        glUniform1i(Loc("uClass"), 4);
        glStencilFunc(GL_EQUAL, 4, 0xFF);
        Draw(fb.fbo, w, h);
    }
    // Lamps and signs (stencil 8): always lit, brighter at night.
    {
        const float tint[3] = { 1.0f, 0.92f, 0.78f };
        glUniform3fv(Loc("uTint"), 1, tint);
        glUniform1f(Loc("uGain"), (0.55f + 1.1f * night) * strength);
        glUniform1i(Loc("uClass"), 8);
        glStencilFunc(GL_EQUAL, 8, 0xFF);
        Draw(fb.fbo, w, h);
    }
    glStencilMask(0xFF);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_BLEND);
    CheckError("emissive");
}

// Point-light shadows. Picks the one or two lights that matter most around the player (bright, close, not a moving
// fairy), culls the frame's recorded casters to each light's range, and, only when that caster set or the light
// changed, renders them into a distance cube map. The cube is sampled by the local-light pass. Nothing here touches
// the game's data, and it only ever removes light.
void PostFxGL::RunPointShadows(const NewDawnPostFxParams& params) {
    const int want = std::clamp(params.localShadows, 0, 2);
    for (auto& ps : mPointShadow)
        ps.light = -1;
    if (want == 0 || mCasterPositions == nullptr || mCasterVertexCount < 3 || mLocalCount == 0 || !mUsable) {
        return;
    }
    const int size = std::clamp(CvInt("gNewDawn.Shadows.PointSize", 256), 64, 1024);

    // Candidates: nearest to the player, scaled by brightness and range; a light already holding a slot gets a margin so
    // two similar lights never swap every frame.
    struct Candidate {
        int index;
        float score;
    };
    std::vector<Candidate> candidates;
    for (int i = 0; i < mLocalCount; ++i) {
        if (mLocalFairy[i])
            continue;
        const float range = mLocalPos[i][3];
        const float lum = 0.2126f * mLocalRgb[i][0] + 0.7152f * mLocalRgb[i][1] + 0.0722f * mLocalRgb[i][2];
        if (lum < 0.05f)
            continue;
        float d2 = 0.0f;
        for (int a = 0; a < 3; ++a)
            d2 += (mLocalPos[i][a] - params.localFocus[a]) * (mLocalPos[i][a] - params.localFocus[a]);
        const float d = std::sqrt(d2);
        if (d > range + 500.0f)
            continue;
        float score = range / std::max(d, 40.0f);
        for (const auto& ps : mPointShadow) {
            if (ps.assigned) {
                float e2 = 0.0f;
                for (int a = 0; a < 3; ++a)
                    e2 += (mLocalPos[i][a] - ps.pos[a]) * (mLocalPos[i][a] - ps.pos[a]);
                if (e2 < 12.0f * 12.0f)
                    score *= 1.5f;
            }
        }
        candidates.push_back({ i, score });
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    if ((int)candidates.size() > want)
        candidates.resize(want);

    // A light that already owns a slot keeps it; the others take the free slots.
    int slotOf[2] = { -1, -1 };
    bool taken[2] = {};
    for (size_t c = 0; c < candidates.size(); ++c) {
        const int i = candidates[c].index;
        for (int k = 0; k < 2; ++k) {
            if (taken[k] || !mPointShadow[k].assigned)
                continue;
            float e2 = 0.0f;
            for (int a = 0; a < 3; ++a)
                e2 += (mLocalPos[i][a] - mPointShadow[k].pos[a]) * (mLocalPos[i][a] - mPointShadow[k].pos[a]);
            if (e2 < 12.0f * 12.0f) {
                slotOf[c] = k;
                taken[k] = true;
                break;
            }
        }
    }
    for (size_t c = 0; c < candidates.size(); ++c) {
        if (slotOf[c] >= 0)
            continue;
        for (int k = 0; k < 2; ++k) {
            if (!taken[k]) {
                slotOf[c] = k;
                taken[k] = true;
                mPointShadow[k].valid = false; // a different light: its old cube is meaningless
                break;
            }
        }
    }
    for (int k = 0; k < 2; ++k) {
        if (!taken[k])
            mPointShadow[k].assigned = false;
    }

    GLint previousActive;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActive);
    bool stateSet = false;
    GLboolean depthClamp = GL_FALSE;
    mPointLastRendered = 0;
    mPointLastTriangles = 0;
    const float* tris = mCasterPositions;
    const uint32_t triCount = mCasterVertexCount / 3;
    const auto cpuStart = std::chrono::steady_clock::now();
    // 64-triangle chunks of the recorded casters (consecutive triangles are spatially coherent): their boxes are shared by
    // every light, so they are computed once per frame.
    const uint32_t chunkCount = (triCount + 63) / 64;
    mPointChunkBox.resize((size_t)chunkCount * 6);
    for (uint32_t ch = 0; ch < chunkCount; ++ch) {
        const uint32_t t1 = std::min((ch + 1) * 64u, triCount);
        float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
        for (uint32_t v = ch * 64u * 3; v < t1 * 3; ++v) {
            for (int a = 0; a < 3; ++a) {
                lo[a] = std::min(lo[a], tris[v * 3 + a]);
                hi[a] = std::max(hi[a], tris[v * 3 + a]);
            }
        }
        std::copy(lo, lo + 3, &mPointChunkBox[(size_t)ch * 6]);
        std::copy(hi, hi + 3, &mPointChunkBox[(size_t)ch * 6 + 3]);
    }
    for (size_t c = 0; c < candidates.size(); ++c) {
        const int i = candidates[c].index;
        PointShadow& ps = mPointShadow[slotOf[c]];
        // Cube range: the light's range rounded up to 32 units so the smoothed value does not re-render it each frame.
        const float lightPos[3] = { std::round(mLocalPos[i][0] * 2.0f) * 0.5f, std::round(mLocalPos[i][1] * 2.0f) * 0.5f,
                                    std::round(mLocalPos[i][2] * 2.0f) * 0.5f };
        const float range = std::min(std::ceil(mLocalPos[i][3] / 32.0f) * 32.0f + 32.0f, 800.0f);
        const float r2 = range * range;

        // Cull to the light's sphere: 64-triangle chunks first (their boxes), then each triangle's box.
        mPointCasters.clear();
        auto overlaps = [&](const float* lo, const float* hi) {
            float d2 = 0.0f;
            for (int a = 0; a < 3; ++a) {
                const float e = std::max(std::max(lo[a] - lightPos[a], lightPos[a] - hi[a]), 0.0f);
                d2 += e * e;
            }
            return d2 < r2;
        };
        for (uint32_t ch = 0; ch < chunkCount; ++ch) {
            const uint32_t t0 = ch * 64u, t1 = std::min(t0 + 64u, triCount);
            if (!overlaps(&mPointChunkBox[(size_t)ch * 6], &mPointChunkBox[(size_t)ch * 6 + 3]))
                continue;
            for (uint32_t t = t0; t < t1; ++t) {
                const float* q = tris + t * 9;
                float tlo[3], thi[3];
                for (int a = 0; a < 3; ++a) {
                    tlo[a] = std::min(q[a], std::min(q[3 + a], q[6 + a]));
                    thi[a] = std::max(q[a], std::max(q[3 + a], q[6 + a]));
                }
                if (overlaps(tlo, thi))
                    mPointCasters.insert(mPointCasters.end(), q, q + 9);
            }
        }
        uint64_t hash = 1469598103934665603ull;
        auto mix = [&](const void* data, size_t bytes) {
            const uint8_t* b = static_cast<const uint8_t*>(data);
            for (size_t n = 0; n < bytes; ++n)
                hash = (hash ^ b[n]) * 1099511628211ull;
        };
        mix(mPointCasters.data(), mPointCasters.size() * sizeof(float));
        mix(lightPos, sizeof(lightPos));
        mix(&range, sizeof(range));
        mix(&size, sizeof(size));

        ps.light = i;
        ps.assigned = true;
        for (int a = 0; a < 3; ++a)
            ps.pos[a] = lightPos[a];
        ps.range = range;
        mPointLastTriangles += (int)(mPointCasters.size() / 9);
        if (ps.valid && ps.hash == hash && ps.size == size)
            continue; // nothing near the light moved: keep last frame's cube

        if (!stateSet) {
            stateSet = true;
            glDisable(GL_BLEND);
            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_STENCIL_TEST);
            glDisable(GL_CULL_FACE);
            depthClamp = glIsEnabled(GL_DEPTH_CLAMP); // the game's renderer keeps it on; the near plane must clip here
            glDisable(GL_DEPTH_CLAMP);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glDepthMask(GL_TRUE);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            Use(mPointShadowDepth);
            glBindVertexArray(mPointShadowVao);
        }
        if (ps.tex == 0 || ps.size != size) {
            if (ps.tex == 0) {
                glGenTextures(1, &ps.tex);
                glGenFramebuffers(1, &ps.fbo);
                glGenRenderbuffers(1, &ps.depth);
            }
            glActiveTexture(GL_TEXTURE0 + kPointShadowUnit0);
            glBindTexture(GL_TEXTURE_CUBE_MAP, ps.tex);
            for (int face = 0; face < 6; ++face)
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_R32F, size, size, 0, GL_RED, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_CUBE_MAP, mPointShadowDummy);
            glBindRenderbuffer(GL_RENDERBUFFER, ps.depth);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, size, size);
            glBindFramebuffer(GL_FRAMEBUFFER, ps.fbo);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, ps.depth);
            ps.size = size;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, ps.fbo);
        glViewport(0, 0, size, size);
        glBindBuffer(GL_ARRAY_BUFFER, mPointShadowVbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(mPointCasters.size() * sizeof(float)), mPointCasters.data(),
                     GL_STREAM_DRAW);
        glUniform3fv(Loc("uLight"), 1, lightPos);
        // The six faces of the GL cube map convention: right, up and forward axes of each face.
        static const float kFaces[6][9] = {
            { 0, 0, -1, 0, -1, 0, 1, 0, 0 },  { 0, 0, 1, 0, -1, 0, -1, 0, 0 }, { 1, 0, 0, 0, 0, 1, 0, 1, 0 },
            { 1, 0, 0, 0, 0, -1, 0, -1, 0 },  { 1, 0, 0, 0, -1, 0, 0, 0, 1 },  { -1, 0, 0, 0, -1, 0, 0, 0, -1 },
        };
        // Near plane: the fixture the light is mounted in (a brazier's rim, a lamp's shade) sits within a few tens of units
        // of the flame and must not close the light in; anything nearer than this does not cast.
        const float zNear = std::clamp(CvFloat("gNewDawn.Shadows.PointNear", 24.0f), 1.0f, 100.0f), zFar = range;
        for (int face = 0; face < 6; ++face) {
            const float* r = kFaces[face];
            const float* u = r + 3;
            const float* f = r + 6;
            // clip = P * V, column-major: V rows (right, up, -forward), P a 90 degree perspective.
            const float a = -(zFar + zNear) / (zFar - zNear), b = -2.0f * zFar * zNear / (zFar - zNear);
            float vp[16];
            const float* rows[3] = { r, u, f };
            for (int col = 0; col < 3; ++col) {
                vp[col * 4 + 0] = rows[0][col];
                vp[col * 4 + 1] = rows[1][col];
                vp[col * 4 + 2] = -a * rows[2][col];
                vp[col * 4 + 3] = rows[2][col];
            }
            float tr[3];
            for (int row = 0; row < 3; ++row)
                tr[row] = -(rows[row][0] * lightPos[0] + rows[row][1] * lightPos[1] + rows[row][2] * lightPos[2]);
            vp[12] = tr[0];
            vp[13] = tr[1];
            vp[14] = -a * tr[2] + b;
            vp[15] = tr[2];
            glUniformMatrix4fv(Loc("uVP"), 1, GL_FALSE, vp);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, ps.tex, 0);
            static const GLfloat kFar[4] = { 1.0e6f, 0, 0, 0 };
            glClearBufferfv(GL_COLOR, 0, kFar);
            glClear(GL_DEPTH_BUFFER_BIT);
            if (!mPointCasters.empty())
                glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(mPointCasters.size() / 3));
        }
        ps.hash = hash;
        ps.valid = true;
        mPointLastRendered++;
        if (const char* dumpDir = Ship::Context::GetRawInstance()->GetConsoleVariables()->GetString("gNewDawn.Shadows.PointDumpDir", "");
            dumpDir != nullptr && *dumpDir != '\0' && c == 0) {
            // Measurement/debug fixture: write the six faces of the first cube once as raw R32F (distance, world units).
            static bool dumped = false;
            if (!dumped && !mPointCasters.empty()) {
                dumped = true;
                std::vector<float> face((size_t)size * size);
                glActiveTexture(GL_TEXTURE0 + kPointShadowUnit0);
                glBindTexture(GL_TEXTURE_CUBE_MAP, ps.tex);
                for (int f = 0; f < 6; ++f) {
                    glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, 0, GL_RED, GL_FLOAT, face.data());
                    char name[512];
                    snprintf(name, sizeof(name), "%s/face%d_%d.f32", dumpDir, f, size);
                    if (FILE* fp = fopen(name, "wb")) {
                        fwrite(face.data(), sizeof(float), face.size(), fp);
                        fclose(fp);
                    }
                }
                glBindTexture(GL_TEXTURE_CUBE_MAP, mPointShadowDummy);
                SPDLOG_INFO("NewDawnPointShadows: dumped cube faces, light ({:.1f},{:.1f},{:.1f}) range {:.0f}, {} triangles",
                            lightPos[0], lightPos[1], lightPos[2], range, mPointCasters.size() / 9);
            }
        }
    }
    if (stateSet) {
        glBindVertexArray(mVao);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        if (depthClamp)
            glEnable(GL_DEPTH_CLAMP);
    }
    glActiveTexture(previousActive);
    if (mStats) {
        static int logged = 0;
        static double cpuMs = 0.0;
        cpuMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count();
        if (logged++ % 300 == 299) {
            SPDLOG_INFO("NewDawnPointShadows: CPU {:.3f} ms/frame (culling {} caster triangles per light, 300 frames)",
                        cpuMs / 300.0, triCount);
            cpuMs = 0.0;
        }
        static int logged2 = 0;
        if (logged2++ % 300 == 0) {
            SPDLOG_INFO("NewDawnPointShadows: {} lights, {} caster triangles, {} cube(s) re-rendered this frame",
                        (int)candidates.size(), mPointLastTriangles, mPointLastRendered);
            for (int i = 0; i < mLocalCount; ++i)
                SPDLOG_INFO("  local light {}: ({:.0f},{:.0f},{:.0f}) range {:.0f}{}", i, mLocalPos[i][0], mLocalPos[i][1],
                            mLocalPos[i][2], mLocalPos[i][3], mLocalFairy[i] ? " (fairy)" : (mLocalFlame[i] ? " (flame)" : ""));
        }
    }
    CheckError("point shadows");
}

void PostFxGL::RunAmbientOcclusion(const NewDawnPostFxParams& params, const FramebufferOGL& fb, const Settings& s) {
    const int w = fb.width, h = fb.height;
    // Raw occlusion at half resolution (gNewDawn.PostFx.AO.HalfRes, default 1): the kernel is wide and the two depth-aware
    // blurs below average it over far more than a pixel anyway. Red holds the occlusion, green the sample's linear depth.
    const int aoStep = CvInt("gNewDawn.PostFx.AO.HalfRes", 1) != 0 ? 2 : 1;
    EnsureTarget(mAoRaw, (w + aoStep - 1) / aoStep, (h + aoStep - 1) / aoStep, GL_RG16F, GL_RG, GL_FLOAT, false, false);
    EnsureTarget(mAoBlur, w, h, GL_RG16F, GL_RG, GL_FLOAT, false, false);
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
    glUniform1i(Loc("uStep"), aoStep);
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
    glUniform1i(Loc("uAoShift"), aoStep - 1);
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
    glUniform1i(Loc("uAoShift"), 0);
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

// ---------------------------------------------------------------------------------------------------------------------
// Temporal anti-aliasing

static bool TaaInvert4x4(const float m[16], float out[16]) {
    // General 4x4 inverse (cofactors, double precision): row-major in, row-major out.
    double a[16], inv[16];
    for (int i = 0; i < 16; i++)
        a[i] = m[i];
    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] +
             a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] -
             a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] +
             a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] -
              a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] -
             a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] +
             a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] -
             a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] +
              a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] +
             a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] -
             a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] +
              a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] -
              a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] -
             a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] +
             a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] -
              a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] -
              a[8] * a[2] * a[5];
    const double det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if (std::fabs(det) < 1e-30)
        return false;
    for (int i = 0; i < 16; i++)
        out[i] = (float)(inv[i] / det);
    return true;
}

// Camera position and view axis out of a row-vector view-projection (clip = v * M): clip x, y and w all vanish at the
// eye, and w is the distance along the view axis. Used only to notice camera cuts.
static bool TaaCameraOf(const float m[16], float eye[3], float fwd[3]) {
    const int cols[3] = { 0, 1, 3 };
    double a[3][4];
    for (int i = 0; i < 3; i++) {
        for (int r = 0; r < 3; r++)
            a[i][r] = m[r * 4 + cols[i]];
        a[i][3] = -m[12 + cols[i]];
    }
    for (int c = 0; c < 3; c++) {
        int piv = c;
        for (int r = c + 1; r < 3; r++)
            if (std::fabs(a[r][c]) > std::fabs(a[piv][c]))
                piv = r;
        if (std::fabs(a[piv][c]) < 1e-12)
            return false;
        std::swap_ranges(a[c], a[c] + 4, a[piv]);
        for (int r = 0; r < 3; r++) {
            if (r == c)
                continue;
            const double f = a[r][c] / a[c][c];
            for (int k = c; k < 4; k++)
                a[r][k] -= f * a[c][k];
        }
    }
    for (int i = 0; i < 3; i++)
        eye[i] = (float)(a[i][3] / a[i][i]);
    const float len = std::sqrt(m[3] * m[3] + m[7] * m[7] + m[11] * m[11]);
    if (len < 1e-12f)
        return false;
    fwd[0] = m[3] / len, fwd[1] = m[7] / len, fwd[2] = m[11] / len;
    return true;
}

void PostFxGL::SetTaaCamera(const float* viewProj, float ndcXScale, float flipY, float zNear, float zFar,
                            float jitterPxX, float jitterPxY, uint32_t frameIndex) {
    mTaaCam.jitter[0] = jitterPxX;
    mTaaCam.jitter[1] = jitterPxY;
    std::copy(viewProj, viewProj + 16, mTaaCam.vp);
    mTaaCam.ndcXScale = ndcXScale;
    mTaaCam.flipY = flipY;
    mTaaCam.zNear = zNear > 0.0f ? zNear : 10.0f;
    mTaaCam.zFar = zFar > mTaaCam.zNear ? zFar : 12800.0f;
    mTaaCam.frame = frameIndex;
    mTaaCam.valid = true;
}

const PostFxGL::Target* PostFxGL::RunTaa(const FramebufferOGL& fb, const Settings& s) {
    mTaaDidReset = false;
    if (!mTaaCam.valid || !mTaa.id || !mTaaMaskFill.id || s.debugView != 0) {
        mTaaValid = false;
        return nullptr;
    }
    const int w = fb.width, h = fb.height;
    // The actor mask only exists while the local-light pass marks stencil for this frame's character draws.
    const bool useMask = mLocalMarking;
    EnsureTarget(mTaaHist[0], w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
    EnsureTarget(mTaaHist[1], w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
    EnsureTarget(mTaaOut, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
    EnsureTarget(mTaaDepth, w, h, GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, true, false);
    if (useMask)
        EnsureTarget(mTaaMask, w, h, GL_RG8, GL_RG, GL_UNSIGNED_BYTE, false, false);
    if (!mUsable) {
        return nullptr;
    }

    float eye[3] = {}, fwd[3] = { 0, 0, 1 };
    const bool haveCamera = TaaCameraOf(mTaaCam.vp, eye, fwd);
    bool cut = false;
    if (mTaaValid && haveCamera) {
        const float dx = eye[0] - mTaaPrevEye[0], dy = eye[1] - mTaaPrevEye[1], dz = eye[2] - mTaaPrevEye[2];
        const float dot = fwd[0] * mTaaPrevFwd[0] + fwd[1] * mTaaPrevFwd[1] + fwd[2] * mTaaPrevFwd[2];
        // Hard cut: a jump no continuous camera makes in one rendered frame (400 units, or about 25 degrees)
        cut = dx * dx + dy * dy + dz * dz > 400.0f * 400.0f || dot < 0.9f;
        if (cut && mStats) {
            SPDLOG_INFO("NewDawnPostFx: TAA camera moved {:.0f} units, view axis dot {:.3f}",
                        std::sqrt(dx * dx + dy * dy + dz * dz), dot);
        }
    }
    const bool reset =
        !mTaaValid || !haveCamera || mTaaCam.frame != mTaaLastFrame + 1 || mTaaPrevW != w || mTaaPrevH != h || cut;
    mTaaDidReset = reset;
    if (reset && mStats) {
        SPDLOG_INFO("NewDawnPostFx: TAA history reset ({}), frame {}",
                    !mTaaValid                           ? "no history"
                    : !haveCamera                        ? "no camera"
                    : mTaaCam.frame != mTaaLastFrame + 1 ? "frame gap"
                    : (mTaaPrevW != w || mTaaPrevH != h) ? "resize"
                                                         : "camera cut",
                    mTaaCam.frame);
    }

    // Depth (and the actor stencil) of the finished world, single sample
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mTaaDepth.fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_DEPTH_BUFFER_BIT | (useMask ? GL_STENCIL_BUFFER_BIT : 0), GL_NEAREST);
    CheckError("TAA depth blit");
    if (useMask) {
        glBindFramebuffer(GL_FRAMEBUFFER, mTaaMask.fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, mTaaDepth.tex, 0);
        glViewport(0, 0, w, h);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_STENCIL_TEST);
        glStencilMask(0);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        Use(mTaaMaskFill);
        // Stencil 1 (characters) lands in red, stencil 2 (grass tufts) in green.
        glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
        glStencilFunc(GL_EQUAL, 1, 0xFF);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_FALSE);
        glStencilFunc(GL_EQUAL, 2, 0xFF);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDisable(GL_STENCIL_TEST);
        glStencilMask(0xFF);
        glStencilFunc(GL_ALWAYS, 0, 0xFF);
        CheckError("TAA actor and grass mask");
    }

    float invVP[16];
    if (!TaaInvert4x4(mTaaCam.vp, invVP)) {
        mTaaValid = false;
        return nullptr;
    }
    const int write = 1 - mTaaRead;
    glBindFramebuffer(GL_FRAMEBUFFER, mTaaHist[write].fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, mTaaOut.tex, 0);
    const GLenum buffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, buffers);
    Bind(0, mScene.tex);
    Bind(1, mTaaDepth.tex);
    Bind(2, mTaaHist[mTaaRead].tex);
    Bind(3, useMask ? mTaaMask.tex : mFogNeutral);
    Use(mTaa);
    glUniform1i(Loc("uScene"), kFirstUnit + 0);
    glUniform1i(Loc("uDepth"), kFirstUnit + 1);
    glUniform1i(Loc("uTaaHistory"), kFirstUnit + 2);
    glUniform1i(Loc("uTaaMask"), kFirstUnit + 3);
    glUniform2i(Loc("uSize"), w, h);
    glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, invVP);
    glUniformMatrix4fv(Loc("uPrevViewProj"), 1, GL_FALSE, reset ? mTaaCam.vp : mTaaPrevVP);
    glUniform2f(Loc("uNdc"), mTaaCam.ndcXScale, mTaaCam.flipY);
    glUniform2f(Loc("uJitter"), mTaaCam.jitter[0], mTaaCam.jitter[1]);
    glUniform1f(Loc("uNear"), mTaaCam.zNear);
    glUniform1f(Loc("uFar"), mTaaCam.zFar);
    glUniform1f(Loc("uFeedback"), std::clamp(CvFloat("gNewDawn.TAA.Feedback", 0.9f), 0.0f, 0.98f));
    glUniform1i(Loc("uReset"), reset ? 1 : 0);
    glUniform1i(Loc("uUseMask"), useMask ? 1 : 0);
    // Tuft pixels (stencil 2) are thin swaying blades that no reprojection can follow, and the soft Gaussian that suits
    // surfaces blurs them: history weight capped at 0.85, a narrower current-frame filter (exponent 5 against 2.29),
    // and a full-strength sharpen below. Tuned on Termina and Swamp grass against the 2x2-averaged gradient of TAA off.
    glUniform4f(Loc("uGrassTune"), 0.85f, 1.1f, 5.0f, 0.0f);
    glUniform1i(Loc("uDebug"), CvInt("gNewDawn.TAA.Debug", 0));
    Draw(mTaaHist[write]);
    CheckError("TAA resolve");

    mTaaRead = write;
    mTaaValid = true;
    mTaaLastFrame = mTaaCam.frame;
    mTaaPrevW = w;
    mTaaPrevH = h;
    std::copy(mTaaCam.vp, mTaaCam.vp + 16, mTaaPrevVP);
    std::copy(eye, eye + 3, mTaaPrevEye);
    std::copy(fwd, fwd + 3, mTaaPrevFwd);
    // The history keeps the unsharpened image; only what goes on to bloom and the display transform is sharpened.
    const float sharpen = std::clamp(CvFloat("gNewDawn.TAA.Sharpen", 0.7f), 0.0f, 1.0f);
    if (sharpen > 0.0f && mTaaSharpen.id) {
        EnsureTarget(mTaaSharp, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
        if (!mUsable) {
            return &mTaaOut;
        }
        Bind(0, mTaaOut.tex);
        Bind(3, useMask ? mTaaMask.tex : mFogNeutral);
        Use(mTaaSharpen);
        glUniform1i(Loc("uScene"), kFirstUnit + 0);
        glUniform1i(Loc("uTaaMask"), kFirstUnit + 3);
        glUniform1i(Loc("uUseMask"), useMask ? 1 : 0);
        glUniform2i(Loc("uSize"), w, h);
        glUniform1f(Loc("uSharpen"), sharpen);
        // Grass gets the full sharpen; the fade keeps true darks (perceptual luma below about 0.04, display below
        // ~0.05) untouched and reaches full strength by 0.14 (display ~0.15), so mid-dark grass is sharpened.
        glUniform4f(Loc("uSharpTune"), 1.0f, 0.04f, 0.14f, 0.0f);
        Draw(mTaaSharp);
        CheckError("TAA sharpen");
        return &mTaaSharp;
    }
    return &mTaaOut;
}

void PostFxGL::RunSceneEnd(const FramebufferOGL& fb, const Settings& s, bool taa) {
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

    // Temporal anti-aliasing resolves the jittered world here: before bloom and the display transform, so the
    // temporal supersampling happens in scene-referred light, and before the HUD, which is drawn after the world.
    const Target* sceneSrc = &mScene;
    if (taa && mHdrActive && mTaaCam.valid && mTaa.id) {
        // Stage 1's timer is open around this function; GL_TIME_ELAPSED queries cannot nest, so TAA gets its own and
        // stage 1 (bloom, display transform) resumes after it.
        if (mTimers)
            glEndQuery(GL_TIME_ELAPSED);
        BeginTimer(kTaaTimerStage);
        if (const Target* resolved = RunTaa(fb, s))
            sceneSrc = resolved;
        EndTimer(kTaaTimerStage);
        if (mTimers)
            glBeginQuery(GL_TIME_ELAPSED, mQueries[1][mQueryNext[1]]);
    }

    const bool useBloom = s.bloomOn && levels > 0;
    if (useBloom) {
        Bind(0, sceneSrc->tex);
        Use(mHdrActive ? mDownHdr : mDown);
        glUniform1i(Loc("uSrc"), kFirstUnit + 0);
        glUniform1f(Loc("uThreshold"),
                    mHdrActive ? std::max(s.bloomThreshold, 1.0f) : std::min(s.bloomThreshold, 0.99f));
        glUniform1i(Loc("uHdr"), mHdrActive);
        for (int l = 0; l < levels; l++) {
            const Target& src = (l == 0) ? *sceneSrc : mBloom[l - 1];
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

    Bind(0, sceneSrc->tex);
    Bind(1, useBloom ? mBloom[0].tex : sceneSrc->tex);
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
    glUniform1i(Loc("uArtOn"), mArt.on && s.postFxOn && s.debugView != 2);
    if (mArt.on) {
        glUniform3fv(Loc("uArtSlope"), 1, mArt.slope);
        glUniform1f(Loc("uArtOffset"), mArt.offset);
        glUniform1f(Loc("uArtPower"), mArt.power);
        glUniform1f(Loc("uArtContrast"), mArt.contrast);
        glUniform1f(Loc("uArtSat"), mArt.sat);
        glUniform3fv(Loc("uArtShadow"), 1, mArt.shadow);
        glUniform3fv(Loc("uArtHigh"), 1, mArt.high);
        glUniform2fv(Loc("uArtGreen"), 1, mArt.green);
    }
    if (s.debugView == 1 && !mHdrActive) {
        glEnable(GL_SCISSOR_TEST);
        glScissor(w / 2, 0, w - w / 2, h);
    }
    Draw(mHdrActive ? mHdrDisplay.fbo : fb.fbo, w, h);
    CheckError("composite pass");
}

} // namespace Fast
#endif
