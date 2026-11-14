#if defined(ENABLE_OPENGL) && !defined(USE_OPENGLES)
#include "fast/backends/gfx_opengl_postfx.h"
#include "fast/newdawn_wind.h"
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"
#include <stb_image.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>

namespace Fast {
namespace {
// Dedicated units: never alias the interpreter, sky, fog or flare samplers.
constexpr int kSpriteUnit = 16, kDepthUnit = 17, kShadowUnit = 18, kLeafUnit = 19, kBackgroundUnit = 24;
const char* kVertex = R"GLSL(#version 330 core
uniform mat4 uViewProj;
uniform vec3 uEye;
uniform vec2 uWindOffset;
uniform float uTime;
uniform vec2 uProjection, uSize;
uniform float uNdcXScale, uFlipY;
uniform int uKind, uBiome;
out vec2 vUv;
out vec3 vWorld;
out float vFade, vViewZ, vNear;
flat out float vSeed;
uint hash(uint n) { n ^= n >> 16u; n *= 0x7feb352du; n ^= n >> 15u; n *= 0x846ca68bu; return n ^ (n >> 16u); }
float random(uint n) { return float(hash(n) & 0xffffffu)/16777216.0; }
void main() {
    uint id = uint(gl_InstanceID) + uint(uKind)*8192u;
    vec3 seed = vec3(random(id*3u+1u), random(id*3u+2u), random(id*3u+3u));
    float phase = seed.x*6.2831853;
    float extent = uBiome == 4 ? 220.0 : (uKind == 2 ? 260.0 : (uKind == 3 ? 400.0 : 380.0));
    vec3 p = (seed-0.5)*extent*2.0;
    p.xz += uWindOffset * (uBiome == 4 ? 0.08 : (uKind == 2 ? 0.1 : (uKind == 3 ? 1.0 : 0.45)));
    // Slow meandering hover (no steady rise or fall) for dust, pollen and fireflies; leaves fall.
    p += vec3(sin(uTime*0.23+phase)*16.0+sin(uTime*0.51+phase*2.0)*5.0, sin(uTime*0.17+phase*1.7)*12.0+sin(uTime*0.43+phase)*4.0,
              cos(uTime*0.19+phase)*14.0+cos(uTime*0.47+phase*3.0)*5.0);
    if (uKind == 3) p.y += uTime * -9.0;
    // Fixed world seeds and camera-relative periodic images. Outer shell fades before wrapping.
    vec3 relative = mod(p-uEye+extent, extent*2.0)-extent;
    p = uEye + relative;
    if (uKind == 2) relative.y = (seed.y-0.5)*90.0 - 28.0 + sin(uTime*0.3+phase)*8.0; // hover low, near grass and water
    p = uEye + relative;
    float distance = length(relative);
    vNear = 1.0 - smoothstep(30.0, 110.0, distance);
    // Dust/pollen vanish close to the lens instead of becoming large defocused discs.
    float nearFade = uKind <= 1 ? smoothstep(150.0, 190.0, distance) : smoothstep(18.0, 55.0, distance);
    vFade = nearFade * (1.0-smoothstep(extent*0.65, extent*0.94, distance));
    if (uKind == 2) vFade *= 0.12+0.88*pow(0.5+0.5*sin(uTime*(0.9+seed.y*1.2)+phase*3.0),2.0);
    vec2 corners[6] = vec2[6](vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
    vec2 q = corners[gl_VertexID];
    vUv = q*0.5+0.5;
    float radius = mix(2.0,5.0,seed.z*seed.z);
    if (uKind == 2) radius = mix(3.2,6.5,seed.z);
    if (uKind == 3) {
        radius = mix(2.5,4.5,seed.z);
        float rotation = uTime*0.7+phase;
        q.x *= 0.45+0.55*abs(cos(uTime*1.1+phase));
        q = mat2(cos(rotation),sin(rotation),-sin(rotation),cos(rotation))*q;
    }
    gl_Position = uViewProj*vec4(p,1.0);
    gl_Position.x /= uNdcXScale;
    gl_Position.y *= uFlipY;
    if (uKind <= 1) {
        // Full sprite diameter in pixels at 1080p; visible soft cores are typically 2–4 px.
        float diameter = mix(0.45, 0.90, seed.z*seed.z) * uProjection.y * 1080.0 / max(gl_Position.w, 1.0);
        diameter = clamp(diameter, 3.0, 8.0) * (uSize.y / 1080.0);
        gl_Position.xy += q * (diameter / uSize) * gl_Position.w;
    } else {
        gl_Position.xy += q * radius * uProjection;
    }
    vViewZ = gl_Position.w;
    vWorld = p;
    vSeed = seed.y;
}
)GLSL";
const char* kFragment = R"GLSL(#version 330 core
in vec2 vUv;
in vec3 vWorld;
in float vFade, vViewZ, vNear;
flat in float vSeed;
uniform float uTime;
out vec4 oColor;
uniform sampler2D uAtmosSprite;
uniform sampler2D uAtmosLeaf;
uniform sampler2D uAtmosDepth;
uniform sampler2D uAtmosShadow;
uniform sampler2D uAtmosBackground;
uniform mat4 uLightViewProj;
uniform vec2 uSize, uNearFar;
uniform bool uShadowReady;
uniform float uNight, uStrength, uRain;
uniform int uKind, uBiome, uLightCount;
uniform vec3 uSunColor;
uniform vec4 uLights[8];
uniform vec3 uLightColors[8];
float linearDepth(float d) {
    return 2.0*uNearFar.x*uNearFar.y / max(uNearFar.y+uNearFar.x-(2.0*d-1.0)*(uNearFar.y-uNearFar.x),0.001);
}
float sunlight() {
    if (!uShadowReady) return 0.18;
    vec4 clip = uLightViewProj*vec4(vWorld,1.0);
    vec3 p = clip.xyz/clip.w*0.5+0.5;
    if (any(lessThan(p,vec3(0.0))) || any(greaterThan(p,vec3(1.0)))) return 0.0;
    vec2 texel = 1.0/vec2(textureSize(uAtmosShadow,0));
    float sum = 0.0;
    for (int y=0;y<2;y++) for (int x=0;x<2;x++) {
        float d = texture(uAtmosShadow,p.xy+(vec2(x,y)-0.5)*texel*2.0).r;
        sum += float(p.z-0.0005 < d)*0.25;
    }
    return sum*smoothstep(0.0,0.08,min(min(p.x,1.0-p.x),min(p.y,1.0-p.y)));
}
void main() {
    if (vFade < 0.001 || vViewZ <= 0.0) discard;
    float scene = linearDepth(texture(uAtmosDepth,gl_FragCoord.xy/uSize).r);
    float soft = clamp((scene-vViewZ)/6.0,0.0,1.0);
    vec4 sprite = uKind == 3 ? texture(uAtmosLeaf,vUv) : texture(uAtmosSprite,vUv);
    float core = uKind == 2 ? mix(sprite.r, sprite.a, 0.7) : sprite.r; // fireflies keep a soft halo
    float alpha = sprite.a*(uKind == 3 ? 1.0 : core)*vFade*soft*uStrength;
    float bright = mix(0.45, 1.7, fract(vSeed*13.7));
    float twinkle = 0.65+0.35*sin(uTime*(0.8+fract(vSeed*5.3)*1.8)+vSeed*40.0);
    vec3 radiance;
    if (uKind == 2) {
        radiance = mix(vec3(1.1,2.8,0.20), vec3(2.8,2.0,0.15),vSeed)*uNight*2.2;
    } else {
        float sun = uBiome == 4 ? 0.0 : sunlight()*(1.0-uNight)*(1.0-clamp(uRain,0.0,1.0));
        radiance = uSunColor*sun*(uKind <= 1 ? 4.5 : 3.4)*bright*twinkle*vec3(1.0,0.88,0.66);
        alpha *= mix(1.0, 0.5, vNear); // defocused near motes are dimmer
        if (uKind == 1) radiance *= vec3(0.95,1.0,0.65);
        for (int i=0;i<uLightCount;i++) {
            float distance = length(vWorld-uLights[i].xyz);
            float influence = max(1.0-distance/max(uLights[i].w,1.0),0.0);
            radiance += uLightColors[i]*influence*influence*0.55;
        }
    }
    if (uKind == 3) {
        radiance = pow(sprite.rgb,vec3(2.2)) * (radiance + vec3(0.18));
        oColor = vec4(radiance*alpha,alpha);
        return;
    }
    if (uKind <= 1) {
        // Sun visibility lights a speck in a beam; a bright background reduces its apparent contrast.
        vec3 background = texture(uAtmosBackground, gl_FragCoord.xy/uSize).rgb;
        float luma = dot(background, vec3(0.2126,0.7152,0.0722));
        radiance *= mix(1.0, 0.16, smoothstep(0.15, 0.8, luma));
    }
    alpha *= uKind == 2 ? 0.9 : 0.65;
    oColor = vec4(radiance*alpha,0.0); // in-scattering; no depth/opaque colour modification
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
        SPDLOG_ERROR("NewDawnAtmos shader: {}", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}
} // namespace

bool PostFxGL::BuildAtmosShader(Program& program, const char* vertex, const char* fragment) {
    GLuint vs = Compile(GL_VERTEX_SHADER, vertex), fs = Compile(GL_FRAGMENT_SHADER, fragment);
    if (!vs || !fs) {
        glDeleteShader(vs);
        glDeleteShader(fs);
        return false;
    }
    program.id = glCreateProgram();
    glAttachShader(program.id, vs);
    glAttachShader(program.id, fs);
    glLinkProgram(program.id);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok;
    glGetProgramiv(program.id, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(program.id, sizeof(log), nullptr, log);
        SPDLOG_ERROR("NewDawnAtmos link: {}", log);
        glDeleteProgram(program.id);
        program.id = 0;
        return false;
    }
    GLint count;
    glGetProgramiv(program.id, GL_ACTIVE_UNIFORMS, &count);
    for (GLint i = 0; i < count; i++) {
        char name[256];
        GLsizei length;
        GLint size;
        GLenum type;
        glGetActiveUniform(program.id, i, sizeof(name), &length, &size, &type, name);
        program.uniforms[name] = glGetUniformLocation(program.id, name);
    }
    return true;
}

bool PostFxGL::BuildAtmosProgram() {
    // Lazy, optional: an atmosphere failure never disables the established postfx pipeline.
    if (mAtmosAttempted)
        return mAtmos.id != 0;
    mAtmosAttempted = true;
    if (!BuildAtmosShader(mAtmos, kVertex, kFragment))
        return false;
    Use(mAtmos);
    glUniform1i(Loc("uAtmosSprite"), kSpriteUnit);
    glUniform1i(Loc("uAtmosDepth"), kDepthUnit);
    glUniform1i(Loc("uAtmosShadow"), kShadowUnit);
    glUniform1i(Loc("uAtmosLeaf"), kLeafUnit);
    glUniform1i(Loc("uAtmosBackground"), kBackgroundUnit);
    int w, h, n;
    auto vars = Ship::Context::GetRawInstance()->GetConsoleVariables();
    auto assetPath = [](const char* path) {
        return std::filesystem::path(path).is_absolute() ? std::string(path)
                                                         : Ship::Context::LocateFileAcrossAppDirs(path);
    };
    const auto spritePath = assetPath(vars->GetString("gNewDawn.Atmos.SpritePath", "assets/newdawn/atmos/dust_64.png"));
    const auto leafPath = assetPath(vars->GetString("gNewDawn.Atmos.LeafPath", "assets/newdawn/atmos/leaf_64.png"));
    auto* pixels = stbi_load(spritePath.c_str(), &w, &h, &n, 4);
    if (!pixels) {
        SPDLOG_ERROR("NewDawnAtmos: CC0 dust sprite could not load");
        glDeleteProgram(mAtmos.id);
        mAtmos.id = 0;
        return false;
    }
    glGenTextures(1, &mAtmosSprite);
    glActiveTexture(GL_TEXTURE0 + kSpriteUnit);
    glBindTexture(GL_TEXTURE_2D, mAtmosSprite);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    stbi_image_free(pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenerateMipmap(GL_TEXTURE_2D);
    pixels = stbi_load(leafPath.c_str(), &w, &h, &n, 4);
    if (!pixels) {
        SPDLOG_ERROR("NewDawnAtmos: CC0 leaf sprite could not load");
        glDeleteProgram(mAtmos.id);
        mAtmos.id = 0;
        return false;
    }
    glGenTextures(1, &mAtmosLeaf);
    glActiveTexture(GL_TEXTURE0 + kLeafUnit);
    glBindTexture(GL_TEXTURE_2D, mAtmosLeaf);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    stbi_image_free(pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenerateMipmap(GL_TEXTURE_2D);
    // A complete texture remains on the dedicated background sampler even while the pass is idle.
    glActiveTexture(GL_TEXTURE0 + kBackgroundUnit);
    glBindTexture(GL_TEXTURE_2D, mAtmosSprite);
    glGenTextures(1, &mAtmosNeutralDepth);
    glActiveTexture(GL_TEXTURE0 + kDepthUnit);
    glBindTexture(GL_TEXTURE_2D, mAtmosNeutralDepth);
    const float depth = 1.0f;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, 1, 1, 0, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glActiveTexture(GL_TEXTURE0 + kShadowUnit);
    glBindTexture(GL_TEXTURE_2D, mAtmosNeutralDepth);
    return true;
}

void PostFxGL::RunAtmos(const NewDawnPostFxParams& p, const FramebufferOGL& fb, const float* lightMatrix,
                        bool shadowReady) {
    if (!BuildAtmosProgram())
        return;
    Use(mAtmos);
    GLuint background = mWaterOpaqueColor.tex;
    if (mWaterOpaqueFrame != mSettingsFrame || mWaterOpaqueColor.w != fb.width || mWaterOpaqueColor.h != fb.height) {
        // No water snapshot in this view: copy to a separate target, never sample the draw framebuffer.
        EnsureTarget(mScene, fb.width, fb.height, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
        glDisable(GL_SCISSOR_TEST);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mScene.fbo);
        glBlitFramebuffer(0, 0, fb.width, fb.height, 0, 0, fb.width, fb.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        background = mScene.tex;
    }
    glActiveTexture(GL_TEXTURE0 + kBackgroundUnit);
    glBindTexture(GL_TEXTURE_2D, background);
    glActiveTexture(GL_TEXTURE0 + kDepthUnit);
    glBindTexture(GL_TEXTURE_2D, mDepth.tex);
    glActiveTexture(GL_TEXTURE0 + kShadowUnit);
    glBindTexture(GL_TEXTURE_2D, shadowReady ? mShadowMap.tex : mAtmosNeutralDepth);
    for (int unit : { kSpriteUnit, kDepthUnit, kShadowUnit, kLeafUnit, kBackgroundUnit })
        glBindSampler(unit, 0);
    glUniformMatrix4fv(Loc("uViewProj"), 1, GL_FALSE, p.shadowViewProj);
    glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, lightMatrix);
    glUniform3fv(Loc("uEye"), 1, p.atmosEye);
    glUniform1f(Loc("uNdcXScale"), p.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), p.shadowFlipY);
    const float projection = 1.0f / std::tan(p.fovyDegrees * 0.00872664626f);
    glUniform2f(Loc("uProjection"), projection * float(fb.height) / fb.width, projection);
    glUniform2f(Loc("uSize"), fb.width, fb.height);
    glUniform2f(Loc("uNearFar"), p.zNear, p.zFar);
    glUniform1i(Loc("uShadowReady"), shadowReady);
    glUniform1i(Loc("uBiome"), p.atmosBiome);
    glUniform1f(Loc("uNight"), p.atmosNight);
    glUniform1f(Loc("uRain"), p.atmosRain);
    glUniform3fv(Loc("uSunColor"), 1, p.atmosSunColor);
    glUniform1i(Loc("uLightCount"), p.atmosLightCount);
    glUniform4fv(Loc("uLights[0]"), NEWDAWN_ATMOS_LIGHTS, &p.atmosLights[0][0]);
    glUniform3fv(Loc("uLightColors[0]"), NEWDAWN_ATMOS_LIGHTS, &p.atmosLightColors[0][0]);
    auto* vars = Ship::Context::GetRawInstance()->GetConsoleVariables().get();
    const float strength = std::clamp(vars->GetFloat("gNewDawn.Atmos.Strength", 1.0f), 0.0f, 2.0f);
    glUniform1f(Loc("uStrength"), strength);
    static const auto epoch = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
    // Double precision drift before converting bounded offsets to shader floats.
    const float t = static_cast<float>(seconds);
    glUniform1f(Loc("uTime"), t);
    double drift = seconds;
    if (vars->GetInteger("gNewDawn.Atmos.Wind", 0))
        drift = NewDawnWindTravel(NewDawnWindSeconds());
    glUniform2f(Loc("uWindOffset"), static_cast<float>(std::fmod(drift * 3.6, 924000.0)),
                static_cast<float>(std::fmod(drift * 1.8, 924000.0)));
    glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
    glViewport(0, 0, fb.width, fb.height);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glUniform1i(Loc("uKind"), 0);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, p.atmosBiome == 4 ? 1024 : 3000);
    if (p.atmosBiome == 2 || p.atmosBiome == 3) {
        if (p.atmosNight < 1.0f) {
            glUniform1i(Loc("uKind"), 1);
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, 320);
        }
        if (p.atmosNight > 0.0f) {
            glUniform1i(Loc("uKind"), 2);
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, p.atmosBiome == 3 ? 320 : 220);
        }
    }
    if (p.atmosBiome == 3) {
        glUniform1i(Loc("uKind"), 3);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, 128);
    }
    // Complete neutral depth bindings remain assigned when disabled or between frames.
    glActiveTexture(GL_TEXTURE0 + kDepthUnit);
    glBindTexture(GL_TEXTURE_2D, mAtmosNeutralDepth);
    glActiveTexture(GL_TEXTURE0 + kShadowUnit);
    glBindTexture(GL_TEXTURE_2D, mAtmosNeutralDepth);
    glActiveTexture(GL_TEXTURE0 + kBackgroundUnit);
    glBindTexture(GL_TEXTURE_2D, mAtmosSprite);
}
} // namespace Fast
#endif
