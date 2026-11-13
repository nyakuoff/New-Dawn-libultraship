#if defined(ENABLE_OPENGL) && !defined(USE_OPENGLES)
#include "fast/backends/gfx_opengl_postfx.h"
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"
#include "fast/newdawn_wind.h"
#include <stb_image.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <vector>

namespace Fast {
namespace {
constexpr int kSprite = 16, kDepth = 17, kRoof = 20, kScene = 21, kEnvironment = 22, kRing = 23;
const char* kWeatherVertex = R"GLSL(#version 330 core
uniform mat4 uViewProj;
uniform vec3 uEye;
uniform float uTime, uNdcXScale, uFlipY;
uniform vec2 uProjection,uSize,uWindVel;
uniform int uSnow;
out vec2 vUv;
out vec3 vWorld;
out float vFade,vViewZ;
uint hash(uint n) { n ^= n>>16u; n*=0x7feb352du; n^=n>>15u; n*=0x846ca68bu; return n^(n>>16u); }
float random(uint n) { return float(hash(n)&0xffffffu)/16777216.0; }
void main() {
    uint id=uint(gl_InstanceID);
    vec3 seed=vec3(random(id*3u+1u),random(id*3u+2u),random(id*3u+3u));
    float extent=uSnow!=0 ? 380.0 : 360.0;
    float speed=uSnow!=0 ? 12.0 : 620.0;
    vec3 p=(seed-0.5)*extent*2.0;
    p+=vec3(3.6*uTime,-speed*uTime,1.8*uTime);
    if (uSnow!=0) p+=vec3(sin(uTime*0.7+seed.x*6.28)*10.0,0.0,cos(uTime*0.6+seed.y*6.28)*10.0);
    vec3 relative=mod(p-uEye+extent,extent*2.0)-extent;
    p=uEye+relative;
    float distance=length(relative);
    vFade=smoothstep(16.0,45.0,distance)*(1.0-smoothstep(extent*0.7,extent*0.95,distance));
    vec2 corners[6]=vec2[6](vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
    vec2 q=corners[gl_VertexID];
    vUv=q*0.5+0.5;
    vec4 c0=uViewProj*vec4(p,1.0);
    c0.x/=uNdcXScale; c0.y*=uFlipY;
    gl_Position=c0;
    if (uSnow!=0) {
        gl_Position.xy+=q*mix(1.0,2.4,seed.z)*uProjection;
    } else {
        // Motion blur: project the exposure-length fall vector; fixed minimum pixel width keeps far rain visible.
        vec4 c1=uViewProj*vec4(p+vec3(uWindVel.x,-speed,uWindVel.y)*0.02,1.0);
        c1.x/=uNdcXScale; c1.y*=uFlipY;
        vec2 n0=c0.xy/max(c0.w,0.01), n1=c1.xy/max(c1.w,0.01);
        vec2 d=(n1-n0)*uSize*0.5;
        float len=clamp(length(d),2.0,260.0);
        vec2 dir=length(d)>0.001 ? normalize(d) : vec2(0.0,-1.0);
        float width=mix(1.6,2.8,seed.z);
        vec2 px=dir*(q.y*0.5+0.5)*len+vec2(-dir.y,dir.x)*q.x*width;
        gl_Position.xy+=px*2.0/uSize*c0.w;
    }
    vWorld=p; vViewZ=gl_Position.w;
}
)GLSL";
const char* kWeatherFragment = R"GLSL(#version 330 core
in vec2 vUv;
in vec3 vWorld;
in float vFade,vViewZ;
out vec4 oColor;
uniform sampler2D uWeatherSprite,uWeatherDepth,uWeatherRoof;
uniform mat4 uRoofViewProj;
uniform vec2 uSize,uNearFar;
uniform vec3 uSkyColor;
uniform float uIntensity;
uniform int uSnow;
float linearDepth(float d) {
    return 2.0*uNearFar.x*uNearFar.y/max(uNearFar.y+uNearFar.x-(2.0*d-1.0)*(uNearFar.y-uNearFar.x),0.001);
}
void main() {
    if (vFade<0.001 || vViewZ<=0.0) discard;
    vec3 roof=(uRoofViewProj*vec4(vWorld,1.0)).xyz*0.5+0.5;
    float ceiling=texture(uWeatherRoof,roof.xy).r;
    if (roof.z>ceiling+0.0008) discard;
    float scene=linearDepth(texture(uWeatherDepth,gl_FragCoord.xy/uSize).r);
    float soft=clamp((scene-vViewZ)/8.0,0.0,1.0);
    vec4 sprite=texture(uWeatherSprite,vUv);
    float alpha=sprite.a*vFade*soft*(uSnow!=0 ? 0.65*uIntensity : mix(0.6,1.0,uIntensity));
    vec3 color=uSnow!=0 ? max(uSkyColor*1.3,vec3(0.9,0.93,1.0)) : max(uSkyColor*1.6,vec3(0.30,0.33,0.38));
    oColor=vec4(color*alpha,alpha);
}
)GLSL";
const char* kWetVertex = R"GLSL(#version 330 core
out vec2 vUv;
void main() { vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));vUv=p;gl_Position=vec4(p*2.0-1.0,0.0,1.0); }
)GLSL";
const char* kWetFragment = R"GLSL(#version 330 core
in vec2 vUv;
out vec4 oColor;
uniform sampler2D uWeatherScene,uWeatherDepth,uWeatherRoof,uWeatherRing;
uniform samplerCube uWeatherEnvironment;
uniform mat4 uInvViewProj,uRoofViewProj;
uniform float uNdcXScale,uFlipY,uWetness,uRain,uTime;
uniform vec3 uEye,uSunDir,uSunColor;
float hash(vec2 p) { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p) {
    vec2 a=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);
    return mix(mix(hash(a),hash(a+vec2(1,0)),f.x),mix(hash(a+vec2(0,1)),hash(a+vec2(1)),f.x),f.y);
}
void main() {
    vec3 color=texture(uWeatherScene,vUv).rgb;
    float depth=texture(uWeatherDepth,vUv).r;
    if (depth>=0.99999) { oColor=vec4(color,1.0);return; }
    vec2 ndc=vUv*2.0-1.0;ndc*=vec2(uNdcXScale,uFlipY);
    vec4 world=uInvViewProj*vec4(ndc,depth*2.0-1.0,1.0);
    vec3 P=world.xyz/world.w;
    vec3 V=normalize(uEye-P);
    vec3 N=normalize(cross(dFdx(P),dFdy(P)));
    if (dot(N,V)<0.0) N=-N;
    vec3 roof=(uRoofViewProj*vec4(P,1.0)).xyz*0.5+0.5;
    float edge=min(min(roof.x,1.0-roof.x),min(roof.y,1.0-roof.y));
    float covered=smoothstep(0.0,0.08,edge)*step(0.0,roof.z)*step(roof.z,1.0);
    float exposed=covered>0.0 ? step(roof.z,texture(uWeatherRoof,roof.xy).r+0.0012)*covered : 0.0;
    float wet=exposed*uWetness;
    float flatness=smoothstep(0.65,0.95,N.y);
    float patch=smoothstep(0.34,0.72,noise(P.xz*0.012));
    float puddle=wet*flatness*patch;
    color*=1.0-wet*0.20;
    float fres=0.04+0.96*pow(1.0-max(dot(N,V),0.0),5.0);
    vec3 environment=textureLod(uWeatherEnvironment,reflect(-V,N),0.8).rgb;
    color=mix(color,environment,clamp(puddle*(0.12+0.65*fres),0.0,0.7));
    vec3 H=normalize(V+uSunDir);
    float roughness=mix(0.16,0.045,patch);
    float a2=roughness*roughness;
    float nh=max(dot(N,H),0.0);
    float den=nh*nh*(a2-1.0)+1.0;
    float spec=min(a2/(3.14159265*den*den),8.0)*max(dot(N,uSunDir),0.0);
    color+=uSunColor*spec*wet*0.08;
    // World-anchored CC0 ring sprites, reprojected onto flatness wet surfaces for splashes/ripples.
    vec2 cell=floor(P.xz/22.0),f=fract(P.xz/22.0);
    vec2 centre=vec2(0.2+0.6*hash(cell),0.2+0.6*hash(cell+3.0));
    float age=fract(uTime*1.8+hash(cell+7.0));
    vec2 ringUv=(f-centre)*22.0/(0.6+age*5.5)*0.5+0.5;
    if (all(greaterThanEqual(ringUv,vec2(0.0))) && all(lessThanEqual(ringUv,vec2(1.0)))) {
        vec4 ring=texture(uWeatherRing,ringUv);
        float splash=ring.a*ring.r*(1.0-age)*uRain*flatness*exposed;
        color+=vec3(0.55,0.62,0.70)*splash;
    }
    oColor=vec4(color,1.0);
}
)GLSL";
} // namespace

bool PostFxGL::BuildWeatherPrograms() {
    if (mWeatherAttempted)
        return WeatherReady();
    mWeatherAttempted = true;
    if (!BuildAtmosProgram() || !BuildAtmosShader(mWeatherParticles, kWeatherVertex, kWeatherFragment) ||
        !BuildAtmosShader(mWeatherWet, kWetVertex, kWetFragment))
        return false;
    Use(mWeatherParticles);
    glUniform1i(Loc("uWeatherSprite"), kSprite);
    glUniform1i(Loc("uWeatherDepth"), kDepth);
    glUniform1i(Loc("uWeatherRoof"), kRoof);
    Use(mWeatherWet);
    glUniform1i(Loc("uWeatherScene"), kScene);
    glUniform1i(Loc("uWeatherDepth"), kDepth);
    glUniform1i(Loc("uWeatherRoof"), kRoof);
    glUniform1i(Loc("uWeatherEnvironment"), kEnvironment);
    glUniform1i(Loc("uWeatherRing"), kRing);
    auto vars = Ship::Context::GetRawInstance()->GetConsoleVariables();
    std::string path = vars->GetString("gNewDawn.Atmos.SplashPath", "assets/newdawn/atmos/splash_64.png");
    if (!std::filesystem::path(path).is_absolute())
        path = Ship::Context::LocateFileAcrossAppDirs(path);
    int w, h, n;
    auto* pixels = stbi_load(path.c_str(), &w, &h, &n, 4);
    if (!pixels) {
        SPDLOG_ERROR("NewDawnAtmos: CC0 splash sprite could not load");
        glDeleteProgram(mWeatherWet.id);
        mWeatherWet.id = 0;
        return false;
    }
    glGenTextures(1, &mWeatherRing);
    glActiveTexture(GL_TEXTURE0 + kRing);
    glBindTexture(GL_TEXTURE_2D, mWeatherRing);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    stbi_image_free(pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenerateMipmap(GL_TEXTURE_2D);
    glGenTextures(1, &mWeatherNeutralCube);
    glActiveTexture(GL_TEXTURE0 + kEnvironment);
    glBindTexture(GL_TEXTURE_CUBE_MAP, mWeatherNeutralCube);
    const float sky[3] = { 0.08f, 0.10f, 0.13f };
    for (int face = 0; face < 6; face++)
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_RGB16F, 1, 1, 0, GL_RGB, GL_FLOAT, sky);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    for (int unit : { kRoof, kScene }) {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, mAtmosNeutralDepth);
    }
    return true;
}

void PostFxGL::RunWeather(const NewDawnPostFxParams& p, const FramebufferOGL& fb, const float* roofMatrix,
                          const float* positions, uint32_t vertexCount) {
    std::copy_n(roofMatrix, 16, mWeatherRoofMatrix);
    mWeatherFrameReady = false;
    if (!BuildWeatherPrograms())
        return;
    static const auto epoch = std::chrono::steady_clock::now();
    const double time = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
    mWeatherTime = time;
    const float rain = std::clamp(p.atmosRain, 0.0f, 1.0f), snow = p.atmosSnow;
    if (mWeatherScene != p.atmosScene || mWeatherLastFrame + 1 != mSettingsFrame) {
        mWeatherScene = p.atmosScene;
        mWeatherWetness = rain;
        mWeatherLastTime = time;
    }
    const float dt = std::clamp(float(time - mWeatherLastTime), 0.0f, 0.25f);
    mWeatherLastTime = time;
    mWeatherWetness = std::max(rain, mWeatherWetness - dt / 30.0f);
    mWeatherLastFrame = mSettingsFrame;
    if (rain <= 0 && snow <= 0 && mWeatherWetness <= 0)
        return;
    if (!positions || vertexCount < 3)
        return; // no safe roof/depth data: keep native rain
    if (!mWeatherRoof.tex) {
        glGenTextures(1, &mWeatherRoof.tex);
        glGenFramebuffers(1, &mWeatherRoof.fbo);
        glActiveTexture(GL_TEXTURE0 + kRoof);
        glBindTexture(GL_TEXTURE_2D, mWeatherRoof.tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindFramebuffer(GL_FRAMEBUFFER, mWeatherRoof.fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, mWeatherRoof.tex, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            SPDLOG_ERROR("NewDawnAtmos: incomplete roof framebuffer; retaining native rain");
            glDeleteTextures(1, &mWeatherRoof.tex);
            glDeleteFramebuffers(1, &mWeatherRoof.fbo);
            mWeatherRoof = {};
            glDeleteProgram(mWeatherWet.id);
            glDeleteProgram(mWeatherParticles.id);
            mWeatherWet.id = mWeatherParticles.id = 0;
            glActiveTexture(GL_TEXTURE0 + kRoof);
            glBindTexture(GL_TEXTURE_2D, mAtmosNeutralDepth);
            return;
        }
    }
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, mWeatherRoof.fbo);
    glViewport(0, 0, 1024, 1024);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glClear(GL_DEPTH_BUFFER_BIT);
    Use(mShadowDepth);
    glUniformMatrix4fv(Loc("uLightViewProj"), 1, GL_FALSE, roofMatrix);
    glBindVertexArray(mShadowVao);
    glBindBuffer(GL_ARRAY_BUFFER, mShadowVbo);
    // Drop sky domes and other huge triangles: only geometry near the camera can shelter rain.
    static std::vector<float> local;
    local.clear();
    for (uint32_t i = 0; i + 2 < vertexCount; i += 3) {
        bool near = true;
        for (int v = 0; v < 3 && near; v++) {
            const float* q = positions + (i + v) * 3;
            near = std::fabs(q[0] - p.atmosEye[0]) < 1000.0f && std::fabs(q[2] - p.atmosEye[2]) < 1000.0f &&
                   q[1] < p.atmosEye[1] + 1000.0f;
        }
        if (near)
            local.insert(local.end(), positions + i * 3, positions + i * 3 + 9);
    }
    glBufferData(GL_ARRAY_BUFFER, local.size() * sizeof(float), local.data(), GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, GLsizei(local.size() / 3));
    if (Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Atmos.Debug.Roof", 0)) {
        static int lastScene=-1;
        if (lastScene!=p.atmosScene) {
            lastScene=p.atmosScene;
            float depth;
            glReadPixels(512,512,1,1,GL_DEPTH_COMPONENT,GL_FLOAT,&depth);
            float eyeDepth=(p.atmosEye[0]*roofMatrix[2]+p.atmosEye[1]*roofMatrix[6]+
                            p.atmosEye[2]*roofMatrix[10]+roofMatrix[14])*0.5f+0.5f;
            float roofY=((2*depth-1)-roofMatrix[14]-p.atmosEye[0]*roofMatrix[2]-p.atmosEye[2]*roofMatrix[10])/roofMatrix[6];
            SPDLOG_INFO("NewDawnAtmos roof: scene {} eye ({},{},{}) eyeDepth {} mapDepth {} topY {} triangles {}",
                        p.atmosScene,p.atmosEye[0],p.atmosEye[1],p.atmosEye[2],eyeDepth,depth,roofY,vertexCount/3);
        }
    }
    glBindVertexArray(mVao);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    for (int unit : { kSprite, kDepth, kRoof, kScene, kEnvironment, kRing })
        glBindSampler(unit, 0);
    glActiveTexture(GL_TEXTURE0 + kRoof);
    glBindTexture(GL_TEXTURE_2D, mWeatherRoof.tex);
    glActiveTexture(GL_TEXTURE0 + kDepth);
    glBindTexture(GL_TEXTURE_2D, mDepth.tex);
    if (mWeatherWetness > 0) {
        const int w = fb.width, h = fb.height;
        EnsureTarget(mScene, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, false, true);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mScene.fbo);
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glActiveTexture(GL_TEXTURE0 + kScene);
        glBindTexture(GL_TEXTURE_2D, mScene.tex);
        glActiveTexture(GL_TEXTURE0 + kEnvironment);
        glBindTexture(GL_TEXTURE_CUBE_MAP, mSkyReflectionsReady ? mSkySpecular.tex : mWeatherNeutralCube);
        Use(mWeatherWet);
        glUniformMatrix4fv(Loc("uInvViewProj"), 1, GL_FALSE, p.shadowInvViewProj);
        glUniformMatrix4fv(Loc("uRoofViewProj"), 1, GL_FALSE, roofMatrix);
        glUniform1f(Loc("uNdcXScale"), p.shadowNdcXScale);
        glUniform1f(Loc("uFlipY"), p.shadowFlipY);
        glUniform1f(Loc("uWetness"), mWeatherWetness);
        glUniform1f(Loc("uRain"), rain);
        glUniform1f(Loc("uTime"), time);
        glUniform3fv(Loc("uEye"), 1, p.atmosEye);
        glUniform3fv(Loc("uSunDir"), 1, p.shadowSunDir);
        glUniform3fv(Loc("uSunColor"), 1, p.atmosSunColor);
        Draw(fb.fbo, w, h);
    }
    mWeatherFrameReady = true;
    glDisable(GL_BLEND);
    for (int unit : { kRoof, kScene, kDepth }) {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, mAtmosNeutralDepth);
    }
    glActiveTexture(GL_TEXTURE0 + kEnvironment);
    glBindTexture(GL_TEXTURE_CUBE_MAP, mWeatherNeutralCube);
}
void PostFxGL::RunWeatherParticles(const NewDawnPostFxParams& p, const FramebufferOGL& fb) {
    if (!mWeatherFrameReady)
        return;
    const float rain = std::clamp(p.atmosRain, 0.0f, 1.0f), snow = p.atmosSnow;
    glActiveTexture(GL_TEXTURE0 + kRoof);
    glBindTexture(GL_TEXTURE_2D, mWeatherRoof.tex);
    glActiveTexture(GL_TEXTURE0 + kDepth);
    glBindTexture(GL_TEXTURE_2D, mDepth.tex);
    const float* roofMatrix = mWeatherRoofMatrix;
    Use(mWeatherParticles);
    glUniformMatrix4fv(Loc("uViewProj"), 1, GL_FALSE, p.shadowViewProj);
    glUniformMatrix4fv(Loc("uRoofViewProj"), 1, GL_FALSE, roofMatrix);
    glUniform3fv(Loc("uEye"), 1, p.atmosEye);
    glUniform1f(Loc("uTime"), mWeatherTime);
    glUniform1f(Loc("uNdcXScale"), p.shadowNdcXScale);
    glUniform1f(Loc("uFlipY"), p.shadowFlipY);
    const float projection = 1.0f / std::tan(p.fovyDegrees * 0.00872664626f);
    glUniform2f(Loc("uProjection"), projection * float(fb.height) / fb.width, projection);
    glUniform2f(Loc("uSize"), fb.width, fb.height);
    // Gusting wind slants rain when the shared wind is enabled; otherwise the original gentle constant drift.
    float windVel[2] = { 3.6f, 1.8f };
    if (Ship::Context::GetRawInstance()->GetConsoleVariables()->GetInteger("gNewDawn.Atmos.Wind", 0)) {
        const auto wind = NewDawnWindAt(NewDawnWindSeconds());
        windVel[0] = wind.dirX * wind.gust * 60.0f;
        windVel[1] = wind.dirZ * wind.gust * 60.0f;
    }
    glUniform2f(Loc("uWindVel"), windVel[0], windVel[1]);
    glUniform2f(Loc("uNearFar"), p.zNear, p.zFar);
    float sky[3];
    for (int i = 0; i < 3; i++)
        sky[i] = std::pow(p.atmosSkyColor[i], 2.2f);
    glUniform3fv(Loc("uSkyColor"), 1, sky);
    glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
    glViewport(0, 0, fb.width, fb.height);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    if (rain > 0) {
        glUniform1i(Loc("uSnow"), 0);
        glUniform1f(Loc("uIntensity"), rain);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, 20000);
    }
    if (snow > 0) {
        glUniform1i(Loc("uSnow"), 1);
        glUniform1f(Loc("uIntensity"), std::min(1.0f, snow * 4.0f));
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, 8000);
    }

    glDisable(GL_BLEND);
    for (int unit : { kRoof, kDepth }) {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, mAtmosNeutralDepth);
    }
}

} // namespace Fast
#endif
