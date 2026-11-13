#include "ship/window/Window.h"
#if defined(ENABLE_OPENGL) && !defined(USE_OPENGLES)
#include "gfx_opengl_mm3d.h"
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <spdlog/spdlog.h>

#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#endif

namespace Fast {
namespace {
// Restore exactly the state touched here, leaving the interpreter's caches valid.
// Units 8..10 are outside its own texture units. Framebuffer/viewport/scissor stay unchanged.
struct State {
    GLint program, vao, vbo, active, unpack, depthFunc, cullMode, frontFace;
    GLint srcRgb, dstRgb, srcA, dstA, eqRgb, eqA;
    GLfloat blendColor[4];
    GLboolean depthMask, colorMask[4], depth, cull, blend, offset, a2c, srgb;
    GLint bindings[3], samplers[3];
    State() {
        glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &vbo);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &active);
        glGetIntegerv(GL_UNPACK_ALIGNMENT, &unpack);
        glGetIntegerv(GL_DEPTH_FUNC, &depthFunc);
        glGetIntegerv(GL_CULL_FACE_MODE, &cullMode);
        glGetIntegerv(GL_FRONT_FACE, &frontFace);
        glGetIntegerv(GL_BLEND_SRC_RGB, &srcRgb); glGetIntegerv(GL_BLEND_DST_RGB, &dstRgb);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &srcA); glGetIntegerv(GL_BLEND_DST_ALPHA, &dstA);
        glGetIntegerv(GL_BLEND_EQUATION_RGB, &eqRgb); glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &eqA);
        glGetFloatv(GL_BLEND_COLOR, blendColor);
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask); glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);
        depth = glIsEnabled(GL_DEPTH_TEST); cull = glIsEnabled(GL_CULL_FACE); blend = glIsEnabled(GL_BLEND);
        offset = glIsEnabled(GL_POLYGON_OFFSET_FILL); a2c = glIsEnabled(GL_SAMPLE_ALPHA_TO_COVERAGE);
        srgb = glIsEnabled(GL_FRAMEBUFFER_SRGB);
        for (int i = 0; i < 3; ++i) {
            glActiveTexture(GL_TEXTURE8 + i);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &bindings[i]);
            glGetIntegeri_v(GL_SAMPLER_BINDING, 8 + i, &samplers[i]);
        }
    }
    static void Set(GLenum cap, bool on) { if (on) glEnable(cap); else glDisable(cap); }
    ~State() {
        glUseProgram(program); glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glPixelStorei(GL_UNPACK_ALIGNMENT, unpack);
        glDepthFunc(depthFunc); glDepthMask(depthMask); glCullFace(cullMode); glFrontFace(frontFace);
        glBlendFuncSeparate(srcRgb, dstRgb, srcA, dstA); glBlendEquationSeparate(eqRgb, eqA);
        glBlendColor(blendColor[0], blendColor[1], blendColor[2], blendColor[3]);
        glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
        Set(GL_DEPTH_TEST, depth); Set(GL_CULL_FACE, cull); Set(GL_BLEND, blend);
        Set(GL_POLYGON_OFFSET_FILL, offset); Set(GL_SAMPLE_ALPHA_TO_COVERAGE, a2c);
        Set(GL_FRAMEBUFFER_SRGB, srgb);
        for (int i = 0; i < 3; ++i) {
            glActiveTexture(GL_TEXTURE8 + i); glBindTexture(GL_TEXTURE_2D, bindings[i]);
            glBindSampler(8 + i, samplers[i]);
        }
        glActiveTexture(active);
    }
};

const char* kVertex = R"GLSL(#version 330 core
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV0;
layout(location=3) in vec4 aColor;
layout(location=4) in vec2 aUV1;
layout(location=5) in vec2 aUV2;
uniform mat4 uProjection;
uniform vec2 uClipScale;
uniform float uFogNear;
uniform vec3 uAmbient, uLightDirection[2], uLightColor[2];
out vec3 vNormal;
out vec4 vColor;
out vec2 vUV0, vUV1, vUV2;
out float vFog;
out vec3 vLighting;
vec3 light(vec3 n) {
    return uAmbient + max(dot(n,uLightDirection[0]),0.0)*uLightColor[0]
                    + max(dot(n,uLightDirection[1]),0.0)*uLightColor[1];
}
void main() {
    vec4 clip = uProjection * vec4(aPosition,1.0);
    // Exact N64 fog-position coefficient convention, evaluated at each vertex.
    float ndcZ = clip.z / max(clip.w,0.001);
    float mul = 128000.0 / max(1000.0-uFogNear,1.0);
    float offset = (500.0-uFogNear)*256.0 / max(1000.0-uFogNear,1.0);
    vFog = clamp((ndcZ*mul+offset)/255.0,0.0,1.0);
    clip.xy *= uClipScale;
    gl_Position = clip;
    vNormal=aNormal; vColor=aColor; vUV0=aUV0; vUV1=aUV1; vUV2=aUV2;
    vLighting=light(normalize(aNormal+vec3(0.000001)));
}
)GLSL";

const char* kFragment = R"GLSL(#version 330 core
in vec3 vNormal;
in vec4 vColor;
in vec2 vUV0, vUV1, vUV2;
in float vFog;
in vec3 vLighting;
out vec4 outColor;
uniform sampler2D uTexture0,uTexture1,uTexture2;
uniform int uHasTexture[3],uSource[3],uMapping[3];
uniform vec4 uUVScaleTranslate[3];
uniform float uUVRotation[3];
uniform vec3 uAmbient,uLightDirection[2],uLightColor[2],uFogColor,uViewRight,uViewUp;
uniform int uFog,uAlphaTest,uAlphaFunction,uHDR,uPerPixel,uFragmentLighting;
uniform float uAlphaReference,uHdrAmbient,uHdrSun,uHdrFog;
uniform vec4 uDiffuse,uMaterialAmbient,uEmission,uConstant[6];
uniform int uStageCount;
uniform int uStages[120];
vec3 decode(vec3 c) { return mix(c/12.92,pow((max(c,vec3(0.0))+0.055)/1.055,vec3(2.4)),step(vec3(0.04045),c)); }
vec3 encode(vec3 c) { return mix(c*12.92,1.055*pow(max(c,vec3(0.0)),vec3(1.0/2.4))-0.055,step(vec3(0.0031308),c)); }
vec2 coord(int i) {
    vec2 uv = uSource[i]==1?vUV1:(uSource[i]==2?vUV2:vUV0);
    if(uMapping[i]==3) { vec3 n=normalize(vNormal); uv=vec2(dot(n,uViewRight),dot(n,uViewUp))*0.5+0.5; }
    vec4 st=uUVScaleTranslate[i]; float angle=uUVRotation[i]*3.14159265359;
    float s=sin(angle),c=cos(angle);
    uv-=vec2(0.5);
    vec2 transformed=vec2(st.x*(c*uv.x-s*uv.y+0.5-st.z),st.y*(s*uv.x+c*uv.y+0.5-st.w));
    transformed.y=1.0-transformed.y; // CMB UVs address the decoded PICA atlas from its opposite edge.
    return transformed;
}
vec4 source(int code,vec4 t0,vec4 t1,vec4 t2,vec4 primary,vec4 lit,vec4 prev,vec4 buffer,vec4 constantColor) {
    if(code==0x84C0) return t0;
    if(code==0x84C1) return t1;
    if(code==0x84C2) return t2;
    if(code==0x8576) return constantColor;
    if(code==0x8577) return primary;
    if(code==0x8578) return prev;
    if(code==0x8579) return buffer;
    if(code==0x6210) return lit;
    if(code==0x6211) return vec4(0.0);
    return vec4(1.0);
}
vec3 operand(vec4 v,int op) {
    if(op==0x301) return 1.0-v.rgb;
    if(op==0x302) return vec3(v.a);
    if(op==0x303) return vec3(1.0-v.a);
    if(op==0x8580) return vec3(v.r);
    if(op==0x8581) return vec3(v.g);
    if(op==0x8582) return vec3(v.b);
    if(op==0x8583) return vec3(1.0-v.r);
    if(op==0x8584) return vec3(1.0-v.g);
    if(op==0x8585) return vec3(1.0-v.b);
    return v.rgb;
}
float alphaOperand(vec4 v,int op) {
    if(op==0x301||op==0x303) return 1.0-v.a;
    if(op==0x8580) return v.r; if(op==0x8581) return v.g; if(op==0x8582) return v.b;
    if(op==0x8583) return 1.0-v.r; if(op==0x8584) return 1.0-v.g; if(op==0x8585) return 1.0-v.b;
    return v.a;
}
vec3 combine(int op,vec3 a,vec3 b,vec3 c) {
    if(op==0x1E01) return a;
    if(op==0x2100) return a*b;
    if(op==0x104) return a+b;
    if(op==0x8574) return a+b-0.5;
    if(op==0x8575) return a*c+b*(1.0-c);
    if(op==0x84E7) return a-b;
    if(op==0x86AE||op==0x86AF) return vec3(dot(a-0.5,b-0.5)*4.0);
    if(op==0x6401) return a*b+c;
    if(op==0x6402) return min(a+b,vec3(1.0))*c;
    return a;
}
bool alphaPass(float a) {
    if(uAlphaFunction==0x200) return false;
    if(uAlphaFunction==0x201) return a<uAlphaReference;
    if(uAlphaFunction==0x202) return abs(a-uAlphaReference)<(0.5/255.0);
    if(uAlphaFunction==0x203) return a<=uAlphaReference;
    if(uAlphaFunction==0x204) return a>uAlphaReference;
    if(uAlphaFunction==0x205) return abs(a-uAlphaReference)>=(0.5/255.0);
    if(uAlphaFunction==0x206) return a>=uAlphaReference;
    return true;
}
void main() {
    vec4 t0=uHasTexture[0]!=0?texture(uTexture0,coord(0)):vec4(1.0);
    vec4 t1=uHasTexture[1]!=0?texture(uTexture1,coord(1)):vec4(1.0);
    vec4 t2=uHasTexture[2]!=0?texture(uTexture2,coord(2)):vec4(1.0);
    vec3 n=normalize(vNormal+vec3(0.000001));
    vec3 irr=uPerPixel!=0 ? uAmbient + max(dot(n,uLightDirection[0]),0.0)*uLightColor[0]
                                         + max(dot(n,uLightDirection[1]),0.0)*uLightColor[1] : vLighting;
    // Baked room colour is retained; the live N64 environment supplies its time/weather tint.
    vec3 bakedTint=clamp(uAmbient+(uLightColor[0]+uLightColor[1])*0.5,vec3(0.05),vec3(1.0));
    // PICA combiners operate on display colours. Encode linear HDR light before
    // the combiner and decode once afterwards; never multiply light a second time.
    vec3 litEnergy=uMaterialAmbient.rgb*uAmbient*(uHDR!=0?uHdrAmbient:1.0)
                  +uDiffuse.rgb*max(irr-uAmbient,0.0)*(uHDR!=0?uHdrSun:1.0)+uEmission.rgb;
    vec3 lightDisplay=uHDR!=0 ? encode(max(litEnergy,vec3(0.0))) : clamp(litEnergy,0.0,1.0);
    // Baked colour already contains the room's lighting; HDR sun/ambient gains apply only to the live 'lit' term.
    vec3 bakedDisplay=bakedTint;
    vec4 primary=vec4(vColor.rgb*bakedDisplay,vColor.a);
    vec4 lit=vec4(lightDisplay,uDiffuse.a);
    vec4 prev=primary,buffer=vec4(0.0);
    for(int i=0;i<6;i++) {
        if(i>=uStageCount) break;
        int b=i*20;
        vec4 cc=uConstant[clamp(uStages[b+18],0,5)];
        vec3 a=operand(source(uStages[b+6],t0,t1,t2,primary,lit,prev,buffer,cc),uStages[b+9]);
        vec3 bb=operand(source(uStages[b+7],t0,t1,t2,primary,lit,prev,buffer,cc),uStages[b+10]);
        vec3 c=operand(source(uStages[b+8],t0,t1,t2,primary,lit,prev,buffer,cc),uStages[b+11]);
        float aa=alphaOperand(source(uStages[b+12],t0,t1,t2,primary,lit,prev,buffer,cc),uStages[b+15]);
        float ab=alphaOperand(source(uStages[b+13],t0,t1,t2,primary,lit,prev,buffer,cc),uStages[b+16]);
        float ac=alphaOperand(source(uStages[b+14],t0,t1,t2,primary,lit,prev,buffer,cc),uStages[b+17]);
        vec3 rgb=max(combine(uStages[b],a,bb,c)*float(uStages[b+2]),vec3(0.0));
        if(uHDR==0) rgb=min(rgb,vec3(1.0)); // float HDR world keeps source radiance above display white.
        vec4 next=vec4(rgb,clamp(combine(uStages[b+1],vec3(aa),vec3(ab),vec3(ac)).r*float(uStages[b+3]),0.0,1.0));
        if(uStages[b+4]==0x8578) buffer.rgb=prev.rgb;
        if(uStages[b+5]==0x8578) buffer.a=prev.a;
        prev=next;
    }
    if(uStageCount==0) prev=primary*t0;
    if(uAlphaTest!=0&&!alphaPass(prev.a)) discard;
    if(uHDR!=0) {
        prev.rgb=decode(prev.rgb);
        // The HDR lighting controls were already applied in the lighting sources above.
    }
    if(uFog!=0) prev.rgb=mix(prev.rgb,uHDR!=0?decode(uFogColor)*uHdrFog:uFogColor,vFog);
    outColor=prev;
}
)GLSL";

GLuint Compile(GLenum type, const char* code) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &code, nullptr); glCompileShader(shader);
    GLint ok; glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096]; glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        SPDLOG_ERROR("MM3D shader compile: {}", log); glDeleteShader(shader); return 0;
    }
    return shader;
}
GLenum Wrap(uint32_t value) { return value == GL_REPEAT || value == GL_MIRRORED_REPEAT ? value : GL_CLAMP_TO_EDGE; }
} // namespace

MM3DGL::~MM3DGL() {
    for (auto& [key, m] : mMeshes) {
        glDeleteVertexArrays(1, &m.vao); glDeleteBuffers(1, &m.vbo); glDeleteBuffers(1, &m.ibo);
    }
    for (auto& [key, texture] : mTextures) glDeleteTextures(1, &texture);
    glDeleteQueries(mQueries.size(), mQueries.data());
    if (mProgram) glDeleteProgram(mProgram);
}

GLint MM3DGL::U(const std::string& name) {
    auto [it, inserted] = mUniforms.emplace(name, -1);
    if (inserted) it->second = glGetUniformLocation(mProgram, name.c_str());
    return it->second;
}

bool MM3DGL::Build() {
    GLuint vs = Compile(GL_VERTEX_SHADER, kVertex), fs = Compile(GL_FRAGMENT_SHADER, kFragment);
    if (!vs || !fs) { if (vs) glDeleteShader(vs); if (fs) glDeleteShader(fs); return false; }
    mProgram = glCreateProgram(); glAttachShader(mProgram, vs); glAttachShader(mProgram, fs);
    glLinkProgram(mProgram); glDeleteShader(vs); glDeleteShader(fs);
    GLint ok; glGetProgramiv(mProgram, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096]; glGetProgramInfoLog(mProgram, sizeof(log), nullptr, log);
        SPDLOG_ERROR("MM3D program link: {}", log); glDeleteProgram(mProgram); mProgram=0; return false;
    }
    glGenQueries(mQueries.size(), mQueries.data());
    return true;
}

bool MM3DGL::Prepare(const NewDawnMM3DRoomParams& params) {
    if (mFailed) return false;
    bool missing = !mProgram;
    for (uint32_t i = 0; !missing && i < params.meshCount; ++i) {
        const auto& mesh = *params.meshes[i];
        missing = !mMeshes.contains(mesh.id);
        for (uint32_t j = 0; !missing && j < mesh.textureCount; ++j) {
            missing = !mTextures.contains(mesh.textures[j].id);
        }
    }
    if (!missing) return true;
    State state;
    auto start = std::chrono::steady_clock::now();
    bool uploaded = false;
    if (!mProgram && !Build()) { mFailed = true; return false; }
    for (uint32_t i = 0; i < params.meshCount; ++i) {
        const auto& mesh = *params.meshes[i];
        if (!mMeshes.contains(mesh.id)) {
            Mesh gpu;
            glGenVertexArrays(1, &gpu.vao); glBindVertexArray(gpu.vao);
            glGenBuffers(1, &gpu.vbo); glBindBuffer(GL_ARRAY_BUFFER, gpu.vbo);
            glBufferData(GL_ARRAY_BUFFER, mesh.vertexCount*16*sizeof(float), mesh.vertices, GL_STATIC_DRAW);
            glGenBuffers(1, &gpu.ibo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpu.ibo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indexCount*sizeof(uint32_t), mesh.indices, GL_STATIC_DRAW);
            const int dims[] = {3,3,2,4,2,2}, offsets[] = {0,3,6,8,12,14};
            for (int j=0;j<6;++j) {
                glEnableVertexAttribArray(j);
                glVertexAttribPointer(j,dims[j],GL_FLOAT,GL_FALSE,64,reinterpret_cast<void*>(offsets[j]*sizeof(float)));
            }
            mMeshes.emplace(mesh.id,gpu); uploaded = true;
        }
        for (uint32_t j=0;j<mesh.textureCount;++j) {
            const auto& texture=mesh.textures[j];
            if (mTextures.contains(texture.id)) continue;
            GLuint gpu; glGenTextures(1,&gpu); glActiveTexture(GL_TEXTURE8); glBindTexture(GL_TEXTURE_2D,gpu);
            glPixelStorei(GL_UNPACK_ALIGNMENT,1);
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,texture.width,texture.height,0,GL_RGBA,GL_UNSIGNED_BYTE,texture.rgba);
            glGenerateMipmap(GL_TEXTURE_2D);
            mTextures.emplace(texture.id,gpu); uploaded = true;
        }
    }
    if (uploaded) SPDLOG_INFO("MM3D: scene {} room {} GL upload {:.3f} ms",params.scene,params.room,
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    for (GLenum error=glGetError();error!=GL_NO_ERROR;error=glGetError()) {
        SPDLOG_ERROR("MM3D prepare: GL error 0x{:X}; keeping N64 rooms",error); mFailed=true;
    }
    return !mFailed;
}

void MM3DGL::Draw(const NewDawnMM3DRoomParams& params,const float* projection,float aspect,float flipY,
                  bool hdr,bool mipmaps,float anisotropy,float lodBias,const NewDawnMM3DScroll* scroll) {
    if (mFailed || !mProgram) return;
    const auto begin=std::chrono::steady_clock::now();
    auto cvars=Ship::Context::GetRawInstance()->GetConsoleVariables();
    bool timing=cvars->GetInteger("gNewDawn.MM3D.Debug.Stats",0)!=0;
    bool query=false;
    if (timing) {
        if (mQueryPending[mQuerySlot]) {
            GLint ready; glGetQueryObjectiv(mQueries[mQuerySlot],GL_QUERY_RESULT_AVAILABLE,&ready);
            if (ready) {
                GLuint64 ns; glGetQueryObjectui64v(mQueries[mQuerySlot],GL_QUERY_RESULT,&ns);
                mGpuMs+=ns/1000000.0; ++mTimerCount; mQueryPending[mQuerySlot]=false;
                if (mTimerCount>=300) {
                    SPDLOG_INFO("MM3D: room draws average {:.3f} ms GPU, {:.3f} ms CPU submit over {} calls",
                                mGpuMs/mTimerCount,mCpuMs/mTimerCount,mTimerCount);
                    mGpuMs=mCpuMs=0; mTimerCount=0;
                }
            }
        }
        query=!mQueryPending[mQuerySlot];
        if(query) glBeginQuery(GL_TIME_ELAPSED,mQueries[mQuerySlot]);
    }
    {
        State state;
        glUseProgram(mProgram);
        glDisable(GL_FRAMEBUFFER_SRGB); glDisable(GL_POLYGON_OFFSET_FILL);
        glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE); glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
        glUniformMatrix4fv(U("uProjection"),1,GL_FALSE,projection);
        glUniform2f(U("uClipScale"),aspect,flipY);
        float right[3] = {projection[0], projection[4], projection[8]};
        float up[3] = {projection[1], projection[5], projection[9]};
        for (float* axis : {right, up}) {
            const float length=std::sqrt(axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2]);
            for (int c=0;c<3;++c) axis[c]/=std::max(length,0.00001f);
        }
        glUniform3fv(U("uViewRight"),1,right); glUniform3fv(U("uViewUp"),1,up);
        glUniform1f(U("uFogNear"),params.fogNear);
        glUniform3fv(U("uAmbient"),1,params.ambient);
        glUniform3fv(U("uLightDirection"),2,&params.lightDirection[0][0]);
        glUniform3fv(U("uLightColor"),2,&params.lightColor[0][0]);
        glUniform3fv(U("uFogColor"),1,params.fogColor);
        glUniform1i(U("uHDR"),hdr);
        glUniform1i(U("uPerPixel"),cvars->GetInteger("gNewDawn.Lighting.PerPixel",0));
        glUniform1f(U("uHdrAmbient"),std::clamp(cvars->GetFloat("gNewDawn.HDR.AmbientFill",0.4f),0.f,4.f));
        glUniform1f(U("uHdrSun"),std::clamp(cvars->GetFloat("gNewDawn.HDR.SunIntensity",1.8f),0.f,16.f));
        glUniform1f(U("uHdrFog"),std::clamp(cvars->GetFloat("gNewDawn.HDR.FogIntensity",1.f),0.f,4.f));
        glUniform1i(U("uTexture0"),8); glUniform1i(U("uTexture1"),9); glUniform1i(U("uTexture2"),10);
        for(uint32_t mi=0;mi<params.meshCount;++mi) {
            const auto& mesh=*params.meshes[mi]; glBindVertexArray(mMeshes.at(mesh.id).vao);
            for(uint32_t di=0;di<mesh.drawCount;++di) {
                const auto& draw=mesh.draws[di]; const auto& mat=draw.material;
                if((mat.blend!=0)!=(params.pass!=0)) continue;
                State::Set(GL_DEPTH_TEST,mat.depthTest); glDepthMask(mat.depthWrite?GL_TRUE:GL_FALSE);
                glDepthFunc(mat.depthFunction);
                State::Set(GL_CULL_FACE,mat.cull!=0); glCullFace(mat.cull==2?GL_FRONT:GL_BACK);
                glFrontFace(flipY<0?GL_CW:GL_CCW);
                State::Set(GL_BLEND,mat.blend!=0);
                if(mat.blend) {
                    glBlendFuncSeparate(mat.blendRgb[0],mat.blendRgb[1],mat.blendAlpha[0],mat.blendAlpha[1]);
                    glBlendEquationSeparate(mat.blendRgb[2],mat.blendAlpha[2]);
                    glBlendColor(mat.blendConstant[0],mat.blendConstant[1],mat.blendConstant[2],mat.blendConstant[3]);
                }
                glUniform1i(U("uAlphaTest"),mat.alphaTest); glUniform1i(U("uAlphaFunction"),mat.alphaFunction);
                glUniform1f(U("uAlphaReference"),mat.alphaReference); glUniform1i(U("uFog"),mat.fog);
                glUniform1i(U("uFragmentLighting"),mat.fragmentLighting);
                glUniform4fv(U("uDiffuse"),1,mat.diffuse); glUniform4fv(U("uMaterialAmbient"),1,mat.ambient);
                glUniform4fv(U("uEmission"),1,mat.emission); glUniform4fv(U("uConstant"),6,&mat.constants[0][0]);
                glUniform1i(U("uStageCount"),mat.combinerCount);
                glUniform1iv(U("uStages"),120,&mat.combiners[0][0]);
                for(int t=0;t<3;++t) {
                    const auto& binding=mat.bindings[t]; const std::string suffix="["+std::to_string(t)+"]";
                    glUniform1i(U("uHasTexture"+suffix),binding.texture>=0);
                    glUniform1i(U("uSource"+suffix),binding.source); glUniform1i(U("uMapping"+suffix),binding.mapping);
                    float uv[4]={binding.transform[0],binding.transform[1],binding.transform[2],binding.transform[3]};
                    if(scroll && mat.animSegment>=8 && mat.animSegment<16 && t<2 && binding.texture>=0) {
                        // Texture coords are scale*(uv-translate): a shift of f periods needs translate += f/scale.
                        const float* o=scroll->offset[mat.animSegment][t];
                        if(uv[0]!=0.f) uv[2]+=o[0]/uv[0];
                        if(uv[1]!=0.f) uv[3]-=o[1]/uv[1]; // the shader flips V: N64 t grows downward, MM3D v upward
                    }
                    glUniform4fv(U("uUVScaleTranslate"+suffix),1,uv);
                    glUniform1f(U("uUVRotation"+suffix),binding.transform[4]);
                    glActiveTexture(GL_TEXTURE8+t); glBindSampler(8+t,0);
                    if(binding.texture<0) { glBindTexture(GL_TEXTURE_2D,0); continue; }
                    glBindTexture(GL_TEXTURE_2D,mTextures.at(mesh.textures[binding.texture].id));
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,mipmaps?GL_LINEAR_MIPMAP_LINEAR:GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,Wrap(binding.wrapS));
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,Wrap(binding.wrapT));
                    if(anisotropy>=1) glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAX_ANISOTROPY_EXT,anisotropy);
                    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_LOD_BIAS,lodBias);
                }
                glDrawElements(GL_TRIANGLES,draw.indexCount,GL_UNSIGNED_INT,
                               reinterpret_cast<void*>(draw.firstIndex*sizeof(uint32_t)));
            }
        }
    }
    if(query) { glEndQuery(GL_TIME_ELAPSED); mQueryPending[mQuerySlot]=true; }
    if(timing) {
        mCpuMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        mQuerySlot=(mQuerySlot+1)%mQueries.size();
    }
    if(cvars->GetInteger("gNewDawn.MM3D.Debug.Inspect",0))
        for(GLenum error=glGetError();error!=GL_NO_ERROR;error=glGetError()) SPDLOG_ERROR("MM3D draw: GL error 0x{:X}",error);
}

bool GfxRenderingAPIOGL::MM3DAvailable() const { return true; }
bool GfxRenderingAPIOGL::PrepareMM3DRoom(const NewDawnMM3DRoomParams& params) {
    if (!mMM3D) mMM3D=std::make_unique<MM3DGL>();
    return mMM3D->Prepare(params);
}
void GfxRenderingAPIOGL::DrawMM3DRoom(const NewDawnMM3DRoomParams& params,const float* projection,float aspect,float flipY,
                                      const NewDawnMM3DScroll* scroll) {
    mMM3D->Draw(params,projection,aspect,flipY,HdrWorldActive(),mMipmapsEnabled,
                 mMaxAnisotropy > 1.0f ? std::min(mRequestedAnisotropy,mMaxAnisotropy) : 0.0f,mLodBias,scroll);
}
} // namespace Fast
#elif defined(ENABLE_OPENGL)
#include "fast/backends/gfx_opengl.h"
namespace Fast {
bool GfxRenderingAPIOGL::MM3DAvailable() const { return false; }
bool GfxRenderingAPIOGL::PrepareMM3DRoom(const NewDawnMM3DRoomParams&) { return false; }
void GfxRenderingAPIOGL::DrawMM3DRoom(const NewDawnMM3DRoomParams&,const float*,float,float,const NewDawnMM3DScroll*) {}
}
#endif
