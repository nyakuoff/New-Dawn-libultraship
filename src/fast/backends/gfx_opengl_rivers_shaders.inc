// Rivers and waterfalls (docs/WATER_SIM.md). Included by gfx_opengl_postfx.cpp.
//  1. Flow buffer: the recorded river triangles are re-rasterised from the camera into an RG16F target (world xz velocity of
//     the native texture scroll, 1 = covered). The surface pass reads it per pixel to advect its streak normals and foam.
//  2. Waterfalls: the recorded fall triangles get a second, flowing layer of streaks, sun glints and foam on top of the
//     native texture; depth-tested by hand against the scene depth.
//  3. Mist: instanced soft billboards at the foot of each fall (rising mist, ballistic spray that catches the sun, a ragged
//     foam disc on the pool), seeded and animated analytically like the atmosphere particles.

// Lit and flowing: two-phase flow-map sampling of the tileable noise. The noise is blurred along the flow direction with a
// handful of taps (a linear operation, so the slope of the blurred height is the blur of the stored slope), which stretches
// it into streaks without ever rotating the sampling frame (a rotation about the world origin would jump at every triangle
// whose direction differs). Returns xy = slope, z = streak mask.
static const char* kRiverFlowLib = R"GLSL(
uniform float uTime;
uniform sampler2D uFlowNoise;
const float RIVER_TILE = 150.0; // world units per noise tile
vec3 flowSurface(vec2 x, vec2 f, vec2 gx, vec2 gy) {
    float speed = length(f);
    vec2 dir = speed > 1e-3 ? f / speed : vec2(1.0, 0.0);
    const float CYCLE = 2.6;
    float T = uTime / CYCLE;
    float blurLen = clamp(speed * 1.1, 14.0, 80.0);
    vec3 acc = vec3(0.0);
    for (int k = 0; k < 2; k++) {
        float ph = fract(T + 0.5 * float(k));
        float w = 1.0 - abs(2.0 * ph - 1.0);
        vec2 o = x - f * (ph * CYCLE) + float(k) * vec2(53.7, 29.1);
        vec3 a = vec3(0.0);
        for (int j = 0; j < 5; j++) {
            float s = (float(j) - 2.0) * 0.25 * blurLen;
            vec4 t = textureGrad(uFlowNoise, (o + dir * s) / RIVER_TILE, gx / RIVER_TILE, gy / RIVER_TILE);
            a += vec3(t.rg * 2.0 - 1.0, t.a);
        }
        acc += w * a * 0.2;
    }
    return acc;
}
)GLSL";

static const char* kRiverFlowVertex = R"GLSL(
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aFlow;
uniform mat4 uViewProj;
uniform float uNdcXScale;
uniform float uFlipY;
flat out vec3 vFlow;
void main() {
    gl_Position = uViewProj * vec4(aPos, 1.0);
    gl_Position.x /= uNdcXScale;
    gl_Position.y *= uFlipY;
    vFlow = aFlow;
}
)GLSL";

static const char* kRiverFlowFragment = R"GLSL(
flat in vec3 vFlow;
out vec4 oColor;
void main() {
    oColor = vec4(vFlow.xz, 1.0, 0.0);
}
)GLSL";

// Falls: same vertex stage; the world position and the downward speed reach the fragment stage.
static const char* kFallVertex = R"GLSL(
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aFlow;
uniform mat4 uViewProj;
uniform float uNdcXScale;
uniform float uFlipY;
out vec3 vWorld;
flat out vec3 vFlow;
void main() {
    gl_Position = uViewProj * vec4(aPos, 1.0);
    gl_Position.x /= uNdcXScale;
    gl_Position.y *= uFlipY;
    vWorld = aPos;
    vFlow = aFlow;
}
)GLSL";

static const char* kFallFragment = R"GLSL(
in vec3 vWorld;
flat in vec3 vFlow;
out vec4 oColor;
uniform sampler2D uVisibleDepth;
uniform float uNear, uFar, uStrength, uPixelAngle;
uniform int uDebug; // 2: paint every recorded face (magenta falls, cyan streams) to find them
uniform int uMode; // 0 waterfall face, 1 painted stream (a river mesh with no WaterBox)
uniform float uFogMul, uFogOff;
uniform vec3 uFogColor, uEye, uLightDir, uLightColor, uSky;
float lin(float d) {
    return 2.0 * uNear * uFar / max(uFar + uNear - (2.0 * d - 1.0) * (uFar - uNear), 1e-3);
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float vd = texelFetch(uVisibleDepth, p, 0).r;
    float me = lin(gl_FragCoord.z);
    if (me > lin(vd) * 1.012 + 3.0) {
        discard; // an actor, bridge or bank is in front of this part of the fall
    }
    if (uDebug == 2) {
        oColor = uMode == 1 ? vec4(0.0, 0.8, 0.8, 0.8) : vec4(0.9, 0.0, 0.9, 0.9);
        return;
    }
    vec3 V = normalize(uEye - vWorld);
    if (uMode == 1) {
        // Painted stream: the same flow-mapped streaks as the river surface pass, added as sun glints and foam over the
        // game's own water texture (which already scrolls the right way).
        float footprintR = length(uEye - vWorld) * uPixelAngle / max(V.y, 0.06);
        vec3 rs = flowSurface(vWorld.xz, vFlow.xz, dFdx(vWorld.xz), dFdy(vWorld.xz));
        vec3 Nr = normalize(vec3(-rs.x * 0.45, 1.0, -rs.y * 0.45));
        vec3 Hr = normalize(uLightDir + V);
        float glintR = pow(max(dot(Nr, Hr), 0.0), 48.0);
        float foamR = smoothstep(0.55, 0.95, rs.z) * (1.0 - smoothstep(4.0, 14.0, footprintR));
        float fade = 1.0 - smoothstep(8.0, 34.0, footprintR);
        float sp = clamp(length(vFlow.xz) / 45.0, 0.0, 1.0);
        float aR = clamp(foamR * (0.25 + 0.45 * sp) + glintR * 0.55 * fade, 0.0, 0.8) * uStrength;
        vec3 colR = mix(uLightColor * (glintR * 2.0 * fade) + uSky * 0.25, (uSky * 0.9 + uLightColor * 0.45), foamR);
        float ndczR = clamp(gl_FragCoord.z * 2.0 - 1.0, -1.0, 1.0);
        float fogR = clamp(ndczR * uFogMul + uFogOff, 0.0, 255.0) / 255.0;
        colR = mix(colR, uFogColor, fogR);
        aR *= 1.0 - 0.9 * fogR;
        oColor = vec4(colR * aR, aR);
        return;
    }
    vec3 N = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    if (dot(N, V) < 0.0) {
        N = -N;
    }
    vec3 T = normalize(vec3(-N.z, 0.0, N.x) + vec3(1e-5, 0.0, 0.0)); // horizontal along the fall face
    float s = dot(vWorld, T);
    float y = vWorld.y;
    float v = max(length(vFlow), 90.0);
    float dist = length(uEye - vWorld);
    float footprint = dist * uPixelAngle;
    const float CYCLE = 1.7;
    float Tm = uTime / CYCLE;
    vec3 acc = vec3(0.0);
    vec2 g = vec2(1.0 / 70.0, 1.0 / 340.0);
    for (int k = 0; k < 2; k++) {
        float ph = fract(Tm + 0.5 * float(k));
        float w = 1.0 - abs(2.0 * ph - 1.0);
        vec2 uv = vec2(s / 70.0 + 0.37 * float(k), (y + v * ph * CYCLE) / 340.0 + 0.61 * float(k));
        vec2 dx = vec2(dFdx(s), dFdx(y)) * g, dy = vec2(dFdy(s), dFdy(y)) * g;
        vec4 t = textureGrad(uFlowNoise, uv, dx, dy);
        acc += w * vec3(t.rg * 2.0 - 1.0, t.a);
    }
    float streak = clamp(acc.z, 0.0, 1.0);
    // Perturbed normal: slope across (along T) and vertical.
    vec3 Nw = normalize(N + T * acc.x * 0.55 + vec3(0.0, acc.y * 0.25, 0.0));
    float ndl = abs(dot(Nw, uLightDir));
    vec3 H = normalize(uLightDir + V);
    float glint = pow(max(dot(Nw, H), 0.0), 36.0) * (0.4 + 0.6 * streak);
    float back = pow(max(dot(-V, uLightDir), 0.0), 3.0); // light through the thin sheet of water
    vec3 albedo = mix(vec3(0.55, 0.74, 0.88), vec3(0.85, 0.94, 1.0), streak); // darker blue channels between bright streaks
    vec3 col = albedo * (uSky * 0.75 + uLightColor * (0.18 + 0.5 * ndl) + uLightColor * (0.7 * back)) + uLightColor * glint * 2.6;
    // Ragged vertical foam bands: strongest in the thick ridges of the streak mask.
    float foam = smoothstep(0.45, 0.9, streak);
    col = mix(col, vec3(1.35) * (uSky * 0.9 + uLightColor * 0.5), foam * 0.7);
    float a = clamp(0.38 + 0.5 * streak + 0.12 * (0.5 + 0.5 * acc.x), 0.0, 0.92) * uStrength;
    a *= 1.0 - smoothstep(3.0, 10.0, footprint); // distant falls just keep their native texture
    float ndcz = clamp(gl_FragCoord.z * 2.0 - 1.0, -1.0, 1.0);
    float fogF = clamp(ndcz * uFogMul + uFogOff, 0.0, 255.0) / 255.0;
    col = mix(col, uFogColor, fogF);
    a *= 1.0 - 0.9 * fogF;
    oColor = vec4(col * a, a);
}
)GLSL";

static const char* kMistVertex = R"GLSL(
uniform mat4 uViewProj;
uniform float uNdcXScale, uFlipY;
uniform vec2 uProjection;
uniform float uTime;
uniform int uCount;
uniform vec4 uEmit[48];  // xyz base of the fall, w = energy
uniform vec4 uEmitB[48]; // xyz = horizontal direction away from the face, w = half width
out vec2 vUv;
out vec3 vWorld;
out float vAlpha, vViewZ, vKind, vSeed;
uint hash(uint n) { n ^= n >> 16u; n *= 0x7feb352du; n ^= n >> 15u; n *= 0x846ca68bu; return n ^ (n >> 16u); }
float rnd(uint n) { return float(hash(n) & 0xffffffu) / 16777216.0; }
const int PER = 30; // 20 mist puffs, 8 spray droplets, 2 foam discs
void main() {
    int e = gl_InstanceID / PER;
    int k = gl_InstanceID % PER;
    vAlpha = 0.0;
    if (e >= uCount) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    uint id = uint(gl_InstanceID) * 7u + 11u;
    vec3 sd = vec3(rnd(id * 3u + 1u), rnd(id * 3u + 2u), rnd(id * 3u + 3u));
    vec4 E = uEmit[e];
    vec3 out_ = uEmitB[e].xyz;
    vec3 side = vec3(-out_.z, 0.0, out_.x);
    float hw = uEmitB[e].w;
    float en = E.w;
    vec2 corners[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
    vec2 q = corners[gl_VertexID];
    vUv = q * 0.5 + 0.5;
    vSeed = sd.x;
    vec3 p;
    float radius;
    bool flat_ = false;
    if (k < 20) {
        // mist: rises and spreads from the pool surface at the foot of the fall
        vKind = 0.0;
        float life = 3.4 + 1.6 * sd.y;
        float age = fract(uTime / life + sd.x);
        float spread = (sd.z - 0.5) * 2.0 * hw;
        p = E.xyz + side * spread + out_ * (6.0 + age * 34.0 * en + sd.y * 10.0) + vec3(0.0, 6.0 + age * 62.0 * (0.6 + sd.z), 0.0);
        p += side * sin(uTime * 0.7 + sd.x * 40.0) * 5.0;
        radius = (12.0 + 46.0 * age) * (0.7 + 0.6 * sd.y) * (0.6 + 0.5 * en);
        vAlpha = sin(3.14159 * age) * 0.34 * min(en, 1.2);
    } else if (k < 28) {
        // spray: ballistic droplets thrown out of the impact
        vKind = 1.0;
        float life = 0.8 + 0.5 * sd.y;
        float t = fract(uTime / life + sd.x) * life;
        float spread = (sd.z - 0.5) * 2.0 * hw;
        vec3 v0 = out_ * (24.0 + 46.0 * sd.y) * en + side * (sd.x - 0.5) * 40.0 + vec3(0.0, 70.0 + 110.0 * sd.z, 0.0) * en;
        p = E.xyz + side * spread + vec3(0.0, 3.0, 0.0) + v0 * t - vec3(0.0, 0.5 * 280.0 * t * t, 0.0);
        radius = 1.6 + 2.6 * sd.y;
        vAlpha = (1.0 - t / life) * smoothstep(0.0, 0.08, t) * 0.9 * min(en, 1.2);
    } else {
        // foam disc: ragged, pulsing white water on the pool surface
        vKind = 2.0;
        flat_ = true;
        p = E.xyz + side * ((float(k - 28) - 0.5) * hw * 0.8) + out_ * 22.0 + vec3(0.0, 1.0, 0.0);
        radius = (40.0 + 16.0 * sin(uTime * 1.3 + float(k))) * (0.7 + 0.45 * en);
        vAlpha = 0.5 * min(en, 1.2);
    }
    vec4 c = uViewProj * vec4(p, 1.0);
    c.x /= uNdcXScale;
    c.y *= uFlipY;
    if (flat_) {
        vec3 pq = p + vec3(q.x, 0.0, q.y) * radius;
        c = uViewProj * vec4(pq, 1.0);
        c.x /= uNdcXScale;
        c.y *= uFlipY;
        vWorld = pq;
    } else {
        c.xy += q * radius * uProjection;
        vWorld = p;
    }
    gl_Position = c;
    vViewZ = c.w;
}
)GLSL";

static const char* kMistFragment = R"GLSL(
in vec2 vUv;
in vec3 vWorld;
in float vAlpha, vViewZ, vKind, vSeed;
out vec4 oColor;
uniform sampler2D uSprite;
uniform sampler2D uVisibleDepth;
uniform sampler2D uFlowNoise;
uniform vec2 uSize;
uniform float uNear, uFar, uStrength, uTime;
uniform float uFogMul, uFogOff;
uniform vec3 uFogColor, uEye, uLightDir, uLightColor, uSky;
float lin(float d) {
    return 2.0 * uNear * uFar / max(uFar + uNear - (2.0 * d - 1.0) * (uFar - uNear), 1e-3);
}
void main() {
    if (vAlpha < 0.002 || vViewZ <= 0.0) {
        discard;
    }
    float scene = lin(texture(uVisibleDepth, gl_FragCoord.xy / uSize).r);
    float soft = clamp((scene - vViewZ) / (vKind < 0.5 ? 26.0 : (vKind < 1.5 ? 4.0 : 10.0)), 0.0, 1.0);
    float sprite = texture(uSprite, vUv).r * texture(uSprite, vUv).a;
    vec3 V = normalize(vWorld - uEye);
    float cosSun = dot(V, uLightDir); // looking towards the sun: the water scatters it forward
    float fwd = pow(max(cosSun, 0.0), 6.0);
    float alpha = vAlpha * soft * uStrength;
    vec3 col;
    if (vKind < 0.5) {
        alpha *= sprite;
        col = uSky * 0.8 + uLightColor * (0.28 + 2.2 * fwd);
        float n = texture(uFlowNoise, vWorld.xz / 90.0 + vWorld.y / 140.0 + vec2(uTime * 0.02, 0.0)).b;
        alpha *= 0.55 + 0.9 * n;
    } else if (vKind < 1.5) {
        alpha *= smoothstep(0.0, 0.5, sprite);
        col = uSky * 0.6 + uLightColor * (0.7 + 6.0 * fwd);
    } else {
        // ragged disc: the streak mask decides where the foam is
        float m = texture(uFlowNoise, vWorld.xz / 60.0 + vec2(uTime * 0.05, -uTime * 0.03)).a;
        float r = length(vUv - 0.5) * 2.0;
        alpha *= (1.0 - smoothstep(0.35, 1.0, r)) * smoothstep(0.1, 0.6, m + 0.3 * sprite);
        col = uSky * 0.85 + uLightColor * 0.45;
    }
    float ndcz = clamp(gl_FragCoord.z * 2.0 - 1.0, -1.0, 1.0);
    float fogF = clamp(ndcz * uFogMul + uFogOff, 0.0, 255.0) / 255.0;
    col = mix(col, uFogColor, fogF);
    alpha *= 1.0 - 0.85 * fogF;
    oColor = vec4(col * alpha, alpha);
}
)GLSL";
