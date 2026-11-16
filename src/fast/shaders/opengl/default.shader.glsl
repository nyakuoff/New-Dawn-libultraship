@prism(type='fragment', name='Fast3D Fragment Shader', version='1.0.0', description='Ported shader to prism', author='Emill & Prism Team')

@{GLSL_VERSION}

@if(VERTEX_SHADER)
    @{attr} vec4 aVtxPos;

    @if(o_per_pixel_lighting)
        @{attr} vec3 aNormal;
        @{attr} vec3 aViewPos;
        @{out} vec3 vNormal;
        @{out} vec3 vViewPos;
        @{update_floats(3)}
        @{update_floats(3)}
    @end
    @if(o_unlit_material)
        @{attr} vec3 aViewPos;
        @{attr} vec3 aNormal;
        @{out} vec3 vViewPos;
        @{out} vec3 vNormal;
        @{update_floats(3)}
        @{update_floats(3)}
    @end
    @if(o_smooth)
        @{attr} vec3 aSmooth0;
        @{attr} vec3 aSmooth1;
        @{out} vec3 vSmooth0;
        @{out} vec3 vSmooth1;
        @{update_floats(3)}
        @{update_floats(3)}
    @end

    @for(i in 0..2)
        @if(o_textures[i])
            @{attr} vec2 aTexCoord@{i};
            @{out} vec2 vTexCoord@{i};
            @{update_floats(2)}
            @for(j in 0..2)
                @if(o_clamp[i][j])
                    @if(j == 0)
                        @{attr} float aTexClampS@{i};
                        @{out} float vTexClampS@{i};
                    @else
                        @{attr} float aTexClampT@{i};
                        @{out} float vTexClampT@{i};
                    @end
                    @{update_floats(1)}
                @end
            @end
        @end
    @end

    @if(o_fog)
        @{attr} vec4 aFog;
        @{out} vec4 vFog;
        @{update_floats(4)}
    @end

    @if(o_grayscale)
        @{attr} vec4 aGrayscaleColor;
        @{out} vec4 vGrayscaleColor;
        @{update_floats(4)}
    @end

    @for(i in 0..o_inputs)
        @if(o_alpha)
            @{attr} vec4 aInput@{i + 1};
            @{out} vec4 vInput@{i + 1};
            @{update_floats(4)}
        @else
            @{attr} vec3 aInput@{i + 1};
            @{out} vec3 vInput@{i + 1};
            @{update_floats(3)}
        @end
    @end

    void main() {
        @for(i in 0..2)
            @if(o_textures[i])
                vTexCoord@{i} = aTexCoord@{i};
                @for(j in 0..2)
                    @if(o_clamp[i][j])
                        @if(j == 0)
                            vTexClampS@{i} = aTexClampS@{i};
                        @else
                            vTexClampT@{i} = aTexClampT@{i};
                        @end
                    @end
                @end
            @end
        @end
        @if(o_fog)
            vFog = aFog;
        @end
        @if(o_grayscale)
            vGrayscaleColor = aGrayscaleColor;
        @end
        @for(i in 0..o_inputs)
            vInput@{i + 1} = aInput@{i + 1};
        @end
        @if(o_per_pixel_lighting)
            vNormal = aNormal;
            vViewPos = aViewPos;
        @end
        @if(o_unlit_material)
            vViewPos = aViewPos;
            vNormal = aNormal;
        @end
        @if(o_smooth)
            vSmooth0 = aSmooth0;
            vSmooth1 = aSmooth1;
        @end
        gl_Position = aVtxPos;
        @if(opengles)
            gl_Position.z *= 0.3f;
        @end
    }
@else
    @if(core_opengl || opengles)
    out vec4 vOutColor;
    @end

    @for(i in 0..2)
        @if(o_textures[i])
            @{attr} vec2 vTexCoord@{i};
            @for(j in 0..2)
                @if(o_clamp[i][j])
                    @if(j == 0)
                        @{attr} float vTexClampS@{i};
                    @else
                        @{attr} float vTexClampT@{i};
                    @end
                @end
            @end
        @end
    @end

    @if(o_fog) @{attr} vec4 vFog;
    @if(o_grayscale) @{attr} vec4 vGrayscaleColor;

    @for(i in 0..o_inputs)
        @if(o_alpha)
            @{attr} vec4 vInput@{i + 1};
        @else
            @{attr} vec3 vInput@{i + 1};
        @end
    @end

    uniform bool uHdrWorld;
    uniform float uHdrSun;
    uniform float uHdrAmbient;
    uniform float uHdrSky;
    uniform float uHdrFog;
    @if(o_alpha)
        // New Dawn foliage light (render only). Frame: eye | sun direction, strength | sun radiance; the eye's w is 1
        // when uFolDraw.yzw is the origin of the actor the leaves belong to. Draw: on | origin. View: game viewport.
        uniform vec4 uFolFrame[3];
        uniform vec4 uFolDraw;
        uniform vec4 uFolView;
        uniform mat4 uFolInv;
    @end
    vec3 hdrDecode(vec3 c) {
        c = max(c, vec3(0.0));
        return mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, lessThanEqual(c, vec3(0.04045)));
    }

    @if(o_per_pixel_lighting)
        @{attr} vec3 vNormal;
        @{attr} vec3 vViewPos;

        #define NEWDAWN_MAX_LIGHTS 32

        uniform int uLightCount;
        uniform vec3 uLightAmbient;
        uniform vec3 uLightDir[NEWDAWN_MAX_LIGHTS];
        uniform vec3 uLightColor[NEWDAWN_MAX_LIGHTS];
        uniform vec3 uLightPos[NEWDAWN_MAX_LIGHTS];
        uniform vec2 uLightAtten[NEWDAWN_MAX_LIGHTS];
        uniform int uLightPositional[NEWDAWN_MAX_LIGHTS];
        uniform mat3 uViewMatrix;
        uniform int uShadeMask;
        uniform float uPointLightQuality;

#ifndef GL_ES
        uniform bool uSkyAmbientEnabled;
        uniform samplerCube uSkyIrradiance;
        vec3 skyAmbientColor(vec3 baseline, vec3 N) {
            if (!uSkyAmbientEnabled) return baseline;
            float y = dot(baseline,vec3(0.2126,0.7152,0.0722));
            return texture(uSkyIrradiance,N).rgb*y;
        }
        uniform bool uSkyReflectionsEnabled;
        uniform samplerCube uSkySpecular;
        uniform vec3 uSkyEye;

#endif

        // Same light model as the CPU path in GfxSpVertex: this replaces the per-vertex shade
        // colour with the same arithmetic evaluated per fragment. Values are kept in the 0..255
        // range the CPU uses, including its per-light truncation, so parity is exact where the
        // interpolated normal and position are constant across a triangle.
        vec3 newDawnIrradianceEx(vec3 shadeNormal, out vec3 ambientPart, float wrap, out vec3 wrapExtra, out vec3 pointPart) {
            vec3 n = shadeNormal / max(length(shadeNormal), 0.0001);
            vec3 lit = (uLightAmbient / 255.0) * uHdrAmbient;
#ifndef GL_ES
            if (uSkyAmbientEnabled) {
                // MM loads view into P; uViewMatrix here is the model-to-world transform.
                // The cofactor handles scaled actors/bones and the existing material normal path.
                mat3 cof = mat3(cross(uViewMatrix[1],uViewMatrix[2]),
                                cross(uViewMatrix[2],uViewMatrix[0]),
                                cross(uViewMatrix[0],uViewMatrix[1]));
                float det = dot(uViewMatrix[0],cof[0]);
                vec3 wn = cof * n * (det<0.0 ? -1.0 : 1.0);
                wn = length(wn)>1e-6 ? normalize(wn) : vec3(0.0,1.0,0.0);
                lit = skyAmbientColor(lit,wn);
            }
#endif
            ambientPart = max(lit, vec3(0.0));
            wrapExtra = vec3(0.0);
            pointPart = vec3(0.0);
            for (int i = 0; i < uLightCount; ++i) {
                float intensity;
                if (uLightPositional[i] != 0) {
                    vec3 d = uLightPos[i] - vViewPos;
                    float d2 = max(dot(d.xy, d.xy) + 2.0 * d.z * d.z, 0.000001);
                    float dist = sqrt(d2);
                    vec3 lm = transpose(uViewMatrix) * d;
                    float attenuation = 1.0 +
                        (dist * uLightAtten[i].x * 2.0 + d2 * uLightAtten[i].y / 8.0) / 65535.0;
                    intensity = clamp(dot(n, clamp(508.0 * lm / d2, -127.0, 127.0)), 0.0, 1.0) / attenuation;
                    pointPart += intensity * uLightColor[i] / 255.0;
                } else {
                    float nl = dot(n, uLightDir[i]);
                    intensity = max(nl, 0.0) * uHdrSun;
                    // Wrapped diffuse (characters): the terminator is soft, the part of the wrap that the plain
                    // Lambert term lacks is returned separately so the caller can colour it (subsurface).
                    if (wrap > 0.0) {
                        wrapExtra += (max((nl + wrap) / (1.0 + wrap), 0.0) * uHdrSun - intensity) * uLightColor[i] / 255.0;
                    }
                }
                lit += intensity * uLightColor[i] / 255.0;
            }
            return max(lit, vec3(0.0));
        }

        vec3 newDawnIrradiance(vec3 shadeNormal) {
            vec3 ambientPart, wrapExtra, pointPart;
            return newDawnIrradianceEx(shadeNormal, ambientPart, 0.0, wrapExtra, pointPart);
        }

        vec3 newDawnShade(vec3 shadeNormal) {
            vec3 lit = uLightAmbient;
            for (int i = 0; i < uLightCount; i++) {
                float intensity = 0.0;
                if (uLightPositional[i] != 0) {
                    vec3 distVec = uLightPos[i] - vViewPos;
                    // The extra *2 on z comes from the CPU reference (and GLideN64).
                    float distSq = max(distVec.x * distVec.x + distVec.y * distVec.y + distVec.z * distVec.z * 2.0,
                                       0.000001);
                    float dist = sqrt(distSq);
                    vec3 lightModel = transpose(uViewMatrix) * distVec;
                    vec3 lightIntensity = clamp(4.0 * lightModel / distSq, -1.0, 1.0);
                    float total = clamp(dot(shadeNormal, lightIntensity), -1.0, 1.0);
                    // uPointLightQuality 0 keeps the CPU's integer distance steps.
                    float distf = mix(floor(dist), dist, uPointLightQuality);
                    float attenuation =
                        (distf * uLightAtten[i].x * 2.0 + distf * distf * uLightAtten[i].y / 8.0) / 65535.0 + 1.0;
                    intensity = total / attenuation;
                } else {
                    intensity = dot(shadeNormal, uLightDir[i]) / 127.0;
                }
                lit += floor(max(intensity, 0.0) * uLightColor[i]);
            }
            return clamp(floor(lit), 0.0, 255.0) / 255.0;
        }

        // New Dawn character lighting. xyz: camera position in the same (world) space as vViewPos; w: strength,
        // 0 for room meshes, the HUD and everything while the feature is off.
        uniform vec4 uCharLight;

        // New Dawn modern character lighting (docs/CHARACTER_LIGHTING.md). x: how far the scene's light colour is moved to a
        // daylight white balance on actors (0 = the scene's light as it is), y: strength of the per-channel highlight shoulder,
        // z: debug view of the applied light (1 total, 2 ambient and sky ambient, 3 lights, 4 applied), w: diffuse wrap.
        uniform vec4 uCharGrade;
        uniform vec4 uCharMat;

        // Same brightness, less hue, and a daylight white balance: the scene's warm sun and warm ambient multiplied onto
        // saturated cloth and skin clip the blue channel, which reads as yellow skin and acid green. The key moves towards a
        // warm white, the fill towards a cool one (a remake's sun and sky), each keeping its brightness (Rec.709 luma) and
        // direction. Local point lights (torches, fires) keep their colour.
        vec3 charWhiteBalance(vec3 irradiance, vec3 target, float amount) {
            float y = dot(irradiance, vec3(0.2126, 0.7152, 0.0722));
            return mix(irradiance, y * target, amount);
        }

        // Per-channel soft shoulder on the lit actor radiance (scene-linear, before the tonemap): below the knee nothing
        // changes, above it each channel rolls off towards knee + 1, so a saturated channel stops running away from the
        // other two (acid greens, hot reds); mid-tones are untouched.
        vec3 charShoulder(vec3 c, float amount) {
            const float knee = 0.55;
            vec3 over = max(c - vec3(knee), vec3(0.0));
            vec3 rolled = min(c, vec3(knee)) + over / (1.0 + over);
            // And a soft floor under the weakest channel (17% of the luma): a saturated texel keeps a trace of every channel, as
            // light scattering in cloth and skin leaves, instead of a pure primary after the display transform.
            float y = dot(rolled, vec3(0.2126, 0.7152, 0.0722));
            vec3 lifted = rolled + max(vec3(0.17 * y) - rolled, vec3(0.0)) * 0.8;
            return mix(c, lifted, amount);
        }

        // Additive linear radiance for lit actor draws: a soft Fresnel rim (sky behind the subject, the scene's own
        // light colours indoors, sun when backlit) and a bounded specular from the same lights. Actor meshes are
        // low-poly, so the rim, the wrap and everything that is not metal read from a softened normal and broad lobes
        // (no facet edges); only bright neutral (metal) texels get a crisp highlight, tinted by their own albedo, plus a
        // reflection of the sky. Both terms are tied to the albedo so dark texels stay dark, and skin hues get no specular.
        // New Dawn character materials. uCharMat: x class of the texture (0 none, 1 skin, 2 costume, 3 hair, 4 eyes and mouth,
        // 5 metal, 6 cloth, 7 leather), y strength, z relief. Additive linear radiance on top of the lit albedo: relief taken
        // from the albedo's own luminance (screen-space derivatives, so no tangents are needed), skin that scatters light red
        // at the terminator and when backlit, cloth sheen, leather and skin sheen lobes, anisotropic hair bands. A costume atlas
        // (one texture with tunic, belt, boots and skin) is split into those by colour.
        vec3 newDawnCharMaterial(vec3 albedoLin, vec3 albedoGamma, vec3 shadeNormal, vec3 ambientIrr) {
            int cls = int(uCharMat.x + 0.5);
            if (cls <= 0 || cls == 4 || uCharMat.y <= 0.0) return vec3(0.0);
            // Derivatives first, before any per-fragment early return.
            vec3 p = vViewPos;
            vec3 dpx = dFdx(p), dpy = dFdy(p);
            float hx = dFdx(dot(albedoGamma, vec3(0.2126, 0.7152, 0.0722))), hy = dFdy(dot(albedoGamma, vec3(0.2126, 0.7152, 0.0722)));
            vec3 n = shadeNormal / max(length(shadeNormal), 0.0001);
            mat3 cof = mat3(cross(uViewMatrix[1], uViewMatrix[2]), cross(uViewMatrix[2], uViewMatrix[0]), cross(uViewMatrix[0], uViewMatrix[1]));
            float det0 = dot(uViewMatrix[0], cof[0]);
            vec3 wn = cof * n * (det0 < 0.0 ? -1.0 : 1.0);
            wn = length(wn) > 1e-6 ? normalize(wn) : vec3(0.0, 1.0, 0.0);
            vec3 v = uCharLight.xyz - vViewPos;
            float vl = length(v);
            if (vl < 1.0) return vec3(0.0);
            v /= vl;
            float fade = 1.0 - smoothstep(250.0, 700.0, vl);

            // Class weights.
            float mx = max(albedoGamma.r, max(albedoGamma.g, albedoGamma.b));
            float mn = min(albedoGamma.r, min(albedoGamma.g, albedoGamma.b));
            float sat = (mx - mn) / max(mx, 0.001);
            float luma = dot(albedoGamma, vec3(0.2126, 0.7152, 0.0722));
            float wSkin = 0.0, wCloth = 0.0, wLeather = 0.0, wMetal = 0.0, wHair = 0.0;
            if (cls == 1) wSkin = 1.0;
            else if (cls == 3) wHair = 1.0;
            else if (cls == 5) wMetal = 1.0;
            else if (cls == 6) wCloth = 1.0;
            else if (cls == 7) wLeather = 1.0;
            else {
                float warm = step(albedoGamma.g, albedoGamma.r) * step(albedoGamma.b, albedoGamma.g);
                wSkin = warm * smoothstep(0.10, 0.20, sat) * (1.0 - smoothstep(0.50, 0.66, sat)) * smoothstep(0.30, 0.45, luma);
                float brown = warm * smoothstep(0.30, 0.45, sat) * (1.0 - smoothstep(0.40, 0.56, luma)) * smoothstep(0.04, 0.12, luma);
                wLeather = brown * (1.0 - wSkin);
                float gold = warm * smoothstep(0.50, 0.66, sat) * smoothstep(0.46, 0.60, luma);
                float grey = (1.0 - smoothstep(0.16, 0.34, sat)) * smoothstep(0.24, 0.5, luma) * (1.0 - smoothstep(0.70, 0.86, luma));
                wMetal = max(gold, grey) * (1.0 - wSkin);
                wCloth = max(1.0 - wSkin - wLeather - wMetal, 0.0);
            }

            // Relief from the albedo: the surface gradient of the luminance (Mikkelsen, bump mapping unparametrized surfaces).
            float classRelief = 1.0 * wCloth + 1.4 * wLeather + 0.8 * wHair + 0.45 * wMetal + 0.25 * wSkin;
            vec3 r1 = cross(dpy, wn), r2 = cross(wn, dpx);
            float det = dot(dpx, r1);
            vec3 grad = sign(det) * (hx * r1 + hy * r2);
            vec3 nb = normalize(abs(det) * wn - 0.32 * uCharMat.z * classRelief * grad * min(1.0, 80.0 / vl));
            if (!(abs(det) > 1e-12)) nb = wn;
            vec3 wnSoft = normalize(wn + 0.5 * v);
            vec3 nbSoft = normalize(nb + 0.5 * v);
            float ndv = clamp(dot(nbSoft, v), 0.0, 1.0);

            vec3 tint = albedoLin / max(max(albedoLin.r, max(albedoLin.g, albedoLin.b)), 0.002);
            vec3 add = vec3(0.0);
            int tm = int(uCharMat.w + 0.5);
            float tRelief = (tm & 1) != 0 ? 1.0 : 0.0, tSkin = (tm & 2) != 0 ? 1.0 : 0.0, tCloth = (tm & 4) != 0 ? 1.0 : 0.0,
                  tLeather = (tm & 8) != 0 ? 1.0 : 0.0, tHair = (tm & 16) != 0 ? 1.0 : 0.0, tSky = (tm & 32) != 0 ? 1.0 : 0.0;
            vec3 up = vec3(0.0, 1.0, 0.0);
            vec3 hairT = up - wn * dot(up, wn);
            hairT = length(hairT) > 0.05 ? normalize(hairT) : vec3(1.0, 0.0, 0.0);
            for (int i = 0; i < uLightCount; ++i) {
                if (uLightPositional[i] != 0) continue; // fires and torches colour the diffuse; no extra lobes from them
                vec3 l = normalize(uViewMatrix * uLightDir[i]);
                vec3 rad = uLightColor[i] / 255.0 * uHdrSun;
                rad = mix(rad, vec3(dot(rad, vec3(0.2126, 0.7152, 0.0722))), 0.5 * uCharGrade.x);
                float nl0 = dot(wn, l);
                float nl = dot(nb, l);
                vec3 h = normalize(l + v);
                float nh = max(dot(nb, h), 0.0);
                float lit = nl > 0.0 ? 1.0 : 0.0;
                // Relief shading of the diffuse term.
                add += tRelief * albedoLin * rad * (max(nl, 0.0) - max(nl0, 0.0)) * classRelief * uCharMat.z * 0.9;
                // Skin: light that scatters under the surface turns red at the terminator and glows through thin parts.
                float band = smoothstep(-0.30, 0.05, nl0) * (1.0 - smoothstep(0.05, 0.55, nl0));
                float back = pow(clamp(dot(-v, l), 0.0, 1.0), 2.5) * (1.0 - clamp(nl0, 0.0, 1.0));
                add += tSkin * wSkin * albedoLin * rad * (vec3(0.95, 0.32, 0.16) * band * 0.40 + vec3(1.0, 0.38, 0.20) * back * 0.45);
                add += tSkin * wSkin * rad * lit * (pow(nh, 12.0) * 0.030 + pow(nh, 50.0) * 0.035);
                // Cloth: a velvet-like sheen that grows towards the silhouette and follows the light.
                add += tCloth * wCloth * mix(albedoLin * 1.4, vec3(dot(albedoLin, vec3(0.2126, 0.7152, 0.0722)) * 1.4), 0.25) * rad *
                       pow(1.0 - ndv, 3.0) * smoothstep(-0.35, 0.5, nl0) * 0.55;
                // Leather: a broad, low highlight.
                add += tLeather * wLeather * mix(vec3(1.0), tint, 0.5) * rad * lit * (pow(nh, 18.0) * 0.12 + pow(nh, 70.0) * 0.06);
                // Hair: two anisotropic bands (Kajiya-Kay) along the strands, the second tinted by the hair.
                vec3 t1 = normalize(hairT + wn * 0.18), t2 = normalize(hairT - wn * 0.28);
                float s1 = sqrt(max(1.0 - dot(t1, h) * dot(t1, h), 0.0)), s2 = sqrt(max(1.0 - dot(t2, h) * dot(t2, h), 0.0));
                add += tHair * wHair * rad * smoothstep(-0.15, 0.35, nl0) * (pow(s1, 70.0) * 0.50 + pow(s2, 20.0) * 0.20 * mix(vec3(1.0), tint, 0.7));
            }
            // Sky light: cloth sheen and skin scattering from the ambient.
            add += tSky * wCloth * albedoLin * ambientIrr * pow(1.0 - ndv, 3.0) * 0.45;
            add += tSky * wSkin * albedoLin * ambientIrr * vec3(1.0, 0.45, 0.30) * (1.0 - ndv) * 0.22;
            vec3 budget = albedoLin * 0.9 + vec3(0.012);
            add = budget * add / max(budget + abs(add), vec3(0.000001));
            return add * uCharMat.y * uCharLight.w * fade;
        }

        vec3 newDawnCharacter(vec3 albedoLin, vec3 albedoGamma, vec3 shadeNormal, vec3 irradiance) {
            vec3 n = shadeNormal / max(length(shadeNormal), 0.0001);
            mat3 cof = mat3(cross(uViewMatrix[1], uViewMatrix[2]),
                            cross(uViewMatrix[2], uViewMatrix[0]),
                            cross(uViewMatrix[0], uViewMatrix[1]));
            float det = dot(uViewMatrix[0], cof[0]);
            vec3 wn = cof * n * (det < 0.0 ? -1.0 : 1.0);
            wn = length(wn) > 1e-6 ? normalize(wn) : vec3(0.0, 1.0, 0.0);
            vec3 v = uCharLight.xyz - vViewPos;
            float vl = length(v);
            if (vl < 1.0) return vec3(0.0);
            v /= vl;

            // Material class from the albedo alone (render state has no material id): bright neutral texels read as
            // metal, skin hues stay matte, saturated cloth gets only a faint broad sheen.
            float mx = max(albedoGamma.r, max(albedoGamma.g, albedoGamma.b));
            float mn = min(albedoGamma.r, min(albedoGamma.g, albedoGamma.b));
            float sat = (mx - mn) / max(mx, 0.001);
            float luma = dot(albedoGamma, vec3(0.2126, 0.7152, 0.0722));
            float metal = (1.0 - smoothstep(0.20, 0.46, sat)) * smoothstep(0.22, 0.50, luma) *
                          (1.0 - smoothstep(0.70, 0.86, luma)); // near-white (cloth, eyes, socks) is not metal
            float skin = step(albedoGamma.g, albedoGamma.r) * step(albedoGamma.b, albedoGamma.g) *
                         smoothstep(0.10, 0.20, sat) * (1.0 - smoothstep(0.55, 0.75, sat)) *
                         smoothstep(0.22, 0.38, luma);
            // The texture's material class replaces the albedo guess where it knows better.
            int matCls = int(uCharMat.x + 0.5);
            if (uCharMat.y > 0.0 && matCls > 0) {
                if (matCls == 5) { metal = 1.0; skin = 0.0; }
                else if (matCls == 1) { metal = 0.0; skin = 1.0; }
                else if (matCls == 3 || matCls == 6 || matCls == 7) { metal = 0.0; skin = 0.0; }
            }
            float specWeight = mix(0.07 * (1.0 - skin), 1.0, metal);

            // The broad terms read from a normal pulled toward the view: facet borders of the low-poly meshes merge
            // into a smooth gradient. Metal keeps the real normal so its highlight stays crisp.
            vec3 wnSoft = normalize(wn + 0.5 * v);
            vec3 ns = normalize(mix(wnSoft, wn, metal));
            float nv = clamp(dot(ns, v), 0.0, 1.0);
            // 0 facing the camera, 1 at the silhouette (0.447 is the soft normal's N.V there): a wide, low-contrast rim.
            float fres = pow(clamp((1.0 - dot(wnSoft, v)) / 0.553, 0.0, 1.0), 1.5);

            float rough = mix(0.72, 0.24, metal);
            float a2 = pow(rough, 4.0);
            // Metal reflects with its own colour: F0 is the albedo's hue pushed to full saturation, not a grey.
            vec3 f0 = mix(vec3(0.04), 0.92 * pow(albedoGamma / max(mx, 0.001), vec3(3.0)), metal);

            vec3 ambient = (uLightAmbient / 255.0) * uHdrAmbient;
            vec3 rimBase = ambient;
#ifndef GL_ES
            if (uSkyAmbientEnabled) rimBase = skyAmbientColor(ambient, reflect(-v, wnSoft));
#endif
            vec3 rim = rimBase * 3.4 * fres;
            vec3 spec = vec3(0.0);
            for (int i = 0; i < uLightCount; ++i) {
                vec3 l;
                vec3 radiance = uLightColor[i] / 255.0;
                if (uLightPositional[i] != 0) {
                    vec3 delta = uLightPos[i] - vViewPos;
                    float dist = length(delta);
                    l = delta / max(dist, 0.0001);
                    float d2 = max(dot(delta.xy, delta.xy) + 2.0 * delta.z * delta.z, 0.000001);
                    float distf = mix(floor(sqrt(d2)), sqrt(d2), uPointLightQuality);
                    radiance /= 1.0 + (distf * uLightAtten[i].x * 2.0 + distf * distf * uLightAtten[i].y / 8.0) / 65535.0;
                    radiance *= mix(0.5, 1.0, metal); // point-light glints on faceted cloth and skin read as noise
                } else {
                    l = normalize(uViewMatrix * uLightDir[i]);
                    radiance *= uHdrSun;
                }
                float nl = dot(ns, l);
                // Light from behind the subject, seen from the camera, wraps around the silhouette.
                float back = smoothstep(-0.1, 0.7, -dot(v, l));
                rim += radiance * fres * back * clamp((nl + 0.6) / 1.6, 0.0, 1.0) * 2.2;
                if (nl <= 0.0 || nv <= 0.0) continue;
                vec3 h = normalize(v + l);
                float nh = max(dot(ns, h), 0.0);
                float vh = clamp(dot(v, h), 0.0, 1.0);
                float denom = nh * nh * (a2 - 1.0) + 1.0;
                float d = a2 / max(denom * denom, 0.00001);
                // Metal: a crisp core plus a broad lobe, so a curved shield shows a highlight gradient, not just a pinpoint.
                float denomB = nh * nh * (0.0625 - 1.0) + 1.0;
                d = mix(d, 0.6 * d + 0.4 * 0.0625 / max(denomB * denomB, 0.00001), metal);
                float gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
                float gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
                float vis = 0.5 / max(gv + gl, 0.00001);
                vec3 fr = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
                spec += radiance * fr * (d * vis * nl);
            }
#ifndef GL_ES
            // Metal also mirrors the sky (the same prefiltered cube the material maps use), tinted by F0.
            if (uSkyReflectionsEnabled && metal > 0.0) {
                float nvm = clamp(dot(wn, v), 0.0, 1.0);
                vec3 env = textureLod(uSkySpecular, reflect(-v, wn), rough * 4.0).rgb;
                spec += env * (f0 + (max(vec3(1.0 - rough), f0) - f0) * pow(1.0 - nvm, 5.0)) * metal * 10.0; // the sky cube holds radiance well below sunlit surfaces; lift it so metal reads as reflective
            }
#endif
            // The specular is bounded per channel by the lit albedo, so a black or dark texel cannot be lifted.
            vec3 budget = albedoLin * (0.03 + 4.5 * metal) + vec3(0.02 * metal);
            spec *= specWeight;
            spec = budget * spec / max(budget + spec, vec3(0.000001));
            // Characters are read at close range; far-away large props drawn as actors fade out (Link is ~60 units).
            // A metal has no diffuse of its own to speak of: dim it so the highlight and sky reflection carry the look.
            vec3 diffuseLoss = albedoLin * irradiance * (0.5 * metal);
            return (albedoLin * rim + spec - diffuseLoss) * uCharLight.w * (1.0 - smoothstep(400.0, 800.0, vl));
        }
    @end

    // New Dawn remastered textures: isotropic materials (grass, dirt, rock, plaster) carry TEX_FLAG_NEWDAWN_STOCH and are
    // sampled stochastically so large surfaces do not visibly repeat: three taps on a skewed triangle grid (Mikkelsen's
    // hex tiling), each shifted by a hash of its grid vertex, blended by sharpened barycentric weights. Albedo and
    // material maps share this, so a pixel reads the same patch in all of them.
    uniform int uNdStoch[2];
    vec2 ndHash2(vec2 p) {
        p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
        return fract(sin(p) * 43758.5453);
    }
    vec4 ndStochTex(sampler2D tex, vec2 uv, vec2 dx, vec2 dy) {
        vec2 p = uv * 2.1;
        vec2 sk = vec2(p.x, -0.57735027 * p.x + 1.15470054 * p.y);
        vec2 base = floor(sk);
        vec2 f = fract(sk);
        float z = 1.0 - f.x - f.y;
        float s = step(0.0, -z);
        float s2 = 2.0 * s - 1.0;
        vec3 w = vec3(-z * s2, s - f.y * s2, s - f.x * s2);
        w = w * w * w;
        w /= w.x + w.y + w.z;
        vec4 c = textureGrad(tex, uv + ndHash2(base + vec2(s, s)), dx, dy) * w.x;
        c += textureGrad(tex, uv + ndHash2(base + vec2(s, 1.0 - s)), dx, dy) * w.y;
        c += textureGrad(tex, uv + ndHash2(base + vec2(1.0 - s, s)), dx, dy) * w.z;
        return c;
    }

    // New Dawn unlit materials (interiors, dungeons): vertex-coloured room geometry carries its light in the vertex colour and has no
    // normals. The maps still add relief: the face normal comes from the world position's screen derivatives, the tangent frame from
    // the UV derivatives, and the texture is shaded by how much the mapped normal differs from the face normal against a fixed world
    // light from above (MM is Y up) plus the height cavities. The baked colour then multiplies it, so the room keeps its mood.
    @if(o_unlit_material)
        @{attr} vec3 vViewPos;
        @{attr} vec3 vNormal; // smooth normal of room geometry (the interpreter welds neighbouring triangles); zero: derive it here
        uniform sampler2D uMaterialNormal;
        uniform sampler2D uMaterialRoughness;
        uniform float uMaterialStrength;
        uniform vec2 uMaterialMirror;
        uniform vec4 uMaterialEye; // xyz: camera position in vViewPos's (world) space, w: 1 when valid
        uniform vec3 uNdUnlit;     // x: normal strength, y: relief shading, z: cavity darkening
        uniform int uMaterialDebug; // 11: relief shade, 12: face normal, 13: world position
        vec4 ndUnlitTex(sampler2D tex, vec2 uv, vec2 dx, vec2 dy) {
            if (uNdStoch[@{material_tile}] != 0) return ndStochTex(tex, uv, dx, dy);
            return textureGrad(tex, uv, dx, dy);
        }
        float ndUnlitShade(vec2 uv, vec2 dx, vec2 dy) {
            vec3 px = dFdx(vViewPos), py = dFdy(vViewPos);
            vec3 n = vNormal;
            float nl = length(n);
            if (nl > 0.1) {
                n /= nl;
            } else {
                n = cross(px, py);
                nl = length(n);
                if (nl < 1e-9) return 1.0;
                n /= nl;
            }
            if (dot(n, (uMaterialEye.w > 0.0 ? uMaterialEye.xyz : vec3(0.0)) - vViewPos) < 0.0) n = -n;
            float det = dx.x * dy.y - dx.y * dy.x;
            float handedness = det < 0.0 ? -1.0 : 1.0;
            vec3 t = px * dy.y - py * dx.y;
            vec3 b = py * dx.x - px * dy.x;
            t = (t - n * dot(n, t)) * handedness;
            b = (b - n * dot(n, b)) * handedness;
            float t2 = dot(t, t), b2 = dot(b, b);
            if (abs(det) < 1e-12 || min(t2, b2) < 1e-16) return 1.0;
            vec2 xy = ndUnlitTex(uMaterialNormal, uv, dx, dy).rg * 2.0 - 1.0;
            xy = (xy - vec2(1.0 / 255.0)) * uMaterialStrength * uNdUnlit.x;
            xy *= mix(vec2(1.0), 1.0 - 2.0 * mod(floor(uv), 2.0), uMaterialMirror);
            vec3 mapped = normalize(vec3(xy, sqrt(max(1.0 - dot(xy, xy), 0.001))));
            vec3 nm = normalize(t * inversesqrt(t2) * mapped.x - b * inversesqrt(b2) * mapped.y + n * mapped.z);
            const vec3 L = vec3(0.3162, 0.7589, 0.5692); // normalize(0.35, 0.84, 0.63)
            float relief = ndUnlitTex(uMaterialRoughness, uv, dx, dy).g - 0.5;
            float shade = 1.0 + (dot(nm, L) - dot(n, L)) * uNdUnlit.y * 1.4 + relief * uNdUnlit.z;
            return clamp(shade, 0.45, 1.6);
        }
    @end

    @if(o_material_maps)
        uniform sampler2D uMaterialNormal;
        uniform sampler2D uMaterialRoughness;
        uniform float uMaterialStrength;
        uniform int uMaterialDebug; // 1: specular only, 2: roughness, 3: N.V sign, 4: diffuse only
        uniform vec2 uMaterialMirror;
        vec4 ndMatTex(sampler2D tex, vec2 uv, vec2 dx, vec2 dy) {
            if (uNdStoch[@{material_tile}] != 0) return ndStochTex(tex, uv, dx, dy);
            return textureGrad(tex, uv, dx, dy);
        }
        // Cotangents use unclamped UV derivatives. The -T direction matches the generator's
        // +Y-up normals (texture rows increase down). Keeping the signed determinant handles
        // mirrored islands; repeat/mirror sampling and tile clamp match the albedo sampler.
        // Orthonormal-ish surface frame from screen derivatives: t/b follow +U/+V (raw*det, then
        // handedness-corrected); det is the signed UV Jacobian determinant.
        void materialFrame(vec2 dx, vec2 dy, out vec3 n, out vec3 t, out vec3 b, out float det, out float handedness) {
            // Cofactor inverse-transpose also works on this backend's GLSL 1.30 path.
            mat3 cofactor = mat3(cross(uViewMatrix[1], uViewMatrix[2]),
                                 cross(uViewMatrix[2], uViewMatrix[0]),
                                 cross(uViewMatrix[0], uViewMatrix[1]));
            float determinant = dot(uViewMatrix[0], cofactor[0]);
            vec3 transformed = cofactor * vNormal * (determinant < 0.0 ? -1.0 : 1.0);
            vec3 px = dFdx(vViewPos), py = dFdy(vViewPos);
            n = transformed / max(length(transformed), 0.0001);
            det = dx.x * dy.y - dx.y * dy.x;
            t = px * dy.y - py * dx.y;
            b = py * dx.x - px * dy.x;
            handedness = det < 0.0 ? -1.0 : 1.0;
        }

        vec3 materialNormal(vec2 uv, vec2 dx, vec2 dy, float weight, out float variance) {
            vec3 n, t, b;
            float det, handedness;
            materialFrame(dx, dy, n, t, b, det, handedness);
            t = (t - n * dot(n, t)) * handedness;
            b = (b - n * dot(n, b)) * handedness;
            float t2 = dot(t, t), b2 = dot(b, b);
            vec2 xy = ndMatTex(uMaterialNormal, uv, dx, dy).rg * 2.0 - 1.0;
            // Remove the half-quantum offset of the RG8 flat normal (128,128).
            xy = (xy - vec2(1.0 / 255.0)) * uMaterialStrength * weight;
            xy *= mix(vec2(1.0), 1.0 - 2.0 * mod(floor(uv), 2.0), uMaterialMirror);
            vec3 mapped = normalize(vec3(xy, sqrt(max(1.0 - dot(xy, xy), 0.001))));
            // Derivatives are evaluated before any branch or alpha test. Broaden the GGX lobe
            // as subpixel normal variation grows; mipmapped RG8 vectors are renormalized above.
            variance = max(dot(dFdx(mapped), dFdx(mapped)), dot(dFdy(mapped), dFdy(mapped)));
            if (abs(det) < 1e-12 || min(t2, b2) < 1e-16) return n;
            return normalize(t * inversesqrt(t2) * mapped.x - b * inversesqrt(b2) * mapped.y + n * mapped.z);
        }


    @if(o_material_maps2)
        uniform sampler2D uMaterialNormal2;
        uniform sampler2D uMaterialRoughness2;
        uniform vec2 uMaterialMirror2;
        vec4 ndMatTex2(sampler2D tex, vec2 uv, vec2 dx, vec2 dy) {
            if (uNdStoch[1 - @{material_tile}] != 0) return ndStochTex(tex, uv, dx, dy);
            return textureGrad(tex, uv, dx, dy);
        }
        // The second layer's normal in the same world frame as the first; faceN is the unperturbed surface normal.
        vec3 materialNormal2(vec2 uv, vec2 dx, vec2 dy, float weight, out float variance, out vec3 faceN) {
            vec3 n, t, b;
            float det, handedness;
            materialFrame(dx, dy, n, t, b, det, handedness);
            faceN = n;
            t = (t - n * dot(n, t)) * handedness;
            b = (b - n * dot(n, b)) * handedness;
            float t2 = dot(t, t), b2 = dot(b, b);
            vec2 xy = ndMatTex2(uMaterialNormal2, uv, dx, dy).rg * 2.0 - 1.0;
            xy = (xy - vec2(1.0 / 255.0)) * uMaterialStrength * weight;
            xy *= mix(vec2(1.0), 1.0 - 2.0 * mod(floor(uv), 2.0), uMaterialMirror2);
            vec3 mapped = normalize(vec3(xy, sqrt(max(1.0 - dot(xy, xy), 0.001))));
            variance = max(dot(dFdx(mapped), dFdx(mapped)), dot(dFdy(mapped), dFdy(mapped)));
            if (abs(det) < 1e-12 || min(t2, b2) < 1e-16) return n;
            return normalize(t * inversesqrt(t2) * mapped.x - b * inversesqrt(b2) * mapped.y + n * mapped.z);
        }
    @end

        // Parallax occlusion mapping: steep ray march through the companion height (roughness map G,
        // 0.5 = the real surface, class gain baked by the generator) with a linear relief refinement.
        // The ray is unfolded in UV space, so wrap, mirror and clamp samplers behave like the albedo.
        uniform vec4 uMaterialEye; // xyz: camera position in vViewPos's (world) space, w: 1 when valid
        uniform vec2 uMaterialParallax; // x: depth as a fraction of the texture's world size (0 = off), y: fade-out distance
        vec2 materialParallax(vec2 uv, vec2 dx, vec2 dy, float weight) {
            vec3 n, t, b;
            float det, handedness;
            materialFrame(dx, dy, n, t, b, det, handedness);
            vec3 toEye = uMaterialEye.xyz - vViewPos;
            float dist = length(toEye);
            float fade = uMaterialEye.w * weight * (1.0 - smoothstep(uMaterialParallax.y * 0.55, uMaterialParallax.y, dist));
            vec3 v = toEye / max(dist, 0.0001);
            float vn = dot(v, n);
            if (fade <= 0.001 || abs(det) < 1e-12 || vn <= 0.02) return uv;
            // World vectors per UV unit, projected into the surface plane.
            vec3 pu = t / det, pv = b / det;
            pu -= n * dot(n, pu);
            pv -= n * dot(n, pv);
            float g11 = dot(pu, pu), g12 = dot(pu, pv), g22 = dot(pv, pv);
            float gd = g11 * g22 - g12 * g12;
            if (gd < 1e-12) return uv;
            vec3 lateral = v - n * vn;
            float a = dot(lateral, pu), c = dot(lateral, pv);
            vec2 duv = vec2(g22 * a - g12 * c, g11 * c - g12 * a) / gd;
            // Depth reference: the texture's world size, capped so large sparse tiles do not get metre-deep relief.
            float worldSize = min(sqrt(0.5 * (g11 + g22)), 28.0);
            // Ray shift over the full depth range, bounded so grazing views cannot sweep across the tile.
            // Skip texels the sampler is already minifying: relief below a few texels per pixel only shimmers.
            vec2 heightSize = vec2(textureSize(uMaterialRoughness, 0));
            vec2 footX = dx * heightSize, footY = dy * heightSize;
            float lod = 0.5 * log2(max(max(dot(footX, footX), dot(footY, footY)), 1.0));
            fade *= 1.0 - smoothstep(1.5, 3.5, lod);
            // Ease the depth out only at very grazing angles, where the shift would shear silhouettes.
            fade *= smoothstep(0.08, 0.36, vn);
            if (fade <= 0.001) return uv;
            vec2 q = duv * (uMaterialParallax.x * fade * worldSize / max(vn, 0.25));
            float reach = max(abs(q.x), abs(q.y));
            q *= min(1.0, 0.09 / max(reach, 0.0001));
            reach = min(reach, 0.09);
            // Layers per ray: enough that one layer shifts less than a joint/crack width (~1/128 UV), and more
            // at grazing angles (N.V) where the ray crosses the height field at a shallow slope.
            float steps = clamp(max(reach * 640.0, mix(56.0, 14.0, vn)), 14.0, 64.0);
            // Spend layers where they are seen: fewer with distance.
            steps = max(mix(steps, 12.0, smoothstep(180.0, 520.0, dist)), 12.0);
            float layer = 1.0 / steps;
            float depth = 0.0, prevDepth = 0.0;
            float surface = 1.0 - ndMatTex(uMaterialRoughness, uv + q * 0.5, dx, dy).g;
            for (int i = 0; i < 64; ++i) {
                if (float(i) >= steps || depth >= surface) break;
                prevDepth = depth;
                depth += layer;
                surface = 1.0 - ndMatTex(uMaterialRoughness, uv + q * (0.5 - depth), dx, dy).g;
            }
            // Relief refinement: a linear (classic POM) estimate inside the last layer, then bisection around it,
            // so the hit is not quantised to the layer grid.
            float before = (1.0 - ndMatTex(uMaterialRoughness, uv + q * (0.5 - prevDepth), dx, dy).g) - prevDepth;
            float after = surface - depth;
            float lerpDepth = mix(depth, prevDepth, clamp(after / min(after - before, -0.000001), 0.0, 1.0));
            float lo = max(prevDepth, lerpDepth - 0.25 * layer), hi = min(depth, lerpDepth + 0.25 * layer);
            if (hi < lo) { lo = prevDepth; hi = depth; }
            for (int i = 0; i < 5; ++i) {
                float mid = 0.5 * (lo + hi);
                float midSurface = 1.0 - ndMatTex(uMaterialRoughness, uv + q * (0.5 - mid), dx, dy).g;
                if (mid >= midSurface) hi = mid; else lo = mid;
            }
            return uv + q * (0.5 - 0.5 * (lo + hi));
        }

#ifndef GL_ES
        vec3 materialSkySpecular(vec3 n, float roughness, float weight) {
            if (!uSkyReflectionsEnabled) return vec3(0.0);
            // MM's modelview attributes are world-space: its camera view is in P.
            vec3 skyView = normalize(uSkyEye - vViewPos);
            float skyNv = max(dot(n, skyView), 0.0);
            if (skyNv > 0.0) {
                // Analytic split-sum BRDF fit, with the prefiltered GGX environment in four roughness steps.
                vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
                vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
                vec4 r = roughness * c0 + c1;
                float a004 = min(r.x * r.x, exp2(-9.28 * skyNv)) * r.x + r.y;
                vec2 ab = vec2(-1.04, 1.04) * a004 + r.zw;
                vec3 environment = textureLod(uSkySpecular, reflect(-skyView, n), roughness * 4.0).rgb;
                return environment * max(0.04 * ab.x + ab.y, 0.0) *
                          min(uMaterialStrength, 1.0) * weight * (1.0-roughness) * (1.0-roughness);
            }
            return vec3(0.0);
        }
#endif

        // Specular only: the diffuse term reuses the game's own per-pixel light model (see shadeNormal),
        // so material surfaces keep the scene's tone and only gain relief and highlights.
        vec3 materialSpecular(vec3 n, float roughness, float weight, bool hdrLit) {
            // The eye in vViewPos's (world) space; -vViewPos only points at the world origin.
            vec3 viewVec = uMaterialEye.w > 0.0 ? uMaterialEye.xyz - vViewPos : -vViewPos;
            vec3 v = viewVec / max(length(viewVec), 0.0001);
            float nv = clamp(dot(n, v), 0.0, 1.0);
            // A BRDF has no response below either hemisphere. Clamping a negative N.V
            // positive instead produced bright grazing lobes on the game's authored normals.
            if (nv <= 0.0) return vec3(0.0);
            float a2 = pow(roughness, 4.0);
            float sun = hdrLit ? uHdrSun : 1.0;
            vec3 result = vec3(0.0);
            for (int i = 0; i < uLightCount; ++i) {
                vec3 l;
                float attenuation = sun;
                if (uLightPositional[i] != 0) {
                    vec3 delta = uLightPos[i] - vViewPos;
                    float dist = length(delta);
                    l = delta / max(dist, 0.0001);
                    // Retain the game's animated light set and attenuation coefficients.
                    float d2 = max(dot(delta.xy, delta.xy) + 2.0 * delta.z * delta.z, 0.000001);
                    float distf = mix(floor(sqrt(d2)), sqrt(d2), uPointLightQuality);
                    attenuation = 1.0 / (1.0 + (distf * uLightAtten[i].x * 2.0 +
                        distf * distf * uLightAtten[i].y / 8.0) / 65535.0);
                } else {
                    l = normalize(uViewMatrix * uLightDir[i]);
                }
                float nl = clamp(dot(n, l), 0.0, 1.0);
                if (nl <= 0.0) continue;
                vec3 h = (v + l) / max(length(v + l), 0.0001);
                float nh = max(dot(n, h), 0.0);
                float vh = clamp(dot(v, h), 0.0, 1.0);
                float denom = nh * nh * (a2 - 1.0) + 1.0;
                // GGX/Smith/Schlick dielectric. The game's irradiance is Lambert without 1/pi, so the
                // specular is pi-scaled to stay in the same radiometric convention.
                float d = a2 / max(denom * denom, 0.00001);
                float gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
                float gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
                float visibility = 0.5 / max(gv + gl, 0.00001);
                float fresnel = 0.04 + 0.96 * pow(1.0 - vh, 5.0);
                float f = fresnel * min(uMaterialStrength, 1.0) * weight;
                result += vec3(f * d * visibility) * (uLightColor[i] / 255.0) * nl * attenuation;
            }
            // These replacements contain painted lighting, not calibrated PBR albedo.
            // A fully rough companion must not add an untextured white lobe over it.
            float smoothness = 1.0 - roughness;
            return max(result, vec3(0.0)) * smoothness * smoothness;
        }
    @end

    @if(o_textures[0]) uniform sampler2D uTex0;
    @if(o_textures[1]) uniform sampler2D uTex1;

    @if(o_masks[0]) uniform sampler2D uTexMask0;
    @if(o_masks[1]) uniform sampler2D uTexMask1;

    @if(o_blend[0]) uniform sampler2D uTexBlend0;
    @if(o_blend[1]) uniform sampler2D uTexBlend1;

    uniform int frame_count;
    uniform float noise_scale;

    @if(o_prim_depth)
    uniform float prim_depth;
    @end

    uniform int texture_width[2];
    uniform int texture_height[2];
    uniform int texture_filtering[2];

    // Explicit gradients keep mip selection correct: TEX_OFFSET's fract() makes automatic
    // derivatives useless, and clamped UVs lose their gradient at the texture edge.
    #define TEX_OFFSET(off) textureGrad(tex, texCoord - off / texSize, ddx, ddy)
    #define WRAP(x, low, high) mod((x)-(low), (high)-(low)) + (low)

    float random(in vec3 value) {
        float random = dot(sin(value), vec3(12.9898, 78.233, 37.719));
        return fract(sin(random) * 143758.5453);
    }

    vec4 fromLinear(vec4 linearRGB){
        bvec3 cutoff = lessThan(linearRGB.rgb, vec3(0.0031308));
        vec3 higher = vec3(1.055)*pow(linearRGB.rgb, vec3(1.0/2.4)) - vec3(0.055);
        vec3 lower = linearRGB.rgb * vec3(12.92);
        return vec4(mix(higher, lower, cutoff), linearRGB.a);
    }

    vec4 filter3point(in sampler2D tex, in vec2 texCoord, in vec2 texSize, in vec2 ddx, in vec2 ddy) {
        vec2 offset = fract(texCoord*texSize - vec2(0.5));
        offset -= step(1.0, offset.x + offset.y);
        vec4 c0 = TEX_OFFSET(offset);
        vec4 c1 = TEX_OFFSET(vec2(offset.x - sign(offset.x), offset.y));
        vec4 c2 = TEX_OFFSET(vec2(offset.x, offset.y - sign(offset.y)));
        return c0 + abs(offset.x)*(c1-c0) + abs(offset.y)*(c2-c0);
    }

    vec4 hookTexture2D(in int id, sampler2D tex, in vec2 uv, in vec2 texSize, in vec2 ddx, in vec2 ddy) {
        if (uNdStoch[id] != 0) return ndStochTex(tex, uv, ddx, ddy);
    @if(o_three_point_filtering)
        if(texture_filtering[id] == @{FILTER_THREE_POINT}) {
            return filter3point(tex, uv, texSize, ddx, ddy);
        }
    @end
        return textureGrad(tex, uv, ddx, ddy);
    }

    #define TEX_SIZE(tex) vec2(texture_width[tex], texture_height[tex])

    void main() {
        @if(o_material_maps)
            float materialWeight = 1.0;
            @if(material_blend)
                materialWeight = clamp(vInput@{material_weight_index}.r, 0.0, 1.0);
                @if(material_weight_invert)
                    materialWeight = 1.0 - materialWeight;
                @end
            @end
        @end
        @for(i in 0..2)
            @if(o_textures[i])
                @{s = o_clamp[i][0]}
                @{t = o_clamp[i][1]}

                vec2 texSize@{i} = TEX_SIZE(@{i});
                vec2 ddx@{i} = dFdx(vTexCoord@{i});
                vec2 ddy@{i} = dFdy(vTexCoord@{i});

                @if(!s && !t)
                    vec2 vTexCoordAdj@{i} = vTexCoord@{i};
                @else
                    @if(s && t)
                        vec2 vTexCoordAdj@{i} = clamp(vTexCoord@{i}, 0.5 / texSize@{i}, vec2(vTexClampS@{i}, vTexClampT@{i}));
                    @elseif(s)
                        vec2 vTexCoordAdj@{i} = vec2(clamp(vTexCoord@{i}.s, 0.5 / texSize@{i}.s, vTexClampS@{i}), vTexCoord@{i}.t);
                    @else
                        vec2 vTexCoordAdj@{i} = vec2(vTexCoord@{i}.s, clamp(vTexCoord@{i}.t, 0.5 / texSize@{i}.t, vTexClampT@{i}));
                    @end
                @end

                @if(o_material_maps)
                    // Parallax shifts the material tile's lookups; gradients stay those of the undisplaced UV.
                    if (@{i} == @{material_tile} && uMaterialParallax.x > 0.0) {
                        vTexCoordAdj@{i} = materialParallax(vTexCoordAdj@{i}, ddx@{i}, ddy@{i}, materialWeight);
                        @if(s)
                            vTexCoordAdj@{i}.s = clamp(vTexCoordAdj@{i}.s, 0.5 / texSize@{i}.s, vTexClampS@{i});
                        @end
                        @if(t)
                            vTexCoordAdj@{i}.t = clamp(vTexCoordAdj@{i}.t, 0.5 / texSize@{i}.t, vTexClampT@{i});
                        @end
                    }
                @end

                vec4 texVal@{i} = hookTexture2D(@{i}, uTex@{i}, vTexCoordAdj@{i}, texSize@{i}, ddx@{i}, ddy@{i});

                @if(o_masks[i])
                    @if(opengles) 
                        vec2 maskSize@{i} = vec2(textureSize(uTexMask@{i}, 0));
                    @else 
                        vec2 maskSize@{i} = textureSize(uTexMask@{i}, 0);
                    @end

                    vec4 maskVal@{i} =
                        hookTexture2D(@{i}, uTexMask@{i}, vTexCoordAdj@{i}, maskSize@{i}, ddx@{i}, ddy@{i});

                    @if(o_blend[i])
                        vec4 blendVal@{i} =
                            hookTexture2D(@{i}, uTexBlend@{i}, vTexCoordAdj@{i}, texSize@{i}, ddx@{i}, ddy@{i});
                    @else
                        vec4 blendVal@{i} = vec4(0, 0, 0, 0);
                    @end

                    texVal@{i} = mix(texVal@{i}, blendVal@{i}, maskVal@{i}.a);
                @end
            @end
        @end

        @if(o_unlit_material)
            float ndShade = ndUnlitShade(vTexCoordAdj@{material_tile}, ddx@{material_tile}, ddy@{material_tile});
            texVal@{material_tile}.rgb *= ndShade;
            if (uMaterialDebug == 11) texVal@{material_tile}.rgb = vec3(ndShade * 0.5);
            if (uMaterialDebug == 12) texVal@{material_tile}.rgb = normalize(cross(dFdx(vViewPos), dFdy(vViewPos))) * 0.5 + 0.5;
            if (uMaterialDebug == 13) texVal@{material_tile}.rgb = fract(vViewPos / 100.0);
        @end

        @if(o_material_maps)
            float materialVariance;
            vec3 materialN = materialNormal(vTexCoordAdj@{material_tile}, ddx@{material_tile}, ddy@{material_tile}, materialWeight, materialVariance);
            @if(o_material_maps2)
                float materialVariance2;
                vec3 materialFaceN;
                vec3 materialN2 = materialNormal2(vTexCoordAdj@{1 - material_tile}, ddx@{1 - material_tile}, ddy@{1 - material_tile}, materialWeight, materialVariance2, materialFaceN);
                materialN = normalize(materialN + materialN2 - materialFaceN);
                materialVariance = max(materialVariance, materialVariance2);
            @end
            float materialRoughness = ndMatTex(uMaterialRoughness, vTexCoordAdj@{material_tile}, ddx@{material_tile}, ddy@{material_tile}).r;
            @if(o_material_maps2)
                materialRoughness = 0.5 * (materialRoughness + ndMatTex2(uMaterialRoughness2, vTexCoordAdj@{1 - material_tile}, ddx@{1 - material_tile}, ddy@{1 - material_tile}).r);
            @end
            materialRoughness = clamp(sqrt(materialRoughness * materialRoughness + min(materialVariance, 0.5)), 0.3, 1.0);
            bool materialDiffuse = ((uShadeMask & @{material_diffuse_mask}) != 0) && materialWeight > 0.0;
        @end

        @if(o_per_pixel_lighting)
            // Material surfaces light the perturbed normal (back in model space) with the game's own
            // per-pixel model; everything else keeps the interpolated vertex normal.
            vec3 shadeNormal = vNormal;
            @if(o_material_maps)
                if (materialDiffuse) shadeNormal = normalize(transpose(uViewMatrix) * materialN) * length(vNormal);
            @end
            vec3 newDawnLit = newDawnShade(shadeNormal);
            bool hdrDiffuse = uHdrWorld && ((uShadeMask & @{hdr_diffuse_mask}) != 0);
            if (hdrDiffuse) newDawnLit = vec3(1.0);
            @for(i in 0..o_inputs)
                @if(o_alpha)
                    vec4 vInputShaded@{i + 1} = vInput@{i + 1};
                    if ((uShadeMask & (1 << @{i})) != 0) {
                        vInputShaded@{i + 1}.rgb = newDawnLit;
                    }
                @else
                    vec3 vInputShaded@{i + 1} = vInput@{i + 1};
                    if ((uShadeMask & (1 << @{i})) != 0) {
                        vInputShaded@{i + 1} = newDawnLit;
                    }
                @end
            @end
        @end

        @if(o_alpha)
            vec4 texel;
        @else 
            vec3 texel;
        @end

        @if(o_2cyc)
            @{f_range = 2}
        @else
            @{f_range = 1}
        @end

        @for(c in 0..f_range)
            @if(c == 1)
                @if(o_alpha)
                    @if(o_c[c][1][2] == SHADER_COMBINED)
                        texel.a = WRAP(texel.a, -1.01, 1.01);
                    @else
                        texel.a = WRAP(texel.a, -0.51, 1.51);
                    @end
                @end

                @if(o_c[c][0][2] == SHADER_COMBINED)
                    texel.rgb = WRAP(texel.rgb, -1.01, 1.01);
                @else
                    texel.rgb = WRAP(texel.rgb, -0.51, 1.51);
                @end
            @end

            @if(!o_color_alpha_same[c] && o_alpha)
                texel = vec4(@{
                append_formula(o_c[c], o_do_single[c][0],
                            o_do_multiply[c][0], o_do_mix[c][0], false, false, true, c == 0,
                            o_per_pixel_lighting)
                }, @{append_formula(o_c[c], o_do_single[c][1],
                            o_do_multiply[c][1], o_do_mix[c][1], true, true, true, c == 0,
                            o_per_pixel_lighting)
                });
            @else
                texel = @{append_formula(o_c[c], o_do_single[c][0],
                            o_do_multiply[c][0], o_do_mix[c][0], o_alpha, false,
                            o_alpha, c == 0, o_per_pixel_lighting)};
            @end
        @end

        texel = WRAP(texel, -0.51, 1.51);
        texel = clamp(texel, 0.0, 1.0);
        if (uHdrWorld) {
            vec3 albedoGamma = texel.rgb;
            texel.rgb = hdrDecode(texel.rgb) * uHdrSky;
            @if(o_per_pixel_lighting)
                if (hdrDiffuse) {
                    vec3 albedoLin = texel.rgb;
                    bool characterLit = uCharLight.w > 0.0;
                    @if(o_material_maps)
                        characterLit = characterLit && !materialDiffuse;
                    @end
                    vec3 charAmbient, charWrap, charPoint;
                    vec3 charIrradiance = newDawnIrradianceEx(shadeNormal, charAmbient, characterLit ? uCharGrade.w : 0.0, charWrap, charPoint);
                    vec3 appliedIrradiance = charIrradiance;
                    // With the white balance and the wrap both at 0 the light is left exactly as the scene computed it.
                    if (characterLit && (uCharGrade.x > 0.0 || uCharGrade.w > 0.0)) {
                        vec3 keyPart = charIrradiance - charAmbient - charPoint;
                        // Subsurface: light wrapping past the terminator takes on the surface's own colour (skin glows
                        // red-orange at the edge of the light, cloth deepens), not the light's.
                        vec3 sss = albedoLin / max(dot(albedoLin, vec3(0.2126, 0.7152, 0.0722)), 1e-4);
                        sss = min(pow(max(sss, vec3(0.0)), vec3(0.6)), vec3(2.5));
                        vec3 warm = vec3(1.04, 1.0, 0.94), cool = vec3(0.94, 0.99, 1.07);
                        appliedIrradiance = charWhiteBalance(charAmbient, cool, uCharGrade.x) + charWhiteBalance(keyPart, warm, uCharGrade.x) +
                                            charWhiteBalance(charWrap, warm, uCharGrade.x) * sss + charPoint;
                    }
                    texel.rgb *= appliedIrradiance;
                    if (characterLit) {
                        texel.rgb += newDawnCharacter(albedoLin, albedoGamma, shadeNormal, appliedIrradiance);
                        texel.rgb += newDawnCharMaterial(albedoLin, albedoGamma, shadeNormal, charAmbient);
                        texel.rgb = charShoulder(texel.rgb, uCharGrade.y);
                        int charDebug = int(uCharGrade.z + 0.5);
                        if (charDebug == 1) texel.rgb = charIrradiance;
                        else if (charDebug == 2) texel.rgb = charAmbient;
                        else if (charDebug == 3) texel.rgb = charIrradiance - charAmbient;
                        else if (charDebug == 4) texel.rgb = appliedIrradiance;
                        else if (charDebug == 5) {
                            int dc = int(uCharMat.x + 0.5);
                            texel.rgb = dc == 1 ? vec3(0.9, 0.5, 0.35) : dc == 2 ? vec3(0.3, 0.3, 0.9) : dc == 3 ? vec3(0.95, 0.85, 0.1) : dc == 4 ? vec3(0.1, 0.8, 0.9) : dc == 5 ? vec3(0.8, 0.8, 0.8) : dc == 6 ? vec3(0.2, 0.8, 0.3) : dc == 7 ? vec3(0.55, 0.3, 0.1) : vec3(0.15);
                        }
                    }
                }
            @end
        }
        @if(o_alpha)
            if (uHdrWorld && uFolDraw.x > 0.5 && uFolFrame[1].w > 0.0) {
                // World position of this leaf fragment from its depth, then the sun's light passing through the card
                // when the sun is behind it as seen from the camera (strongest at dawn and dusk), and a darker interior
                // towards the trunk.
                vec2 fp = (gl_FragCoord.xy - uFolView.xy) / max(uFolView.zw, vec2(1.0));
                vec4 wp = uFolInv * vec4(fp * 2.0 - 1.0, gl_FragCoord.z * 2.0 - 1.0, 1.0);
                vec3 fpos = wp.xyz / wp.w;
                vec3 fv = fpos - uFolFrame[0].xyz;
                fv /= max(length(fv), 1.0);
                vec3 fl = uFolFrame[1].xyz;
                float fback = clamp(dot(fv, fl), 0.0, 1.0);
                vec3 fn = cross(dFdx(fpos), dFdy(fpos));
                fn = length(fn) > 1e-8 ? normalize(fn) : vec3(0.0, 1.0, 0.0);
                float fside = clamp(dot(fn, fv) * dot(fn, fl) * 3.0 + 0.35, 0.0, 1.0);
                float fglow = pow(fback, 2.0) * fside;
                float fluma = dot(texel.rgb, vec3(0.2126, 0.7152, 0.0722));
                // The leaf's hue (not its baked brightness: backlit leaves are baked dark) tints the light passing through.
                vec3 fhue = texel.rgb / max(max(texel.r, max(texel.g, texel.b)), 0.002);
                vec3 trans = fhue * fhue * uFolFrame[2].xyz * fglow * uFolFrame[1].w * 1.1 * (0.6 + 0.4 * clamp(fluma * 6.0, 0.0, 1.0));
                float fao = 1.0;
                if (uFolFrame[0].w > 0.5) {
                    vec3 rel = fpos - uFolDraw.yzw;
                    fao = mix(0.62, 1.0, smoothstep(6.0, 60.0, length(rel.xz))) * mix(0.8, 1.0, smoothstep(0.0, 70.0, rel.y));
                }
                texel.rgb = texel.rgb * mix(1.0, fao, min(uFolFrame[1].w, 1.0)) + trans;
            }
        @end
        @if(o_material_maps)
            if (materialDiffuse) {
                vec3 specular = materialSpecular(materialN, materialRoughness, materialWeight, hdrDiffuse);
#ifndef GL_ES
                if (hdrDiffuse) specular += materialSkySpecular(materialN, materialRoughness, materialWeight);
#endif
                // Bound the additive lobe relative to the already lit linear texel, per
                // channel: painted cracks and shadowed/black texels must stay dark. The
                // smooth knee preserves small highlights; rough surfaces get at most 2%
                // extra linear radiance, while smoother wet/metal surfaces have more room.
                vec3 diffuse = uHdrWorld ? texel.rgb : hdrDecode(texel.rgb);
                float smoothness2 = (1.0 - materialRoughness) * (1.0 - materialRoughness);
                vec3 budget = max(diffuse, vec3(0.0)) * (0.02 + 0.5 * smoothness2 * smoothness2);
                specular = budget * specular / max(budget + specular, vec3(0.000001));
                if (uMaterialDebug == 4) specular = vec3(0.0);
                if (uHdrWorld) texel.rgb = diffuse + specular;
                else texel.rgb = fromLinear(vec4(diffuse + specular, 1.0)).rgb;
                if (uMaterialDebug == 1) texel.rgb = uHdrWorld ? specular : fromLinear(vec4(specular, 1.0)).rgb;
                if (uMaterialDebug == 2) texel.rgb = vec3(materialRoughness);
                if (uMaterialDebug == 3) {
                    float nv = dot(materialN, normalize(uMaterialEye.w > 0.0 ? uMaterialEye.xyz - vViewPos : -vViewPos));
                    texel.rgb = nv > 0.0 ? vec3(0.0, nv, 0.0) : vec3(-nv, 0.0, 0.0);
                }
            }
        @end
        // TODO discard if alpha is 0?
        @if(o_fog)
            @if(o_alpha)
                texel = vec4(mix(texel.rgb, uHdrWorld ? hdrDecode(vFog.rgb) * uHdrFog : vFog.rgb, vFog.a), texel.a);
            @else
                texel = mix(texel, uHdrWorld ? hdrDecode(vFog.rgb) * uHdrFog : vFog.rgb, vFog.a);
            @end
        @end

        @if(o_texture_edge && o_alpha)
            @if(o_alpha_to_coverage)
                // Same 0.19 cutout threshold, but the alpha ramp across this pixel becomes MSAA coverage (the draw has
                // blending off and GL_SAMPLE_ALPHA_TO_COVERAGE on). Exactly half coverage where alpha == 0.19.
                texel.a = clamp((texel.a - 0.19) / max(fwidth(texel.a), 1.0 / 256.0) + 0.5, 0.0, 1.0);
            @else
                if (texel.a > 0.19) texel.a = 1.0; else discard;
            @end
        @end

        @if(o_alpha && o_noise)
            texel.a *= floor(clamp(random(vec3(floor(gl_FragCoord.xy * noise_scale), float(frame_count))) + texel.a, 0.0, 1.0));
        @end

        @if(o_grayscale)
            float intensity = (texel.r + texel.g + texel.b) / 3.0;
            vec3 new_texel = (uHdrWorld ? hdrDecode(vGrayscaleColor.rgb) : vGrayscaleColor.rgb) * intensity;
            texel.rgb = mix(texel.rgb, new_texel, vGrayscaleColor.a);
        @end

        @if(o_alpha)
            @if(o_alpha_threshold)
                if (texel.a < 8.0 / 256.0) discard;
            @end
            @if(o_invisible)
                texel.a = 0.0;
            @end
            @{vOutColor} = texel;
        @else
            @{vOutColor} = vec4(texel, 1.0);
        @end

        @if(srgb_mode)
            if (!uHdrWorld) @{vOutColor} = fromLinear(@{vOutColor});
        @end

        @if(o_prim_depth)
            gl_FragDepth = prim_depth;
        @end
    }
@end