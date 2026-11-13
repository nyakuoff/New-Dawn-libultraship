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

#endif

        // Same light model as the CPU path in GfxSpVertex: this replaces the per-vertex shade
        // colour with the same arithmetic evaluated per fragment. Values are kept in the 0..255
        // range the CPU uses, including its per-light truncation, so parity is exact where the
        // interpolated normal and position are constant across a triangle.
        vec3 newDawnIrradiance(vec3 shadeNormal) {
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
                } else {
                    intensity = max(dot(n, uLightDir[i]), 0.0) * uHdrSun;
                }
                lit += intensity * uLightColor[i] / 255.0;
            }
            return max(lit, vec3(0.0));
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
    @end

    @if(o_material_maps)
        uniform sampler2D uMaterialNormal;
        uniform sampler2D uMaterialRoughness;
        uniform float uMaterialStrength;
        uniform int uMaterialDebug; // 1: specular only, 2: roughness, 3: N.V sign, 4: diffuse only
        uniform vec2 uMaterialMirror;
#ifndef GL_ES
        uniform bool uSkyReflectionsEnabled;
        uniform samplerCube uSkySpecular;
        uniform vec3 uSkyEye;
#endif

        // Cotangents use unclamped UV derivatives. The -T direction matches the generator's
        // +Y-up normals (texture rows increase down). Keeping the signed determinant handles
        // mirrored islands; repeat/mirror sampling and tile clamp match the albedo sampler.
        vec3 materialNormal(vec2 uv, vec2 dx, vec2 dy, float weight, out float variance) {
            // Cofactor inverse-transpose also works on this backend's GLSL 1.30 path.
            mat3 cofactor = mat3(cross(uViewMatrix[1], uViewMatrix[2]),
                                 cross(uViewMatrix[2], uViewMatrix[0]),
                                 cross(uViewMatrix[0], uViewMatrix[1]));
            float determinant = dot(uViewMatrix[0], cofactor[0]);
            vec3 transformed = cofactor * vNormal * (determinant < 0.0 ? -1.0 : 1.0);
            vec3 px = dFdx(vViewPos), py = dFdy(vViewPos);
            vec3 n = transformed / max(length(transformed), 0.0001);
            float det = dx.x * dy.y - dx.y * dy.x;
            vec3 t = px * dy.y - py * dx.y;
            vec3 b = py * dx.x - px * dy.x;
            float handedness = det < 0.0 ? -1.0 : 1.0;
            t = (t - n * dot(n, t)) * handedness;
            b = (b - n * dot(n, b)) * handedness;
            float t2 = dot(t, t), b2 = dot(b, b);
            vec2 xy = textureGrad(uMaterialNormal, uv, dx, dy).rg * 2.0 - 1.0;
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
            vec3 v = -vViewPos / max(length(vViewPos), 0.0001);
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
    @if(o_three_point_filtering)
        if(texture_filtering[id] == @{FILTER_THREE_POINT}) {
            return filter3point(tex, uv, texSize, ddx, ddy);
        }
    @end
        return textureGrad(tex, uv, ddx, ddy);
    }

    #define TEX_SIZE(tex) vec2(texture_width[tex], texture_height[tex])

    void main() {
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

        @if(o_material_maps)
            float materialWeight = 1.0;
            @if(material_blend)
                materialWeight = clamp(vInput@{material_weight_index}.r, 0.0, 1.0);
                @if(material_weight_invert)
                    materialWeight = 1.0 - materialWeight;
                @end
            @end
            float materialVariance;
            vec3 materialN = materialNormal(vTexCoordAdj@{material_tile}, ddx@{material_tile}, ddy@{material_tile}, materialWeight, materialVariance);
            float materialRoughness = textureGrad(uMaterialRoughness, vTexCoordAdj@{material_tile}, ddx@{material_tile}, ddy@{material_tile}).r;
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
            texel.rgb = hdrDecode(texel.rgb) * uHdrSky;
            @if(o_per_pixel_lighting)
                if (hdrDiffuse) texel.rgb *= newDawnIrradiance(shadeNormal);
            @end
        }
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
                    float nv = dot(materialN, normalize(-vViewPos));
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
