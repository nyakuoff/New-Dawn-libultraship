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

        // Same light model as the CPU path in GfxSpVertex: this replaces the per-vertex shade
        // colour with the same arithmetic evaluated per fragment. Values are kept in the 0..255
        // range the CPU uses, including its per-light truncation, so parity is exact where the
        // interpolated normal and position are constant across a triangle.
        vec3 newDawnShade() {
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
                    float total = clamp(dot(vNormal, lightIntensity), -1.0, 1.0);
                    // uPointLightQuality 0 keeps the CPU's integer distance steps.
                    float distf = mix(floor(dist), dist, uPointLightQuality);
                    float attenuation =
                        (distf * uLightAtten[i].x * 2.0 + distf * distf * uLightAtten[i].y / 8.0) / 65535.0 + 1.0;
                    intensity = total / attenuation;
                } else {
                    intensity = dot(vNormal, uLightDir[i]) / 127.0;
                }
                lit += floor(max(intensity, 0.0) * uLightColor[i]);
            }
            return clamp(floor(lit), 0.0, 255.0) / 255.0;
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
        @if(o_per_pixel_lighting)
            vec3 newDawnLit = newDawnShade();
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
        // TODO discard if alpha is 0?
        @if(o_fog)
            @if(o_alpha)
                texel = vec4(mix(texel.rgb, vFog.rgb, vFog.a), texel.a);
            @else
                texel = mix(texel, vFog.rgb, vFog.a);
            @end
        @end

        @if(o_texture_edge && o_alpha)
            if (texel.a > 0.19) texel.a = 1.0; else discard;
        @end

        @if(o_alpha && o_noise)
            texel.a *= floor(clamp(random(vec3(floor(gl_FragCoord.xy * noise_scale), float(frame_count))) + texel.a, 0.0, 1.0));
        @end

        @if(o_grayscale)
            float intensity = (texel.r + texel.g + texel.b) / 3.0;
            vec3 new_texel = vGrayscaleColor.rgb * intensity;
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
            @{vOutColor} = fromLinear(@{vOutColor});
        @end

        @if(o_prim_depth)
            gl_FragDepth = prim_depth;
        @end
    }
@end