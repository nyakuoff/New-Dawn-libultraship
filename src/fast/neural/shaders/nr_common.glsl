// New Dawn neural rendering, FP16 path: what every kernel shares.
//
// The network's values are E4M3 bytes or IEEE halves at every kernel boundary (OpenDLSS-NR docs/numerics.md, MIT).
// This path has no FP8 type: E4M3 values travel as halves that sit on the E4M3 grid, and the grid is applied at
// each publication point by quantE4. Activations are f16 [tokens][channels] with the tokens in 4x4 pixel tiles
// of 16 (tiledIndex), so a quarter of an 8x8 window is 16 contiguous rows.
//
// What RADV / RDNA3 asks of a kernel (measured; see docs/NEURAL_RENDERING.md):
//  - a cooperative matrix loads lane by lane: keep each lane's 16 values contiguous (A row-major, B column-major
//    out of k16-tile-major weights), or every element becomes its own 2-byte load;
//  - there are no per-element matrix operations: accumulator -> shared memory -> lanes -> shared memory -> operand;
//  - shared memory is what limits occupancy (64 KB per compute unit): each subgroup publishes its own 16x16 tile
//    through a private scratch, with no barrier.
#ifndef NR_COMMON_GLSL
#define NR_COMMON_GLSL
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_16bit_storage : require

#define ACC coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#define ACC16 coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#define MA coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MB coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>

struct Oct { f16vec4 lo, hi; };   // eight consecutive halves: one 16-byte word

uvec4 packOct(Oct v) {
  return uvec4(packFloat2x16(v.lo.xy), packFloat2x16(v.lo.zw), packFloat2x16(v.hi.xy), packFloat2x16(v.hi.zw));
}
Oct unpackOct(uvec4 w) {
  return Oct(f16vec4(unpackFloat2x16(w.x), unpackFloat2x16(w.y)), f16vec4(unpackFloat2x16(w.z), unpackFloat2x16(w.w)));
}

// Four halves rounded to the E4M3 grid (RNE at 3 mantissa bits, the 2^-9 grid below 2^-6, saturation at 448).
// The grid step of a binade is the ulp of 1.5 * 2^(e + 7), so (|x| + M) - M in half arithmetic is the publication:
// two packed adds instead of per-element bit surgery. M / step = 1536 is even, so the tie rule is E4M3's.
// Checked against the reference conversion for every non-NaN half (nd-nr selftest). NaN must not get here (min and
// max drop it): the cosine normalizations guard their zero rows instead, which is where the reference's NaN -> +0
// rule applies.
f16vec4 quantE4(f16vec4 x) {
  f16vec4 mag = min(abs(x), f16vec4(448.0));
  f16vec4 binade = max(mag, f16vec4(0.015625));
  uvec2 lift = (uvec2(packFloat2x16(binade.xy), packFloat2x16(binade.zw)) & 0x7c007c00u) + 0x1e001e00u;
  f16vec4 M = f16vec4(unpackFloat2x16(lift.x), unpackFloat2x16(lift.y));
  precise f16vec4 sum = mag + M;
  precise f16vec4 q = sum - M;
  uvec2 sign = uvec2(packFloat2x16(x.xy), packFloat2x16(x.zw)) & 0x80008000u;
  uvec2 bits = uvec2(packFloat2x16(q.xy), packFloat2x16(q.zw)) | sign;
  return f16vec4(unpackFloat2x16(bits.x), unpackFloat2x16(bits.y));
}
Oct quantOct(Oct v) { return Oct(quantE4(v.lo), quantE4(v.hi)); }

// The network's cubic SiLU in half FMAs (OpenDLSS-NR siluHalf).
f16vec4 silu(f16vec4 value) {
  f16vec4 bounded = clamp(value, f16vec4(-4.0), f16vec4(4.0));
  precise f16vec4 inner = fma(f16vec4(-0.055908203125), abs(bounded), f16vec4(0.447265625));
  precise f16vec4 polynomial = fma(bounded, inner, f16vec4(0.89453125));
  precise f16vec4 result = value * polynomial;
  return result;
}

// Token index of pixel (x, y) at a level `width4` tiles wide, and back.
uint tiledIndex(uint x, uint y, uint width4) { return ((y >> 2u) * width4 + (x >> 2u)) * 16u + (y & 3u) * 4u + (x & 3u); }
uvec2 tiledPixel(uint token, uint width4) {
  const uint tile = token >> 4u;
  return uvec2((tile % width4) * 4u + (token & 3u), (tile / width4) * 4u + ((token >> 2u) & 3u));
}
#endif
