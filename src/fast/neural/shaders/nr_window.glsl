// Window attention for one 8x8 window per workgroup, shared by block32.comp and attention.comp.
//
// Four subgroups of 32, one 16-token tile each (the window in tiled order: its tokens, queries and keys alike, are
// tile-major). Each lane owns one token row of its tile (lane >> 1) and eight of every sixteen columns
// (lane & 1). A subgroup keeps a private [16][16] staging tile and a [16][32] operand in shared memory; only the
// normalized K and V cross subgroups, behind the one barrier of the kernel. Row reductions (the cosine norm, the
// softmax denominator) meet their other half with a lane shuffle.
//
// The includer binds the weight arena as `w8` (uvec4 view) and includes nr_common.glsl first.
#ifndef NR_WINDOW_GLSL
#define NR_WINDOW_GLSL
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_shader_subgroup_extended_types_float16 : require

// Shared memory, in halves: 14 KB per window.
const uint PRIVATE = 768u;   // per subgroup: [16][16] staging tile, then a [16][32] operand
const uint KN = 3072u;       // [64][32] normalized K
const uint VT = 5120u;       // [32][64] V transposed: a channel's 64 keys are contiguous (the B operand of P V)
shared float16_t lds[7168];

const int ROW = gl_CooperativeMatrixLayoutRowMajor;
const int COL = gl_CooperativeMatrixLayoutColumnMajor;

Oct loadLds(uint index) {
  return Oct(f16vec4(lds[index], lds[index + 1u], lds[index + 2u], lds[index + 3u]),
             f16vec4(lds[index + 4u], lds[index + 5u], lds[index + 6u], lds[index + 7u]));
}
void storeLds(uint index, Oct v) {
  lds[index] = v.lo.x; lds[index + 1u] = v.lo.y; lds[index + 2u] = v.lo.z; lds[index + 3u] = v.lo.w;
  lds[index + 4u] = v.hi.x; lds[index + 5u] = v.hi.y; lds[index + 6u] = v.hi.z; lds[index + 7u] = v.hi.w;
}

// Global row of the first token of the window's local tile `tile` (0..3), or -1 outside the level. The window grid
// starts `shift` tiles (0 or 1) before the level's origin: the network's four window phases.
int windowTileRow(uint tile, uvec2 shift, uvec2 size4) {
  ivec2 t = ivec2(gl_WorkGroupID.xy) * 2 - ivec2(shift) + ivec2(tile & 1u, tile >> 1u);
  if (t.x < 0 || t.y < 0 || t.x >= int(size4.x) || t.y >= int(size4.y)) return -1;
  return (t.y * int(size4.x) + t.x) * 16;
}

// The window exponential, a bit trick on the half: 2^(1.4375 s - 5.375) with a linear mantissa. Per half:
// (bits << 5) ^ 0x8000 kept to 16 bits (the clamp bounds the pattern to 0x3c20..0x3e47, so bit 15 of the shifted
// value is always set and the reference's "+ 0x8000" is this xor). On packed pairs: the mask drops what would shift
// out of each half.
f16vec4 expWeight(f16vec4 s) {
  precise f16vec4 affine = fma(s, f16vec4(0.044921875), f16vec4(1.30078125));
  affine = clamp(affine, f16vec4(1.03125), f16vec4(1.5693359375));
  uvec2 bits = uvec2(packFloat2x16(affine.xy), packFloat2x16(affine.zw));
  bits = ((bits & 0x07ff07ffu) << 5u) ^ 0x80008000u;
  return f16vec4(unpackFloat2x16(bits.x), unpackFloat2x16(bits.y));
}

// One of Q (which = 0), K (1), V (2) of this lane's token: columns c and c + 16 for its eight c. Q and K are
// cosine normalized (Q also takes the head's learned scale), all three land on the E4M3 grid: Q in the subgroup's
// operand, K and V in the window's shared tables.
void publishQkv(uint which, Oct v0, Oct v1, float16_t qScale, uint operand, uint row, uint part, uint token) {
  if (which == 2u) {
    [[unroll]] for (uint t = 0u; t < 2u; ++t) {
      const Oct q = quantOct(t == 0u ? v0 : v1);
      const uint c = VT + (t * 16u + part * 8u) * 64u + token;
      lds[c] = q.lo.x;         lds[c + 64u] = q.lo.y;   lds[c + 128u] = q.lo.z;  lds[c + 192u] = q.lo.w;
      lds[c + 256u] = q.hi.x;  lds[c + 320u] = q.hi.y;  lds[c + 384u] = q.hi.z;  lds[c + 448u] = q.hi.w;
    }
    return;
  }
  // 1 / |v| over the token's 32 halves, the reference's reduction: r[c] = fma(v[c], v[c], f16(v[c + 16]^2)), then a
  // stride 8/4/2/1 tree. This lane holds c and c + 16 for eight c; its neighbour holds the other eight.
  precise f16vec4 hl = v1.lo * v1.lo, hh = v1.hi * v1.hi;
  precise f16vec4 rl = fma(v0.lo, v0.lo, hl), rh = fma(v0.hi, v0.hi, hh);
  precise f16vec4 ol = subgroupShuffleXor(rl, 1u), oh = subgroupShuffleXor(rh, 1u);
  precise f16vec4 s8l = rl + ol, s8h = rh + oh;      // stride 8 (commutative: both lanes agree)
  precise f16vec4 s4 = s8l + s8h;                    // stride 4
  precise f16vec2 s2 = s4.xy + s4.zw;                // stride 2
  precise float16_t s1 = s2.x + s2.y;
  // A zero row (every token outside the level) would be 0 * inf: it publishes as +0, the reference's NaN rule.
  const float16_t norm = s1 > float16_t(0.0) ? float16_t(inversesqrt(float(s1))) : float16_t(0.0);
  [[unroll]] for (uint t = 0u; t < 2u; ++t) {
    const Oct v = t == 0u ? v0 : v1;
    precise f16vec4 lo = v.lo * norm, hi = v.hi * norm;
    if (which == 0u) { lo *= qScale; hi *= qScale; }
    const uint c = t * 16u + part * 8u;
    storeLds((which == 0u ? operand + row * 32u : KN + token * 32u) + c, Oct(quantE4(lo), quantE4(hi)));
  }
}

// After the barrier: scores Q[16][32] * K^T + prior, exponentials, softmax, P[16][64] * V. Leaves the attended
// values of this subgroup's 16 tokens in the two accumulators (columns 0-15 and 16-31), still raw.
// `priorBase`: this head's [64 query][64 key] f16 prior in the arena, in halves (a multiple of 8).
void windowAttend(uint stage, uint operand, uint row, uint part, uint token, uint priorBase, out ACC out0, out ACC out1) {
  const uint mine = stage + row * 16u + part * 8u;
  Oct e[4];   // this lane's half of the row's weights, in registers throughout
  MA A0, A1;
  coopMatLoad(A0, lds, operand, 32u, ROW);
  coopMatLoad(A1, lds, operand + 16u, 32u, ROW);
  precise f16vec4 sum = f16vec4(0.0);
  [[unroll]] for (uint t = 0u; t < 4u; ++t) {
    MB B0, B1;
    coopMatLoad(B0, lds, KN + t * 512u, 32u, COL);
    coopMatLoad(B1, lds, KN + t * 512u + 16u, 32u, COL);
    ACC acc = coopMatMulAdd(A0, B0, ACC(0.0));
    acc = coopMatMulAdd(A1, B1, acc);
    coopMatStore(ACC16(acc), lds, stage, 16u, ROW);
    const Oct s = loadLds(mine);
    const Oct prior = unpackOct(w8[(priorBase + token * 64u + t * 16u + part * 8u) >> 3u]);
    precise f16vec4 sl = s.lo + prior.lo, sh = s.hi + prior.hi;
    e[t] = Oct(expWeight(sl), expWeight(sh));
    sum += e[t].lo;
    sum += e[t].hi;
  }
  precise f16vec2 s2 = sum.xy + sum.zw;
  precise float16_t partial = s2.x + s2.y;
  precise float16_t total = partial + subgroupShuffleXor(partial, 1u);
  const float16_t reciprocal = float16_t(1.0 / float(total));
  ACC acc0 = ACC(0.0), acc1 = ACC(0.0);
  [[unroll]] for (uint kt = 0u; kt < 4u; ++kt) {   // one key tile at a time through the staging tile
    precise f16vec4 lo = e[kt].lo * reciprocal, hi = e[kt].hi * reciprocal;
    storeLds(mine, Oct(quantE4(lo), quantE4(hi)));
    MA P;
    MB V0, V1;
    coopMatLoad(P, lds, stage, 16u, ROW);
    coopMatLoad(V0, lds, VT + kt * 16u, 64u, COL);
    coopMatLoad(V1, lds, VT + 1024u + kt * 16u, 64u, COL);
    acc0 = coopMatMulAdd(P, V0, acc0);
    acc1 = coopMatMulAdd(P, V1, acc1);
  }
  out0 = acc0;
  out1 = acc1;
}
#endif
