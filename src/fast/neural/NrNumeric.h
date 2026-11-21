// E4M3FN / IEEE binary16 helpers and the reference arithmetic of the DLSS-NR network on the CPU.
//
// From OpenDLSS-NR (MIT, see THIRD_PARTY.md): src/numeric.h and the scalar parts of src/reference.cpp. The logic is
// unchanged; these functions are the specification the FP16 kernels are measured against, so they must not be
// "simplified".
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace Fast::Neural::Num {

inline uint32_t F32Bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    return bits;
}

inline float F32FromBits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, 4);
    return value;
}

// IEEE RNE f32 -> f16.
inline uint16_t F16Bits(float value) {
    uint32_t bits = F32Bits(value);
    uint32_t sign = (bits >> 16) & 0x8000;
    uint32_t exponent = (bits >> 23) & 0xff;
    uint32_t mantissa = bits & 0x7fffff;
    if (exponent == 0xff) {
        return (uint16_t)(sign | (mantissa ? 0x7e00 : 0x7c00));
    }
    int halfExponent = (int)exponent - 112;
    if (halfExponent >= 31) {
        return (uint16_t)(sign | 0x7c00);
    }
    if (halfExponent <= 0) {
        if (halfExponent < -10) {
            return (uint16_t)sign;
        }
        uint32_t normalized = mantissa | 0x800000;
        uint32_t shift = (uint32_t)(14 - halfExponent);
        uint32_t rounded = normalized >> shift;
        uint32_t remainder = normalized & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1))) {
            rounded += 1;
        }
        return (uint16_t)(sign | rounded);
    }
    uint32_t rounded = mantissa >> 13;
    uint32_t remainder = mantissa & 0x1fff;
    if (remainder > 0x1000 || (remainder == 0x1000 && (rounded & 1))) {
        rounded += 1;
        if (rounded == 0x400) {
            rounded = 0;
            halfExponent += 1;
        }
    }
    if (halfExponent >= 31) {
        return (uint16_t)(sign | 0x7c00);
    }
    return (uint16_t)(sign | ((uint32_t)halfExponent << 10) | rounded);
}

inline float F16ToF32(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits & 0x8000) << 16;
    uint32_t exponent = (bits >> 10) & 0x1f;
    uint32_t mantissa = bits & 0x3ff;
    if (exponent == 0) {
        if (mantissa == 0) {
            return F32FromBits(sign);
        }
        int shift = 0;
        while (!(mantissa & 0x400)) {
            mantissa <<= 1;
            shift += 1;
        }
        mantissa &= 0x3ff;
        return F32FromBits(sign | ((uint32_t)(113 - shift) << 23) | (mantissa << 13));
    }
    if (exponent == 31) {
        return F32FromBits(sign | 0x7f800000 | (mantissa << 13));
    }
    return F32FromBits(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

inline float RoundF16(float value) {
    return F16ToF32(F16Bits(value));
}

// E4M3FN byte to float; the NaN code decodes to NaN.
inline float E4m3ToF32(uint8_t byte) {
    int sign = (byte & 0x80) ? -1 : 1;
    int exponent = (byte >> 3) & 0xf;
    int mantissa = byte & 0x7;
    if (exponent == 0) {
        return sign * (float)mantissa * (1.0f / 512.0f);
    }
    if (exponent == 0xf && mantissa == 0x7) {
        return NAN;
    }
    return sign * (1.0f + mantissa / 8.0f) * std::ldexp(1.0f, exponent - 7);
}

// An already rounded half -> E4M3FN code: RNE, finite saturation at 448, NaN -> +0; the sign of a zero survives.
inline uint8_t E4m3FromF16Bits(uint16_t half) {
    bool isNaN = (half & 0x7c00) == 0x7c00 && (half & 0x03ff) != 0;
    if (isNaN) {
        return 0;
    }
    if ((half & 0x7fff) == 0) {
        return (uint8_t)((half >> 8) & 0x80);
    }
    uint32_t negative = (half & 0x8000) ? 0x80 : 0;
    uint32_t exponent = (half >> 10) & 0x1f;
    uint32_t mantissa = half & 0x3ff;
    auto shiftRne = [](uint32_t value, uint32_t shift) {
        uint32_t quotient = value >> shift;
        uint32_t remainder = value & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        return quotient + ((remainder > halfway || (remainder == halfway && (quotient & 1))) ? 1u : 0u);
    };
    uint32_t code;
    if (exponent == 31) {
        code = 0x7e;
    } else if (exponent <= 8) {
        uint32_t significand = exponent == 0 ? mantissa : 1024 + mantissa;
        uint32_t shift = exponent == 0 ? 15 : 16 - exponent;
        code = shiftRne(significand, shift);
        if (code > 8) {
            code = 8;
        }
    } else {
        uint32_t e4Exponent = exponent - 8;
        uint32_t e4Mantissa = shiftRne(mantissa, 7);
        if (e4Mantissa == 8) {
            e4Mantissa = 0;
            e4Exponent += 1;
        }
        code = (e4Exponent > 15 || (e4Exponent == 15 && e4Mantissa > 6)) ? 0x7e : ((e4Exponent << 3) | e4Mantissa);
    }
    return (uint8_t)(negative | code);
}

// The E4M3 publication of a value, as a value: half first, then the grid. NaN -> +0.
inline float Fp8Domain(float value) {
    return E4m3ToF32(E4m3FromF16Bits(F16Bits(value)));
}

inline int NormalExponent(float value) {
    uint32_t bits = F32Bits(std::fabs(value));
    return (int)((bits >> 23) & 0xff) - 127;
}
inline int E4m3Exponent(float value) {
    return std::max(NormalExponent(value), -6);
}
inline int F16Exponent(float value) {
    return std::max(NormalExponent(value), -14);
}

// The network's cubic SiLU: five half operations, each rounded once.
inline float CubicSilu(float value) {
    float bounded = RoundF16(std::min(std::max(value, -4.0f), 4.0f));
    float absolute = RoundF16(std::fabs(bounded));
    float inner = RoundF16(-0.055908203125f * absolute + 0.447265625f);
    float polynomial = RoundF16(bounded * inner + 0.89453125f);
    return RoundF16(value * polynomial);
}

// Half of a tensor-core k32 step (m16n8k32 E4M3 x E4M3 -> f16): 16 products and the incoming half accumulator at a
// shared exponent, 13 fractional bits, truncated, one RNE to half.
inline float Fp8Dot16(const float* a, const float* b, float accumulator) {
    if (std::isinf(accumulator) || std::isnan(accumulator)) {
        return accumulator;
    }
    int maximumExponent = -21;
    if (accumulator != 0.0f) {
        maximumExponent = F16Exponent(accumulator);
    }
    for (int i = 0; i < 16; ++i) {
        if (a[i] != 0.0f && b[i] != 0.0f) {
            maximumExponent = std::max(maximumExponent, E4m3Exponent(a[i]) + E4m3Exponent(b[i]));
        }
    }
    double fused = 0.0;
    if (accumulator != 0.0f) {
        int exponent = F16Exponent(accumulator);
        float significand = accumulator * std::ldexp(1.0f, -exponent);
        fused += std::trunc(significand * std::ldexp(1.0f, exponent - maximumExponent + 13)) * 0.0001220703125;
    }
    for (int i = 0; i < 16; ++i) {
        if (a[i] != 0.0f && b[i] != 0.0f) {
            int ea = E4m3Exponent(a[i]), eb = E4m3Exponent(b[i]);
            float sa = a[i] * std::ldexp(1.0f, -ea), sb = b[i] * std::ldexp(1.0f, -eb);
            fused += std::trunc(sa * sb * std::ldexp(1.0f, ea + eb - maximumExponent + 13)) * 0.0001220703125;
        }
    }
    return RoundF16((float)(fused * std::ldexp(1.0, maximumExponent)));
}

// Half of an f16 k16 step (m16n8k16): 8 products, 24 fractional bits, exact integer accumulation.
inline float F16Dot8(const float* a, const float* b, float accumulator) {
    auto roundShiftRightEven = [](uint32_t value, uint32_t shift) -> uint32_t {
        if (shift == 0) {
            return value;
        }
        if (shift > 31) {
            return 0;
        }
        uint32_t truncated = value >> shift;
        uint32_t remainder = value & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        return truncated + ((remainder > halfway || (remainder == halfway && (truncated & 1))) ? 1u : 0u);
    };
    int maximumExponent = -21;
    if (accumulator != 0.0f) {
        maximumExponent = F16Exponent(accumulator);
    }
    for (int i = 0; i < 8; ++i) {
        if (a[i] != 0.0f && b[i] != 0.0f) {
            maximumExponent = std::max(maximumExponent, F16Exponent(a[i]) + F16Exponent(b[i]));
        }
    }
    int64_t fixedSum = 0;
    if (accumulator != 0.0f) {
        int exponent = F16Exponent(accumulator);
        float significand = accumulator * std::ldexp(1.0f, -exponent);
        fixedSum += (int64_t)std::trunc(significand * std::ldexp(1.0f, exponent - maximumExponent + 24));
    }
    for (int i = 0; i < 8; ++i) {
        if (a[i] != 0.0f && b[i] != 0.0f) {
            int ea = F16Exponent(a[i]), eb = F16Exponent(b[i]);
            float sa = a[i] * std::ldexp(1.0f, -ea), sb = b[i] * std::ldexp(1.0f, -eb);
            fixedSum += (int64_t)std::trunc(sa * sb * std::ldexp(1.0f, ea + eb - maximumExponent + 24));
        }
    }
    // Exact signed F24 integer -> IEEE half.
    const int binaryExponent = maximumExponent - 24;
    if (fixedSum == 0) {
        return 0.0f;
    }
    bool negative = fixedSum < 0;
    uint32_t magnitude = (uint32_t)(negative ? -fixedSum : fixedSum);
    uint32_t msb = 31;
    while (!(magnitude & (1u << msb))) {
        msb -= 1;
    }
    int valueExponent = (int)msb + binaryExponent;
    uint32_t halfBits = negative ? 0x8000 : 0;
    if (valueExponent >= -14) {
        uint32_t significand = msb > 10 ? roundShiftRightEven(magnitude, msb - 10) : magnitude << (10 - msb);
        if (significand >= 2048) {
            significand = 1024;
            valueExponent += 1;
        }
        if (valueExponent >= 16) {
            halfBits |= 0x7c00;
        } else {
            halfBits |= (uint32_t)(valueExponent + 15) << 10;
            halfBits |= significand - 1024;
        }
    } else {
        int subnormalScale = binaryExponent + 24;
        uint32_t mantissa = subnormalScale >= 0 ? magnitude << subnormalScale
                                                : roundShiftRightEven(magnitude, (uint32_t)-subnormalScale);
        halfBits |= std::min(mantissa, 1024u);
    }
    return F16ToF32((uint16_t)halfBits);
}

inline float AddF16(float a, float b) {
    return RoundF16((float)((double)a + (double)b));
}

// The window exponential: a bit trick on the half, 2^(1.4375 s - 5.375) with a linear mantissa.
inline float WindowExpWeight(float score) {
    float affine = RoundF16(std::fma(score, 0.044921875f, 1.30078125f));
    affine = std::min(std::max(affine, 1.03125f), 1.5693359375f);
    return F16ToF32((uint16_t)(((uint32_t)F16Bits(affine) << 5) + 0x8000u));
}

// The ViT's: 2^(1.4326 s - 3.6502). Its constants are f32 literals converted to half before the fma.
inline float VitExpWeight(float score) {
    const float slope = RoundF16(0.08953946828842163f), offset = RoundF16(1.7093614339828491f);
    float affine = RoundF16((float)((double)score * (double)slope + (double)offset));
    affine = std::min(std::max(affine, 1.439453125f), 1.9775390625f);
    return F16ToF32((uint16_t)(((uint32_t)F16Bits(affine) << 4) + 0x4000u));
}

} // namespace Fast::Neural::Num
