// The shape of the DLSS-NR network: the padded field and its six levels, the window phases, and where each
// matrix, scale vector and prior sits inside a block's tensor record.
//
// Ported from OpenDLSS-NR (MIT, see THIRD_PARTY.md): nr_graph.cpp Geometry / FusedLayout and docs/network.md,
// docs/weights.md. These are facts about the network, shared by the FP16 graph and by the CPU reference.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace Fast::Neural {

inline uint32_t AlignUp(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// The image is padded to a field whose size the network's own shape rules dictate, so that six halvings and the
// decoder's whole-window upsample all fit. The padding decides which tokens exist: it is part of the arithmetic.
struct Geometry {
    uint32_t mValidWidth = 0, mValidHeight = 0;
    uint32_t mFullWidth = 0, mFullHeight = 0;
    uint32_t mWidth[6] = {}, mHeight[6] = {}; // level 0 (half the field) .. level 5 (the ViT's)

    static uint32_t FieldAlignment(uint32_t valid) {
        uint32_t reductions = 0, size = valid;
        for (int level = 0; level < 6; ++level) {
            const uint32_t half = AlignUp((size + 1) / 2, 4);
            if (half < size) {
                ++reductions;
            }
            if (level == 0 && half % 8 != 0) {
                ++reductions; // the level-0 upsample emits whole 8-pixel windows
            }
            size = half;
        }
        return 1u << reductions;
    }

    static Geometry FromValid(uint32_t validWidth, uint32_t validHeight) {
        Geometry g;
        g.mValidWidth = validWidth;
        g.mValidHeight = validHeight;
        const uint32_t alignWidth = FieldAlignment(validWidth), alignHeight = FieldAlignment(validHeight);
        g.mFullWidth = std::max(320u, AlignUp(validWidth, alignWidth));
        g.mFullHeight = std::max(320u, AlignUp(validHeight, alignHeight));
        // Native's rule, with no stated reason; it has to be reproduced because the field decides the window grid.
        if (g.mFullWidth % (4 * alignWidth) == 0 && g.mFullHeight % (4 * alignHeight) == 0) {
            g.mFullWidth += alignWidth;
        }
        uint32_t width = g.mFullWidth, height = g.mFullHeight;
        for (int level = 0; level < 6; ++level) {
            width = AlignUp((width + 1) / 2, 4);
            height = AlignUp((height + 1) / 2, 4);
            g.mWidth[level] = width;
            g.mHeight[level] = height;
        }
        if (g.mWidth[0] % 8 || g.mHeight[0] % 8 || g.mWidth[0] * 2 != g.mFullWidth ||
            g.mHeight[0] * 2 != g.mFullHeight) {
            throw std::runtime_error("unsupported neural rendering size " + std::to_string(validWidth) + "x" +
                                     std::to_string(validHeight));
        }
        return g;
    }

    uint32_t FieldTokens() const {
        return mFullWidth * mFullHeight;
    }
    uint32_t Tokens(int level) const {
        return mWidth[level] * mHeight[level];
    }
    uint32_t PaddedVitTokens() const {
        return AlignUp(Tokens(5), 64);
    }
};

// The window grid of a block: origin (-x, -y) from a multiple of 8. The phase advances once per block at a level,
// and a decoder stage continues the count its encoder stage left.
inline void WindowPhase(uint32_t index, uint32_t& shiftX, uint32_t& shiftY) {
    static const uint32_t kPhases[4][2] = { { 0, 0 }, { 4, 4 }, { 4, 0 }, { 0, 4 } };
    shiftX = kPhases[index & 3][0];
    shiftY = kPhases[index & 3][1];
}

// Byte offsets inside the tensor record of a window block of C channels.
struct BlockLayout {
    uint32_t mChannels = 0;
    uint32_t mHeads = 0;
    uint32_t mExperts = 0;  // C / 32 for C >= 64; 0 for the dense 32-channel FFN
    uint32_t mExpand = 0;   // W1: [32][128], or per expert [C][128]
    uint32_t mContract = 0; // W2: [128][32], or per expert [128][32] followed by W3 [C][C]
    uint32_t mW3 = 0;
    uint32_t mFfnScale = 0;   // f16 per channel
    uint32_t mQkv = 0;        // [C][3C], head-major Q | K | V
    uint32_t mPrior = 0;      // [heads][64][64] f16
    uint32_t mQScale = 0;     // f32 per head
    uint32_t mProjection = 0; // [C][C]
    uint32_t mAttnScale = 0;  // f16 per channel
    uint32_t mEnd = 0;        // what follows: 16 bytes of padding, or the next stage's C -> 2C matrix
    // Variants:
    uint32_t mInputAdapter = 0;    // block 0: f16 [16][32]
    uint32_t mUpsample = 0;        // first decoder block of a stage: [2C][C]
    uint32_t mTransitionScale = 0; // ... and the skip merge's f16 scale per channel
    uint32_t mInputScale = 0;      // block 70: post blend scale of the upsampled block 69
    uint32_t mAdapterScale = 0;    // block 70: post blend scale of block 0's output
    uint32_t mHead = 0;            // block 70: f16 [32][4]
};

inline BlockLayout StandardLayout(uint32_t channels) {
    BlockLayout l;
    l.mChannels = channels;
    l.mHeads = channels / 32;
    l.mExperts = channels >= 64 ? channels / 32 : 0;
    l.mExpand = 0;
    const uint32_t expandBytes = l.mExperts ? l.mExperts * channels * 128 : channels * 128;
    l.mContract = expandBytes;
    l.mW3 = l.mContract + l.mExperts * 128 * 32;
    const uint32_t ffnBytes =
        l.mExperts ? expandBytes + l.mExperts * 128 * 32 + channels * channels : expandBytes + 128 * channels;
    l.mFfnScale = ffnBytes + 16;
    l.mQkv = l.mFfnScale + channels * 2 + 16;
    l.mPrior = l.mQkv + channels * channels * 3;
    l.mQScale = l.mPrior + l.mHeads * 8192;
    l.mProjection = l.mQScale + AlignUp(l.mHeads * 4, 16);
    l.mAttnScale = l.mProjection + channels * channels;
    l.mEnd = l.mAttnScale + channels * 2;
    return l;
}

// Block 0: the f16 input adapter sits after W2.
inline BlockLayout PreLayout() {
    BlockLayout l = StandardLayout(32);
    l.mInputAdapter = 8208;
    l.mFfnScale = 9232;
    l.mQkv = 9312;
    l.mPrior = 12384;
    l.mQScale = 20576;
    l.mProjection = 20592;
    l.mAttnScale = 21616;
    l.mEnd = 21680;
    return l;
}

// The first block of a decoder stage: the 2C -> C upsample matrix and the skip merge's scale are in its record.
inline BlockLayout UpsampleLayout(uint32_t channels) {
    BlockLayout l = StandardLayout(channels);
    const uint32_t narrowPadding = channels == 32 ? 16 : 0;
    const uint32_t expandBytes = l.mExperts ? l.mExperts * channels * 128 : channels * 128;
    const uint32_t ffnBytes =
        l.mExperts ? expandBytes + l.mExperts * 128 * 32 + channels * channels : expandBytes + 128 * channels;
    l.mUpsample = ffnBytes;
    l.mFfnScale = l.mUpsample + 2 * channels * channels + narrowPadding;
    l.mTransitionScale = l.mFfnScale + channels * 2 + narrowPadding;
    l.mQkv = l.mTransitionScale + channels * 2;
    l.mPrior = l.mQkv + channels * channels * 3;
    l.mQScale = l.mPrior + l.mHeads * 8192;
    l.mProjection = l.mQScale + AlignUp(l.mHeads * 4, 16);
    l.mAttnScale = l.mProjection + channels * channels;
    l.mEnd = l.mAttnScale + channels * 2;
    return l;
}

// Block 70: the two post blend scales, and the f16 RGBA head at the end.
inline BlockLayout PostLayout() {
    BlockLayout l = StandardLayout(32);
    l.mFfnScale = 8208;
    l.mInputScale = 8272;
    l.mAdapterScale = 8336;
    l.mQkv = 8400;
    l.mPrior = 11472;
    l.mQScale = 19664;
    l.mProjection = 19680;
    l.mAttnScale = 20704;
    l.mHead = 20784;
    l.mEnd = 21808;
    return l;
}

// The stages. Encoder: block 0 (field), 1-4 (level 0, 32), 5-8 (64), 9-14 (128), 15-22 (256), 23-30 (512);
// ViT 31-38 (level 5, 1024); decoder: 39 is the 1024 -> 512 transition, 40-47 (512), 48-55 (256), 56-61 (128),
// 62-65 (64), 66-69 (32), 70 (field).
struct Stage {
    int mLevel;
    uint32_t mChannels;
    int mFirst, mLast;
};
inline constexpr Stage kEncoderStages[] = {
    { 0, 32, 1, 4 }, { 1, 64, 5, 8 }, { 2, 128, 9, 14 }, { 3, 256, 15, 22 }, { 4, 512, 23, 30 }
};
inline constexpr Stage kDecoderStages[] = {
    { 4, 512, 40, 47 }, { 3, 256, 48, 55 }, { 2, 128, 56, 61 }, { 1, 64, 62, 65 }, { 0, 32, 66, 69 }
};

} // namespace Fast::Neural
