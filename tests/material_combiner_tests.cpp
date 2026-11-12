#include <gtest/gtest.h>
#include "fast/interpreter.h"

// Actual world combiner IDs, after the interpreter's canonicalization. These catch
// the two-tile and primitive-tint cases that excluded real surfaces in captures.
TEST(MaterialCombiner, LitAlbedoWithPrimitiveTint) {
    CCFeatures features{};
    gfx_cc_get_features(0xd0000d0280000108ULL, SHADER_OPT(_2CYC), &features);
    bool blend = true;
    EXPECT_EQ(NewDawnMaterialShadeMask(features, blend), 3u);
    EXPECT_FALSE(blend);
}

TEST(MaterialCombiner, AlbedoTimesDetailThenShade) {
    CCFeatures features{};
    gfx_cc_get_features(0x020d010d818a080aULL, SHADER_OPT(_2CYC), &features);
    bool blend = true;
    EXPECT_EQ(NewDawnMaterialShadeMask(features, blend), 1u);
    EXPECT_FALSE(blend);
}

TEST(MaterialCombiner, TwoAlbedoBlendThenShade) {
    CCFeatures features{};
    gfx_cc_get_features(0xd000020d818a818aULL, SHADER_OPT(_2CYC), &features);
    bool blend = false;
    EXPECT_EQ(NewDawnMaterialShadeMask(features, blend), 2u);
    EXPECT_TRUE(blend);
}

TEST(MaterialCombiner, SingleCycleAndUnlitTexture) {
    CCFeatures features{};
    bool blend = false;
    gfx_cc_get_features(0x0108, 0, &features); // texel0 * input1
    EXPECT_EQ(NewDawnMaterialShadeMask(features, blend), 1u);
    gfx_cc_get_features(0x8000, 0, &features); // texel0, no separable shade
    EXPECT_EQ(NewDawnMaterialShadeMask(features, blend), 0u);
}

TEST(MaterialCombiner, RejectsUnsupportedSecondCycle) {
    CCFeatures features{};
    bool blend = false;
    gfx_cc_get_features(0x0000800000000108ULL, SHADER_OPT(_2CYC), &features);
    EXPECT_EQ(NewDawnMaterialShadeMask(features, blend), 0u);
}
