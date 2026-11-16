#ifndef NEWDAWN_LOOK_H
#define NEWDAWN_LOOK_H

/* New Dawn look profiles (docs/LOOK_PLAN.md). The look of a frame is one flat array of floats. The game resolves it
   once per game frame from its profiles (scene, clock, weather, story flags; all read-only) and hands it over inside
   NewDawnPostFxParams, so it travels in the display list: interpolated frames and the frozen-frame fixture see the
   values of the frame they draw. With gNewDawn.Look.Profiles off the array is not used and every feature takes its
   numbers from where it took them before.

   One row per field: X(id, name in the profile files, default, minimum, maximum, cvar, offset).
   - The default is the value of the New Dawn looks (NewDawnPresets.cpp), which is what profiles are authored against.
   - cvar: the player's own setting for the same quantity, or NULL. It stays a control on top of the profile: the look's
     value is multiplied by (cvar / default), or, when offset is 1 (exposure, in EV), shifted by (cvar - default). At
   the preset's values a profile holding the defaults therefore gives exactly what the cvars alone give. */
#define NEWDAWN_LOOK_FIELDS(X)                                                                     \
    X(KEY_LEVEL, "light.keyLevel", 1.2f, 0.0f, 16.0f, "gNewDawn.HDR.SunIntensity", 0)              \
    X(FILL_LEVEL, "light.fillLevel", 0.72f, 0.0f, 4.0f, "gNewDawn.HDR.AmbientFill", 0)             \
    X(SHADOW_STRENGTH, "light.shadowStrength", 0.85f, 0.0f, 1.0f, "gNewDawn.Shadows.Strength", 0)  \
    X(SHADOW_FLOOR, "light.shadowFloor", 0.30f, 0.0f, 1.0f, "gNewDawn.Shadows.Floor", 0)           \
    X(DARKEN, "light.darken", 0.0f, 0.0f, 1.0f, NULL, 0)                                           \
    X(LAMP_GAIN, "practicals.lampGain", 1.0f, 0.0f, 4.0f, NULL, 0)                                 \
    X(EXPOSURE, "exposure.ev", 0.05f, -4.0f, 4.0f, "gNewDawn.HDR.Exposure", 1)                     \
    X(CONTRAST, "tone.contrast", 1.08f, 0.5f, 1.5f, "gNewDawn.PostFx.Grade.Contrast", 0)           \
    X(SATURATION, "tone.saturation", 1.12f, 0.0f, 2.0f, "gNewDawn.PostFx.Grade.Saturation", 0)     \
    X(GRADE_GAIN, "grade.gain", 1.0f, 0.0f, 2.0f, NULL, 0)                                         \
    X(GRADE_WARMTH, "grade.warmth", 0.0f, -2.0f, 2.0f, NULL, 0)                                    \
    X(GRADE_TINT, "grade.tint", 0.0f, -2.0f, 2.0f, NULL, 0)                                        \
    X(GRADE_LIFT, "grade.lift", 0.0f, -0.2f, 0.2f, NULL, 0)                                        \
    X(GRADE_GAMMA, "grade.gamma", 1.0f, 0.5f, 2.0f, NULL, 0)                                       \
    X(GRADE_CONTRAST, "grade.contrast", 0.0f, -0.5f, 0.5f, NULL, 0)                                \
    X(GRADE_SAT, "grade.saturation", 1.0f, 0.0f, 2.0f, NULL, 0)                                    \
    X(GRADE_SHADOW_R, "grade.shadowR", 0.0f, -0.3f, 0.3f, NULL, 0)                                 \
    X(GRADE_SHADOW_G, "grade.shadowG", 0.0f, -0.3f, 0.3f, NULL, 0)                                 \
    X(GRADE_SHADOW_B, "grade.shadowB", 0.0f, -0.3f, 0.3f, NULL, 0)                                 \
    X(GRADE_HIGH_R, "grade.highR", 0.0f, -0.3f, 0.3f, NULL, 0)                                     \
    X(GRADE_HIGH_G, "grade.highG", 0.0f, -0.3f, 0.3f, NULL, 0)                                     \
    X(GRADE_HIGH_B, "grade.highB", 0.0f, -0.3f, 0.3f, NULL, 0)                                     \
    X(GRADE_GREEN_SHIFT, "grade.greenShift", 0.0f, -1.0f, 1.0f, NULL, 0)                           \
    X(GRADE_GREEN_SAT, "grade.greenSaturation", 1.0f, 0.0f, 2.0f, NULL, 0)                         \
    X(FOG_LEVEL, "air.fogLevel", 0.85f, 0.0f, 4.0f, "gNewDawn.HDR.FogIntensity", 0)                \
    X(FOG_DENSITY, "air.fogDensity", 0.0f, 0.0f, 4.0f, NULL, 0)                                    \
    X(FOG_HEIGHT, "air.fogHeight", 700.0f, 50.0f, 5000.0f, NULL, 0)                                \
    X(FOG_BASE, "air.fogBase", 0.0f, -5000.0f, 5000.0f, NULL, 0)                                   \
    X(RAYS, "air.rays", 0.30f, 0.0f, 1.0f, "gNewDawn.SunShafts.Strength", 0)                       \
    X(SKY_LEVEL, "sky.level", 0.85f, 0.0f, 4.0f, "gNewDawn.HDR.SkyIntensity", 0)                   \
    X(SKY_COVERAGE, "sky.coverage", 0.55f, 0.0f, 1.0f, "gNewDawn.Sky.Coverage", 0)                 \
    X(BLOOM_STRENGTH, "bloom.strength", 0.18f, 0.0f, 2.0f, "gNewDawn.PostFx.Bloom.Strength", 0)    \
    X(BLOOM_THRESHOLD, "bloom.threshold", 1.0f, 0.0f, 16.0f, "gNewDawn.PostFx.Bloom.Threshold", 0) \
    X(GI_SUN, "gi.sunGain", 0.5f, 0.0f, 4.0f, "gNewDawn.GI.WorldSunGain", 0)                       \
    X(GI_LAMP, "gi.lampGain", 3.0f, 0.0f, 16.0f, "gNewDawn.GI.WorldLocalGain", 0)                  \
    X(GI_BAKED, "gi.bakedGain", 0.6f, 0.0f, 4.0f, "gNewDawn.GI.WorldBakedGain", 0)

enum {
#define NEWDAWN_LOOK_ENUM(id, name, def, lo, hi, cvar, offset) NEWDAWN_LOOK_##id,
    NEWDAWN_LOOK_FIELDS(NEWDAWN_LOOK_ENUM)
#undef NEWDAWN_LOOK_ENUM
        NEWDAWN_LOOK_COUNT
};

#ifdef __cplusplus
#include <cstddef>

namespace Fast {

struct NewDawnLookField {
    const char* name;
    float def, lo, hi;
    const char* cvar;
    int offset;
};

inline const NewDawnLookField& NewDawnLookFieldInfo(int id) {
    static const NewDawnLookField kFields[NEWDAWN_LOOK_COUNT] = {
#define NEWDAWN_LOOK_ROW(id, name, def, lo, hi, cvar, offset) { name, def, lo, hi, cvar, offset },
        NEWDAWN_LOOK_FIELDS(NEWDAWN_LOOK_ROW)
#undef NEWDAWN_LOOK_ROW
    };
    return kFields[id];
}

struct NewDawnLookFrame {
    bool on = false;
    float v[NEWDAWN_LOOK_COUNT] = {};
};

// The look of the display list being drawn (render thread only); set by the interpreter at every post-fx marker.
inline NewDawnLookFrame& NewDawnLookNow() {
    static NewDawnLookFrame frame;
    return frame;
}

// What a feature should use for field `id` when the player's cvar for it reads `cvarValue`.
inline float NewDawnLookValue(const NewDawnLookFrame& look, int id, float cvarValue) {
    if (!look.on) {
        return cvarValue;
    }
    const NewDawnLookField& f = NewDawnLookFieldInfo(id);
    return f.offset ? look.v[id] + (cvarValue - f.def) : look.v[id] * (cvarValue / f.def);
}

} // namespace Fast
#endif

#endif
