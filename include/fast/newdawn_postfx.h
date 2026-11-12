#pragma once
#include <stdint.h>

/*
 * Parameters for the New Dawn scene post-processing marker (gSPNewDawnPostFx).
 * Plain C so game code can fill it. Allocate it from the per-frame graphics pool: the display list is walked once
 * per interpolated output frame, so the pointer must stay valid until the frame is done.
 */
#define NEWDAWN_POSTFX_STAGE_OPAQUE_END 0   /* ambient occlusion / shadows on the opaque geometry drawn so far */
#define NEWDAWN_POSTFX_STAGE_SCENE_END 1    /* bloom, colour grade and dithering of the finished world */
#define NEWDAWN_POSTFX_STAGE_SHADOW_BEGIN 2 /* start recording opaque shadow casters */

typedef struct NewDawnPostFxParams {
    float zNear;
    float zFar;
    float fovyDegrees;
    /* The fog range the game applies to the world, as passed to Gfx_SetFog (>= 997 means no fog, < 0 full fog). */
    int32_t fogNear;
    int32_t fogFar;
    uint32_t flags; /* reserved, 0 */

    /* New Dawn real-time shadows. */
    int32_t shadowEnabled;
    float shadowStrength;
    float shadowRadius;
    int32_t shadowMapSize;
    float shadowSunDir[3];   /* unit vector pointing towards the sun, world space */
    float shadowAmbient[3];  /* 0..1 */
    float shadowDirColor[3]; /* 0..1 */
    float shadowFocus[3];
    float shadowCameraEye[3];
    /* Filled by the interpreter at the opaque-end marker: exact P and its inverse (row-major). */
    int32_t shadowUseExactProj;
    float shadowNdcXScale;
    float shadowFlipY;
    float shadowViewProj[16];
    float shadowInvViewProj[16];

    /* New Dawn sun shafts: raymarched in-scattering through the shadow map. Only used when shadowEnabled is set. */
    int32_t shaftEnabled;
    float shaftStrength;     /* already multiplied by the sun-height and sun-colour fades */
    float shaftFogColor[3];  /* 0..1, the game's fog colour */

    /* New Dawn water: glints and Fresnel over the scene's water planes, applied at the scene-end marker. */
#define NEWDAWN_MAX_WATER_BOXES 16
    int32_t waterEnabled;
    int32_t waterReflections; /* opt-in opaque screen-space reflections */
    float waterReflectionStrength; /* already multiplied by the per-scene factor */
    int32_t waterDebug;      /* 1: paint the planes */
    float waterStrength;     /* already multiplied by the per-scene factor and the light fade */
    float waterTime;         /* seconds, game time (the renderer adds the interpolation fraction) */
    int32_t waterBoxCount;
    float waterBoxes[NEWDAWN_MAX_WATER_BOXES][5]; /* minX, minZ, maxX, maxZ, surface Y */
    float waterLightDir[3];   /* unit vector towards the sun (or moon at night) */
    float waterLightColor[3]; /* glint colour, already scaled by the light's fade */
    float waterSkyColor[3];   /* what the water reflects at grazing angles (the game's fog colour) */
    float waterCameraEye[3];
} NewDawnPostFxParams;
