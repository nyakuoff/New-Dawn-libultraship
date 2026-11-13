#pragma once
#include <stdint.h>

/*
 * Parameters for the New Dawn scene post-processing marker (gSPNewDawnPostFx).
 * Plain C so game code can fill it. Allocate it from the per-frame graphics pool: the display list is walked once
 * per interpolated output frame, so the pointer must stay valid until the frame is done.
 */
#define NEWDAWN_POSTFX_STAGE_SUN_FLARE_BEGIN 7 /* native sun flare draw scope; environment updates still run */
#define NEWDAWN_POSTFX_STAGE_SUN_FLARE_END 8
#define NEWDAWN_POSTFX_STAGE_SKY_BEGIN 4 /* explicit skybox colour calibration */
#define NEWDAWN_POSTFX_STAGE_SKY_END 5
#define NEWDAWN_POSTFX_STAGE_PHYSICAL_SKY 6 /* after the ordinary skybox, before celestial objects/world */
#define NEWDAWN_POSTFX_STAGE_WORLD_BEGIN 3 /* linear HDR world, before sky */
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

    /* Read-only normal outdoor sky state. Unknown/scripted skies never emit the physical-sky marker. */
    float skySunDir[3];
    float skyFogColor[3];
    float skyPaletteColor[3]; /* current time-of-day sky palette, sRGB; read after UpdateSkybox */
    float skyGroundAlbedo[3]; /* bounded representative ground reflectance, linear */
    int32_t skySceneId;       /* cache invalidation only; never used to change game state */
    float skyWeather; /* 0 clear, 1 overcast */
    float skyOvercast; /* 1: winter Mountain Village's overcast deck (replaces its opaque fog fill) */

    float flareSunPos[3]; /* native sun position, world space, read-only */

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

    /* Render-only height fog. Normal outdoor scenes only; native distance fog remains authoritative. */
    int32_t fogVolumetric;
    float fogBaseHeight;
    float fogHeight;
    float fogDensityScale;

    /* New Dawn water: glints and Fresnel over the scene's water planes, applied at the scene-end marker. */
#define NEWDAWN_MAX_WATER_BOXES 16
    int32_t waterEnabled;
    int32_t waterReflections; /* opt-in opaque screen-space reflections */
    float waterReflectionStrength; /* already multiplied by the per-scene factor */
    int32_t waterDebug;      /* 1: paint planes, 2: reflection hit mask (green hit, blue miss, red rejected) */
    float waterStrength;     /* already multiplied by the per-scene factor and the light fade */
    float waterTime;         /* seconds, game time (the renderer adds the interpolation fraction) */
    int32_t waterBoxCount;
    float waterBoxes[NEWDAWN_MAX_WATER_BOXES][5]; /* minX, minZ, maxX, maxZ, surface Y */
    float waterLightDir[3];   /* unit vector towards the sun (or moon at night) */
    float waterLightColor[3]; /* glint colour, already scaled by the light's fade */
    float waterSkyColor[3];   /* what the water reflects at grazing angles (the game's fog colour) */
    float waterCameraEye[3];
    float waterFoam;       /* shoreline/contact foam strength, already multiplied by the per-scene factor */
    float waterRefraction; /* refraction + depth absorption strength, already multiplied by the per-scene factor */
    float waterSurfaceVisibility; /* transmission of the native full-screen sun glare */
    float waterRestyle;           /* 0..1: replace the painted water colour with the absorption/scattering model */
    float waterScatter[3];        /* linear in-scatter colour of deep water (lit by the scene's light) */
    float waterAbsorb[3];         /* per-channel absorption coefficient, 1/world units */
} NewDawnPostFxParams;
