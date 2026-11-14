#pragma once
#include <stdint.h>

/*
 * Parameters for the New Dawn scene post-processing marker (gSPNewDawnPostFx).
 * Plain C so game code can fill it. Allocate it from the per-frame graphics pool: the display list is walked once
 * per interpolated output frame, so the pointer must stay valid until the frame is done.
 */
#define NEWDAWN_POSTFX_STAGE_GROUND_END 13 /* room geometry done, actors not yet drawn: grass takes its ground colour here */
#define NEWDAWN_POSTFX_STAGE_WIND_BEGIN 11 /* render-only foliage sway scope (room geometry) */
#define NEWDAWN_POSTFX_STAGE_WIND_END 12
#define NEWDAWN_POSTFX_STAGE_WEATHER_BEGIN 9 /* native rain draw scope, updates/RNG still run */
#define NEWDAWN_POSTFX_STAGE_WEATHER_END 10
#define NEWDAWN_POSTFX_STAGE_SUN_FLARE_BEGIN 7 /* native sun flare draw scope; environment updates still run */
#define NEWDAWN_POSTFX_STAGE_SUN_FLARE_END 8
#define NEWDAWN_POSTFX_STAGE_SKY_BEGIN 4 /* explicit skybox colour calibration */
#define NEWDAWN_POSTFX_STAGE_SKY_END 5
#define NEWDAWN_POSTFX_STAGE_PHYSICAL_SKY 6 /* after the ordinary skybox, before celestial objects/world */
#define NEWDAWN_POSTFX_STAGE_WORLD_BEGIN 3 /* linear HDR world, before sky */
#define NEWDAWN_POSTFX_STAGE_OPAQUE_END 0   /* ambient occlusion / shadows on the opaque geometry drawn so far */
#define NEWDAWN_POSTFX_STAGE_SCENE_END 1    /* bloom, colour grade and dithering of the finished world */
#define NEWDAWN_POSTFX_STAGE_SHADOW_BEGIN 2 /* start recording opaque shadow casters */

/* One recorded grass ground triangle: world xyz of its three vertices. */
#define NEWDAWN_GRASS_TRI_FLOATS 9

/* One recorded VFX sprite: cls, ndcX, ndcY, ndcZ, viewDepth, halfW, halfH, seed, primRGBA(4), world xyz, unused, envRGBA(4). */
#define NEWDAWN_VFX_FLOATS 20

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
    /* gNewDawn.Sky.Special variants, all read-only: 0 normal, 1 winter overcast, 2 final day, 3 cursed Ikana,
       4 poisoned swamp. skyMood is the final day's 0..1 intensity derived from the game clock. */
    int32_t skyVariant;
    float skyMood;

    float flareSunPos[3]; /* native sun position, world space, read-only */
    float screenShaftStrength; /* HDR radial rays; independent of shadows and volumetric fog */
    float screenShaftColor[3]; /* sun/sky tint, sRGB; decoded by the renderer */

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
    /* Cascaded sun shadows: 1 = the single shadowRadius map, 2..4 = the map is split into that many tiles that cover
       the view out to shadowDistance (view-axis distance in world units). */
    int32_t shadowCascades;
    float shadowDistance;

    /* Screen-space diffuse bounce. Camera/scene are read-only history rejection inputs. */
    int32_t giEnabled, giSceneId;
    float giStrength;
    float giCameraEye[3];

    /* New Dawn sun shafts: raymarched in-scattering through the shadow map. Only used when shadowEnabled is set. */
    int32_t shaftEnabled;
    float shaftStrength;     /* already multiplied by the sun-height and sun-colour fades */
    float shaftFogColor[3];  /* 0..1, the game's fog colour */

    /* Render-only height fog. Normal outdoor scenes only; native distance fog remains authoritative. */
    int32_t fogVolumetric; /* 1: sun-shadowed outdoor fog, 2: local-light interior fog */
    float fogBaseHeight;
    float fogHeight;
    float fogDensityScale;
    float fogAmbient[3];
    float fogColor[3];
#define NEWDAWN_FOG_POINT_LIGHTS 4
    int32_t fogPointCount;
    float fogPointPosRadius[NEWDAWN_FOG_POINT_LIGHTS][4];
    float fogPointColor[NEWDAWN_FOG_POINT_LIGHTS][3];

    /* Render-only ambient particles. Scene and lights are read-only snapshots. */
#define NEWDAWN_ATMOS_LIGHTS 8
    int32_t atmosBiome, atmosScene, atmosParticles, atmosWeather;
    float atmosSnow;
    float atmosNight, atmosRain;
    float atmosEye[3], atmosSunColor[3], atmosSkyColor[3];
    int32_t atmosLightCount;
    float atmosLights[NEWDAWN_ATMOS_LIGHTS][4]; /* position, radius */
    float atmosLightColors[NEWDAWN_ATMOS_LIGHTS][3];

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
    float waterCaustics;          /* 0..1: animated caustics on lit surfaces below (and just above) the water planes */
    float waterCausticDir[3];     /* unit vector towards the light that forms the caustics */
    float waterCausticColor[3];   /* its colour, already scaled by the light's fade */
    float waterUnderwater;        /* 0..1: underwater fog/absorption/god rays/refraction; 0 = off */
    float waterEyeSurfaceY;       /* surface Y of the WaterBox the camera is below, valid when waterUnderwaterState > 0 */
    int32_t waterCausticSun;      /* 1: the caustic light is the sun/moon the shadow map was built for */
    float waterEyeRect[4];        /* XZ rect (minX, minZ, maxX, maxZ) of that WaterBox */
    float waterUnderwaterState;   /* smoothed 0..1: 1 once the camera is below a surface, ramps down after leaving it */
    /* New Dawn art-directed grade, applied in display space after the base grade at the scene-end marker. The game
       computes the final (time-of-day interpolated, scene cross-faded, strength-scaled) look; 0 = skipped entirely. */
    int32_t artEnabled;
    float artSlope[3];      /* per-channel gain (warmth/tint folded in) */
    float artOffset;        /* black lift, added before the power */
    float artPower;         /* gamma exponent (>1 darkens mid-tones) */
    float artContrast;      /* extra S-curve, 0 = none */
    float artSat;           /* overall saturation multiplier */
    float artShadowTint[3]; /* added, weighted towards the shadows */
    float artHighTint[3];   /* added, weighted towards the highlights */
    float artGreen[2];      /* green hue range: shift towards yellow (+) or teal (-), saturation multiplier */
    /* New Dawn VFX replacement: native fire/smoke/spark sprites are recorded by the interpreter (render-side only) and
       redrawn as HD HDR sprites at the scene-end marker. Needs HDR. */
    int32_t vfxEnabled;
    float vfxStrength;  /* HDR emission multiplier */
    float vfxEmbers;    /* ember density 0..2 */
    /* New Dawn local lighting: the game's own point lights (read-only snapshot, nearest first) lit per pixel from the
       depth buffer in HDR at the opaque-end marker. The renderer appends the VFX flames that have no game light. */
#define NEWDAWN_LOCAL_LIGHTS 16
    int32_t localEnabled;
    float localStrength; /* HDR intensity multiplier */
    float localFlicker;  /* 0..1 fire flicker amount */
    int32_t localCount;
    float localLights[NEWDAWN_LOCAL_LIGHTS][4];      /* world position, game radius */
    float localColors[NEWDAWN_LOCAL_LIGHTS][3];      /* the light's own colour, 0..1 */
    float localDarken;       /* 0..~0.8: how far the unlit scene is pulled down (night outdoors, lamp-lit interiors) */
    float localGain;         /* light intensity multiplier (interiors are lit harder) */
    /* New Dawn grass: render-only instanced tufts on grass-textured ground, drawn at the opaque-end marker. Needs HDR. */
    int32_t grassEnabled;
    float grassDensity;      /* 0..2 multiplier on the tuft count */
    float grassDistance;     /* world units; blades fade out and disappear here */
    float grassHeight;       /* blade height multiplier */
    float grassFocus[3];     /* the player's position: blades close to it are laid low (read-only snapshot) */
    float grassSunDir[3];    /* unit vector towards the sun; all zero when there is none */
    float grassSunColor[3];  /* sRGB 0..1 */
} NewDawnPostFxParams;
