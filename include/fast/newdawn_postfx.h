#pragma once
#include <stdint.h>

/*
 * Parameters for the New Dawn scene post-processing marker (gSPNewDawnPostFx).
 * Plain C so game code can fill it. Allocate it from the per-frame graphics pool: the display list is walked once
 * per interpolated output frame, so the pointer must stay valid until the frame is done.
 */
#define NEWDAWN_POSTFX_STAGE_OPAQUE_END 0 /* ambient occlusion on the opaque geometry drawn so far */
#define NEWDAWN_POSTFX_STAGE_SCENE_END 1  /* bloom, colour grade and dithering of the finished world */

typedef struct NewDawnPostFxParams {
    float zNear;
    float zFar;
    float fovyDegrees;
    /* The fog range the game applies to the world, as passed to Gfx_SetFog (>= 997 means no fog, < 0 full fog). */
    int32_t fogNear;
    int32_t fogFar;
    uint32_t flags; /* reserved, 0 */
} NewDawnPostFxParams;
