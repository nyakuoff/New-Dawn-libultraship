#pragma once
// Read-only GPU timings for the New Dawn performance overlay. Call NewDawnPerfGet from the render (GL) thread.
#define NEWDAWN_PERF_STAGES 26

struct NewDawnPerfInfo {
    float gpuFrameMs;                    // whole-frame GPU time (timestamps), 0 until the first result
    float stageMs[NEWDAWN_PERF_STAGES];  // smoothed ms per post-FX stage; 0 for stages that did not run
    int vramTotalMB;                     // -1 when the driver does not report it
    int vramFreeMB;                      // -1 when the driver does not report it
};

namespace Fast {
// Timers only run while the cvar gNewDawn.Perf.Overlay is 1 (or gNewDawn.PostFx.Debug.Stats is).
void NewDawnPerfGet(NewDawnPerfInfo* out);
const char* NewDawnPerfStageName(int stage);
} // namespace Fast
