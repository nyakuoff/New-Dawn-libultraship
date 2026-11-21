#pragma once
// New Dawn neural rendering (gNewDawn.Neural.*): what the menu and the performance overlay show about it.
// Call from the render (GL) thread.

struct NewDawnNeuralInfo {
    int state;           // 0 off, 1 working (loading, extracting), 2 running, 3 unavailable or failed
    char status[192];    // one line for the menu
    int width, height;   // the image the network sees; 0 when not running
    float runMs;         // GPU time of one network run (Vulkan), smoothed
    float runsPerSecond; // how often a new result lands
};

namespace Fast {
void NewDawnNeuralGet(NewDawnNeuralInfo* out);
} // namespace Fast
