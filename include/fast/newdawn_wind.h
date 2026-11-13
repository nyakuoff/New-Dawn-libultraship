#pragma once
// New Dawn: one render-side wind shared by foliage sway and rain. Driven by wall-clock time only (never game time,
// RNG or state), so it cannot influence gameplay.
#include <chrono>
#include <cmath>

namespace Fast {
struct NewDawnWindState {
    float dirX, dirZ; // unit horizontal direction (matches the 2:1 cloud wind direction)
    float gust;       // 0.3 .. 1.6, slow swells with faster flutter
};

inline double NewDawnWindSeconds() {
    static const auto epoch = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
}

inline NewDawnWindState NewDawnWindAt(double t) {
    const float slow = 0.5f + 0.5f * std::sin(float(std::fmod(t, 628.318)) * 0.21f);
    const float fast = 0.5f + 0.5f * std::sin(float(std::fmod(t, 62.8318)) * 1.9f + 1.3f);
    return { 0.894427f, 0.447214f, 0.3f + 1.0f * slow * slow + 0.3f * fast };
}
} // namespace Fast
