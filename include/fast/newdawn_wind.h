#pragma once
// New Dawn: render-side wind shared by clouds, particles, foliage and precipitation.
// Wall-clock time only (never game time, RNG or state), so it cannot influence gameplay.
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
    const float slow = 0.5f + 0.5f * std::sin(t * 0.21);
    const float fast = 0.5f + 0.5f * std::sin(t * 1.9 + 1.3);
    return { 0.894427f, 0.447214f, 0.3f + 1.0f * slow * slow + 0.3f * fast };
}
// Analytic integral of gust: no frame-rate-dependent accumulation or velocity-times-time jumps.
// Derivative is exactly NewDawnWindAt(t).gust; callers choose speed and periodic domain.
inline double NewDawnWindTravel(double t) {
    return 0.825 * t + 0.5 * (1.0 - std::cos(0.21 * t)) / 0.21 -
           0.125 * std::sin(0.42 * t) / 0.42 +
           0.15 * (std::cos(1.3) - std::cos(1.9 * t + 1.3)) / 1.9;
}
} // namespace Fast
