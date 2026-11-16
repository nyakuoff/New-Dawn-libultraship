#pragma once
// New Dawn: render-side wind shared by clouds, particles, foliage and precipitation.
// Wall-clock time only (never game time, RNG or state), so it cannot influence gameplay.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>

namespace Fast {
struct NewDawnWindState {
    float dirX, dirZ; // unit horizontal direction (matches the 2:1 cloud wind direction)
    float gust;       // 0.3 .. 1.6, slow swells with faster flutter
};

// gNewDawn.Debug.FreezeTime (test fixtures only): the interpreter mirrors the cvar here once per frame. While set, the
// shared render clock stands still at 0 and every render-frame counter (TAA jitter, GI/light smoothing steps) is held,
// so a frozen game state renders the same pixels on every run.
inline std::atomic<bool>& NewDawnTimeFrozen() {
    static std::atomic<bool> frozen{ false };
    return frozen;
}

// Frozen-frame fixture (gNewDawn.Debug.FreezeFrame, mm/2s2h/DeveloperTools): the game sets Landed once it has stopped
// the game logic. The interpreter then counts the displayed frames since (Tick, 0 = not landed), bumps Epoch once so
// every temporal accumulator (TAA, screen-space and world GI) restarts from nothing, and after kFreezeSettleFrames
// frames those accumulators stop writing their history (Hold). Exactly kFreezeSettleFrames identical frames from an
// empty history is what makes a frozen capture the same pixels on every run, whatever the frame rate or wall time.
constexpr uint32_t kFreezeSettleFrames = 240;
inline std::atomic<bool>& NewDawnFreezeLanded() {
    static std::atomic<bool> landed{ false };
    return landed;
}
inline std::atomic<uint32_t>& NewDawnFreezeTick() {
    static std::atomic<uint32_t> tick{ 0 };
    return tick;
}
inline std::atomic<uint32_t>& NewDawnFreezeEpoch() {
    static std::atomic<uint32_t> epoch{ 0 };
    return epoch;
}
inline bool NewDawnFreezeHold() {
    return NewDawnFreezeTick().load(std::memory_order_relaxed) > kFreezeSettleFrames;
}

// The one wall-clock source of every render-side animation (wind, clouds, flicker, VFX, sky, grass sway, weather).
inline double NewDawnClockSeconds() {
    if (NewDawnTimeFrozen().load(std::memory_order_relaxed)) {
        return 0.0;
    }
    static const auto epoch = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
}

inline double NewDawnWindSeconds() {
    return NewDawnClockSeconds();
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
