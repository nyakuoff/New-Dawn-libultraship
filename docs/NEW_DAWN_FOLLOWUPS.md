# Fog shafts and water specks investigation — 2026-11-14

Base renderer: `e4aba70e` (`origin/new-dawn`), game: `f7f8b1f1c` (`origin/main`).
Branch in both repositories: `fix/fog-shafts-water-specks`. **Neither requested fix is complete.**
All experimental renderer changes were rejected and removed after the three-attempt limit.
This commit contains documentation only; there is no rendering or gameplay change.

## Observations and hypotheses

`src/fast/backends/gfx_opengl_postfx.cpp`, `kFogMarchShader` and `kFogApplyShader`:

- Both programs skip sky-depth pixels. Consequently the existing pass cannot show local sunlight
  through canopy gaps backed by sky. This is a confirmed missing rendering path, not proof that
  adding that path alone produces shafts.
- The march extends to 6,000 world units; the default shadow map has a radius of 1,500 about the
  player. Out-of-map samples receive full sunlight. This can dilute local occlusion with broad
  scattering. It was not established as the sole cause of the missing shafts.
- The inspected dusk Field view includes conifers but also Clock Town behind them. Terrain/building
  occlusion and insufficient nearby canopy remain unresolved. The shadow-map diagnostic shows
  substantial solid caster coverage; it does not prove that the relevant branches cast correctly.

`traceReflection`, `RunWater`, and the opaque-end capture in the same source file:

- SSR reads colour and depth captured before translucent world/HUD draws. HUD contamination is
  therefore not supported by the source trace.
- Sky depth, below-water-plane hits and excessive depth mismatch already have rejection tests.
  There is no runtime evidence here establishing invalid hits, NaNs or a filter defect.
- The reflection composite attenuates glints by reflection weight. This remains a hypothesis for
  dark marks, not a demonstrated defect. No changes to it or to the sharp filter were retained.

## Attempts and results

Same RX 7800 XT / Mesa OpenGL machine, 1920×1080, MSAA 4, preset/full local mods.
Build: `cmake --build build -j6`. Captures run sequentially within this task through:

```sh
python3 tools/newdawn/capture.py --bin-dir build/mm --preset \
  --mods /home/loganp/dev/ghm-mods/sets/full ...
```

A private companion pins time/camera only in disposable capture processes. Scratch, logs, fixtures,
rejected patches and full images: `/home/loganp/dev/ghm-mods/work/followups/`.

| Fog view/candidate | Stage 9 GPU ms | Result |
| --- | ---: | --- |
| Main, Field 17:30 | 0.286 | Haze, no distinct shafts |
| 1: sky sunlight, HG g=0.80, 48 steps | 0.995 | No distinct shafts; exceeds 0.8 ms total |
| 2: separate local sun integration, g=0.80, 32 steps | 0.500 | Within cost limit; no distinct shafts |
| 3: candidate 2 with SunIntensity=8 | 0.500 | No distinct shafts; stopped |

Values are means of the available 300-run stage reports, including early samples. Other workers
were using this shared GPU, so these are indicative timings, not an isolated performance benchmark.
The main fixed Coast water view measured stage 4 at 0.418 ms; no water candidate was implemented.

Water reproduction also stopped after three attempts: the ordinary third-person Coast view and
the fixed view using the old recorded eye/direction did not show the reference's isolated specks;
the legacy keyboard look recipe faced inland. A later SSR-off control also ended facing inland,
despite an earlier correct camera memory sample, and was discarded. **No water before/after proof.**

## Visual checks and limits

All captured images were inspected, including the rejected/invalid views. Matched Field split and
1:1 crops are saved in `/home/loganp/dev/ghm-mods/showcase/followups/`, explicitly labelled **rejected**.
No obvious march bands appear in the inspected crops with the preset's dither enabled.

The original keyboard turning recipe did not reliably aim the camera. A replacement test-only
camera companion sets the camera status to `CAM_STATUS_CUT`, writes verified camera/view vectors,
and rotates the view at 12 degrees/second. `turn-fixed/` contains eight visibly different angles
0.3 seconds apart and `camera.jsonl`; all eight 1:1 crops were inspected. No obvious fog flashes
were seen in those samples. This coarse burst does not prove high-frame-rate flicker safety.
An initial version wrote the wrong view-up offset and produced black world frames; `turn-candidate/`
is invalid and excluded. No camera/test-process writes are part of the shipped renderer.

Frozen Stock Pot Inn, noon, frame 30, `gMatchRefreshRate=0`, `gInterpolationFPS=20`, fog/flare and
Foam/Refraction/Restyle off: candidate 2 and restored exact main are PNG byte-identical in two
repeats each, both HDR and LDR. HDR SHA-256:
`46ad33f92c94c188f017961d297107a7295ca957e83e1ce39ec54ea04103329d`;
LDR: `f8599206cd7f8a60f248349b1f062f905e823aadba73ee30a6344eea1893b391`.
The restored executable also matches the saved original main executable's SHA-256.
No GL/shader/framebuffer errors were found in the retained capture logs with `MESA_DEBUG=context`.
Neither copy of `default.shader.glsl` was edited. An initial main proof launch overlapped the tail
of the restore build and failed; it was excluded and the completed-build proof rerun.

The requested canopy result, clean-away comparison, exact cause of SSR specks, and a verified
water fix remain open. A future investigation needs a genuinely sunlit canopy viewpoint and a
repeatable current-main water-speck frame before another shader edit. Do not count the rejected
split as a visual improvement.
