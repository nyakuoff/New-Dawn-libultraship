# Neural rendering

The DLSS-NR generative re-rendering network (the 71-block Swin / ViT U-net of `nvngx_dlssnr.dll` 310.8.0) as Vulkan
compute kernels, running as an opt-in New Dawn post-process beside the OpenGL renderer. This directory is the
network, the runtime the renderer drives, and a standalone runner; the renderer's half is `RunNeural` in
`backends/gfx_opengl_postfx.cpp`.

**Nothing here is NVIDIA's**, and nothing NVIDIA-derived may be committed. The weights come from a
`nvngx_dlssnr.dll` the player supplies; `ExtractModel` turns it into a model directory in their cache. See
[THIRD_PARTY.md](THIRD_PARTY.md) for the projects this is built on.

| File | What it is |
| --- | --- |
| `NrNumeric.h` | E4M3 / half conversions and the reference's tensor-core arithmetic, on the CPU |
| `NrLayout.h` | the padded field and its levels, window phases, the layout of every tensor record |
| `NrModel.*` | the extractor (DLL -> model directory) and the model directory reader |
| `NrReference.*` | the whole network on the CPU, in exact FP8 arithmetic or in this path's FP16 semantics (tool only) |
| `NrVk.*` | the Vulkan context; the loader is opened at run time, the game does not link against Vulkan |
| `NrGraph.*` | the network as FP16 cooperative-matrix dispatches: the AMD / generic path |
| `NrRuntime.*` | what the renderer talks to: background loading, the shared image buffers, a run split over frames |
| `shaders/` | the kernels; `embed_spirv.py` embeds them as `NrSpirv.inc`, which is committed |
| `tool/` | `nd-nr`, the standalone runner |

## The standalone runner

```
cmake -S libultraship/src/fast/neural -B build-neural -G Ninja && cmake --build build-neural
build-neural/nd-nr devices                                   # what each GPU can run
build-neural/nd-nr extract <nvngx_dlssnr.dll> <model dir>    # the player's file -> a model directory
build-neural/nd-nr synth <model dir>                         # or a stand-in with random weights, for development
build-neural/nd-nr bench <model dir> 960 540                 # ms per frame and per stage
build-neural/nd-nr verify <model dir> 640 360 --fp8          # GPU against the CPU reference, step by step
build-neural/nd-nr runtime <model dir> 960 540               # the in-game path without a renderer, checked on the CPU
build-neural/nd-nr fixture <model dir> 256 256 <dir>         # the CPU reference as an OpenDLSS-NR parity fixture
build-neural/nd-nr selftest                                  # the arithmetic tricks, exhaustively, on the CPU
```

`verify` is the gate after any kernel change: every step must stay within rounding noise of the FP16 reference
(30 dB and up; a wiring mistake shows as single digits), and `--fp8` says what the FP16 arithmetic costs against the
exact one. The network is chaotic at its output: `sensitivity` shows a one-ulp change of a few inputs moving the head
to about 18 dB, so two correct arithmetics agree on the composed image (40 dB and up), not on the head's bits. After editing a shader run `shaders/embed_spirv.py` (needs `glslangValidator`).

## Requirements of the FP16 path

Vulkan 1.3 with `VK_KHR_cooperative_matrix` offering 16x16x16 half-float matrices with a 32-bit accumulator at
subgroup scope, and a subgroup size of 32. Measured on an RX 7800 XT with Mesa RADV 26.2; `nd-nr devices` names
what a device lacks.

## In the game

Off by default (`gNewDawn.Neural.Enabled`), Linux and OpenGL only for now, and only with the HDR renderer. The menu
entry is Advanced > Experimental > Neural Rendering; the player points it at their own `nvngx_dlssnr.dll` and the
model directory is built in their install's folder (`newdawn-neural-model`, or `gNewDawn.Neural.ModelDir`).

Each run, the renderer draws two small images at the network's resolution (`gNewDawn.Neural.Height`, 540 by
default): the **proxy** (the finished world image as the display would show it, sRGB code values, clip w in alpha)
and the **history** (the proxy with the last result carried onto it). It reads them into two buffers that Vulkan
allocated and OpenGL imported (`GL_EXT_memory_object_fd`), signals a shared semaphore, and the runtime queues the
network: a feature kernel, the 96 graph segments as pre-recorded command buffers, a composite kernel. With
`gNewDawn.Neural.Interleave` = N a run starts every N frames at most and its segments are submitted over those N
frames. Nothing waits: the renderer polls a fence, and uploads the result when it has landed.

What comes back is not an image but a **residual**: the network's output minus the proxy, in display code values,
with the run's clip w. Every frame the apply pass fetches it through the camera of the run (the surface point of
each full-resolution pixel, reprojected; the four residual texels around it, minus those that saw another surface
then), and applies it to the scene-referred image as a per-channel ratio, before bloom and the display transform.
So the picture keeps this frame's full-resolution detail and animation, and the network's change follows the camera
between runs. The HUD is drawn later and never reaches the network. A camera cut drops the residual; the next one
fades in over eight frames. `gNewDawn.Neural.BeforeTaa` moves both passes in front of the TAA resolve.

Controls: `Strength` (0..1), `Style` (0 default, 1 natural, 2 cinematic: the network's own conditioning lane),
`Tone`, `Structure`, `Debug` (1: the residual, 2: where it reached), `VitReuse` / `VitRefresh` (experimental, see
the comment at their use: reusing the global-attention result for several runs makes the temporal loop oscillate).

Measured on the RX 7800 XT at 1080p output, Ultra preset, South Clock Town and Termina Field: 60 fps held with the
network at 540p every 2nd frame (30 results a second), 720p every 2nd or 3rd, and 540p "every frame" (which lands
41 results a second: a run rarely finishes before the next frame asks). OpenGL-side cost 0.2 ms a frame. A still
frame changes by 0.13 8-bit codes from one result to the next.
