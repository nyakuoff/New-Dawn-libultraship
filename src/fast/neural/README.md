# Neural rendering

The DLSS-NR generative re-rendering network (the 71-block Swin / ViT U-net of `nvngx_dlssnr.dll` 310.8.0) as Vulkan
compute kernels, to run as a New Dawn post-process beside the OpenGL renderer. This directory is the network and
its standalone runner; the in-game pass is not wired up yet.

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
