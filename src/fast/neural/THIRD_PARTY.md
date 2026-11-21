# Third-party work in the neural rendering module

Nothing in this directory is NVIDIA's. It contains no NVIDIA software, weights, headers or files, and builds none into
the game: the network's weights come from the `nvngx_dlssnr.dll` a player supplies themselves, are read on their
machine, and stay in their local cache. "DLSS" is a trademark of NVIDIA Corporation, used here only to say what the
network implemented by this code is compatible with. New Dawn is not affiliated with, endorsed by or supported by
NVIDIA, and the player is responsible for the licence of the file they supply.

## OpenDLSS-NR (MIT)

<https://github.com/maanHimself/OpenDLSS-NR>, revision `9d08f41`. A Vulkan reimplementation of the DLSS-NR network
and the documentation of its graph, numerics and weight layouts.

Used here:

- `NrNumeric.h`: the E4M3 / half conversions and the tensor-core dot products (`src/numeric.h`, `src/reference.cpp`);
- `NrSha256.h`: `src/sha256.h`;
- `NrLayout.h`, `NrModel.cpp`: the padded field, the window phases, the tensor record layouts and the weight
  fragment indexing (`src/nr_graph.cpp`, `src/nr_model.cpp`, `docs/network.md`, `docs/weights.md`);
- `NrReference.cpp`, `NrGraph.cpp`: the data flow of the 71-block graph and every publication rule
  (`src/nr_graph.cpp`, the GLSL reference kernels in `shaders/`, `docs/numerics.md`);
- `shaders/`: the kernels are written for this path (half-float cooperative matrices on RADV), after the structure
  and the arithmetic of `shaders/common.glsl`, `fused_block32.comp`, `attention_common.glsl`, `global_normalize.comp`,
  `global_attend.comp` and `ops.comp`.

    MIT License

    Copyright (c) 2026 maan


    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## lmxxf/dlss5-on-amd-9070xt-porting (MIT)

<https://github.com/lmxxf/dlss5-on-amd-9070xt-porting>, revision `6035309`. A port of the network to AMD RDNA 4.

Used here: the framing of the weight archive inside the DLL (`Development/parse_weights_archive.py`), which
`NrModel.cpp`'s `ExtractModel` follows. No code, binary or data file of that repository is included.

    MIT License

    Copyright (c) 2026 Kien


    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## Fetched at build time, linked into the game

- **volk** (MIT, Copyright (c) 2018-2026 Arseny Kapoulkine), tag `vulkan-sdk-1.4.363.0`: loads the Vulkan loader at
  run time. <https://github.com/zeux/volk>
- **Vulkan-Headers** (Apache-2.0 OR MIT, The Khronos Group Inc.), tag `vulkan-sdk-1.4.363.0`: headers only.
  <https://github.com/KhronosGroup/Vulkan-Headers>
