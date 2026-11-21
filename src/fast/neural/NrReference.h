// The DLSS-NR network on the CPU, as a specification: every publication point of the 71-block graph written out
// in order. Two arithmetics over one data flow:
//
//   Fp8   the reference's own: E4M3 x E4M3 products summed in groups of 16 at 13 truncated fractional bits with a
//         half accumulator (the tensor cores' m16n8k32), in K order, with the split-K partitions of the ViT.
//   Fp16  what the FP16 kernels compute: the same publications, f32 accumulation over the whole K.
//
// Fp8 is what the FP16 path is measured against (dB); Fp16 is what the GPU kernels are checked against (bugs).
// It is slow by design (minutes per megapixel in Fp8) and used by the standalone tool only.
//
// The data flow, the layouts and the publication order follow OpenDLSS-NR (MIT, see THIRD_PARTY.md): nr_graph.cpp,
// reference.cpp, the GLSL reference kernels and docs/. Tensors here are [tokens][channels] floats in natural
// (row-major) token order, each value an exact half or an E4M3 grid value.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "NrLayout.h"
#include "NrModel.h"

namespace Fast::Neural {

enum class Arithmetic { Fp8, Fp16 };

class Reference {
  public:
    Reference(const Model& model, const Geometry& geometry, Arithmetic arithmetic);

    // `features`: [field][16] f32. Returns the head, [field][4]. When `boundaries` is given it receives the E4M3
    // state after every block ("block-N", N = 0..69) and every encoder transition ("transition-N-M"), as the
    // reference's fixtures name them. `lastBlock` < 70 stops early (the head is then empty).
    std::vector<float> Run(const std::vector<float>& features,
                           std::map<std::string, std::vector<float>>* boundaries = nullptr, int lastBlock = 70) const;

    // Input features for a proxy image (RGBA f32 code values, `width` x `height` = the valid size), as the
    // reference's preprocess: noise lanes, the centred proxy twice (no history), the conditioning lanes.
    struct Conditioning { // the defaults are the reference demo's
        uint32_t mSeed = 0;
        float mStyle = 0.0f;
        float mLocalTone = 1.0f;
        float mLocalStructure = 1.0f;
        float mSkinStructure = -1.0f;
        bool mAutoMask = true;
    };
    static std::vector<float> Features(const Geometry& geometry, const std::vector<float>& proxyRgba,
                                       const Conditioning& conditioning);

    // With boundaries: also keep block 0's intermediates ("transition-b0-adapter", "-hidden", "-ffn", "-attended"),
    // to find which kernel a disagreement with another implementation starts in.
    bool mCaptureBlock0 = false;

  private:
    struct Impl;
    const Model& mModel;
    Geometry mGeometry;
    Arithmetic mArithmetic;
};

} // namespace Fast::Neural
