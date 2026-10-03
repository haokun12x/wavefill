// Modified by Wavefill (2026): rolled K-phase loop for K=17408. See docs/wavefill/.
#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"

#include <type_traits>

namespace ninfer::ops::detail {
namespace {

// K=17408 is 34 K-phases. Fully unrolled that is a ~100 KB code image which every launch of a
// decode step fetches cold, so the long shape rolls the loop by 4 and reads its scales per lane.
template <class Geometry>
using DecodeSchedule = std::conditional_t<
    Geometry::kInputRows == 17408,
    Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 2, 4>,
    Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>;

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule        = DecodeSchedule<Geometry>;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    auto* output          = static_cast<__nv_bfloat16*>(residual.data);
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse,
        Nvfp4AddResidualEpilogue{output, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{output, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream) {
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N5120K6144:
        launch<Nvfp4N5120K6144>(x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch<Nvfp4N5120K17408>(x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
