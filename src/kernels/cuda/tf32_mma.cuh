#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace strata::kernels::detail {

// Shared by the corrected QSA scorer and the opt-in HC up projection.
__device__ __forceinline__ uint32_t tf32_hi(float x) {
#if !defined(__HIPCC__) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
#else
    return __float_as_uint(x);
#endif
}

__device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
#if !defined(__HIPCC__) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

}  // namespace strata::kernels::detail
