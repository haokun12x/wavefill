// How many clusters of S whole-SM CTAs (96 KiB dynamic smem) the device holds at once.
#include <cuda_runtime.h>
#include <cstdio>
__global__ void k(int* p) { extern __shared__ int s[]; if (p) p[0] = s[0]; }
int main() {
    const int smem = 96 * 1024;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    for (int s = 1; s <= 16; ++s) {
        cudaLaunchAttribute a{};
        a.id = cudaLaunchAttributeClusterDimension;
        a.val.clusterDim.x = s; a.val.clusterDim.y = 1; a.val.clusterDim.z = 1;
        cudaLaunchConfig_t c{};
        c.gridDim = dim3(s); c.blockDim = dim3(128); c.dynamicSmemBytes = smem; c.attrs = &a; c.numAttrs = 1;
        int n = 0;
        cudaError_t e = cudaOccupancyMaxActiveClusters(&n, k, &c);
        std::printf("S=%2d clusters=%3d CTAs=%3d idle_SMs=%3d %s\n", s, n, n * s, sms - n * s,
                    e == cudaSuccess ? "" : cudaGetErrorString(e));
        cudaGetLastError();
    }
}
