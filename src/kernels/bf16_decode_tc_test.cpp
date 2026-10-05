#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

template <class F> static void captured(F&& run, cudaStream_t stream) {
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    run();
    ck(cudaStreamEndCapture(stream, &graph));
    ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    ck(cudaGraphLaunch(executable, stream));
    ck(cudaStreamSynchronize(stream));
    ck(cudaGraphExecDestroy(executable));
    ck(cudaGraphDestroy(graph));
}

// A literal distinguishes supported BF16 arithmetic from a valid lower-PTX FP32 fallback.
// 1 + 1/256 is exactly representable in FP32 and rounds to 1 in BF16 (ties to even).
static bool uses_bf16_arithmetic() {
    std::vector<uint16_t> w(16 * 1024, 0x3f80);
    std::vector<float> x(32, 0.f), y(2 * 1024, -999.f);
    x[0] = x[16] = 1.00390625f;
    uint16_t* dw = nullptr;
    float *dx = nullptr, *dy = nullptr;
    ck(cudaMalloc((void**) &dw, w.size() * 2));
    ck(cudaMalloc((void**) &dx, x.size() * 4));
    ck(cudaMalloc((void**) &dy, y.size() * 4));
    ck(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice));
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    captured([&] { strata::kernels::bf16_gemv_fp32_mmvf_multi(dx, 16, dw, dy, 1024, 16, 1024, 2, stream); }, stream);
    ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    ck(cudaStreamDestroy(stream));
    ck(cudaFree(dw)); ck(cudaFree(dx)); ck(cudaFree(dy));
    const bool bf16 = std::all_of(y.begin(), y.end(), [](float v) { return v == 1.f; });
    const bool fp32 = std::all_of(y.begin(), y.end(), [](float v) { return v == 1.00390625f; });
    if (!bf16 && !fp32) throw std::runtime_error("BF16 dispatch literal probe produced invalid outputs");
    return bf16;
}

int main(int argc, char** argv) {
    try {
        bool rounded = argc > 1 && std::string(argv[1]) == "--bf16";
        int device = 0, major = 0;
        ck(cudaGetDevice(&device));
        ck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        if (rounded && major < 8) return 3;
        if (argc > 1 && std::string(argv[1]) == "--ptx75" && major < 8) return 3;
        if (rounded) rounded = uses_bf16_arithmetic();
        int failures = 0;
        std::mt19937 random(41);
        std::uniform_real_distribution<float> values(-1.f, 1.f);
        for (const int ni : {16, 2560, 10240}) {
            for (const int no : {1, 48, 512, 4096, 10240}) {
                if (ni == 10240 && no == 10240) continue; // exceeds spare VRAM beside the loaded model
                const int ldx = ni + 16, ldy = no + 16;
                std::vector<uint16_t> w((size_t) ni * no);
                for (auto& v : w) v = strata::kernels::bf16_from_f32(values(random) * .02f);
                uint16_t* dw = nullptr;
                ck(cudaMalloc((void**) &dw, w.size() * 2));
                ck(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
                for (const int nt : {2, 3, 4, 6, 8}) {
                    std::vector<float> x((size_t) nt * ldx), y((size_t) nt * ldy, -999.f);
                    for (auto& v : x) v = values(random);
                    float *dx = nullptr, *dy = nullptr;
                    ck(cudaMalloc((void**) &dx, x.size() * 4));
                    ck(cudaMalloc((void**) &dy, y.size() * 4));
                    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
                    ck(cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice));
                    cudaStream_t stream;
                    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                    const auto run = [&] {
                        strata::kernels::bf16_gemv_fp32_mmvf_multi(dx, ldx, dw, dy, ldy, ni, no, nt, stream);
                    };
                    captured(run, stream);
                    ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
                    double err = 0, norm = 0;
                    bool padding = true, finite = true;
                    for (int t = 0; t < nt; ++t) {
                        for (int o = 0; o < no; ++o) {
                            double ref = 0;
                            for (int i = 0; i < ni; ++i) {
                                float v = x[(size_t) t * ldx + i];
                                const bool tensorCase = (ni == 16 || ni == 2560) && (no == 4096 || no == 10240);
                                if (rounded && tensorCase) v = strata::kernels::f32_from_bf16(strata::kernels::bf16_from_f32(v));
                                ref += (double) v * strata::kernels::f32_from_bf16(w[(size_t) o * ni + i]);
                            }
                            const float got = y[(size_t) t * ldy + o];
                            finite = finite && std::isfinite(got);
                            err += ((double) got - ref) * ((double) got - ref);
                            norm += ref * ref;
                        }
                        for (int o = no; o < ldy; ++o) padding = padding && y[(size_t) t * ldy + o] == -999.f;
                    }
                    const double rel = std::sqrt(err / std::max(norm, 1e-30));
                    if (!finite || !padding || rel > 3e-6) ++failures;
                    cudaEvent_t a, b;
                    ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
                    for (int warm = 0; warm < 10; ++warm) run();
                    ck(cudaEventRecord(a, stream));
                    for (int repeat = 0; repeat < 100; ++repeat) run();
                    ck(cudaEventRecord(b, stream)); ck(cudaEventSynchronize(b));
                    float ms = 0;
                    ck(cudaEventElapsedTime(&ms, a, b));
                    std::printf("{\"n_in\":%d,\"n_out\":%d,\"tokens\":%d,\"us\":%.3f,\"relativeError\":%.9g,\"padding\":%s,\"finite\":%s}\n",
                                ni, no, nt, ms * 10.f, rel, padding ? "true" : "false", finite ? "true" : "false");
                    ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b));
                    ck(cudaStreamDestroy(stream));
                    ck(cudaFree(dx)); ck(cudaFree(dy));
                }
                ck(cudaFree(dw));
            }
        }
        std::printf("{\"failures\":%d,\"bf16Reference\":%s}\n", failures, rounded ? "true" : "false");
        return failures ? 1 : 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
