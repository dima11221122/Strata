// Q6_K scheduling regression: independent scalar dequantization and FP64 dots
// over the same Q8_1 bytes, including ragged row tails and captured launches.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

double run_case(int n_in, int n_out, cudaStream_t stream) {
    constexpr int columns = 8, guard = 8;
    constexpr float sentinel = -12345.0f;
    const int blocks = n_in / 256;
    std::mt19937 random(20261005u + unsigned(n_in + n_out));
    std::vector<uint8_t> weights(std::size_t(n_out) * blocks * 210);
    for (auto& value : weights) value = uint8_t(random());
    for (std::size_t offset = 0; offset < weights.size(); offset += 210) {
        const uint16_t scale = uint16_t((random() & 0x8000u) | ((9u + random() % 6u) << 10) | (random() & 1023u));
        weights[offset + 208] = uint8_t(scale);
        weights[offset + 209] = uint8_t(scale >> 8);
    }
    std::vector<float> x(std::size_t(columns) * n_in);
    for (auto& value : x) value = float(int(random() % 2001u) - 1000) / 317.0f;
    const std::size_t qbytes = strata::kernels::native_q8_1_bytes(n_in, columns);
    void *dw = nullptr, *dx = nullptr, *dq = nullptr, *dy = nullptr;
    check(cudaMalloc(&dw, weights.size()));
    check(cudaMalloc(&dx, x.size() * sizeof(float)));
    check(cudaMalloc(&dq, qbytes));
    std::vector<float> output(std::size_t(columns) * n_out + 2 * guard, sentinel);
    check(cudaMalloc(&dy, output.size() * sizeof(float)));
    check(cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    strata::kernels::native_quantize_q8_1(static_cast<const float*>(dx), dq, n_in, columns, stream);
    check(cudaStreamSynchronize(stream));
    std::vector<uint8_t> quant(qbytes);
    check(cudaMemcpy(quant.data(), dq, qbytes, cudaMemcpyDeviceToHost));

    std::vector<double> reference(std::size_t(columns) * n_out, 0.0);
    float row[256];
    for (int r = 0; r < n_out; ++r) {
        for (int kb = 0; kb < blocks; ++kb) {
            strata::dequantize_q6_K(weights.data() + (std::size_t(r) * blocks + kb) * 210, row);
            for (int j = 0; j < columns; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 256; ++k) {
                    const uint8_t* act = quant.data() + (std::size_t(j) * n_in / 32 + kb * 8 + k / 32) * 36;
                    int8_t code;
                    std::memcpy(&code, act + 4 + k % 32, 1);
                    sum += double(row[k]) * strata::fp16_to_fp32(strata::read_u16(act)) * code;
                }
                reference[std::size_t(j) * n_out + r] += sum;
            }
        }
    }
    double worst = 0.0;
    bool valid = true;
    for (int nc = 1; nc <= columns; ++nc) {
        std::fill(output.begin(), output.end(), sentinel);
        check(cudaMemcpy(dy, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice));
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        if (nc % 2) check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        strata::kernels::native_q6_k_mmvq(dw, dq, static_cast<float*>(dy) + guard, n_in, n_out, nc, stream);
        if (nc % 2) {
            check(cudaStreamEndCapture(stream, &graph));
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            check(cudaGraphLaunch(executable, stream));
        }
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(output.data(), dy, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (executable) check(cudaGraphExecDestroy(executable));
        if (graph) check(cudaGraphDestroy(graph));
        double error = 0.0, norm = 0.0;
        for (std::size_t i = 0; i < output.size(); ++i) {
            if (i < guard || i >= std::size_t(guard + nc * n_out)) {
                valid = valid && output[i] == sentinel;
                continue;
            }
            const double ref = reference[i - guard];
            valid = valid && std::isfinite(output[i]);
            error += (double(output[i]) - ref) * (double(output[i]) - ref);
            norm += ref * ref;
        }
        const double relative = std::sqrt(error / (norm + 1e-30));
        valid = valid && relative <= 3e-6 && norm > 0.0;
        worst = std::max(worst, relative);
    }
    cudaFree(dw); cudaFree(dx); cudaFree(dq); cudaFree(dy);
    if (!valid) throw std::runtime_error("Q6_K numerical/tail failure: " + std::to_string(n_in) + " x " + std::to_string(n_out));
    return worst;
}
}

int main() {
    try {
        cudaStream_t stream;
        check(cudaStreamCreate(&stream));
        double worst = 0.0;
        for (int n_in : {256, 512, 2560, 8192})
            for (int n_out : {1, 5, 33}) worst = std::max(worst, run_case(n_in, n_out, stream));
        check(cudaStreamDestroy(stream));
        std::printf("Q6_K: 96 cases, scalar FP64 reference, tails and graph capture; worst relative L2 %.3e\n", worst);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
