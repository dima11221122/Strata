// Diagnostic-only, paired draft/target states for offline final-mixer training.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace strata::core {

class DraftAlignmentCapture {
public:
    ~DraftAlignmentCapture();
    bool init(const std::string& dir, int64_t hidden, int64_t streams, int steps, std::string& err);
    uint64_t vram_bytes() const { return bytes_; }
    void begin_request();
    bool snapshot(int step, const float* residual, const float* head_input, cudaStream_t stream, std::string& err);
    void remember(int64_t next_position, int count, const int32_t* proposals, const float* probabilities);
    bool observe(int T, int64_t position, const int32_t* window, const int32_t* targets,
                 const float* teacher_head, bool usable, int accepted, std::string& err);

private:
    int64_t hidden_ = 0, residual_ = 0, position_ = -1;
    int steps_ = 0, pending_ = 0;
    uint64_t bytes_ = 0, request_ = 0, rows_ = 0;
    float* device_ = nullptr;
    std::vector<float> states_, draft_heads_, teacher_heads_, probabilities_;
    std::vector<int32_t> proposals_;
    std::ofstream residual_file_, draft_file_, teacher_file_, target_file_, metadata_;
};

} // namespace strata::core
