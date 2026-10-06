#include "strata/core/draft_alignment_capture.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>

namespace strata::core {

DraftAlignmentCapture::~DraftAlignmentCapture() {
    if (device_) cudaFree(device_);
}

bool DraftAlignmentCapture::init(const std::string& dir, int64_t hidden, int64_t streams,
                                 int steps, std::string& err) {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(__HIPCC__) || defined(_WIN32)
    err = "draft alignment: diagnostic requires CUDA and POSIX owner permissions";
    return false;
#endif
    // This experiment is scoped to the retained single-device Qwen MTP geometry.
    if (hidden != 2560 || streams != 4 || steps < 1 || steps > 7) {
        err = "draft alignment: unsupported geometry";
        return false;
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path root(dir);
    if (!root.is_absolute() || fs::exists(root, ec) || ec || !fs::create_directories(root, ec) || ec) {
        err = "draft alignment: output must be a new absolute directory";
        return false;
    }
    fs::permissions(root, fs::perms::owner_all, fs::perm_options::replace, ec);
    if (ec) { err = "draft alignment: directory permissions"; return false; }
    auto open = [&](std::ofstream& file, const char* name) {
        const fs::path path = root / name;
        file.open(path, std::ios::binary | std::ios::out);
        if (!file) return false;
        fs::permissions(path, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);
        return !ec;
    };
    if (!open(residual_file_, "states.f32") || !open(draft_file_, "draft_heads.f32") ||
        !open(teacher_file_, "teacher_heads.f32") || !open(target_file_, "targets.i32") ||
        !open(metadata_, "rows.jsonl")) {
        err = "draft alignment: opening outputs failed";
        return false;
    }
    std::ofstream manifest;
    if (!open(manifest, "format.json")) { err = "draft alignment: opening format failed"; return false; }
    manifest << "{\"version\":1,\"byte_order\":\"little\",\"hidden\":" << hidden
             << ",\"streams\":" << streams << ",\"max_steps\":" << steps
             << ",\"pairing\":\"next verifier window, including rejected suffix\"}\n";
    manifest.flush();
    if (!manifest) { err = "draft alignment: format write failed"; return false; }
    hidden_ = hidden;
    residual_ = hidden * streams;
    steps_ = steps;
    bytes_ = uint64_t(steps) * uint64_t(residual_ + hidden_) * sizeof(float);
    if (cudaMalloc(&device_, bytes_) != cudaSuccess) {
        err = "draft alignment: device snapshots do not fit";
        return false;
    }
    states_.resize(size_t(steps_) * size_t(residual_));
    draft_heads_.resize(size_t(steps_) * size_t(hidden_));
    teacher_heads_.resize(size_t(steps_) * size_t(hidden_));
    proposals_.resize(size_t(steps_));
    probabilities_.resize(size_t(steps_));
    return true;
}

void DraftAlignmentCapture::begin_request() {
    ++request_;
    pending_ = 0;
    position_ = -1;
}

bool DraftAlignmentCapture::snapshot(int step, const float* residual, const float* head_input,
                                     cudaStream_t stream, std::string& err) {
    if (step < 0 || step >= steps_) { err = "draft alignment: snapshot index"; return false; }
    float* head = device_ + size_t(steps_) * size_t(residual_) + size_t(step) * size_t(hidden_);
    if (cudaMemcpyAsync(device_ + size_t(step) * size_t(residual_), residual,
                        size_t(residual_) * sizeof(float), cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(head, head_input, size_t(hidden_) * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        err = "draft alignment: recording snapshots failed";
        return false;
    }
    return true;
}

void DraftAlignmentCapture::remember(int64_t next_position, int count, const int32_t* proposals,
                                     const float* probabilities) {
    position_ = next_position;
    pending_ = count;
    if (count < 0 || count > steps_) { pending_ = 0; return; }
    std::copy_n(proposals, count, proposals_.begin());
    std::copy_n(probabilities, count, probabilities_.begin());
}

bool DraftAlignmentCapture::observe(int T, int64_t position, const int32_t* window, const int32_t* targets,
                                    const float* teacher_head, bool usable, int accepted, std::string& err) {
    const int fed = T - 1;
    const int pending = pending_;
    // Row T-1 also gives a valid label for a computed but unoffered proposal,
    // when one exists. This includes the confidence-stopped proposal at T=1.
    const int count = std::min(T, pending);
    pending_ = 0; // Every observation consumes this draft, including skipped lookup/sampled windows.
    if (!usable || count <= 0 || pending == 0) return true;
    if (T < 1 || position != position_ || fed > pending || accepted < 0 || accepted > fed || !teacher_head ||
        !std::equal(window + 1, window + T, proposals_.begin())) {
        err = "draft alignment: next-window pairing mismatch";
        return false;
    }
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(__HIPCC__)
    cudaPointerAttributes attr{};
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess || cudaPointerGetAttributes(&attr, teacher_head) != cudaSuccess ||
        attr.device != current) {
        err = "draft alignment: teacher must be on the draft device";
        return false;
    }
#endif
    if (cudaMemcpy(states_.data(), device_, size_t(count) * size_t(residual_) * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(draft_heads_.data(), device_ + size_t(steps_) * size_t(residual_),
                   size_t(count) * size_t(hidden_) * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(teacher_heads_.data(), teacher_head, size_t(count) * size_t(hidden_) * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "draft alignment: reading paired states failed";
        return false;
    }
    auto finite = [](const std::vector<float>& values, size_t n) {
        return std::all_of(values.begin(), values.begin() + n, [](float x) { return std::isfinite(x); });
    };
    if (!finite(states_, size_t(count) * size_t(residual_)) ||
        !finite(draft_heads_, size_t(count) * size_t(hidden_)) ||
        !finite(teacher_heads_, size_t(count) * size_t(hidden_)) || !finite(probabilities_, size_t(count))) {
        err = "draft alignment: non-finite paired data";
        return false;
    }
    residual_file_.write(reinterpret_cast<const char*>(states_.data()), std::streamsize(count) * residual_ * sizeof(float));
    draft_file_.write(reinterpret_cast<const char*>(draft_heads_.data()), std::streamsize(count) * hidden_ * sizeof(float));
    teacher_file_.write(reinterpret_cast<const char*>(teacher_heads_.data()), std::streamsize(count) * hidden_ * sizeof(float));
    target_file_.write(reinterpret_cast<const char*>(targets), std::streamsize(count) * sizeof(int32_t));
    for (int j = 0; j < count; ++j) {
        metadata_ << std::setprecision(9) << "{\"row\":" << rows_++ << ",\"request\":" << request_ << ",\"position\":" << position
                  << ",\"step\":" << j << ",\"proposal\":" << proposals_[size_t(j)]
                  << ",\"target\":" << targets[j] << ",\"probability\":" << probabilities_[size_t(j)]
                  << ",\"accepted_prefix\":" << accepted << ",\"width\":" << T
                  << ",\"fed\":" << (j < fed ? "true" : "false") << "}\n";
    }
    for (auto* file : {&residual_file_, &draft_file_, &teacher_file_, &target_file_, &metadata_}) {
        file->flush();
        if (!*file) { err = "draft alignment: paired output write failed"; return false; }
    }
    return true;
}

} // namespace strata::core
