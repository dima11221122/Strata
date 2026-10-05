#pragma once

#include <cstdint>

namespace strata::core {

/// Host metadata for immutable native blobs in the verifier's existing staging arena.
/// The host may admit a blob only after protecting every current-layer hit. The
/// next layer cannot observe its tag until the GPU finishes this layer's fetch/join.
class StagingCache {
public:
    static constexpr int capacity = 64; // metadata maximum; begin() bounds it to the allocated slots

    void clear() {
        if (source_ == nullptr) return;
        for (auto& tag : tags_) tag = {};
        clock_ = 0;
        source_ = nullptr;
    }

    void begin(const void* source, uint64_t base, uint64_t stride, int slots) {
        if (source_ != source || base_ != base || stride_ != stride || slots_ != slots) clear();
        source_ = source;
        base_ = base;
        stride_ = stride;
        slots_ = slots;
    }

    int find(int64_t layer, int32_t expert, uint64_t bytes) {
        for (int i = 0; i < slots_; ++i) {
            auto& tag = tags_[i];
            if (tag.valid && tag.layer == layer && tag.expert == expert && tag.bytes == bytes) {
                tag.age = ++clock_;
                return i;
            }
        }
        return -1;
    }

    int reserve(int64_t layer, int32_t expert, uint64_t bytes, bool* protected_slots) {
        int victim = -1;
        for (int i = 0; i < slots_; ++i) {
            if (protected_slots[i]) continue;
            if (!tags_[i].valid) { victim = i; break; }
            if (victim < 0 || tags_[i].age < tags_[victim].age) victim = i;
        }
        if (victim < 0) return -1;
        tags_[victim] = {layer, expert, bytes, ++clock_, true};
        protected_slots[victim] = true;
        return victim;
    }

private:
    struct Tag {
        int64_t layer = -1;
        int32_t expert = -1;
        uint64_t bytes = 0, age = 0;
        bool valid = false;
    };
    Tag tags_[capacity] = {};
    const void* source_ = nullptr;
    uint64_t base_ = 0, stride_ = 0, clock_ = 0;
    int slots_ = 0;
};

}  // namespace strata::core
