#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::core {

// Measured costs in microseconds: CPU/resident/PCIe are affine in group and
// input counts; fetch is affine in copied blobs. Original quantized bytes stay intact.
struct PcieCost {
    int gu_type = 0, down_type = 0;
    uint64_t blob_bytes = 0;
    double cpu[3] = {}, resident[3] = {}, fetch[2] = {}, pcie[3] = {};
};

inline void select_pcie_cost(const PcieCost& cost, const int* reuse, const bool* missed,
                             const bool* eligible, int count, int quota, int resident_groups,
                             int resident_entries, bool overlap, bool* selected) {
    std::fill(selected, selected + count, false);
    int cpu_groups = 0, cpu_entries = 0;
    for (int i = 0; i < count; ++i) if (missed[i]) { ++cpu_groups; cpu_entries += reuse[i]; }
    const double resident = cost.resident[0] + cost.resident[1] * resident_groups + cost.resident[2] * resident_entries;
    const auto completion = [&](int cpu_n, int cpu_inputs, int fetch_n, int fetch_inputs) {
        const double cpu = cpu_n > 0 ? cost.cpu[0] + cost.cpu[1] * cpu_n + cost.cpu[2] * cpu_inputs : 0;
        const double fetch = cost.fetch[0] + cost.fetch[1] * fetch_n;
        const double gpu = (overlap ? std::max(resident, fetch) : resident + fetch) +
                           cost.pcie[0] + cost.pcie[1] * fetch_n + cost.pcie[2] * fetch_inputs;
        return std::max(cpu, gpu);
    };
    int fetch_inputs = 0;
    double current = completion(cpu_groups, cpu_entries, 0, 0);
    for (int n = 0; n < quota; ++n) {
        int best = -1;
        double best_time = current - 5.0; // margin against tiny model differences
        for (int i = 0; i < count; ++i) {
            if (!missed[i] || !eligible[i] || selected[i]) continue;
            const double trial = completion(cpu_groups - 1, cpu_entries - reuse[i], n + 1, fetch_inputs + reuse[i]);
            if (trial < best_time) { best = i; best_time = trial; }
        }
        if (best < 0) break;
        selected[best] = true;
        --cpu_groups; cpu_entries -= reuse[best]; fetch_inputs += reuse[best]; current = best_time;
    }
}

} // namespace strata::core
