#pragma once

#include <algorithm>

namespace strata::core {

// A fixed transfer budget: offload the experts serving the most window entries.
// Break ties from the end, as the routing-order policy does. Callers exclude
// resident, peer-owned and unpinned experts before assigning this budget.
inline void select_pcie_reuse(const int* reuse, const bool* eligible, int count, int quota, bool* selected) {
    std::fill(selected, selected + count, false);
    for (int q = 0; q < quota; ++q) {
        int best = -1;
        for (int i = 0; i < count; ++i)
            if (eligible[i] && !selected[i] && (best < 0 || reuse[i] >= reuse[best])) best = i;
        if (best < 0) break;
        selected[best] = true;
    }
}

}  // namespace strata::core
