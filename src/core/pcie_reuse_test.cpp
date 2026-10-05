#include "strata/core/pcie_reuse.hpp"

#include <array>
#include <cstdio>

int main() {
    struct Case {
        const char* name;
        std::array<int, 5> reuse;
        std::array<bool, 5> eligible;
        int quota;
        std::array<bool, 5> expected;
    };
    const Case cases[] = {
        {"offload the most reused misses", {3, 1, 2, 1, 1}, {true, true, true, true, true}, 2,
         {true, false, true, false, false}},
        {"keep routing-order tie behavior", {1, 1, 1, 1, 1}, {true, true, true, true, true}, 2,
         {false, false, false, true, true}},
        {"skip resident or unpinned experts", {8, 7, 2, 1, 1}, {false, false, true, true, true}, 2,
         {false, false, true, false, true}},
        {"zero quota offloads nothing", {3, 1, 2, 1, 1}, {true, true, true, true, true}, 0,
         {false, false, false, false, false}},
        {"quota cannot exceed eligible experts", {3, 1, 2, 1, 1}, {false, true, false, false, true}, 9,
         {false, true, false, false, true}},
    };
    int failures = 0;
    for (const Case& c : cases) {
        std::array<bool, 5> selected = {true, true, true, true, true};
        strata::core::select_pcie_reuse(c.reuse.data(), c.eligible.data(), 5, c.quota, selected.data());
        if (selected != c.expected) { std::fprintf(stderr, "FAIL: %s\n", c.name); ++failures; }
    }
    std::printf("pcie_reuse_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
