#include "strata/artifact/mtp_expert_manifest.hpp"
#include <cstdio>
#include <sstream>

int main() {
    using namespace strata::artifact;
    struct Case { const char* text; bool valid; MtpExpertKind kind; };
    const Case cases[] = {
        {"canonical-q2_0 2560 640 512\n", true, MtpExpertKind::CanonicalQ2},
        {"native-q8_0 2560 640 512\n", true, MtpExpertKind::NativeQ8},
        {"native-q5_0 2560 640 512\n", true, MtpExpertKind::NativeQ5},
        {"native-q5_0 2560 640 256", false, MtpExpertKind::CanonicalQ2},
        {"native-q8_0 2560 640 256", false, MtpExpertKind::CanonicalQ2},
        {"native-q8_0 1280 640 512", false, MtpExpertKind::CanonicalQ2},
        {"native-q8_0 2560 1280 512", false, MtpExpertKind::CanonicalQ2},
        {"native-q4_0 2560 640 512", false, MtpExpertKind::CanonicalQ2},
        {"native-q8_0 2560 640", false, MtpExpertKind::CanonicalQ2},
        {"native-q8_0 2560 640 512 garbage", false, MtpExpertKind::CanonicalQ2},
        {"", false, MtpExpertKind::CanonicalQ2},
    };
    int failures = 0;
    for (const auto& c : cases) {
        std::istringstream in(c.text);
        auto kind = MtpExpertKind::CanonicalQ2;
        std::string error;
        const bool ok = parse_mtp_expert_manifest(in, 2560, 640, 512, kind, error);
        if (ok != c.valid || (ok && kind != c.kind) || (!ok && error.empty())) {
            std::fprintf(stderr, "FAIL expert manifest: %s\n", c.text);
            ++failures;
        }
    }
    std::printf("mtp_expert_manifest_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
