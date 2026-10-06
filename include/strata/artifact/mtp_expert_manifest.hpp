#pragma once
#include <cstdint>
#include <istream>
#include <string>

namespace strata::artifact {
enum class MtpExpertKind { CanonicalQ2, NativeQ5, NativeQ8, NativeQ3KQ2 };
inline bool parse_mtp_expert_manifest(std::istream& input, int64_t h, int64_t ff, int64_t ne,
                                      MtpExpertKind& kind, std::string& error) {
    std::string name, extra;
    int64_t got_h = 0, got_ff = 0, got_ne = 0;
    if (!(input >> name >> got_h >> got_ff >> got_ne) || (input >> extra) ||
        got_h != h || got_ff != ff || got_ne != ne ||
        (name != "canonical-q2_0" && name != "native-q5_0" && name != "native-q8_0" && name != "native-q3_k-q2_0")) {
        error = "mtp: malformed or incompatible experts.txt";
        return false;
    }
    kind = name == "native-q8_0" ? MtpExpertKind::NativeQ8
         : name == "native-q5_0" ? MtpExpertKind::NativeQ5
         : name == "native-q3_k-q2_0" ? MtpExpertKind::NativeQ3KQ2 : MtpExpertKind::CanonicalQ2;
    return true;
}
} // namespace strata::artifact
