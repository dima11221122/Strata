// Preserve the existing AVX2 reduction when a single input uses the paired kernel.
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main(int argc,char** argv) {
    const bool paired=argc==2 && std::strcmp(argv[1],"--paired")==0;
    namespace cpu=strata::kernels::cpu;
    if(!cpu::cpu_avx2_ok()) return 77;
    int failures=0,cases=0;
    std::mt19937 rng(173);
    for(int n:{512,2560}) for(int rows:{640,768}) {
        cpu::NativeFmt f;std::string err;
        if(!cpu::native_fmt(GGML_TYPE_IQ3_S,42,n,rows,f,err)) {std::fprintf(stderr,"%s\n",err.c_str());return 1;}
        std::vector<uint8_t> blob(f.bytes);
        for(auto& v:blob) v=static_cast<uint8_t>(rng());
        for(int role=0;role<2;++role) for(int r=0;r<rows;++r) for(int b=0;b<n/256;++b) {
            const size_t at=(role?f.up_off:0)+size_t(r)*f.gu_row+size_t(b)*110;
            // A small normal half scale: finite, mixed-sign dot products with visible rounding differences.
            blob[at]=0x00;blob[at+1]=0x18;
        }
        std::vector<std::vector<uint8_t>> acts(4,std::vector<uint8_t>(f.act_bytes));
        std::vector<float> x(n);
        const void* ap[4];
        for(int t=0;t<4;++t) {
            for(auto& v:x) v=float(int(rng()%2001)-1000)/251.f;
            cpu::native_quant_act(f,x.data(),acts[t].data());ap[t]=acts[t].data();
        }
        std::vector<float> gate(4*rows),up(4*rows),want(4*rows,-123.f),got(4*rows,-123.f);
        float *gp[4],*up_p[4],*out[4];
        for(int t=0;t<4;++t) {gp[t]=gate.data()+t*rows;up_p[t]=up.data()+t*rows;out[t]=got.data()+t*rows;}
        for(int nt=1;nt<=4;++nt) {
            cpu::iq256_rows(21,blob.data(),f.gu_row,n,ap,nt,gp,0,rows);
            cpu::iq256_rows(21,blob.data()+f.up_off,f.gu_row,n,ap,nt,up_p,0,rows);
            if(!paired && nt==1) {
                const auto* tc=ggml_get_type_traits_cpu(GGML_TYPE_IQ3_S);
                for(int r=0;r<rows;++r) {
                    tc->vec_dot(n,&gate[r],0,blob.data()+size_t(r)*f.gu_row,0,ap[0],0,1);
                    tc->vec_dot(n,&up[r],0,blob.data()+f.up_off+size_t(r)*f.gu_row,0,ap[0],0,1);
                }
            }
            for(const auto range:{std::pair<int,int>{0,rows},{7,rows-9},{rows-17,rows-3}}) {
                std::fill(want.begin(),want.end(),-123.f);std::fill(got.begin(),got.end(),-123.f);
                for(int t=0;t<nt;++t) for(int r=range.first;r<range.second;++r) {
                    const size_t at=size_t(t)*rows+r;
                    want[at]=(gate[at]/(1.f+std::exp(-gate[at])))*up[at];
                }
                cpu::native_gu_rows(f,blob.data(),ap,nt,out,range.first,range.second);
                bool finite=true;for(float v:got) finite=finite&&std::isfinite(v);
                const bool ok=finite && std::memcmp(want.data(),got.data(),want.size()*sizeof(float))==0;
                ++cases;
                if(!ok) {++failures;std::printf("FAIL n=%d rows=%d nt=%d range=%d:%d paired=%d\n",n,rows,nt,range.first,range.second,paired);}
            }
        }
    }
    std::printf("iq3 single pair: %d cases, %d failures\n",cases,failures);
    return failures?1:0;
}
