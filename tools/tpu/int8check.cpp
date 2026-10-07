// int8check DIR: run the packaged predictors' test vectors through the same int8 FC-chain loop as
// llama-estore.cpp (estore_int8_chain) and compare with the Python reference outputs (expect.i8).
// Checks that the C++ CPU fallback is bit-exact with tools/predictor/train.py:int8_fc.
//
//   g++ -O2 -o int8check tools/tpu/int8check.cpp && ./int8check traces/<pkg>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static int8_t q8(double v) { return (int8_t) std::min(127.0, std::max(-128.0, std::nearbyint(v))); }

static std::vector<char> slurp(const std::string & p) {
    std::vector<char> v;
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(2); }
    char b[65536];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) v.insert(v.end(), b, b + n);
    fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s PKG_DIR\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    FILE * m = fopen((dir + "/manifest.txt").c_str(), "r");
    int span;
    if (!m || fscanf(m, " span %d", &span) != 1) { fprintf(stderr, "bad manifest\n"); return 2; }
    int il, ns;
    long bad = 0, total = 0;
    while (fscanf(m, "%d %d", &il, &ns) == 2) {
        std::vector<int> d(ns + 1);
        std::vector<double> a(ns + 1), w(ns);
        for (auto & x : d) if (fscanf(m, "%d", &x) != 1) return 2;
        for (auto & x : a) if (fscanf(m, "%lf", &x) != 1) return 2;
        for (auto & x : w) if (fscanf(m, "%lf", &x) != 1) return 2;
        const std::string base = dir + "/pred_" + std::to_string(il);
        auto X = slurp(base + ".in.i8"), Y = slurp(base + ".expect.i8");
        std::vector<std::vector<char>> W(ns), B(ns);
        for (int k = 0; k < ns; ++k) {
            W[k] = slurp(base + ".s" + std::to_string(k) + ".w.i8");
            B[k] = slurp(base + ".s" + std::to_string(k) + ".b.i32");
        }
        const int n = (int) (X.size() / d[0]);
        for (int t = 0; t < n; ++t) {
            std::vector<int8_t> in(X.begin() + (size_t) t * d[0], X.begin() + (size_t) (t + 1) * d[0]), out;
            for (int k = 0; k < ns; ++k) {
                out.assign(d[k + 1], 0);
                const double mult = a[k] * w[k] / a[k + 1];
                const int32_t * bias = (const int32_t *) B[k].data();
                for (int o = 0; o < d[k + 1]; ++o) {
                    int32_t acc = bias[o];
                    for (int i = 0; i < d[k]; ++i) acc += (int32_t) (int8_t) W[k][(size_t) o * d[k] + i] * in[i];
                    out[o] = q8(acc * mult);
                }
                in = out;
            }
            for (int o = 0; o < d[ns]; ++o) {
                bad += out[o] != (int8_t) Y[(size_t) t * d[ns] + o];
                total++;
            }
        }
    }
    printf("%s: %ld / %ld outputs differ from the Python int8 reference\n", dir.c_str(), bad, total);
    return bad != 0;
}
