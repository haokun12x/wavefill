// Added by Wavefill (2026). See docs/wavefill/technical-report.zh-CN.md §8.4.
// Agent-turn prefill latency: a long conversation history whose state the engine already holds,
// followed by a short new message (a tool result, a file, a user turn). Measures what an agent
// loop waits for on every step: prefill of the delta only, at a large context offset.
//
// For each (history H, delta d): a fresh conversation whose H-token history is prefilled untimed,
// then turns that each append d new tokens to the previous prompt + output (exactly what an agent
// loop sends; the engine keeps the latest endpoint, so only the d new tokens are prefilled), with
// requested_output_tokens = 1. reused_prompt_tokens in the output confirms the reuse.
//
//   tools/wavefill/build-agent-turns.sh build
//   build/agent_turns --weights A.ninfer --corpus ids --history 32768,65536 \
//       --deltas 128,256,512,1024,2048,4096 --reps 3 [--kv bf16|int8|fp8] [--csv out.csv]

#include "ninfer/engine.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint32_t> parse_list(const std::string& s) {
    std::vector<std::uint32_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    return out;
}

std::vector<ninfer::TokenId> load_ids(const std::string& path) {
    std::ifstream in(path);
    std::vector<ninfer::TokenId> ids;
    long long v = 0;
    while (in >> v) ids.push_back(static_cast<ninfer::TokenId>(v));
    if (ids.empty()) throw std::runtime_error("empty corpus " + path);
    return ids;
}

ninfer::RequestOptions one_token_request() {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.execution.allow_prefix_reuse      = true;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    options.output.preserve_special_tokens    = true;
    return options;
}

} // namespace

int main(int argc, char** argv) {
    std::string weights, corpus_path, csv_path, kv = "bf16";
    std::vector<std::uint32_t> histories{65536}, deltas{128, 256, 512, 1024, 2048, 4096};
    int reps = 3;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--weights") weights = v;
        else if (a == "--corpus") corpus_path = v;
        else if (a == "--history") histories = parse_list(v);
        else if (a == "--deltas") deltas = parse_list(v);
        else if (a == "--reps") reps = std::stoi(v);
        else if (a == "--kv") kv = v;
        else if (a == "--csv") csv_path = v;
        else throw std::invalid_argument("unknown option " + a);
    }
    const auto corpus = load_ids(corpus_path);

    std::uint32_t max_h = 0, max_d = 0;
    for (auto h : histories) max_h = std::max(max_h, h);
    for (auto d : deltas) max_d = std::max(max_d, d);
    const std::uint32_t max_context =
        ((max_h + (max_d + 1) * static_cast<std::uint32_t>(reps + 1) + 16 + 1023) / 1024) * 1024;

    ninfer::EngineOptions eo;
    eo.artifact_path = weights;
    eo.max_context   = max_context;
    eo.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
    eo.prefill_chunk = 1024;
    eo.kv_cache      = kv == "int8"  ? ninfer::KvCacheStorage::Int8Group64
                       : kv == "fp8" ? ninfer::KvCacheStorage::Fp8E4M3Row256
                                     : ninfer::KvCacheStorage::BFloat16;
    ninfer::Engine engine(eo);

    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        csv << "history,delta,rep,reused,prefill_s,first_token_s\n";
    }
    std::printf("%8s %6s %4s %8s %10s %10s\n", "history", "delta", "rep", "reused", "prefill_ms",
                "ttft_ms");

    for (const auto h : histories) {
        for (const auto d : deltas) {
            if (h + (d + 1) * (reps + 1) > corpus.size()) throw std::runtime_error("corpus too short");
            // A fresh conversation per (history, delta): untimed full prefill of the history, then
            // one warm-up turn and `reps` measured turns, each extending the previous turn's
            // prompt + output exactly (the engine's private endpoint).
            std::vector<ninfer::TokenId> prompt(corpus.begin(), corpus.begin() + h);
            auto seed = engine.generate(engine.prepare_tokens(prompt, true), one_token_request());
            prompt.insert(prompt.end(), seed.generated_token_ids.begin(),
                          seed.generated_token_ids.end());
            std::size_t cursor = h;
            for (int r = -1; r < reps; ++r) {
                const std::size_t context = prompt.size();
                prompt.insert(prompt.end(), corpus.begin() + cursor, corpus.begin() + cursor + d);
                cursor += d;
                auto res = engine.generate(engine.prepare_tokens(prompt, true), one_token_request());
                prompt.insert(prompt.end(), res.generated_token_ids.begin(),
                              res.generated_token_ids.end());
                if (r < 0) continue;
                std::printf("%8zu %6u %4d %8u %10.2f %10.2f\n", context, d, r,
                            res.reused_prompt_tokens, res.timings.prefill_seconds * 1e3,
                            res.timings.first_token_seconds * 1e3);
                std::fflush(stdout);
                if (csv) {
                    csv << context << ',' << d << ',' << r << ',' << res.reused_prompt_tokens << ','
                        << res.timings.prefill_seconds << ',' << res.timings.first_token_seconds
                        << '\n';
                }
            }
        }
    }
    return 0;
}
