// llama-expert-store: prepare, verify and benchmark the files a tiered MoE
// cache streams experts from. A GGUF already holds each expert bank in place;
// a native safetensors model holds its canonical banks in a repack cache.

#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-moe-store.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace fs = std::filesystem;

namespace {

// One expert bank: n_expert slices of expert_size bytes, contiguous from offset.
struct bank {
    std::string name;
    std::string path;
    int layer = -1;
    int file = -1;
    uint64_t offset = 0;
    int64_t n_expert = 0;
    size_t expert_size = 0;
};

int layer_of(const std::string & name) {
    return name.rfind("blk.", 0) == 0 ? atoi(name.c_str() + 4) : -1;
}

// The moe-cache name rule, less the per-expert scale tensors that stay resident.
bool is_bank(const std::string & name, int64_t n_expert) {
    return (name.find("_exps.weight") != std::string::npos || name.find("_chexps.weight") != std::string::npos) &&
        n_expert > 1;
}

void add_gguf(const std::string & path, std::vector<bank> & banks) {
    ggml_context * ctx = nullptr;
    gguf_init_params params = { /*no_alloc =*/ true, /*ctx =*/ &ctx };
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf) {
        throw std::runtime_error("cannot read GGUF " + path);
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); i++) {
        const char * name = gguf_get_tensor_name(gguf, i);
        const ggml_tensor * tensor = ggml_get_tensor(ctx, name);
        if (!tensor || !is_bank(name, tensor->ne[2])) {
            continue;
        }
        banks.push_back({name, path, layer_of(name), -1,
                gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i), tensor->ne[2], tensor->nb[2]});
    }
    gguf_free(gguf);
    ggml_free(ctx);
}

// A repack cache entry is <dir>/<id>/weights with a receipt naming its layout,
// "name:type:ne0:ne1:ne2:ne3".
void add_repack_cache(const std::string & dir, std::vector<bank> & banks) {
    for (const auto & entry : fs::directory_iterator(dir)) {
        const fs::path receipt = entry.path() / "receipt.json";
        const fs::path weights = entry.path() / "weights";
        if (!entry.is_directory() || !fs::is_regular_file(receipt) || !fs::is_regular_file(weights)) {
            continue;
        }
        std::ifstream input(receipt);
        const nlohmann::json json = nlohmann::json::parse(input, nullptr, false);
        if (!json.is_object() || !json.contains("layout") || !json["layout"].is_string()) {
            continue;
        }
        const std::string layout = json["layout"].get<std::string>();
        std::vector<std::string> fields;
        std::stringstream parts(layout);
        for (std::string field; std::getline(parts, field, ':');) {
            fields.push_back(field);
        }
        if (fields.size() != 6) {
            continue;
        }
        const int64_t n_expert = atoll(fields[4].c_str());
        const uint64_t size = fs::file_size(weights);
        if (!is_bank(fields[0], n_expert) || size % (uint64_t)n_expert != 0) {
            continue;
        }
        banks.push_back({fields[0], weights.string(), layer_of(fields[0]), -1, 0, n_expert,
                (size_t)(size / (uint64_t)n_expert)});
    }
}

std::vector<bank> collect(const std::vector<std::string> & sources) {
    std::vector<bank> banks;
    for (const std::string & source : sources) {
        if (fs::is_directory(source)) {
            add_repack_cache(source, banks);
        } else {
            add_gguf(source, banks);
        }
    }
    std::sort(banks.begin(), banks.end(), [](const bank & a, const bank & b) {
        return a.layer != b.layer ? a.layer < b.layer : a.name < b.name;
    });
    return banks;
}

void * aligned_alloc_bytes(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, GGML_MOE_STORE_ALIGN);
#else
    void * ptr = nullptr;
    return posix_memalign(&ptr, GGML_MOE_STORE_ALIGN, size) == 0 ? ptr : nullptr;
#endif
}

// Room for one slice at any file offset, however it is aligned.
size_t slot_span(size_t size) {
    return ggml_moe_store_span(GGML_MOE_STORE_ALIGN - 1, size);
}

void aligned_free_bytes(void * ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Counts outstanding reads; each read's done callback signals one completion.
struct waiter {
    std::mutex mu;
    std::condition_variable cv;
    int pending = 0;
    int failed = 0;

    static void done(void * user, bool ok) {
        waiter * self = (waiter *)user;
        std::lock_guard<std::mutex> lock(self->mu);
        self->pending--;
        self->failed += ok ? 0 : 1;
        self->cv.notify_all();
    }
};

struct options {
    std::vector<std::string> sources;
    std::string model;
    std::string cache;
    int threads = 8;
    size_t piece = 2304 * 1024;
    int inflight = 4;
    double seconds = 10.0;
    int64_t sample = 0;
};

ggml_moe_store * open_store(const options & opt, std::vector<bank> & banks) {
    ggml_moe_store * store = ggml_moe_store_init(opt.threads, opt.piece);
    if (!store) {
        throw std::runtime_error("cannot start the reader pool");
    }
    std::map<std::string, int> files;
    for (bank & b : banks) {
        auto it = files.find(b.path);
        if (it == files.end()) {
            it = files.emplace(b.path, ggml_moe_store_open(store, b.path.c_str())).first;
        }
        if (it->second < 0) {
            ggml_moe_store_free(store);
            throw std::runtime_error("cannot open " + b.path);
        }
        b.file = it->second;
    }
    return store;
}

// Random experts, each read as every bank of its layer at once, with
// opt.inflight experts outstanding, the way decode misses arrive.
int cmd_bench(const options & opt) {
    std::vector<bank> banks = collect(opt.sources);
    if (banks.empty()) {
        fprintf(stderr, "no expert banks found\n");
        return 1;
    }
    std::vector<std::vector<size_t>> layers;
    for (size_t i = 0; i < banks.size(); i++) {
        if (layers.empty() || banks[layers.back()[0]].layer != banks[i].layer) {
            layers.emplace_back();
        }
        layers.back().push_back(i);
    }
    ggml_moe_store * store = open_store(opt, banks);
    size_t slot_bytes = 0;
    for (const auto & layer : layers) {
        size_t bytes = 0;
        for (size_t i : layer) {
            bytes += slot_span(banks[i].expert_size);
        }
        slot_bytes = std::max(slot_bytes, bytes);
    }
    std::vector<void *> slots(opt.inflight);
    for (void *& slot : slots) {
        slot = aligned_alloc_bytes(slot_bytes);
    }

    struct expert_read {
        waiter wait;
        std::chrono::steady_clock::time_point start;
        size_t bytes = 0;
    };
    std::vector<std::unique_ptr<expert_read>> reads(opt.inflight);
    std::mt19937_64 rng(1234);
    std::vector<double> latencies;
    uint64_t total_bytes = 0;
    int failures = 0;
    auto issue = [&](int k) {
        const auto & layer = layers[rng() % layers.size()];
        const int64_t expert = (int64_t)(rng() % (uint64_t)banks[layer[0]].n_expert);
        auto read = std::make_unique<expert_read>();
        read->start = std::chrono::steady_clock::now();
        char * dst = (char *)slots[k];
        read->wait.pending = (int)layer.size();
        for (size_t i : layer) {
            const bank & b = banks[i];
            const uint64_t offset = b.offset + (uint64_t)expert * b.expert_size;
            ggml_moe_store_read(store, b.file, offset, b.expert_size, dst,
                    GGML_MOE_STORE_DEMAND, waiter::done, &read->wait);
            dst += slot_span(b.expert_size);
            read->bytes += b.expert_size;
        }
        reads[k] = std::move(read);
    };
    const auto t0 = std::chrono::steady_clock::now();
    const auto t_end = t0 + std::chrono::duration<double>(opt.seconds);
    for (int k = 0; k < opt.inflight; k++) {
        issue(k);
    }
    int live = opt.inflight;
    // Round-robin: wait for the oldest outstanding expert, then reuse its slot.
    for (int k = 0;; k = (k + 1) % opt.inflight) {
        expert_read & read = *reads[k];
        {
            std::unique_lock<std::mutex> lock(read.wait.mu);
            read.wait.cv.wait(lock, [&] { return read.wait.pending == 0; });
        }
        const auto now = std::chrono::steady_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(now - read.start).count());
        total_bytes += read.bytes;
        failures += read.wait.failed;
        if (now >= t_end) {
            // Each slot is visited once more after the deadline, in order.
            if (--live == 0) {
                break;
            }
            continue;
        }
        issue(k);
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::sort(latencies.begin(), latencies.end());
    const auto pct = [&](double p) { return latencies[std::min(latencies.size() - 1, (size_t)(p * latencies.size()))]; };
    printf("banks=%zu layers=%zu expert=%.2f MiB threads=%d piece=%zu inflight=%d  %.2f GB/s  %.1f experts/s  "
           "latency ms p50=%.2f p90=%.2f p99=%.2f  failures=%d\n",
           banks.size(), layers.size(), slot_bytes / 1048576.0, opt.threads, opt.piece, opt.inflight,
           total_bytes / dt / 1e9, latencies.size() / dt, pct(0.5), pct(0.9), pct(0.99), failures);
    ggml_moe_store_free(store);
    for (void * slot : slots) {
        aligned_free_bytes(slot);
    }
    return failures ? 1 : 0;
}

// Reads expert slices through the reader pool and compares them byte for byte
// with ordinary buffered reads of the same file.
int cmd_verify(const options & opt) {
    std::vector<bank> banks = collect(opt.sources);
    if (banks.empty()) {
        fprintf(stderr, "no expert banks found\n");
        return 1;
    }
    ggml_moe_store * store = open_store(opt, banks);
    std::mt19937_64 rng(42);
    int64_t checked = 0;
    int64_t mismatches = 0;
    std::vector<char> expect;
    for (const bank & b : banks) {
        std::ifstream file(b.path, std::ios::binary);
        void * slot = aligned_alloc_bytes(slot_span(b.expert_size));
        expect.resize(b.expert_size);
        const int64_t n = opt.sample > 0 ? std::min(opt.sample, b.n_expert) : b.n_expert;
        for (int64_t i = 0; i < n; i++) {
            const int64_t expert = opt.sample > 0 ? (int64_t)(rng() % (uint64_t)b.n_expert) : i;
            const uint64_t offset = b.offset + (uint64_t)expert * b.expert_size;
            waiter wait;
            wait.pending = 1;
            const char * got = (const char *)ggml_moe_store_read(store, b.file, offset, b.expert_size, slot,
                    GGML_MOE_STORE_DEMAND, waiter::done, &wait);
            {
                std::unique_lock<std::mutex> lock(wait.mu);
                wait.cv.wait(lock, [&] { return wait.pending == 0; });
            }
            file.seekg((std::streamoff)offset);
            file.read(expect.data(), (std::streamsize)b.expert_size);
            const bool ok = !wait.failed && file && memcmp(got, expect.data(), b.expert_size) == 0;
            if (!ok) {
                fprintf(stderr, "mismatch: %s expert %lld\n", b.name.c_str(), (long long)expert);
                mismatches++;
            }
            checked++;
        }
        aligned_free_bytes(slot);
    }
    ggml_moe_store_free(store);
    printf("verified %lld expert slices in %zu banks: %lld mismatches\n",
           (long long)checked, banks.size(), (long long)mismatches);
    return mismatches ? 1 : 0;
}

// Linux cgroup page cache in bytes (the "file" line of memory.stat), or -1.
long long cgroup_file_bytes() {
    std::ifstream cgroup("/proc/self/cgroup");
    std::string line;
    std::string path;
    while (std::getline(cgroup, line)) {
        if (line.rfind("0::", 0) == 0) {
            path = "/sys/fs/cgroup" + line.substr(3) + "/memory.stat";
        }
    }
    std::ifstream stat(path);
    for (std::string key; stat >> key;) {
        long long value = 0;
        stat >> value;
        if (key == "file") {
            return value;
        }
    }
    return -1;
}

// Loads a native safetensors model once, CPU only, so its canonical expert
// banks are written to the repack cache.
int cmd_pack(const options & opt) {
    if (opt.model.empty() || opt.cache.empty()) {
        fprintf(stderr, "pack needs --model DIR and --repack-cache DIR\n");
        return 1;
    }
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    ggml_backend_dev_t no_devices[] = { nullptr };
    params.devices = no_devices;
    params.n_gpu_layers = 0;
    params.load_mode = LLAMA_LOAD_MODE_MMAP;
    params.repack_cache = opt.cache.c_str();
    const long long file_before = cgroup_file_bytes();
    const auto t0 = std::chrono::steady_clock::now();
    llama_model * model = llama_model_load_from_file(opt.model.c_str(), params);
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const long long file_after = cgroup_file_bytes();
    if (!model) {
        fprintf(stderr, "failed to load %s\n", opt.model.c_str());
        return 1;
    }
    llama_model_free(model);
    llama_backend_free();
    printf("packed %s into %s in %.1f s; cgroup page cache %.2f -> %.2f GiB\n", opt.model.c_str(),
           opt.cache.c_str(), dt, file_before / 1073741824.0, file_after / 1073741824.0);
    return 0;
}

void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s pack   --model DIR --repack-cache DIR\n"
        "       %s verify [--sample N] SOURCE...\n"
        "       %s bench  [--threads N] [--piece BYTES] [--inflight N] [--seconds S] SOURCE...\n"
        "SOURCE is a GGUF file (every shard) or a repack cache directory.\n",
        argv0, argv0, argv0);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    const std::string cmd = argv[1];
    options opt;
    for (int i = 2; i < argc; i++) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--threads" && has_value) {
            opt.threads = atoi(argv[++i]);
        } else if (arg == "--piece" && has_value) {
            opt.piece = (size_t)atoll(argv[++i]);
        } else if (arg == "--inflight" && has_value) {
            opt.inflight = std::max(1, atoi(argv[++i]));
        } else if (arg == "--seconds" && has_value) {
            opt.seconds = atof(argv[++i]);
        } else if (arg == "--sample" && has_value) {
            opt.sample = atoll(argv[++i]);
        } else if (arg == "--model" && has_value) {
            opt.model = argv[++i];
        } else if (arg == "--repack-cache" && has_value) {
            opt.cache = argv[++i];
        } else if (!arg.empty() && arg[0] == '-') {
            usage(argv[0]);
            return 1;
        } else {
            opt.sources.push_back(arg);
        }
    }
    try {
        if (cmd == "pack") {
            return cmd_pack(opt);
        }
        if (cmd == "verify") {
            return cmd_verify(opt);
        }
        if (cmd == "bench") {
            return cmd_bench(opt);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    usage(argv[0]);
    return 1;
}
