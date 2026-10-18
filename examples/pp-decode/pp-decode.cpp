// Pipelined greedy decode over a layer-split GLM5-Next trunk [EXPERIMENTAL]
//
// The head context runs layers [0, s) and the tail [s, n_layer). While the tail finishes the
// confirmed token at p, the head already runs the NextN draft for p+1 on the other GPUs. A hit
// keeps the head's work, a miss restores the head's sequence and reruns it with the true token.
// The output must match a plain greedy run of the full trunk token for token.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../../src/llama-context.h"
#include "../../src/llama-ext.h"
#include "../../src/llama-model.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

static llama_token argmax(llama_context * ctx, int32_t i, int32_t n_vocab, float * gap = nullptr) {
    const float * logits = llama_get_logits_ith(ctx, i);
    const llama_token best = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
    if (gap) {
        float second = -INFINITY;
        for (int32_t k = 0; k < n_vocab; ++k) {
            if (k != best) {
                second = std::max(second, logits[k]);
            }
        }
        *gap = logits[best] - second;
    }
    return best;
}

// the greedy id of the one output row: from the in-graph argmax when enabled, which skips the logits copy
static llama_token top1(llama_context * ctx, int32_t n_vocab) {
    const int32_t * ids = llama_get_logits_argmax(ctx);
    return ids ? ids[0] : argmax(ctx, -1, n_vocab);
}

static double now_ms() {
    return ggml_time_us() / 1000.0;
}

static int env_int(const char * name) {
    const char * v = getenv(name);
    return v ? atoi(v) : 0;
}

// a persistent thread for the tail, so its decodes keep their per-thread CUDA stream
struct worker {
    std::mutex mtx;
    std::condition_variable cv;
    std::function<void()> job;
    bool busy = false;
    bool quit = false;
    std::thread th; // last: the thread starts once the members above exist

    worker() : th([this] { loop(); }) {}
    ~worker() {
        { std::lock_guard<std::mutex> lk(mtx); quit = true; }
        cv.notify_all();
        th.join();
    }
    void loop() {
        std::unique_lock<std::mutex> lk(mtx);
        while (true) {
            cv.wait(lk, [this] { return quit || busy; });
            if (quit) {
                return;
            }
            auto f = std::move(job);
            job = nullptr;
            lk.unlock();
            f();
            lk.lock();
            busy = false;
            cv.notify_all();
        }
    }
    void run(std::function<void()> f) {
        { std::lock_guard<std::mutex> lk(mtx); job = std::move(f); busy = true; }
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [this] { return !busy; });
    }
};

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    params.load_mtp = true;
    const int n_gen = params.n_predict > 0 ? params.n_predict : 128;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_model   * model   = init->model();
    llama_context * ctx_ref = init->context();
    if (!model || !ctx_ref) {
        LOG_ERR("failed to load the model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t n_embd  = llama_model_n_embd_out(model);
    const int32_t n_layer = llama_model_n_layer(model);

    // split where the layers change GPU halves, unless given
    int split = env_int("PP_SPLIT");
    if (split <= 0) {
        const int n_dev = (int) model->devices.size();
        for (int il = 1; il < n_layer && split <= 0; ++il) {
            const ggml_backend_dev_t dev = model->dev_layer(il);
            for (int d = n_dev/2; d < n_dev; ++d) {
                if (model->devices[d].dev == dev) {
                    split = il;
                    break;
                }
            }
        }
    }
    LOG_INF("split at layer %d of %d\n", split, n_layer);

    auto cp = common_context_params_to_llama(params);
    cp.n_batch  = cp.n_ubatch;
    cp.n_rs_seq = 0;

    auto cp_head = cp;
    cp_head.layer_begin = 0;
    cp_head.layer_end   = split;
    cp_head.n_seq_max   = 2;
    cp_head.kv_unified  = true;

    auto cp_tail = cp;
    cp_tail.layer_begin = split;
    cp_tail.layer_end   = 0;
    cp_tail.n_seq_max   = 1;

    llama_context * ctx_head = llama_init_from_model(model, cp_head);
    llama_context * ctx_tail = llama_init_from_model(model, cp_tail);
    if (!ctx_head || !ctx_tail) {
        LOG_ERR("failed to create the split contexts\n");
        return 1;
    }

    auto cp_mtp = cp;
    cp_mtp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    cp_mtp.n_seq_max = 1;
    cp_mtp.ctx_other = ctx_tail;
    llama_context * ctx_mtp = llama_init_from_model(model, cp_mtp);
    if (!ctx_mtp) {
        LOG_ERR("failed to create the MTP context\n");
        return 1;
    }
    llama_set_embeddings_nextn(ctx_tail, true, /*masked*/ false);

    std::vector<llama_token> prompt = common_tokenize(ctx_ref, params.prompt, true, true);
    const int n_prompt = (int) prompt.size();
    const int n_ub = (int) cp.n_ubatch;

    // reference: plain greedy on the full trunk
    std::vector<llama_token> ref;
    std::vector<float> ref_gap(n_gen, 0.0f); // top-1 minus top-2 logit, to tell near ties from bugs
    double t_ref = 0.0;
    {
        const int n_batch_ref = (int) llama_n_batch(ctx_ref);
        llama_batch b = llama_batch_init(n_batch_ref, 0, 1);
        for (int i0 = 0; i0 < n_prompt; i0 += n_batch_ref) {
            common_batch_clear(b);
            for (int i = i0; i < std::min(n_prompt, i0 + n_batch_ref); ++i) {
                common_batch_add(b, prompt[i], i, { 0 }, i == n_prompt - 1);
            }
            if (llama_decode(ctx_ref, b)) {
                LOG_ERR("reference prefill failed\n");
                return 1;
            }
        }
        llama_token tok = argmax(ctx_ref, -1, n_vocab);
        ref.push_back(tok);
        const double t0 = now_ms();
        for (int i = 1; i < n_gen; ++i) {
            common_batch_clear(b);
            common_batch_add(b, tok, n_prompt + i - 1, { 0 }, true);
            if (llama_decode(ctx_ref, b)) {
                LOG_ERR("reference decode failed\n");
                return 1;
            }
            tok = argmax(ctx_ref, -1, n_vocab, &ref_gap[i]);
            ref.push_back(tok);
        }
        t_ref = now_ms() - t0;
        llama_batch_free(b);
    }

    llama_batch bh = llama_batch_init(n_ub, 0, 1);
    llama_batch bt = llama_batch_init(n_ub, 0, 1);
    // MTP rows carry a token and an embedding (same workaround as common/speculative.cpp)
    llama_batch bm = llama_batch_init(n_ub, n_embd, 1);
    bm.token = (llama_token *) malloc(sizeof(llama_token) * n_ub);

    llama_memory_t mem_head = llama_get_memory(ctx_head);

    LOG_INF("reference done, pipelined run starts\n");

    // prefill one ubatch at a time: the hop holds two ubatches of rows
    std::vector<float> h_prev(n_embd, 0.0f); // tail hidden row of the position before
    llama_token next = -1;
    for (int i0 = 0; i0 < n_prompt; i0 += n_ub) {
        const int n = std::min(n_ub, n_prompt - i0);
        common_batch_clear(bh);
        common_batch_clear(bt);
        for (int i = 0; i < n; ++i) {
            common_batch_add(bh, prompt[i0 + i], i0 + i, { 0 }, false);
            common_batch_add(bt, prompt[i0 + i], i0 + i, { 0 }, i0 + i == n_prompt - 1);
        }
        if (llama_decode(ctx_head, bh)) {
            LOG_ERR("head prefill failed\n");
            return 1;
        }
        llama_synchronize(ctx_head);
        if (llama_decode(ctx_tail, bt)) {
            LOG_ERR("tail prefill failed\n");
            return 1;
        }
        // MTP rows: token at p with the tail hidden of p-1
        const float * h = llama_get_embeddings_nextn(ctx_tail);
        common_batch_clear(bm);
        for (int i = 0; i < n; ++i) {
            common_batch_add(bm, prompt[i0 + i], i0 + i, { 0 }, false);
            memcpy(bm.embd + (size_t) i*n_embd, i == 0 ? h_prev.data() : h + (size_t) (i - 1)*n_embd, n_embd*sizeof(float));
        }
        memcpy(h_prev.data(), h + (size_t) (n - 1)*n_embd, n_embd*sizeof(float));
        if (i0 + n == n_prompt) {
            next = argmax(ctx_tail, -1, n_vocab);
        }
        if (llama_decode(ctx_mtp, bm)) {
            LOG_ERR("MTP prefill failed\n");
            return 1;
        }
    }

    // next = x_{p+1}, h_prev = h_p, the head holds [0, p] in seq 0
    // the head always decodes in seq 0, so its graph is reused, and backs it up into seq 1
    worker tail_thr;
    std::vector<llama_token> out = { next };
    llama_token spec = -1;
    int n_hit = 0;
    double t_mtp = 0.0, t_slot = 0.0, t_redo = 0.0, t_head = 0.0, t_bak = 0.0, t_tail = 0.0;
    bool ok = true;
    // PP_SERIAL=1: never speculate, the head runs each true token before the tail
    const bool serial = env_int("PP_SERIAL") != 0;
    // debug: PP_MISS=1 speculates a wrong token every step, exercising the restore
    const bool force_miss = env_int("PP_MISS") != 0;
    std::vector<int> miss_at;

    // PP_HOSTARGMAX=1: copy the full logits and take the argmax on the host
    if (env_int("PP_HOSTARGMAX") == 0) {
        ctx_tail->set_dflash_target_argmax(1);
        ctx_mtp->set_dflash_target_argmax(1);
    }

    const double t0 = now_ms();
    for (int p = n_prompt - 1; (int) out.size() < n_gen; ++p) {
        double t = now_ms();
        if (spec == next) {
            n_hit++;
        } else {
            if (spec >= 0) {
                miss_at.push_back((int) out.size());
                const bool rm_ok = llama_memory_seq_rm(mem_head, 0, -1, -1);
                llama_memory_seq_cp(mem_head, 1, 0, -1, -1);
                if (!rm_ok || llama_memory_seq_pos_max(mem_head, 0) != p) {
                    LOG_ERR("restore at p = %d: rm %d, pos_max %d\n", p, rm_ok, llama_memory_seq_pos_max(mem_head, 0));
                }
            }
            common_batch_clear(bh);
            common_batch_add(bh, next, p + 1, { 0 }, false);
            ok &= llama_decode(ctx_head, bh) == 0;
            llama_synchronize(ctx_head);
        }
        t_redo += now_ms() - t;

        // the tail confirms x_{p+1} while the MTP drafts x_{p+2} from x_{p+1} and h_p, and the head speculates it
        t = now_ms();
        common_batch_clear(bm);
        common_batch_add(bm, next, p + 1, { 0 }, true);
        memcpy(bm.embd, h_prev.data(), n_embd*sizeof(float));

        llama_token next_tail = -1;
        bool ok_tail = true;
        tail_thr.run([&, p, tok = next] {
            const double tt = now_ms();
            common_batch_clear(bt);
            common_batch_add(bt, tok, p + 1, { 0 }, true);
            ok_tail = llama_decode(ctx_tail, bt) == 0;
            next_tail = top1(ctx_tail, n_vocab);
            memcpy(h_prev.data(), llama_get_embeddings_nextn(ctx_tail), n_embd*sizeof(float));
            t_tail += now_ms() - tt;
        });

        llama_token draft = -1;
        if (!serial) {
            const double tm = now_ms();
            ok &= llama_decode(ctx_mtp, bm) == 0;
            draft = top1(ctx_mtp, n_vocab);
            if (force_miss) {
                draft = (draft + 1) % n_vocab;
            }
            t_mtp += now_ms() - tm;

            const double th = now_ms();
            llama_memory_seq_rm(mem_head, 1, -1, -1);
            llama_memory_seq_cp(mem_head, 0, 1, -1, -1);
            t_bak += now_ms() - th;
            common_batch_clear(bh);
            common_batch_add(bh, draft, p + 2, { 0 }, false);
            ok &= llama_decode(ctx_head, bh) == 0;
            llama_synchronize(ctx_head);
            t_head += now_ms() - th;
        }
        tail_thr.wait();
        next = next_tail;
        ok &= ok_tail;
        t_slot += now_ms() - t;

        spec = draft;
        out.push_back(next);
        if (!ok) {
            LOG_ERR("decode failed at p = %d\n", p);
            break;
        }
    }
    const double t_pp = now_ms() - t0;

    int n_match = 0;
    while (n_match < (int) std::min(out.size(), ref.size()) && out[n_match] == ref[n_match]) {
        n_match++;
    }
    const int n_steps = (int) out.size() - 1;

    LOG_INF("\nprompt %d tokens, generated %d\n", n_prompt, n_gen);
    LOG_INF("reference: %.2f tok/s (%.2f ms/token)\n", (n_gen - 1)*1000.0/t_ref, t_ref/(n_gen - 1));
    LOG_INF("pipelined: %.2f tok/s (%.2f ms/token), hit %d miss %d (%.1f%%)\n",
            n_steps*1000.0/t_pp, t_pp/n_steps, n_hit, (int) miss_at.size(), 100.0*n_hit/std::max(1, n_hit + (int) miss_at.size()));
    LOG_INF("  per token: redo %.2f ms, mtp %.2f ms, slot %.2f ms (head %.2f incl. backup %.2f, tail %.2f)\n",
            t_redo/n_steps, t_mtp/n_steps, t_slot/n_steps, t_head/n_steps, t_bak/n_steps, t_tail/n_steps);
    const std::pair<const char *, llama_context *> ctxs[] = { { "head", ctx_head }, { "tail", ctx_tail }, { "mtp", ctx_mtp } };
    for (const auto & [name, c] : ctxs) {
        const auto pd = llama_perf_context(c);
        LOG_INF("  %s: %d evals, %d graphs reused\n", name, pd.n_eval, pd.n_reused);
    }
    LOG_INF("match: %d / %d tokens %s\n", n_match, (int) ref.size(), n_match == (int) ref.size() ? "IDENTICAL" : "DIVERGED");
    if (n_match < (int) ref.size()) {
        LOG_INF("  first divergence: ref %d, ours %d, ref top-2 gap %.4f\n", ref[n_match], out[n_match], ref_gap[n_match]);
        std::string ms;
        for (int i : miss_at) {
            if (i > n_match + 1) break;
            ms += " " + std::to_string(i);
        }
        LOG_INF("  misses up to it (token index):%s\n", ms.c_str());
    }
    LOG_INF("text: %s\n", common_detokenize(ctx_ref, out).c_str());

    llama_batch_free(bm);
    llama_batch_free(bh);
    llama_batch_free(bt);
    llama_free(ctx_mtp);
    llama_free(ctx_tail);
    llama_free(ctx_head);
    llama_backend_free();

    return n_match == (int) ref.size() ? 0 : 2;
}
