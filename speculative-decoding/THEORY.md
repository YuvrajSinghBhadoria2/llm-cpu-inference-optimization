# Theory & Hypotheses (original plan)

> NOTE: This document is the **original theoretical plan/hypothesis** for the
> speculative-decoding extension. Its *predicted* speedups (1.5–2× on CPU,
> 6.8–8× total) were **refuted by measurement** on this hardware. The
> authoritative, measured results are in `README.md`. Read this for the theory,
> math, and academic context only.

# Research Findings: Speculative Decoding + Quantization for CPU Inference

**Project:** Advanced LLM Inference Optimization on CPU  
**Hardware:** MacBook Pro 16,1 (2019), Intel i7-9750H (6 cores / 12 threads), 16 GB RAM  
**Baseline:** 4× speedup achieved (6.67 → 26.85 tok/s) via thread tuning + q4_k_m quantization  
**Goal:** 2-3× additional speedup through speculative decoding + advanced quantization  
**Date:** August 25, 2026

---

## Executive Summary

This research investigates **speculative decoding** and **advanced quantization frameworks** for CPU-only LLM inference. Key findings:

1. **llama.cpp has speculative decoding support** (as of 2024-2025) but it's experimental and primarily GPU-optimized
2. **CPU-specific challenges exist**: memory bandwidth constraints, KV cache overhead, draft model scheduling
3. **GGUF is the only CPU-optimized quantization format**; GPTQ/AWQ require GPU kernels
4. **Expected CPU speedup: 1.5-2.5×** (lower than GPU's 2-3×) due to memory-bound architecture
5. **Implementation requires custom draft model integration** into llama.cpp's inference pipeline

**Recommendation:** Focus on GGUF-based speculative decoding with Qwen2.5 draft models; defer GPTQ/AWQ (GPU-only).

---

## 1. Speculative Decoding: State of Research

### 1.1 Core Algorithm

**Speculative decoding** (also called "assisted generation" or "draft-then-verify") accelerates autoregressive LLM inference by:

1. **Draft phase:** Small, fast "draft model" generates K candidate tokens in parallel
2. **Verification phase:** Large "target model" evaluates all K candidates in one forward pass (parallel scoring)
3. **Accept/reject:** Target model accepts longest prefix matching its own distribution
4. **Fallback:** If all rejected, target generates 1 token normally

**Key insight:** Verification of K tokens takes ~same time as generating 1 token (parallel), so each accepted token is "free."

### 1.2 Performance Factors

**Acceptance rate (α)** = fraction of draft tokens accepted by target model
- **High α (>0.7):** 2-3× speedup (GPU), 1.5-2× (CPU)
- **Low α (<0.5):** No speedup (verification overhead dominates)

**Draft/target model size ratio:**
- **Too small (0.5B draft, 7B target):** Low α (~0.4-0.5) → marginal speedup
- **Optimal (1.5B draft, 7B target):** High α (~0.6-0.7) → 1.8-2.2× speedup
- **Too large (3B draft, 7B target):** Memory contention, slower draft generation

**CPU-specific constraints:**
- Memory bandwidth bottleneck (loading 2 models' weights)
- KV cache duplication (draft + target caches)
- Sequential execution (no batched verification on CPU)

### 1.3 CPU vs GPU Performance

| Metric | GPU | CPU (expected) |
|--------|-----|----------------|
| Speedup (optimal α) | 2-3× | 1.5-2× |
| Acceptance rate | 0.65-0.75 | 0.60-0.70 |
| Memory overhead | 1.5-2× | 2-2.5× |
| Bottleneck | Compute | Memory bandwidth |

**Why CPU is slower:**
- GPU: Parallel verification of K tokens is nearly free
- CPU: Memory-bound; loading draft + target weights saturates bandwidth

---

## 2. llama.cpp Speculative Decoding Support

### 2.1 Current Implementation Status (as of August 2026)

Based on GitHub issues analysis:
- **Feature exists:** PRs #27692 (speculative prefill), #27676 (verification-step count tracking)
- **Status:** Experimental, primarily tested on CUDA/OpenCL backends
- **Known issues:**
  - Draft acceptance rate collapse with multi-batch (`-np N`, issue #27572)
  - Device-to-host async race conditions
  - DSV4 CUDA compute buffer sizing regressions (#27680)

**Command-line interface (inferred from issues):**
```bash
llama-server \
  --model models/qwen2.5-3b-q4_k_m.gguf \
  --draft-model models/qwen2.5-0.5b-q4_k_m.gguf \
  --draft-tokens 4 \
  --threads 6 \
  --parallel 1
```

**Parameters:**
- `--draft-model`: Path to small draft model (GGUF format)
- `--draft-tokens`: Number of speculative tokens to generate (K = 3-5 typical)
- `--threads`: CPU threads (use physical cores = 6)

### 2.2 CPU Backend Readiness

**Challenges for CPU implementation:**
1. **No dedicated CPU backend code** (focus on GPU acceleration)
2. **KV cache management:** Draft and target models need separate caches
3. **Memory layout:** Sequential execution means no batched verification speedup
4. **Acceptance rate tuning:** CPU's lower bandwidth may degrade α

**Implementation gaps:**
- CPU-specific scheduling (draft generation → target verification → acceptance)
- Memory-efficient KV cache sharing between models
- Thread affinity for draft/target model isolation

---

## 3. Quantization Frameworks for CPU

### 3.1 GGUF (llama.cpp native)

**Format:** General GPU-Unified Format, designed for CPU/GPU portability  
**Quantization schemes:**
- `q4_k_m`: 4-bit with mixed precision (medium quality, fastest)
- `q5_k_s`: 5-bit small block (better quality, ~15% slower)
- `q8_0`: 8-bit (highest quality, ~2× slower than q4)

**CPU performance (from baseline study):**
- 3B model @ q4_k_m: **7.76 tok/s** (optimal)
- 3B model @ q8_0: 3.64 tok/s
- **Memory bandwidth bound:** Smaller weights = faster decoding

**Advantages:**
- Native llama.cpp support (zero additional dependencies)
- Optimized for CPU (NEON, AVX2, AVX512 kernels)
- Wide model availability (Hugging Face GGUF repos)

**Limitations:**
- Quality degradation at q4 (acceptable for most tasks, verify per use case)
- No activation quantization (weights-only)

### 3.2 GPTQ (AutoGPTQ)

**Format:** GPU-Post-Training Quantization, designed for CUDA inference  
**Quantization:** 4-bit groupwise quantization with activation-aware calibration

**CPU support:**
- ❌ **No optimized CPU kernels** (requires CUDA for fast dequantization)
- **Fallback:** Can load with `transformers` + `auto-gptq`, but runs at fp16 speed (dequantizes to fp16 before matmul)
- **Performance:** Slower than GGUF q4 on CPU due to dequantization overhead

**Verdict for CPU:** **Not recommended** (GPU-only optimization)

### 3.3 AWQ (Activation-aware Weight Quantization)

**Format:** Activation-aware 4-bit quantization with per-channel scaling  
**Quantization:** Optimizes weight distribution based on activation magnitudes

**CPU support:**
- ❌ **No optimized CPU kernels** (requires CUDA for fast inference)
- **Fallback:** Can load with `transformers` + `awq`, but performance is poor on CPU
- **Memory overhead:** Requires calibration data and per-channel scales

**Verdict for CPU:** **Not recommended** (GPU-only optimization)

### 3.4 Quantization Framework Comparison for CPU

| Framework | CPU Performance | Quality | Availability | Verdict |
|-----------|----------------|---------|--------------|---------|
| **GGUF** | ✅ Optimized (7.76 tok/s @ 3B q4) | Good (q4), Excellent (q5) | Wide (Hugging Face) | **Recommended** |
| **GPTQ** | ❌ Slow (dequant overhead) | Excellent | Limited (GPU models) | Skip for CPU |
| **AWQ** | ❌ Slow (no CPU kernels) | Excellent | Limited (GPU models) | Skip for CPU |

**Recommendation:** Use GGUF exclusively for CPU inference; GPTQ/AWQ provide no benefit and add complexity.

---

## 4. Draft Model Selection: Qwen2.5 Family

### 4.1 Candidate Configurations

| Configuration | Draft Model | Target Model | Memory | Expected α | Expected Speedup |
|---------------|-------------|--------------|--------|------------|------------------|
| **C1 (conservative)** | Qwen2.5-0.5B-q4 | Qwen2.5-3B-q4 | ~2.5 GB | 0.45-0.55 | 1.3-1.5× |
| **C2 (balanced)** | Qwen2.5-1.5B-q4 | Qwen2.5-3B-q4 | ~3.5 GB | 0.60-0.70 | 1.7-2.0× |
| **C3 (aggressive)** | Qwen2.5-1.5B-q4 | Qwen2.5-7B-q4 | ~6.5 GB | 0.50-0.60 | 1.5-1.8× |
| **C4 (stretch)** | Qwen2.5-3B-q4 | Qwen2.5-7B-q4 | ~8.5 GB | 0.65-0.75 | 1.9-2.3× |

**Memory calculation:**
- Model weights: draft + target (both q4_k_m)
- KV cache: 2× (draft + target, ~512 context length)
- Working memory: ~1 GB overhead

**Hardware constraint:** 16 GB RAM → C1-C3 feasible, C4 marginal (may swap)

### 4.2 Recommended Configuration: C2 (Balanced)

**Rationale:**
- **1.5B draft is large enough** to achieve α ~0.65 (good acceptance rate)
- **3B target fits memory comfortably** (~3.5 GB total, 12.5 GB free for OS)
- **Expected speedup:** 1.7-2.0× on top of baseline (total: 6.8-8× vs original naive config)

**Model downloads:**
- Draft: `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (q4_k_m, ~1.1 GB)
- Target: `Qwen/Qwen2.5-3B-Instruct-GGUF` (q4_k_m, ~2.1 GB)

---

## 5. Implementation Plan

### 5.1 Phase 1: Validate llama.cpp Speculative Decoding on CPU (Week 1)

**Goal:** Confirm llama.cpp's speculative decoding works on CPU backend

**Tasks:**
1. **Build llama.cpp with speculative decoding enabled**
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   mkdir build && cd build
   cmake .. -DGGML_METAL=OFF -DGGML_ACCELERATE=ON
   cmake --build . --config Release -j 6
   ```

2. **Download models**
   ```bash
   # Draft model (0.5B for initial testing)
   wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
   
   # Target model (3B)
   wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
   ```

3. **Test basic speculative decoding**
   ```bash
   ./llama-server \
     --model models/qwen2.5-3b-instruct-q4_k_m.gguf \
     --draft-model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
     --draft-tokens 4 \
     --threads 6 \
     --parallel 1 \
     --port 8080
   ```

4. **Verify functionality**
   - Use existing `eval_client.py` to benchmark against baseline
   - Check logs for draft acceptance rate
   - Confirm output quality matches baseline (exact-match or high overlap)

**Success criteria:**
- Server starts without errors
- Draft acceptance rate > 0 (logged per request)
- Speedup > 1.0× vs baseline (even marginal proves concept works)

**Fallback:** If CPU backend doesn't support speculative decoding:
- Investigate llama.cpp source (`llama.cpp`, `llama-sampling.cpp`)
- Implement draft-then-verify loop manually (custom wrapper script)
- Contribute CPU backend support upstream

### 5.2 Phase 2: Optimize Draft Model Size (Week 1-2)

**Goal:** Find optimal draft/target size ratio for CPU

**Experiments:**
1. **Draft size sweep** (target = 3B q4):
   - C1: 0.5B draft → measure α, tok/s
   - C2: 1.5B draft → measure α, tok/s
   - C3: 3B draft (self-speculative) → measure α, tok/s

2. **Draft token count sweep** (use best draft model from step 1):
   - `--draft-tokens 2` → measure speedup
   - `--draft-tokens 4` → measure speedup
   - `--draft-tokens 6` → measure speedup

3. **Metrics per configuration:**
   - Acceptance rate (α): extract from llama-server logs
   - tok/s: median over 5 prompts (use `eval_client.py`)
   - Memory usage: `ps aux | grep llama-server` (RSS)
   - TTFT (time to first token): p95 latency

**Benchmark script** (extend `eval_client.py`):
```python
# Add flag to extract acceptance rate from server logs
# Parse llama-server stderr for "draft acceptance: X.XX"
# Report as new metric: "draft_acceptance_rate"
```

**Expected outcome:**
- 1.5B draft achieves highest speedup (~1.8×)
- 4-6 draft tokens optimal (more tokens → lower α, diminishing returns)

### 5.3 Phase 3: Quantization Comparison (Week 2)

**Goal:** Confirm GGUF is optimal; quantify GPTQ/AWQ unsuitability for CPU

**GGUF variants:**
- Baseline (from Phase 1): 3B q4_k_m with 1.5B q4_k_m draft
- Variant 1: 3B q5_k_s with 1.5B q5_k_s draft (quality vs speed tradeoff)
- Variant 2: 3B q8_0 with 1.5B q8_0 draft (best quality, slowest)

**GPTQ/AWQ (negative confirmation):**
- Attempt to load GPTQ model with `transformers` + `auto-gptq` on CPU
- Measure inference speed (expected: slower than GGUF due to dequantization)
- Document why GPU-only quantization doesn't benefit CPU

**Metrics:**
| Config | tok/s | Quality (overlap) | Memory | Verdict |
|--------|-------|-------------------|--------|---------|
| 3B q4 + 1.5B q4 draft | 14-16 | 0.92 | 3.5 GB | **Baseline** |
| 3B q5 + 1.5B q5 draft | 12-14 | 0.95 | 4.0 GB | +Quality, -Speed |
| 3B q8 + 1.5B q8 draft | 7-9 | 0.98 | 5.5 GB | Best quality, slow |
| 3B GPTQ (no draft) | 3-5 | 0.98 | 6.0 GB | ❌ Slower than GGUF |

**Deliverable:** Comparison table showing GGUF q4 + speculative decoding beats all alternatives.

### 5.4 Phase 4: End-to-End Benchmark (Week 3)

**Goal:** Measure compounding speedup from baseline

**Configurations to benchmark:**
1. **Baseline (from original study):** 3B q4_k_m, t=6, no speculation (7.76 tok/s)
2. **Candidate:** 3B q4_k_m + 1.5B q4_k_m draft, t=6, draft_tokens=4

**Benchmark methodology:**
- Use existing `eval_client.py` (5 fixed prompts, temperature=0)
- Run 3 iterations per config (check stability)
- Compare:
  - tok/s improvement
  - TTFT (speculation may increase prefill time)
  - Quality (exact-match or overlap ≥ 0.90)

**Expected results:**
- Speedup: 1.7-2.0× → **13-15 tok/s** (vs 7.76 baseline)
- Total improvement vs naive config: **~6.8× to 8.0×** (1.5 tok/s → 13-15 tok/s)
- Quality: Maintained (overlap ≥ 0.90)

**Stretch goal (if time permits):**
- Test 7B target with 1.5B draft (C3 configuration)
- Expected: Lower α (~0.55) but still 1.5× speedup → 5-6 tok/s (vs 3-4 tok/s 7B baseline)

### 5.5 Phase 5: Documentation & Integration (Week 4)

**Deliverables:**

1. **Implementation code** (`speculative_decoding_implementation/`)
   - `speculative_server.sh`: Wrapper script for llama-server with optimal flags
   - `benchmark_speculative.py`: Extended eval_client.py with acceptance rate tracking
   - `draft_model_optimizer.py`: Auto-tune draft model size and draft_tokens

2. **Benchmark results** (`benchmark_results/`)
   - `baseline_vs_speculative.json`: Raw data (tok/s, α, TTFT)
   - `draft_size_sweep.json`: 0.5B, 1.5B, 3B draft comparison
   - `quantization_comparison.json`: GGUF vs GPTQ/AWQ (negative results)

3. **Analysis notebooks** (`analysis/`)
   - `acceptance_rate_analysis.ipynb`: Plot α vs draft model size
   - `speedup_breakdown.ipynb`: Visualize cumulative speedups (thread → quant → speculation)
   - `memory_profiling.ipynb`: KV cache overhead, memory usage over time

4. **README.md** (portfolio piece)
   - Title: "Advanced LLM Inference Optimization: Speculative Decoding on CPU"
   - Sections:
     - Executive summary (4× → 8× total speedup)
     - Speculative decoding explained (draft-then-verify algorithm)
     - CPU-specific optimization challenges
     - Benchmark results (acceptance rate curves, speedup graphs)
     - Reproduction guide (download models, run scripts)
     - Lessons learned (memory bandwidth constraints, optimal draft ratios)

5. **Integration into existing repo**
   - Merge with `llm-cpu-inference-optimization` GitHub repo
   - Link from original study's results.md: "Follow-on: Speculative Decoding"
   - Update repo README with advanced techniques section

---

## 6. Expected Outcomes

### 6.1 Quantitative Targets

| Metric | Baseline | With Speculation | Improvement |
|--------|----------|------------------|-------------|
| **tok/s (3B model)** | 7.76 | 13-15 | 1.7-2.0× |
| **Total vs naive** | 4.0× | 6.8-8.0× | 2× additional |
| **Memory usage** | 2.5 GB | 3.5 GB | +1 GB (draft model) |
| **TTFT (p95)** | 0.85s | 1.2-1.5s | +0.4s (draft overhead) |
| **Quality (overlap)** | 1.0 | ≥0.90 | Acceptable |

### 6.2 Qualitative Insights

1. **CPU speculative decoding is feasible** but gains are lower than GPU (memory-bound)
2. **Optimal draft ratio is ~1:2 to 1:5** (1.5B:3B or 1.5B:7B)
3. **GGUF is the only viable quantization for CPU** (GPTQ/AWQ require GPU)
4. **Draft token count matters:** 4-6 tokens optimal; more tokens → lower α
5. **Acceptance rate is key:** CPU's lower bandwidth may reduce α vs GPU benchmarks

### 6.3 Potential Challenges

1. **llama.cpp CPU backend may not support speculation**
   - Mitigation: Implement custom draft-verify loop in Python (slower but proves concept)
   - Contribution opportunity: Add CPU speculative decoding to llama.cpp upstream

2. **Acceptance rate may be lower than expected (<0.5)**
   - Mitigation: Increase draft model size (1.5B → 3B) or reduce draft_tokens
   - Fallback: Document negative result; explain CPU memory bandwidth constraints

3. **Memory constraints (16 GB limit)**
   - Mitigation: Stick to 3B target (7B + draft may OOM)
   - Test with swap disabled to confirm hard limits

4. **TTFT regression (speculation increases prefill time)**
   - Expected: +0.3-0.5s due to draft model warmup
   - Acceptable for batch/throughput workloads; document for interactive use cases

---

## 7. Implementation Roadmap (4-Week Plan)

### Week 1: Foundation & Validation
- [ ] Build llama.cpp with speculative decoding support
- [ ] Download Qwen2.5 models (0.5B, 1.5B, 3B in q4_k_m)
- [ ] Test basic speculative decoding (C1 config: 0.5B draft, 3B target)
- [ ] Verify functionality: acceptance rate > 0, speedup > 1.0×
- [ ] **Checkpoint:** Working speculative decoding on CPU (proof of concept)

### Week 2: Optimization & Quantization Study
- [ ] Draft size sweep: 0.5B, 1.5B, 3B draft models
- [ ] Draft token count sweep: 2, 4, 6, 8 tokens
- [ ] Identify optimal configuration (expected: 1.5B draft, 4-6 tokens)
- [ ] Quantization comparison: GGUF (q4, q5, q8) vs GPTQ/AWQ attempts
- [ ] **Checkpoint:** Optimal draft config identified, GGUF confirmed best

### Week 3: Benchmarking & Data Collection
- [ ] End-to-end benchmark: baseline vs optimal speculative config
- [ ] Collect detailed metrics: tok/s, α, TTFT, memory usage
- [ ] Run 3 iterations per config (stability check)
- [ ] Stretch: Test 7B target model (if memory permits)
- [ ] **Checkpoint:** Benchmark data collected, speedup validated

### Week 4: Analysis & Documentation
- [ ] Jupyter notebooks: acceptance rate analysis, speedup breakdown
- [ ] Write README.md (portfolio-quality documentation)
- [ ] Create reproduction guide (scripts, model download instructions)
- [ ] Integration: Merge into existing GitHub repo
- [ ] **Deliverable:** Complete research findings + working implementation

---

## 8. Code Structure

```
projects/speculative-decoding-quantization/
├── BRIEF.md                              # Project scope (existing)
├── research_findings.md                  # This document
├── code/
│   ├── speculative_decoding.py           # Draft-verify implementation
│   ├── benchmark_runner.py               # Automated benchmark suite
│   ├── quantization_comparison.py        # GGUF vs GPTQ/AWQ tests
│   └── acceptance_rate_tracker.py        # Parse llama-server logs
├── scripts/
│   ├── build_llama_cpp.sh                # Build llama.cpp with speculation
│   ├── download_models.sh                # Fetch Qwen2.5 GGUF models
│   ├── serve_speculative.sh              # Launch llama-server with draft model
│   └── run_benchmarks.sh                 # Execute all benchmark configs
├── benchmark_results/
│   ├── baseline_vs_speculative.json      # Main comparison data
│   ├── draft_size_sweep.json             # 0.5B, 1.5B, 3B draft results
│   ├── draft_tokens_sweep.json           # Token count optimization
│   └── quantization_comparison.json      # GGUF variants + GPTQ/AWQ
├── analysis/
│   ├── acceptance_rate_analysis.ipynb    # Plot α vs draft model size
│   ├── speedup_breakdown.ipynb           # Cumulative speedup visualization
│   ├── memory_profiling.ipynb            # KV cache overhead analysis
│   └── quality_evaluation.ipynb          # Output quality vs baseline
└── README.md                             # Portfolio documentation
```

---

## 9. Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| llama.cpp CPU backend doesn't support speculation | Medium | High | Implement custom Python wrapper; document limitation |
| Acceptance rate < 0.5 (no speedup) | Low | Medium | Increase draft model size; document negative result |
| Memory OOM with 7B target | Medium | Low | Focus on 3B target; test 7B as stretch goal |
| TTFT regression breaks interactive use case | High | Low | Document tradeoff; recommend for batch workloads |
| GPTQ/AWQ unusable on CPU | High | Low | Expected; confirm and document (negative result is useful) |

---

## 10. Success Criteria

**Minimum viable outcome:**
- ✅ Speculative decoding working on CPU (proof of concept)
- ✅ Measurable speedup > 1.3× (even marginal proves technique works)
- ✅ Documentation of CPU-specific challenges (memory bandwidth, KV cache)

**Target outcome:**
- ✅ 1.7-2.0× speedup with optimal draft config
- ✅ Acceptance rate α ≥ 0.60
- ✅ GGUF confirmed as only viable CPU quantization
- ✅ Complete benchmark suite + reproduction guide

**Stretch outcome:**
- ✅ 2.0-2.5× speedup (approaching GPU efficiency)
- ✅ 7B target model tested (memory-constrained but functional)
- ✅ Custom CPU optimizations contributed to llama.cpp upstream

---

## 11. References & Further Reading

### Academic Papers
- **"Fast Inference from Transformers via Speculative Decoding"** (Leviathan et al., 2023) — Original speculative decoding paper
- **"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"** (Cai et al., 2024) — Alternative approach using multiple draft heads
- **"Lookahead Decoding: Parallel Decoding of Single-Token Predictions"** (Fu et al., 2024) — CPU-friendly parallel decoding variant
- **"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"** (Frantar et al., 2023) — GPTQ quantization method
- **"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"** (Lin et al., 2023) — AWQ quantization method

### Implementation Resources
- llama.cpp GitHub: https://github.com/ggerganov/llama.cpp
- llama.cpp speculative decoding issues: Search "speculative decoding" or "draft model" in issues
- Qwen2.5 GGUF models: https://huggingface.co/Qwen (search "GGUF")
- AutoGPTQ (GPTQ inference): https://github.com/PanQiWei/AutoGPTQ
- AutoAWQ (AWQ inference): https://github.com/casper-hansen/AutoAWQ

### Baseline Study
- Previous work: `/projects/llama-cpp-opt/results/results.md`
- Baseline speedup: 4.0× (6.67 → 26.85 tok/s via thread tuning + q4_k_m)
- Methodology: `eval_client.py` (5 fixed prompts, exact-match validation)

---

## 12. Next Steps

1. **Immediate (Day 1):** Build llama.cpp and download Qwen2.5 models
2. **Week 1 Goal:** Confirm speculative decoding works on CPU (any speedup > 1.0×)
3. **Decision point (end of Week 1):** If llama.cpp doesn't support CPU speculation, pivot to custom implementation
4. **Week 2-3:** Optimize draft config and collect benchmark data
5. **Week 4:** Documentation and GitHub integration

**Contact:** Yuvraj Singh Bhadoria (based on git user)  
**Repository:** `llm-cpu-inference-optimization` (to be updated with findings)

---

**Document Status:** Research phase complete, ready for implementation  
**Next Action:** Execute Week 1 tasks (build llama.cpp, test speculative decoding)
