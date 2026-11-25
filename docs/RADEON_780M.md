# Radeon 780M / 760M (Ryzen 7040 / 8040, gfx1103): build from source on Linux

The Radeon 780M (Phoenix / Hawk Point, RDNA3, `gfx1103`) runs Strata through the same HIP backend as the Radeon
cards in [AMD_HIP.md](AMD_HIP.md). **Status: experimental and opt-in** (`STRATA_EXPERIMENTAL_GFX1103=1`): the port
is measured on one machine (the numbers below), the maintainers have not run one, and the chip is not Strix Halo.
Everything here was measured on a Ryzen 7 8745HS (8 cores, AVX-512), Radeon 780M (12 CUs), 60 GB DDR5, GTT at
50 GiB, Ubuntu kernel 7.0.0-38, `cwsr_enable=0`, engine 0.1.43 built from this tree by hand, Qwen3.8-Flash-Next
GSQ-RCO Q2_0 with the MTP draft layer, 32K context, Int8 KV.

## 1. What the port changed

- **Native dot4.** gfx1103 was outside the `dp4a` guard in `hip_compat/intrinsics.hpp`, so every quantized dot
  product ran a four-step byte loop instead of `v_dot4_i32_iu8`. gfx1103 now compiles the native instruction
  (decode +55% measured: 8.70 -> 13.44 tok/s at a 32K prompt, same greedy token ids as before).
- **gfx11 WMMA kernels.** gfx1103 was outside the `__gfx11xx__` list of the prompt attention, block scorer,
  prompt GEMMs and prompt experts, so those fell back to hipBLAS / portable kernels. It is now in the list and
  the parity tests run on it (`hip_prefill_wmma_gemm_parity`, `hip_prompt_attn_wmma`, `hip_prefill_fused_moe`,
  `hip_prefill_fused_iq`, `qsa_*`, `hip_gemm_f16_io_parity`, `hip_prefill_mmq_parity`).
- **A hipBLASLt table** for gfx1103 + hipBLASLt `100401`: `terminal/tools/hip/gfx1103-hipblaslt-100401.txt` (section 3).
- **The gfx1103 exact speed switches** (section 4), written into the config's `env` by setup for the opt-in install
  (the engine keeps gfx1103 out of its exact-switch arch defaults (the prompt set of section 5 is separate), as it does for gfx1150). Whether the chip leaves the
  `STRATA_EXPERIMENTAL_GFX1103=1` opt-in is the maintainers' call; the numbers here are what they need to make it.

## 2. Toolchain and build

ROCm 10.1 (`rocm/dev-ubuntu-24.04:10.1.0-full`) is what this build and the table were tested with: HIP
7.16.26385, hipBLASLt 1.4.1 (`100401`), clang 24. Its hipBLASLt carries gfx1103 kernels. setup's TheRock index
for `gfx110X-all` has no 10.x (its newest is 7.14.0a20260612); the gfx1103 opt-in keeps setup's nightly pin, and
the wheel route was not exercised by the numbers below - they come from a hand build.

By hand, the steps of [STRIX_HALO.md](STRIX_HALO.md) with the architecture and the paths changed:

```sh
cmake -S engine -B build-780m -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1103 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  "-DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/lib/llvm/amdgcn/bitcode"
cmake --build build-780m -j 12
ctest --test-dir build-780m
```

`ctest` on this machine: 57 passed, 2 skipped (`hip_prefill_hcd_exact_parity`, `hip_prefill_hipblaslt_gemm` -
they run and pass with the table set), 2 environmental failures (`ple_parity` needs the `engine/bench/micro` fixtures,
`platform_memory_test` needs a memlock limit above 8 MiB; this host's hard limit is 8 MiB).

The model: a GSQ-RCO Q2_0 pack (`terminal/tools/iq_pack.py`, the experts read from the GGUF in place), the PLE table on
the SSD (`--ple-io direct`), `--mmap-experts --expert-cache 24576`, `--prefill auto`, MTP from
`terminal/tools/mtp_fetch.py` + `mtp_pack.py` + `mtp_rt.py`. GTT must be raised for this model (`ttm.pages_limit`; setup
changes no host setting).

## 3. The hipBLASLt table

```sh
export STRATA_HIPBLASLT_TUNING=$PWD/tools/hip/gfx1103-hipblaslt-100401.txt
```

setup hands it over automatically when the installed hipBLASLt reports 1.4.1 and Linux `amdgpu.cwsr_enable=0` is
active (#6132, #6574). Without `amdgpu.cwsr_enable=0` on the kernel command line, hipBLASLt warm-up on `gfx1103` can
trigger a KFD/MES `REMOVE_QUEUE` reset, so auto-selection skips the table and falls back to plain hipBLAS until
`amdgpu.cwsr_enable=0` is set (or `STRATA_HIPBLASLT_TABLE=on` forces it). On `gfx1103` with hipBLASLt 1.2.0 (`100200`),
a minimal 6-row zero-workspace subset covering the 3 main prompt GEMM shapes (f16/bf16) captures essentially all of the
whole-model prompt speedup (~217-225 tok/s vs 165-212 tok/s plain hipBLAS on cold 56K prompts; #6574).

The shipped `gfx1103-hipblaslt-100401.txt` table was calibrated with
`tools/hip/tune_hipblaslt` over the gfx1150 table's 144 GEMM cases at T = 64..16384 (the same case list the 890M
run used), two passes, 32 MiB workspace, each row the solution with the lowest median of all six repetitions: 143
rows (the `bf16 N=1 K=2560 T=64` shape has no candidate faster than plain hipBLAS and is left out), 1.05-23.30x per
GEMM, geometric mean 4.95x. `hip_prefill_hipblaslt_gemm` reports `fallbacks=0`; `hip_prefill_hcd_exact_parity`
skips because the table picks no 1176 / 1177 for the HC down GEMM (the gfx1150 table gets the same skip).

## 4. The exact speed switches

With `STRATA_EXPERIMENTAL_GFX1103=1` setup writes the 17 exact switches below into the config's `env`, the same way
it writes the hipBLASLt table; turning the opt-in off takes them out again. The engine keeps gfx1103 out of its exact-switch
arch defaults (`arch_default_env("gfx1103")` stays empty, checked by `arch_defaults_test`). Measured on the 780M:
+13% to +15% prompt with the table (183-186 -> 211-212 tok/s at 4K) and +3.2% decode (16.63 -> 17.17 tok/s, 640
greedy tokens, 3 of 3 pairs), with the greedy token ids unchanged. They are the gfx1151 table less
`STRATA_HCD_EXACT` (which needs hipBLASLt solutions 1176 / 1177; the gfx1103 table picks others).

| Switch | What it does |
|---|---|
| `STRATA_GDN_HEAD`, `STRATA_GDN_PP=2`, `STRATA_GDN_CONVL2`, `STRATA_GDN_NOY`, `STRATA_CVEC_FUSE` | the prompt's gated-delta-net recurrence and the steered layer's fused write |
| `STRATA_PF_PAD` | padded GEMM row strides (acts with `STRATA_PF_GEMM`; inert without it) |
| `STRATA_Q8_PACKED`, `STRATA_Q6_PACKED`, `STRATA_MMVF_ROWS`, `STRATA_ATTN_LANECELL` | decode layouts and kernels (packed Q8_0 / Q6_K weights, 4-row BF16 GEMV, one-cell-per-thread attention scores) |
| `STRATA_EXPERT_V2`, `STRATA_TSUM`, `STRATA_LFUSE`, `STRATA_GDN_SPLIT`, `STRATA_QFUSE`, `STRATA_PLE_BATCH`, `STRATA_SH_STREAM` | grouped decode experts, butterfly warp sums, fewer launches, the GDN step over four blocks per head, producer-written activation images, batched PLE projections, the shared expert on its own stream |

## 5. The matrix-core prompt set (on by default on the 780M; it changes bits)

On by default on the Radeon 780M (`gfx1103`, integrated, Linux; AMD_HIP.md, "Integrated GPUs"): the engine sets `STRATA_HIP_WMMA`, `STRATA_PF_GEMM`,
`STRATA_PA_FAST`, `STRATA_SELECT_WMMA` and `STRATA_HC_UPMIX` itself. `STRATA_IGPU_PROMPT_SET=0` turns the whole set off, `NAME=0` one switch. The 880M / 890M
(`gfx1150`) and Windows keep it off (no hardware / not tested); `STRATA_IGPU_PROMPT_SET=1` turns it on there. `STRATA_PF_FUSED` (the prompt experts on Strata's own int8 kernels instead of llama.cpp's MMQ) is on by itself on the 780M under Linux; `STRATA_PF_FUSED=0` keeps MMQ in a build that has it (the release builds of 0.2.16 and later have none: `=0` there runs the prompt experts on the native int8 kernels instead, #6251). Q2_0, 3 interleaved rounds: prompt x1.30 at 2K tokens and x1.13 at 8K, no change at 300 and 1,000 tokens; forced-token KL against the FP16 path 0.0060 against 0.0054 for MMQ (24 prompts, 64 tokens each, ratio 1.10, 95% CI 0.93 to 1.37).

Measured (021-minipc, P2006, Q2_0, fresh engine per run, 10 interleaved rounds, 6 complete pairs): prompt 1K 98.5 -> 157.3 tok/s (x1.60, 6/6 pairs faster),
4K 115.1 -> 233.3 (x2.03, 6/6), decode 22.02 -> 21.99 (code) and 15.09 -> 15.05 (story) (x0.999), greedy token ids identical in 6 of 6 pairs.

The 16-bit prompt GEMMs run FP16 in and out by themselves on the 780M under Linux (`STRATA_HIP_PROMPT_F16`; AMD_HIP.md has the numbers and the check against the FP16 path); `STRATA_HIP_PROMPT_F16=0` keeps the BF16 route.

**GPU resets.** The 780M's current Ubuntu firmware (gc 11.0.1 MES) can still reset the GPU now and then, with or without this set: when the kernel moves memory that the
GPU has registered, the process's queues are evicted, and a kernel that sits in a barrier at that moment never answers. The engine recovers
("device wedged, but recovered through reset") but the running request fails. In the same runs the default engine crashed 4 of 10 times and the set 0 of 10, so it did
not add resets. Until AMD fixes the firmware a mini PC may still see them; the engine does not touch the firmware. A newer `linux-firmware` is the thing to try.
The prompt levers need the table to pay: with the table, the WMMA set
read 248 tok/s at 4K against 186 without it, and at 32K 272 against 178. No KL was measured on this chip.

```sh
export STRATA_HIP_WMMA=1      # prompt attention on matrix cores
export STRATA_PF_GEMM=1       # the prompt's FP16 projections on the WMMA GEMM
export STRATA_PA_FAST=1       # prompt attention with single FP16 q and p
export STRATA_SELECT_WMMA=1   # the block scorer on matrix cores
export STRATA_PF_FUSED=1      # the prompt experts on the fused int8 kernels (on by default; =0 keeps MMQ where a build has it, the native int8 kernels in the release builds)
export STRATA_HC_UPMIX=1      # the hyper-connection up projection with its epilogue
```

## 6. Engine flags

`--spec 2` is the best MTP window on this chip: 15.65 tok/s decode at a 4K prompt against 14.18 at `--spec 4` and
11.16 at `--spec 6` (greedy, 256 tokens, medians of three interleaved restarts). `--lookup-chain 4` adds +2.3% (3
of 3 pairs, byte-identical greedy answers) and setup's `--spec-min-p 0.5` is 4.5% slower here than the default;
`--mtp-q4` is neutral and `--mtp-window 4096` measured +2% at 32K in two pairs (weak). The raw runs are in the
results folder.

## 7. Measured

bench/bench_vs_llama.py, 256 tokens out, temperature 0 and 0.7, interleaved; each cell one round (medians need
more rounds - treat these as one-machine samples):

| arm | cell | prompt tok/s | decode tok/s | TTFT s |
| --- | --- | ---: | ---: | ---: |
| plain (no table, no opt-ins) | 4K | 90 | 13.1 | 44.9 |
| plain (no table, no opt-ins) | 32K | 90 | 13.1 | 351 |
| table + all opt-ins | 4K | 277 | 13.9 | 14.6 |
| table + WMMA set | 32K | 272 | 14.3 | 118 |

A context ladder on the shipping configuration (131K context, `--spec 2 --lookup-chain 4`) with fresh prompts from
this checkout's own documentation, 128 greedy tokens each: prefill holds ~310 tok/s to 32K (276.7 at 4K, 314.3 /
314.6 at 8K / 16K, 308.7 at 32K), then 258.2 at 64K, 215.5 at 96K and 198.5-221.4 at 128K across two runs - that is
the average over the full read, and the last ~14K tokens at 128K depth read at ~206 tok/s with the prefix cached.
Decode holds ~16 tok/s to 32K, then 12.1 at 64K and 9.8-9.9 at 96-128K (draft acceptance stays ~70%, so it is
context cost, not the drafter). Memory stays flat: 38.1 of 50 GiB GTT at 131K. Raw:
[docs/bench-results/2026-10-12-community-gfx1103](bench-results/2026-10-12-community-gfx1103/README.md).

## 7a. Inkling-2-Mini on the 780M

`strata-inkling` is a separate engine: none of the switches in sections 4 and 5, and not the hipBLASLt table, apply to it.
It keeps hipBLASLt off on RDNA3 (`STRATA_HIPBLASLT=1` read prompts 26% slower here: 252 -> 186 tok/s at 4K, 2 of 2 runs).

On gfx1103 it uses one-token decode kernels by itself (`STRATA_INK_DECODE_780M`; `=0` turns them off, `=1` turns them on
for another HIP card, not measured there):

- decode attention runs the 3 query heads of each KV head in one block, so each key and value is read once, not 3 times;
- the expert gate/up kernel picks its codec once per block, at compile time, and does 4 rows per wave;
- the expert down kernel has its sizes at compile time and does 4 rows per wave;
- the unembed reads the hidden row into LDS once.

The arithmetic is the same: the greedy token ids were identical (SQ-15: 2 prompts x 256 tokens and a 12K-token prompt
x 128; SQ-10: 1 x 256).

`STRATA_INK_TRUNK_I8=1` (opt-in, off by default) makes the one-token step read 8-bit copies
of the attention projection and output weights (int8 with one fp16 scale per 32 values, made at load: about 0.65 GB more
memory). The prompt and the several-token steps keep BF16. This changes the output. Measured against the one-token BF16
engine on 4 real texts (this repository's docs and code), 1,500 tokens each, fed one token at a time:

| compared with the BF16 one-token step | mean KL | top-1 agreement | perplexity |
| --- | ---: | ---: | ---: |
| `STRATA_INK_TRUNK_I8=1` | 0.0039 | 97.8% | -0.3% to +0.3% |
| the prompt (batched) path, for scale | 0.0031 | 98.3% | -0.05% to +0.8% |
| the SQ-10 build, for scale | 0.061 | 91.5% | +0.2% to +3.6% |

Measured on a Ryzen 7 255, Radeon 780M, 60 GB DDR5, Fedora kernel 7.3.4, `cwsr_enable=0`, ROCm 10.1 image,
`strata-inkling --bench`, `--draft off`, one slot, SQ-15, 128 tokens out, 3 interleaved rounds (all 3 the same to 0.1 tok/s):

| prompt | engine 0.1.67 | decode kernels | + `STRATA_INK_TRUNK_I8=1` |
| ---: | ---: | ---: | ---: |
| 2K | 10.8 tok/s | 13.0 tok/s | 15.6 tok/s |
| 32K | 7.3 tok/s | 10.6 tok/s | 12.3 tok/s |

SQ-10 at 2K before the attention and gate/up changes: 10.8 -> 12.3 tok/s (1 pair). Prompt speed did not change (about
259 tok/s at 2K). `--max-context` did not change decode speed (32K, 128K and 256K). With `STRATA_HIP_PROF=2` the SQ-15
step at 2K went from 92 ms to 77 ms (63.7 ms with the 8-bit trunk); at 32K the attention went from 50.2 to 18.3 ms.
Tried and slower here: loading the next key ahead in the attention, more rows per wave in the output projection, and
an occupancy cap on the expert down kernel.

**DSpark drafting** (`--drafting dspark`) was 7-18% slower here: checking a draft tree cost 4-5x a plain step (6 nodes:
321 ms against 77). On gfx1103 the check now uses the same switch's kernels: tree attention with the 3 query heads of a KV
head in one block, the experts in expert order (the rows that share an expert read it from L2), the verify unembed and the
drafter's vocabulary screen with their sizes at compile time. A 3-node check went from 183 to 123 ms and a 6-node one from
321 to 197 ms; the screen from 9.5 to 6.3 ms. After a probe that does not pay, the server now waits 40, 80, 160 and then
320 plain steps before the next one (it was always 40). The text is the same as without drafting (each row is the
one-token step's value, #6671): the three replies below were identical with drafting on and off.

Server, SQ-15, greedy, `reasoning_effort` none, one request each (tok/s):

| request | drafting off | dspark before | dspark now |
| --- | ---: | ---: | ---: |
| code (an LRU cache class with tests, 600 tokens) | 13.6 | 12.0 | 16.2 |
| list to JSON (600 tokens) | 13.2 | 14.4 | 21.6 |
| short story (301 tokens) | 13.6 | 11.2 | 12.8 |

**With `STRATA_INK_TRUNK_I8=1`** the checks read the same 8-bit copies with the one-token step's arithmetic per row, so
drafting now works in the 8-bit mode (it was turned off there) and its text is the 8-bit mode's text without drafting
(code, story and list replies identical, story repeated 5 times). Server, SQ-15, one request each: code 16.6 (8-bit, no
drafting) / 15.5 (8-bit + dspark) tok/s, list to JSON 16.0 / 21.3, story 16.6 / 15.2. A 3-node check: 111 ms (BF16: 122).

**Unsloth GGUF builds with `STRATA_INK_TRUNK_I8=1`**: the attention, output projection and unembed matrices that the file
stores as Q8_0 (49 in UD-Q4_K_XL) are used as stored, the file's own int8 values and scales, instead of being re-quantized
from the BF16 expansion. UD-Q4_K_XL, 2K prompt, 3 interleaved rounds: 11.1-11.3 tok/s (BF16) -> 14.2-14.3 (13.0-13.1 with
the re-quantized copies). Against the BF16 path on the 4 texts above: mean KL 0.0031, perplexity -0.4% to +0.3%.

**Reading long prompts** (agents). The engine reads a prompt in batches of at most 512 tokens on the 780M under Linux (a
guard that keeps each GPU submission well under a second), so each expert gets about 32 tokens per batch and the grouped
expert kernels' 128-token tiles run three quarters empty. `STRATA_INK_PREFILL_CHUNK=2048` together with the engine flag
`--chunk 2048` (in Docker: `ENGINE_ARGS=--chunk 2048`) reads 2,048 tokens per batch; the engine then flushes every 2
layer visits instead of 8, so a submission stays as long as before. The prompt attention also runs the 3 query heads of
a KV head in one block now (on by default with `STRATA_INK_DECODE_780M`; the logits of a 3,000-token prompt were
byte-identical to before). SQ-15, `--bench`, one run each (tok/s):

| prompt | before | 2,048-token batches | + prompt attention change |
| ---: | ---: | ---: | ---: |
| 8K | 242 | 299-305 | 317 |
| 32K | 208 | 255-256 | 281 |

Through the server (8-bit trunk on), a 12,769-token prompt took 59.8 s before and 42.5 s with both (214 -> 301 tok/s).
The ring check (`--ring-check`, 12,000 tokens in 2,048-token batches against one token at a time) passed: KL median
1.1e-6, top-1 100%. No GPU reset in the kernel log over these runs (one machine, `cwsr_enable=0`). 1,024 gave 281 tok/s
at 8K, 4,096 gave 303-306.

The prompt's grouped expert kernels then got a pipelined K loop (same switch): for SF4, S2_K, S3_K and S4_K the next
step's raw token words and weight bytes are loaded into registers before the current step's matrix products and
unpacked into a second LDS stage after them, one barrier per step instead of two; the codec is fixed per block at compile
time. Builds that skipped one part at a time had put the weight unpacking at about 50% of the gate/up kernel and the
token-row reads at about 35%. The unpacking formulas are the same, so the logits of a 3,000-token prompt stayed
byte-identical (and SQ-10's greedy ids). Per 2,048-token batch: expert gate/up 2,271 -> 1,740 ms, down 1,111 -> 819 ms;
prompt 8K 305-312 -> 348-361 tok/s, 32K 281 -> 313-315 tok/s; a 12.7K-token prompt through the server 42.6 s -> 38.3 s.
Not kept (measured slower or no faster): one launch per codec, 512-thread expert blocks, the prompt attention with K and V
in separate buffers and 2 barriers per tile instead of 4 (same output, no faster). At the end of a 16K prompt the
attention's K/V reads are about 40% of its time and its softmax about 19%; the kernel is at 256 VGPRs and 63.5 of
64 KB LDS, so prefetching the next tile needs a different layout first.

**Repeatable logits for several tokens.** Reading the same 3,000-token prompt ten times gave five different logit dumps
(6 of 10 alike). The rows that differed were single rows, never their followers, off by at most about 0.05, and two
launches of the batched unembed (`k_unembed_softcap`) on identical inputs in one process gave a different row about once
in 1,500 to 15,000 rows. The final hidden rows were identical every time. Pinning the clocks, deterministic
expert buckets and a second process on the GPU did not remove it. On gfx1103 the several-token unembed now runs on
`k_unembed_softcap_rows_sk` (each token's row read into LDS once, the same arithmetic): the dumps came out identical
in 7 of 7 runs, and repeated launches still differed now and then, about 10 times less often. The one-token kernels the
server decodes with did not differ in 1,596 launches, which is not enough to say they never do. A low rate of read
errors on this machine (non-ECC SO-DIMMs of two different builds) fits these results; a memory test would tell.

Also measured, no change kept: GPU clocks forced to `high` (prompt +2%, decode unchanged: decode is bound by DDR5-5600);
FP8 KV at 32K (decode 10.4 -> 10.5 tok/s, prompt 5% slower; k8v8 is not supported on HIP).

What still costs the most in a check is the experts: rows that pick different experts have to read them. The drafter's
own 8-row matrix products (about 9 ms per draft) did not get faster with fewer barriers. The server's greedy output of a
long reply (900+ tokens) can differ from run to run with drafting off too (seen with the engine before these changes;
the command-line generation repeats exactly): not caused by drafting, not yet explained.

## 8. Not tested

- The TheRock-wheel install for gfx1103 (the hand build above is what was measured).
- The opt-in set beyond one round; KL of any bit-changing switch; other models; contexts past 131K.
- CUDA and SYCL were not built here (no toolchains on this box); the added gfx1103 macros are undefined for them.
- Reset stability looks good but is one machine: a 2 h 07 m soak (`terminal/tools/serve_load.py`, 924 requests) and the
  reset-prone combination from 7.10 (partial expert cache + `STRATA_IO_PREFETCH=1` + the adaptive tier) both ran
  with zero `amdgpu` resets in the kernel log. The known 780M issue ([DETAILS.md](DETAILS.md)) was seen on ROCm
  7.10; with `cwsr_enable=0` and this stack it did not reproduce.
