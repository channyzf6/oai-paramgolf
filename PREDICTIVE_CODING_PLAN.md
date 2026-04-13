# Predictive Coding Transformer + Adaptive Early Exit

## Submission Plan for OpenAI Parameter Golf

**Author:** channyzf6
**Date:** April 13, 2026
**Target:** Top 10 on merged leaderboard + OpenAI novelty recognition

---

## Thesis

Everyone in the competition is stacking the same techniques (SP8192, depth recurrence, parallel residuals, TTT, GPTQ). We differentiate with two **completely unexplored** innovations:

1. **Predictive Coding** — Each transformer layer predicts the next layer's hidden state via auxiliary losses, providing direct gradient supervision to every layer instead of relying on diluted backprop from the final LM head. Faster convergence = better model in 10 minutes.

2. **Per-Token Adaptive Early Exit** — Not all tokens need full depth. Easy tokens ("the", "of") exit early; hard tokens (rare words, complex syntax) get full depth. This improves throughput → more training steps → better model.

These two innovations are **synergistic**: predictive coding makes intermediate representations independently useful, which is exactly what early exit requires. Without predictive coding, early layers produce representations too poor for direct prediction.

**Neither technique has been attempted in the competition** (verified via GitHub search across 1500+ PRs).

---

## Competitive Landscape (as of April 13, 2026)

### Merged Leaderboard (Top 10)

| Rank | BPB | Author | PR | Key Techniques |
|------|-----|--------|-----|----------------|
| 1 | **1.0810** | bigbag | #1493 | SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT |
| 2 | 1.0822 | aryanbhosale | #1477 | SP8192 + Parallel Residuals + Score-First TTT |
| 3 | 1.0828 | dexhunter | #1413 | SP8192 + QK-Gain 5 + Legal Score-First TTT |
| 4 | 1.0835 | Robby Sneiderman | #1412 | SP8192 + Parallel Residuals + Hessian-Aware SDClip |
| 5 | 1.0856 | Kevin Clark | #1394 | SP8192 + GPTQ Embeddings + Depth Recurrence + SDClip |
| 6 | 1.0897 | aryanbhosale | #1334 | SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R |
| 7 | 1.0912 | dexhunter | #1285 | MuonEq-R + Depth Recurrence + WD=0.090 + All-Int6 GPTQ |
| 8 | 1.0979 | Kevin Clark | #1218 | 4096-Vocab + Larger Model + High WD |
| 9 | 1.1063 | Marko Sisovic | #1204 | Parallel Residuals + Mini Depth Recurrence |
| 10 | **1.1147** | abaybektursun | #1019 | 11L AR Self-Gen GPTQ + XSA (our previous best) |

### Key Pending PRs

| PR | BPB | Status | Notes |
|----|-----|--------|-------|
| #1585 | 1.0639 | OPEN | Casefold tokenizer (legality TBD) |
| #1517 | 1.0632 | OPEN | Depth Recurrence + Pre-Quant TTT (18ep) |
| #1487 | 1.0600 | OPEN | Recur345 + Par7 + Pre-Quant TTT 10ep |
| #1586 | 1.0749 | OPEN | Per-Layer Adaptive GPTQ Clip |
| #1560 | 1.0741 | OPEN | VarLen Attention + Triton Fused MLP + Doc-TTT |
| #1555 | 1.0764 | OPEN | TMA Megakernel |

### What's Been Explored (and Failed) for Novel Approaches

| Approach | PRs | Outcome |
|----------|-----|---------|
| MAML/Meta-TTT | #873, #494, #296, #384, #1501, #1502 | **Ceiling hit.** PR #1502 proved TTT is architecture-limited, not initialization-limited. |
| Self-Distillation | #1029, #896 | **Net negative.** I/O overhead costs ~280 training steps. |
| JEPA | #832, #1312, #1556 | **Mostly negative.** PR #1556: gains were torch.compile confound. |
| MoE | #1538, #660, #480, #981 | **Not competitive.** Sparse routing collapses under 16MB. |
| GDN/SSM Hybrid | #1576, #1545, #1355, #1574 | **PR #1576 had BPB bug** (double-counted space bytes, real score ~1.18). Best valid SSM: 1.15. |

### What's Truly Unexplored (Zero PRs)

1. **Inter-layer predictive coding** (our approach)
2. **Per-token adaptive early exit / halting** (our approach)
3. **Information bottleneck regularization**
4. **Progressive model growing**
5. **Megakernels** (OpenAI wishlist, only 1 attempt)

---

## Architecture

### Base Stack (Rebase onto SOTA ~1.08)

We start from the proven April 2026 SOTA stack (PR #1493 by bigbag):

| Component | Setting | Source |
|-----------|---------|--------|
| Tokenizer | SP8192 | PR #1394 |
| Layers | 11 physical (512d, 8 heads, 4 KV heads) | Baseline |
| Depth Recurrence | Loop layers 3-5 three times (17 virtual layers), activate at 35% | PR #1493 |
| Parallel Residuals | Layers 7-10 (GPT-J style) | PR #1412 |
| MLP | 4x expansion (2048 hidden), LeakyReLU(0.5)^2 | PR #1493 |
| Optimizer | MuonEq-R (row-normalized Muon) | PR #1285 |
| QK-Gain | 5.0-5.25 | PR #1493 |
| Quantization | GPTQ SDClip: int6 matrices (k=12.85), int8 embeddings (k=20) | PR #1394 |
| Compression | Brotli + byte-shuffle (or LZMA preset=9) | PR #1493 |
| Weight Avg | EMA decay=0.9965 | PR #1493 |
| Warmdown | Fraction 0.72 of total steps | PR #1493 |
| Weight Decay | 0.085-0.095 (Muon), 0.095 (Adam) | PR #1493 |
| TTT | Score-first, SGD(lr=0.005, mom=0.9), 3 epochs, cosine decay | PR #1493 |
| Attention | XSA on all layers + SmearGate + BigramHash | PR #1019 |
| Skip Connections | U-Net encoder-decoder with learned gates | Baseline |
| Eval | Sliding window, stride 64 | Standard |

### Novel Addition 1: Predictive Coding Auxiliary Losses

```
                    Standard Transformer          Predictive Coding Transformer
                    ════════════════════          ═════════════════════════════

                    Layer 0                       Layer 0
                      │                             │
                      ▼                             ├──► pred_head_0 ──► MSE(pred, Layer 1 output)
                    Layer 1                       Layer 1
                      │                             │
                      ▼                             ├──► pred_head_1 ──► MSE(pred, Layer 2 output)
                    Layer 2                       Layer 2
                      │                             │
                      ...                           ...
                      │                             │
                      ▼                             ▼
                    Layer 10                      Layer 10
                      │                             │
                      ▼                             ▼
                    LM Head ──► CE Loss           LM Head ──► CE Loss
                                                    +
                                                  α * Σ(prediction_errors)
```

**Implementation details:**

```python
class PredictiveCodingHead(nn.Module):
    """Predicts next layer's hidden state from current layer's output."""
    def __init__(self, d_model):
        super().__init__()
        # Single linear projection — minimal parameters
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, current_hidden, next_hidden):
        """Returns cosine prediction loss (scale-invariant)."""
        pred = self.proj(current_hidden.detach())  # stop gradient from pred back to encoder
        # Cosine similarity loss — invariant to representation scale
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(next_hidden.detach(), dim=-1)
        return 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()
```

**Key design choices:**
- **Cosine loss** (not MSE): invariant to representation scale, which changes across layers
- **Stop-gradient on both sides**: the prediction head learns to predict, but doesn't distort the representations. Current layer and next layer learn from the main CE loss only. The auxiliary loss only trains the prediction heads, which provide richer gradient signals during backprop.
- **Actually, reconsider**: We want the auxiliary loss to also provide gradients to the current layer. So we should NOT detach `current_hidden`. We SHOULD detach `next_hidden` (the target). This way:
  - Current layer gets gradient from both CE loss AND prediction loss (richer signal)
  - Next layer's representation is the fixed target (avoids representation collapse)
  - Prediction head learns to bridge between layers

**Corrected implementation:**

```python
def forward(self, current_hidden, next_hidden):
    pred = self.proj(current_hidden)  # gradients flow back to current layer
    target = next_hidden.detach()     # target is fixed (no collapse)
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()
```

**Hyperparameters:**
- `PC_ALPHA = 0.1` — weight of auxiliary loss (sweep: 0.01, 0.05, 0.1, 0.2, 0.5)
- `PC_START_LAYER = 0` — first layer with prediction head
- `PC_END_LAYER = 9` — last layer with prediction head (layer 10 uses LM head)
- `PC_WARMUP_STEPS = 200` — linearly ramp α from 0 to target (let representations stabilize first)
- `PC_DECAY = True` — decay α during warmdown (predictions become less useful as model converges)

**Parameter cost:**
- 10 prediction heads x 512x512 = 2.62M params (all training-only, discarded before quantization)
- Runtime overhead: ~2-3% (10 extra linear forward passes, no backward through them during eval)
- **Artifact cost: ZERO** (prediction heads are not saved)

**Why this should work for Parameter Golf specifically:**
- The competition is convergence-limited (only ~7000 steps in 10 min). Any technique that accelerates convergence directly improves the final BPB.
- With 11 layers, gradients to layer 0 pass through 10 matrix multiplications. Predictive coding gives layer 0 a gradient signal from layer 1's state, which is only 1 hop away.
- The technique is compatible with ALL existing techniques (XSA, depth recurrence, parallel residuals, etc.)

### Novel Addition 2: Per-Token Adaptive Early Exit

```
                    ┌─────────────────────────────────────────────────┐
                    │  Forward Pass with Adaptive Exit                │
                    │                                                 │
                    │  x = embed(tokens)                              │
                    │                                                 │
                    │  for i, layer in enumerate(layers):             │
                    │      x = layer(x)                               │
                    │                                                 │
                    │      if i >= MIN_EXIT_LAYER:                    │
                    │          logits_i = lm_head(norm(x))            │
                    │          entropy_i = -Σ(p * log p)              │
                    │                                                 │
                    │          if entropy_i < EXIT_THRESHOLD:         │
                    │              ┌─── EXIT HERE ───┐                │
                    │              │ Return logits_i  │  ◄── easy     │
                    │              │ for this token   │      tokens   │
                    │              └──────────────────┘                │
                    │                                                 │
                    │  return lm_head(norm(x))  ◄── hard tokens      │
                    │                           use full depth        │
                    └─────────────────────────────────────────────────┘
```

**Implementation strategy — training vs eval:**

During **training**, early exit is tricky (need gradients through all layers). We use a weighted mixture approach:

```python
def forward_with_early_exit(self, x, targets, training=True):
    all_logits = []
    for i, layer in enumerate(self.layers):
        x = layer(x)
        if i >= self.min_exit_layer:
            logits_i = self.lm_head(self.final_norm(x))
            all_logits.append((i, logits_i))

    if training:
        # Weighted sum of losses from all exit points
        # Later layers get higher weight (they're better)
        total_loss = 0
        for depth, logits in all_logits:
            weight = (depth + 1) / sum(d + 1 for d, _ in all_logits)
            loss_i = F.cross_entropy(logits.view(-1, V), targets.view(-1))
            total_loss += weight * loss_i
        return total_loss
    else:
        # During eval: per-token exit based on entropy
        # Use the earliest confident prediction for each token
        final_logits = all_logits[-1][1]  # start with full-depth
        for depth, logits in all_logits[:-1]:
            probs = logits.softmax(dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1)  # [B, T]
            confident = entropy < self.exit_threshold      # [B, T]
            # Replace full-depth logits with early-exit logits where confident
            final_logits = torch.where(confident.unsqueeze(-1), logits, final_logits)
        return final_logits
```

**But wait — there's a simpler and better approach for training:**

Instead of per-token exit during training (which is complex), we use the **auxiliary classifier** approach:
- Every `EXIT_INTERVAL` layers, compute CE loss from intermediate logits
- Add to main loss with a small weight
- This trains ALL layers to produce useful representations for prediction
- During **eval only**, we actually do per-token early exit

This is exactly what predictive coding already gives us! The predictive coding heads train each layer to produce useful representations. We just need to add intermediate classifier heads (sharing the LM head) and exit at eval time.

**Combined architecture (Predictive Coding + Early Exit):**

```python
# Training: auxiliary losses at each layer (predictive coding + intermediate CE)
loss = main_ce_loss
for i in range(num_layers - 1):
    # Predictive coding: predict next layer's hidden state
    loss += pc_alpha * pred_heads[i](hidden[i], hidden[i+1])
    # Intermediate CE: train each layer's representation for direct prediction
    if i >= min_exit_layer:
        exit_logits = lm_head(final_norm(hidden[i]))
        loss += exit_alpha * F.cross_entropy(exit_logits, targets) * (i / num_layers)

# Eval: per-token early exit using entropy threshold
```

**Hyperparameters:**
- `MIN_EXIT_LAYER = 5` — don't exit before layer 5 (too shallow for useful predictions)
- `EXIT_THRESHOLD = 1.5` — entropy threshold for confident exit (sweep: 0.5, 1.0, 1.5, 2.0, 2.5)
- `EXIT_ALPHA = 0.05` — weight of intermediate CE losses during training
- Early exit is **eval-only** for throughput — during training, run all layers but add auxiliary losses

**Expected throughput impact:**
- During eval: ~15-25% fewer forward-pass FLOPs (easy tokens exit at layer 5-7 instead of 17)
- During training: ~2-3% overhead from intermediate logit computation
- Net: faster eval → more time budget for TTT → better final BPB

**Synergy with predictive coding:**
- Predictive coding trains each layer to produce a representation that accurately predicts the next layer's output
- This means intermediate layers develop independently-useful features
- Which means intermediate LM-head predictions are more accurate
- Which means early exit works better (more tokens can confidently exit early)
- Without predictive coding, early exit from layer 5 would produce garbage predictions

---

## Implementation Plan

### Phase 1: Rebase onto SOTA Stack (Days 1-3)

**Goal:** Reproduce ~1.08 BPB using the proven April SOTA techniques.

We need to upgrade our March 25 train_gpt.py (1.1147 BPB) with:

| Technique | Our Current | Target | Source |
|-----------|-------------|--------|--------|
| Tokenizer | SP1024 | SP8192 | PR #1394 |
| MLP expansion | 3x (1536) | 4x (2048) | PR #1493 |
| Depth recurrence | None | Loop L3-5, 3x, at 35% | PR #1493 |
| Parallel residuals | None | L7+ GPT-J style | PR #1412 |
| Optimizer | Parallel Muon | MuonEq-R | PR #1285 |
| QK-Gain | 1.5 | 5.0-5.25 | PR #1493 |
| Weight decay | 0.04 | 0.085-0.095 | PR #1493 |
| Quantization | int6 GPTQ + lzma | SDClip int6 (k=12.85) + int8 emb (k=20) | PR #1394 |
| Compression | LZMA preset=9 | Brotli + byte-shuffle | PR #1493 |
| EMA decay | 0.997 | 0.9965 | PR #1493 |
| Warmdown | 3500 iters (~18%) | Fraction 0.72 | PR #1493 |
| TTT | None (dropped) | Score-first SGD, 3 epochs | PR #1493 |
| BigramHash | 3072x112 | Keep or tune | Our work |

**Steps:**

1. Download SP8192 dataset:
   ```bash
   python3 data/cached_challenge_fineweb.py --variant sp8192
   ```

2. Modify train_gpt.py with all SOTA techniques (reference PRs #1493, #1477, #1412, #1394)

3. Smoke test on 1xH100 (200 steps): verify loss decreases, no NaN

4. Full run on 8xH100: verify ~1.08 BPB, <600s, <16MB artifact

**Deliverable:** Reproduction of ~1.08 BPB. This is our safety net — even without novel techniques, this is a top-10 submission.

### Phase 2: Predictive Coding Integration (Days 3-5)

**Goal:** Add predictive coding auxiliary losses and measure impact.

**Steps:**

1. **Add PredictiveCodingHead modules** (10 heads, one per layer 0-9)
   - Single 512x512 linear projection per head
   - Cosine similarity loss: `1 - cos_sim(proj(h_i), sg(h_{i+1}))`
   - Stop-gradient on target (next layer's output) to prevent collapse

2. **Modify GPT.forward()**
   - Collect hidden states at each layer boundary
   - Compute predictive coding loss for layers 0-9
   - Add `pc_alpha * pc_loss` to main CE loss

3. **Add warmup and decay for PC alpha**
   - Ramp from 0 to `PC_ALPHA` over first 200 steps
   - Decay to 0 during warmdown (representations are stable by then)

4. **Ensure PC heads are excluded from artifact**
   - PC heads are training-only — strip them before quantization
   - Verify no artifact size increase

5. **A/B test on 1xH100** (1000 steps)
   - With PC (alpha=0.1) vs without
   - Measure convergence speed (loss at step 500, 1000)
   - If PC converges faster, proceed. If neutral, try alpha=0.2, 0.5.

6. **Sweep PC_ALPHA on 1xH100** (short runs, 500 steps each)
   - Values: 0.01, 0.05, 0.1, 0.2, 0.5
   - Pick the alpha with fastest convergence

7. **Full run on 8xH100 with best alpha**
   - Measure final BPB vs baseline (no PC)
   - Expected: 0.005-0.015 BPB improvement

**Deliverable:** Measured impact of predictive coding on convergence and final BPB.

### Phase 3: Adaptive Early Exit (Days 5-8)

**Goal:** Add eval-time early exit for throughput improvement.

**Steps:**

1. **Add intermediate classifier losses during training**
   - At layers 5, 7, 9: compute CE loss using shared LM head
   - Weight: `exit_alpha * (layer_idx / num_layers)` (later layers get more weight)
   - This trains intermediate representations for direct token prediction

2. **Implement eval-time early exit**
   - For each token at each exit point (layers 5, 7, 9):
     - Compute entropy of predicted distribution
     - If entropy < threshold, use this prediction (exit early)
     - Otherwise, continue to next layer
   - Measure: what fraction of tokens exit at each layer?

3. **Profile throughput impact**
   - Measure eval ms/token with and without early exit
   - Target: 15-25% speedup on eval
   - More eval budget → more TTT iterations → better BPB

4. **Sweep exit threshold**
   - Too low: few tokens exit (no speedup)
   - Too high: too many tokens exit early with bad predictions (worse BPB)
   - Find the sweet spot where eval speed improves without BPB degradation

5. **Test synergy with predictive coding**
   - Run early exit WITH vs WITHOUT predictive coding
   - Hypothesis: PC makes early exit work much better (intermediate layers are better trained)
   - This is the key experiment that validates our thesis

**Deliverable:** Measured impact of early exit on eval throughput and BPB.

### Phase 4: Combined Optimization (Days 8-11)

**Goal:** Tune the full stack (SOTA base + PC + early exit) for best BPB.

**Steps:**

1. **Joint hyperparameter sweep** (1xH100, short runs)
   - PC_ALPHA x EXIT_THRESHOLD x EXIT_ALPHA grid
   - Also sweep: QK-Gain (5.0-5.5), WD (0.08-0.10), warmdown (0.68-0.76)

2. **Interaction effects**
   - Does PC interact with depth recurrence? (layers 3-5 are looped — PC targets shift)
   - Does early exit interact with parallel residuals? (layers 7+ have different structure)
   - Does PC interact with TTT? (better-trained representations → more/less TTT benefit?)

3. **Ablation table** (8xH100, 1-seed each)
   - Baseline SOTA stack: ~1.08
   - + Predictive coding: ~1.07?
   - + Early exit (eval only): ~1.07?
   - + Both: ~1.06?
   - Document each component's contribution

4. **Artifact budget check**
   - Verify model + code fits under 16MB
   - PC heads are NOT in artifact (training only)
   - If tight: adjust BigramHash size or MLP expansion

**Deliverable:** Optimized full stack with measured ablation of each novel component.

### Phase 5: Submission (Days 11-14)

**Goal:** 3-seed validation, write-up, and PR submission.

**Steps:**

1. **3-seed runs** (8xH100, seeds 42, 314, 999)
   - Record: BPB, training time, artifact size, steps completed
   - Compute mean, std
   - Welch's t-test vs current SOTA

2. **Clean room verification**
   - Fresh RunPod pod
   - Clone repo, install deps, run from scratch
   - Verify train <600s, eval <600s, artifact <16MB

3. **Write submission**
   - Folder: `records/track_10min_16mb/YYYY-MM-DD_PredictiveCoding_AdaptiveExit/`
   - Files: `train_gpt.py`, `README.md`, `submission.json`, `requirements.txt`, 3 seed logs

4. **README emphasis**
   - Novel contribution: first application of predictive coding to competitive LM pretraining
   - Novel contribution: first per-token adaptive early exit in the competition
   - Synergy story: PC enables early exit by making intermediate representations useful
   - Full ablation table showing each component's contribution
   - Comparison to SOTA showing improvement

5. **PR title format:**
   ```
   Record: Predictive Coding + Adaptive Early Exit — val_bpb X.XXXX (3-seed mean)
   ```

---

## Risk Assessment

### Low Risk
- **Predictive coding hurts performance:** Alpha can be tuned to near-zero. Worst case: 2% training overhead with no benefit. Mitigation: decay alpha to 0 during warmdown.
- **Artifact size:** PC heads are training-only (not in artifact). Zero artifact impact.
- **Compatibility:** Both techniques are additive — they don't modify any existing component.

### Medium Risk
- **Early exit hurts eval BPB:** If the threshold is too aggressive, early-exited tokens get worse predictions. Mitigation: conservative threshold, only exit tokens with very low entropy.
- **SOTA rebase takes too long:** The gap from our 1.1147 to the 1.0810 stack is 0.034 BPB across many techniques. Mitigation: use PR #1493's code directly as reference, copy technique by technique.
- **Interaction with depth recurrence:** Layers 3-5 are looped 3x, so the "next layer" relationship is complex. Mitigation: only apply PC to the first pass of recurrent layers, or skip recurrent layers entirely for PC.

### High Risk
- **Novel techniques provide no measurable improvement:** Predictive coding might not accelerate convergence enough to matter in ~7000 steps. Early exit might not save enough eval time to justify the complexity. Mitigation: we still have a competitive ~1.08 submission from Phase 1. Submit that with the novel techniques as an ablation study — still scientifically interesting even if the improvement is marginal.

---

## Compute Budget

| Phase | GPU Hours | Estimated Cost |
|-------|-----------|----------------|
| Phase 1: SOTA rebase (3-5 runs) | 4-6 hrs (8xH100) | $80-120 |
| Phase 2: PC integration + sweeps | 3-4 hrs (1xH100) + 2 hrs (8xH100) | $50-80 |
| Phase 3: Early exit + sweeps | 3-4 hrs (1xH100) + 2 hrs (8xH100) | $50-80 |
| Phase 4: Joint optimization | 4-6 hrs (1xH100) + 3 hrs (8xH100) | $70-100 |
| Phase 5: 3-seed validation | 3 hrs (8xH100) | $60 |
| Buffer | 5 hrs mixed | $80 |
| **Total** | **~30-40 hrs** | **$390-520** |

---

## BPB Projections

| Configuration | Expected BPB | Confidence |
|---------------|-------------|------------|
| Phase 1: SOTA rebase | ~1.08 | High (proven stack) |
| + Predictive coding | ~1.07-1.075 | Medium |
| + Adaptive early exit (eval only) | ~1.065-1.075 | Medium |
| + Both (synergy) | ~1.06-1.07 | Medium-Low |
| Optimistic (with tuning) | ~1.05-1.06 | Low |

Even the conservative case (just SOTA rebase at ~1.08) is a top-10 submission. The novel techniques are upside.

---

## Key Insight: Why This Story Is Compelling

The standard training paradigm supervises only the final layer. In a 10-minute budget with ~7000 training steps, this means:

- **Layer 0** gets gradients diluted through 17 virtual layers of backprop
- **Layer 5** gets gradients diluted through 12 virtual layers
- **Layer 10** gets relatively clean gradients

Predictive coding gives EVERY layer a direct learning signal. In the convergence-limited regime of this competition, this could be the difference between a model that's 90% converged and one that's 95% converged.

And once every layer is well-trained, early exit becomes natural: why force every token through all 17 virtual layers when layer 7's representation is already sufficient for "the cat sat on the ___"?

The narrative: **Richer supervision enables adaptive computation, which enables better efficiency, which enables better final quality.** This is a general principle, not a competition hack.

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `train_gpt.py` | Primary — all model, training, eval, PC, early exit code |
| `README.md` | Submission documentation with ablations |
| `submission.json` | Metadata and results |
| `requirements.txt` | Dependencies (flash-attn-3, sentencepiece, brotli) |

## References

- Predictive Coding: Rao & Ballard (1999), "Predictive coding in the visual cortex"
- Forward-Forward: Hinton (2022), "The Forward-Forward Algorithm"
- Deep Supervision: Lee et al. (2015), "Deeply-Supervised Nets"
- Adaptive Computation: Graves (2016), "Adaptive Computation Time for RNNs"
- Early Exit: Teerapittayanon et al. (2016), "BranchyNet"
- Universal Transformers: Dehghani et al. (2019), "Universal Transformers"
