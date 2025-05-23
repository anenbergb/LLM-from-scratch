# Alignment

# Alignment: Reinforcement Learning from Human Feedback (RLHF)

## Overview
- Goal: Improve and align LM outputs using instruction tuning and RLHF.
- Problem: Pretraining data is broad but misaligned; fine-tuning data is costly and limited.

![image](https://github.com/user-attachments/assets/53bb3c38-1464-4c01-a18d-a864cc4ac6d0)
OpenAI InstructGPT "Training language models to follow instructions
with human feedback" https://arxiv.org/abs/2203.02155

## Instruction Tuning (Supervised Fine-Tuning / SFT)
- Based on expert demonstrations.
- Effective when extracting *pretrained* knowledge and behaviors.
- **Datasets:**
  - **FLAN**: Aggregated NLP tasks, structured, task-specific.
  - **OpenAssistant (Oasst)**: Open-source, structured instruction tasks.
  - **Alpaca**: Seed human instructions + LLM-generated expansions.

### Key Points
- Data quality matters more than quantity due to smaller dataset size.
- Longer and list-style responses are often preferred by humans.
- Fine-tuning on facts not learned during pretraining leads to hallucinations.
- Ideal to fine-tune on pretraining-known facts only.
- Safety tuning possible with ~500 samples. But there's a risk that you might refuse legitimate requests such as (e.g., handling questions like "How can I kill a Python process") 

## Scaling Instruction Tuning
- **Instruction tuning as pretraining**:
  1. Pretrain on web-scale data.
  2. Mix in instruction data.
  3. Follow up with a brief instruction-tuning phase.
- **Midtraining / Two-phase Training**:
  - Blend instruction data into late pretraining phases to avoid catastrophic forgetting.
  - Used in MiniCPM, JetMoE.
 
<img width="800" src="https://github.com/user-attachments/assets/861e533e-2713-4ecd-8554-945fe5538353" />


## Reinforcement Learning with Human Feedback (RLHF)

### Motivation
- SFT data is expensive; RLHF reduces annotation cost.
- Scalar feedback (preferences) is cheaper than constructing ideal completions.

### RLHF Pipeline
1. **Collect Pairwise Feedback**:
   - Two model responses, human ranks the better one.
   - Human reviewers often favor longer outputs and stylistic features.
   - AI-based feedback (e.g., GPT-4) is becoming popular and cost-effective.
2. **Train a Reward Model**:
   - Uses ranked pairs to predict response quality.
3. **Policy Optimization**:
   - **PPO (Proximal Policy Optimization)**: Complex, gradient clipping to stabilize updates.
   - **DPO (Direct Preference Optimization)**: Simpler, no reward model or on-policy rollouts.

### Trade-offs and Pitfalls
- **Overoptimization**: Excessive tuning leads to overfitting and mode collapse.
- **Mode Collapse**: Loss of diversity and probabilistic modeling in outputs.
- **Style Bias**: Preference data can introduce unintended biases (e.g., toward verbosity).

## Summary
- SFT is best used to extract, not inject, knowledge.
- RLHF provides a scalable path for aligning models, but introduces new risks.
- DPO is a promising alternative to PPO with fewer moving parts.
- Effective alignment requires balancing data quality, diversity, and safety.

# Lecture 16 Outline: Reinforcement Learning from Verifiable Rewards (RLVR)

## I. Introduction & Motivation
- RLHF has limitations: overoptimization, lack of calibration, mode collapse.
- Verifiable rewards enable more stable, scalable RL in domains with clear correctness.
- Empirical results in RL are setup-dependent—careful interpretation is essential.

---

## II. Core RL Algorithms for Language Models

### A. PPO (Proximal Policy Optimization)
- **Theory**:
  - Starts from policy gradients: ∇θ Eₚθ[R(z) ∇θ log pθ(z)]
  - TRPO introduces KL constraints.
  - PPO simplifies this with a clipping mechanism to avoid large policy shifts.

- **Practice**:
  - Requires value model for advantage estimation.
  - KL-penalized per-token rewards + full-sequence reward.
  - Generalized Advantage Estimation (GAE) reduces variance.
  - Drawback: complex and memory-intensive, especially due to the value model.

### B. DPO (Direct Preference Optimization)
- **Goal**: Remove the need for reward models and on-policy rollouts.
- **Method**:
  - Non-parametric assumption links policy and reward.
  - Optimize implied reward via supervised loss (MLE on pairwise comparisons).
- **Pros**: Simpler implementation, competitive performance.
- **Cons**: Requires pairwise data (not always available), typically offline.

### C. GRPO (Group Regularized Policy Optimization)
- **Goal**: Simpler alternative to PPO for verifiable rewards.
- **Key Features**:
  - No value model or GAE.
  - Advantage = z-score within group.
  - Normalize rewards by group std, enabling online updates.
- **Biases**:
  - Length normalization can cause bias towards longer responses.
  - Standard deviation as baseline does not preserve unbiasedness.
- **Fixes & Critiques**:
  - Alternatives proposed (e.g., leave-one-out baselining).
  - Still prone to reward overfitting and language mixing in output.

---

## III. Case Studies in RLVR

### A. DeepSeek R1
- **Setup**:
  - Uses GRPO (no process supervision).
  - Verifiable accuracy and formatting rewards.
- **Training Phases**:
  1. **R1-Zero**: GRPO RL on verifiable tasks only.
  2. **R1**: Adds SFT with long chain-of-thought (CoT), then non-verifiable RLHF.
- **Notable**:
  - Beats OpenAI O1.
  - Introduces language-consistency loss.
  - Distillation step used to train smaller models like Qwen.

### B. Kimi K1.5
- **Pipeline**:
  1. Data curation (difficulty filtering).
  2. SFT on long CoTs.
  3. RL using a DPO-inspired loss with reward model.
- **Innovations**:
  - Length control reward for efficient CoT.
  - Curriculum learning by sampling hard examples more often.
- **Infrastructure**:
  - Uses vLLM workers with dummy weights for rollout, torn down after each iteration.

### C. Qwen 3
- **Distinctive Features**:
  - Low-data RLVR (GRPO on ~4k samples).
  - “Thinking mode fusion”: toggle thinking vs. non-thinking responses.
  - Special tokens for early stopping and controlling generation length.
- **Pipeline**:
  - Filtering of too-easy questions and low-quality CoTs.
  - RLHF follows reasoning RL and SFT.

---

## IV. Practical Considerations in RL Infrastructure
- **Challenges**:
  - On-policy learning (PPO/GRPO) is compute-intensive.
  - Long CoTs cause uneven batch processing.
  - Switching between inference and training requires complex systems coordination.
- **Solutions**:
  - Sync LLM weights between inference and RL workers.
  - Use dummy weight vLLMs to manage memory constraints.

---

## V. Key Themes & Takeaways
- **Overoptimization**: RL models may overfit proxy reward signals, diverging from human intent.
- **Biases in Learning**: GRPO may skew towards easy/hard tasks or longer completions.
- **RLVR Frameworks**: Verifiable rewards + simple RL (like GRPO) scale well in structured domains.
- **Successful RLVR Recipes**:
  - **Deepseek R1**: GRPO + SFT + RLHF (best known open result).
  - **Kimi K1.5**: RL with DPO-style loss + length control.
  - **Qwen 3**: Efficient GRPO + innovative data tagging.

