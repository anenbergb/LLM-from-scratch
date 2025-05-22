# Reinforcement Learning from Human Feedback (RLHF) / Alignment

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
