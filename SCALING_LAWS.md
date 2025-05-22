# Scaling Laws
> Given a fixed compute budget C, what combination of model size N, number of training tokens D, and other hyperparameters minimizes final training loss?

- Scaling laws enable prediction, optimization, and resource-efficient design.
- Predict optimal:
  - Model size
  - Data scale
  - Hyperparameters
  - Compute allocation
- Guide training decisions before large-scale investment.


### Motivation
- Designing LLMs with limited time and massive compute (e.g. 10,000 H100s).
- Need efficient strategies: infra setup, data collection, model design.
- **Key question:** Which model size and configuration yields best performance?

### Introduction to Scaling Laws
- **Old approach:** Tune large models directly (expensive).
- **Modern approach:** Tune small models, extrapolate with scaling laws.

### Practical Data Scaling Considerations
- **Data selection and quality:** Impacts offset, not slope.
- **Repetition effects:** Repeated data has diminishing returns.
- **Adaptive selection:** Data choice should depend on training scale.

### Model Engineering via Scaling Laws
- **Architecture:** Transformers scale better than LSTMs.
- **Optimizer:** Adam generally outperforms SGD in scaling behavior.
- **Depth vs Width:** Tradeoffs; embedding layers behave differently.
- **Batch Size:** Diminishing returns after critical batch size.
- **Learning Rates:** Use scale-aware methods (e.g. muP).

### Joint Model-Data Scaling Laws
- Joint error models:  
  \[
  $\text{Error} = n^{-\alpha} + m^{-\beta} + C$
  \]
- Helps balance model size and dataset size under budget.

### Compute Tradeoffs
- **Chinchilla insight:** More tokens per parameter yields better results under fixed compute.
- Recent models (e.g. LLaMA 3) use >200 tokens per parameter.

### Fitting Scaling Laws (Chinchilla Methods)
1. **Method 1:** Minimum over all training runs.
2. **Method 2:** IsoFLOPs — minimum loss per fixed compute budget.
3. **Method 3:** Joint fits (less reliable, data forensics required).



### Key Challenges
- **Extrapolation**: Generalizing from small-scale experiments to large-scale performance.
- **Tradeoffs**: Balancing between data size and model size for a fixed compute budget.

References:
- Chinchilla (by Hoffmann) https://arxiv.org/abs/2203.15556
- Kaplan https://arxiv.org/abs/2001.08361
- Yang https://arxiv.org/abs/2203.03466


## Scaling Laws from IsoFLOPs Profiles

The **IsoFLOPs** approach evaluates performance while keeping total compute cost fixed $C \approx 6ND$, where:
- N: number of model parameters
- D: number of training tokens

For each compute budget C, models with varying sizes N are trained using datasets sized $D = C / (6N)$, resulting in different final training losses.

### Empirical Findings
- Hoffmann et al. [2022] observed a **quadratic relationship** between final loss and model size under a fixed compute budget.
- Small models cannot fit the data well, resulting in high loss.
- As model size increases, loss decreases smoothly until it eventually **increases again** due to limited gradient updates at fixed compute.


## Fitting the Scaling Laws

Rather than fitting a full curve to each profile, a **simplified method** is used:
- For each compute budget $C_i$, identify the model with the **lowest training loss**.
- This gives a set of optimal model sizes $N_{opt}(C_i)$.
- Fit a **power law**: 
  \[
  $N_{opt} \propto C^a \quad \text{and} \quad D_{opt} \propto C^b$
  \]
- These relationships allow extrapolation to larger budgets for determining the best model/data allocation.

**For example, we can fit power laws for optimal model size and data size given a fixed compute budget** 

<img width="400" src="https://github.com/user-attachments/assets/2157bcb0-44bb-4a8e-afa7-1c233bdcc022" />
<img width="400" src="https://github.com/user-attachments/assets/3a704ca1-34db-41d7-98d9-ebd850ce47d2" />

Then we can extrapolate the optimal model size and data size for large compute budget sizes
```
compute budget: 10^23 FLOPS, optimal model size: 5.00e+10 params, optimal data size: 3.37e+11 tokens
compute budget: 10^24 FLOPS, optimal model size: 1.27e+11 params, optimal data size: 1.33e+12 tokens
```

## Key Strategies and Solutions
**muP (μ-Parametrization)**: 
- Makes hyperparameter tuning invariant to model width.
- Used in Cerebras-GPT and MiniCPM to stabilize training across different model sizes.

**Learning Rate Schedules**:
- **WSD (Warmup-Stable-Decay)** schedule used to approximate full scaling law sweeps more efficiently.
- Preferred over cosine schedules when performing Chinchilla-style analysis.

**Scaling Law Fit Methods**:
- **Chinchilla Method 1**: Lower envelope of training curves.
- **Chinchilla Method 3**: Joint fitting of multiple runs for better efficiency.
- **IsoFLOPs**: Compute-budget-based analysis used by DeepSeek and others.

### Model-Specific Notes
- **Cerebras-GPT**: Combines muP with Chinchilla scaling laws.
- **MiniCPM**: Uses muP and WSD learning rate; fits Method 3.
- **DeepSeek**: Does not use muP; performs direct batch/LR fits and Method 2 (IsoFLOPs).
- **LLaMA 3 & Hunyuan-1**: Rely on isoflops-style scaling; fewer tuning details disclosed.
- **MiniMax-01**: Focuses on architectural scaling and Method 1 analysis.
