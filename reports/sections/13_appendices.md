# 13. Appendices

## Appendix A: Reproducibility Guide

### A.1 Hardware and Environment Specifications
All experiments, training runs, and evaluations were executed in the following local environment:
- **Operating System:** Microsoft Windows 11 Home / Pro (x86_64 architecture).
- **Python Runtime:** Python 3.14.3 (tags/v3.14.3, 64-bit AMD64).
- **Primary Package Dependencies:**
  - `numpy==2.2.3`
  - `pandas==2.2.3`
  - `scipy==1.15.2`
  - `scikit-learn==1.6.1`
  - `matplotlib==3.10.1`
  - `streamlit==1.42.0`
  - `plotly==6.0.0`
  - `pytest==8.3.4`
  - `python-pptx==1.0.2`
- **Hardware Resources:** Standard multi-core laptop CPU (no dedicated GPU acceleration was required or utilized).

### A.2 Execution Commands
All results, models, and application interfaces can be fully reproduced from the repository root:

```bash
# 1. Install required Python dependencies
pip install -r requirements-app.txt
pip install python-pptx

# 2. Re-run complete test suite (20 automated tests)
python -m pytest app/tests -v

# 3. Launch the operational Streamlit decision application
streamlit run app/streamlit_app.py

# 4. Re-generate comparison plots and slide figures
python scripts/generate_slide_figures.py

# 5. Build presentation slide deck (.pptx)
python scripts/build_slide_deck.py
```

## Appendix B: Design-Decision Log

**Table B.1: Architectural Design Decisions, Rationales, and Source Files**

| Decision ID | Plan Proposal | As-Built Implementation | Rationale and Context | Source File Reference |
| :---: | :--- | :--- | :--- | :--- |
| **DD-01** | Global random sequence per run | Synchronized per-round random seeds | Eliminates environmental variance between competing policies; all policies face identical customer arrivals. | `src/simulation/generator.py` |
| **DD-02** | Single online training run | Dual-mode evaluation (Online & Frozen) | Distinguishes the cost of exploratory learning (20k rounds) from mature operational policy quality (5k held-out rounds). | `src/evaluation/evaluator.py` |
| **DD-03** | Tune on evaluation seeds | Separate tuning seeds (101–105) | Prevents hyperparameter over-fitting on reported evaluation seeds (Seeds 42–51). | `results/alpha_sensitivity.csv` |
| **DD-04** | Soft penalty reward for loyalty | Hard programmatic guardrails | Soft penalties collapse discount volume globally; hard rules guarantee 100% policy compliance without distorting bandit learning. | `src/guardrails/rules.py` |
| **DD-05** | Real customer purchase logs | Synthetic 8,000-customer registry | Public Kaggle retail dataset lacks individual customer IDs; longitudinal RFM profiles required simulation. | `src/simulation/registry.py` |
| **DD-06** | Naive matrix inversion $\mathcal{O}(d^3)$ | Sherman-Morrison rank-1 update $\mathcal{O}(d^2)$ | Enables sub-millisecond inference and high-throughput training in pure Python/NumPy. | `src/policies/linucb.py` |
| **DD-07** | Empty set failure crash | Fallback to $\{0\%\}$ full price | Guarantees operational fault tolerance if commercial margin floors eliminate all promotional arms. | `src/guardrails/rules.py` |
| **DD-08** | Combined tier input in app | Decoupled Profile Tier vs. Context Tier | Preserves transparency when merchandisers test edge-case RFM inputs while inspecting specific tier rule envelopes. | `app/streamlit_app.py` |
| **DD-09** | Sole focus on net profit | Paired tracking of Profit, Revenue, and Conceded Discount | Exposes the critical business trade-off where bandits capture highest revenue by trading away net margin. | `src/evaluation/ledger.py` |

## Appendix C: Glossary of Terminology and Notation

- **Round ($t$):** A single atomic decision instance consisting of an observable customer context $x_t$, a product inventory state, a recommended discount arm $a_t$, and a stochastic conversion outcome $Y_t$.
- **Policy ($\pi$):** A decision-making mapping $\pi: \mathcal{X} \to \mathcal{A}$ that selects an action based on context.
- **Arm ($a$):** A discrete promotional discount choice from the action set $\mathcal{A} = \{0\%, 5\%, 10\%, 15\%, 20\%\}$.
- **Eligible Set ($\mathcal{A}_t$):** The subset of actions $\mathcal{A}_t \subseteq \mathcal{A}$ permitted after filtering context $x_t$ through loyalty tier caps and margin floor guardrails.
- **Guardrail:** A deterministic commercial rule enforced prior to algorithmic scoring that prevents illegal or below-cost recommendations.
- **Oracle ($a_t^*$):** The counterfactual arm within the eligible set $\mathcal{A}_t$ that maximizes true mathematical expected net profit under simulator ground-truth probabilities.
- **Expected Regret ($\Delta_t$):** The difference in expected profit between the oracle action and the policy's chosen action: $\Delta_t = \mu_t^* - \mu_t(a_t)$.
- **Cumulative Regret ($R(T)$):** The running summation of instantaneous expected regret: $R(T) = \sum_{t=1}^T \Delta_t$.
- **Financial Ledger:** An immutable transaction log recording per-round revenues, unit costs, realized profits, and discount concessions.
- **Discount Offered:** The nominal percentage discount presented to the consumer ($a_t$).
- **Discount Given (Conceded):** The actual currency value surrendered upon completed transaction: $Y_t \cdot (\text{BasePrice} \times a_t) \cdot \text{BasketSize}$.
- **Online Evaluation:** Protocol where the bandit begins from a cold start, actively explores, and updates model parameters after every transaction.
- **Frozen Evaluation:** Protocol where pre-trained model weights are locked, evaluating steady-state operational performance without updates on held-out rounds.
- **Off-Policy Replay:** Evaluation method using historical transaction logs via rejection sampling (Li et al. [2]).
- **Simulated Rupees (₹):** Simulated monetary units generated within the retail simulation environment.

## Appendix D: Complete Hyperparameters and Settings Table

**Table D.1: Master Algorithmic and Experimental Hyperparameters**

| Category | Hyperparameter Name | Symbol | Value | Source File |
| :--- | :--- | :---: | :---: | :--- |
| **Bandit** | Action Space | $\mathcal{A}$ | $\{0\%, 5\%, 10\%, 15\%, 20\%\}$ | `src/policies/base.py` |
| **Bandit** | Context Dimension | $d$ | 20 | `models/linucb_v1_meta.json` |
| **Bandit** | LinUCB Exploration Parameter | $\alpha$ | 0.5 | `models/linucb_v1_meta.json` |
| **Bandit** | Ridge Regularization Factor | $\lambda_2$ | 1.0 ($I_d$) | `src/policies/linucb.py` |
| **Bandit** | Thompson Sampling Noise Multiplier | $\sigma$ | 0.5 | `models/thompson_v1_meta.json` |
| **Simulation**| Online Horizon | $T_{\text{online}}$ | 20,000 rounds | `models/linucb_v1_meta.json` |
| **Simulation**| Frozen Evaluation Horizon | $T_{\text{frozen}}$ | 5,000 rounds | `src/evaluation/evaluator.py` |
| **Simulation**| Evaluation Random Seeds | $\mathcal{S}_{\text{eval}}$ | Seeds 42 through 51 | `results/comparison_online.csv` |
| **Simulation**| Hyperparameter Tuning Seeds | $\mathcal{S}_{\text{tune}}$ | Seeds 101 through 105 | `results/alpha_sensitivity.csv` |
| **Simulation**| Synthetic Customer Registry | $N_{\text{cust}}$ | 8,000 profiles | `src/simulation/registry.py` |
| **Simulation**| Assumed Cost Margin Rate | $m_{\text{cost}}$ | 40% of Base Price | `src/simulation/simulator.py` |
| **Simulation**| Demand Logistic Calibration AUC | $\text{AUC}$ | 0.6322 | `reports/evaluation_findings.md` |
| **Guardrails**| New Tier Discount Cap | $a_{\max}^{\text{New}}$ | 5% | `src/guardrails/rules.py` |
| **Guardrails**| Regular Tier Discount Cap | $a_{\max}^{\text{Reg}}$ | 15% | `src/guardrails/rules.py` |
| **Guardrails**| Loyal Tier Discount Cap | $a_{\max}^{\text{Loyal}}$ | 20% | `src/guardrails/rules.py` |
| **Guardrails**| Default Unit Margin Floor Buffer | $\Delta_{\text{floor}}$ | ₹0.00 | `src/guardrails/rules.py` |

## Appendix E: 50-Milestone Completion Map

**Table E.1: Complete Roadmap Completion Record (Phases 1 through 9)**

| Phase | Milestone Range | Primary Objective | Key Output Artifact | Status |
| :---: | :---: | :--- | :--- | :---: |
| **Phase 1** | M01 – M08 | Data foundation, cleaning, feature engineering, registry | `data/processed_retail.csv`, `src/simulation/registry.py` | **Complete** |
| **Phase 2** | M09 – M14 | Demand calibration, simulator core, counterfactuals, oracle | `src/simulation/simulator.py`, calibration logs | **Complete** |
| **Phase 3** | M15 – M19 | Programmatic guardrails, margin floor, historical audit | `src/guardrails/rules.py`, `reports/fairness_audit.json` | **Complete** |
| **Phase 4** | M20 – M22 | Baseline policies (Random, Fixed 10%, Rule-based) | `src/policies/baselines.py` | **Complete** |
| **Phase 5** | M23 – M28 | Bandit implementations (LinUCB, Thompson Sampling, Sherman-Morrison) | `src/policies/linucb.py`, `src/policies/thompson.py` | **Complete** |
| **Phase 6** | M29 – M31 | 20,000-round training, model serialization, logging | `models/linucb_v1.npz`, `models/thompson_v1.npz` | **Complete** |
| **Phase 7** | M32 – M40 | Multi-seed evaluation, regret, ledger, replay, sensitivity, ablation | `results/*.csv`, `outputs/plots/*.png`, `reports/evaluation_findings.md` | **Complete** |
| **Phase 8** | M41 – M47 | Frontend Streamlit app, test suite, dashboard UI | `app/streamlit_app.py`, `app/tests/test_app.py` (20/20 pass) | **Complete** |
| **Phase 9** | M48 – M50 | Final report, claims audit, slides & charts, viva preparation | `reports/final_report.md`, `slides/deck.pptx`, `viva/fact_sheet.md` | **Complete** |
