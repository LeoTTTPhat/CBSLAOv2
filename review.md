# ICSOC 2026 Research Track Review

**Paper:** Cost- and SLA-Bounded Orchestration for LLM-Agent Tool/Service Composition  
**Track:** Research papers  
**Reviewer role:** ICSOC reviewer  
**Recommendation:** Weak Accept  
**Reviewer confidence:** 4 / 5

---

## Summary

This paper studies cost- and SLA-bounded orchestration for LLM-agent tool/service composition. The core problem, CBSLAO, asks an agent controller to select tool invocations adaptively while satisfying chance-constrained budget and deadline requirements. The proposed algorithm, CBUC, combines schema-type closure pruning with a budget-aware UCB selector and chance-constraint shrinkage.

The paper has a substantial theoretical package: NP-hardness, a Max-$k$-Coverage inapproximability result at the $(1-1/e+\varepsilon)$ threshold, a matching offline $(1-1/e)$ approximation for the monotone-submodular non-adaptive case, an adaptivity-gap discussion, and a regret bound for CBUC under sub-Gaussian assumptions. The evaluation is a deterministic synthetic simulator with an extended sweep of 3,960 rows and several constraint-aware baselines. CBUC achieves very low budget-overrun and zero deadline-miss rates, but with a severe utility cost relative to the Lagrangian primal-dual baseline.

## Fit to ICSOC

The paper is a strong topical fit for ICSOC 2026. The call explicitly lists service composition, service monitoring and adaptive management, AI-empowered service construction and composition, AI tools for service science and engineering, data governance service-oriented architectures, and service-oriented computing foundations as relevant areas. This paper sits at the intersection of service composition, AI-agent orchestration, and service governance.

The service-oriented framing is not merely superficial: the schema-typing model and type-closure pruning make the problem recognizably about service/tool discovery and composition rather than generic LLM prompting. The strongest fit is to **Focus Area 1: Service-Oriented Technology Basics and Trends** and **Focus Area 2: AI for Services and as-a-Service**.

## Strengths

1. **Timely and relevant problem.** LLM agents increasingly act as service orchestrators, and budget/SLA governance is a real deployment issue. The paper identifies a concrete gap in existing agent controllers.

2. **Service-composition-specific formalization.** The CBSLAO model ties tool schemas, type closure, stochastic cost, latency, and utility into a single optimization problem. This is more aligned with ICSOC than a generic LLM-agent evaluation paper would be.

3. **Strong theoretical contribution.** The Max-$k$-Coverage reduction and matching $(1-1/e)$ approximation result give a tight offline non-adaptive characterization. The adaptivity-gap discussion further clarifies what is and is not captured by the regret target.

4. **Good baseline set.** The evaluation no longer compares only against weak capped ReAct-style baselines. Lagrangian primal-dual, CVaR-BwK, pre-check UCB, and CC-knapsack are meaningful constraint-aware comparators.

5. **Honest trade-off analysis.** The paper acknowledges that CBUC’s safety comes with a large utility loss: 0.587 utility vs. 0.806 for Lagrangian primal-dual, approximately a 27% relative reduction. The new connection to the $(1-1/e)$ offline approximation bound is helpful.

6. **Reproducibility orientation.** The paper provides deterministic seeds, code paths, raw result filenames, and a short runtime claim. This aligns well with ICSOC’s emphasis on artifact availability and reproducibility.

7. **Improved presentation.** The added schematic figures make the paper easier to follow: the governance-layer diagram, theorem landscape, CBUC control loop, and safety–utility frontier are all useful.

## Weaknesses

1. **Evaluation remains synthetic.** The largest remaining weakness is the absence of live tool/API traces. Calibrated simulation is useful, but the paper’s motivating claims concern real LLM-agent service orchestration. A small replay over public tool-use traces, or even a limited live API/tool benchmark, would substantially strengthen construct validity.

2. **Strict schema typing is central to both algorithm and lower bound.** The paper now includes a useful caveat about fuzzy MCP/OpenAPI registries, but the theory still depends heavily on exact type incidence. Real service registries often use partial schemas, optional fields, natural-language descriptions, and fuzzy semantic matching. The paper does not evaluate degradation under schema ambiguity.

3. **The utility cost is severe.** CBUC is the safest policy, but it is not clearly the best deployment choice in all settings. Pre-check UCB gives 0.701 utility with only 2.8% overrun and no deadline misses, while Lagrangian primal-dual gives 0.806 utility but with 23.1% overrun. The paper frames this as a frontier, which is appropriate, but the abstract and conclusion still risk making CBUC sound like the dominant solution rather than the conservative endpoint.

4. **The regret theorem and simulator assumptions diverge.** Theorem 6 assumes bounded sub-Gaussian costs, while headline results aggregate over Pareto $\alpha=1.5$ cells where variance is infinite. The paper acknowledges this, but the main empirical claim still depends on a regime outside the theorem.

5. **Some proof details may be too compressed for a research-track paper.** The coupled empirical-Bernstein concentration and claimed constant-factor saving over decoupled analysis are interesting, but the reader must trust several appendix-level steps. The paper would benefit from a clearer statement of exactly where dependence among cost, latency, and utility is used.

6. **Reproducibility statement is not yet submission-ready.** The repository link is still an anonymous placeholder. ICSOC’s call emphasizes artifact availability and reproducibility statements; the final submission should include an anonymized artifact package or clear instructions.

7. **Potential page-limit risk.** ICSOC research papers should be 10–15 pages including references at initial submission. The source has many tables, figures, and a long appendix. I could not compile the PDF in this environment, so I cannot verify page count, but the authors should check this carefully.

## Detailed Comments

### Theoretical Contribution

The theoretical section is the paper’s strongest part. The progression from exact NP-hardness to Max-$k$-Coverage inapproximability and then to a matching offline approximation result is clean and compelling. The strict/fuzzy schema typing remark is important because the Max-$k$-Coverage reduction relies on exact element-set incidence encoded as schema outputs.

The adaptivity-gap material is also useful, but the paper should ensure that all references to the empirical non-adaptive/informed-knapsack ratio are backed by a visible table or appendix calculation. Otherwise, the “median 1.00” statement may feel ungrounded.

### Algorithm

CBUC is technically plausible and well motivated. The two-stage structure is simple: prune by type closure, then sequence with budget-aware UCB and conservative feasibility checks. The algorithmic description is clear.

The main concern is not the algorithm itself but its positioning. CBUC is conservative by design. That is good for hard-budget settings, but the paper should be careful not to imply that it is the best controller for all service orchestration contexts.

### Evaluation

The evaluation is broad and much improved by the constraint-aware baselines. Table 1 is now internally consistent with `out/results_stronger.csv`, and the safety–utility frontier figure is an effective summary.

However, the evaluation does not yet establish that the simulator reflects real LLM-agent behavior. The calibrated benchmark replay helps, but it is still parameterized simulation rather than replaying actual traces. For ICSOC, where service deployments and system evidence matter, this is the main barrier to a stronger accept.

### Writing and Presentation

The paper is generally clear. The newly added figures help the reader navigate the contribution. The abstract is dense but effective. The limitations section is unusually honest, which improves credibility.

The paper should watch for overclaiming in phrases like “drop-in production controller” or “governance-layer proposal.” The current limitations already say contextual-bandit extensions are required before production; that caveat should remain visible in the introduction/conclusion framing.

## Questions for Authors

1. Can the authors add even a small trace replay or live-tool pilot before camera-ready?

2. How sensitive is CBUC to schema noise: missing types, extra permissive types, or fuzzy similarity thresholds?

3. Does the coupled empirical-Bernstein analysis materially change the chosen policy in experiments, or is it primarily a proof-level improvement?

4. Can the authors report paired-seed significance tests for the utility gaps, especially CBUC vs. pre-check UCB and CBUC vs. Lagrangian?

5. Can the authors include an anonymized artifact link and exact reproduction commands in the submission version?

## Scores

| Criterion | Score | Rationale |
|---|---:|---|
| Originality | 4 / 5 | The service-governance formalization and schema-aware hardness result are novel and relevant. |
| Technical Soundness | 4 / 5 | Strong theory and careful evaluation, with caveats around heavy-tailed costs and proof compression. |
| Significance | 4 / 5 | Important problem for AI-enabled service composition; significance limited by synthetic-only evaluation. |
| Clarity | 4 / 5 | Clear writing and good figures; dense theory may be challenging for some readers. |
| Reproducibility | 4 / 5 | Strong deterministic setup, but needs an actual anonymous artifact link. |
| ICSOC Fit | 5 / 5 | Directly relevant to service composition, AI for services, and service governance. |

## Recommendation

**Weak Accept.**

This is a strong ICSOC submission with a timely service-oriented computing problem, a substantial theoretical contribution, and a serious empirical comparison against constraint-aware baselines. I would accept it because the contribution is original, technically credible, and highly relevant to AI-enabled service composition.

The main reason I do not recommend a stronger accept is the lack of real trace or live-service evaluation. The paper makes deployment-facing claims, but the evidence is still simulator-only. If the authors can add even a small trace replay and an anonymized artifact link, the paper would become a clear accept.

## Confidence

**4 / 5.** I am comfortable assessing the service-composition framing, online-learning/bandit structure, and empirical claims. I did not compile the PDF locally and therefore did not verify page length or rendered figure placement.
