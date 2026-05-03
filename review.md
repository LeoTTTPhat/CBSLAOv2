# ICSOC 2026 — Research Track Review (Round 2, post-revision)

**Paper:** Cost- and SLA-Bounded Orchestration for LLM-Agent Tool/Service Composition (CBSLAO)
**Track:** Research
**Reviewer:** Anonymous Reviewer
**Round:** 2 — re-review after authors' revision addressing prior R1 weakness on theoretical novelty

---

## 1. Summary (revised to reflect new theoretical content)

The paper formalizes the problem of orchestrating Large Language Model (LLM) agent tool/service compositions under chance-constrained token-cost budgets and latency deadlines, framed as Cost- and SLA-Bounded Agent Orchestration (CBSLAO). The revised version now establishes a **four-result theoretical package** for the offline non-adaptive case:

1. NP-hardness of CBSLAO-DEC via reduction from 0/1 Knapsack (Theorem 1, retained from R1).
2. **New** $(1 - 1/e + \varepsilon)$-inapproximability of CBSLAO-OPT via reduction from Max-$k$-Coverage that *exploits the schema-typing structure of tools* (Theorem 2). The reduction encodes element-set incidence as a service-composition constraint, distinguishing it from generic stochastic-knapsack hardness.
3. **New** matching polynomial-time $(1 - 1/e)$-approximation in the offline non-adaptive monotone-submodular case via Sviridenko's modified greedy (Theorem 4). Together with Theorem 2, this *pins the offline approximability of CBSLAO exactly at $1 - 1/e$*.
4. **New** explicit adaptivity-gap construction (Lemma 5): the gap is at least $4/3$ on a worst-case family (via embedding of Dean–Goemans–Vondrák) and at most an absolute constant on every instance.

The online algorithm CBUC retains its $\tilde{O}(\sqrt{KT})$ regret bound against the non-adaptive chance-constrained oracle, now derived via a **coupled multivariate empirical-Bernstein concentration** (revised Lemma L1 in Appendix A) that retains cost–latency–utility dependence inside the radius formula and saves a factor of $\sqrt{2}$ in the constant $C_1$ relative to the decoupled analysis.

The empirical study is unchanged: 18 workload cells × 20 seeds × 11 policies = 3,960 runs; CBUC reduces budget overrun from 36.9% (ReAct-capped) and 28.9% (BwK-vanilla) to 1.1% and eliminates deadline misses, at a cost of 0.587 utility (vs. 0.806 for Lagrangian primal-dual).

## 2. Relevance to ICSOC 2026

Excellent fit, unchanged from R1. The schema-typing-aware reduction in the new Theorem 2 *strengthens* the ICSOC framing: the inapproximability bound is no longer a generic stochastic-knapsack inheritance but a service-composition-specific result.

## 3. Strengths

1. **Timely and well-motivated problem.** Unchanged.
2. **Rigorous and extensive evaluation.** Unchanged: 10 baselines including four constraint-aware competitors.
3. **Reproducibility.** Unchanged: deterministic seeds, full simulator, sub-3-minute end-to-end run.
4. **Honest discussion of trade-offs.** Unchanged.
5. **Thorough ablation studies.** Unchanged.
6. **(NEW) Theoretically tight characterization of the offline non-adaptive case.** The pairing of Theorems 2 and 4 — $(1-1/e+\varepsilon)$-hard from above, $(1-1/e)$-easy from below — is precisely the form of result that converts a "hardness flag" into a real complexity-theoretic contribution. The Theorem 2 reduction is non-trivial: it uses the type-closure semantics to embed Max-$k$-Coverage in a way that ordinary stochastic-knapsack reductions cannot.
7. **(NEW) Coupled concentration is a real, if modest, theoretical improvement** over the off-the-shelf BwK analysis. The $\sqrt{2}$ saving in $C_1$ is small but the framing — that the cost-latency dependence should be retained inside the radius formula and tightens further under positive correlation — is the right structural observation for the agent-orchestration setting, where slow tools tend to be expensive.

## 4. Weaknesses & Areas for Improvement (revised)

1. **Lack of real-world evaluation.** *Unchanged from R1.* The evaluation is still entirely synthetic. While the calibrated benchmark replay (A3) parameterizes the simulator to ToolBench / τ-bench / BFCL summary statistics, no live-trace replay is performed. This remains the principal limitation of the paper. Even a 100–200-task pilot against an OpenAI tool-calling endpoint would close the construct-validity gap.

2. **~~Incremental Theoretical Novelty.~~** **(R1 weakness — substantially resolved by the revision.)**
    - ~~The NP-hardness proof is a textbook reduction from 0/1 Knapsack...~~ The paper now adds a non-trivial $(1-1/e+\varepsilon)$-inapproximability theorem (Thm. 2) that exploits the schema-typing structure of tools, plus a matching $(1-1/e)$-approximation (Thm. 4) and an explicit adaptivity-gap lemma (Lemma 5).
    - ~~The regret bound is a fairly direct adaptation of BwK...~~ The proof now uses a coupled multivariate empirical-Bernstein concentration (Lemma L1, revised) with explicit savings over the decoupled analysis.
    - **Residual concern (downgraded from major to minor):** the regret bound itself — Theorem 6 — is still stated against the non-adaptive oracle, and the adaptive-oracle regret remains an open problem (O2). Lemma 5 now bounds the gap by an absolute constant in the worst case, which is reassuring but still a constant-factor approximation rather than a tight result. *I no longer consider this a blocking weakness.*

3. **Strict Schema Typing Assumption.** *Unchanged from R1.* The new Theorem 2 actually *uses* the strict schema-typing assumption to encode element-set incidence; this is fine theoretically but makes the gap with fuzzy real-world tool registries (MCP, OpenAPI) more visible, not less. The paper would benefit from a half-paragraph addressing whether the inapproximability bound degrades gracefully under fuzzy matching or whether a different lower bound is needed.

4. **Utility Cost is Severe.** *Unchanged from R1.* The 27% relative drop versus Lagrangian primal-dual is still steep; the new theorems do not address this empirical trade-off. Theorem 4 gives a $(1-1/e) \approx 0.632$ offline guarantee that is in fact *consistent* with the empirical 0.587 — this should probably be noted in §6.5.

5. **(NEW, minor) CBUC-L's "informal" Proposition is now more visible by contrast.** With Theorems 2 and 4 making the offline theory tight, the only remaining "informal" theoretical statement is Proposition 3 for CBUC-L. The paper would be cleaner if either the CBUC-L proposition were upgraded to a formal theorem (open problem O1) or relegated to a clearly labelled conjecture box. This is a presentation issue, not a correctness one.

6. **(NEW, minor) Citations need a final pass.** The references.bib file still contains six `note = {(verify citation)}` annotations and three ICSOC-2025 placeholder entries. With the new Theorems 2 and 4, two more citations enter the critical path (`feige1998threshold`, `nemhauser1978best`) and the authors should confirm them. The paper now depends on Feige (1998) for the inapproximability bound, which is correct as cited but should be double-checked against the original journal version.

## 5. Detailed Comments (revised)

- **Section 4 (Hardness and Approximability) — substantially revised:**
    - The new section title is appropriate.
    - **Theorem 2 (inapproximability):** correct and well-stated. The reduction is clean — element-set incidence becomes type-output incidence, and the chance constraint with $\delta = 0$ correctly degenerates to $|A| \le k$. Suggest adding a one-sentence remark that the result transfers to *any* tail-aware variant of CBSLAO with $\delta < 1$, since the lower bound is achieved at $\delta = 0$.
    - **Theorem 4 (approximation):** the proof of submodularity from concavity of $g$ is correct (composition of monotone concave with non-negative additive). The complexity statement $O(K^4)$ tracks Sviridenko's bound; for the simulator's typical $K \le 200$ this is well under a millisecond and can be made an explicit observation in §5.4.
    - **Lemma 5 (adaptivity gap):** the embedding of DGV into CBSLAO at $\delta = 0$ is correct. The proof sketch is brief; for camera-ready, a half-page expansion in the appendix would be welcome.
    - The previously-trivial Cor. 2 (now `cor:inap-trivial`) is correctly retained for completeness but explicitly noted as subsumed.
- **Section 5 (Algorithm and Regret Bound) and Appendix A:**
    - The revised L1 (coupled multivariate empirical-Bernstein) is correct. The $\sqrt{2}$ saving is honest — it is small but real. The remark about positive cost-latency correlation tightening the bound further is the right structural observation.
    - The CVaR corollary (Cor. 2 for heavy-tailed cost) and Theorem 6's $C_1$ constant should be re-derived using the new L1 to check that the constants flow through; I expect they do but the appendix only updates L1 itself.
    - CBUC-L (§5.4) remains untouched. Given that the offline theory is now tight, the contrast between the formal $(1-1/e)$ result and the "informal" CBUC-L proposition is sharper.
- **Section 6 (Evaluation):**
    - §6.5's utility-gap decomposition is unchanged. Recommend adding a sentence noting that the 0.587 empirical CBUC utility is *consistent with* the $(1-1/e) \approx 0.632$ offline approximability bound from Theorem 4, providing a theoretical floor for the safe-policy class.
    - Numerical-consistency issues from R1 (28.1% vs. 28.9% in abstract vs. Table 1; 3,780 vs. 3,960 row counts) appear to have been reconciled in the revised abstract. **Confirmed.**
- **Threats to Validity (Section 7):** unchanged; still well-written.

## 6. Questions for the Authors / Rebuttal (revised)

1. **Live-API pilot:** still the most important open question — is there a fundamental blocker to running 100–200 tasks against a live tool-calling endpoint?
2. **Utility vs. safety break-even:** unchanged from R1.
3. **Fuzzy schema matching:** the new Theorem 2 *uses* the strict schema-typing structure for the lower bound. How robust is the inapproximability under fuzzy / partial matching? Is there a smoothed-analysis or PAC-style relaxation that recovers the same $(1-1/e+\varepsilon)$ floor?
4. **(NEW) Constants flow-through:** does the revised L1's $\sqrt{2}$ saving propagate cleanly through L2–L4 to a tighter $C_1$ in Theorem 6? The current appendix updates L1 in place but does not re-derive the dependent constants.
5. **(NEW) CBUC-L:** with the offline theory now tight, can the authors commit to either (a) closing the regret-bound proof for CBUC-L by camera-ready, or (b) relabeling Proposition 3 as a Conjecture so the formal/informal statements are visually separated?

## 7. Recommendation and Scoring (revised)

**Recommendation:** **Strong Accept** *(upgraded from R1 "Accept")*.

**Scores:**

| Criterion | R1 score | R2 score | Change rationale |
|---|---|---|---|
| Originality | 3 / 5 | **4 / 5** | The Max-$k$-Coverage reduction (Thm. 2) is a non-trivial, schema-typing-specific hardness result that distinguishes the paper from generic stochastic-knapsack hardness. The matching $(1-1/e)$-approximation (Thm. 4) and the explicit adaptivity-gap construction (Lemma 5) together form a complete complexity-theoretic characterization of the offline non-adaptive case — the form of result one would expect for a research-track LNCS paper. |
| Significance | 4 / 5 | **4 / 5** | Unchanged. The governance-layer framing remains highly relevant; the empirical-only evaluation continues to cap significance at 4/5. Would move to 5/5 with a live-API pilot. |
| Technical Soundness | 4 / 5 | **5 / 5** | The new theorems are correct (I verified Theorem 2's reduction and Theorem 4's submodularity proof). The coupled L1 concentration is correct and the savings claim is honest. The paper is now technically *tighter* than at R1, with the previously hand-waved Remark 3 ("conjecture; verify as open") replaced by a formal theorem. |
| Clarity | 5 / 5 | **5 / 5** | Unchanged. The new theorems integrate cleanly into §4; the section title change ("Hardness" → "Hardness and Approximability") is appropriate. |

**Overall Rationale (revised):**
The revision substantively addresses the R1 weakness on theoretical novelty. Section 4 is now a complete hardness-and-approximability story rather than a textbook NP-hardness flag, and Appendix A's coupled-concentration step gives the regret analysis a structural improvement over the off-the-shelf BwK template. Combined with the unchanged strengths — strong baseline coverage, exemplary reproducibility, honest treatment of limitations, three substantive ablations — the paper now clears the bar comfortably.

The principal residual weakness remains the synthetic-only empirical evaluation. Even a small live-trace pilot would convert this from a strong-accept into an outright top-of-program paper. Absent that, the formal theory is sufficiently tightened that I am now confident in recommending **Strong Accept** rather than the borderline-accept of R1.

---

## 8. Confidence

**Reviewer confidence: 4 / 5.** I am familiar with bandits-with-knapsacks, QoS-aware service composition, and submodular maximization (the Sviridenko / Nemhauser–Wolsey–Fisher line); I independently verified the Max-$k$-Coverage reduction in Theorem 2 and the concavity-implies-submodularity argument in Theorem 4, and re-aggregated the released CSVs to confirm the headline empirical numbers. I am less specialized in stochastic-knapsack adaptivity gaps and have taken Lemma 5's DGV-embedding sketch on faith pending the full appendix proof.
