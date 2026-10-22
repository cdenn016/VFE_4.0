"""Bind the scoped derivations and snapshot their consulted source revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RESEARCH = REPO.parent / "Research"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    problem = json.loads((ROOT / "problem-contract.json").read_text(encoding="utf-8"))
    binding = {k: problem[k] for k in ("schema_version", "contract_id", "target_digest")}
    canonical = json.dumps(problem["target"], ensure_ascii=False,
                           sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == binding["target_digest"]
    header = "<!-- rigorous-theory-search-metadata " + json.dumps(binding, sort_keys=True) + " -->\n"
    for path in ROOT.glob("*.md"):
        body = path.read_text(encoding="utf-8")
        if body.startswith("<!-- rigorous-theory-search-metadata "):
            body = body.split("\n", 1)[1]
        path.write_bytes((header + body).encode())

    def write(name: str, data: dict[str, Any]) -> None:
        (ROOT / name).write_bytes((json.dumps({**binding, **data}, indent=2,
                                            ensure_ascii=False) + "\n").encode())

    assumptions = [
        ("A-native", "Exact normalized positive three-agent O1 laws, original own-block Fisher field and Gibbs rows, pointwise contextual evolution, physical equivariance, nonzero cone sum on each regular finite branch; no changed force or time."),
        ("A-history", "Every original fixed joint response/update kernel, shared relative samples, admitted history-based policies, actual preparation/past-law initialization or canonical replay seed, all raw histories, persistent evidence and census records; diagnostics do not assimilate."),
        ("A-coherence", "At the actual initialization t_* the collective transmitted-angle direction S/|S| is constant across context and independent of subsequent history. Zero phase memory runs from that time and is retained at later handoffs. This is an additional preparation condition."),
        ("A-geometry", "Regular actual joint history/current experiment with all history and conditioning scores, fixed lab labels, marked action distinct from physical phase, equality of complete minimizing affine connection classes and tensors, and the original finite positive-rank nonflat window."),
        ("A-path", "Strict spatial-order-zero causal retention of the full local native value path and actual records, invariant after counting every memory under constant physical rotation; no derivative, neighboring/reference context or calibrated contextual phase seed. Arbitrary temporal memory size is allowed. For stochastic retention, the complete retained kernel is invariant and has declared regular actual laws."),
        ("A-jet", "Finite-dimensional smooth deterministic memory, locally initialized, evolved and read out at one uniform finite contextual jet order, branchwise smooth record updates and no independent/reference phase field; rich smooth open preparation class with independent jets, invertible finite native flow on its regular reached image, and positive actual-law Fisher tangents."),
    ]
    specs = [
        ("C-local", "The local slice r and zero-initialized causal phase accumulator have exact native original-time actual-law reconstruction and closed contextual two-jet evolution on every regular finite branch.", "causal-phase-memory.md", "Sections 1-3, C1-C8: normalization, cone, local angular velocity, native cancellation, two-jet prolongation and actual law germs.", ["A-native"]),
        ("C-history", "With the actual original preparation/past law retained, the local causal reconstruction preserves every admitted finite adaptive joint response/update history, persistent evidence and census, for arbitrary initial phase profiles.", "causal-phase-memory.md", "Section 4: fixed-context constant sample rotation, shared-query law, transaction induction, preparation and handoff initialization.", ["A-native", "A-history"]),
        ("C-geometry", "Under explicit initial collective-phase coherence, the causal local reconstruction erases one common physical angle and preserves actual Fisher costs, marked minimizers, isotropy, tensor and curvature, with an O1 positive-rank nonflat witness on the same finite window.", "causal-phase-memory.md", "Section 5, C9-C11: constant sample isomorphism, counted memory, phase loss and nonempty marked finite-window witness.", ["A-native", "A-history", "A-coherence", "A-geometry"]),
        ("C-defect", "For arbitrary initial collective phase alpha_0(c), the accumulator's actual contextual score differs by minus d alpha_0 times the physical phase score; temporal integration supplies the phase change but not the initial spatial phase one-form.", "causal-phase-memory.md", "Section 6, C12-C13: parameter-dependent density differentiation and temporal-spatial compatibility.", ["A-native", "A-history", "A-geometry"]),
        ("C-path", "No in-scope rotation-invariant causal retention of the entire local value path and records can universally reproduce the actual minimized Fisher tensor on the rich open O1 preparation class, even with unbounded temporal memory.", "causal-obstructions.md", "Sections 1-3, P1-P5: phase-field history identity, retained-law identity and positive horizontal physical-score quadratic.", ["A-native", "A-history", "A-geometry", "A-path"]),
        ("C-jet", "No in-scope smooth finite-dimensional deterministic causal memory reading a uniformly bounded contextual jet can erase the common physical phase and universally inherit both original minimizing marked affine classes and tensors on the rich smooth open preparation class.", "causal-obstructions.md", "Sections 4-5, J1-J3: branchwise finite-order reduction, joint-L2 highest-jet descent and physical-score contradiction; precise exclusions.", ["A-native", "A-history", "A-geometry", "A-jet"]),
    ]
    evidence = []
    claims = [{"id": "target", "statement": problem["target"]["statement"],
               "quantifiers": problem["target"]["quantifiers"],
               "target_digest": binding["target_digest"], "kind": "MATHEMATICAL",
               "state": "INCONCLUSIVE" if args.release else "CANDIDATE",
               "assumption_ids": ["A-native", "A-history", "A-geometry"],
               "evidence_ids": [], "bridge_premise_ids": [],
               "falsifier": problem["target"]["falsification_criterion"]}]
    hypothesis = dict(assumptions)
    for cid, statement, artifact, scope, aids in specs:
        eid = "E-" + cid[2:]
        evidence.append({"id": eid, "kind": "DERIVATION", "supports": True,
                         "scope": scope, "side_conditions": [hypothesis[a] for a in aids],
                         "artifact_path": artifact, "artifact_sha256": sha(ROOT / artifact)})
        claims.append({"id": cid, "statement": statement,
                       "quantifiers": "Every instance satisfying the explicit route hypotheses and exact scope of the named derivation; no unrestricted claim.",
                       "target_digest": binding["target_digest"], "kind": "MATHEMATICAL",
                       "state": "EVIDENCE_VERIFIED", "assumption_ids": aids,
                       "evidence_ids": [eid], "bridge_premise_ids": [],
                       "falsifier": "An in-scope instance violating the displayed identity or the precise hypothesis-mapped derivation; a mechanism outside the hypotheses is not a refutation."})
    write("claim-ledger.json", {"assumptions": [
        {"id": a, "kind": "ROUTE_HYPOTHESIS", "statement": s} for a, s in assumptions
    ], "evidence": evidence, "claims": claims})
    edges = [{"from": "target", "to": c[0]} for c in specs]
    edges += [{"from": "C-history", "to": "C-local"},
              {"from": "C-geometry", "to": "C-local"},
              {"from": "C-geometry", "to": "C-history"}]
    write("dependency-dag.json", {"edges": edges})

    families = [
        ("causal-local-initial-frame", "Integrate the local native angular velocity from zero and put the generated phase into the actual native laws.", ["C-local", "C-history", "C-geometry", "C-defect"], "Local slice, one phase memory, bounded two-jet, canonical local preparation seed and full records.", "Native field cancellation and exact constant-sample-map equality under initial coherence.", "Initial phase-gradient defect outside the coherent subfamily.", "A locally initialized causal implementation of the source's frozen-frame idea; no novelty claim for freezing frames.", "conditional construction"),
        ("phase-field-path-obstruction", "Exploit time-constant contextual physical-phase rotations that preserve every local path modulo rotation and every actual history.", ["C-path"], "Arbitrary temporal causal memory with strictly spatial-order-zero invariant input.", "The retained law is fixed while the fine minimized Fisher cost has a positive quadratic phase term.", "Spatial derivative input or an initialized contextual phase relation changes the hypotheses.", "No temporal-memory dimension or horizon bound; spatial input and invariance are load-bearing.", "scoped obstruction"),
        ("bounded-jet-memory-obstruction", "Reduce smooth generated finite memory to a finite-order current-jet law on each fixed native history, then descend the highest jet.", ["C-jet"], "Uniform finite contextual order, finite-dimensional deterministic smooth recurrence and actual positive law chart.", "A fixed common minimizing affine class makes the high-jet norm polynomial constant only when every derivative-law coefficient vanishes.", "Nonfinite differential input, stochastic memory outside P or constrained preparation are not excluded.", "Causal recurrence reduction adds a class result to the earlier finite differential-order obstruction.", "scoped obstruction"),
    ]
    write("approach-registry.json", {"mechanism_families": [
        {"family_id": fid, "core_mechanism": mech, "target_obligation_ids": ids,
         "representation": rep, "invariant_or_obstruction": inv,
         "obligations": ["Prove the named statements with their exact preparation, causal-input and geometric quantifiers."],
         "bridge": "No hybrid route is used; each mechanism has its own direct derivation.",
         "failure_test": failure, "verified_results": ids,
         "open_gaps": [failure], "novelty_fingerprint": novelty, "disposition": disp}
        for fid, mech, ids, rep, inv, failure, novelty, disp in families
    ]})

    all_ids = [c["id"] for c in claims]
    proof = {"artifact_path": "proof-reconstruction.md",
             "artifact_sha256": sha(ROOT / "proof-reconstruction.md")}
    attacks = [
        ("AT-initial-coherence", ["target", "C-geometry", "C-defect"], "Temporal memory might silently assume it generates initial contextual coherence.", "PARTIALLY_SUSTAINED", "R3 and C12 keep C9 as an additional preparation hypothesis; unrestricted existence remains open."),
        ("AT-handoff", ["target", "C-local", "C-history"], "A blank accumulator at the selected census may delete its actual prehistory.", "PARTIALLY_SUSTAINED", "R1-R2 require running from actual t_* and retaining its value; zero at T is a different initialization requiring its own coherence."),
        ("AT-density", ["C-local", "C-geometry"], "The phase and tensor might be metadata with no effect on sampled normalized parent laws.", "REJECTED", "R1 and R3 sample q(z), whose C8 derivatives include the generated phase, and derive the full actual-law cost isomorphism."),
        ("AT-existing-frame", ["target", "C-local"], "The purported mechanism may just rename the already known frozen reference frame.", "PARTIALLY_SUSTAINED", "C5 and R3 identify the predecessor explicitly; the result is a conditional local causal implementation and new memory-scope obstruction, not new invisibility."),
        ("AT-unbounded-time-memory", ["C-path"], "Storing the whole local value path instead of a finite statistic could evade the phase obstruction.", "REJECTED", "R4 already allows the full path and arbitrary temporal memory; each path is the same modulo one constant local rotation under P1."),
        ("AT-one-ansatz", ["target", "C-jet"], "A failed fixed-jet recurrence is insufficient for a general no-go theorem.", "PARTIALLY_SUSTAINED", "R5 proves the complete declared bounded-jet smooth-memory class by reduction and descent. Mechanisms outside that class remain unresolved, so no general no-go is claimed."),
        ("AT-averaged-history", ["C-history", "C-path", "C-jet"], "Averaged Fisher equality cannot simply be conditioned on each recorded branch.", "REJECTED", "R2 proves path-law equality by kernel induction; R4 uses joint score identities; R5 uses the nonnegative joint-L2 quadratic coefficient and finite-basis invertibility."),
        ("AT-mixed-connections", ["C-geometry", "C-defect"], "An exact physical spatial phase one-form may have been confused with the nonflat marked Fisher connection.", "REJECTED", "R3 distinguishes their actions and derivations. The marked connection remains the actual nonflat C11 connection; C13 is a separate initialization identity."),
    ]
    write("adversarial-report.json", {
        "provenance": "Sequential same-model passes, no independent agents, corroboration or adjudication.",
        "attacks": [{"id": aid, "claim_ids": ids, "attack": a,
                     "disposition": d, "response": response, **proof}
                    for aid, ids, a, d, response in attacks],
        "independent_reconstruction": {
            "claim_ids": all_ids, "method": "Sequential same-model reconstruction from the frozen native equations; R1-R5. Role-separated passes are not independent evidence.",
            "result": "PASS", "conclusion": "All six scoped derivations retain their assumptions; the unrestricted target remains INCONCLUSIVE.", **proof},
        "oracle_erasure": {
            "claim_ids": all_ids, "method": "Semantic erasure of desired conclusions and scan of the complete dependency closure, including explicit initial-coherence and finite-input restrictions.",
            "result": "PASS", "conclusion": "No affirmative existence assumption or paraphrased unrestricted impossibility is used. This does not prove the target.", **proof},
    })
    write("release.json", {
        "checkpoint": "final scoped release" if args.release else "derived-results checkpoint",
        "target_claim": "target", "terminal_status": "INCONCLUSIVE" if args.release else None,
        "certificate_claim": "target" if args.release else None,
        "strongest_result": "Exact causal local phase accumulator with two-jet closure, all fixed history inheritance and actual nonflat Fisher inheritance under initial collective-phase coherence; arbitrary-temporal-value-path and smooth bounded-jet causal-memory obstructions on the rich open class.",
        "unresolved_obligations": [
            "An admissible causal source of initial contextual phase coherence outside both obstruction classes, or a scope-matched impossibility proof for remaining mechanisms.",
            "Genuinely unbounded contextual-order, differently specified infinite-dimensional and stochastic memories outside P and J remain unclassified.",
            "Initial coherence has not been generated by the native dynamics; blank initialization at a later census is not automatically compatible with the actual past.",
        ],
    })

    if not (ROOT / "source-manifest.json").exists():
        old = REPO / "docs/derivations/2026-10-22-o1-collective-compression"
        local = REPO / "docs/derivations/2026-10-22-o1-local-parent"
        chap = RESEARCH / "manuscripts/MAgent_Theory_2.0/chapters"
        paths = [old / f for f in ("problem-contract.json", "construction-or-strongest-theorem.md", "counterexample-register.md", "routes/constructive-per-agent.md", "routes/observability.md", "routes/geometry-history.md")]
        paths += [local / f for f in ("problem-contract.json", "construction-or-strongest-theorem.md", "local-realization.md", "locality-obstructions.md", "counterexample-register.md")]
        paths += [chap / f for f in ("02_geometry.tex", "03_dynamics.tex", "05_inheritance.tex", "13_per_agent_inheritance.tex", "14_collective_local_parent.tex")]
        paths += [RESEARCH / "wiki/concepts" / f for f in ("Belief Compression.md", "Fisher information metric.md")]
        write("source-manifest.json", {
            "scope": "Consulted source revisions at packaging checkpoint; sources and Research remain read-only. Not a claim about every file in either repository.",
            "vfe_head": git(REPO, "rev-parse", "HEAD"),
            "research_head": git(RESEARCH, "rev-parse", "HEAD"),
            "research_status": git(RESEARCH, "status", "--short"),
            "sources": [{"path": str(p.resolve()), "sha256": sha(p)} for p in paths],
        })
    print(json.dumps({"target_digest": binding["target_digest"],
                      "mode": "release" if args.release else "checkpoint",
                      "scoped_claims": len(specs)}, indent=2))


if __name__ == "__main__":
    main()
