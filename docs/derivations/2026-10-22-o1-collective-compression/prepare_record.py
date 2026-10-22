"""Package the O1 analytical certificate; this script does not prove it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT / "problem-contract.json").read_text(encoding="utf-8"))
META = {key: CONTRACT[key] for key in ("schema_version", "contract_id", "target_digest")}


def artifact(path: str) -> dict[str, str]:
    """Record the bytes of one contained supporting artifact."""
    return {"artifact_path": path, "artifact_sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


def write(name: str, fields: dict[str, Any]) -> None:
    """Write a contract-bound release record."""
    (ROOT / name).write_text(json.dumps({**META, **fields}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Bind the accepted scoped derivations and reconstruction records."""
    header = "<!-- rigorous-theory-search-metadata " + json.dumps(META, sort_keys=True, separators=(",", ":")) + " -->"
    for path in ROOT.rglob("*.md"):
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].startswith("<!-- rigorous-theory-search-metadata "):
            lines = lines[1:]
        path.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")

    specs = [
        ("C1", "The normalized three-agent bit/two-circle model has exactly the own-agent Fisher and prescribed null fields C15 and C17, with Gibbs rows C14 and C16.",
         "Positive exponential-family recognition, fixed normalized evaluator and copying/resampling channels, persistent psi-independent likelihoods, positive priors (1/5,2/5,2/5), and exact unit-temperature row reduction.",
         "routes/constructive-per-agent.md", "Sections 1-2, C1-C17, and finite-horizon existence in Section 6.",
         "An incorrect normalization, omitted own derivative, extra source-role derivative, or changed null row would invalidate the field."),
        ("C2", "A relatively open initial class selects a persistent full three-agent block at a common fixed census time, whereas the prescribed null never selects it.",
         "The C1 model, fixed initial public record pattern (+,+,-), arbitrary retained angular records, small independent positive epsilon and delta, public graph rule C34, and fixed census times 0,T.",
         "routes/constructive-per-agent.md", "Section 7, C26-C34: invariant full/null boxes, signed null barrier, stable scaled field, and common finite entrance.",
         "Failure of a box boundary inequality, stability margin, or uniform entrance would defeat the persistent partition contrast."),
        ("C3", "Complete pairwise relative-angle responses identify all nonzero circular laws up to one common rotation; a proper block with any fixed nonzero exterior law has no noninjective local law retention preserving those responses and direct remaining-law queries.",
         "Three nonzero von Mises physical factors on a complete probe graph, direct public/marked and presentation law queries, known retained frame embeddings, and exterior-fixed block-local inheritance.",
         "routes/constructive-per-agent.md", "Section 4, C19-C22, including strict monotonicity of the circular moment ratio and direct-law identifiability.",
         "A zero factor, absent crossing edge, or unrecorded shared exterior reference would change the identification hypothesis."),
        ("C4", "The nonzero-anchor whole-section quotient has normalized canonical parent port laws, exact autonomous original-time free evolution, and every admitted finite adaptive joint response-record history.",
         "Entire C2 contextual sections with one fixed basepoint, common cone, complete records and ports, fresh samples between transactions, actual shared-sample edge kernels, psi-independent evidence, and history-based policies.",
         "routes/constructive-per-agent.md", "Sections 3, 5-6, C23-C25; the joint-kernel argument is also reconstructed as A19-A20 in observability.md.",
         "A zero anchor, exposed absolute psi output, phase-sensitive assimilation, or failure of the actual joint response/update kernel invalidates descent."),
        ("C5", "The section quotient preserves the entire actual joint contextual/marked-orbit Fisher quadratic cost, hence the minimizing connection class and minimized contextual tensor.",
         "A fixed prepared contextual family; complete random histories; fixed lab labels for every stored marked output; covariantly transformed local embeddings and constitutive directions; regular dominated finite-episode experiment; fixed-census events with strict margins.",
         "routes/geometry-history.md", "G12-G17, including G16a, full history score, the two distinct parameter derivatives, and fixed-census regularity.",
         "Pointwise contextual anchoring, an unproved action on numerical history angles, or an omitted event/history score can destroy the cost identity."),
        ("C6", "The same positive-parameter selected family has rank-two actual minimized Fisher tensor and nonzero marked curvature on a common finite post-census window, with exact parent equality.",
         "C1-C2 and C5, fixed positive marked amplitudes independent of epsilon and delta, a compact two-context patch, the same finite T, the initial evidence episode held fixed, and sufficiently small positive parameters.",
         "routes/geometry-history.md", "G18-G20 and the finite-time C2 continuation of fine costs before applying the regular positive-parameter quotient.",
         "A vanishing limiting inertia, a nonuniform entrance time, or using the singular delta=0 anchor would invalidate this bridge."),
        ("C7", "The frozen O1 existential collective alternative is realized by one normalized per-agent model with a selected persistent group, a genuinely collective physical loss, exact original-time dynamics and histories, and actual nonflat inherited contextual geometry.",
         "All explicitly declared witness hypotheses C1-C6, including the whole-section parent, fixed relative protocol, lab-label marked-frame convention, and common finite geometric window; no newly created symmetry or local parent is asserted.",
         "construction-or-strongest-theorem.md", "The assembled same-model theorem and its source, protocol, geometry, physical/gauge and locality boundaries.",
         "A mismatch between the model, complete history convention, selector or parent state across C1-C6 would defeat the assembly."),
        ("C8", "Operational equivalence cannot first appear at finite time along a common regular analytic fixed-history flow segment for an analytic, wait-closed experiment family.",
         "Finite-dimensional analytic field, analytic measure-determining finite-experiment expectations on a common connected domain, prefix-wait closure, and no change of record-conditioned stratum along the compared segment.",
         "routes/observability.md", "Lemma A1 and Theorem A2, with C2, event, finite-record and schedule limitations in Sections 4-6.",
         "Failure of analyticity, common-domain continuation, wait closure, or fixed-history conditions removes this obstruction; preexisting collective equivalence is allowed."),
        ("C9", "For a fixed equivariant sample coarsening retaining history, the minimized Fisher tensor is preserved exactly when the fine horizontal residual is measurable from retained data; equality of minimizing affine classes additionally requires unchanged orbit isotropy.",
         "Regular dominated joint families, the same marked Lie algebra convention, a fixed equivariant sample kernel, and constant fine/coarse orbit ranks; this does not treat a nonlinear law-section retention as a fixed statistic.",
         "routes/geometry-history.md", "G21-G22 and their orthogonal-decomposition proof, with explicit moving-map and finite-record failure cases.",
         "A nonzero residual conditional variance or new orbit isotropy invalidates the corresponding equality."),
    ]
    assumptions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    for cid, statement, hypotheses, path, scope, falsifier in specs:
        assumptions.append({"id": "A-" + cid, "statement": hypotheses, "kind": "ROUTE_HYPOTHESIS", "source": path})
        evidence.append({"id": "E-" + cid, "kind": "DERIVATION", "supports": True, "scope": scope, "side_conditions": [hypotheses], **artifact(path)})
        claims.append({"id": cid, "statement": statement, "quantifiers": hypotheses, "kind": "MATHEMATICAL", "target_digest": META["target_digest"], "state": "EVIDENCE_VERIFIED", "assumption_ids": ["A-" + cid], "evidence_ids": ["E-" + cid], "bridge_premise_ids": [], "falsifier": falsifier})
    target = CONTRACT["target"]
    evidence.append({"id": "E-target", "kind": "DERIVATION", "supports": True, "scope": "Existential collective branch: the complete assembled witness, not interaction-created equivalence or pointwise parent geometry.", "side_conditions": [specs[6][2]], **artifact("construction-or-strongest-theorem.md")})
    claims.append({"id": "target", "statement": target["statement"], "quantifiers": target["quantifiers"], "kind": target["kind"], "target_digest": META["target_digest"], "state": "EVIDENCE_VERIFIED", "assumption_ids": ["A-C7"], "evidence_ids": ["E-target"], "bridge_premise_ids": [], "falsifier": target["falsification_criterion"]})
    write("claim-ledger.json", {"assumptions": assumptions, "evidence": evidence, "claims": claims, "interpretation": "Direct analytical derivations accepted by a same-root sequential reconstruction under their exact hypotheses. Not formal proof, independent adjudication, or numerical evidence for universal claims."})
    edges = [{"from": "target", "to": "C7", "reason": "The explicit same-model witness realizes the existential collective alternative."}]
    edges += [{"from": "C7", "to": cid, "reason": "Required interface of the complete constructive witness."} for cid in ("C1", "C2", "C3", "C4", "C5", "C6")]
    edges += [{"from": "C2", "to": "C1", "reason": "Uses the exact native and prescribed null fields."}, {"from": "C4", "to": "C1", "reason": "Uses native equivariance, complete kernels, and finite-time cone regularity."}, {"from": "C5", "to": "C4", "reason": "Uses the actual normalized section parent and carried histories."}]
    edges += [{"from": "C6", "to": cid, "reason": "Combines the same native model, common grouping time, and exact positive-parameter geometric equality."} for cid in ("C1", "C2", "C5")]
    write("dependency-dag.json", {"edges": edges})

    families = [
        ("analytic-observability", "Continue analytic experiment expectations along actual common trajectories.", ["C8"], "Regular analytic law-state charts and fixed-history experiment functions.", "Wait closure prevents first appearance of future equivalence on one segment.", "Does not cover general C2 data, arbitrary record changes, or rule out a preexisting collective quotient."),
        ("relative-angle-collectivity", "Transmit psi between peers and expose all relative laws on a homogeneous complete graph.", ["C1", "C3", "C4"], "Positive bit/two-circle exponential families and exact Gibbs rows.", "Pair first moments identify every proper block against its exterior, but preserve one global rotation.", "The symmetry is predeclared; a pointwise quotient alone does not preserve contextual geometry."),
        ("fixed-score-sufficiency", "Compute the exact conditional-variance and orbit-isotropy loss under a fixed sample coarsening.", ["C9"], "Joint record/current-state L2 score spaces.", "Horizontal residual measurability is necessary and sufficient for the minimized tensor.", "A law-dependent section canonicalization is not a fixed sample statistic."),
        ("pointwise-canonicalization", "Choose a separate physical-angle representative at every evaluation context.", [], "Context-dependent sample diffeomorphisms.", "The moving-map score removes positive private phase information.", "Retired: it fails the required geometric equality on explicit nonflat families."),
        ("anchored-record-geometry", "Use one contextual basepoint and fixed lab history labels to construct an actual joint-sample isometry.", ["C5", "C6"], "Whole C2 sections, complete histories, and a passive marked-frame action on local embeddings.", "Context and orbit scores are both preserved; a two-mark limit gives a common nonflat window.", "No pointwise or fixed-size parent result; exact crossing timestamps require another regular experiment."),
        ("integrated-collective-o1", "Combine native persistent grouping with complete-graph identification and the whole-section geometric parent.", ["C2", "C7"], "One normalized model and fixed protocol with two small positive parameters.", "Common census time, null barrier, explicit quotient field, joint kernels, and actual full-cost equality.", "The collective existential branch closes; interaction-created invisibility and second-stage recursion are not claimed."),
    ]
    write("approach-registry.json", {
        "mechanism_families": [{"family_id": fid, "core_mechanism": mechanism, "target_obligation_ids": ["target"], "representation": representation, "invariant_or_obstruction": invariant, "obligations": [scope], "bridge": invariant, "failure_test": scope, "verified_results": results, "open_gaps": [scope], "novelty_fingerprint": mechanism + " Compared with the consulted local sources; no global novelty claim.", "disposition": "Retired after explicit geometric failure." if fid == "pointwise-canonicalization" else "Scoped direct analytical result."} for fid, mechanism, results, representation, invariant, scope in families],
        "rounds": [{"round": 1, "result": "Analytic obstruction, explicit transmitted-angle collective model, exact fields, and whole-section candidate."}, {"round": 2, "result": "Same-model grouping and nonflat window assembled; fixed census, fixed lab labels, direct-query labels, and history-based policies close the identified geometry/history interfaces."}],
        "stopping_reason": "The existential collective witness is assembled under its explicit scope. No further route search or background continuation is needed for this request."
    })

    all_ids = [claim["id"] for claim in claims]
    attacks = [
        ("native-and-null", ["C1", "C2", "C7", "target"], "The construction might reuse reciprocal derivatives or alter the null self row, making the grouping contrast inapplicable.", "C10-C17 derive the own-agent envelope field and retained-prior null; C26-C34 prove its invariant null barrier and full stable entrance.", "REJECTED"),
        ("hidden-primitive", ["C3", "C4", "C7", "target"], "The final collective witness might simply rename the old invisible primitive factor because its global symmetry preexists selection.", "Rejected for the exact final claim: C19-C22 identify every proper block against a nonzero exterior and exactly one global fiber. The frozen target permits genuinely collective loss. The stronger interaction-created-invisibility claim is excluded and remains open.", "REJECTED"),
        ("incomplete-joint-history", ["C4", "C7", "target"], "Separate marginals, dropped records, raw absolute-angle outputs, or unstated sample reuse could invalidate exact histories.", "The protocol uses actual joint edge pushforwards, fresh transactions, complete history, fixed lab labels, and joint response/next-state descent. Absolute psi outputs are excluded initially and no probe is retired.", "REJECTED"),
        ("pointwise-and-record-geometry", ["C5", "C6", "C7", "target"], "The final whole-section history-dependent map might still remove contextual or orbit scores through a moving evaluation anchor or an unproved marked action.", "Rejected after the explicit interface corrections: the final parent retains a whole section, G16a fixes every stored lab label and transforms embeddings, and history-based policies preserve covariance. G14-G17 prove full-cost equality. The failed pointwise and numerical-history-action variants remain documented as counterexamples, outside the final claims.", "REJECTED"),
        ("event-and-zero-limit", ["C2", "C5", "C6", "C7", "target"], "Exact event timestamps or a singular zero-anchor limit could invalidate regular Fisher geometry at the same grouping time.", "The fixed census is declared before execution and all its data are retained with strict uniform margins. The fine costs are continued from the zero-parameter limit; canonicalization is applied only at positive parameters with a nonzero cone anchor.", "REJECTED"),
        ("analytic-overreach", ["C8"], "An analytic segment argument could be promoted to universal nonexistence across C2 laws or record-changing jumps.", "A1-A2 use exact analytic wait-closure and common-domain hypotheses. A6 and the record/event analysis delimit the conclusion; no universal negative claim is made.", "REJECTED"),
        ("records-repair-score", ["C9"], "Retained noisy history might compensate a positive conditional-current horizontal score loss.", "The orthogonal decomposition G21-G22 proves the two nonnegative defects. History-only scores are already retained and cannot cancel the missing conditional residual; G23 gives a strict example.", "REJECTED"),
    ]
    reconstruction = artifact("proof-reconstruction.md")
    write("adversarial-report.json", {
        "workflow": "Substantive analytical failure tests and same-root sequential reconstruction; no independent adjudication panel.",
        "attacks": [{"id": aid, "claim_ids": ids, "attack": attack, "response": response, "disposition": disposition, **reconstruction} for aid, ids, attack, response, disposition in attacks],
        "independent_reconstruction": {"claim_ids": all_ids, "method": "A labeled root pass from the frozen contract, rebuilding the normalization, fields, bounds, kernels, and score interfaces; not independent corroboration.", "result": "PASS", "conclusion": "The whole-section collective witness reconstructs under the exact final protocol and frame convention. Stronger locality and interaction-created-equivalence claims do not follow.", **reconstruction},
        "oracle_erasure": {"claim_ids": all_ids, "method": "Remove expectations of a positive outcome and inspect every assumption, dependency and bridge; the frozen contract contains no affirmative search prior.", "result": "PASS", "conclusion": "The direct model formulas, inequalities, kernel maps and score identities establish the scoped witness without assuming its conclusions. Scope restrictions remain explicit.", **reconstruction}
    })
    write("release.json", {
        "checkpoint": "scoped collective O1 analytical release",
        "target_claim": "target",
        "terminal_status": "COMPLETE_AFFIRMATIVE",
        "certificate_claim": "target",
        "strongest_result": "A genuinely collective one-angle physical quotient under current normalized per-agent dynamics, with persistent full/null-separated grouping, exact original-time whole-section parent dynamics, complete admitted histories, and actual nonflat inherited contextual geometry on a common finite window.",
        "unresolved_obligations": [],
        "outside_scope_open": ["Interaction-created operational invisibility.", "Pointwise local or fixed-size parent with the same geometry.", "Regular geometry with continuously reported parameter-dependent crossing times.", "Second-stage recursion and future_TODO extensions."],
        "certificate_limitations": "Direct analytical derivations and same-root sequential reconstruction. Whole contextual section and fixed relative/lab-label protocol are required. No numerical parameter threshold, simulation, formal proof, independent adjudication, or infinite-time curvature bound."
    })
    print(json.dumps({"claims": len(claims), "direct_derivations": len(evidence), "terminal_status": "COMPLETE_AFFIRMATIVE"}))


if __name__ == "__main__":
    main()
