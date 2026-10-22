"""Package scoped analytical results; this script does not prove them."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT / "problem-contract.json").read_text(encoding="utf-8"))
META = {key: CONTRACT[key] for key in ("schema_version", "contract_id", "target_digest")}


def artifact(path: str) -> dict[str, str]:
    """Identify the exact supporting derivation."""
    return {"artifact_path": path, "artifact_sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


def write(name: str, fields: dict[str, Any]) -> None:
    """Write one contract-bound record within this investigation."""
    (ROOT / name).write_text(json.dumps({**META, **fields}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Save direct derivations, explicit hypotheses, and open obligations."""
    header = "<!-- rigorous-theory-search-metadata " + json.dumps(META, sort_keys=True, separators=(",", ":")) + " -->"
    for path in ROOT.rglob("*.md"):
        body = path.read_text(encoding="utf-8").splitlines()
        if body and body[0].startswith("<!--"):
            body = body[1:]
        path.write_text(header + "\n" + "\n".join(body) + "\n", encoding="utf-8")

    specs = [
        ("C1", "The four-state conditional-interface model has the exact per-agent and prescribed null fields displayed in the transfer route; reciprocal equilibrium and stability claims require new conditions.",
         "Positive four-state laws, normalized fixed local evaluator, complete sender-dependent channel substitution, exact unit-temperature Gibbs rows, product recognition, and own-block Fisher dynamics.",
         "routes/per-agent-transfer.md", "Sections 1-5, equations (1)-(25).",
         "An omitted source derivative within an own objective, incorrect self-row weighting, or a failed normalization would invalidate the transfer."),
        ("C2", "The transferred four-state model admits a full-open interaction-dependent selected pair with persistent nonconstant records and a nonzero persistence window.",
         "The N=2 constants (26), sufficiently small positive epsilon and record strength h, and the explicit threshold selector on a full open positive-table neighborhood.",
         "routes/per-agent-transfer.md", "Section 6, equations (26)-(29), full-table null cone and finite-time crossing proof.",
         "Failure of the strict null sign cone or the full-flow threshold crossing would invalidate F5 for this witness."),
        ("C3", "Three identical per-agent species with conflicting persistent records form a persistent full block on an open class, whereas the prescribed null never forms that block.",
         "The positive Bernoulli model, pi_self=1/5, pi_peer=2/5, record vector (+,+,-), sufficiently small positive epsilon, and selector (R7).",
         "routes/persistent-record-grouping.md", "R1-R7, stable affine limit, local attraction, and all-time null sign invariance.",
         "A failure of the stable equilibrium or the strict graph margins would invalidate the claimed partition contrast."),
        ("C4", "The marked-spectrum and canonical-circle parents preserve exact original-time port dynamics and all admitted finite adaptive response-record histories while deleting physical hidden phases.",
         "The declared invariant circle family and fixed phase-insensitive pre-handoff protocol, full retained records, shape parameters, primitive ports and event data; product port inheritance is used.",
         "routes/operational-compression.md", "O14-O38, with separate recognition-dependent and fixed-own-model channel conventions.",
         "An admitted hidden-phase response or a retained update depending on the erased phase defeats the corresponding quotient."),
        ("C5", "The integrated fixed-own-model circle extension retains a persistent interaction-dependent group with an all-time null obstruction.",
         "The hybrid model and regular invariant family in the bridge, weak positive epsilon, J=epsilon^3, positive fixed-small marked amplitudes, and its predeclared public selector.",
         "routes/integrated-grouping-bridge.md", "Exact parameter fields, invariant null box, stable full-flow limit, and fixed-amplitude compatibility with geometry.",
         "A false Gibbs-row cancellation, failure of the null box, or loss of the public sign and ratio margins invalidates this bridge."),
        ("C6", "The canonical phase parent preserves the actual collective minimizing connection classes and contextual Fisher tensors, including retained history scores, on the declared regular nonflat family.",
         "Context-constant discarded phases, marked-circle passive frame action with constitutive and record-coordinate transport, actual normalized comparison experiment, and a fixed finite grouping/geometry window.",
         "routes/record-geometry.md", "Record-sensitive score isometry, nonflat two-mark calculation, and integrated fixed-own-model extension.",
         "A context-dependent erased phase, omitted random-history score, or changed frame action can invalidate equality; nonflatness is only asserted on the declared patch."),
        ("C7", "One normalized per-agent hybrid combines interaction-dependent grouping with physically noninjective conditional inheritance of port dynamics, histories, persistent records and nonflat collective geometry.",
         "All hypotheses of C4-C6, including a declared phase-insensitive interface fixed before selection and the retained primitive port architecture.",
         "construction-or-strongest-theorem.md", "The assembled conditional theorem and its explicit interface, protocol, geometry and recursion restrictions.",
         "Any mismatch of the constitutive model across grouping, quotient and geometry would invalidate assembly; the result does not assert endogenous creation of the interface."),
    ]
    assumptions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    for cid, statement, quantifiers, path, scope, falsifier in specs:
        assumptions.append({"id": "A-" + cid, "statement": quantifiers, "kind": "ROUTE_HYPOTHESIS", "source": path})
        evidence.append({"id": "E-" + cid, "kind": "DERIVATION", "supports": True, "scope": scope, "side_conditions": [quantifiers], **artifact(path)})
        claims.append({"id": cid, "statement": statement, "quantifiers": quantifiers, "kind": "MATHEMATICAL", "target_digest": META["target_digest"], "state": "EVIDENCE_VERIFIED", "assumption_ids": ["A-" + cid], "evidence_ids": ["E-" + cid], "bridge_premise_ids": [], "falsifier": falsifier})
    obligations = {
        "O1": "Derive an interaction-selected sufficient interface or genuinely collective physical loss whose noninjectivity is not supplied by a permanently invisible primitive factor, with the same exact field, complete response-record laws, and nonflat geometry.",
        "O2": "Extension: construct a second genuinely noninjective inherited handoff and prove the specified counterfactual dependence; concatenating already canonical ports is insufficient."
    }
    for cid, statement in obligations.items():
        claims.append({"id": cid, "statement": statement, "quantifiers": "Existence within the frozen target and its explicitly declared recursive extension; no witness is supplied here.", "kind": "MATHEMATICAL", "target_digest": META["target_digest"], "state": "INCONCLUSIVE", "assumption_ids": [], "evidence_ids": [], "bridge_premise_ids": [], "falsifier": "An admissible complete construction closes existence; failure of this particular candidate does not prove nonexistence."})
    target = CONTRACT["target"]
    claims.append({"id": "target", "statement": target["statement"], "quantifiers": target["quantifiers"], "kind": target["kind"], "target_digest": META["target_digest"], "state": "INCONCLUSIVE", "assumption_ids": [], "evidence_ids": [], "bridge_premise_ids": [], "falsifier": target["falsification_criterion"]})
    write("claim-ledger.json", {"assumptions": assumptions, "evidence": evidence, "claims": claims, "interpretation": "EVIDENCE_VERIFIED denotes root-accepted direct analytical derivations under their explicit hypotheses. It does not denote formal proof, numerical verification of the theorem, or independent adjudication."})
    edges = [{"from": "target", "to": cid, "reason": "Scoped result or open obligation characterizing this investigation; O2 is the explicitly declared extension."} for cid in [s[0] for s in specs] + list(obligations)]
    edges += [{"from": "C2", "to": "C1", "reason": "Uses the transferred own-block and null fields."}]
    edges += [{"from": "C7", "to": cid, "reason": "Same-model assembly of dynamics, response-record and geometry inheritance."} for cid in ("C4", "C5", "C6")]
    write("dependency-dag.json", {"edges": edges})
    families = [
        ("four-state-transfer", "Rederive own-objective conditional-channel dynamics and compare full and null sign evolution.", ["C1", "C2"], "Finite positive tables and conditional channels.", "Full-table null sign cones and a transverse selected crossing.", "No compressed parent in this route."),
        ("persistent-record-consensus", "Use peer influence to overcome opposing persistent record forces.", ["C3"], "Three Bernoulli law pairs with positive source priors.", "Stable affine small-parameter limit versus all-time null sign invariance.", "Grouping alone preserves every child coordinate."),
        ("marked-conditional-spectrum", "Retain public laws and record-marked spectra, then realize them by regular canonical laws.", ["C4"], "General bounded profiles and smooth two-circle invariant families.", "Exact joint response/next-state factorization.", "Invisible phase supplied by the declared interface."),
        ("record-inclusive-geometry", "Carry actual history and current-state scores through phase canonicalization.", ["C6"], "Regular dominated families with a marked-circle comparison action.", "Equality of complete collective quadratic costs.", "Context-varying discarded phases and phase-reading records fail."),
        ("integrated-fixed-model-hybrid", "Couple marked angle and public sign through a weak normalized evaluator while retaining fixed-own-model semantics.", ["C5", "C7"], "Binary public variable and two compact angular coordinates.", "Common Gibbs cost cancellation, invariant null box, and exact canonical phase quotient.", "Grouping is endogenous; sufficient phase interface is predeclared. Second compressive join remains open.")
    ]
    write("approach-registry.json", {"mechanism_families": [{"family_id": fid, "core_mechanism": mechanism, "target_obligation_ids": ["target", "O1", "O2"], "representation": rep, "invariant_or_obstruction": invariant, "obligations": [gap], "bridge": invariant, "failure_test": gap, "verified_results": results, "open_gaps": [gap], "novelty_fingerprint": mechanism + " Relative to consulted local sources; no worldwide priority claim.", "disposition": "Scoped direct derivation; complete target remains open."} for fid, mechanism, results, rep, invariant, gap in families], "rounds": [{"round": 1, "result": "Exact per-agent transfer, persistent-record grouping, conditional operational compression, and record-sensitive geometry."}, {"round": 2, "result": "One fixed-own-model hybrid combines persistent grouping, exact conditional compression and finite-window nonflat geometry; an invariant null box supplies the all-time control."}], "stopping_reason": "The combined route gives checkable progress but retains a predeclared invisible factor. A new selected-interface mechanism is required to close O1; no background continuation scheduled."})
    all_ids = [claim["id"] for claim in claims]
    reconstruction = artifact("proof-reconstruction.md")
    write("adversarial-report.json", {
        "workflow": "Same-root sequential reconstruction and concrete failure tests; no independent adjudication panel.",
        "attacks": [{"id": "model-interface-history-geometry", "claim_ids": all_ids, "attack": "The per-agent force, null control, persistent records, geometric experiment or compression interface might be changed between routes, or a pre-existing quotient mistaken for formation.", "response": "The integrated route explicitly fixes one model, proves the all-time null bound, retains complete records and port state, and proves cost equality on a declared section. The supplied phase-invisible interface remains an unresolved formation gap.", "disposition": "PARTIALLY_SUSTAINED", **reconstruction}],
        "independent_reconstruction": {"claim_ids": all_ids, "method": "Sequential root reconstruction from the frozen contract; not independent corroboration.", "result": "PASS", "conclusion": "Scoped derivations reconstruct; the endogenous-interface and second-stage extensions do not follow.", **reconstruction},
        "oracle_erasure": {"claim_ids": all_ids, "method": "Remove any expectation of a positive target and inspect every premise and bridge; the contract has no search prior.", "result": "PASS", "conclusion": "The conditional theorem remains valid with its supplied interface; no complete existential or negative certificate results.", **reconstruction}
    })
    write("release.json", {"checkpoint": "analytical partial-results release", "target_claim": "target", "terminal_status": "INCONCLUSIVE", "certificate_claim": None, "strongest_result": "Exact per-agent transfer and persistent interaction-dependent grouping, plus a same-model conditional phase-compressive parent preserving port dynamics, admitted histories, records and nonflat finite-window collective geometry.", "unresolved_obligations": [key + ": " + value for key, value in obligations.items()], "certificate_limitations": "Direct analytical derivations and same-root sequential reconstruction. The sufficient invisible interface is supplied from the outset. Seven symbolic identities are ancillary checks; no dynamics simulation or independent adjudication."})
    research = ROOT.parents[3] / "Research"
    manuscript = research / "manuscripts" / "MAgent_Theory_2.0"
    source_paths = [manuscript / "AUTHORING.md"]
    source_paths += [manuscript / "chapters" / name for name in (
        "02_geometry.tex", "03_dynamics.tex", "05_inheritance.tex",
        "07_smooth_closure.tex", "09_public_compression.tex", "12_open_formation.tex")]
    source_paths += [research / "wiki" / "concepts" / "Communication-constrained inference.md",
                     ROOT.parent / "2026-10-22-formation-next-step-reassessment.md"]
    previous = ROOT.parent / "2026-10-22-endogenous-interface-selection" / "routes"
    source_paths += [previous / name for name in (
        "variational-selection.md", "history-selection.md",
        "selection-observability-bridge.md", "moving-interface-geometry.md")]
    revisions = {}
    for repo in (ROOT.parents[2], research):
        revisions[str(repo)] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    write("source-manifest.json", {
        "revisions": revisions,
        "scope": "Current source bytes read for continuity. No remote synchronization or source correctness is certified by these identifiers.",
        "sources": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in source_paths],
    })
    print(json.dumps({"claims": len(claims), "direct_derivations": len(evidence), "terminal_status": "INCONCLUSIVE"}))


if __name__ == "__main__":
    main()
