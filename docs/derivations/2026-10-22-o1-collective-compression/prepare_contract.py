"""Freeze the bounded O1 target and record the consulted source revisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    """Bind the initial research record to the complete O1 target."""
    root = Path(__file__).resolve().parent
    previous = root.parent / "2026-10-22-per-agent-formation" / "problem-contract.json"
    target = json.loads(previous.read_text(encoding="utf-8"))["target"]
    target["statement"] = "Resolve O1: construct genuinely collective or interaction-selected physical compression under current normalized per-agent dynamics, preserving exact original-time inherited dynamics, all fixed-protocol joint histories and persistent records, and actual nonflat collective geometry, without a permanently invisible factor; or establish a scope-matched obstruction."
    target["quantifiers"] = "There exist a finite population under the normalized per-agent mixture-source model, positive fixed source priors and a declared unit-temperature row convention, fixed normalized constitutive data and a fixed nonempty operational protocol, and an initially unpartitioned nonempty open class or declared invariant family, such that interaction selects a persistent nontrivial block differently from the prescribed null after event times are removed. Its parent retention is genuinely noninjective in a positive-dimensional nongauge physical distinction after interface state, event information and complete persistent records are included. The loss must be genuinely collective or interaction-selected and cannot be supplied by a permanently invisible primitive factor or a quotient available for every partition throughout the admitted class. The parent has normalized laws and exact original-time flow, preserves every admitted finite adaptive joint response-record history, and preserves the actual collective minimizing connection class and positive-rank Fisher tensor with nonzero curvature on a declared common regular window. Second-stage recursion and every future_TODO extension are excluded from this target."
    target["codomains"] = [
        "A one-stage selected parent with inherited normalized belief/model laws, evaluators, operational kernels, complete required memory and autonomous original-time dynamics.",
        "Every admitted joint adaptive response-record process, including response/next-state dependence and events, and actual collective nonflat Fisher geometry on a common declared window.",
        "If incomplete, proved scope-specific obstructions and exact conditions still required by an admissible construction; no general negative claim follows from an analytic or finite-table special case.",
    ]
    target["premises"] = [
        "Use current normalized per-agent own-block Fisher dynamics with persistent record likelihoods. Do not replace it by a reciprocal potential, an assumed global free energy, or a separately designed parent force.",
        "Declare the complete probe/record protocol before choosing a retention. Include action labels, waiting times, event data and retained memories whenever accessible. Keep the old prohibition on deleting future probes, inserting flat erasure gates, resetting the system or supplying a membership tree.",
        "A whole population has no assumed probability law over its law configurations. Any filtering construction must introduce and label its preparation law and prove the appropriate conditional equivalence rather than substitute it for pointwise inheritance.",
        "The existing per-agent grouping, conditional phase parent, history identity and geometric results are scoped source material. Their permanently invisible phase does not close O1. A collective candidate must show exactly why it is not the same supplied factor or all-partition quotient in different coordinates.",
        "Fixed state-dependent constitutive formulas are allowed only within the existing model and with complete own-block derivatives. Additional analyticity, finite dimensions or probing richness are route-specific assumptions, not premises of the full existential target.",
    ]
    target["search_priors"] = []
    digest = hashlib.sha256(json.dumps(target, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    metadata = {"schema_version": "rigorous-theory-search/v1", "contract_id": "contract-sha256-" + digest, "target_digest": digest}
    for path in root.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(metadata)
        if path.name == "problem-contract.json":
            data["target"] = target
        elif path.name == "claim-ledger.json":
            for claim in data.get("claims", []):
                if claim.get("id") == "target":
                    claim.update({key: target[key] for key in ("statement", "kind", "quantifiers")})
                    claim["target_digest"] = digest
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    header = "<!-- rigorous-theory-search-metadata " + json.dumps(metadata, sort_keys=True, separators=(",", ":")) + " -->\n"
    for path in root.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        if text.startswith("<!-- rigorous-theory-search-metadata "):
            text = text.split("\n", 1)[1]
        path.write_text(header + text, encoding="utf-8")
    (root / "routes").mkdir(exist_ok=True)
    sources = [
        Path("C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0") / relative
        for relative in ("AUTHORING.md", "chapters/03_dynamics.tex", "chapters/04_prediction.tex", "chapters/05_inheritance.tex", "chapters/08_smooth_erasure.tex", "chapters/09_public_compression.tex", "chapters/12_open_formation.tex", "chapters/13_per_agent_inheritance.tex", "chapters/02_geometry.tex", "future_TODO.md")
    ]
    sources += [root.parent.parent.parent / "future_TODO.md"]
    manifest = {**metadata, "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]}
    (root / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "research-record.md").write_text(header + "# O1 research record\n\nThe user requested O1 only under the current per-agent dynamics. The Research vault and both future_TODO files remain read-only. Second-stage recursion is deferred. There are at most three substantive rounds, with early stopping after two rounds without checkable progress.\n\nThe existing sources supply an exact canonical-parent criterion, a joint response-transition criterion, a path-KL memory identity, positive-support observability obstructions, interaction-dependent grouping, and a conditional phase parent. The new task is not another transfer calculation. The missing result is compression not supplied by a permanently invisible factor or an all-partition quotient.\n\nRound 1 opens three analytical routes: observability and new equivalences; constructive normalized per-agent collective mechanisms; geometry and persistent-history compatibility. No compute-heavy scripts, simulations, GPU operations, or Lean work are authorized. Independent reconstruction at release will be a labeled root pass; no unsolicited adjudication panel is part of the task.\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
