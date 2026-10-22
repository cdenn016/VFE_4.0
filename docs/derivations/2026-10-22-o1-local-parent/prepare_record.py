"""Bind the completed conditional derivations to the frozen research record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', action='store_true')
    args = parser.parse_args()
    problem = json.loads((ROOT / 'problem-contract.json').read_text(encoding='utf-8'))
    binding = {k:problem[k] for k in ('schema_version','contract_id','target_digest')}
    header = '<!-- rigorous-theory-search-metadata ' + json.dumps(binding, sort_keys=True) + ' -->\n'
    for path in ROOT.glob('*.md'):
        body = path.read_text(encoding='utf-8')
        if body.startswith('<!-- rigorous-theory-search-metadata '):
            body = body.split('\n',1)[1]
        path.write_text(header+body,encoding='utf-8')

    def write(name: str, obj: dict[str, object]) -> None:
        (ROOT/name).write_text(json.dumps({**binding,**obj},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

    hypotheses = [
        ('A-native', 'The exact positive O1 exponential family, native own-block Fisher field, pointwise contextual dependence, original Gibbs rows, fixed protocol and finite-history regular branch.'),
        ('A-reference', 'One fixed c0 and nonzero pointed-cone anchor, full finite native reference tuple, coherent same-record updates, local two-jet and complete raw history; reference is unsampled memory.'),
        ('A-history', 'Actual compatible past-law initialization from a retained preparation seed and the original normalized chronological kernels; stored lab labels and covariant marked embeddings; the public-pattern normalizer is retained.'),
        ('A-geometry', 'A regular finite joint history/current experiment and the published common finite nonflat window; evaluation derivative holds the prepared section and its c0 reference fixed.'),
        ('A-nojet', 'Rich smooth open section class with independently variable finite jets, smooth finite-order local actual-law rule, positive Fisher chart, same minimizing marked connection classes and tensors; no separately phase-bearing reference memory.'),
        ('A-global', 'One smooth finite-dimensional encoding preserving direct public response laws at every c for a C2-open class; all finite smooth memory counted, history stratum fixed.'),
        ('A-generator', 'An explicitly known finite C2 preparation generator f(c;p), equivariant physical canonical seed, native chronological replay, and complete raw history; a restricted initialization family, not the entire C2 class.')
    ]
    specs = [
        ('C-local','A canonical local native-law two-jet and a finite native reference configuration have exact closed original-time evolution under every finite admitted record word.','local-realization.md','Sections 2-4 and 8, L1-L4: normalization, quotient, anchor positivity, local jet prolongation and actual law germs.',['A-native','A-reference']),
        ('C-history','With compatible actual past-law initialization, the finite-reference realization preserves every admitted finite adaptive joint response and next-state history.','local-realization.md','Section 5: actual normalized joint kernels, coherent persistent coefficients, seed/past-law distinction and causal induction.',['A-native','A-reference','A-history']),
        ('C-geometry','The finite-reference realization preserves actual Fisher costs and therefore minimizing marked classes, tensors and curvature on the original regular nonflat window, at the exact local two-jet scope.','local-realization.md','Sections 6 and 8, L5-L6: actual joint sample isomorphism, commutation, private score check and common nonflat window.',['A-native','A-reference','A-history','A-geometry']),
        ('C-reference','The published canonical value is not determined by the local germ away from c0, and its angular velocity is not a function of the anchor sum alone on the native open chart.','locality-obstructions.md','Section 1: disjoint reference perturbation and exact Gibbs formula for dOmega/du with S fixed.',['A-native']),
        ('C-nojet','Every finite-order strictly local actual-law rule satisfying N1 on a rich smooth open section class loses derivative dependence, so it cannot erase the common physical phase while preserving the original connection class and tensor.','locality-obstructions.md','Sections 2-3, N1-N3: positive norm polynomial, joint-L2 history argument, and positive horizontal private phase score.',['A-native','A-history','A-nojet']),
        ('C-global','A single smooth finite-dimensional state cannot reproduce all direct public law responses of an arbitrary open section class at every contextual point.','locality-obstructions.md','Section 4, N2: independent supported public perturbations and response-Jacobian rank contradiction.',['A-native','A-global']),
        ('C-generator','An explicit finite preparation generator admits an exact canonical finite seed plus complete-history whole-patch realization with native dynamics, all admitted kernels, physical loss and actual nonflat Fisher-cost inheritance.','finite-generator.md','Sections 1-4, F1-F3: nonempty witness family, chronological native solution map, normalized path law and same sample isomorphism.',['A-native','A-history','A-geometry','A-generator'])
    ]
    evidence = []
    claims = []
    assumptions = [{'id':i,'kind':'ROUTE_HYPOTHESIS','statement':s} for i,s in hypotheses]
    for cid,statement,path,scope,aids in specs:
        eid='E-'+cid[2:]
        evidence.append({'id':eid,'kind':'DERIVATION','supports':True,'scope':scope,'side_conditions':[dict(hypotheses)[a] for a in aids],'artifact_path':path,'artifact_sha256':digest(path)})
        claims.append({'id':cid,'statement':statement,'quantifiers':'For every instance satisfying the listed route hypotheses and precise locality/experiment scope in the referenced derivation.','target_digest':binding['target_digest'],'kind':'MATHEMATICAL','state':'EVIDENCE_VERIFIED','assumption_ids':aids,'evidence_ids':[eid],'bridge_premise_ids':[],'falsifier':'An admitted instance violating the stated identity or the exact proof hypothesis mapping in the named derivation.'})
    claims.insert(0, {'id':'target','statement':problem['target']['statement'],'quantifiers':problem['target']['quantifiers'],'target_digest':binding['target_digest'],'kind':'MATHEMATICAL','state':'INCONCLUSIVE','assumption_ids':[],'evidence_ids':[],'bridge_premise_ids':[],'falsifier':problem['target']['falsification_criterion']})
    write('claim-ledger.json',{'assumptions':assumptions,'evidence':evidence,'claims':claims})
    edges=[{'from':'target','to':s[0]} for s in specs]
    edges += [{'from':'C-history','to':'C-local'},{'from':'C-geometry','to':'C-local'},{'from':'C-geometry','to':'C-history'},{'from':'C-generator','to':'C-history'},{'from':'C-generator','to':'C-geometry'}]
    write('dependency-dag.json',{'edges':edges})

    families = [
        ('finite-reference','Native local/reference values and two-jets','Keep a finite context-reference state to close the global sample frame','Nonzero anchor and exact same-time chain rule',['C-local','C-history','C-geometry','C-reference'],'Explicit actual-law isomorphism L5-L6','Drop reference law coordinates at fixed S; Omega changes','No strictly germ-local initialization or arbitrary-profile finite global encoding','SUPPORTED_SCOPED'),
        ('strict-local-differential','Finite jets mapped to actual normalized laws','Recover geometry using derivative-valued local law coordinates','Highest-jet positive norm polynomial',['C-nojet'],'N1 forces derivative independence, then physical score loss','Independent next jets, same minimizer class and positive private score','Does not cover separately phase-bearing memory or constrained initial generators','OBSTRUCTED_SCOPED'),
        ('finite-global-encoding','One regular finite state for arbitrary section functions','Encode every contextual direct law response in fixed finite smooth coordinates','Rank of independent contextual public responses',['C-global'],'Disjoint supported perturbations and finite tangent dimension','More response coordinates than parent dimension','No obstruction to a local field, nonfinite seed, irregular encoding or finite generator','OBSTRUCTED_SCOPED'),
        ('preparation-generator','Finite canonical preparation seed, full chronological history and law evaluations','Replay the native controlled flow from a known finite section generator','Physical rotation equivariance and actual causal law equality',['C-generator'],'F1-F3 and L6 on the same positive finite window','Allow an arbitrary unknown bump outside the generator image','Initialization is restricted; raw history and replay work remain unbounded','SUPPORTED_CONDITIONAL')
    ]
    write('approach-registry.json',{'mechanism_families':[{'family_id':f,'representation':r,'core_mechanism':m,'invariant_or_obstruction':inv,'target_obligation_ids':['target',*ids],'obligations':ids,'bridge':bridge,'failure_test':test,'verified_results':ids,'open_gaps':[gap],'novelty_fingerprint':r+' | '+m+' | '+inv+' | '+bridge+' | '+gap,'disposition':disp} for f,r,m,inv,ids,bridge,test,gap,disp in families]})
    attacks = [
        ('AT-reference',['target','C-local','C-reference'],'Calling the two-jet construction strictly local conceals its use of c0 data.','PARTIALLY_SUSTAINED','L2 retains a disclosed finite native reference tuple; R1 reconstructs closure. This supports finite-reference locality only and leaves the unrestricted target open.'),
        ('AT-past',['target','C-history'],'Current laws and one observed record word do not determine its unconditional past probability.','REJECTED','Section 5 and R2 require the actual normalized preparation/record kernels and a replay seed or initialized pushforward law; no past probability is inferred from the snapshot alone.'),
        ('AT-density',['target','C-geometry'],'A stored two-jet or copied geometry is not an actual normalized parent family.','REJECTED','L5-L6 and the polynomial initial-seed realization in Section 8 use normalized native laws. Reference memory is unsampled and the true contextual phase dependence remains in B.'),
        ('AT-frozen',['target','C-local','C-generator'],'Freezing the handoff frame could restore the lost phase or hide nonlocal initialization.','PARTIALLY_SUSTAINED','R3 counts canonical seed memory and does not retain U inverse; common physical rotation remains lost. The reference-coherent initialization condition remains explicit and is not called a germ-local rule.'),
        ('AT-nojet',['target','C-nojet'],'The derivative-order proof might apply only to a failed ansatz, or incorrectly infer conditional equality from averaged Fisher equality.','REJECTED','N1 covers smooth finite differential-order rules under its rich-jet hypotheses. Its joint-L2 norm polynomial, branchwise invertible native derivative and finite basis argument avoid the conditional inference. R4 checks this repair.'),
        ('AT-allcontexts',['target','C-global'],'The finite-encoding argument may invent context-relocation probes or ignore a finite generator.','PARTIALLY_SUSTAINED','N2 is explicitly about one global encoding of the indexed experiment; it adds no protocol action. N2 excludes neither evaluation-local states nor the restricted F1 generator. R5 states these boundaries.'),
        ('AT-generator',['target','C-generator'],'A finite preparation family might silently replace the published arbitrary C2 class.','PARTIALLY_SUSTAINED','F1 is an added explicit sufficient condition, with a concrete O1 witness inside its image. The evolved field is not projected into an ansatz. This conditional construction cannot close the unrestricted target, as R6 records.')
    ]
    covered=['target',*[s[0] for s in specs]]
    reconstruction={'claim_ids':covered,'method':'Sequential same-model reconstruction from the frozen contract; no independent worker or corroboration. Sections R1-R6 reconstruct every load-bearing interface.','result':'PASS','conclusion':'All seven scoped derivations retain their assumptions; the unrestricted target remains INCONCLUSIVE.','artifact_path':'proof-reconstruction.md','artifact_sha256':digest('proof-reconstruction.md')}
    erasure={**reconstruction,'method':'Semantic removal of any affirmative expectation; R6 rereads the full dependency closure and conditional hypotheses.','conclusion':'No affirmative prior or paraphrased existence assumption is needed for the scoped results. This does not prove the unresolved target.'}
    write('adversarial-report.json',{'provenance':'Sequential same-model failure-directed passes, not an independent panel or adjudication.','attacks':[{'id':i,'claim_ids':ids,'attack':attack,'disposition':disp,'response':resp,'artifact_path':'proof-reconstruction.md','artifact_sha256':digest('proof-reconstruction.md')} for i,ids,attack,disp,resp in attacks],'independent_reconstruction':reconstruction,'oracle_erasure':erasure})
    unresolved=['Strictly germ-local actual-law realization on the full arbitrary section class with a more general permitted phase-bearing memory outside N1, or a scope-matched impossibility proof.','A single finite global representation of all arbitrary contextual profiles cannot be inferred from local jets; N2 requires an explicit restriction, nonfinite memory, or a departure from its regular encoding class.','Minimal native reference memory has not been determined; the constructive reference is sufficient and contextually nonlocal.']
    write('release.json',{'checkpoint':'final scoped release' if args.release else 'completed derivation checkpoint','target_claim':'target','terminal_status':'INCONCLUSIVE' if args.release else None,'certificate_claim':'target' if args.release else None,'strongest_result':'Exact finite-reference local two-jet realization with actual-law Fisher-cost inheritance; conditional finite preparation-and-history realization; scoped finite-order and regular finite-encoding obstructions.','unresolved_obligations':unresolved})
    (ROOT/'research-record.md').write_text(header+'# Research record\n\nThree substantive rounds completed sequentially. Source reading, finite-reference dynamics/history/geometry derivation, strict finite-order and regular global encoding obstructions, conditional finite-generator realization, seven exact algebra checks, sequential reconstruction and semantic erasure are complete. The unrestricted target remains INCONCLUSIVE. No independent corroboration is claimed.\n\n'+ '\n\n'.join(unresolved)+'\n\nThe daily change log is updated. The configured tool inventory exposes no work_state_read or work_state_write tool; this durable record tracks state because those requested tools are unavailable. No automation, background continuation, Lean, GPU probe or publication was performed.\n',encoding='utf-8')
    print('Bound seven scoped claims; target INCONCLUSIVE; release='+str(args.release))


if __name__ == '__main__':
    main()
