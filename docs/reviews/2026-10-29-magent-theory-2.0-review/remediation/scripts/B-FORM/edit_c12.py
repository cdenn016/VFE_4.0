p12=r'C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/12_open_formation.tex'
pc=r'C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/companion/c12_open_formation.tex'
def apply(p,pairs):
    d=open(p,'rb').read().decode('utf-8'); n=d.count('\r\n')
    for a,b in pairs:
        assert d.count(a)==1,(p[-20:],d.count(a),a[:70]); d=d.replace(a,b)
    d=d.replace(chr(13)+chr(10),chr(10)).replace(chr(10),chr(13)+chr(10))
    open(p,'wb').write(d.encode('utf-8')); print(n,d.count('\r\n'))
apply(p12,[
 (r"\chapter{Preferential Two-Stage Selection from Open Law Classes}", r"\chapter{Preferential Selection in Two Stages from Open Law Classes}"),
 (r"\theoremheading{Open-law two-stage selection with exact inheritance}{open:formation}", r"\theoremheading{Open-law selection in two stages with exact inheritance}{open:formation}"),
])
apply(pc,[
 (r"In the regime of the proof, integrations indicate that switching off every peer interaction leaves the sequence of groups unchanged. Every group relation is temporary, and the same rule eventually merges everything. Measured against the formation standard, this is a two-stage selection with exact inheritance, not a recursive formation.",
  r"In the regime of the proof, integrations of the limiting equation indicate that switching off every peer interaction leaves the sequence of groups unchanged; this is not proved. Every group relation is temporary, and the same rule eventually merges everything. Measured against the formation standard, the construction is not shown to be a recursive formation: interaction necessity and the counterfactual clause are neither established nor shown to fail."),
 (r"With the proof's choices the far pair sits near $u\approx2\times10^7$, and the temperature implied by the written estimates is below $10^{-15}$, or about $10^{-21}$ once the slower decay of distant means in the exact field is accounted for.",
  r"With illustrative choices the far pair sits near $u\approx2\times10^7$, and the temperature is below $10^{-15}$ by admissibility alone, or about $10^{-21}$ once the slower decay of distant means in the exact field is accounted for, both before the Gronwall factor of the persistence lemma, which makes it far smaller still."),
 (r"The formation standard asks whether a grouping would also occur with the interaction switched off. Deleting every peer cost leaves $\dot u=-u$,",
  r"The formation standard asks whether a grouping would also occur with the interaction switched off, and it offers two ways of switching it off: keeping the peer labels and priors, or removing them. On the bit laws used in the calculation every self cost vanishes and the two agree; for correlated laws they differ through the self weight of the angular self cost. Deleting every peer cost leaves $\dot u=-u$,"),
 (r"Integrations from the proof's initial data, at $A=8$, $64$, and $4096$,",
  r"Integrations from reference configurations of the proof's form, at $A=8$, $64$, and $4096$ with a third pair much closer than in the proof,"),
 (r"What is proved is that peer forces contract the pairs faster than the null flow does.",
  r"What is proved is that peer forces contract the pairs at a rate above one, while the null flow contracts them at rate exactly one; a lower bound of $0.977$ on the rate would not show this, and the proof uses the explicit row structure."),
])
