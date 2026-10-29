p = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/companion/c17_communication_links.tex"
s = open(p, encoding="utf-8").read()
reps = [
(r"A_{ij}=\bar T_j\bar\theta_{ij}^{-1}.", r"\hat A_{ij}=\bar T_j\bar\theta_{ij}^{-1}."),
(r"This accounts for the inverse in $A_{ij}$.", r"This accounts for the inverse in $\hat A_{ij}$."),
(r"For regular non-Gaussian location noise, the same formula uses its location Fisher matrix instead of $R^{-1}$, with the same kernel when that matrix is positive definite.",
 r"For non-Gaussian location noise whose density has a square root in $H^1$, which includes Laplace noise, the same formula uses its location Fisher matrix instead of $R^{-1}$; that matrix is then positive definite and the kernel is the same. Uniform noise has no such Fisher form, and only the stabilizer statement remains. These identification statements assume that the receiver knows the sender's reference code. If it does not, messages depend only on $\bar T\bar\theta^{-1}$, which is unchanged when both factors are multiplied on the right by one group element."),
(r"If source labels are hidden, their uncertainty reduces information in the positive-semidefinite order, by the missing-information identity \citep{Louis1982}.",
 r"If source labels are hidden, their uncertainty reduces information in the positive-semidefinite order: by the law of total variance for the complete-data score, the observed information is the complete-data information minus a positive-semidefinite term, as in the identity of \citet{Louis1982}."),
(r"This is the same fixed-section mechanism underlying connection Laplacians \citep{SingerWu2012,BandeiraSingerSpielman2013}.",
 r"This is the global-section condition of group synchronization \citep{Singer2011}. On undirected graphs with the inverse convention the corresponding quadratic form is the connection Laplacian of \citet{SingerWu2012}, whose least eigenvalue measures frustration \citep{BandeiraSingerSpielman2013}."),
(r"Grounded edge records can identify the dictionaries, and hence their loop product, without relaying one realized sample around the entire loop.",
 r"When the codes are injective, grounded edge records can identify the dictionaries, and hence their loop product, without relaying one realized sample around the entire loop. For lossy codes they identify each dictionary only modulo its code stabilizer, and the loop product only modulo products of stabilizer elements."),
(r"If $Q$ is the $W$-orthogonal projection onto the complement of that subspace, the message-identifiable residual is", r"If $Q_{\mathcal D}$ is the $W$-orthogonal projection onto the complement of that subspace, the message-identifiable residual is"),
(r"=\|QB\omega\|_W^2.", r"=\|Q_{\mathcal D}B\omega\|_W^2."),
(r"These are compatible assignments of content vectors through the codes, the global sections of a cellular sheaf \citep{HansenGhrist2021}.",
 r"These are compatible assignments of content vectors through the codes, the global sections of a cellular sheaf \citep{HansenGhrist2019}, of the kind used for discourse by \citet{HansenGhrist2021}."),
]
for a, b in reps:
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok", len(reps))
