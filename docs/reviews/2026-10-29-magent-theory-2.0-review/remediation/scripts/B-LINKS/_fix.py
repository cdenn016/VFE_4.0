root = "C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/"
B = chr(92)
fixes = {
 "chapters/17_communication_links.tex": [(r"\nabla\sqrt f\,\nabla\sqrt f^T", r"\nabla\sqrt f(\nabla\sqrt f)^T")],
 "chapters/18_link_dynamics.tex": [(r"\int p(dh)\,q_\eta", r"\int q_\eta(\cdot\mid h)p(dh)"[:0] + r"\int p(dh)q_\eta"), (r"(m/M)\,|\nabla V|\,|\dot y|", r"(m/M)|\nabla V||\dot y|")],
 "chapters/19_curvature_hierarchy.tex": [(r"\mu(dz_A)\,\pi_b(d\theta)", r"\mu(dz_A)\pi_b(d\theta)")],
 "companion/c19_curvature_hierarchy.tex": [(r"$g\,dz$", r"$g(z)dz$")],
}
for f, reps in fixes.items():
    s = open(root + f, encoding="utf-8").read()
    for a, b in reps:
        assert s.count(a) == 1, (f, a)
        s = s.replace(a, b)
    open(root + f, "w", encoding="utf-8").write(s)
print("ok")
