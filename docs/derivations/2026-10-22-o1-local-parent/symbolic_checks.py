"""Small exact algebra checks; no model, numerical experiment, or GPU use."""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def main() -> None:
    results: list[dict[str, object]] = []

    def check(name: str, expression: sp.Expr | sp.MatrixBase) -> None:
        entries = list(expression) if isinstance(expression, sp.MatrixBase) else [expression]
        passed = all(sp.simplify(sp.trigsimp(entry)) == 0 for entry in entries)
        results.append({'name': name, 'passed': passed})
        if not passed:
            raise AssertionError(name)

    x, y, vx, vy = sp.symbols('x y vx vy', real=True)
    r = sp.sqrt(x*x + y*y)
    U = sp.Matrix([[x, y], [-y, x]]) / r
    J = sp.Matrix([[0, -1], [1, 0]])
    omega = (x*vy-y*vx)/(x*x+y*y)
    check('moving-frame derivative, nonzero anchor', U.diff(x)*vx+U.diff(y)*vy+omega*J*U)
    check('canonical anchor direction', U*sp.Matrix([x,y])-sp.Matrix([r,0]))

    alpha = sp.symbols('alpha', real=True)
    R = sp.Matrix([[sp.cos(alpha), -sp.sin(alpha)], [sp.sin(alpha), sp.cos(alpha)]])
    z = R*sp.Matrix([x,y])
    U_rot = sp.Matrix([[z[0],z[1]],[-z[1],z[0]]])/r
    check('equivariance of rotation slice', U_rot*R-U)

    s,k,E,u = sp.symbols('s k E u', positive=True)
    D = s+k*E+k*u
    w = (4*k*u/D-2*k*E/D-2*k/(s+2*k))/5
    check('reference sum alone does not close its angular rate', sp.diff(w,u)-(4*k*s+6*k*k*E)/(5*D*D))

    c = sp.symbols('c', real=True)
    f = sp.Function('f')(c)
    f0,f1,f2,b = sp.symbols('f0 f1 f2 b', real=True)
    rhs = f*f+c*f+c*c+b
    actual = sp.diff(rhs,c,2).subs({sp.diff(f,c,2):f2,sp.diff(f,c):f1,f:f0})
    expected = (2*f0+c)*f2+2*f1*f1+2*f1+2
    check('second-jet chain rule, no third derivative', actual-expected)

    t,a1,a2,w1,w2,g1,g2 = sp.symbols('t a1 a2 w1 w2 g1 g2', real=True)
    norm = g1*(a1+t*w1)**2+g2*(a2+t*w2)**2
    check('highest-jet norm coefficient', sp.diff(norm,t,2)/2-(g1*w1*w1+g2*w2*w2))

    angle, bx, by = sp.symbols('angle bx by', real=True)
    B = sp.Matrix([bx,by])
    e = sp.Matrix([sp.cos(angle),sp.sin(angle)])
    check('actual circular exponential under common sample rotation', (R*B).dot(R*e)-B.dot(e))

    report = {'scope': 'Seven exact symbolic identities only; side conditions and universal arguments are in the derivations.', 'sympy_version': sp.__version__, 'passed': all(bool(r['passed']) for r in results), 'checks': results, 'not_run': ['model simulations', 'training', 'GPU probes', 'Lean', 'independent agent verification']}
    path = Path(__file__).with_name('symbolic-checks.json')
    binding = json.loads(path.with_name('problem-contract.json').read_text(encoding='utf-8'))
    report.update({key:binding[key] for key in ('schema_version','contract_id','target_digest')})
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
