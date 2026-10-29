"""Text-edit helper: rename the common-psi score symbols in chapter 14 files."""
import sys
for p in sys.argv[1:]:
    s = open(p, newline='').read()
    n1 = s.count(r'\ell_{\mathrm{phys}}'); n2 = s.count(r'I_{\mathrm{phys}}')
    s = s.replace(r'\ell_{\mathrm{phys}}', r'\ell_\psi').replace(r'I_{\mathrm{phys}}', r'I_\psi')
    open(p, 'w', newline='').write(s)
    print(p, n1, n2)
