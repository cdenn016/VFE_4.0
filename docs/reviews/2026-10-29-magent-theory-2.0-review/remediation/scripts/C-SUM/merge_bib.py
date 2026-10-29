"""Merge the verified staged entries into references.bib and update stale records.

Usage: python merge_bib.py <references.bib> <staged.bib>
Keeps CRLF line endings. Refuses to add a key that already exists.
"""
from __future__ import annotations

import re
import sys

REPLACE = {
    "LetacMassam2017": r"""@article{LetacMassam2017,
  author  = {Letac, G{\'e}rard and Massam, H{\'e}l{\`e}ne},
  title   = {The {L}aplace transform $(\det s)^{-p}\exp \mathrm{tr}(s^{-1}w)$ and the existence of non-central {W}ishart distributions},
  journal = {Journal of Multivariate Analysis},
  volume  = {163},
  pages   = {96--110},
  year    = {2018},
  doi     = {10.1016/j.jmva.2017.10.005}
}""",
    "ChanEtAl2016": r"""@article{ChanEtAl2016,
  author  = {Chan, Chung and Al-Bashabsheh, Ali and Zhou, Qiaoqiao and Kaced, Tarik and Liu, Tie},
  title   = {Info-Clustering: A Mathematical Theory for Data Clustering},
  journal = {IEEE Transactions on Molecular, Biological and Multi-Scale Communications},
  volume  = {2},
  number  = {1},
  pages   = {64--91},
  year    = {2016},
  doi     = {10.1109/TMBMC.2016.2630054}
}""",
    "ShaliziCrutchfield2001": r"""@article{ShaliziCrutchfield2001,
  author  = {Shalizi, Cosma Rohilla and Crutchfield, James P.},
  title   = {Computational Mechanics: Pattern and Prediction, Structure and Simplicity},
  journal = {Journal of Statistical Physics},
  volume  = {104},
  number  = {3--4},
  pages   = {817--879},
  year    = {2001},
  doi     = {10.1023/A:1010388907793}
}""",
    "Shannon1959": r"""@article{Shannon1959,
  author  = {Shannon, Claude E.},
  title   = {Coding Theorems for a Discrete Source with a Fidelity Criterion},
  journal = {IRE International Convention Record},
  volume  = {7},
  number  = {4},
  pages   = {142--163},
  year    = {1959}
}""",
    "FournelEtAl2014": r"""@article{FournelEtAl2014,
  author  = {Fournel, C{\'e}dric and Fran{\c{c}}ois, Jordan and Lazzarini, Serge and Masson, Thierry},
  title   = {Gauge invariant composite fields out of connections, with examples},
  journal = {International Journal of Geometric Methods in Modern Physics},
  volume  = {11},
  number  = {3},
  pages   = {1450016},
  year    = {2014},
  doi     = {10.1142/S0219887814500169}
}""",
    "MassambaThompson2003": r"""@article{MassambaThompson2003,
  author  = {Massamba, Fortun{\'e} and Thompson, George},
  title   = {The Universal Connection and Metrics on Moduli Spaces},
  journal = {Communications in Mathematical Physics},
  volume  = {253},
  number  = {2},
  pages   = {323--335},
  year    = {2005},
  doi     = {10.1007/s00220-004-1191-7}
}""",
    "Kallenberg2021": r"""@book{Kallenberg2021,
  author    = {Kallenberg, Olav},
  title     = {Foundations of Modern Probability},
  edition   = {3},
  series    = {Probability Theory and Stochastic Modelling},
  volume    = {99},
  publisher = {Springer},
  address   = {Cham},
  year      = {2021},
  doi       = {10.1007/978-3-030-61871-1}
}""",
    "CasselEtAl2024": r"""@article{CasselEtAl2024,
  author  = {Cassel, Jonas and Boll, Bastian and Petra, Stefania and Albers, Peter and Schn{\"o}rr, Christoph},
  title   = {Sigma Flows for Image and Data Labeling and Learning Structured Prediction},
  journal = {Journal of Mathematical Imaging and Vision},
  volume  = {67},
  number  = {6},
  pages   = {54},
  year    = {2025},
  doi     = {10.1007/s10851-025-01270-w}
}""",
    "EellsSampson1964": r"""@article{EellsSampson1964,
  author  = {Eells, Jr., James and Sampson, J. H.},
  title   = {Harmonic Mappings of {R}iemannian Manifolds},
  journal = {American Journal of Mathematics},
  volume  = {86},
  number  = {1},
  pages   = {109--160},
  year    = {1964},
  doi     = {10.2307/2373037}
}""",
    "Minka2001EP": r"""@inproceedings{Minka2001EP,
  author    = {Minka, Thomas P.},
  title     = {Expectation Propagation for Approximate {B}ayesian Inference},
  booktitle = {Proceedings of the Seventeenth Conference on Uncertainty in Artificial Intelligence},
  year      = {2001},
  pages     = {362--369},
  publisher = {Morgan Kaufmann},
  url       = {https://tminka.github.io/papers/ep/minka-ep-uai.pdf}
}""",
}
DELETE = {"Narasimhan1962", "LewisNigamOlver2003"}

ENTRY = re.compile(r"@(\w+)\{([^,\s]+),.*?\n\}[ \t]*\n?", re.S)


def main(bib_path: str, staged_path: str) -> None:
    raw = open(bib_path, encoding="utf-8", newline="").read()
    text = raw.replace("\r\n", "\n")
    existing = {m.group(2) for m in ENTRY.finditer(text)}
    for key in list(REPLACE) + list(DELETE):
        assert key in existing, key

    def sub(m: re.Match) -> str:
        key = m.group(2)
        if key in DELETE:
            return ""
        if key in REPLACE:
            return REPLACE[key] + "\n"
        return m.group(0)

    text = ENTRY.sub(sub, text)
    staged = open(staged_path, encoding="utf-8").read().replace("\r\n", "\n")
    added = []
    blocks = []
    for m in ENTRY.finditer(staged):
        key = m.group(2)
        if key in existing:
            raise SystemExit(f"key exists already: {key}")
        added.append(key)
        blocks.append(m.group(0).rstrip("\n") + "\n")
    text = text.rstrip("\n") + "\n\n" + "\n".join(blocks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    open(bib_path, "w", encoding="utf-8", newline="").write(text.replace("\n", "\r\n"))
    print("replaced:", sorted(REPLACE))
    print("deleted:", sorted(DELETE))
    print("added", len(added), ":", added)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
