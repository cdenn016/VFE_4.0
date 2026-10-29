r"""Bibliography bookkeeping for the remediation (read-only).

Lists references.bib entries never cited by the three documents, keys cited
but missing, and, for the keys that the 2026-10-22 remediation logs declare
"from memory" or "not re-fetched", where they are cited.
"""
import glob
import os
import re

ROOT = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s01_bib_audit.out.txt")

CITE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
ENTRY = re.compile(r"@\w+\{([^,]+),")

# Keys the remediation logs flag as written from memory / not re-fetched.
FROM_MEMORY = {
    "R-FOUND": ["KobayashiNomizu1963", "Kechris1995", "Abbas2009", "BattistelliChisci2014", "DeGroot1974",
                "DonskerVaradhan1975", "DupuisEllis1997", "HegselmannKrause2002", "HofbauerSigmund1998",
                "MondererShapley1996", "Shahshahani1979", "Lojasiewicz1963", "AbsilMahonyAndrews2005",
                "Cencov1982", "Haynsworth1968", "Dieudonne1960", "FruchartEtAl2021", "NealHinton1998",
                "JordanEtAl1999", "Csiszar1984", "Petz1988", "DiaconisZabell1982", "Csiszar1975",
                "HalmosSavage1949", "Fritz2020"],
    "R-GEOM": ["ONeill1966", "Procesi1976", "Higham2008", "LeCamYang2000", "Cencov1982", "Zamir1998",
               "Kullback1959", "Reinhart1959", "Molino1988"],
    "R-INH": ["RogersPitman1981", "Dynkin1965", "WeiKuo1969", "LiRabitz1989", "ONeill1966", "BairdWood2003",
              "Lee2013", "LarsenSkou1991", "Dawid1979", "Lauritzen1996", "GolubitskyStewartSchaeffer1988",
              "DixonMortimer1996", "BaumPetrie1966", "EphraimMerhav2002", "GenestZidek1986", "Abbas2009",
              "KirchhoffEtAl2018", "PalaciosEtAl2020", "Fritz2020", "Kechris1995"],
    "R-FUS": ["Brezis1973", "Filippov1988", "Clarke1983", "HegselmannKrause2002", "Price1970", "Blackwell1953",
              "Torgersen1991", "AmbrosioGigliSavare2008", "Dieudonne1960"],
    "R-ERA": ["LaSalle1960", "PappasLafferriereSastry2000", "MauroyMezicMoehlis2013", "BatesLuZeng2000",
              "GenestZidek1986", "Abbas2009", "GolubitskyStewart2002"],
    "R-BND": ["Palais1979", "GolubitskyStewartSchaeffer1988", "Field2007", "Walters1982"],
    "R-CND": ["LanceWilliams1967", "Muirhead1982", "Smale1970", "ONeill1966", "HegselmannKrause2002"],
    "R-OPEN": ["BartlettRudolphSpekkens2007", "RowleyEtAl2003", "FelsOlver1998", "ONeill1966", "Cheng1995",
               "CarreiraPerpinan2006", "GeshkovskiEtAl2025", "ChenEtAl2023", "MardiaJupp2000", "Prony1795"],
}


def main() -> None:
    bib = open(os.path.join(ROOT, "references.bib"), encoding="utf-8").read()
    keys = ENTRY.findall(bib)
    files = (glob.glob(os.path.join(ROOT, "*.tex")) + glob.glob(os.path.join(ROOT, "chapters", "*.tex"))
             + glob.glob(os.path.join(ROOT, "appendices", "*.tex")) + glob.glob(os.path.join(ROOT, "supplements", "*.tex"))
             + glob.glob(os.path.join(ROOT, "companion", "*.tex")))
    files = [f for f in files if not os.path.basename(f).startswith("Item_")]
    where = {}
    for f in files:
        rel = os.path.relpath(f, ROOT).replace("\\", "/")
        for n, line in enumerate(open(f, encoding="utf-8"), 1):
            for m in CITE.finditer(line):
                for k in m.group(1).split(","):
                    where.setdefault(k.strip(), []).append("%s:%d" % (rel, n))
    out = ["references.bib entries: %d" % len(keys)]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    out.append("duplicate keys: %s" % dups)
    unc = [k for k in keys if k not in where]
    out.append("entries never cited by main/supplement/companion: %d" % len(unc))
    out.append("  " + ", ".join(unc))
    other_bibs = ""
    for b in ["item1_refs.bib", "item2_refs.bib", "item7_refs.bib"]:
        p = os.path.join(ROOT, b)
        if os.path.exists(p):
            other_bibs += open(p, encoding="utf-8").read()
    other_keys = set(ENTRY.findall(other_bibs))
    missing = sorted(k for k in where if k not in keys and k not in other_keys)
    out.append("cited but absent from references.bib and item bibs: %s" % missing)
    out.append("cited keys found only in item bibs: %s" % sorted(k for k in where if k not in keys and k in other_keys))
    out.append("")
    out.append("# keys flagged in remediation logs as from memory / not re-fetched")
    allmem = {}
    for log, ks in FROM_MEMORY.items():
        for k in ks:
            allmem.setdefault(k, []).append(log)
    for k in sorted(allmem):
        present = k in keys
        locs = where.get(k, [])
        out.append("%s\tlogs=%s\tin_bib=%s\tcited=%d\t%s" % (k, ",".join(allmem[k]), present, len(locs), "; ".join(locs[:6])))
    open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out[:8]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
