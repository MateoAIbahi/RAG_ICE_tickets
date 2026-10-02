#!/usr/bin/env python3
"""
Import en masse d'une arborescence livrée par ICE, par ex. :

    Fichiers pour IA/
      PCCN V1/
        AUTO/...              -> pccn V1, tranche AUTO
        TRA/...               -> pccn V1, tranche TRA
        Dicodata 3.2/...      -> pccn V1, toutes tranches, dicodata 3.2
      PCCN V2/...
      PCCN V3/...

Les fichiers sont COPIÉS dans la boîte d'entrée avec leur .meta.json ; ils
seront ingérés au prochain passage (vendredi soir) ou tout de suite avec
`python -m src.ingestion.sync --docs-only`.

Script volontairement sans dépendance : il se lance directement sur l'hôte.
    python3 src/ingestion/import_tree.py "/chemin/Fichiers pour IA" --dry-run
    python3 src/ingestion/import_tree.py "/chemin/Fichiers pour IA" --inbox /data/uploads
"""
import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SUPPORTED_EXT = {".pdf", ".doc", ".docx", ".odt", ".rtf", ".ppt", ".pptx", ".odp"}
IGNORED_NAMES = {"thumbs.db", ".ds_store", "desktop.ini"}
TRANCHES = {"AUTO", "TRA", "RAME"}

RE_PCCN = re.compile(r"\bP+C+N\s*V?\s*(\d+)\b", re.IGNORECASE)   # tolère "PPCNV1"
RE_DICODATA = re.compile(r"\bdicodata\s*v?\s*(\d+(?:[.,]\d+)*)", re.IGNORECASE)


def classify(rel: Path, forced_pccn: str | None):
    pccn = forced_pccn
    tranche = None
    dicodata = None
    for part in rel.parts[:-1]:
        if pccn is None and (m := RE_PCCN.search(part)):
            pccn = f"V{int(m.group(1))}"
        if part.strip().upper() in TRANCHES:
            tranche = part.strip().upper()
        if m := RE_DICODATA.search(part):
            dicodata = m.group(1).replace(",", ".")
    if dicodata:
        tranche = None  # Dicodata : communes à toutes les tranches d'un palier
    return pccn, tranche, dicodata


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dossier racine extrait (contenant PCCN V1, V2...)")
    ap.add_argument("--inbox", type=Path, default=Path("/data/uploads"))
    ap.add_argument("--pccn", help="forcer la version (si la racine est déjà un dossier PCCN Vx)")
    ap.add_argument("--dry-run", action="store_true", help="affiche le classement sans rien copier")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"Dossier introuvable : {root}")

    plan, unsupported, no_version = [], Counter(), []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.name.lower() in IGNORED_NAMES or f.name.startswith("~$"):
            continue
        rel = f.relative_to(root)
        if f.suffix.lower() not in SUPPORTED_EXT:
            unsupported[f.suffix.lower() or "(aucune)"] += 1
            continue
        pccn, tranche, dicodata = classify(rel, args.pccn)
        if pccn is None:
            no_version.append(rel)
            continue
        plan.append((f, rel, pccn, tranche, dicodata))

    groups = Counter((p, t or "toutes", d or "-") for _, _, p, t, d in plan)
    print(f"\n{len(plan)} fichier(s) à importer depuis {root}\n")
    print(f"{'PCCN':<6}{'Tranche':<10}{'Dicodata':<10}Fichiers")
    for (p, t, d), n in sorted(groups.items()):
        print(f"{p:<6}{t:<10}{d:<10}{n}")
    if unsupported:
        print("\nIgnorés (format non géré) : " + ", ".join(f"{e} x{n}" for e, n in unsupported.most_common()))
    if no_version:
        print(f"\n{len(no_version)} fichier(s) SANS version PCCN détectée (non importés), ex. :")
        for r in no_version[:10]:
            print(f"  {r}")
        print("  -> relancer avec --pccn Vx sur le bon sous-dossier.")

    if args.dry_run:
        print("\n(dry-run : rien n'a été copié)")
        return

    batch = args.inbox / f"import_{datetime.now():%Y%m%d_%H%M%S}"
    for f, rel, pccn, tranche, dicodata in plan:
        dest = batch / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        meta = {
            "pccn_version": pccn,
            "tranche": tranche,
            "dicodata_version": dicodata,
            "relpath": rel.as_posix(),
            "origin": "import",
            "imported_at": datetime.now().isoformat(timespec="seconds"),
        }
        (dest.parent / f"{dest.name}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"\n{len(plan)} fichier(s) copiés dans {batch}")
    print("Ingestion au prochain passage, ou tout de suite : deploy/run_ingestion.sh --docs-only")


if __name__ == "__main__":
    main()
