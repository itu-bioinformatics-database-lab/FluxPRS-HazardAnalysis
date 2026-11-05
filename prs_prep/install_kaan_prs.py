"""Install Kaan PRS_results_* into cohort Final_Prs_Scores as sample_id,score CSVs."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/cnjume/Desktop/FluxPRS-HazardAnalysis_Private")
SRC = ROOT / "incoming" / "kaan_pgs_scores"
CLIN = ROOT / "ROSMAP-AD" / "raw" / "ROSMAP_clinical.csv"
DICAD_MAP = ROOT / "incoming" / "prs_prep" / "phenotypes" / "dicad_id_map.tsv"
DICAD_IDS = ROOT / "DIABETES" / "processed" / "splits" / "all_ids.csv"


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_prs(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "score"])
        w.writerows(rows)


def rosmap_r_to_gwas() -> dict[str, str]:
    out = {}
    with CLIN.open(newline="") as fh:
        for rec in csv.DictReader(fh):
            rid = (rec.get("individualID") or "").strip().strip('"')
            study = (rec.get("Study") or "").strip().strip('"')
            proj = (rec.get("projid") or "").strip().strip('"')
            if rid and study and proj:
                out[rid] = f"{study}{proj}"
    return out


def dicad_iid_to_barcodes() -> dict[str, list[str]]:
    geno_to_indiv = {}
    with DICAD_MAP.open(newline="") as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            geno_to_indiv[rec["genotype_id"]] = rec["individual_id"]
    barcodes = defaultdict(set)
    with DICAD_IDS.open(newline="") as fh:
        for rec in csv.DictReader(fh):
            barcodes[rec["participant_id"]].add(rec["sample_id"])
    iid_to_barcodes = defaultdict(set)
    for geno, indiv in geno_to_indiv.items():
        for b in barcodes.get(indiv, ()):
            iid_to_barcodes[geno].add(b)
            iid_to_barcodes[indiv].add(b)
    return {k: sorted(v) for k, v in iid_to_barcodes.items()}


def matching_group(src_dir: Path) -> dict[str, str]:
    path = src_dir / "matching_summary.tsv"
    return {r["MODEL_ID"]: r["GROUP"] for r in read_tsv(path)}


def convert_scores(src_dir: Path, dest_root: Path, id_fn, group_keep: set[str] | None) -> dict:
    groups = matching_group(src_dir)
    n_models = n_rows = n_unmapped = 0
    mapped_ids = set()
    for score_path in sorted(src_dir.glob("*/*_scores.tsv")):
        model = score_path.parent.name
        if group_keep is not None and groups.get(model) not in group_keep:
            continue
        rows = []
        for rec in read_tsv(score_path):
            score = rec.get("PRS_SUM") or rec.get("PRS_AVG") or rec.get("PRS_Z")
            if score in (None, "", "NA"):
                continue
            for sid in id_fn(rec):
                rows.append((sid, score))
                mapped_ids.add(sid)
        n_unmapped += sum(1 for rec in read_tsv(score_path) if not id_fn(rec))
        write_prs(dest_root / model / "prs_scores.csv", rows)
        n_models += 1
        n_rows += len(rows)
    return {"models": n_models, "rows": n_rows, "unique_ids": len(mapped_ids), "unmapped_rows": n_unmapped}


def main() -> None:
    rmap = rosmap_r_to_gwas()
    dmap = dicad_iid_to_barcodes()

    def rosmap_ids(rec):
        rid = rec.get("INDIVIDUAL_ID") or ""
        gwas = rmap.get(rid)
        return [gwas] if gwas else []

    def dicad_ids(rec):
        iid = rec.get("IID") or ""
        return dmap.get(iid, [])

    def mayo_ids(rec):
        sid = rec.get("INDIVIDUAL_ID") or rec.get("IID") or ""
        return [sid] if sid and sid != "NA" else []

    summary = {
        "ROSMAP-AD": convert_scores(SRC / "PRS_results_ROSMAP", ROOT / "ROSMAP-AD" / "Final_Prs_Scores", rosmap_ids, {"AD"}),
        "DIABETES": convert_scores(SRC / "PRS_results_DiCAD", ROOT / "DIABETES" / "Final_Prs_Scores", dicad_ids, {"DIABETES"}),
        "MayoRNASeq-AD": convert_scores(SRC / "PRS_results_MayoRNAseq", ROOT / "MayoRNASeq-AD" / "Final_Prs_Scores", mayo_ids, {"AD"}),
        "MayoRNASeq-PSP": convert_scores(SRC / "PRS_results_MayoRNAseq", ROOT / "MayoRNASeq-PSP" / "Final_Prs_Scores", mayo_ids, {"PSP"}),
    }
    # notebooks historically read data/Final_Prs_Scores (ROSMAP copy)
    data_dest = ROOT / "data" / "Final_Prs_Scores"
    if data_dest.exists():
        import shutil
        shutil.rmtree(data_dest)
    convert_scores(SRC / "PRS_results_ROSMAP", data_dest, rosmap_ids, {"AD"})

    report = ROOT / "incoming" / "kaan_pgs_scores" / "install_report.json"
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
