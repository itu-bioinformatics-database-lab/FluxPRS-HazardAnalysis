#!/usr/bin/env python3
"""Build ROSMAP and DiCAD phenotype tables for Kaan's prs_pipeline.py.

ROSMAP: cogdx 1 -> control, cogdx 4 and 5 -> Alzheimer Disease.
         IDs are Affy imputed gwas_id (ROS/MAP). MCI/other are omitted.
DiCAD:  diabetes_cat2 NGT -> control, Diabetes -> DIABETES, Pre-diabetes -> PRE-DIABETES.
         IDs are dicad_id; specimenIDs like 600003-2 are written to the ID map.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNP_META = Path("/home/cnjume/Desktop/SNP Data Download/metadata")

COGDX_KEEP = {1, 4, 5}
COGDX_TO_DX = {
    1: "control",
    4: "Alzheimer Disease",
    5: "Alzheimer Disease",
}
DICAD_CAT2 = {
    "ngt": "control",
    "nct": "control",
    "control": "control",
    "healthy": "control",
    "diabetes": "DIABETES",
    "pre-diabetes": "PRE-DIABETES",
    "prediabetes": "PRE-DIABETES",
    "pre diabetes": "PRE-DIABETES",
}


def _open(path: Path):
    return path.open(newline="", encoding="utf-8-sig")


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_tsv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def pad_projid(raw: str) -> str:
    s = (raw or "").strip().strip('"')
    if re.fullmatch(r"\d+", s):
        return s.zfill(8)
    return s


def build_rosmap(out_dir: Path) -> dict:
    clinical = SNP_META / "syn3191087" / "ROSMAP_clinical.csv"
    gwas_list = SNP_META / "syn12136606" / "AMP-AD_ROSMAP_Rush-Broad_AffymetrixGenechip6_Imputed.csv"
    if not clinical.is_file():
        raise SystemExit("missing %s" % clinical)
    if not gwas_list.is_file():
        raise SystemExit("missing %s" % gwas_list)

    gwas_ids = []
    with _open(gwas_list) as fh:
        for row in csv.DictReader(fh):
            gid = (row.get("gwas_id") or "").strip()
            if gid:
                gwas_ids.append(gid)
    gwas_set = set(gwas_ids)

    by_gwas = {}
    cogdx_all = Counter()
    n_clinical = 0
    n_unmapped_id = 0
    with _open(clinical) as fh:
        for row in csv.DictReader(fh):
            n_clinical += 1
            study = (row.get("Study") or "").strip()
            projid = pad_projid(row.get("projid") or "")
            gid = "%s%s" % (study, projid) if study and projid else ""
            cog_raw = (row.get("cogdx") or "").strip()
            try:
                cogdx = int(float(cog_raw)) if cog_raw and cog_raw.upper() != "NA" else None
            except ValueError:
                cogdx = None
            if cogdx is not None:
                cogdx_all[cogdx] += 1
            rec = {
                "gwas_id": gid,
                "individualID": gid,
                "rush_individualID": (row.get("individualID") or "").strip(),
                "projid": projid,
                "Study": study,
                "cogdx": "" if cogdx is None else str(cogdx),
                "diagnosis": COGDX_TO_DX.get(cogdx, ""),
                "apoe_genotype": (row.get("apoe_genotype") or "").strip(),
            }
            if gid in gwas_set:
                by_gwas[gid] = rec
            elif gid:
                n_unmapped_id += 1

    pheno_rows = []
    id_map_rows = []
    cogdx_in_geno = Counter()
    for gid in gwas_ids:
        rec = by_gwas.get(gid)
        if not rec:
            continue
        try:
            cogdx = int(rec["cogdx"]) if rec["cogdx"] else None
        except ValueError:
            cogdx = None
        if cogdx is not None:
            cogdx_in_geno[cogdx] += 1
        if cogdx not in COGDX_KEEP:
            continue
        pheno_rows.append(rec)
        if rec["rush_individualID"]:
            id_map_rows.append({
                "genotype_id": gid,
                "individual_id": rec["rush_individualID"],
            })

    write_csv(out_dir / "rosmap_pheno.csv",
              ["individualID", "gwas_id", "rush_individualID", "projid", "Study",
               "cogdx", "diagnosis", "apoe_genotype"],
              pheno_rows)
    write_tsv(out_dir / "rosmap_id_map.tsv",
              ["genotype_id", "individual_id"],
              id_map_rows)

    n_ad = sum(1 for r in pheno_rows if r["cogdx"] in ("4", "5"))
    n_ad5 = sum(1 for r in pheno_rows if r["cogdx"] == "5")
    n_ctrl = sum(1 for r in pheno_rows if r["cogdx"] == "1")
    summary = {
        "cohort": "ROSMAP",
        "n_clinical_rows": n_clinical,
        "n_imputed_gwas_ids": len(gwas_ids),
        "n_clinical_matched_to_gwas": len(by_gwas),
        "n_gwas_without_clinical": len(gwas_set - set(by_gwas)),
        "n_clinical_ids_not_in_imputed": n_unmapped_id,
        "cogdx_all_clinical": dict(sorted(cogdx_all.items())),
        "cogdx_among_imputed": {str(k): v for k, v in sorted(cogdx_in_geno.items())},
        "n_pheno_control_cogdx1": n_ctrl,
        "n_pheno_ad_cogdx4_plus_5": n_ad,
        "n_pheno_ad_cogdx5_only": n_ad5,
        "n_pheno_total": len(pheno_rows),
        "n_id_map": len(id_map_rows),
        "pheno": str(out_dir / "rosmap_pheno.csv"),
        "id_map": str(out_dir / "rosmap_id_map.tsv"),
        "note": "Phenotype IDs are Affy imputed gwas_id (ROS/MAP). --ad-groups AD,CONTROL",
    }
    return summary


def _norm_header(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").replace("\ufeff", "").strip())


def load_dicad_metabolomics(path: Path):
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise SystemExit("cannot decode %s" % path)
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:8]):
        if "dicad_id" in line and "diabetes_cat2" in line:
            header_idx = i
            break
    if header_idx is None:
        raise SystemExit("dicad_id/diabetes_cat2 header not found in %s" % path)
    rdr = csv.DictReader(lines[header_idx:])
    rows = []
    for row in rdr:
        rec = {_norm_header(k): (v or "").strip() for k, v in row.items() if k is not None}
        dicad = rec.get("dicad_id", "")
        if not re.fullmatch(r"\d+", dicad):
            continue
        sample_type = rec.get("Sample Type", "").lower()
        if sample_type and sample_type not in ("sample", "unknown", ""):
            if any(tok in sample_type for tok in ("calibrat", "qc", "blank", "standard", "pool")):
                continue
        rows.append(rec)
    return rows


def map_dicad_dx(cat2: str) -> str:
    key = re.sub(r"\s+", " ", (cat2 or "").strip().lower())
    return DICAD_CAT2.get(key, "")


def build_dicad(out_dir: Path) -> dict:
    metab = SNP_META / "syn25178479" / "DiCAD_metabolomics_JULY2020.csv"
    snp_assay = SNP_META / "syn25178488" / "DiCAD_snparray_assay_metadata.csv"
    indiv = SNP_META / "syn25178454" / "DiCAD_individual_metadata.csv"
    if not metab.is_file():
        raise SystemExit("missing %s" % metab)

    meta_rows = load_dicad_metabolomics(metab)
    by_id = {}
    extra_ids = defaultdict(set)
    conflicts = []
    cat2_counts = Counter()
    n_visits = 0
    for rec in meta_rows:
        dicad = rec["dicad_id"]
        cat2 = rec.get("diabetes_cat2", "")
        dx = map_dicad_dx(cat2)
        if not dx:
            continue
        n_visits += 1
        cat2_counts[cat2] += 1
        for extra in (rec.get("Sample Bar Code", ""), rec.get("Sample Identification", "")):
            if extra:
                extra_ids[dicad].add(extra)
        prev = by_id.get(dicad)
        if prev and prev["diabetes_cat2"] != cat2:
            conflicts.append(dicad)
            continue
        if prev is None:
            by_id[dicad] = {
                "individualID": dicad,
                "dicad_id": dicad,
                "diabetes_cat2": cat2,
                "diagnosis": dx,
                "apoeGenotype": "",
                "snp_specimenID": "",
            }

    apoe = {}
    if indiv.is_file():
        with _open(indiv) as fh:
            for row in csv.DictReader(fh):
                iid = (row.get("individualID") or "").strip()
                apoe[iid] = (row.get("apoeGenotype") or "").strip()

    specimens = defaultdict(list)
    if snp_assay.is_file():
        with _open(snp_assay) as fh:
            for row in csv.DictReader(fh):
                spec = (row.get("specimenID") or "").strip()
                if not spec:
                    continue
                dicad = spec.split("-")[0]
                specimens[dicad].append(spec)

    pheno_rows = []
    id_map_rows = []
    seen_map = set()
    dx_counts = Counter()
    for dicad in sorted(by_id, key=lambda x: int(x) if x.isdigit() else x):
        rec = by_id[dicad]
        rec["apoeGenotype"] = apoe.get(dicad, "")
        specs = specimens.get(dicad, [])
        rec["snp_specimenID"] = ";".join(specs)
        pheno_rows.append(rec)
        dx_counts[rec["diagnosis"]] += 1
        keys = [dicad, "ID%s" % dicad]
        keys.extend(sorted(extra_ids.get(dicad, [])))
        for spec in specs:
            keys.append(spec)
            tail = spec.split("-", 1)
            if len(tail) == 2:
                keys.append("ID%s_%s" % (dicad, tail[1]))
                keys.append("%s_%s" % (dicad, tail[1]))
        for key in keys:
            if key and key not in seen_map:
                id_map_rows.append({"genotype_id": key, "individual_id": dicad})
                seen_map.add(key)

    write_csv(out_dir / "dicad_pheno.csv",
              ["individualID", "dicad_id", "snp_specimenID", "diabetes_cat2",
               "diagnosis", "apoeGenotype"],
              pheno_rows)
    write_tsv(out_dir / "dicad_id_map.tsv",
              ["genotype_id", "individual_id"],
              id_map_rows)

    return {
        "cohort": "DiCAD",
        "n_metabolomics_labeled_visits": n_visits,
        "n_participants_with_label": len(pheno_rows),
        "diabetes_cat2_visit_counts": dict(cat2_counts),
        "diagnosis_counts": dict(dx_counts),
        "n_conflicting_repeat_labels_skipped": len(set(conflicts)),
        "n_with_snp_specimen": sum(1 for r in pheno_rows if r["snp_specimenID"]),
        "n_snp_assay_specimens": sum(len(v) for v in specimens.values()),
        "n_id_map": len(id_map_rows),
        "pheno": str(out_dir / "dicad_pheno.csv"),
        "id_map": str(out_dir / "dicad_id_map.tsv"),
        "ad_groups_for_kaan": "CONTROL,OTHER(DIABETES),OTHER(PRE-DIABETES)",
        "note": "Kaan norm_dx maps control->CONTROL and DIABETES/PRE-DIABETES to OTHER(...).",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "phenotypes"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "rosmap": build_rosmap(out_dir),
        "dicad": build_dicad(out_dir),
    }
    summary_path = out_dir / "phenotype_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
