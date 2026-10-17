"""
Create symbolic links in he_out_dir and codex_out_dir for matched HE–CODEX TMA cores.

Reads a mapping CSV with columns tma_id_he, tma_id_codex, and tma_id. For each row,
if corresponding ome.tif/ome.tiff files exist in he_dir and codex_dir, creates
symlinks in he_out_dir and codex_out_dir. Symlink names use the suffix from the
first dot in the source filename (e.g. core1.ome.tif -> .ome.tif, foo.other.ext -> .other.ext).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OMETIF_SUFFIXES = (".ome.tif", ".ome.tiff")


def _suffix_from_first_dot(path: Path) -> str:
    """Return the suffix from the first dot in the filename (e.g. core1.ome.tif -> .ome.tif)."""
    name = path.name
    i = name.find(".")
    return name[i:] if i >= 0 else ""


def _resolve_ometif(path: Path, base_name: str) -> Path | None:
    """Return path to existing file base_name.ome.tif or base_name.ome.tiff."""
    for suf in OMETIF_SUFFIXES:
        candidate = path / f"{base_name}{suf}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create symlinks for matched HE–CODEX TMA cores from a mapping CSV."
    )
    parser.add_argument("--he-dir", type=Path, required=True, help="Directory of HE TMA core ome.tif files.")
    parser.add_argument("--codex-dir", type=Path, required=True, help="Directory of CODEX TMA core ome.tif files.")
    parser.add_argument("--he-out-dir", type=Path, required=True, help="Output directory for HE symlinks.")
    parser.add_argument("--codex-out-dir", type=Path, required=True, help="Output directory for CODEX symlinks.")
    parser.add_argument("--mapping-csv", type=Path, required=True, help="CSV with tma_id_he, tma_id_codex, tma_id.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done.")
    args = parser.parse_args()

    df = pd.read_csv(args.mapping_csv)
    required = {"tma_id_he", "tma_id_codex", "tma_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Mapping CSV must have columns {required}. Missing: {missing}. "
            f"Available: {list(df.columns)}"
        )

    args.he_out_dir.mkdir(parents=True, exist_ok=True)
    args.codex_out_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    skipped_missing = 0
    for _, row in df.iterrows():
        tma_id_he = str(row["tma_id_he"]).strip()
        tma_id_codex = str(row["tma_id_codex"]).strip()
        tma_id = str(row["tma_id"]).strip()
        if not tma_id_he or not tma_id_codex or not tma_id:
            continue

        he_src = _resolve_ometif(args.he_dir, tma_id_he)
        codex_src = _resolve_ometif(args.codex_dir, tma_id_codex)
        if he_src is None or codex_src is None:
            skipped_missing += 1
            continue

        # Use suffix from first dot in source filename (e.g. .ome.tif, .ome.tiff, or any other)
        he_ext = _suffix_from_first_dot(he_src)
        codex_ext = _suffix_from_first_dot(codex_src)
        he_link = args.he_out_dir / f"{tma_id}.HE{he_ext}"
        codex_link = args.codex_out_dir / f"{tma_id}.IF{codex_ext}"

        if args.dry_run:
            print(f"Would link: {he_link} -> {he_src}")
            print(f"Would link: {codex_link} -> {codex_src}")
        else:
            he_link.symlink_to(he_src.resolve())
            codex_link.symlink_to(codex_src.resolve())
        linked += 1

    print(f"Created {linked} matched core symlinks. Skipped {skipped_missing} rows (missing file or empty id).")


if __name__ == "__main__":
    main()
