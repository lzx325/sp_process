"""Run the HE/CODEX TMA processing pipeline (steps 01-08) for one TMA described by a YAML config.

Steps:
    1 extract_cores   extract HE and CODEX TMA cores from the WSIs
    2 match_cores     match HE and CODEX cores via the metadata CSV
    3 registration    register each CODEX core to its HE core
    4 tissue_mask     convert QuPath tissue-detection GeoJSON to masks
    5 cellpose        cellpose cell segmentation on aligned CODEX cores
    6 marker_stats    per-channel marker statistics within the tissue masks
    7 quantification  per-cell marker quantification -> AnnData
    8 clustering      cell-type clustering

Example:
    python run_pipeline.py --config configs/TA-29.yaml --steps 3-8 --dry-run
"""
import argparse
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
SETUP_CONDA = PROJECT_DIR / "shell" / "setup_conda.sh"


def load_config(fp):
    with open(fp) as f:
        cfg = yaml.safe_load(f)
    fmt = {"data_dir": cfg["data_dir"], "tma_name": cfg["tma_name"]}
    fmt["tma_dir"] = cfg.get("tma_dir", "{data_dir}/{tma_name}").format(**fmt)
    cfg["tma_dir"] = Path(fmt["tma_dir"])
    cfg["inputs"] = {k: Path(v.format(**fmt)) for k, v in cfg.get("inputs", {}).items()}
    cfg.setdefault("steps", {})
    return cfg


class Runner:
    def __init__(self, cfg, dry_run=False):
        self.cfg = cfg
        self.dry_run = dry_run

    def step_cfg(self, name):
        return self.cfg["steps"].get(name) or {}

    def run(self, step_name, cmd):
        """Run `cmd` (list of args) inside the conda env configured for `step_name`."""
        step = self.step_cfg(step_name)
        env_prefix = self.cfg["conda_envs"][step.get("env", "default")]
        cmd = [str(c) for c in cmd] + [str(a) for a in step.get("extra_args", [])]
        shell_cmd = f"source {shlex.quote(str(SETUP_CONDA))} {shlex.quote(env_prefix)} && {shlex.join(cmd)}"
        print(f"[{step_name}] {shlex.join(cmd)}", flush=True)
        if not self.dry_run:
            subprocess.run(["bash", "-c", shell_cmd], check=True)


def py(script, *args):
    return ["python", SRC_DIR / script, *args]


def step_extract_cores(r):
    inputs, tma_dir = r.cfg["inputs"], r.cfg["tma_dir"]
    diameter = r.step_cfg("extract_cores")["core_diameter_um"]
    for modality in ("he", "codex"):
        r.run("extract_cores", py(
            "extract_tma_cores.py",
            inputs[f"{modality}_tma_wsi"],
            inputs[f"{modality}_tma_tsv"],
            tma_dir / f"{modality.upper()}_cores",
            "--core-diameter-um", diameter,
        ))


def step_match_cores(r):
    tma_dir = r.cfg["tma_dir"]
    r.run("match_cores", py(
        "match_he_codex_cores.py",
        "--he-dir", tma_dir / "HE_cores",
        "--codex-dir", tma_dir / "CODEX_cores",
        "--he-out-dir", tma_dir / "HE_cores_matched",
        "--codex-out-dir", tma_dir / "CODEX_cores_matched",
        "--mapping-csv", r.cfg["inputs"]["metadata_csv"],
    ))


def step_registration(r):
    tma_dir = r.cfg["tma_dir"]
    step = r.step_cfg("registration")
    codex_fps = sorted((tma_dir / "CODEX_cores_matched").glob("*.IF.ome.tif"))
    if not codex_fps:
        if r.dry_run:
            print(f"[registration] (no cores found yet in {tma_dir / 'CODEX_cores_matched'})")
            return
        sys.exit(f"No *.IF.ome.tif cores found in {tma_dir / 'CODEX_cores_matched'}")
    cmds = []
    for codex_fp in codex_fps:
        core_name = codex_fp.name[: -len(".IF.ome.tif")]
        he_fp = tma_dir / "HE_cores_matched" / f"{core_name}.HE.ome.tif"
        cmd = py("codex_registeration_batch.py", codex_fp, he_fp, tma_dir / "registration")
        if step.get("align_he_to_codex", False):
            cmd.append("--align-he-to-codex")
        cmds.append(cmd)
    n_jobs = step.get("n_jobs", 1)
    if n_jobs <= 1:
        for cmd in cmds:
            r.run("registration", cmd)
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            for fut in [pool.submit(r.run, "registration", cmd) for cmd in cmds]:
                fut.result()


def step_tissue_mask(r):
    tma_dir = r.cfg["tma_dir"]
    geojson_dir = r.cfg["inputs"]["tissue_detection_geojson_dir"]
    if not r.dry_run and not (geojson_dir.is_dir() and any(geojson_dir.iterdir())):
        sys.exit(
            f"Tissue detection GeoJSON not found in {geojson_dir}.\n"
            f"Run qupath_scripts/scripts/tissue_detection.groovy in QuPath on {tma_dir / 'HE_cores_matched'} first."
        )
    r.run("tissue_mask", py(
        "qupath_tissue_detection_to_mask.py",
        "--image_dir", tma_dir / "HE_cores_matched",
        "--geojson_dir", geojson_dir,
        "--output_dir", tma_dir / "tissue_detection" / "masks",
    ))


def step_cellpose(r):
    tma_dir = r.cfg["tma_dir"]
    step = r.step_cfg("cellpose")
    r.run("cellpose", py(
        "cellpose_detection.py",
        "--input-dir", tma_dir / "registration" / "aligned",
        "--output-dir", tma_dir / "cellpose" / "masks",
        "--dapi_channel", step.get("dapi_channel", 0),
        "--n_gpu", step.get("n_gpu", 1),
    ))


def step_marker_stats(r):
    tma_dir = r.cfg["tma_dir"]
    step = r.step_cfg("marker_stats")
    r.run("marker_stats", py(
        "sp_marker_stats.py",
        "--input_dir", tma_dir / "registration" / "aligned",
        "--output_fp", tma_dir / "processed" / "channel_stats.json",
        "--tissue_mask_dir", tma_dir / "tissue_detection" / "masks",
        "--n_workers", step.get("n_workers", 8),
    ))


def step_quantification(r):
    tma_dir = r.cfg["tma_dir"]
    step = r.step_cfg("quantification")
    cmd = py(
        "sp_quantification.py",
        "--input_image_dir", tma_dir / "registration" / "aligned",
        "--mask_dir", tma_dir / "cellpose" / "masks",
        "--output_fp", tma_dir / "processed" / "quantification.h5ad",
        "--n_workers", step.get("n_workers", 8),
    )
    if step.get("slide_split", False):
        cmd.append("--slide-split")
    r.run("quantification", cmd)


def step_clustering(r):
    tma_dir = r.cfg["tma_dir"]
    step = r.step_cfg("clustering")
    cmd = ["python", "-u", SRC_DIR / "sp_cell_type_clustering.py",
           "--input_anndata_fp", tma_dir / "processed" / "quantification.h5ad",
           "--output_dir", tma_dir / "processed"]
    for flag in ("run_preprocess", "run_leiden", "run_umap", "use_bbknn"):
        if step.get(flag, False):
            cmd.append(f"--{flag}")
    r.run("clustering", cmd)


STEPS = [
    (1, "extract_cores", step_extract_cores),
    (2, "match_cores", step_match_cores),
    (3, "registration", step_registration),
    (4, "tissue_mask", step_tissue_mask),
    (5, "cellpose", step_cellpose),
    (6, "marker_stats", step_marker_stats),
    (7, "quantification", step_quantification),
    (8, "clustering", step_clustering),
]


def parse_steps(spec):
    """Parse e.g. '1-8', '3,5', 'registration,cellpose' into a sorted list of step numbers."""
    name_to_num = {name: num for num, name, _ in STEPS}
    selected = set()
    for part in spec.split(","):
        part = part.strip()
        if part in name_to_num:
            selected.add(name_to_num[part])
        elif "-" in part:
            lo, hi = part.split("-")
            selected.update(range(int(lo), int(hi) + 1))
        else:
            selected.add(int(part))
    unknown = selected - set(name_to_num.values())
    if unknown:
        raise ValueError(f"Unknown step(s): {sorted(unknown)}")
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config of the TMA to process")
    parser.add_argument("--steps", default="1-8", help="Steps to run, by number or name (default: 1-8)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        selected = parse_steps(args.steps)
    except ValueError as exc:
        parser.error(str(exc))

    runner = Runner(cfg, dry_run=args.dry_run)
    for num, name, fn in STEPS:
        if num in selected:
            print(f"===== step {num:02d}: {name} ({cfg['tma_name']})", flush=True)
            try:
                fn(runner)
            except subprocess.CalledProcessError as exc:
                sys.exit(f"Step {num:02d} ({name}) failed with exit status {exc.returncode}")


if __name__ == "__main__":
    main()
