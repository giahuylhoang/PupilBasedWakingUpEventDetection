#!/usr/bin/env python3
"""
Copy arteriole diameter CSVs from data/raw/**/**/cycle*/ to data/pupil_whisker/
with names: {group}_{day}_{cycle}_arteriole-diameter.csv
(same pattern as pupil / whisker in collect_pupil_whisker.py)
"""
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
RAW_ROOT = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "pupil_whisker")


def find_first_file_with_pattern(folder, pattern, ext=".csv"):
    for name in os.listdir(folder):
        if pattern.lower() in name.lower() and name.lower().endswith(ext):
            return os.path.join(folder, name)
    return None


def normalize_cycle_name(cycle_folder_name: str) -> str:
    return cycle_folder_name.strip().replace(" ", "")


def collect_arteriole(dry_run: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isdir(RAW_ROOT):
        raise RuntimeError(f"RAW_ROOT does not exist: {RAW_ROOT}")

    for group_name in os.listdir(RAW_ROOT):
        group_path = os.path.join(RAW_ROOT, group_name)
        if not os.path.isdir(group_path):
            continue

        for day_name in os.listdir(group_path):
            day_path = os.path.join(group_path, day_name)
            if not os.path.isdir(day_path):
                continue

            for cycle_name in os.listdir(day_path):
                cycle_path = os.path.join(day_path, cycle_name)
                if not os.path.isdir(cycle_path):
                    continue

                arteriole_file = find_first_file_with_pattern(cycle_path, "arteriole")
                if not arteriole_file:
                    continue

                norm_cycle = normalize_cycle_name(cycle_name)
                out_name = f"{group_name}_{day_name}_{norm_cycle}_arteriole-diameter.csv"
                out_path = os.path.join(OUT_DIR, out_name)

                print(f"[CYCLE] {cycle_path}")
                print(f"  Arteriole -> {out_path}")

                if not dry_run:
                    shutil.copy2(arteriole_file, out_path)


if __name__ == "__main__":
    collect_arteriole(dry_run=False)  # set True to preview only
