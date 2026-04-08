#!/usr/bin/env python3
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))  # adjust if you put script elsewhere
RAW_ROOT = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "pupil_whisker")


def find_first_file_with_pattern(folder, pattern, ext=".csv"):
    """Return full path of first file in folder that contains pattern and endswith ext, or None."""
    for name in os.listdir(folder):
        if pattern.lower() in name.lower() and name.lower().endswith(ext):
            return os.path.join(folder, name)
    return None


def normalize_cycle_name(cycle_folder_name: str) -> str:
    """Convert 'cycle 5' -> 'cycle5', 'cycle_3' -> 'cycle3', etc."""
    name = cycle_folder_name.strip()
    # simple rule: remove spaces
    name = name.replace(" ", "")
    return name


def collect_pupil_whisker(dry_run: bool = False):
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

                # find pupil and whisker files in this cycle folder
                pupil_file = find_first_file_with_pattern(cycle_path, "pupil")
                whisker_file = find_first_file_with_pattern(cycle_path, "whisker")

                if not pupil_file and not whisker_file:
                    continue  # nothing relevant here
                if not (pupil_file and whisker_file):
                    # skip cycles that don't have both signals
                    print(f"Skipping (missing one): {cycle_path}")
                    continue

                norm_cycle = normalize_cycle_name(cycle_name)

                # pupil output name
                pupil_out_name = f"{group_name}_{day_name}_{norm_cycle}_pupil.csv"
                pupil_out_path = os.path.join(OUT_DIR, pupil_out_name)

                # whisker output name: decide suffix
                whisker_suffix = "whisker-gradient" if "grad" in os.path.basename(whisker_file).lower() else "whisker"
                whisker_out_name = f"{group_name}_{day_name}_{norm_cycle}_{whisker_suffix}.csv"
                whisker_out_path = os.path.join(OUT_DIR, whisker_out_name)

                print(f"[CYCLE] {cycle_path}")
                print(f"  Pupil  -> {pupil_out_path}")
                print(f"  Whisker-> {whisker_out_path}")

                if not dry_run:
                    shutil.copy2(pupil_file, pupil_out_path)
                    shutil.copy2(whisker_file, whisker_out_path)


if __name__ == "__main__":
    # Set dry_run=True first to just print what would be done.
    collect_pupil_whisker(dry_run=False)