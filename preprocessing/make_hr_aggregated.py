"""
Rebuild the per-participant heart-rate files `<id>_hr_aggregated.csv` that
survey_heart_rate.ipynb reads from data/personai/HR_Aggregated.

The script that originally created these files is lost. This is a
reconstruction, checked against the 80 original files. By default it reads the
anonymized copies of the raw sensor exports in data/personai/Person_AI_Data/Data/<id>/ :
    Cozie/sample_anon.csv              columns index, ts_heart_rate, ws_heart_rate
    Oura/heartrate_<n>_<id>_anon.csv   columns timestamp, bpm
In these copies every other column is set to 'x' (the original Cozie export holds
locations and credentials, the Oura export names and e-mail addresses). The
original files are not read.

Output columns: time, bpm, source, bpm_cozie, bpm_oura, mean, deviation

Rules:
    - Cozie readings are all values of ts_heart_rate and ws_heart_rate; the same
      (exact time, value) pair is kept once.
    - Readings below 40 or above 200 bpm are dropped (the originals have none).
    - Of several Oura readings in the same second only the first one is kept.
    - A Cozie reading and an Oura reading with exactly the same time become one
      row (source "cozie", bpm = Cozie value, bpm_cozie and bpm_oura filled), whatever
      the values. Cozie readings with a fractional second therefore stay separate
      rows. All other readings are rows of their own (source "cozie" or "oura").
    - The time is cut to whole seconds after merging and sorting.
    - mean is the rounded mean of bpm over the whole file, deviation = bpm - mean.

Known difference to the original files (80 participants checked):
    When two different Oura readings fall into the same second the originals keep
    one of them by a rule that is not known; keeping the first picks the same one
    in about 83 % of these seconds. 1,655 of 6.2 million rows (0.013 %) differ
    because of this. survey_heart_rate.ipynb gives a final file that is cell for
    cell equal to data/personai/all_aircontrol_with_surveys_hr_with_new_bpm_new.csv.

Usage:
    py preprocessing\\make_hr_aggregated.py <output folder> [id001 id002 ...]

Without ids every participant with a heart-rate file is rebuilt. Files that
already exist in the output folder are never overwritten.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SRC_DIR, "..", "data", "personai", "Person_AI_Data")
BPM_MIN, BPM_MAX = 40, 200
OUT_COLUMNS = ["time", "bpm", "source", "bpm_cozie", "bpm_oura", "mean", "deviation"]


def empty_readings(value_column):
    return pd.DataFrame({"time": pd.Series(dtype="datetime64[ns, UTC]"), value_column: pd.Series(dtype=float)})


def in_range(readings, column):
    return readings[(readings[column] >= BPM_MIN) & (readings[column] <= BPM_MAX)]


def finish(rows):
    # time is cut to whole seconds only now; mean and deviation as in the original files
    if rows.empty:
        return rows.reindex(columns=OUT_COLUMNS)
    rows = rows.sort_values("time", kind="stable").reset_index(drop=True)
    rows["time"] = rows["time"].dt.floor("s")
    rows["mean"] = int(round(rows["bpm"].mean()))
    rows["deviation"] = rows["bpm"] - rows["mean"]
    return rows[OUT_COLUMNS]


# ---- raw exports (default)

def raw_participant_folder(participant):
    data = os.path.join(RAW_DIR, "Data")
    return os.path.join(data if os.path.isdir(data) else RAW_DIR, participant)


def read_raw_cozie(folder):
    path = os.path.join(folder, "Cozie", "sample_anon.csv")
    if not os.path.exists(path):
        return empty_readings("bpm")
    header = pd.read_csv(path, nrows=0).columns
    columns = [c for c in ("index", "ts_heart_rate", "ws_heart_rate") if c in header]
    cozie = pd.read_csv(path, usecols=columns, low_memory=False)
    cozie["time"] = pd.to_datetime(cozie["index"], utc=True)
    parts = [cozie.dropna(subset=[c])[["time", c]].rename(columns={c: "bpm"}) for c in ("ts_heart_rate", "ws_heart_rate") if c in cozie]
    if not parts:
        return empty_readings("bpm")
    return in_range(pd.concat(parts).drop_duplicates(["time", "bpm"]), "bpm").reset_index(drop=True)


def read_raw_oura(folder, participant):
    found = glob.glob(os.path.join(folder, "Oura", "heartrate_*_%s_anon.csv" % participant))
    if not found:
        return empty_readings("bpm_oura")
    oura = pd.read_csv(found[0], usecols=["timestamp", "bpm"]).rename(columns={"bpm": "bpm_oura"})
    oura["time"] = pd.to_datetime(oura["timestamp"], utc=True)
    oura = in_range(oura, "bpm_oura").drop_duplicates("time", keep="first")
    return oura[["time", "bpm_oura"]].reset_index(drop=True)


def aggregate_raw(participant):
    folder = raw_participant_folder(participant)
    cozie, oura = read_raw_cozie(folder), read_raw_oura(folder, participant)

    # one row per Cozie reading, with the Oura reading of exactly the same time if there is one
    cozie = cozie.merge(oura, on="time", how="left")
    cozie["source"] = "cozie"
    cozie["bpm_cozie"] = cozie["bpm"]

    oura = oura[~oura["time"].isin(set(cozie.dropna(subset=["bpm_oura"])["time"]))].copy()
    oura["source"] = "oura"
    oura["bpm"] = oura["bpm_oura"]
    oura["bpm_cozie"] = np.nan

    columns = ["time", "bpm", "source", "bpm_cozie", "bpm_oura"]
    return finish(pd.concat([oura[columns], cozie[columns]], ignore_index=True))


def list_participants():
    data = os.path.join(RAW_DIR, "Data")
    root = data if os.path.isdir(data) else RAW_DIR
    return sorted(f for f in os.listdir(root) if re.fullmatch(r"id\d+", f))


def main():
    parser = argparse.ArgumentParser(description="Rebuild the <id>_hr_aggregated.csv files from the raw sensor exports.")
    parser.add_argument("output_folder", help="folder for the new files (created if missing, existing files are kept)")
    parser.add_argument("ids", nargs="*", help="participant ids, e.g. id001 id002 (default: all with heart-rate data)")
    args = parser.parse_args()

    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError("input folder not found: " + os.path.abspath(RAW_DIR))
    ids = args.ids or list_participants()
    os.makedirs(args.output_folder, exist_ok=True)

    written, skipped = 0, 0
    for participant in ids:
        target = os.path.join(args.output_folder, participant + "_hr_aggregated.csv")
        if os.path.exists(target):
            print(participant, "file exists, not overwritten")
            skipped += 1
            continue
        rows = aggregate_raw(participant)
        if rows.empty:
            print(participant, "no heart-rate readings (or no *_anon.csv files), skipped")
            skipped += 1
            continue
        rows.to_csv(target + ".tmp", index=False)
        os.replace(target + ".tmp", target)
        print(participant, len(rows), "rows")
        written += 1
    print("written:", written, "| skipped:", skipped)


if __name__ == "__main__":
    main()
