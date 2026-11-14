"""Rebuild the two finalist submissions from saved model outputs and validate by day."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import person_vote as pv
import redetect as r

MEMBERS = ["base", "s1", "s2", "base_flip", "s1_flip", "s2_flip"]
LEADER = "standard_box_check.csv"
POSE = "enhanced_person_check.csv"


def predictions(split):
    rows = r.load(f"{split}.csv", dtype={"id": str})
    if split == "train":
        rows = rows[rows.class_name != "table"].copy()
        rows["day"] = rows.frame_id.map(r.frame_day("train"))
        members = [r.oof_features(rows, member) for member in MEMBERS]
    else:
        members = [r.agreement_features(rows, pd.read_csv(r.pred_path("all", member)))
                   for member in MEMBERS]
    df = pv.add_signals(rows, split, members)
    _, df[LEADER] = pv.scores(df, legs=True, skip_window=True)
    hall = (df.class_name == "person") & ((df.x1 + df.x2) / 2 >= pv.WINDOW_X)
    vote = np.where(hall, (df.D + df.P) / 2, df.D)
    df[POSE] = (1 - df.confidence + vote + 2 * np.maximum(df.veto, df.L)) / 4
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=r.DATA)
    parser.add_argument("--out", type=Path, default=r.ROOT / "reproduced")
    args = parser.parse_args()
    r.DATA = args.data
    r.WORK = r.ROOT / "outputs/redetect_strong"
    for name in ("scoring_policy.json", "train.csv", "train_frames.csv", "test.csv", "sample_submission.csv"):
        if not (r.DATA / name).is_file():
            parser.error(f"Missing {r.DATA / name}; pass --data with the unpacked competition directory")
    if args.out.resolve() == (r.ROOT / "submissions").resolve():
        parser.error("Choose a separate output directory for the rebuilt files")
    train = predictions("train")
    test = predictions("test").set_index("id", verify_integrity=True)
    template = r.load("sample_submission.csv", dtype={"id": str}).set_index("id", verify_integrity=True)
    if set(template.index) != set(test.index) or len(template) != 362:
        raise ValueError("Unexpected test IDs")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in (LEADER, POSE):
        ap = train.groupby("day").apply(lambda day: average_precision_score(day.error, day[name]))
        print(name, "AP", ap.round(5).to_dict())
        out = test.loc[template.index, [name]].rename(columns={name: "score"})
        if not out.score.between(0, 1).all():
            raise ValueError("Invalid test scores")
        out.to_csv(args.out / name)
        ref = pd.read_csv(r.ROOT / "submissions" / name, dtype={"id": str}).set_index("id", verify_integrity=True)
        pd.testing.assert_index_equal(out.index, ref.index)
        np.testing.assert_allclose(out.score, ref.score, atol=1e-12, rtol=0)
        print(f"Verified {len(out)} scores against submitted file: {args.out / name}")


if __name__ == "__main__":
    main()
