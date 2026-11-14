from pathlib import Path

import numpy as np
import pandas as pd

import redetect as R

ROOT = Path(__file__).resolve().parents[1]
VETO = ROOT / "outputs" / "expert_veto"
BOX = ["x1", "y1", "x2", "y2"]
POSE_CONF = 0.5      # рамки pose-модели, которые считаем «её разметкой»
LEG_BELOW = 0.35     # насколько ниже рамки должна быть точка ноги, в высотах рамки
KEYPOINT_CONF, SHOULDER_CONF, SKELETON_CONF = 0.5, 0.3, 0.15
LEG_POINTS = ("lkne", "rkne", "lank", "rank")
WINDOW_X = 330       # центр рамки левее этой границы — места у окна слева (контровой свет)


def pose_vote(rows, pose):
    """P: 1, если рамка person не получила пару (один к одному, IoU >= 0.6) среди рамок pose-модели.

    Берём только прогон на полном кадре: тайлы обрезают людей, и обрезанная рамка «подтверждает»
    обрезанную рамку заказчика. Для остальных классов P = 0.
    """
    pose = pose[(pose.src == "full") & (pose.conf >= POSE_CONF)]
    vote = pd.Series(0.0, index=rows.id)
    people = rows[rows.class_name == "person"]
    for frame, group in people.groupby("frame_id"):
        theirs = pose[pose.frame_id == frame]
        iou = R.iou_matrix(group[BOX].to_numpy(), theirs[BOX].to_numpy())
        matched = R.one_to_one(iou, np.ones_like(iou, dtype=bool))
        vote[group.id.to_numpy()] = (~matched).astype(float)
    return vote.to_numpy()


def legs_below(rows, pose):
    """На сколько высот рамки ниже её нижней границы лежит самая нижняя уверенная точка ноги
    у скелета, чьи плечи внутри рамки. Тайлы (увеличение вдвое) здесь нужны: на них видно ступни.
    -9, если таких точек нет."""
    pose = pose[pose.conf >= SKELETON_CONF].copy()
    pose["shx"] = (pose.lsho_x + pose.rsho_x) / 2
    pose["shy"] = (pose.lsho_y + pose.rsho_y) / 2
    pose = pose[np.minimum(pose.lsho_c, pose.rsho_c) > SHOULDER_CONF]
    by_frame = dict(tuple(pose.groupby("frame_id")))
    result = []
    for row in rows.itertuples():
        best = -9.0
        skeletons = by_frame.get(row.frame_id)
        if row.class_name == "person" and skeletons is not None:
            own = skeletons[(skeletons.shx > row.x1) & (skeletons.shx < row.x2)
                            & (skeletons.shy > row.y1) & (skeletons.shy < row.y2)]
            for point in LEG_POINTS:
                seen = own[own[f"{point}_c"] > KEYPOINT_CONF]
                if len(seen):
                    best = max(best, float(((seen[f"{point}_y"] - row.y2) / (row.y2 - row.y1)).max()))
        result.append(best)
    return np.array(result)


def add_signals(df, split, member_frames):
    """Собрать голоса детекторов, P, вето V и флаг ног L в одну таблицу."""
    pose = pd.read_csv(VETO / f"{split}_pose.csv")
    veto = pd.read_csv(VETO / f"{split}_features.csv", dtype={"id": str})[["id", "veto"]]
    df = df.merge(veto, on="id", how="left", validate="one_to_one")
    votes = np.stack([(~frame.set_index("id").matched.reindex(df.id)).to_numpy(dtype=float) for frame in member_frames])
    df["votes"], df["n_members"] = votes.sum(axis=0), len(member_frames)
    df["D"] = votes.mean(axis=0)
    df["P"] = pose_vote(df, pose)
    df["L"] = (legs_below(df, pose) > LEG_BELOW).astype(float)
    return df


def scores(df, legs, skip_window=False):
    """Текущая формула и формула с голосом pose-модели."""
    person = df.class_name == "person"
    if skip_window:  # у окна оставляем прежнюю долю D
        person &= (df.x1 + df.x2) / 2 >= WINDOW_X
    d_new = np.where(person, (df.votes + df.P) / (df.n_members + 1), df.D)
    veto_new = np.maximum(df.veto, df.L) if legs else df.veto
    current = (1 - df.confidence + df.D + 2 * df.veto) / 4
    new = (1 - df.confidence + d_new + 2 * veto_new) / 4
    return current, new
