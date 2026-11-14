import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rascar-hack-case-1"
WORK = ROOT / "outputs" / "redetect_strong"
DAYS = ["20261025", "20261026"]
CLASSES = ["person", "nude_hands", "cup", "tray"]  # оцениваемые классы, которые реально встречаются
EXTRA = ["table", "phone", "empty_plate", "full_plate", "personal_items", "bottle", "hair", "uniform_not_ok"]
TRAIN_CLASSES = CLASSES + EXTRA
FLIP = '_flip'


def run_name(name, member):
    """Имя запуска и файла предсказаний участника: day_20261025 для base, day_20261025_s1 для s1."""
    return name if member in ("", "base") else f"{name}_{member}"


def pred_path(name, member):
    """Файл предсказаний участника. Участник «s1_flip» — те же веса s1 на отражённых кадрах (TTA):
    pred_all_s1_flip.csv; для base — pred_all_flip.csv."""
    if member.endswith(FLIP):
        return WORK / f"pred_{run_name(name, member[:-len(FLIP)])}{FLIP}.csv"
    return WORK / f"pred_{run_name(name, member)}.csv"


def load(name, **kw):
    df = pd.read_csv(DATA / name, **kw)
    if "class_name" in df:
        df["class_name"] = df["class_name"].replace(json.loads((DATA / "scoring_policy.json").read_text(encoding="utf-8"))["aliases"])
    return df


def frame_day(split):
    fr = load(f"{split}_frames.csv", dtype={"day": str})
    return fr.set_index("frame_id")["day"]


def build():
    human = load("human_boxes_train.csv")
    human = human[human.class_name.isin(TRAIN_CLASSES)]
    day_of = frame_day("train")
    targets = load("train_context.csv").frame_id  # кадры с ручной разметкой
    sets = {f"day_{d}": [f for f in targets if day_of[f] == d] for d in DAYS}
    sets["all"] = list(targets)
    for name, frames in sets.items():
        root = WORK / "yolo" / name
        for sub in ("images", "labels"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        for fid in frames:
            img = root / "images" / f"{fid}.jpg"
            if not img.exists():
                try:
                    os.symlink(DATA / f"{fid}.jpg", img)
                except OSError:
                    # Windows may disallow symlinks without Developer Mode.
                    shutil.copy2(DATA / f"{fid}.jpg", img)
            rows = human[human.frame_id == fid]
            (root / "labels" / f"{fid}.txt").write_text("".join(
                f"{TRAIN_CLASSES.index(r.class_name)} {r.cx:.6f} {r.cy:.6f} {r.w:.6f} {r.h:.6f}\n" for r in rows.itertuples()))
        # val = тот же набор: другой день нельзя использовать для отбора эпох
        (root / "data.yaml").write_text(
            f"path: {root}\ntrain: images\nval: images\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(TRAIN_CLASSES)))
        print(name, len(frames), "кадров,", int(human.frame_id.isin(frames).sum()), "боксов")


def train(args):
    from ultralytics import YOLO
    for name in args.sets:
        if args.resume:
            checkpoint = WORK / "runs" / run_name(name, args.member) / "weights" / "last.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            YOLO(checkpoint).train(resume=True, device=args.device)
            continue
        model = YOLO(args.weights)  # публичные COCO-веса Ultralytics, скачиваются автоматически
        data_dir = WORK / "yolo" / (f"student_{name}{args.student_tag}" if args.student else name)
        if not (data_dir / "data.yaml").is_file():
            raise FileNotFoundError(f"{data_dir / 'data.yaml'}: сначала build" + (" и pseudo" if args.student else ""))
        model.train(data=str(data_dir / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=args.device, project=str(WORK / "runs"),
                    name=run_name(name, args.member), exist_ok=True, seed=args.seed, deterministic=True, val=False, patience=0, plots=False, flipud=0.0, verbose=False, workers=2,
                    **({} if args.freeze is None else {"freeze": args.freeze}))


def predict(args):
    from ultralytics import YOLO
    jobs = {  # какой детектор на каких кадрах
        "day_20261025": ("train", "20261026"),
        "day_20261026": ("train", "20261025"),
        "all": ("test", None),
    }
    for name in args.sets:
        split, day = jobs[name]
        frames = load(f"{split}_frames.csv", dtype={"day": str})
        if day:
            frames = frames[frames.day == day]
        model = YOLO(WORK / "runs" / run_name(name, args.member) / "weights" / "last.pt")
        out = []
        paths = [str(DATA / p) for p in frames.image_path]
        # Bound memory even when the backend interprets a list as one batch.
        for start in range(0, len(paths), args.predict_batch):
            batch = paths[start:start + args.predict_batch]
            if args.flip:  # отражаем кадр по горизонтали, боксы потом отражаем обратно
                import cv2
                batch = [np.ascontiguousarray(cv2.imread(p)[:, ::-1]) for p in batch]
            results = model.predict(batch, imgsz=args.imgsz,
                                    conf=0.01, iou=0.6, device=args.device,
                                    stream=True, verbose=False, max_det=200)
            for fid, res in zip(frames.frame_id.iloc[start:start + args.predict_batch], results):
                b = res.boxes
                xyxy_all = b.xyxy.cpu().numpy()
                if args.flip:
                    w = res.orig_shape[1]
                    xyxy_all = np.stack([w - xyxy_all[:, 2], xyxy_all[:, 1], w - xyxy_all[:, 0], xyxy_all[:, 3]], axis=1)
                for xyxy, c, k in zip(xyxy_all, b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
                    out.append((fid, model.names[int(k)], float(c), *map(float, xyxy)))
            print(f"{name}: {min(start + args.predict_batch, len(paths))}/{len(paths)} frames", flush=True)
        df = pd.DataFrame(out, columns=["frame_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        df.to_csv(pred_path(name, args.member + (FLIP if args.flip else "")), index=False)
        print(name, "->", len(frames), "кадров,", len(df), "детекций")


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def one_to_one(m, same):
    """Официальное правило метки: взаимно однозначные пары одного класса с IoU >= 0.6,
    сначала максимум числа пар, затем максимум суммарного IoU. Возвращает флаг «бокс получил пару»."""
    matched = np.zeros(m.shape[0], dtype=bool)
    if m.size:
        w = np.where(same & (m >= 0.6), 1000.0 + m, 0.0)
        rows, cols = linear_sum_assignment(w, maximize=True)
        matched[rows[w[rows, cols] > 0]] = True
    return matched


def agreement_features(boxes, ours, min_conf=0.05):
    """Для каждого исходного бокса — насколько с ним согласен наш детектор.
    boxes должны содержать все боксы кадра оцениваемых классов: дубли делят одну пару."""
    ours = ours[ours.confidence >= min_conf]
    feats = []
    xy = ["x1", "y1", "x2", "y2"]
    for fid, g in boxes.groupby("frame_id"):
        o = ours[ours.frame_id == fid]
        m = iou_matrix(g[xy].to_numpy(), o[xy].to_numpy())
        same = g.class_name.to_numpy()[:, None] == o.class_name.to_numpy()[None, :]
        oc = o.confidence.to_numpy()
        matched = one_to_one(m, same)
        for i, r in enumerate(g.itertuples()):
            ms = np.where(same[i], m[i], 0.0) if len(o) else np.zeros(0)
            mo = np.where(~same[i], m[i], 0.0) if len(o) else np.zeros(0)
            j = int(ms.argmax()) if ms.size else -1
            ok = ms >= 0.6
            feats.append(dict(
                id=r.id,
                iou_same=ms.max() if ms.size else 0.0,                         # лучший IoU с нашим боксом того же класса
                conf_at_best=oc[j] if j >= 0 and ms[j] > 0 else 0.0,            # уверенность нашего в этом боксе
                match_conf=(oc * ok).max() if ms.size else 0.0,                 # макс. уверенность среди «пар» с IoU>=0.6
                iou_other=mo.max() if mo.size else 0.0,                         # совпадение с нашим боксом ДРУГОГО класса
                other_conf=oc[int(mo.argmax())] if mo.size and mo.max() > 0 else 0.0,
                n_same_03=int((ms >= 0.3).sum()) if ms.size else 0,             # сколько наших боксов рядом
                matched=bool(matched[i]),                                        # пара по правилу один к одному
            ))
    return pd.DataFrame(feats)


def oof_features(train_df, member):
    parts = []
    for det_day, eval_day in (("20261025", "20261026"), ("20261026", "20261025")):
        ours = pd.read_csv(pred_path("day_" + det_day, member))
        g = train_df[train_df.day == eval_day]
        parts.append(g.merge(agreement_features(g, ours), on="id"))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run the six finalist detector views")
    parser.add_argument("cmd", choices=["build", "train", "predict"])
    parser.add_argument("--sets", nargs="+", choices=["day_20261025", "day_20261026", "all"],
                        default=["day_20261025", "day_20261026", "all"])
    parser.add_argument("--member", choices=["base", "s1", "s2"], default="base")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    args.seed = {"base": 0, "s1": 1, "s2": 2}[args.member]
    args.weights = str(ROOT / "weights/yolo11l.pt")
    args.epochs, args.imgsz, args.batch, args.predict_batch = 250, 1280, 4, 4
    args.resume, args.student, args.freeze = False, False, None
    if args.cmd == "build":
        build()
    elif args.cmd == "train":
        train(args)
    else:
        predict(args)
