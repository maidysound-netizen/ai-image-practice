"""중고거래 사진 도우미 — 핵심 로직.

사진 한 장을 받아
  1) detection(yolo26n.pt)  : 무엇이 어디 있는지 → 박스·클래스·점수 → 카테고리 자동 태그
  2) segmentation(yolo26n-seg.pt) : 물건의 실제 윤곽 → 마스크 → 배경을 지운 '누끼' 사진
  3) 사진 품질 점검 : 물건이 너무 작게 찍힘, 다른 물건이 같이 찍힘, 사람(개인정보)·동물(거래 금지)
을 만들고, 모든 결과를 outputs/yolo/<사진이름>/conf_<값>/ 에 파일로 저장한다.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_ROOT = ROOT / "outputs" / "yolo"

DET_WEIGHTS = "yolo26n.pt"
SEG_WEIGHTS = "yolo26n-seg.pt"
IMGSZ = 640
DEVICE = "cpu"

# COCO 80종 중 중고거래에 올릴 수 있는 물건 → 중고거래 카테고리
CATEGORY = {
    "가구/인테리어": ["chair", "couch", "bed", "dining table", "potted plant", "vase", "clock"],
    "디지털기기": ["laptop", "tv", "cell phone", "keyboard", "mouse", "remote"],
    "생활가전": ["microwave", "oven", "toaster", "refrigerator", "hair drier"],
    "생활/주방": ["cup", "bowl", "bottle", "wine glass", "fork", "knife", "spoon", "scissors", "toothbrush"],
    "스포츠/레저": ["bicycle", "sports ball", "skateboard", "surfboard", "tennis racket", "skis",
                 "snowboard", "frisbee", "baseball bat", "baseball glove", "kite"],
    "가방/잡화": ["backpack", "handbag", "suitcase", "tie", "umbrella"],
    "도서": ["book"],
    "유아동/완구": ["teddy bear"],
    "중고차/오토바이": ["car", "motorcycle"],
}
CLASS_TO_CATEGORY = {c: cat for cat, classes in CATEGORY.items() for c in classes}

KO_NAME = {
    "chair": "의자", "couch": "소파", "bed": "침대", "dining table": "식탁", "potted plant": "화분",
    "vase": "꽃병", "clock": "시계", "laptop": "노트북", "tv": "TV/모니터", "cell phone": "휴대폰",
    "keyboard": "키보드", "mouse": "마우스", "remote": "리모컨", "microwave": "전자레인지",
    "oven": "오븐", "toaster": "토스터", "refrigerator": "냉장고", "hair drier": "드라이기",
    "cup": "컵", "bowl": "그릇", "bottle": "병", "wine glass": "와인잔", "fork": "포크",
    "knife": "칼", "spoon": "숟가락", "scissors": "가위", "toothbrush": "칫솔",
    "bicycle": "자전거", "sports ball": "공", "skateboard": "스케이트보드", "surfboard": "서핑보드",
    "tennis racket": "테니스 라켓", "skis": "스키", "snowboard": "스노보드", "frisbee": "원반",
    "baseball bat": "야구 배트", "baseball glove": "야구 글러브", "kite": "연",
    "backpack": "백팩", "handbag": "핸드백", "suitcase": "캐리어", "tie": "넥타이", "umbrella": "우산",
    "book": "책", "teddy bear": "곰인형", "car": "자동차", "motorcycle": "오토바이",
    "person": "사람", "bus": "버스", "truck": "트럭",
}
ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

# 품질 점검 기준 (근거: 썸네일에서 물건이 화면의 15% 미만이면 무엇인지 알아보기 어렵다고 판단)
MIN_MAIN_AREA_RATIO = 0.15
# 대표 물건 점수가 이보다 낮거나 두 모델 의견이 다르면 자동 등록하지 않고 '확인 필요'로 둔다
REVIEW_SCORE = 0.50
# 박스가 사진 가장자리에서 이 픽셀 이내면 '잘린 물건'으로 본다
EDGE_PX = 3


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _models():
    from ultralytics import YOLO

    if not hasattr(_models, "cache"):
        _models.cache = (YOLO(str(MODEL_DIR / DET_WEIGHTS)), YOLO(str(MODEL_DIR / SEG_WEIGHTS)))
    return _models.cache


def _versions() -> dict:
    import torch
    import ultralytics

    return {
        "python": sys.version.split()[0],
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }


def _box(b) -> list[float]:
    return [round(float(v), 1) for v in b]


def analyze(image_path: str | Path, conf: float = 0.25, out_root: Path = OUT_ROOT) -> dict:
    """사진 한 장을 분석해 결과 파일을 저장하고 요약(dict)을 돌려준다."""
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")
    h, w = img.shape[:2]
    img_area = float(h * w)

    out = out_root / image_path.stem / f"conf_{conf:.2f}"
    (out / "masks").mkdir(parents=True, exist_ok=True)

    det_model, seg_model = _models()
    det = det_model.predict(str(image_path), imgsz=IMGSZ, device=DEVICE, conf=conf, verbose=False)[0]
    seg = seg_model.predict(str(image_path), imgsz=IMGSZ, device=DEVICE, conf=conf, verbose=False)[0]

    # ---- 1) detection: 박스·클래스·점수
    detections = []
    for i, (cls, score, xyxy) in enumerate(
        zip(det.boxes.cls.tolist(), det.boxes.conf.tolist(), det.boxes.xyxy.tolist())
    ):
        name = det.names[int(cls)]
        x1, y1, x2, y2 = xyxy
        detections.append({
            "id": i,
            "class": name,
            "class_ko": KO_NAME.get(name, name),
            "score": round(score, 3),
            "box_xyxy": _box(xyxy),
            "box_area_ratio": round((x2 - x1) * (y2 - y1) / img_area, 4),
        })
    cv2.imwrite(str(out / "detect.jpg"), det.plot())

    # ---- 2) segmentation: 원본 해상도 마스크(폴리곤을 원본 크기로 다시 그림)
    segments = []
    union = np.zeros((h, w), np.uint8)
    if seg.masks is not None:
        for i, (cls, score, xyxy, poly) in enumerate(
            zip(seg.boxes.cls.tolist(), seg.boxes.conf.tolist(), seg.boxes.xyxy.tolist(), seg.masks.xy)
        ):
            name = seg.names[int(cls)]
            mask = np.zeros((h, w), np.uint8)
            if len(poly):
                cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
            mask_file = f"masks/{i:02d}_{name.replace(' ', '_')}.png"
            cv2.imwrite(str(out / mask_file), mask)
            union = np.maximum(union, mask)
            x1, y1, x2, y2 = xyxy
            box_area = max((x2 - x1) * (y2 - y1), 1.0)
            mask_px = int((mask > 0).sum())
            segments.append({
                "id": i,
                "class": name,
                "class_ko": KO_NAME.get(name, name),
                "score": round(score, 3),
                "box_xyxy": _box(xyxy),
                "mask_file": mask_file,
                "mask_area_px": mask_px,
                "mask_area_ratio": round(mask_px / img_area, 4),
                # 박스 중 실제 물건이 차지하는 비율 → 나머지는 박스 안에 섞여 있던 배경
                "mask_fill_of_box": round(mask_px / box_area, 3),
            })
    cv2.imwrite(str(out / "segment.jpg"), seg.plot())

    # 마스크만 반투명으로 겹친 화면 (저장된 마스크 파일과 같은 데이터로 그림)
    overlay = img.copy()
    overlay[union > 0] = (0.45 * overlay[union > 0] + 0.55 * np.array([60, 200, 60])).astype(np.uint8)
    cv2.imwrite(str(out / "mask_overlay.jpg"), overlay)

    # ---- 3) 판매용 정리: 대표 물건 고르기 → 카테고리·누끼·경고
    for s in segments:
        x1, y1, x2, y2 = s["box_xyxy"]
        s["touches_edge"] = x1 <= EDGE_PX or y1 <= EDGE_PX or x2 >= w - EDGE_PX or y2 >= h - EDGE_PX
    sellable = [s for s in segments if s["class"] in CLASS_TO_CATEGORY]

    def rank(s):
        # 크게·확실하게 찍힌 물건이 대표. 가장자리에서 잘린 물건(배경의 모니터 등)은 절반만 쳐준다.
        return s["mask_area_ratio"] * s["score"] * (0.5 if s["touches_edge"] else 1.0)

    main = max(sellable, key=rank, default=None)
    warnings, review_reasons = [], []

    # 대표 물건을 detection 모델도 같은 이름으로 봤는지 (박스가 IoU 0.5 이상 겹치는 같은 클래스)
    det_agrees = main is not None and any(
        d["class"] == main["class"] and _iou(d["box_xyxy"], main["box_xyxy"]) >= 0.5 for d in detections
    )

    if main is None:
        det_sellable = [d for d in detections if d["class"] in CLASS_TO_CATEGORY]
        if det_sellable:
            best = max(det_sellable, key=lambda d: d["score"])
            review_reasons.append(
                f"segmentation은 판매 물건을 못 찾았고, detection만 '{best['class_ko']}'(점수 {best['score']})로 봤습니다. "
                f"누끼는 만들지 않았습니다."
            )
        else:
            review_reasons.append("판매할 만한 물건을 찾지 못했습니다(모델이 아는 80종 밖이거나 점수가 임계값보다 낮음).")
    else:
        if main["score"] < REVIEW_SCORE:
            review_reasons.append(f"대표 물건 '{main['class_ko']}'의 점수가 {main['score']}로 낮습니다(기준 {REVIEW_SCORE}).")
        if not det_agrees:
            det_names = ", ".join(f"{d['class_ko']}({d['score']})" for d in detections) or "없음"
            review_reasons.append(f"detection 모델은 다르게 봤습니다: {det_names}.")
        if main["touches_edge"]:
            warnings.append(f"{main['class_ko']}이(가) 사진 가장자리에서 잘렸습니다. 전체가 나오게 찍으세요.")
        mask = cv2.imread(str(out / main["mask_file"]), cv2.IMREAD_GRAYSCALE)
        x1, y1, x2, y2 = (int(round(v)) for v in main["box_xyxy"])
        pad = int(0.05 * max(x2 - x1, y2 - y1))
        x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
        x2, y2 = min(x2 + pad, w), min(y2 + pad, h)
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask
        cv2.imwrite(str(out / "cutout.png"), rgba[y1:y2, x1:x2])  # 투명 배경
        white = img.copy()
        white[mask == 0] = 255
        cv2.imwrite(str(out / "cutout_white.jpg"), white[y1:y2, x1:x2])  # 흰 배경

        if main["mask_area_ratio"] < MIN_MAIN_AREA_RATIO:
            warnings.append(
                f"{main['class_ko']}이(가) 사진의 {main['mask_area_ratio']:.0%}만 차지합니다. "
                f"더 가까이에서 찍으면 목록 썸네일에서 잘 보입니다."
            )
        others = [s for s in sellable if s is not main]
        if others:
            names = ", ".join(sorted({s["class_ko"] for s in others}))
            warnings.append(f"다른 물건({names})도 같이 찍혀 있습니다. 판매 물건이 무엇인지 헷갈릴 수 있어요.")

    people = [s for s in segments if s["class"] == "person"]
    if people:
        warnings.append(f"사람 {len(people)}명이 찍혀 있습니다. 얼굴·신상이 드러나지 않게 가리거나 다시 찍으세요(개인정보).")
    animals = [s for s in segments if s["class"] in ANIMALS]
    if animals:
        warnings.append("동물이 찍혀 있습니다. 대부분의 중고거래 앱은 반려동물 거래를 금지합니다.")

    # detection과 segmentation이 서로 다르게 본 경우도 그대로 기록(숨기지 않음)
    det_classes = sorted(d["class"] for d in detections)
    seg_classes = sorted(s["class"] for s in segments)

    listing = {
        "status": "확인 필요" if review_reasons else "자동 태그 가능",
        "review_reasons": review_reasons,
        "main_item": None if main is None else {
            "class": main["class"],
            "name_ko": main["class_ko"],
            "category": CLASS_TO_CATEGORY[main["class"]],
            "score": main["score"],
            "segment_id": main["id"],
            "detection_agrees": det_agrees,
            "touches_edge": main["touches_edge"],
            "cutout_png": "cutout.png",
            "cutout_white_jpg": "cutout_white.jpg",
        },
        "suggested_tags": sorted({CLASS_TO_CATEGORY[s["class"]] for s in sellable}),
        "warnings": warnings,
        "models_disagree": det_classes != seg_classes,
    }

    result = {
        "input": str(image_path.name),
        "image_size_wh": [w, h],
        "settings": {"imgsz": IMGSZ, "device": DEVICE, "conf": conf,
                     "det_weights": DET_WEIGHTS, "seg_weights": SEG_WEIGHTS},
        "detections": detections,
        "segments": segments,
        "listing": listing,
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["out_dir"] = str(out)
    return result


def compare(image_path: str | Path, confs=(0.25, 0.50), out_root: Path = OUT_ROOT) -> dict:
    """같은 사진을 두 임계값으로 실제로 다시 돌려, 사라진 대상과 나란히 놓은 그림을 저장한다."""
    image_path = Path(image_path)
    runs = {c: analyze(image_path, c, out_root) for c in confs}
    lo, hi = runs[confs[0]], runs[confs[1]]

    def dropped(kind):
        # 낮은 임계값에서만 나온 대상 = 점수가 두 임계값 사이에 있던 후보
        hi_keys = {(d["class"], tuple(round(v) for v in d["box_xyxy"])) for d in hi[kind]}
        return [d for d in lo[kind] if (d["class"], tuple(round(v) for v in d["box_xyxy"])) not in hi_keys]

    report = {
        "input": image_path.name,
        "confs": list(confs),
        "count": {f"conf_{c:.2f}": {"detections": len(r["detections"]), "segments": len(r["segments"])}
                  for c, r in runs.items()},
        "only_at_low_conf": {"detections": dropped("detections"), "segments": dropped("segments")},
        "main_item": {f"conf_{c:.2f}": r["listing"]["main_item"] for c, r in runs.items()},
    }
    base = out_root / image_path.stem
    (base / "compare.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 원본 | conf 낮음 | conf 높음  (윗줄 detection, 아랫줄 segmentation)
    def row(name):
        tiles = [cv2.imread(str(image_path))] + [cv2.imread(str(Path(runs[c]["out_dir"]) / name)) for c in confs]
        th = 360
        tiles = [cv2.resize(t, (int(t.shape[1] * th / t.shape[0]), th)) for t in tiles]
        labels = ["original"] + [f"{name.split('.')[0]} conf={c}" for c in confs]
        for t, label in zip(tiles, labels):
            cv2.putText(t, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(t, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return np.hstack(tiles)

    cv2.imwrite(str(base / "side_by_side.jpg"), np.vstack([row("detect.jpg"), row("segment.jpg")]))
    report["runs"] = runs
    return report


def write_manifest(inputs: list[dict], runs: list[str], out_root: Path = OUT_ROOT) -> Path:
    """입력 파일·모델·패키지 버전·실행 위치·설정을 기록한다."""
    manifest = {
        "service": "중고거래 사진 도우미 (YOLO detection + segmentation)",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_location": {
            "where": "local (내 Mac, CPU)",
            "os": platform.platform(),
            "machine": platform.machine(),
        },
        "models": {
            "detection": {"weights": DET_WEIGHTS, "path": f"models/{DET_WEIGHTS}"},
            "segmentation": {"weights": SEG_WEIGHTS, "path": f"models/{SEG_WEIGHTS}"},
            "classes": "COCO 80 classes (pretrained, 추가 학습 없음)",
        },
        "packages": _versions(),
        "settings": {"imgsz": IMGSZ, "device": DEVICE, "confs": [0.25, 0.50],
                     "min_main_area_ratio": MIN_MAIN_AREA_RATIO},
        "inputs": inputs,
        "outputs": runs,
    }
    path = out_root / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
