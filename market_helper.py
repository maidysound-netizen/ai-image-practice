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
    # 판매 카테고리에는 없지만 화면에 '다른 AI는 ○○로 봤어요'로 나올 수 있는 나머지 COCO 클래스
    "airplane": "비행기", "train": "기차", "boat": "배", "traffic light": "신호등", "fire hydrant": "소화전",
    "stop sign": "정지 표지판", "parking meter": "주차 미터기", "bench": "벤치", "bird": "새", "cat": "고양이",
    "dog": "개", "horse": "말", "sheep": "양", "cow": "소", "elephant": "코끼리", "bear": "곰",
    "zebra": "얼룩말", "giraffe": "기린", "banana": "바나나", "apple": "사과", "sandwich": "샌드위치",
    "orange": "오렌지", "broccoli": "브로콜리", "carrot": "당근", "hot dog": "핫도그", "pizza": "피자",
    "donut": "도넛", "cake": "케이크", "toilet": "변기", "sink": "싱크대",
}
ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

# 품질 점검 기준 (근거: 썸네일에서 물건이 화면의 15% 미만이면 무엇인지 알아보기 어렵다고 판단)
MIN_MAIN_AREA_RATIO = 0.15
# 대표 물건 점수가 이보다 낮거나 두 모델 의견이 다르면 자동 등록하지 않고 '확인 필요'로 둔다
REVIEW_SCORE = 0.50
# 박스가 사진 가장자리에서 이 픽셀 이내면 '잘린 물건'으로 본다
EDGE_PX = 3
# 다른 물건 마스크의 이 비율 이상이 대표 물건 마스크 안에 있으면 대표 물건의 일부로 본다
# (laptop-desk.jpg 에서 노트북 자체의 키보드를 '다른 물건'으로 세던 오경보를 막기 위해)
PART_OF_RATIO = 0.5
# 중고거래 앱 목록 썸네일: 정사각형, 흰 배경, 물건 둘레 여백
THUMB_SIZE = 1080
THUMB_MARGIN = 0.08


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


def _padded_box(box, w: int, h: int, pad_ratio: float = 0.05) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    pad = int(pad_ratio * max(x2 - x1, y2 - y1))
    return max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, w), min(y2 + pad, h)


def _square_thumb(crop: np.ndarray, size: int = THUMB_SIZE, margin: float = THUMB_MARGIN) -> np.ndarray:
    """잘라 낸 물건(BGR 또는 투명 배경 BGRA)을 흰 정사각형 가운데에 놓는다."""
    if crop.shape[2] == 4:
        a = crop[:, :, 3:4].astype(np.float32) / 255.0
        crop = (crop[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    inner = int(size * (1 - 2 * margin))
    s = inner / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                      interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y0, x0 = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return canvas


def draft_listing(listing: dict, category: str | None = None, name_ko: str | None = None) -> dict:
    """판매 글 초안. 사진으로 알 수 없는 것(브랜드·상태·가격)은 지어내지 않고 빈칸으로 둔다."""
    item = listing["main_item"]
    name = name_ko or item["name_ko"]
    cat = category or item["category"]
    body = [
        f"{name} 판매합니다.",
        "",
        "- 브랜드/모델명: (직접 입력)",
        "- 구매 시기·사용 기간: (직접 입력)",
        "- 상태: (흠집·고장·구성품 여부 직접 입력)",
        "- 가격: (직접 입력)",
        "- 거래 방법: (직거래 장소 / 택배)",
    ]
    todo = [c["detail"] for c in listing.get("checks", []) if c["ok"] is not True]
    return {
        "title": f"{name} 팝니다",
        "category": cat,
        "tags": sorted({name, cat.split("/")[0]}),
        "body": "\n".join(body),
        "auto_filled": ["물건 이름", "카테고리"],
        "needs_seller": ["브랜드/모델명", "사용 기간", "상태", "가격", "거래 방법"],
        "photo_todo": todo,
        "note": "물건 이름과 카테고리는 사진 인식으로 추천한 값입니다. 올리기 전에 확인하세요.",
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
    # 같은 폴더에 이전 실행의 파일(예: 이번에는 만들지 않은 cutout.png)이 남지 않게 비우고 시작한다
    if out.exists():
        for f in list(out.glob("masks/*")) + [f for f in out.iterdir() if f.is_file()]:
            # exFAT: macOS가 파일을 지울 때 짝인 ._파일도 함께 지우므로 이미 없을 수 있다
            f.unlink(missing_ok=True)
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

    fallback = None  # segmentation이 못 찾았을 때 detection 결과로 대신하는 대표 물건
    checks = []  # 판매자에게 보여 줄 사진 점검표: {"item", "ok", "detail"}
    parts = []   # 대표 물건의 일부로 본 대상(예: 노트북 안의 keyboard) → '다른 물건'으로 세지 않는다
    thumb_source = None

    if main is None:
        det_sellable = [d for d in detections if d["class"] in CLASS_TO_CATEGORY]
        if det_sellable:
            best = max(det_sellable, key=lambda d: d["score"])
            fallback = best
            review_reasons.append(
                f"segmentation은 판매 물건을 못 찾았고, detection만 '{best['class_ko']}'(점수 {best['score']})로 봤습니다. "
                f"배경을 지우지 못해 박스 영역만 잘라 썸네일을 만들었습니다."
            )
        else:
            review_reasons.append("판매할 만한 물건을 찾지 못했습니다(모델이 아는 80종 밖이거나 점수가 임계값보다 낮음).")
    else:
        if main["score"] < REVIEW_SCORE:
            review_reasons.append(f"대표 물건 '{main['class_ko']}'의 점수가 {main['score']}로 낮습니다(기준 {REVIEW_SCORE}).")
        if not det_agrees:
            det_names = ", ".join(f"{d['class_ko']}({d['score']})" for d in detections) or "없음"
            review_reasons.append(f"detection 모델은 다르게 봤습니다: {det_names}.")
        mask = cv2.imread(str(out / main["mask_file"]), cv2.IMREAD_GRAYSCALE)
        x1, y1, x2, y2 = _padded_box(main["box_xyxy"], w, h)
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask
        cv2.imwrite(str(out / "cutout.png"), rgba[y1:y2, x1:x2])  # 투명 배경
        white = img.copy()
        white[mask == 0] = 255
        cv2.imwrite(str(out / "cutout_white.jpg"), white[y1:y2, x1:x2])  # 흰 배경
        cv2.imwrite(str(out / "listing_thumb.jpg"), _square_thumb(rgba[y1:y2, x1:x2]))
        thumb_source = "segmentation 누끼"

        # 다른 물건 중 마스크의 절반 이상이 대표 물건 마스크 안에 있으면 '부속'(예: 노트북의 키보드)으로 본다.
        main_px = mask > 0
        for s in sellable:
            if s is main:
                continue
            m = cv2.imread(str(out / s["mask_file"]), cv2.IMREAD_GRAYSCALE) > 0
            inside = float((m & main_px).sum()) / max(int(m.sum()), 1)
            s["inside_main_ratio"] = round(inside, 3)
            if inside >= PART_OF_RATIO:
                parts.append(s)

    if main is None and fallback is not None:
        x1, y1, x2, y2 = _padded_box(fallback["box_xyxy"], w, h)
        cv2.imwrite(str(out / "listing_thumb.jpg"), _square_thumb(img[y1:y2, x1:x2]))
        thumb_source = "detection 박스 자르기(배경 그대로)"

    # ---- 사진 점검표 (판매자용). 근거 숫자를 함께 적는다.
    if main is not None:
        big = main["mask_area_ratio"] >= MIN_MAIN_AREA_RATIO
        checks.append({"item": "물건이 충분히 크게 찍힘", "ok": big,
                       "detail": f"사진의 {main['mask_area_ratio']:.0%} (기준 {MIN_MAIN_AREA_RATIO:.0%})"
                                 + ("" if big else " → 더 가까이에서 찍으면 목록 썸네일에서 잘 보입니다.")})
        checks.append({"item": "물건 전체가 사진 안에 있음", "ok": not main["touches_edge"],
                       "detail": "가장자리에 닿지 않음" if not main["touches_edge"] else "가장자리에서 잘림 → 전체가 나오게 다시 찍으세요."})
        others = [s for s in sellable if s is not main and s not in parts]
        names = ", ".join(sorted({s["class_ko"] for s in others}))
        detail = f"함께 찍힌 물건: {names}" if others else "판매 물건만 찍힘"
        if parts:
            detail += f" (대표 물건의 일부로 보고 제외: {', '.join(sorted({p['class_ko'] for p in parts}))})"
        checks.append({"item": "판매 물건만 찍힘", "ok": not others, "detail": detail})
        if main["touches_edge"]:
            warnings.append(f"{main['class_ko']}이(가) 사진 가장자리에서 잘렸습니다. 전체가 나오게 찍으세요.")
        if not big:
            warnings.append(f"{main['class_ko']}이(가) 사진의 {main['mask_area_ratio']:.0%}만 차지합니다. "
                            f"더 가까이에서 찍으면 목록 썸네일에서 잘 보입니다.")
        if others:
            warnings.append(f"다른 물건({names})도 같이 찍혀 있습니다. 판매 물건이 무엇인지 헷갈릴 수 있어요.")

    # 사람: 한 모델만 봐도 경고하고 자동으로 흐리게 한다(개인정보는 놓치는 쪽이 더 위험).
    people = [s for s in segments if s["class"] == "person"]
    privacy_file = None
    if people:
        warnings.append(f"사람 {len(people)}명이 찍혀 있습니다. privacy_safe.jpg 에 사람을 흐리게 처리했습니다(개인정보).")
        blur_mask = np.zeros((h, w), np.uint8)
        for p in people:
            blur_mask |= cv2.imread(str(out / p["mask_file"]), cv2.IMREAD_GRAYSCALE)
        k = max(15, (min(h, w) // 20) | 1)
        blur_mask = cv2.dilate(blur_mask, np.ones((k, k), np.uint8))
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=k)
        safe = img.copy()
        safe[blur_mask > 0] = blurred[blur_mask > 0]
        cv2.imwrite(str(out / "privacy_safe.jpg"), safe)
        privacy_file = "privacy_safe.jpg"
    checks.append({"item": "사람이 찍히지 않음", "ok": not people,
                   "detail": "사람 없음" if not people else f"사람 {len(people)}명 → 흐리게 처리한 사진을 쓰세요."})

    # 동물: 두 모델이 모두 동물로 봤을 때만 경고. 한쪽만 보면 오인일 수 있어 '확인 필요'로 돌린다
    # (실제로 chair.jpg 에서 segmentation 이 의자 다리를 dog 로 봤다).
    animals = [s for s in segments if s["class"] in ANIMALS]
    det_animals = [d for d in detections if d["class"] in ANIMALS]
    agreed = [a for a in animals if any(_iou(a["box_xyxy"], d["box_xyxy"]) >= 0.3 for d in det_animals)]
    if agreed or (det_animals and not animals):
        warnings.append("동물이 찍혀 있습니다. 대부분의 중고거래 앱은 반려동물 거래를 금지합니다.")
    elif animals:
        names = ", ".join(f"{a['class']}({a['score']})" for a in animals)
        review_reasons.append(f"segmentation만 동물로 봤습니다: {names}. detection은 동물을 찾지 않아 오인일 수 있습니다.")
    # ok: True=통과, False=문제 있음, None=모델끼리 의견이 달라 사람이 봐야 함
    checks.append({"item": "동물이 찍히지 않음",
                   "ok": False if (agreed or det_animals) else (None if animals else True),
                   "detail": "두 모델 모두 동물로 봄" if (agreed or det_animals)
                   else ("한 모델만 동물로 봄 → 확인 필요" if animals else "동물 없음")})

    # detection과 segmentation이 서로 다르게 본 경우도 그대로 기록(숨기지 않음)
    det_classes = sorted(d["class"] for d in detections)
    seg_classes = sorted(s["class"] for s in segments)

    item = main or fallback
    listing = {
        "status": "확인 필요" if review_reasons else "자동 태그 가능",
        "review_reasons": review_reasons,
        "main_item": None if item is None else {
            "class": item["class"],
            "name_ko": item["class_ko"],
            "category": CLASS_TO_CATEGORY[item["class"]],
            "score": item["score"],
            "source": "segmentation" if main else "detection",
            "segment_id": main["id"] if main else None,
            "detection_agrees": det_agrees,
            "touches_edge": main["touches_edge"] if main else None,
            "cutout_png": "cutout.png" if main else None,
            "cutout_white_jpg": "cutout_white.jpg" if main else None,
        },
        "listing_thumb": "listing_thumb.jpg" if thumb_source else None,
        "listing_thumb_source": thumb_source,
        "privacy_safe": privacy_file,
        "parts_of_main": [p["class"] for p in parts],
        "suggested_tags": sorted({CLASS_TO_CATEGORY[s["class"]] for s in sellable if s not in parts}),
        "checks": checks,
        "warnings": warnings,
        "models_disagree": det_classes != seg_classes,
    }
    listing["draft"] = draft_listing(listing) if item else None

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
