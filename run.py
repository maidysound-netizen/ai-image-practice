"""명령줄 실행: input/ 의 사진(또는 지정한 사진)을 conf 0.25 / 0.50 으로 분석하고 manifest.json 을 남긴다.

    python run.py                      # input/ 의 모든 jpg/png
    python run.py input/my-photo.jpg   # 특정 사진만
"""

import json
import sys
from pathlib import Path

from market_helper import ROOT, compare, write_manifest

SOURCES_FILE = ROOT / "input" / "SOURCES.json"  # 사진별 출처·공유 조건


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]]
    images = args or sorted(p for p in (ROOT / "input").iterdir()
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith("._"))
    sources = json.loads(SOURCES_FILE.read_text(encoding="utf-8")) if SOURCES_FILE.exists() else {}

    inputs, outputs = [], []
    for img in images:
        report = compare(img)
        inputs.append({"file": f"input/{img.name}", **sources.get(img.name, {"source": "기록 없음"})})
        outputs.append(f"outputs/yolo/{img.stem}/")
        print(f"\n== {img.name}")
        for key, c in report["count"].items():
            print(f"  {key}: detection {c['detections']}개, segmentation {c['segments']}개")
        low = report["only_at_low_conf"]["detections"]
        print(f"  0.25에서만 나온 박스: {[(d['class'], d['score']) for d in low] or '없음'}")
        listing = report["runs"][0.25]["listing"]
        main_item = listing["main_item"]
        print(f"  대표 물건(0.25): {main_item['name_ko'] + ' / ' + main_item['category'] if main_item else '없음'}"
              f"  → {listing['status']}")
        for r in listing["review_reasons"]:
            print(f"  ? {r}")
        for w in listing["warnings"]:
            print(f"  ⚠ {w}")

    path = write_manifest(inputs, outputs)
    print(f"\nmanifest: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
