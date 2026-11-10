"""웹 화면: 사진을 올리고 임계값 슬라이더를 움직이면 결과가 바뀐다.

    python app.py   →  http://127.0.0.1:7860
"""

import json
from pathlib import Path

import gradio as gr

from market_helper import OUT_ROOT, analyze

WEB_OUT = OUT_ROOT / "web"  # 웹 화면 결과는 명령줄 결과(outputs/yolo/<사진>)를 덮어쓰지 않게 따로 둔다

EXAMPLES = [[str(p)] for p in sorted((Path(__file__).parent / "input").glob("*.jpg")) if not p.name.startswith("._")]


def run(image_path, conf):
    if not image_path:
        return None, None, None, None, "사진을 올려 주세요.", None
    r = analyze(image_path, conf, out_root=WEB_OUT)
    out = Path(r["out_dir"])
    listing = r["listing"]
    main = listing["main_item"]

    lines = [f"### 결과: **{listing['status']}**"]
    if main:
        lines.append(f"- 대표 물건: **{main['name_ko']}** ({main['class']}, 점수 {main['score']})")
        lines.append(f"- 추천 카테고리: **{main['category']}**")
    if listing["suggested_tags"]:
        lines.append(f"- 태그 후보: {', '.join(listing['suggested_tags'])}")
    for reason in listing["review_reasons"]:
        lines.append(f"- ❓ {reason}")
    for w in listing["warnings"]:
        lines.append(f"- ⚠️ {w}")
    lines.append(f"\n저장 위치: `{out}`")

    cutout = str(out / "cutout.png") if (out / "cutout.png").exists() and main else None
    data = {k: r[k] for k in ("settings", "detections", "segments", "listing")}
    return (str(out / "detect.jpg"), str(out / "segment.jpg"), str(out / "mask_overlay.jpg"),
            cutout, "\n".join(lines), json.dumps(data, ensure_ascii=False, indent=2))


with gr.Blocks(title="중고거래 사진 도우미") as demo:
    gr.Markdown(
        "# 중고거래 사진 도우미\n"
        "상품 사진을 올리면 **detection**(yolo26n)이 물건을 찾아 카테고리를 붙이고, "
        "**segmentation**(yolo26n-seg)이 윤곽을 따서 배경을 지운 사진을 만듭니다. "
        "점수가 낮거나 두 모델 의견이 다르면 자동으로 정하지 않고 '확인 필요'로 표시합니다."
    )
    with gr.Row():
        with gr.Column(scale=1):
            # image_mode=None: Gradio 기본값 image_mode="RGB"는 흑백(L) 사진을 RGB로 바꾼 뒤 JPEG로 다시 저장한다.
            # 그러면 픽셀이 바뀌어 점수가 달라진다(실제로 chair.jpg에서 0.326 → 0.297, 픽셀 최대 차이 23).
            # None으로 두면 업로드 파일을 그대로 넘겨 명령줄 실행과 같은 점수가 나온다.
            # (format="png"는 Gradio 6에서 업로드에는 적용되지 않는다. 원래 확장자를 따른다.)
            image = gr.Image(type="filepath", image_mode=None, label="상품 사진")
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="점수 임계값 (conf)")
            btn = gr.Button("분석", variant="primary")
            gr.Examples(EXAMPLES, inputs=[image])
        with gr.Column(scale=2):
            summary = gr.Markdown()
            with gr.Row():
                det_img = gr.Image(label="detection: 박스·클래스·점수")
                seg_img = gr.Image(label="segmentation: 마스크")
            with gr.Row():
                overlay_img = gr.Image(label="저장된 마스크를 원본에 겹친 화면")
                cutout_img = gr.Image(label="누끼 (투명 배경 PNG)", image_mode="RGBA")
            data = gr.Code(label="result.json", language="json")

    outputs = [det_img, seg_img, overlay_img, cutout_img, summary, data]
    btn.click(run, [image, conf], outputs)
    conf.release(run, [image, conf], outputs)
    image.change(run, [image, conf], outputs)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
