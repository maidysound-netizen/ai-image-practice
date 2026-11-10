"""웹 화면: 판매자가 사진을 올리면 판매용 사진·카테고리·글 초안·사진 점검표를 받는다.

    python app.py   →  http://127.0.0.1:7860

모델이 확신하지 못하면('확인 필요') 카테고리와 물건 이름을 판매자가 고칠 수 있고, 고치면 글 초안이 바로 바뀐다.
모델 근거(박스·마스크·JSON)는 아래 '모델 근거' 칸에 접어 둔다.
"""

import json
from pathlib import Path

import gradio as gr

from market_helper import CATEGORY, OUT_ROOT, REVIEW_SCORE, analyze, draft_listing

WEB_OUT = OUT_ROOT / "web"  # 웹 화면 결과는 명령줄 결과(outputs/yolo/<사진>)를 덮어쓰지 않게 따로 둔다
CATEGORIES = list(CATEGORY) + ["기타"]
EXAMPLES = [[str(p)] for p in sorted((Path(__file__).parent / "input").glob("*.jpg")) if not p.name.startswith("._")]
MARK = {True: "✅", False: "⚠️", None: "❓"}


def _status_md(listing: dict) -> str:
    main = listing["main_item"]
    if listing["status"] == "자동 태그 가능":
        head = "## 🟢 바로 올릴 수 있어요"
    elif main:
        head = "## 🟡 확인이 필요해요 — 카테고리와 이름을 확인해 주세요"
    else:
        head = "## 🔴 판매 물건을 찾지 못했어요"
    lines = [head]
    if main:
        how = "배경 제거" if main["source"] == "segmentation" else "박스 자르기(배경 제거 실패)"
        lines.append(f"추천: **{main['name_ko']}** · {main['category']} · 점수 {main['score']} · 썸네일: {how}")
    for r in listing["review_reasons"]:
        lines.append(f"- ❓ {r}")
    return "\n".join(lines)


def _checks_md(listing: dict) -> str:
    rows = ["### 사진 점검", "| | 항목 | 근거 |", "|---|---|---|"]
    rows += [f"| {MARK[c['ok']]} | {c['item']} | {c['detail']} |" for c in listing["checks"]]
    return "\n".join(rows)


def run(image_path, conf):
    empty = (None, "사진을 올려 주세요.", "", gr.update(value=None), gr.update(value=""), "", "", None,
             None, None, None, None, None)
    if not image_path:
        return empty
    r = analyze(image_path, conf, out_root=WEB_OUT)
    out = Path(r["out_dir"])
    listing = r["listing"]
    main = listing["main_item"]

    gallery = []
    if listing["listing_thumb"]:
        gallery.append((str(out / "listing_thumb.jpg"), "목록 썸네일 1080×1080"))
    if main and main["cutout_png"]:
        gallery.append((str(out / "cutout.png"), "누끼 (투명 배경 PNG)"))
    if listing["privacy_safe"]:
        gallery.append((str(out / "privacy_safe.jpg"), "사람 흐림 처리본"))

    downloads = [str(out / g[0].split("/")[-1]) for g in gallery] + [str(out / "result.json")]
    draft = listing["draft"] or {}
    data = {k: r[k] for k in ("settings", "detections", "segments", "listing")}
    return (
        gallery or None,
        _status_md(listing),
        _checks_md(listing),
        gr.update(value=main["category"] if main else None),
        gr.update(value=main["name_ko"] if main else ""),
        draft.get("title", ""),
        draft.get("body", ""),
        downloads,
        str(out / "detect.jpg"), str(out / "segment.jpg"), str(out / "mask_overlay.jpg"),
        json.dumps(data, ensure_ascii=False, indent=2),
        listing,
    )


def redraft(listing, category, name):
    """판매자가 카테고리·이름을 고치면 글 초안을 다시 만든다(모델을 다시 돌리지 않음)."""
    if not listing or not listing.get("main_item") or not category:
        return gr.update(), gr.update()
    d = draft_listing(listing, category=category, name_ko=name or None)
    return d["title"], d["body"]


with gr.Blocks(title="중고거래 사진 도우미") as demo:
    gr.Markdown(
        "# 중고거래 사진 도우미\n"
        "상품 사진 한 장을 올리면 **판매용 썸네일**, **카테고리 추천**, **글 초안**, **사진 점검표**를 만들어 드려요. "
        f"모델이 확신하지 못하면(점수 {REVIEW_SCORE} 미만이거나 두 모델 의견이 다름) 자동으로 정하지 않고 확인을 부탁드려요."
    )
    state = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            # image_mode=None: Gradio 기본값 image_mode="RGB"는 흑백(L) 사진을 RGB로 바꾼 뒤 JPEG로 다시 저장한다.
            # 그러면 픽셀이 바뀌어 점수가 달라진다(실제로 chair.jpg에서 0.326 → 0.297, 픽셀 최대 차이 23).
            # None으로 두면 업로드 파일을 그대로 넘겨 명령줄 실행과 같은 점수가 나온다.
            # (format="png"는 Gradio 6에서 업로드에는 적용되지 않는다. 원래 확장자를 따른다.)
            image = gr.Image(type="filepath", image_mode=None, label="상품 사진", height=320)
            btn = gr.Button("분석", variant="primary")
            gr.Examples(EXAMPLES, inputs=[image], label="예시 사진")
            with gr.Accordion("고급: 점수 임계값", open=False):
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="conf (기본 0.25)",
                                 info="낮추면 후보가 늘고, 높이면 확실한 것만 남아요.")
        with gr.Column(scale=2):
            status = gr.Markdown()
            with gr.Row():
                gallery = gr.Gallery(label="판매용 사진", columns=3, height=300, object_fit="contain")
                checks = gr.Markdown()
            with gr.Row():
                category = gr.Dropdown(CATEGORIES, label="카테고리 (고칠 수 있어요)", allow_custom_value=False)
                name = gr.Textbox(label="물건 이름 (고칠 수 있어요)")
            title = gr.Textbox(label="글 제목 초안")
            body = gr.Textbox(label="글 내용 초안 — (직접 입력) 칸은 사진으로 알 수 없어 비워 두었어요", lines=7)
            files = gr.File(label="내려받기", file_count="multiple")
            with gr.Accordion("모델 근거 (박스·마스크·JSON)", open=False):
                with gr.Row():
                    det_img = gr.Image(label="detection: 박스·클래스·점수")
                    seg_img = gr.Image(label="segmentation: 마스크")
                    overlay_img = gr.Image(label="저장된 마스크를 원본에 겹친 화면")
                data = gr.Code(label="result.json", language="json")

    outputs = [gallery, status, checks, category, name, title, body, files,
               det_img, seg_img, overlay_img, data, state]
    btn.click(run, [image, conf], outputs)
    conf.release(run, [image, conf], outputs)
    image.change(run, [image, conf], outputs)
    category.input(redraft, [state, category, name], [title, body])
    name.submit(redraft, [state, category, name], [title, body])
    name.blur(redraft, [state, category, name], [title, body])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
