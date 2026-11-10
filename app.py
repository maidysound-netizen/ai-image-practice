"""웹 화면: 판매자가 사진을 올리면 판매용 사진·카테고리·글 초안·사진 점검표를 받는다.

    python app.py   →  http://127.0.0.1:7860

처음 들어온 사람이 '무엇에 쓰는 물건인지' 바로 알 수 있게
  위: 한 줄 소개 + 3단계 안내
  사진 올리기 전: 예시 전후 비교(이 사진을 넣으면 → 이런 판매용 사진이 나온다)
  사진 올린 후: 판매자가 일하는 순서대로 ① 판매용 사진 ② 카테고리·이름 ③ 글 초안(복사) ④ 사진 점검
  맨 아래: AI 판단 근거(박스·마스크·JSON)는 접어 둔다.
모델이 확신하지 못하면('확인 필요') 카테고리와 물건 이름을 판매자가 고칠 수 있고, 고치면 글 초안이 바로 바뀐다.
"""

import html
import json
from pathlib import Path

import gradio as gr

from market_helper import CATEGORY, OUT_ROOT, REVIEW_SCORE, analyze, draft_listing

ROOT = Path(__file__).parent
WEB_OUT = OUT_ROOT / "web"  # 웹 화면 결과는 명령줄 결과(outputs/yolo/<사진>)를 덮어쓰지 않게 따로 둔다
CATEGORIES = list(CATEGORY) + ["기타"]
EXAMPLES = [[str(p)] for p in sorted((ROOT / "input").glob("*.jpg")) if not p.name.startswith("._")]
# 사진을 올리기 전에 보여 줄 전후 예시 (run.py 가 만들어 둔 결과)
DEMO_BEFORE = ROOT / "input" / "laptop-desk.jpg"
DEMO_AFTER = OUT_ROOT / "laptop-desk" / "conf_0.25" / "listing_thumb.jpg"

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
#hero-block .html-container { padding: 0 !important; }  /* 제목을 아래 사진 칸과 같은 선에 맞춘다 */
#hero h1 { font-size: 1.9rem; margin: 0 0 .3rem; }
#hero p.lead { font-size: 1.05rem; opacity: .85; margin: 0 0 1rem; }
.steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .6rem; margin-bottom: .4rem; }
.step { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: .7rem .85rem;
        background: var(--block-background-fill); }
.step b { display: block; font-size: 1rem; margin-bottom: .15rem; }
.step span { font-size: .88rem; opacity: .8; }
.step .num { display: inline-block; width: 1.5rem; height: 1.5rem; line-height: 1.5rem; text-align: center;
             border-radius: 50%; background: var(--color-accent); color: white; font-weight: 700;
             font-size: .85rem; margin-right: .35rem; }
@media (max-width: 640px) {
  .steps { grid-template-columns: 1fr; gap: .35rem; }
  .step { padding: .45rem .7rem; }
  .step span:not(.num) { display: none; }  /* 좁은 화면에서는 단계 제목만 */
  #hero h1 { font-size: 1.5rem; }
  #hero p.lead { font-size: .95rem; }
}
.banner { border-radius: 12px; padding: .85rem 1rem; margin-bottom: .2rem; border: 1px solid transparent; }
.banner h3 { margin: 0 0 .25rem; font-size: 1.15rem; }
.banner p { margin: .15rem 0; font-size: .93rem; }
.banner.ok   { background: rgba(34,197,94,.12);  border-color: rgba(34,197,94,.45); }
.banner.warn { background: rgba(234,179,8,.14);  border-color: rgba(234,179,8,.5); }
.banner.bad  { background: rgba(239,68,68,.12);  border-color: rgba(239,68,68,.45); }
.checks { list-style: none; padding: 0; margin: 0; }
.checks li { display: flex; gap: .55rem; padding: .45rem 0; border-bottom: 1px solid var(--border-color-primary); }
.checks li:last-child { border-bottom: 0; }
.checks .icon { flex: 0 0 1.4rem; font-size: 1.05rem; }
.checks .what { font-weight: 600; }
.checks .why { font-size: .86rem; opacity: .8; }
.section-title h3 { margin: .6rem 0 0 !important; }
.section-title p { margin: .1rem 0 0 !important; font-size: .88rem; opacity: .75; }
.footnote { font-size: .85rem; opacity: .75; }
"""

HERO = """
<div id="hero">
  <h1>📦 중고거래 사진 도우미</h1>
  <p class="lead">팔 물건 사진 한 장만 올리세요. <b>판매용 사진</b>, <b>카테고리</b>, <b>판매 글 초안</b>을 만들고
  사진에 문제가 없는지 점검해 드려요.</p>
  <div class="steps">
    <div class="step"><b><span class="num">1</span>사진 올리기</b><span>팔 물건을 찍은 사진을 끌어다 놓거나, 아래 예시 사진을 눌러 보세요.</span></div>
    <div class="step"><b><span class="num">2</span>결과 확인</b><span>배경을 지운 판매용 사진과 추천 카테고리를 보고, 틀렸으면 고쳐요.</span></div>
    <div class="step"><b><span class="num">3</span>글 복사해서 올리기</b><span>글 초안의 (직접 입력) 칸만 채워서 중고거래 앱에 붙여 넣어요.</span></div>
  </div>
</div>
"""

MARK = {True: "✅", False: "⚠️", None: "❓"}


def _section(title: str, sub: str) -> gr.Markdown:
    return gr.Markdown(f"### {title}\n{sub}", elem_classes="section-title")


ANIMAL_KO = {"bird": "새", "cat": "고양이", "dog": "개", "horse": "말", "sheep": "양", "cow": "소",
             "elephant": "코끼리", "bear": "곰", "zebra": "얼룩말", "giraffe": "기린"}


def _banner(r: dict) -> str:
    """판정 이유를 모델 용어(detection/segmentation/임계값) 없이 판매자 말로 바꿔 보여 준다.
    원래 문장은 result.json 의 review_reasons 에 그대로 남아 있다."""
    listing = r["listing"]
    main = listing["main_item"]
    e = html.escape
    if listing["status"] == "자동 태그 가능":
        cls, head = "ok", "🟢 바로 올릴 수 있어요"
    elif main:
        cls, head = "warn", "🟡 AI가 확신하지 못했어요 — 카테고리와 이름을 한 번 확인해 주세요"
    else:
        cls, head = "bad", "🔴 팔 물건을 찾지 못했어요"
    lines = []
    if main:
        lines.append(f"<p>이 물건은 <b>{e(main['name_ko'])}</b> 같아요 · 추천 카테고리 <b>{e(main['category'])}</b>"
                     f" · AI 점수 {main['score']:.2f} (1에 가까울수록 확실)</p>")
        if main["score"] < REVIEW_SCORE:
            lines.append(f"<p>❓ AI 점수가 낮아요(기준 {REVIEW_SCORE}). 맞는지 확인해 주세요.</p>")
        if main["source"] == "segmentation" and not main["detection_agrees"]:
            other = ", ".join(sorted({d["class_ko"] for d in r["detections"]})) or "아무것도 못 찾음"
            lines.append(f"<p>❓ 두 AI의 의견이 달라요. 다른 AI는 이 사진을 <b>{e(other)}</b>(으)로 봤어요.</p>")
        if main["source"] != "segmentation":
            lines.append("<p>❓ 물건의 윤곽을 따지 못해서 배경을 지우지 못했어요. 물건 부분만 잘라 판매용 사진을 만들었어요.</p>")
    else:
        lines.append("<p>AI가 아는 물건(가구·가전·디지털기기·가방 등 80종)이 아니거나, 사진에서 물건이 잘 안 보여요. "
                     "물건만 크게, 밝은 곳에서 다시 찍어 보세요.</p>")
    animal_check = next((c for c in listing["checks"] if c["item"] == "동물이 찍히지 않음"), None)
    if animal_check and animal_check["ok"] is None:
        seen = ", ".join(sorted({ANIMAL_KO.get(s["class"], s["class"]) for s in r["segments"]
                                 if s["class"] in ANIMAL_KO}))
        lines.append(f"<p>❓ 한 AI가 사진 일부를 <b>{e(seen)}</b>(으)로 봤지만 다른 AI는 동물을 못 찾았어요. "
                     "착각일 가능성이 커요.</p>")
    if listing["privacy_safe"]:
        lines.append("<p>🔒 사람이 찍혀 있어서, 사람을 흐리게 가린 사진을 따로 만들었어요.</p>")
    return f'<div class="banner {cls}"><h3>{head}</h3>{"".join(lines)}</div>'


def _checks(listing: dict) -> str:
    items = "".join(
        f'<li><span class="icon">{MARK[c["ok"]]}</span><span><span class="what">{html.escape(c["item"])}</span><br>'
        f'<span class="why">{html.escape(c["detail"])}</span></span></li>'
        for c in listing["checks"]
    )
    return f'<ul class="checks">{items}</ul>'


def run(image_path, conf):
    hidden, shown = gr.update(visible=False), gr.update(visible=True)
    if not image_path:
        return (shown, hidden, hidden) + (None,) * 12
    r = analyze(image_path, conf, out_root=WEB_OUT)
    out = Path(r["out_dir"])
    listing = r["listing"]
    main = listing["main_item"]

    gallery = []
    if listing["listing_thumb"]:
        gallery.append((str(out / "listing_thumb.jpg"), "목록용 정사각형 사진 (흰 배경)"))
    if main and main["cutout_png"]:
        gallery.append((str(out / "cutout.png"), "배경을 지운 사진 (투명 PNG)"))
    if listing["privacy_safe"]:
        gallery.append((str(out / "privacy_safe.jpg"), "사람을 흐리게 가린 사진"))

    downloads = [p for p, _ in gallery] + [str(out / "result.json")]
    draft = listing["draft"] or {}
    data = {k: r[k] for k in ("settings", "detections", "segments", "listing")}
    return (
        hidden, shown, gr.update(visible=main is not None),
        _banner(r),
        gallery or None,
        gr.update(value=main["category"] if main else None),
        gr.update(value=main["name_ko"] if main else ""),
        draft.get("title", ""),
        draft.get("body", ""),
        _checks(listing),
        downloads,
        str(out / "detect.jpg"), str(out / "segment.jpg"),
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
    state = gr.State()
    gr.HTML(HERO, padding=False, elem_id="hero-block")

    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=300):
            # image_mode=None: Gradio 기본값 image_mode="RGB"는 흑백(L) 사진을 RGB로 바꾼 뒤 JPEG로 다시 저장한다.
            # 그러면 픽셀이 바뀌어 점수가 달라진다(실제로 chair.jpg에서 0.326 → 0.297, 픽셀 최대 차이 23).
            # None으로 두면 업로드 파일을 그대로 넘겨 명령줄 실행과 같은 점수가 나온다.
            # (format="png"는 Gradio 6에서 업로드에는 적용되지 않는다. 원래 확장자를 따른다.)
            image = gr.Image(type="filepath", image_mode=None, label="팔 물건 사진", height=340,
                             sources=["upload", "clipboard"])
            btn = gr.Button("판매 준비 시작하기", variant="primary", size="lg")
            gr.Examples(EXAMPLES, inputs=[image], label="사진이 없으면 예시로 해 보세요")
            with gr.Accordion("고급 설정 (발표·실험용)", open=False):
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="AI 점수 기준 (conf)",
                                 info="낮추면 더 많이 찾지만 틀린 것도 섞이고, 높이면 확실한 것만 남아요. 기본 0.25.")
            gr.Markdown(
                "**알아 두세요**\n"
                "- AI가 아는 물건은 80종(가구·가전·디지털기기·가방·스포츠용품 등)이에요. 옷·신발·화장품은 아직 못 알아봐요.\n"
                f"- AI 점수가 {REVIEW_SCORE} 미만이거나 두 AI의 의견이 다르면 자동으로 정하지 않고 확인을 부탁드려요.\n"
                "- 브랜드·상태·가격은 사진으로 알 수 없어서 지어내지 않아요.\n"
                "- 사진은 이 컴퓨터 안에서만 처리돼요.",
                elem_classes="footnote",
            )

        with gr.Column(scale=3, min_width=320):
            with gr.Column(visible=True) as intro_col:
                _section("이렇게 바뀌어요", "책상 위 노트북 사진을 넣으면, 배경을 지운 목록용 사진과 글 초안이 나와요.")
                with gr.Row():
                    gr.Image(str(DEMO_BEFORE), label="올린 사진", interactive=False, height=260,
                             buttons=["fullscreen"])
                    gr.Image(str(DEMO_AFTER) if DEMO_AFTER.exists() else None, label="만들어진 판매용 사진",
                             interactive=False, height=260, buttons=["fullscreen"])
                gr.Markdown(
                    "함께 나오는 것: **카테고리 추천**(디지털기기) · **글 초안** · **사진 점검**"
                    "(\"노트북이 사진의 9%만 차지해요, 더 가까이 찍으세요\" 같은 조언)",
                    elem_classes="footnote",
                )

            with gr.Column(visible=False) as result_col:
                banner = gr.HTML()
                _section("① 판매용 사진", "중고거래 앱 목록에 올릴 사진이에요. 눌러서 크게 보고 내려받을 수 있어요.")
                gallery = gr.Gallery(show_label=False, columns=3, height=280, object_fit="contain",
                                     buttons=["download", "fullscreen"])
                with gr.Column(visible=True) as draft_col:
                    _section("② 카테고리와 물건 이름", "AI가 추천한 값이에요. 틀렸으면 고치세요. 글 초안이 바로 바뀌어요.")
                    with gr.Row():
                        category = gr.Dropdown(CATEGORIES, label="카테고리", allow_custom_value=False)
                        name = gr.Textbox(label="물건 이름", placeholder="예: 백팩, 노트북")
                    _section("③ 판매 글 초안", "오른쪽 위 복사 버튼으로 복사해서 (직접 입력) 칸만 채우세요.")
                    title = gr.Textbox(label="제목", buttons=["copy"])
                    body = gr.Textbox(label="내용", lines=8, buttons=["copy"])
                _section("④ 사진 점검", "올리기 전에 고치면 좋은 점이에요.")
                checks = gr.HTML()
                files = gr.File(label="한꺼번에 내려받기", file_count="multiple")
                with gr.Accordion("AI가 어떻게 판단했나 (발표·검토용)", open=False):
                    gr.Markdown("두 AI가 따로 본 결과예요. 박스는 '무엇이 어디 있나', 색칠은 '물건의 실제 윤곽'이에요.",
                                elem_classes="footnote")
                    with gr.Row():
                        det_img = gr.Image(label="AI 1 (detection): 박스·이름·점수")
                        seg_img = gr.Image(label="AI 2 (segmentation): 윤곽")
                    data = gr.Code(label="result.json (전체 결과)", language="json")

    outputs = [intro_col, result_col, draft_col, banner, gallery, category, name, title, body, checks, files,
               det_img, seg_img, data, state]
    btn.click(run, [image, conf], outputs)
    conf.release(run, [image, conf], outputs)
    image.change(run, [image, conf], outputs)
    category.input(redraft, [state, category, name], [title, body])
    name.submit(redraft, [state, category, name], [title, body])
    name.blur(redraft, [state, category, name], [title, body])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=CSS,
                theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"))
