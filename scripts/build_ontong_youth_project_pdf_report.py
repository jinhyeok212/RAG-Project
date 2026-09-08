import csv
import json
import math
import textwrap
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/processed/ontong_youth_project_report"
OUT_PDF = OUT_DIR / "ontong_youth_rag_mvp_report_readable_v3.pdf"
OUT_SUMMARY = OUT_DIR / "ontong_youth_rag_mvp_report_summary.json"

PATHS = {
    "collection_manifest": ROOT / "data/processed/ontong_youth_catalog/ontong_youth_policy_final_collection_manifest.json",
    "normalization_manifest": ROOT / "data/processed/ontong_youth_normalization/normalization_manifest.json",
    "mvp_manifest": ROOT / "data/processed/ontong_youth_mvp_400/ontong_youth_mvp_400_manifest.json",
    "document_manifest": ROOT / "data/processed/ontong_youth_mvp_400_documents/document_manifest.json",
    "documents": ROOT / "data/processed/ontong_youth_mvp_400_documents/documents.jsonl",
    "document_schema_md": ROOT / "data/processed/ontong_youth_mvp_400_documents/DOCUMENT_SCHEMA.md",
    "eval_manifest": ROOT / "data/processed/ontong_youth_eval_questions_v2_final_47/eval_questions_manifest.json",
    "eval_questions": ROOT / "data/processed/ontong_youth_eval_questions_v2_final_47/eval_questions.csv",
    "qrels": ROOT / "data/processed/ontong_youth_eval_questions_v2_final_47/qrels.csv",
    "resolved_label2": ROOT / "data/processed/ontong_youth_eval_questions_v2_final_47/resolved_label2_actions.csv",
    "v1_feedback": ROOT / "data/processed/ontong_youth_eval_questions_v2_refined/v1_human_feedback_analysis.json",
    "adjudication_summary": ROOT / "data/processed/ontong_youth_eval_questions_v2_refined/adjudication_comparison_summary.json",
    "claude_repair_summary": ROOT / "data/processed/ontong_youth_eval_questions_v2_repaired/claude_repair_review_summary.md",
}


KST = timezone(timedelta(hours=9))
PAGE_SIZE = (11.69, 8.27)  # A4 landscape in inches
FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")
BOLD_FONT_PATH = Path("C:/Windows/Fonts/malgunbd.ttf")


def setup_font():
    fm.fontManager.addfont(str(FONT_PATH))
    fm.fontManager.addfont(str(BOLD_FONT_PATH))
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path, limit=None):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def make_fig():
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="#fbfbf8")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def add_title(ax, title, subtitle=None):
    ax.text(0.055, 0.925, title, fontsize=22, fontweight="bold", color="#111827", va="top")
    if subtitle:
        ax.text(0.055, 0.875, subtitle, fontsize=10.5, color="#4b5563", va="top")


def add_footer(ax, page, note="온통청년 정책 RAG MVP 구축 보고서"):
    ax.plot([0.055, 0.945], [0.06, 0.06], color="#d1d5db", lw=0.8)
    ax.text(0.055, 0.035, note, fontsize=8, color="#6b7280", va="center")
    ax.text(0.945, 0.035, str(page), fontsize=8, color="#6b7280", va="center", ha="right")


def wrap_text(text, width=56, break_long_words=False):
    if text is None:
        return ""
    out = []
    for para in str(text).splitlines():
        if not para.strip():
            out.append("")
        else:
            out.extend(textwrap.wrap(para, width=width, break_long_words=break_long_words, replace_whitespace=False))
    return "\n".join(out)


def paragraph(ax, x, y, text, width=78, fontsize=10, color="#1f2937", line_gap=0.032, bullet=False):
    # Keep prose inside the visible page. Korean glyphs and mixed English paths
    # render wider than plain ASCII, so a raw character count can run past x=1.
    safe_width = max(24, int((0.91 - x) * (82 if fontsize <= 10.5 else 68)))
    effective_width = min(width, safe_width)
    lines = wrap_text(text, effective_width, break_long_words=True).splitlines()
    cursor = y
    for i, line in enumerate(lines):
        prefix = "• " if bullet and i == 0 else ("  " if bullet else "")
        ax.text(x, cursor, prefix + line, fontsize=fontsize, color=color, va="top")
        cursor -= line_gap
    return cursor


def card(ax, x, y, w, h, title, body, accent="#2563eb"):
    rect = patches.FancyBboxPatch(
        (x, y - h),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.01",
        linewidth=0.8,
        edgecolor="#e5e7eb",
        facecolor="#ffffff",
    )
    ax.add_patch(rect)
    ax.add_patch(patches.Rectangle((x, y - h), 0.008, h, color=accent, lw=0))
    ax.text(x + 0.022, y - 0.035, title, fontsize=11.5, fontweight="bold", color="#111827", va="top")
    paragraph(ax, x + 0.022, y - 0.075, body, width=42, fontsize=9.2, color="#374151", line_gap=0.027)


def table(ax, x, y, rows, col_widths, row_h=0.044, fontsize=8.7, header=True):
    cursor = y
    for r_idx, row in enumerate(rows):
        fill = "#eef2ff" if header and r_idx == 0 else ("#ffffff" if r_idx % 2 else "#f9fafb")
        xcur = x
        for c_idx, cell in enumerate(row):
            w = col_widths[c_idx]
            rect = patches.Rectangle((xcur, cursor - row_h), w, row_h, facecolor=fill, edgecolor="#d1d5db", lw=0.5)
            ax.add_patch(rect)
            ax.text(
                xcur + 0.008,
                cursor - 0.012,
                wrap_text(str(cell), max(8, int(w * 88)), break_long_words=True),
                fontsize=fontsize,
                fontweight="bold" if header and r_idx == 0 else "normal",
                color="#111827",
                va="top",
            )
            xcur += w
        cursor -= row_h
    return cursor


def draw_text_box(ax, x, y, w, h, title, body, fontsize=8.8, accent="#2563eb", body_width=76):
    rect = patches.FancyBboxPatch(
        (x, y - h),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.008",
        linewidth=0.7,
        edgecolor="#d1d5db",
        facecolor="#ffffff",
    )
    ax.add_patch(rect)
    ax.add_patch(patches.Rectangle((x, y - h), 0.006, h, color=accent, lw=0))
    ax.text(x + 0.018, y - 0.026, title, fontsize=10, fontweight="bold", color="#111827", va="top")
    paragraph(ax, x + 0.018, y - 0.062, body, width=body_width, fontsize=fontsize, color="#374151", line_gap=0.024)


def draw_pipeline(ax, steps):
    xs = [0.08 + i * 0.12 for i in range(len(steps))]
    y = 0.54
    for i, (label, count, color) in enumerate(steps):
        ax.add_patch(
            patches.FancyBboxPatch(
                (xs[i], y - 0.11),
                0.1,
                0.16,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                facecolor=color,
                edgecolor="#ffffff",
                lw=1.2,
            )
        )
        ax.text(xs[i] + 0.05, y + 0.012, label, ha="center", va="center", fontsize=9, fontweight="bold", color="#111827")
        ax.text(xs[i] + 0.05, y - 0.043, count, ha="center", va="center", fontsize=13, fontweight="bold", color="#111827")
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i] + 0.122, y - 0.03), xytext=(xs[i] + 0.103, y - 0.03), arrowprops=dict(arrowstyle="->", lw=1.2, color="#6b7280"))


def small_bar(ax, data, title, x, y, w, h, color="#3b82f6"):
    sub = ax.figure.add_axes([x, y, w, h])
    labels = list(data.keys())
    values = list(data.values())
    sub.barh(range(len(labels)), values, color=color)
    sub.set_yticks(range(len(labels)))
    sub.set_yticklabels(labels, fontsize=8)
    sub.invert_yaxis()
    sub.set_title(title, fontsize=10, pad=6)
    sub.tick_params(axis="x", labelsize=8)
    sub.grid(axis="x", alpha=0.2)
    for spine in sub.spines.values():
        spine.set_visible(False)
    for i, v in enumerate(values):
        sub.text(v + max(values) * 0.02, i, str(v), va="center", fontsize=8)


def sample_policy(docs):
    for d in docs:
        if "청년창업 특례보증" in d.get("title", ""):
            return d
    return docs[0]


def find_eval_example(eval_rows, keyword):
    for r in eval_rows:
        if keyword in r.get("query", "") or keyword in r.get("source_title", ""):
            return r
    return eval_rows[0]


def build_report():
    setup_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    collection = read_json(PATHS["collection_manifest"])
    norm = read_json(PATHS["normalization_manifest"])
    mvp = read_json(PATHS["mvp_manifest"])
    doc_manifest = read_json(PATHS["document_manifest"])
    eval_manifest = read_json(PATHS["eval_manifest"])
    docs = read_jsonl(PATHS["documents"], limit=None)
    eval_rows = read_csv(PATHS["eval_questions"])
    qrels_rows = read_csv(PATHS["qrels"])
    resolved = read_csv(PATHS["resolved_label2"])

    doc_categories = Counter(d["metadata"]["primary_category"] for d in docs)
    doc_middle = Counter(d["metadata"]["primary_mclsf"] for d in docs)
    doc_lengths = [len(d.get("retrieval_text", "")) for d in docs]
    eval_types = Counter(r["question_type"] for r in eval_rows)
    eval_categories = Counter(r["primary_category"] for r in eval_rows)
    resolution_actions = Counter(r["resolution_action"] for r in eval_rows)

    multi_q = [qid for qid, c in Counter(r["query_id"] for r in qrels_rows).items() if c > 1]
    policy_ex = sample_policy(docs)
    eval_ex = find_eval_example(eval_rows, "청년창업 특례보증")
    qrels_ex = [r for r in qrels_rows if r["query_id"] == eval_ex["query_id"]]

    pages = []
    with PdfPages(OUT_PDF) as pdf:
        page = 1

        # Cover
        fig, ax = make_fig()
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, color="#f8fafc"))
        ax.add_patch(patches.Circle((0.86, 0.78), 0.18, color="#dbeafe", alpha=0.7))
        ax.add_patch(patches.Circle((0.76, 0.2), 0.13, color="#dcfce7", alpha=0.8))
        ax.text(0.06, 0.76, "온통청년 정책 RAG MVP\n데이터 구축 보고서", fontsize=30, fontweight="bold", color="#111827", va="top")
        ax.text(0.06, 0.58, "수집 → 정규화 → MVP 400 문서 선정 → RAG 문서 스키마 → 평가질문 47개와 qrels 답안지 구축", fontsize=13, color="#374151")
        stats = [
            ("API 수집", f"{collection['final_record_count']:,}건"),
            ("MVP 문서", f"{mvp['selected_count']:,}개"),
            ("평가 질문", f"{eval_manifest['counts']['eval_questions']}개"),
            ("qrels", f"{eval_manifest['counts']['qrels_rows']}행"),
        ]
        for i, (k, v) in enumerate(stats):
            card(ax, 0.07 + i * 0.22, 0.38, 0.18, 0.16, k, v, ["#2563eb", "#059669", "#7c3aed", "#f97316"][i])
        ax.text(0.06, 0.14, f"생성일: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')}", fontsize=10, color="#6b7280")
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Executive summary
        fig, ax = make_fig()
        add_title(ax, "1. 한눈에 보는 현재 상태", "2번 담당 범위: 데이터 수집, 정제, 문서 스키마, 평가셋/qrels 준비까지 완료")
        draw_pipeline(
            ax,
            [
                ("API 수집", "2,728", "#dbeafe"),
                ("정규화", "2,728", "#e0f2fe"),
                ("MVP 선정", "400", "#dcfce7"),
                ("문서 스키마", "400", "#fef3c7"),
                ("평가질문", "47", "#f3e8ff"),
                ("qrels", "50", "#ffedd5"),
                ("3번 인계", "준비", "#e5e7eb"),
            ],
        )
        y = 0.32
        y = paragraph(ax, 0.07, y, "현재까지 만든 핵심 산출물은 검색 대상 문서 400개와 평가 질문 47개, 그리고 문서 단위 정답표(qrels) 50행이다. qrels가 질문 수보다 많은 이유는 3개 질문에서 정답으로 인정해야 하는 문서가 2개씩 있기 때문이다.", width=105, fontsize=11)
        y = paragraph(ax, 0.07, y - 0.015, "아직 하지 않은 일은 chunking, chunk_id 생성, ground_truth_chunk_ids 매핑, embedding, Chroma 적재, 실제 retrieval 평가다. 이 부분은 다음 담당자가 이어서 수행한다.", width=105, fontsize=11)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Data collection
        fig, ax = make_fig()
        add_title(ax, "2. 원본 데이터 수집", "온통청년 청년정책 API에서 전체 정책을 JSON으로 수집")
        rows = [
            ["항목", "값"],
            ["API endpoint", "https://www.youthcenter.go.kr/go/ythip/getPlcy"],
            ["최종 수집 건수", f"{collection['final_record_count']:,}건"],
            ["API key 저장 여부", "파일에 저장하지 않음"],
            ["중복 정책번호", f"{collection['duplicate_policy_number_count']}건"],
            ["수집 방식", "pageSize=10 기본 수집 후 실패 페이지는 pageSize=1로 복구"],
        ]
        table(ax, 0.06, 0.80, rows, [0.23, 0.66], row_h=0.055, fontsize=9.2)
        paragraph(ax, 0.07, 0.39, "구현 관점에서는 API 키를 스크립트에 직접 박아 넣지 않고 환경변수로 받는 방식으로 다뤘다. 원본 API 응답은 raw 영역에 보존하고, 검색과 분석에 편한 catalog/jsonl을 별도로 생성했다.", width=95, fontsize=10.5)
        code_box = patches.FancyBboxPatch((0.07, 0.13), 0.86, 0.18, boxstyle="round,pad=0.012", facecolor="#111827", edgecolor="#111827")
        ax.add_patch(code_box)
        code = 'GET /go/ythip/getPlcy?apiKeyNm=...&pageNum=1&pageSize=10&rtnType=json\n→ raw page JSON 보존\n→ 실패 페이지는 pageSize=1로 재요청\n→ JSONL + CSV catalog 생성'
        ax.text(0.09, 0.275, code, fontsize=10, color="#f9fafb", family="Malgun Gothic", va="top")
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Raw categories before normalization
        fig, ax = make_fig()
        add_title(ax, "3. 원본 분류의 문제", "API 원본 분류명은 중복 표기, 옛 명칭, 복합 분류가 섞여 있어 그대로 쓰기 어렵다")
        raw_top = dict(collection["top_lclsfNm"][:8])
        small_bar(ax, raw_top, "원본 대분류 Top 8", 0.08, 0.18, 0.38, 0.55, "#60a5fa")
        small_bar(ax, dict(list(norm["primary_category_counts"].items())[:6]), "정규화 후 대분류", 0.56, 0.18, 0.34, 0.55, "#34d399")
        paragraph(ax, 0.08, 0.84, "예를 들어 원본에는 '복지문화', '금융･복지･문화', '교육', '교육･직업훈련', '참여권리', '참여･기반'처럼 비슷한 의미가 다른 문자열로 들어온다. 그래서 이후 선정과 평가가 흔들리지 않도록 정규화 규칙을 먼저 만들었다.", width=105, fontsize=10.5)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Normalization
        fig, ax = make_fig()
        add_title(ax, "4. 분류명 정규화 기준", "정책 분야를 MVP에서 균형 있게 다루기 위한 공통 분류 체계")
        rows = [
            ["정규화 대분류", "건수", "설명"],
            ["일자리", norm["primary_category_counts"].get("일자리", 0), "취업, 창업, 재직자"],
            ["복지문화", norm["primary_category_counts"].get("복지문화", 0), "건강, 문화활동, 예술인지원, 금융/복지 성격"],
            ["참여권리", norm["primary_category_counts"].get("참여권리", 0), "청년참여, 권익보호, 정책인프라"],
            ["교육", norm["primary_category_counts"].get("교육", 0), "교육비지원, 미래역량강화, 온라인교육"],
            ["주거", norm["primary_category_counts"].get("주거", 0), "전월세, 주택, 거주지, 기숙사"],
            ["UNKNOWN", norm["primary_category_counts"].get("UNKNOWN", 0), "분류값이 비어 있거나 매핑 불가"],
        ]
        table(ax, 0.06, 0.80, rows, [0.18, 0.11, 0.60], row_h=0.058, fontsize=9)
        paragraph(ax, 0.07, 0.35, "정규화 결과 전체 2,728건 중 UNKNOWN은 1건뿐이었다. 이 기준 덕분에 MVP 400개를 카테고리별로 의도적으로 뽑을 수 있고, 나중에 평가 질문도 분야별로 균형을 볼 수 있다.", width=105, fontsize=10.5)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # MVP selection
        fig, ax = make_fig()
        add_title(ax, "5. MVP 400개 선정", "전체 2,728건을 모두 쓰기 전, 실험 가능한 고품질 corpus 400개를 먼저 구성")
        small_bar(ax, mvp["category_counts"], "MVP 400 대분류 분포", 0.07, 0.22, 0.38, 0.50, "#10b981")
        small_bar(ax, dict(doc_middle.most_common(10)), "MVP 중분류 Top 10", 0.55, 0.22, 0.36, 0.50, "#8b5cf6")
        paragraph(ax, 0.07, 0.84, "선정 규칙은 category_target + primary_mclsf_balance + document_quality_score_desc였다. 즉 분야별 목표 수를 먼저 잡고, 그 안에서 중분류 균형과 문서 품질점수를 반영해 400건을 뽑았다.", width=105, fontsize=10.5)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Quality scoring
        fig, ax = make_fig()
        add_title(ax, "6. 문서 품질점수의 의미", "검색 문서로 쓸 만한 정책인지 판단하기 위해 핵심 필드의 충실도를 점수화")
        fields = [
            ("정책명", "title_ok"),
            ("정책 설명", "description_rich"),
            ("지원 내용", "support_content_rich"),
            ("지원 대상/자격", "target_condition_present"),
            ("신청 기간", "application_period_present"),
            ("신청 방법", "application_method_present"),
            ("제출 서류", "submission_docs_present"),
            ("URL", "url_present"),
            ("키워드", "keyword_present"),
            ("정규화 상태", "normalization_ok"),
        ]
        rows = [["반영 필드", "품질 신호", "왜 중요한가"]]
        reasons = {
            "정책명": "문서를 식별하고 검색 결과를 설명할 때 필요",
            "정책 설명": "문서의 목적과 상황 검색 단서를 제공",
            "지원 내용": "사용자가 묻는 혜택의 핵심 답변",
            "지원 대상/자격": "eligibility 질문의 근거",
            "신청 기간": "period 질문의 근거",
            "신청 방법": "application_method 질문의 근거",
            "제출 서류": "documents 질문의 근거",
            "URL": "검수와 실제 안내 링크",
            "키워드": "분류와 보조 검색 단서",
            "정규화 상태": "카테고리 균형과 후속 분석 안정성",
        }
        for name, signal in fields:
            rows.append([name, signal, reasons[name]])
        table(ax, 0.04, 0.83, rows, [0.16, 0.25, 0.53], row_h=0.052, fontsize=8.2)
        ax.text(0.06, 0.17, f"선정된 400개 품질점수: 최소 {mvp['quality_score_summary']['min']} / 평균 {mvp['quality_score_summary']['avg']} / 최대 {mvp['quality_score_summary']['max']}", fontsize=11, fontweight="bold", color="#111827")
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Document schema
        fig, ax = make_fig()
        add_title(ax, "7. RAG용 documents.jsonl 스키마", "정책 1건을 검색 가능한 document 1건으로 변환")
        x0, y0 = 0.07, 0.72
        schema_boxes = [
            ("document_id", "검색/평가에서 쓰는 고유 ID"),
            ("title", "정책명"),
            ("source / url", "출처와 원문 링크"),
            ("content", "사람이 읽는 정책 본문"),
            ("retrieval_text", "chunking과 검색에 사용할 본문"),
            ("metadata", "분류, 기관, 기간, 나이, 지역 등"),
            ("raw_record", "API 원본 60개 필드 보존"),
        ]
        for i, (k, desc) in enumerate(schema_boxes):
            col = i % 2
            row = i // 2
            card(ax, x0 + col * 0.46, y0 - row * 0.15, 0.39, 0.11, k, desc, "#0ea5e9" if col == 0 else "#22c55e")
        paragraph(ax, 0.07, 0.16, "핵심은 원본 API 값을 잃지 않는 것이다. 검색용 텍스트를 만들더라도 raw_record를 그대로 보존해 나중에 필드 누락, 오분류, 답변 근거 문제를 다시 추적할 수 있게 했다.", width=105, fontsize=10.5)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Document example
        fig, ax = make_fig()
        add_title(ax, "8. 문서 예시", "documents.jsonl의 실제 정책 1건이 어떤 모양인지")
        rows = [
            ["필드", "예시"],
            ["document_id", policy_ex["document_id"]],
            ["title", policy_ex["title"]],
            ["category", f"{policy_ex['metadata']['primary_category']} / {policy_ex['metadata']['primary_mclsf']}"],
            ["url", policy_ex.get("url", "")],
        ]
        table(ax, 0.05, 0.82, rows, [0.18, 0.75], row_h=0.060, fontsize=8.8)
        draw_text_box(
            ax,
            0.05,
            0.45,
            0.90,
            0.28,
            "retrieval_text 앞부분",
            policy_ex.get("retrieval_text", "")[:520].replace("\n", " / "),
            fontsize=8.4,
            accent="#0ea5e9",
            body_width=128,
        )
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Length distribution
        fig, ax = make_fig()
        add_title(ax, "9. 문서 길이 분포", "chunking 전 corpus의 retrieval_text 길이 확인")
        sub = fig.add_axes([0.08, 0.22, 0.52, 0.50])
        sub.hist(doc_lengths, bins=25, color="#60a5fa", edgecolor="white")
        sub.set_title("retrieval_text 길이 histogram", fontsize=11)
        sub.set_xlabel("문자 수")
        sub.set_ylabel("문서 수")
        sub.grid(alpha=0.2)
        for spine in sub.spines.values():
            spine.set_visible(False)
        rows = [
            ["지표", "값"],
            ["문서 수", len(doc_lengths)],
            ["최소", min(doc_lengths)],
            ["평균", f"{sum(doc_lengths)/len(doc_lengths):.1f}"],
            ["중앙값", sorted(doc_lengths)[len(doc_lengths)//2]],
            ["최대", max(doc_lengths)],
        ]
        table(ax, 0.68, 0.70, rows, [0.13, 0.13], row_h=0.052, fontsize=9)
        paragraph(ax, 0.66, 0.34, "길이 분포를 확인한 이유는 chunk size 실험의 출발점을 잡기 위해서다. 너무 짧은 문서와 긴 문서의 비율을 알아야 300/500/700자 같은 chunk size 비교가 의미 있어진다.", width=36, fontsize=9.5)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Evaluation question history
        fig, ax = make_fig()
        add_title(ax, "10. 평가 질문 생성과 검수 흐름", "처음 만든 질문을 그대로 쓰지 않고, 사람 검수와 재수정을 거쳐 품질을 올림")
        draw_pipeline(
            ax,
            [
                ("V1 생성", "100", "#fee2e2"),
                ("V1 검수", "전부 0", "#fecaca"),
                ("V2 refined", "100", "#dbeafe"),
                ("판정", "1=14", "#dcfce7"),
                ("2번 수리", "33", "#fef3c7"),
                ("수리 통과", "25+8", "#bbf7d0"),
                ("최종", "47", "#e9d5ff"),
            ],
        )
        y = 0.33
        y = paragraph(ax, 0.07, y, "V1은 정책명이 과도하게 드러나거나 너무 포괄적인 질문이 많아 평가 질문으로 부적합했다. 그래서 V2 refined에서는 질문 유형을 나누고, query/source_title/reference_answer의 삼각관계를 중심으로 재검수했다.", width=105, fontsize=10.5)
        paragraph(ax, 0.07, y - 0.015, "최종적으로 원본 V2에서 바로 통과한 14개, label 2를 고친 뒤 통과한 25개, qrels/질문/정답문서 조정으로 복구한 8개를 합쳐 47개 평가 질문을 만들었다.", width=105, fontsize=10.5)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Question types
        fig, ax = make_fig()
        add_title(ax, "11. 최종 평가 질문 47개의 구성", "질문 유형과 정책 분야 분포")
        small_bar(ax, dict(eval_types), "질문 유형 분포", 0.07, 0.20, 0.40, 0.52, "#a78bfa")
        small_bar(ax, dict(eval_categories), "평가 질문 대분류 분포", 0.57, 0.20, 0.32, 0.52, "#f59e0b")
        paragraph(ax, 0.07, 0.84, "질문은 정책 발견형(specific_policy_search)과 정보 조회형(application_method, support_content, eligibility, period, required_documents 등)으로 나뉜다. MVP에서는 완벽한 균형보다 먼저 채점 가능한 고품질 질문 확보를 우선했다.", width=105, fontsize=10.5)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Eval/Qrels concept
        fig, ax = make_fig()
        add_title(ax, "12. eval_questions와 qrels의 차이", "문제지와 채점용 답안지는 역할이 다르다")
        card(ax, 0.07, 0.75, 0.38, 0.28, "eval_questions.csv/jsonl", "질문 문장, 정답 정책명, 참고 답변(reference_answer), 대표 정답 문서 ID를 담는다. 사람이 검토하기 좋은 문제지 역할이다.", "#2563eb")
        card(ax, 0.56, 0.75, 0.34, 0.28, "qrels.csv/jsonl", "query_id와 document_id의 연결만 담는다. 검색기가 어떤 문서를 찾으면 정답으로 채점할지 알려주는 답안지다.", "#f97316")
        ax.annotate("", xy=(0.53, 0.59), xytext=(0.46, 0.59), arrowprops=dict(arrowstyle="->", lw=2, color="#6b7280"))
        rows = [
            ["구분", "질문/답 문장", "정답 문서 ID", "사용자"],
            ["eval_questions", "있음", "있음", "사람 검수, 평가 실행 입력"],
            ["qrels", "없음", "있음", "채점 코드"],
        ]
        table(ax, 0.17, 0.35, rows, [0.20, 0.18, 0.18, 0.26], row_h=0.06, fontsize=9.5)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Concrete qrels example
        fig, ax = make_fig()
        add_title(ax, "13. 실제 질문과 답안지 예시", "질문에 대한 답과 qrels가 가리키는 문서는 서로 다른 층위")
        rows = [
            ["항목", "값"],
            ["query_id", eval_ex["query_id"]],
            ["질문", eval_ex["query"]],
            ["정답 정책명", eval_ex["source_title"]],
            ["qrels document_id", ", ".join(r["document_id"] for r in qrels_ex)],
        ]
        table(ax, 0.04, 0.82, rows, [0.18, 0.76], row_h=0.070, fontsize=8.7)
        draw_text_box(
            ax,
            0.04,
            0.45,
            0.92,
            0.16,
            "reference_answer",
            eval_ex["reference_answer"][:430].replace("\n", " / "),
            fontsize=8.3,
            accent="#7c3aed",
            body_width=134,
        )
        paragraph(ax, 0.06, 0.22, "실제 답변은 신청 방법 문장이지만, qrels는 그 문장을 저장하지 않고 답변을 포함한 정답 문서 ID만 가리킨다.", width=105, fontsize=10.5)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Multi qrels
        fig, ax = make_fig()
        add_title(ax, "14. multi-qrels가 필요한 이유", "정답 문서가 하나만 있다고 가정하면 정상 검색 결과가 오답 처리될 수 있다")
        y = 0.78
        for qid in multi_q:
            q_docs = [r["document_id"] for r in qrels_rows if r["query_id"] == qid]
            ev = next(r for r in eval_rows if r["query_id"] == qid)
            draw_text_box(
                ax,
                0.06,
                y,
                0.88,
                0.13,
                f"{qid} | 정답 문서 {len(q_docs)}개",
                ev["resolution_note"],
                fontsize=8.6,
                accent="#f97316",
                body_width=124,
            )
            y -= 0.16
        paragraph(ax, 0.06, 0.26, "예: 서울청년문화패스와 서울청년문화패스 지원처럼 같은 질문에 답할 수 있는 형제 문서가 corpus에 같이 있으면 두 문서 모두 relevance=1로 넣어야 한다. 그렇지 않으면 검색기가 맞는 형제 문서를 찾아도 qrels에 없어서 오답이 된다.", width=105, fontsize=10.2)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Resolved label2
        fig, ax = make_fig()
        add_title(ax, "15. 보류 8건을 어떻게 복구했는가", "질문 자체 문제와 답안지 문제를 분리해서 처리")
        rows = [["query_id", "조치", "최종 처리"]]
        for r in resolved:
            rows.append([r["repair_query_id"], r["resolution_action"], r["resolution_note"]])
        table(ax, 0.035, 0.82, rows, [0.18, 0.20, 0.57], row_h=0.072, fontsize=7.4)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # File tree
        fig, ax = make_fig()
        add_title(ax, "16. 최종 산출물 위치", "3번 담당자에게 넘길 때 가장 중요한 파일")
        rows = [
            ["역할", "파일"],
            ["MVP 검색 corpus", r"C:\말똥가리\data\processed\ontong_youth_mvp_400_documents\documents.jsonl"],
            ["문서 스키마 설명", r"C:\말똥가리\data\processed\ontong_youth_mvp_400_documents\DOCUMENT_SCHEMA.md"],
            ["평가 질문", r"C:\말똥가리\data\processed\ontong_youth_eval_questions_v2_final_47\eval_questions.jsonl"],
            ["평가 질문 CSV", r"C:\말똥가리\data\processed\ontong_youth_eval_questions_v2_final_47\eval_questions.csv"],
            ["문서 단위 qrels", r"C:\말똥가리\data\processed\ontong_youth_eval_questions_v2_final_47\qrels.jsonl"],
            ["qrels CSV", r"C:\말똥가리\data\processed\ontong_youth_eval_questions_v2_final_47\qrels.csv"],
        ]
        table(ax, 0.04, 0.82, rows, [0.20, 0.74], row_h=0.065, fontsize=8)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Handoff
        fig, ax = make_fig()
        add_title(ax, "17. 다음 담당자에게 넘기는 작업", "2번은 문서와 평가셋 준비까지, 3번은 chunking 이후 평가 준비")
        card(ax, 0.07, 0.78, 0.36, 0.26, "3번이 입력으로 받을 것", "documents.jsonl, eval_questions.jsonl, qrels.jsonl. qrels는 document_id 기준 정답표이며 아직 chunk_id 기준이 아니다.", "#2563eb")
        card(ax, 0.55, 0.78, 0.36, 0.26, "3번이 생성할 것", "chunks.jsonl, chunk_id, document_id→chunk_id 매핑, ground_truth_chunk_ids. 이후 embedding과 Chroma 적재가 가능해진다.", "#16a34a")
        paragraph(ax, 0.07, 0.41, "중요: chunk size나 overlap이 바뀌면 chunk_id와 ground_truth_chunk_ids도 바뀐다. 따라서 지금 만든 qrels(document_id 기준)는 고정 기준으로 두고, chunking 실험마다 정답 chunk를 다시 매핑해야 한다.", width=105, fontsize=11)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Not done and risks
        fig, ax = make_fig()
        add_title(ax, "18. 아직 하지 않은 일과 주의점", "이번 보고서는 2번 단계의 산출물 설명서")
        rows = [
            ["항목", "상태", "이유"],
            ["Chunking", "미수행", "3번 담당 범위"],
            ["Embedding", "미수행", "chunk가 확정된 뒤 수행"],
            ["Chroma 적재", "미수행", "embedding 이후 수행"],
            ["ground_truth_chunk_ids", "미생성", "chunking 설정마다 달라짐"],
            ["전체 2,728건 실서비스 corpus", "미확정", "MVP 400으로 먼저 실험"],
            ["상황 설명형 질문 보강", "권장", "현재 47개는 known-item 성격이 일부 존재"],
        ]
        table(ax, 0.05, 0.80, rows, [0.22, 0.16, 0.55], row_h=0.064, fontsize=8.7)
        paragraph(ax, 0.06, 0.31, "현재 47개는 평가를 시작하기에는 충분하지만, 더 안정적인 MVP 평가를 위해서는 상황 설명형 질문을 추가로 만들어 50~100개 수준까지 확장하는 것이 좋다.", width=105, fontsize=10.5)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Appendix scripts
        fig, ax = make_fig()
        add_title(ax, "19. 구현 스크립트 요약", "주요 변환은 재실행 가능한 스크립트로 남겨 두었다")
        script_rows = [
            ["스크립트", "역할"],
            ["normalize_ontong_youth_catalog.py", "원본 catalog/raw JSONL을 정규화 분류 체계로 변환"],
            ["select_ontong_youth_mvp_400.py", "품질점수와 카테고리 목표 기반으로 MVP 400개 선정"],
            ["build_ontong_youth_mvp_documents.py", "RAG용 documents.jsonl과 schema 문서 생성"],
            ["generate_ontong_youth_eval_question_candidates_v2_refined.py", "V2 refined 평가 질문 후보 생성"],
            ["repair_ontong_youth_eval_label2.py", "보류 33건을 수정 질문 후보로 변환"],
            ["build_ontong_youth_eval_final_47.py", "검수/수정 결과를 최종 eval_questions와 qrels로 조립"],
            ["build_ontong_youth_project_pdf_report.py", "현재 PDF 보고서 생성"],
        ]
        table(ax, 0.04, 0.82, script_rows, [0.35, 0.58], row_h=0.062, fontsize=8.2)
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1

        # Closing
        fig, ax = make_fig()
        add_title(ax, "20. 결론", "2번 담당 산출물은 MVP 기준으로 인계 가능한 상태")
        paragraph(ax, 0.08, 0.78, "온통청년 API 원본 2,728건에서 출발해 정규화된 정책 catalog를 만들고, 그중 품질과 분야 균형을 고려해 MVP 400개 문서를 선정했다. 이후 RAG용 documents.jsonl 스키마를 만들고, 검색 평가를 위한 질문 47개와 qrels 50행을 구축했다.", width=100, fontsize=12, line_gap=0.04)
        paragraph(ax, 0.08, 0.55, "따라서 다음 단계의 핵심은 documents.jsonl을 chunking하고, qrels의 정답 document_id를 기준으로 각 chunking 결과에 맞는 ground_truth_chunk_ids를 생성하는 것이다. 그 다음 embedding, Chroma 적재, retrieval 평가로 이어진다.", width=100, fontsize=12, line_gap=0.04)
        card(ax, 0.18, 0.34, 0.28, 0.15, "2번 완료", "수집·정규화·MVP corpus·문서 스키마·평가셋/qrels", "#16a34a")
        card(ax, 0.55, 0.34, 0.28, 0.15, "다음 단계", "Chunking·정답 chunk 매핑·Embedding·Chroma·평가", "#f97316")
        add_footer(ax, page)
        pdf.savefig(fig)
        plt.close(fig)

    summary = {
        "created_at": datetime.now(KST).isoformat(),
        "pdf": str(OUT_PDF),
        "page_count": page,
        "input_counts": {
            "raw_policies": collection["final_record_count"],
            "normalized_policies": norm["quality"]["normalized_catalog_rows"],
            "mvp_documents": len(docs),
            "eval_questions": len(eval_rows),
            "qrels_rows": len(qrels_rows),
            "multi_qrels_queries": len(multi_q),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build_report()
