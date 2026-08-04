"""검색 평가 결과를 보여주는 정적 HTML 대시보드를 생성한다."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from rag_utils import OUTPUT_DIR, PROJECT_ROOT


UI_DIR = PROJECT_ROOT / "ui"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_summary() -> list[dict[str, str]]:
    summary_path = OUTPUT_DIR / "eval_summary_all.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            "outputs/eval_summary_all.csv가 없습니다. 먼저 검색 평가를 실행하세요."
        )
    return read_csv_rows(summary_path)


def load_retrieval_groups() -> list[dict]:
    groups: list[dict] = []
    pattern = re.compile(r"retrieval_results_chunk_(\d+)_top_(\d+)\.csv")

    for path in sorted(OUTPUT_DIR.glob("retrieval_results_chunk_*_top_*.csv")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue

        chunk_size, top_k = match.groups()
        rows = read_csv_rows(path)
        by_question: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_question[row["question_id"]].append(row)

        questions = []
        for question_id, hits in by_question.items():
            hits.sort(key=lambda item: int(item["rank"]))
            first = hits[0]
            answer_found = any(hit["answer_in_chunk"] == "O" for hit in hits)
            questions.append(
                {
                    "question_id": question_id,
                    "question": first["question"],
                    "answer": first["answer"],
                    "expected_doc_id": first["expected_doc_id"],
                    "answer_found": answer_found,
                    "hits": hits,
                }
            )

        questions.sort(key=lambda item: item["question_id"])
        groups.append(
            {
                "key": f"chunk_{chunk_size}_top_{top_k}",
                "chunk_size": int(chunk_size),
                "top_k": int(top_k),
                "questions": questions,
            }
        )

    return groups


def build_html(summary: list[dict[str, str]], retrieval_groups: list[dict]) -> str:
    payload = {
        "summary": summary,
        "retrievalGroups": retrieval_groups,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Retrieval Dashboard</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #64748b;
      --line: #d9dee8;
      --accent: #0f766e;
      --accent-soft: #d9f3ee;
      --warn: #b45309;
      --warn-soft: #fff2d7;
      --bad: #b42318;
      --bad-soft: #ffe4e0;
      --ok: #146c43;
      --ok-soft: #ddf7e9;
      --blue: #2756a3;
      --blue-soft: #e4ecfb;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", "Malgun Gothic", Arial, sans-serif;
      line-height: 1.5;
    }}

    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 20px 28px 18px;
      position: sticky;
      top: 0;
      z-index: 10;
    }}

    .topbar {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 20px;
      max-width: 1320px;
      margin: 0 auto;
    }}

    h1 {{
      font-size: 24px;
      margin: 0 0 4px;
      letter-spacing: 0;
    }}

    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}

    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 22px 28px 42px;
    }}

    .tabs {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }}

    .tab {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      height: 36px;
      padding: 0 14px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
    }}

    .tab.active {{
      border-color: var(--accent);
      background: var(--accent-soft);
      color: #064e45;
    }}

    .view {{
      display: none;
    }}

    .view.active {{
      display: block;
    }}

    .grid {{
      display: grid;
      gap: 14px;
    }}

    .summary-grid {{
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      margin-bottom: 16px;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}

    .metric-label {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}

    .metric-value {{
      font-size: 28px;
      font-weight: 750;
    }}

    .metric-sub {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }}

    .comparison {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 14px;
      align-items: start;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}

    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}

    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      background: #fbfcfe;
    }}

    .bar-row {{
      display: grid;
      grid-template-columns: 120px 1fr 64px;
      gap: 10px;
      align-items: center;
      margin: 10px 0;
      font-size: 13px;
    }}

    .bar-track {{
      height: 12px;
      background: #e7ebf2;
      border-radius: 999px;
      overflow: hidden;
    }}

    .bar-fill {{
      height: 100%;
      background: var(--accent);
      border-radius: 999px;
    }}

    .controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }}

    label {{
      font-size: 13px;
      color: var(--muted);
      font-weight: 700;
    }}

    select, input {{
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--ink);
      min-width: 160px;
    }}

    input {{
      min-width: min(420px, 100%);
      flex: 1;
    }}

    .two-col {{
      display: grid;
      grid-template-columns: 390px 1fr;
      gap: 14px;
      min-height: 620px;
    }}

    .question-list {{
      max-height: 720px;
      overflow: auto;
      padding: 0;
    }}

    .question-button {{
      width: 100%;
      border: 0;
      border-bottom: 1px solid var(--line);
      background: #fff;
      padding: 12px;
      text-align: left;
      cursor: pointer;
      color: var(--ink);
    }}

    .question-button:hover,
    .question-button.active {{
      background: #eef7f6;
    }}

    .question-text {{
      font-weight: 700;
      font-size: 14px;
      margin-bottom: 6px;
    }}

    .status {{
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}

    .status.ok {{
      color: var(--ok);
      background: var(--ok-soft);
    }}

    .status.bad {{
      color: var(--bad);
      background: var(--bad-soft);
    }}

    .hit {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-top: 12px;
      background: #fff;
    }}

    .hit-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
    }}

    .rank {{
      font-weight: 800;
      color: var(--blue);
      background: var(--blue-soft);
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
    }}

    .score {{
      color: var(--muted);
      font-size: 13px;
    }}

    .chunk-text {{
      white-space: pre-wrap;
      font-size: 14px;
    }}

    mark {{
      background: #fff0a6;
      padding: 0 2px;
      border-radius: 2px;
    }}

    .pipeline {{
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}

    .step {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 12px;
      min-height: 112px;
    }}

    .step-num {{
      width: 28px;
      height: 28px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #064e45;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      margin-bottom: 8px;
    }}

    .step-title {{
      font-weight: 800;
      margin-bottom: 4px;
    }}

    .step-desc {{
      color: var(--muted);
      font-size: 13px;
    }}

    .note {{
      border-left: 4px solid var(--warn);
      background: var(--warn-soft);
      padding: 12px 14px;
      border-radius: 6px;
      color: #713f12;
      margin-top: 14px;
    }}

    @media (max-width: 980px) {{
      .summary-grid,
      .comparison,
      .two-col,
      .pipeline {{
        grid-template-columns: 1fr;
      }}

      header {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <h1>RAG Retrieval Dashboard</h1>
        <p class="subtitle">AIHub 기계독해 데이터로 context 검색, 청킹, Top-k 차이를 확인하는 화면</p>
      </div>
    </div>
  </header>

  <main>
    <div class="tabs" role="tablist">
      <button class="tab active" data-view="overview">실험 비교</button>
      <button class="tab" data-view="questions">질문별 검색</button>
      <button class="tab" data-view="pipeline">파이프라인</button>
    </div>

    <section id="overview" class="view active">
      <div class="grid summary-grid" id="metricCards"></div>

      <div class="comparison">
        <div class="panel">
          <h2>실험 결과 표</h2>
          <table>
            <thead>
              <tr>
                <th>임베딩 모델</th>
                <th>청크</th>
                <th>Top-k</th>
                <th>질문</th>
                <th>정답 포함</th>
                <th>Recall</th>
              </tr>
            </thead>
            <tbody id="summaryTable"></tbody>
          </table>
        </div>

        <div class="panel">
          <h2>Recall 비교</h2>
          <div id="recallBars"></div>
          <div class="note">Recall은 전체 질문 중 검색된 Top-k 청크 안에 정답 문자열이 들어온 비율입니다.</div>
        </div>
      </div>
    </section>

    <section id="questions" class="view">
      <div class="controls panel">
        <label for="experimentSelect">실험 조건</label>
        <select id="experimentSelect"></select>
        <label for="statusSelect">상태</label>
        <select id="statusSelect">
          <option value="all">전체</option>
          <option value="found">정답 포함</option>
          <option value="missed">정답 없음</option>
        </select>
        <input id="searchInput" type="search" placeholder="질문 또는 정답 검색" />
      </div>

      <div class="two-col">
        <div class="panel question-list" id="questionList"></div>
        <div class="panel" id="questionDetail"></div>
      </div>
    </section>

    <section id="pipeline" class="view">
      <div class="panel">
        <h2>우리가 만든 Retrieval-only RAG</h2>
        <div class="pipeline">
          <div class="step">
            <div class="step-num">1</div>
            <div class="step-title">데이터 분리</div>
            <div class="step-desc">context는 검색 문서, question은 입력, answer는 평가 기준으로 분리합니다.</div>
          </div>
          <div class="step">
            <div class="step-num">2</div>
            <div class="step-title">청킹</div>
            <div class="step-desc">긴 context를 300자 또는 700자 조각으로 나눕니다.</div>
          </div>
          <div class="step">
            <div class="step-num">3</div>
            <div class="step-title">임베딩</div>
            <div class="step-desc">문서 청크를 E5 모델로 숫자 벡터로 바꿉니다.</div>
          </div>
          <div class="step">
            <div class="step-num">4</div>
            <div class="step-title">벡터 검색</div>
            <div class="step-desc">질문 벡터와 가까운 청크를 FAISS에서 Top-k로 가져옵니다.</div>
          </div>
          <div class="step">
            <div class="step-num">5</div>
            <div class="step-title">평가</div>
            <div class="step-desc">검색된 청크 안에 answer가 들어있는지 확인합니다.</div>
          </div>
        </div>
        <div class="note">이 화면은 아직 LLM 답변 생성을 하지 않습니다. 검색 단계가 잘 되는지 먼저 분리해서 보는 것이 목적입니다.</div>
      </div>
    </section>
  </main>

  <script id="dashboard-data" type="application/json">{payload_json}</script>
  <script>
    const data = JSON.parse(document.getElementById("dashboard-data").textContent);
    const state = {{
      groupKey: data.retrievalGroups[0]?.key || "",
      selectedQuestionId: null,
      status: "all",
      search: ""
    }};

    const pct = value => `${{(Number(value) * 100).toFixed(2)}}%`;
    const escapeHtml = value => String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

    function highlight(text, answer) {{
      const source = escapeHtml(text);
      if (!answer) return source;
      const safeAnswer = escapeHtml(answer);
      return source.replaceAll(safeAnswer, `<mark>${{safeAnswer}}</mark>`);
    }}

    function renderOverview() {{
      const best = [...data.summary].sort((a, b) => Number(b.recall) - Number(a.recall))[0];
      const totalQuestions = best?.question_count || "0";
      const bestRecall = best ? pct(best.recall) : "0%";
      const modelName = best?.model_name || "-";
      const experimentCount = data.summary.length;

      document.getElementById("metricCards").innerHTML = `
        <div class="panel"><div class="metric-label">실험 조건 수</div><div class="metric-value">${{experimentCount}}</div><div class="metric-sub">chunk/top-k 조합</div></div>
        <div class="panel"><div class="metric-label">평가 질문 수</div><div class="metric-value">${{totalQuestions}}</div><div class="metric-sub">AIHub question 사용</div></div>
        <div class="panel"><div class="metric-label">최고 Recall</div><div class="metric-value">${{bestRecall}}</div><div class="metric-sub">Top-k 안 정답 포함률</div></div>
        <div class="panel"><div class="metric-label">임베딩 모델</div><div class="metric-value" style="font-size:18px">${{escapeHtml(modelName)}}</div><div class="metric-sub">multilingual E5</div></div>
      `;

      document.getElementById("summaryTable").innerHTML = data.summary.map(row => `
        <tr>
          <td>${{escapeHtml(row.model_name)}}</td>
          <td>${{row.chunk_size}}자</td>
          <td>${{row.top_k}}</td>
          <td>${{row.question_count}}</td>
          <td>${{row.answer_found_count}}</td>
          <td><strong>${{pct(row.recall)}}</strong></td>
        </tr>
      `).join("");

      document.getElementById("recallBars").innerHTML = data.summary.map(row => {{
        const label = `${{row.chunk_size}}자 / Top-${{row.top_k}}`;
        return `
          <div class="bar-row">
            <div>${{label}}</div>
            <div class="bar-track"><div class="bar-fill" style="width: ${{Number(row.recall) * 100}}%"></div></div>
            <div><strong>${{pct(row.recall)}}</strong></div>
          </div>
        `;
      }}).join("");
    }}

    function currentGroup() {{
      return data.retrievalGroups.find(group => group.key === state.groupKey) || data.retrievalGroups[0];
    }}

    function filteredQuestions() {{
      const group = currentGroup();
      const search = state.search.trim().toLowerCase();
      return group.questions.filter(item => {{
        if (state.status === "found" && !item.answer_found) return false;
        if (state.status === "missed" && item.answer_found) return false;
        if (!search) return true;
        return item.question.toLowerCase().includes(search) || item.answer.toLowerCase().includes(search);
      }});
    }}

    function renderExperimentSelect() {{
      const select = document.getElementById("experimentSelect");
      select.innerHTML = data.retrievalGroups.map(group => `
        <option value="${{group.key}}">청크 ${{group.chunk_size}}자 / Top-${{group.top_k}}</option>
      `).join("");
      select.value = state.groupKey;
    }}

    function renderQuestionList() {{
      const list = filteredQuestions();
      if (!state.selectedQuestionId || !list.some(item => item.question_id === state.selectedQuestionId)) {{
        state.selectedQuestionId = list[0]?.question_id || null;
      }}

      document.getElementById("questionList").innerHTML = list.map(item => `
        <button class="question-button ${{item.question_id === state.selectedQuestionId ? "active" : ""}}" data-question-id="${{escapeHtml(item.question_id)}}">
          <div class="question-text">${{escapeHtml(item.question)}}</div>
          <span class="status ${{item.answer_found ? "ok" : "bad"}}">${{item.answer_found ? "정답 포함" : "정답 없음"}}</span>
        </button>
      `).join("") || `<div style="padding:16px;color:var(--muted)">조건에 맞는 질문이 없습니다.</div>`;

      document.querySelectorAll(".question-button").forEach(button => {{
        button.addEventListener("click", () => {{
          state.selectedQuestionId = button.dataset.questionId;
          renderQuestions();
        }});
      }});
    }}

    function renderQuestionDetail() {{
      const group = currentGroup();
      const selected = group.questions.find(item => item.question_id === state.selectedQuestionId);
      const detail = document.getElementById("questionDetail");
      if (!selected) {{
        detail.innerHTML = `<h2>질문 없음</h2><p class="subtitle">왼쪽 목록에서 질문을 선택하세요.</p>`;
        return;
      }}

      detail.innerHTML = `
        <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start">
          <div>
            <h2 style="margin-top:0">${{escapeHtml(selected.question)}}</h2>
            <p class="subtitle">정답: <strong>${{escapeHtml(selected.answer)}}</strong> · 기대 문서: ${{escapeHtml(selected.expected_doc_id)}}</p>
          </div>
          <span class="status ${{selected.answer_found ? "ok" : "bad"}}">${{selected.answer_found ? "정답 포함" : "정답 없음"}}</span>
        </div>
        ${{selected.hits.map(hit => `
          <div class="hit">
            <div class="hit-top">
              <div>
                <span class="rank">Rank ${{hit.rank}}</span>
                <span class="score">score ${{Number(hit.score).toFixed(4)}} · ${{escapeHtml(hit.chunk_id)}}</span>
              </div>
              <span class="status ${{hit.answer_in_chunk === "O" ? "ok" : "bad"}}">${{hit.answer_in_chunk === "O" ? "정답 포함" : "정답 없음"}}</span>
            </div>
            <div class="chunk-text">${{highlight(hit.retrieved_text, selected.answer)}}</div>
          </div>
        `).join("")}}
      `;
    }}

    function renderQuestions() {{
      renderExperimentSelect();
      renderQuestionList();
      renderQuestionDetail();
    }}

    function setupEvents() {{
      document.querySelectorAll(".tab").forEach(tab => {{
        tab.addEventListener("click", () => {{
          document.querySelectorAll(".tab").forEach(item => item.classList.remove("active"));
          document.querySelectorAll(".view").forEach(item => item.classList.remove("active"));
          tab.classList.add("active");
          document.getElementById(tab.dataset.view).classList.add("active");
        }});
      }});

      document.getElementById("experimentSelect").addEventListener("change", event => {{
        state.groupKey = event.target.value;
        state.selectedQuestionId = null;
        renderQuestions();
      }});

      document.getElementById("statusSelect").addEventListener("change", event => {{
        state.status = event.target.value;
        state.selectedQuestionId = null;
        renderQuestions();
      }});

      document.getElementById("searchInput").addEventListener("input", event => {{
        state.search = event.target.value;
        state.selectedQuestionId = null;
        renderQuestions();
      }});
    }}

    renderOverview();
    renderQuestions();
    setupEvents();
  </script>
</body>
</html>
"""


def main() -> None:
    UI_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    retrieval_groups = load_retrieval_groups()
    html = build_html(summary, retrieval_groups)
    output_path = UI_DIR / "index.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"대시보드 생성: {output_path}")


if __name__ == "__main__":
    main()

