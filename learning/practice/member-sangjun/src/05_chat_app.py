"""ChatGPT처럼 질문을 입력하고 RAG 검색 결과를 확인하는 로컬 UI 서버."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from rag_utils import (
    OUTPUT_DIR,
    SAMPLE_DIR,
    VECTOR_DIR,
    answer_in_text,
    e5_query,
    read_csv,
    read_faiss_index,
    read_jsonl,
)


CHAT_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Chat</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #16202a;
      --muted: #637083;
      --line: #d8dee8;
      --user: #0f766e;
      --user-text: #ffffff;
      --assistant: #ffffff;
      --accent: #0f766e;
      --soft: #e0f3ef;
      --code: #f8fafc;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", "Malgun Gothic", Arial, sans-serif;
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }

    header {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 14px 20px;
    }

    .header-inner {
      max-width: 980px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }

    h1 {
      font-size: 20px;
      margin: 0;
      letter-spacing: 0;
    }

    .sub {
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
    }

    .settings {
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }

    select {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 0 8px;
      color: var(--ink);
    }

    main {
      overflow-y: auto;
      padding: 22px 20px;
    }

    .chat {
      max-width: 980px;
      margin: 0 auto;
      display: grid;
      gap: 16px;
    }

    .message {
      display: grid;
      gap: 8px;
    }

    .bubble {
      max-width: 86%;
      border-radius: 8px;
      padding: 14px 16px;
      line-height: 1.55;
      font-size: 15px;
      white-space: pre-wrap;
      border: 1px solid var(--line);
    }

    .message.user {
      justify-items: end;
    }

    .message.user .bubble {
      background: var(--user);
      color: var(--user-text);
      border-color: var(--user);
    }

    .message.assistant {
      justify-items: start;
    }

    .message.assistant .bubble {
      background: var(--assistant);
    }

    .sources {
      width: min(86%, 860px);
    }

    .source-panel {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }

    .source-panel > summary {
      cursor: pointer;
      padding: 10px 12px;
      font-weight: 700;
      color: var(--ink);
    }

    .source-list {
      border-top: 1px solid var(--line);
      padding: 12px;
      display: grid;
      gap: 10px;
      background: #fbfcfe;
    }

    .source-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
    }

    .source-card.cited {
      border-color: #f2c14e;
      background: #fffdf4;
    }

    .source-title {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 7px;
      font-weight: 800;
    }

    .source-score {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }

    .source-body {
      color: #243140;
      font-size: 14px;
      line-height: 1.55;
      white-space: pre-wrap;
    }

    .source-body.cited {
      background: #fffdf4;
    }

    .meta {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }

    .welcome {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }

    .welcome-title {
      font-size: 17px;
      font-weight: 800;
      margin-bottom: 8px;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }

    .examples {
      margin-top: 16px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }

    .examples-title {
      font-weight: 800;
      margin-bottom: 4px;
    }

    .examples-sub {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }

    .chip {
      border: 1px solid var(--line);
      background: var(--soft);
      color: #07564e;
      border-radius: 999px;
      padding: 7px 10px;
      cursor: pointer;
      font-size: 13px;
    }

    footer {
      background: var(--panel);
      border-top: 1px solid var(--line);
      padding: 14px 20px;
    }

    form {
      max-width: 980px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: end;
    }

    textarea {
      resize: none;
      min-height: 48px;
      max-height: 160px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 13px;
      font: inherit;
      line-height: 1.45;
      outline: none;
    }

    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14);
    }

    button[type="submit"] {
      height: 48px;
      min-width: 84px;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      font-weight: 800;
      cursor: pointer;
    }

    button[type="submit"]:disabled {
      opacity: 0.55;
      cursor: wait;
    }

    .hint {
      max-width: 980px;
      margin: 7px auto 0;
      color: var(--muted);
      font-size: 12px;
    }

    mark {
      background: #ffe66d;
      border-radius: 2px;
      padding: 1px 3px;
      box-shadow: inset 0 -2px 0 rgba(180, 83, 9, 0.22);
    }

    .cite-badge {
      display: inline-flex;
      align-items: center;
      height: 22px;
      padding: 0 8px;
      margin-left: 6px;
      border-radius: 999px;
      background: #fff2d7;
      color: #92400e;
      font-size: 12px;
      font-weight: 800;
    }

    .citation-note {
      border-left: 4px solid #f59e0b;
      background: #fffbeb;
      padding: 8px 10px;
      border-radius: 6px;
      margin-bottom: 10px;
      color: #713f12;
      font-size: 13px;
    }

    .source-empty {
      color: var(--muted);
      font-size: 13px;
      padding: 4px 0;
    }

    .eval-card {
      width: min(86%, 860px);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fff;
      display: grid;
      gap: 5px;
    }

    .eval-card.success {
      border-color: #9ad5b3;
      background: #f0fbf5;
    }

    .eval-card.failure {
      border-color: #f3b0aa;
      background: #fff5f4;
    }

    .eval-card.unknown {
      background: #f8fafc;
    }

    .eval-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 850;
    }

    .eval-badge {
      display: inline-flex;
      align-items: center;
      height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 850;
    }

    .eval-badge.success {
      background: var(--ok-soft);
      color: var(--ok);
    }

    .eval-badge.failure {
      background: var(--bad-soft);
      color: var(--bad);
    }

    .eval-badge.unknown {
      background: #e8edf5;
      color: #475569;
    }

    .eval-desc {
      color: var(--muted);
      font-size: 13px;
    }

    .answer-badge {
      display: inline-flex;
      align-items: center;
      height: 22px;
      padding: 0 8px;
      margin-left: 6px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 850;
    }

    .answer-badge.hit {
      color: var(--ok);
      background: var(--ok-soft);
    }

    .answer-badge.miss {
      color: var(--bad);
      background: var(--bad-soft);
    }

    @media (max-width: 720px) {
      .header-inner {
        align-items: flex-start;
        flex-direction: column;
      }

      form {
        grid-template-columns: 1fr;
      }

      button[type="submit"] {
        width: 100%;
      }

      .bubble,
      .sources {
        max-width: 100%;
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div>
        <h1>RAG Chat</h1>
        <div class="sub">질문을 입력하면 AIHub context에서 관련 청크를 검색합니다.</div>
      </div>
      <div class="settings">
        <label for="topK">Top-k</label>
        <select id="topK">
          <option value="3">3</option>
          <option value="5" selected>5</option>
        </select>
      </div>
    </div>
  </header>

  <main id="scrollArea">
    <div class="chat" id="chat">
      <div class="welcome">
        <div class="welcome-title">Retrieval-only RAG 채팅 화면</div>
        <div>아직 LLM을 붙이지 않았기 때문에 자연어 답변을 새로 생성하지는 않습니다. 대신 질문과 가장 가까운 문서 청크를 찾아서 근거와 함께 보여줍니다.</div>
        <div class="examples">
          <div class="examples-title">지금 바로 물어볼 수 있는 질문 예시</div>
          <div class="examples-sub">현재 벡터DB에 들어간 AIHub 샘플에서 정답 근거가 있는 질문입니다.</div>
          <div class="chips" id="exampleChips"></div>
        </div>
      </div>
    </div>
  </main>

  <footer>
    <form id="chatForm">
      <textarea id="questionInput" placeholder="질문을 입력하세요" rows="1"></textarea>
      <button id="sendButton" type="submit">보내기</button>
    </form>
    <div class="hint">Enter로 전송, Shift+Enter로 줄바꿈</div>
  </footer>

  <script>
    const chat = document.getElementById("chat");
    const form = document.getElementById("chatForm");
    const input = document.getElementById("questionInput");
    const button = document.getElementById("sendButton");
    const topK = document.getElementById("topK");
    const scrollArea = document.getElementById("scrollArea");

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function highlightCitedText(text, citedText) {
      const safeText = escapeHtml(text);
      if (!citedText) return safeText;

      const safeCitation = escapeHtml(citedText);
      const index = safeText.indexOf(safeCitation);
      if (index < 0) return safeText;

      return [
        safeText.slice(0, index),
        `<mark>${safeCitation}</mark>`,
        safeText.slice(index + safeCitation.length)
      ].join("");
    }

    function evaluationHtml(evaluation) {
      const timeText = evaluation?.response_time_text
        ? ` · 응답 시간 ${evaluation.response_time_text}`
        : "";

      if (!evaluation || !evaluation.has_reference) {
        return `
          <div class="eval-card unknown">
            <div class="eval-title"><span class="eval-badge unknown">평가 기준 없음</span><span>${timeText.replace(" · ", "")}</span></div>
            <div class="eval-desc">이 질문은 평가 질문 파일에 없는 질문이라 answer_in_chunk 기준으로 성공/실패를 자동 판별하지 않습니다.</div>
          </div>
        `;
      }

      const statusClass = evaluation.success ? "success" : "failure";
      const statusText = evaluation.success ? "검색 성공" : "검색 실패";
      const desc = evaluation.success
        ? `Top-k ${evaluation.top_k}개 중 정답 문자열이 포함된 근거 ${evaluation.answer_hit_count}개를 찾았습니다.`
        : `Top-k ${evaluation.top_k}개 안에 정답 문자열이 없습니다. 청킹, 임베딩, Top-k, 데이터 전처리를 확인해야 합니다.`;

      return `
        <div class="eval-card ${statusClass}">
          <div class="eval-title">
            <span class="eval-badge ${statusClass}">${statusText}</span>
            <span>answer_in_chunk 기준${timeText}</span>
          </div>
          <div class="eval-desc">${desc}</div>
        </div>
      `;
    }

    function addMessage(role, text, sources = [], evaluation = null) {
      const wrapper = document.createElement("div");
      wrapper.className = `message ${role}`;

      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;
      wrapper.appendChild(bubble);

      if (role === "assistant") {
        const evaluationBox = document.createElement("div");
        evaluationBox.innerHTML = evaluationHtml(evaluation);
        wrapper.appendChild(evaluationBox.firstElementChild);
      }

      if (sources.length) {
        const sourceBox = document.createElement("div");
        sourceBox.className = "sources";
        const citedCount = sources.filter(source => source.is_cited).length;
        sourceBox.innerHTML = sources.map((source, index) => `
          <div class="source-card ${source.is_cited ? 'cited' : ''}">
            <div class="source-title">
              <span>
                근거 ${index + 1}
                ${source.is_cited ? '<span class="cite-badge">인용</span>' : ''}
                ${source.has_reference ? `<span class="answer-badge ${source.answer_in_chunk ? 'hit' : 'miss'}">${source.answer_in_chunk ? '정답 포함' : '정답 없음'}</span>` : ''}
              </span>
              <span class="source-score">score ${Number(source.score).toFixed(4)}</span>
            </div>
            <div class="meta">doc_id: ${escapeHtml(source.doc_id)} · ${escapeHtml(source.chunk_id)}</div>
            <div class="source-body">${highlightCitedText(source.excerpt || source.cited_text || source.text, source.cited_text)}</div>
          </div>
        `).join("");
        sourceBox.innerHTML = `
          <details class="source-panel">
            <summary>근거 보기 · ${sources.length}개 검색됨${citedCount ? ` · 인용 ${citedCount}개` : ''}</summary>
            <div class="source-list">
              ${citedCount ? '<div class="citation-note">노란색 부분이 답변에 사용된 핵심 근거입니다. 전체 원문 대신 관련 발췌문만 보여줍니다.</div>' : '<div class="source-empty">검색된 근거의 핵심 발췌문입니다.</div>'}
              ${sourceBox.innerHTML}
            </div>
          </details>
        `;
        wrapper.appendChild(sourceBox);
      }

      chat.appendChild(wrapper);
      scrollArea.scrollTop = scrollArea.scrollHeight;
    }

    async function loadExamples() {
      try {
        const response = await fetch("/api/examples");
        const result = await response.json();
        const chips = document.getElementById("exampleChips");
        chips.innerHTML = result.examples.map(item => `
          <button class="chip" data-example="${escapeHtml(item.question)}">${escapeHtml(item.question)}</button>
        `).join("");
        bindExampleButtons();
      } catch (error) {
        document.getElementById("exampleChips").innerHTML = "<span class='sub'>예시 질문을 불러오지 못했습니다.</span>";
      }
    }

    async function ask(question) {
      addMessage("user", question);
      button.disabled = true;
      button.textContent = "검색 중";

      try {
        const response = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, top_k: Number(topK.value) })
        });

        if (!response.ok) {
          throw new Error(await response.text());
        }

        const result = await response.json();
        addMessage("assistant", result.answer, result.sources, result.evaluation);
      } catch (error) {
        addMessage("assistant", `오류가 발생했습니다.\\n${error.message}`);
      } finally {
        button.disabled = false;
        button.textContent = "보내기";
        input.focus();
      }
    }

    form.addEventListener("submit", event => {
      event.preventDefault();
      const question = input.value.trim();
      if (!question) return;
      input.value = "";
      input.style.height = "48px";
      ask(question);
    });

    input.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    input.addEventListener("input", () => {
      input.style.height = "48px";
      input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
    });

    function bindExampleButtons() {
      document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
          input.value = chip.dataset.example;
          form.requestSubmit();
        });
      });
    }

    loadExamples();
  </script>
</body>
</html>
"""


class SearchEngine:
    def __init__(self, index_dir: Path, model_name: str | None = None) -> None:
        self.index_dir = index_dir
        self.config = self._load_config()
        self.model_name = model_name or self.config.get("model_name", "intfloat/multilingual-e5-small")
        self.eval_questions = self._load_eval_questions()
        self.examples = self._load_examples()
        self.answer_by_question = {
            normalize_question(row["question"]): row["answer"]
            for row in self.eval_questions
            if row.get("question") and row.get("answer")
        }
        self._load_dependencies()

    def _load_config(self) -> dict:
        config_path = self.index_dir / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text(encoding="utf-8"))
        return {}

    def _load_eval_questions(self) -> list[dict[str, str]]:
        eval_path = SAMPLE_DIR / "eval_questions.csv"
        if not eval_path.exists():
            return []

        rows = read_csv(eval_path)
        return [
            {
                "question_id": row.get("question_id", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "doc_id": row.get("doc_id", ""),
            }
            for row in rows
            if row.get("question")
        ]

    def _load_examples(self) -> list[dict[str, str]]:
        return self._load_eval_questions()[:12]

    def _load_dependencies(self) -> None:
        try:
            import numpy as np
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "필요한 패키지가 없습니다. 먼저 pip install -r requirements.txt 를 실행하세요."
            ) from exc

        self.np = np
        self.faiss = faiss
        self.index = read_faiss_index(faiss, self.index_dir / "index.faiss")
        self.metadata = read_jsonl(self.index_dir / "metadata.jsonl")
        self.model = SentenceTransformer(self.model_name)

    def search(self, question: str, top_k: int) -> dict:
        query_embedding = self.model.encode(
            [e5_query(question)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(self.np.asarray(query_embedding, dtype="float32"), top_k)
        sources = []
        known_answer = self.answer_by_question.get(normalize_question(question))
        for rank, (score, index_id) in enumerate(zip(scores[0], indices[0]), start=1):
            if index_id < 0:
                continue
            item = self.metadata[int(index_id)]
            source_answer_in_chunk = answer_in_text(known_answer, item["text"]) if known_answer else False
            sources.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": item["chunk_id"],
                    "doc_id": item["doc_id"],
                    "title": item.get("title", ""),
                    "text": item["text"],
                    "is_cited": False,
                    "cited_text": "",
                    "has_reference": bool(known_answer),
                    "answer_in_chunk": source_answer_in_chunk,
                }
            )

        evaluation = build_retrieval_evaluation(sources, known_answer, top_k)
        if evaluation["should_refuse"]:
            answer = UNAVAILABLE_MESSAGE
            display_sources = []
        else:
            attach_citations(question, sources, known_answer)
            attach_excerpts(question, sources, known_answer)
            answer = build_retrieval_answer(question, sources, known_answer)
            display_sources = sources

        return {"question": question, "answer": answer, "sources": display_sources, "evaluation": evaluation}


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().lower()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？다])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def question_keywords(question: str) -> set[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", question.lower())
    stopwords = {
        "무엇", "누가", "언제", "어디", "어떤", "어떻게", "하는", "있는", "없는",
        "인가", "입니까", "했어", "했나요", "되나요", "되었나", "것은", "곳은",
    }
    return {token for token in tokens if token not in stopwords}


def compact_evidence(question: str, source_text: str, known_answer: str | None = None) -> str:
    sentences = split_sentences(source_text)
    if not sentences:
        return source_text[:260]

    if known_answer:
        for sentence in sentences:
            if known_answer in sentence:
                return sentence[:360]

    keywords = question_keywords(question)
    if not keywords:
        return sentences[0][:360]

    ranked = []
    for sentence in sentences:
        lower = sentence.lower()
        score = sum(1 for keyword in keywords if keyword in lower)
        ranked.append((score, len(sentence), sentence))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    best = ranked[0][2] if ranked else sentences[0]
    return best[:360]


def attach_citations(
    question: str,
    sources: list[dict],
    known_answer: str | None = None,
) -> None:
    if not sources:
        return

    cited_any = False
    if known_answer:
        for source in sources:
            if known_answer in source["text"]:
                source["is_cited"] = True
                source["cited_text"] = compact_evidence(question, source["text"], known_answer)
                cited_any = True

    if not cited_any:
        sources[0]["is_cited"] = True
        sources[0]["cited_text"] = compact_evidence(question, sources[0]["text"], known_answer)


def attach_excerpts(
    question: str,
    sources: list[dict],
    known_answer: str | None = None,
) -> None:
    for source in sources:
        source["excerpt"] = source.get("cited_text") or compact_evidence(
            question,
            source["text"],
            known_answer,
        )


def build_retrieval_answer(
    question: str,
    sources: list[dict],
    known_answer: str | None = None,
) -> str:
    if not sources:
        return (
            "관련 문서를 찾지 못했습니다.\n\n"
            "현재 버전은 LLM이 없는 Retrieval-only RAG라서, 검색된 근거가 없으면 답변을 만들지 않습니다."
        )

    best = sources[0]
    cited_source = next((source for source in sources if source.get("is_cited")), best)
    evidence = cited_source.get("cited_text") or compact_evidence(question, best["text"], known_answer)

    if known_answer:
        return f"답변: {known_answer}\n\n핵심 근거: {evidence}"

    return (
        f"핵심 근거: {evidence}\n\n"
        "참고: 현재는 LLM이 없는 Retrieval-only 버전이라 검색된 문장에서 핵심 근거만 추려 보여줍니다."
    )


UNAVAILABLE_MESSAGE = "제공된 문서에서 확인할 수 없습니다."


def build_retrieval_evaluation(
    sources: list[dict],
    known_answer: str | None,
    top_k: int,
) -> dict:
    if not known_answer:
        return {
            "has_reference": False,
            "success": None,
            "answer": "",
            "top_k": top_k,
            "answer_hit_count": 0,
            "should_refuse": True,
        }

    answer_hit_count = sum(1 for source in sources if source.get("answer_in_chunk"))
    success = answer_hit_count > 0
    return {
        "has_reference": True,
        "success": success,
        "answer": known_answer,
        "top_k": top_k,
        "answer_hit_count": answer_hit_count,
        "should_refuse": not success,
    }


class ChatHandler(BaseHTTPRequestHandler):
    engine: SearchEngine

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ["/", "/index.html"]:
            self.send_text(CHAT_HTML, "text/html; charset=utf-8")
            return

        if parsed.path == "/api/examples":
            self.send_json({"examples": self.engine.examples})
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/search":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body)
            question = str(payload.get("question", "")).strip()
            top_k = int(payload.get("top_k", 5))
            top_k = max(1, min(top_k, 10))

            if not question:
                self.send_json({"error": "질문을 입력하세요."}, status=400)
                return

            start = time.perf_counter()
            result = self.engine.search(question, top_k)
            response_time_ms = round((time.perf_counter() - start) * 1000, 2)
            response_time_text = format_duration_ms(response_time_ms)
            result["response_time_ms"] = response_time_ms
            result["response_time_text"] = response_time_text
            result["evaluation"]["response_time_ms"] = response_time_ms
            result["evaluation"]["response_time_text"] = response_time_text
            write_chat_log(result)
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args) -> None:
        return

    def send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def write_chat_log(result: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "chat_logs.csv"
    fieldnames = [
        "timestamp",
        "question",
        "top_k",
        "response_time_ms",
        "response_time_text",
        "has_reference",
        "success",
        "answer_hit_count",
        "answer",
        "top_score",
        "top_chunk_id",
    ]

    evaluation = result.get("evaluation", {})
    sources = result.get("sources", [])
    top_source = sources[0] if sources else {}
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": result.get("question", ""),
        "top_k": evaluation.get("top_k", ""),
        "response_time_ms": result.get("response_time_ms", ""),
        "response_time_text": result.get("response_time_text", ""),
        "has_reference": evaluation.get("has_reference", ""),
        "success": evaluation.get("success", ""),
        "answer_hit_count": evaluation.get("answer_hit_count", ""),
        "answer": evaluation.get("answer", ""),
        "top_score": top_source.get("score", ""),
        "top_chunk_id": top_source.get("chunk_id", ""),
    }

    exists = log_path.exists()
    with log_path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def format_duration_ms(duration_ms: float) -> str:
    total_seconds = int(round(duration_ms / 1000))
    if total_seconds < 1:
        return "1초 미만"

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    parts = []
    if hours:
        parts.append(f"{hours}시간")
    if minutes:
        parts.append(f"{minutes}분")
    if seconds:
        parts.append(f"{seconds}초")

    return " ".join(parts) if parts else "1초 미만"


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 채팅형 로컬 UI 서버를 실행합니다.")
    parser.add_argument("--index-dir", type=Path, default=VECTOR_DIR / "faiss" / "chunk_700")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()

    if not (args.index_dir / "index.faiss").exists():
        raise SystemExit(
            f"FAISS 인덱스를 찾지 못했습니다: {args.index_dir}\n"
            "먼저 python src/02_build_index.py --chunk-size 700 을 실행하세요."
        )

    ChatHandler.engine = SearchEngine(args.index_dir, args.model_name)
    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"RAG Chat UI: http://{args.host}:{args.port}/")
    print("종료하려면 Ctrl+C를 누르세요.")
    server.serve_forever()


if __name__ == "__main__":
    main()
