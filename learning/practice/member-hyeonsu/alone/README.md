# 슈퍼 상담 RAG 챗봇

기존 질문-only Retrieval baseline을 재사용한 Streamlit 데모입니다. 질문을 임베딩하고, 인텐트 분류 Top-3를 이용해 `top3_fallback` 방식으로 기존 Chroma 문서를 검색합니다. 배포 기본 모드는 Ollama를 호출하지 않는 Retrieval-only입니다.

## 실행 흐름

`질문 → 인텐트 Top-3 → Chroma 검색 → 검색 질문·답변·거리 표시`

사이드바에서 Retrieval-only 체크를 해제하면 Ollama 최종 답변을 선택적으로 사용할 수 있습니다. Streamlit Cloud에는 Ollama가 설치되어 있지 않으므로 배포에서는 체크 상태를 유지하세요.

## 로컬 실행

```powershell
python -m pip install -r requirements.txt
$env:STREAMLIT_SERVER_FILE_WATCHER_TYPE="none"
streamlit run streamlit_app.py
```

필수 배포 파일은 `streamlit_app.py`, `src/`, `models/super_intent_classifier.joblib`, `chroma_db/`입니다. `.env`가 없어도 기본 상대경로와 Retrieval-only로 실행할 수 있습니다.

## GitHub 및 Streamlit Cloud

```powershell
git init
git add streamlit_app.py src requirements.txt runtime.txt .streamlit models/super_intent_classifier.joblib chroma_db README.md .gitignore
git commit -m "Add Streamlit retrieval demo"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

Streamlit Community Cloud에서 저장소를 선택하고 Main file path에 `streamlit_app.py`를 입력한 뒤 Deploy합니다. 배포 URL은 앱 설정의 Share/앱 주소에서 확인할 수 있습니다.

Notion에서는 `/embed` 블록에 배포 URL을 붙여 넣습니다. 임베드가 차단되면 URL을 일반 링크 또는 버튼으로 삽입해 새 탭에서 열도록 하세요.

## 파일 크기 및 제외 기준

현재 모델은 약 97MB, Chroma DB는 약 94MB입니다. GitHub 단일 파일 100MB 제한에 근접하므로 모델은 Git LFS 사용을 권장합니다. Training/Validation 원본, 실험 결과, 로그는 저장소에 올리지 않습니다. `chroma_db/`와 모델은 실행에 필수이므로 `.gitignore`에서 제외하지 않습니다.

## 문제 해결

- `torchvision` 경고가 보이면 `.streamlit/config.toml`의 watcher 비활성화 설정을 확인하세요.
- Chroma 컬렉션 오류가 나면 저장소 루트에 `chroma_db/`가 포함되어 있는지 확인하세요.
- 모델 파일 오류가 나면 `models/super_intent_classifier.joblib`이 커밋되었는지 확인하세요.
- Ollama 오류는 Retrieval-only에서는 발생하지 않습니다. 생성 기능을 사용할 때만 `ollama serve`와 모델 설치가 필요합니다.
