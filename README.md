# Clean_PDF ✏️ → 📄

PDF에서 필기를 지우고, **필기에 가려진 인쇄 글씨를 AI가 복원**합니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hnr09/Clean_PDF/blob/main/Clean_PDF.ipynb)

| 필기 종류 | 사용할 도구 | 결과 |
|---|---|---|
| 종이 + **검정 펜** (흑백 스캔) | `erase_ai.py` (AI) | 필기 감지 → 지우기 → 밑의 글씨 복원 |
| 종이 + 색 펜 / 형광펜 / 연필 | `erase_handwriting.py` | 색·밝기 필터 (빠름) |
| 태블릿 필기 (PDF 주석) | `erase_handwriting.py --mode annot` | 무손실 삭제 |

## Colab에서 쓰기 (추천)
1. 위의 **Open in Colab** 버튼 클릭
2. `런타임 → 런타임 유형 변경 → T4 GPU` (선택, 빠름)
3. 셀을 위에서부터 ▶ 실행 → PDF 업로드 → 결과 비교 → 다운로드
   - 필기가 남으면 **민감도↓**, 인쇄 글씨가 지워지면 **민감도↑**
   - 결과가 아쉬우면 3번 셀 **학습하기** 체크 → 올린 스캔본으로 바로 미세조정 (GPU 10~20분)
   - 7번 셀로 학습된 `eraser.pt`를 받아 이 저장소에 덮어쓰면 다음부터 좋아진 모델을 사용

## 내 컴퓨터에서 쓰기
```bash
git clone https://github.com/hnr09/Clean_PDF && cd Clean_PDF
pip install -r requirements.txt
python erase_ai.py 시험지.pdf                 # AI: 검정 펜 필기 지우기 + 복원 → 시험지_clean.pdf
python erase_ai.py 시험지.pdf --thr 0.35      # 더 많이 지우기
python erase_ai.py 시험지.pdf --save-mask     # 필기로 감지한 영역도 따로 저장
python erase_ai.py 시험지.pdf --no-protect    # 그래프/도형 보호 끄기 (그래프 위에 쓴 필기까지 지울 때)
python erase_handwriting.py 노트.pdf          # 색 펜 / 태블릿 주석
```

## 모델 학습
모델은 **합성 데이터**로 학습합니다. 깨끗한 시험지 페이지(자동 생성 또는 직접 넣은 PDF)에 가짜 손글씨·체크·동그라미를 그려 넣고
"필기 있는 이미지 → 깨끗한 이미지"를 배우게 하므로, 필기 밑에 가려진 글자 획도 이어서 그려 넣습니다.

```bash
sudo apt install fonts-nanum fonts-nanum-extra   # 합성용 한글 글꼴
# 필기 있는 내 스캔본으로 미세조정 (기존 필기는 자동으로 학습에서 제외) ★ 가장 효과적
python train.py --resume eraser.pt --scan-pdf 내스캔.pdf --steps 3000
# 필기 없는 원본 시험지 PDF도 있으면 함께 (같은 글꼴·표 양식 학습)
python train.py --resume eraser.pt --scan-pdf 내스캔.pdf --clean-pdf 기출원본.pdf --steps 3000
# 처음부터 학습
python train.py --steps 8000 --out eraser.pt
```

## 파일 구성
| 파일 | 설명 |
|---|---|
| `Clean_PDF.ipynb` | Colab 노트북 |
| `erase_ai.py` | AI 필기 제거 + 복원 |
| `erase_handwriting.py` | 색상/연필 필터, 태블릿 주석 삭제 |
| `protect.py` | 인쇄된 그래프·도형·아이콘·표 선 보호 |
| `train.py` / `synth.py` / `model.py` | 학습 / 합성 데이터 / U-Net 모델 |
| `eraser.pt` | 학습된 모델 (CPU로 짧게 학습한 초기 버전) |

## 한계
- 필기가 글자를 **완전히 덮은** 경우는 원래 글자를 알 수 없으므로 주변으로 추정해 채웁니다(틀릴 수 있음).
- 인쇄된 그래프(좌표축·함수 곡선), 채워진 아이콘(나무·●), 제목 배지, 표 선은 자동으로 보호합니다. 대신 **그래프 위에 쓴 필기는 남을 수 있어요** → `--no-protect`.
- 짧은 세로획 필기는 표 선과 헷갈려 남을 수 있습니다 → 추가 학습으로 개선.
- 결과 PDF는 이미지 PDF입니다(텍스트 검색 불가).
