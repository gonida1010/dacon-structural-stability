# dacon-structural-stability

> **DACON 구조물 안정성 추론 공모전** — DINOv2 Attention Masking + EVA-Giant 2-Stage Pipeline

📊 **최종 순위**: 16등 / 851명 중 상위 4% &nbsp;&nbsp;|&nbsp;&nbsp; 📅 **기간**: 2026.03.03 ~ 03.30 &nbsp;&nbsp;|&nbsp;&nbsp; 👥 5인 팀  
🏆 **Private LogLoss**: 0.01903 &nbsp;&nbsp;|&nbsp;&nbsp; 🏢 **팀 아카이브**: [Dacon-contest Organization](https://github.com/Dacon-contest)

---

## 개인 기여

이 레포는 팀 내에서 "**최고 성능 파이프라인**"을 설계·구현한 개인 작업물입니다.

팀원들과 가볍게 시작했지만, 코랩 파일만 11번 버전을 넘기며 로컬과 A100 환경을 오가며 실험을 반복했습니다. 단순히 모델을 바꿔보는 수준이 아니라, **"왜 Dev에서는 완벽한데 Test에서는 나빠지는가?"** 라는 질문에서 출발해 DINOv2 Attention Masking → EVA-Giant 2-Stage Curriculum Learning 파이프라인을 설계했고, 이 파이프라인이 팀 최종 제출의 핵심이 되었습니다.

### 📝 연구 정리 블로그
- [Vision AI 최종 실험 — 2-Stage Pipeline의 이론적 기반과 실험 결과](https://pak1010pak.tistory.com/176)
- [Vision 시각화 검증 상세 분석 — DINOv2·EVA Attention 분석](https://pak1010pak.tistory.com/170)

---

## 문제 인식에서 해결까지

### 문제: Dev-Test 갭

v10까지의 실험에서 DINOv2와 EVA-Giant를 독립 학습 후 앙상블하면 Dev에서는 거의 완벽한 성능(OOF LogLoss = 0.001792)을 보였지만, 리더보드(Test)에서는 LogLoss가 0.0476으로 급격히 벌어졌습니다.

| 모델 | 강점 | 약점 |
|---|---|---|
| **DINOv2-Large** | 구조물을 "물체"로 인식하는 공간 이해 능력 탁월 | 안정/불안정의 미세한 차이 구분 부족 |
| **EVA-Giant** | 세밀한 시각적 패턴으로 분류 정밀도 높음 | **배경(체커보드 바닥)에 의존하여 오분류** |

### 원인 진단

EVA가 배경 텍스처(체커보드 패턴, 조명 각도)를 shortcut으로 학습하고 있었습니다. Dev와 Train은 같은 렌더링 엔진에서 유사한 배경을 공유하지만, Test는 미세하게 다른 배경 분포를 가질 수 있다는 가설을 세웠습니다.

### 핵심 발상

> "DINO는 객체를 잘 찾고, EVA는 잘 분류한다.  
> 그러면 DINO가 배경을 날려주고, EVA가 구조물만 보면?"

```
v10: DINOv2(전체 이미지) + EVA(전체 이미지) → Meta-Learner → 예측
v11: DINOv2(전체 이미지) → Attention Mask → EVA(마스크 이미지) → 예측
            ↑ 배경 제거 전문                    ↑ 구조물 분석 전문
```

---

## 2-Stage Curriculum Learning Pipeline

### Phase 1: DINOv2 Attention Masking + EVA 학습

DINOv2의 self-supervised 학습 특성을 활용했습니다. DINOv2는 라벨 없이 학습되었기 때문에 분류 단서가 아닌 **객체 자체의 경계**에 attention을 집중합니다. 이 특성을 이용해:

1. Finetuned DINOv2의 마지막 4개 레이어에서 CLS→patch attention 추출
2. Multi-head, multi-layer 평균 → 24×24 heatmap 생성
3. Otsu 이진화 + morphological processing → binary mask
4. 배경을 검은색으로 제거한 마스크 이미지 생성
5. **EVA-Giant가 마스크 이미지로 5-fold CV 학습** → 배경 shortcut 차단

### Phase 2: 원본 이미지 적응 (Curriculum Learning)

마스크 이미지로만 학습하면 테스트 시 원본 이미지와의 도메인 갭이 발생합니다. Phase 1 가중치에서 출발해 극히 낮은 learning rate(bb: 1e-6, head: 1e-5)로 단 3 epoch만 원본 이미지에 적응시켰습니다.

- Phase 1에서 학습한 "구조물 중심 표현"은 대부분 보존
- 원본 이미지의 색상/조명/질감 분포에만 미세 적응
- 배경 shortcut 재학습은 방지

### 결과

```
                        Dev LogLoss    Accuracy    Phase 효과
Phase 1 (마스크만)        0.133277      0.9300        —
Phase 2 (원본 적응)       0.018457      1.0000     7.2배 개선
```

단 3 epoch의 추가 학습으로 Dev LogLoss가 0.133 → 0.018로 **7.2배 개선**되었습니다.

---

## 시각화 분석 — 모델이 실제로 보는 것

EVA Attention Rollout을 통해 모델이 이미지의 어디를 보는지 시각화한 결과입니다.

<p>
  <img src="https://github.com/user-attachments/assets/0b97eb82-a2af-4c70-8480-f6f92188b2cc" alt="EVA Attention Rollout 분석" />
</p>

- **UNSTABLE (P≈0.999)**: 구조물 상단의 꺾임점, 기울어진 블록에 attention 집중. 배경에는 거의 attention 없음 → 마스킹 성공
- **BORDERLINE (P≈0.3)**: attention이 구조물 전체에 분산되어 특정 위험 부위를 찾지 못함 → 불확실한 판단
- **STABLE (P≈0.001)**: 구조물의 넓은 기반부에 고르게 분포. 기울어짐이나 꺾임 없음 → 자신 있는 stable 판단

이 패턴은 배경과 무관하게 **구조물의 기하학적 불균형**을 학습했음을 보여줍니다.

---

## 사용 모델

| 모델 | 파라미터 | 입력 | 환경 | 용도 |
|---|---|---|---|---|
| EVA-Giant/14 | 1.0B | 336px | A100 | 구조물 분류 (메인) |
| DINOv2 ViT-L/14 reg4 | 304M | 336px | Local | Attention Masking + 앙상블 |
| EVA02-Large/14 | 305M | 448px | Local | 로컬 실험 검증 |

## 적용 기법 요약

### 모델 아키텍처
- **Shared Backbone Dual-View**: front/top 이미지를 동일 백본으로 인코딩
- **Attention Gate Fusion**: 두 뷰 특징의 학습 가능한 가중 합산
- **CrossViewFusion (DINOv2)**: 학습 가능한 CLS 토큰이 front+top 1152개 패치에 attend하는 4-layer Transformer
- **MLP Head**: LayerNorm → 512 → 256 → 2 (Dropout 0.3/0.15)

### 학습 전략
- **ShapeStacks h=6 Pretrain** → **5-Fold CV Finetune** 2단계 학습
- **CutMix (30%) + Mixup (30%) + Clean (40%)** 배치 단위 증강
- **Focal Loss** (α=0.25, γ=2.0) × 0.7 + **Label Smoothing** (ε=0.05) × 0.3
- Differential LR (백본 0.1× / 헤드 1×) + Layer-wise LR Decay
- Cosine Annealing + Warmup 2 epoch + AMP fp16

### 추론
- **멀티 백본 앙상블**: 여러 백본 × 5-fold 체크포인트 평균
- **TTA**: 원본 + HFlip + Brightness + CenterCrop (4종)
- **Temperature Scaling**: 확률 보정으로 LogLoss 개선

---

## 실행 방법

### 환경 설정

```bash
cd dacon-structural-stability
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 로컬 학습 (12GB GPU — RTX 4070 Ti)

```bash
# ShapeStacks Pretrain + 5-Fold Finetune (한번에)
python train.py --backbone eva02_large --stage both --pretrain_epochs 15 --finetune_epochs 50 --include_dev --grad_checkpointing --resume
python train.py --backbone dinov2_large --stage both --pretrain_epochs 15 --finetune_epochs 50 --include_dev --grad_checkpointing --resume
```

### 추론

```bash
# Dev 성능 검증
python inference.py --backbones eva02_large dinov2_large --tta --validate

# 최종 제출 파일 생성
python inference.py --backbones eva02_large dinov2_large --tta --temperature 1.0
```

### Google Colab (A100)

`colab_train.ipynb`를 Colab에 업로드하여 사용합니다. 체크포인트는 Google Drive에 자동 저장되어 세션이 끊겨도 이어서 학습이 가능합니다. 상세 사용법은 노트북 내 셀 주석을 참고하세요.

---

## 관련 저장소

| Repository | 설명 |
|---|---|
| [Dacon-contest (팀 아카이브)](https://github.com/Dacon-contest) | 팀 전체 실험·전처리·모델 아카이브 |
