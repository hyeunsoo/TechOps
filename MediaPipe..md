# Google MediaPipe 종합 정리

## 1. MediaPipe 개요

Google MediaPipe는 Google에서 개발한 **오픈소스 크로스플랫폼 ML(머신러닝) 프레임워크**로, 실시간으로 다양한 비전(Vision) 및 오디오 관련 AI 작업을 수행할 수 있게 해주는 도구입니다.

### 핵심 특징

- **크로스플랫폼 지원** — Android, iOS, 웹(JavaScript), Python 환경에서 모두 사용할 수 있어 다양한 디바이스와 플랫폼에 쉽게 배포할 수 있습니다.
- **온디바이스 추론** — 클라우드 서버 없이 디바이스 자체에서 실시간으로 ML 모델을 실행할 수 있어 빠른 응답 속도와 프라이버시 보호가 가능합니다.
- **낮은 진입 장벽** — ML 전문 지식 없이도 사전 학습된 솔루션을 API 호출만으로 사용할 수 있습니다.

-----

## 2. 주요 솔루션 (Tasks)

MediaPipe는 크게 **Vision**, **Text**, **Audio** 세 가지 카테고리의 태스크를 제공합니다.

### Vision 영역

- **Hand Landmark Detection** — 손의 21개 관절 포인트를 실시간 추적 (수어 인식, 제스처 컨트롤 등에 활용)
- **Face Detection / Face Mesh** — 얼굴 검출 및 468개 이상의 3D 랜드마크 추출 (AR 필터, 얼굴 분석 등)
- **Pose Estimation** — 전신 33개 포인트의 자세 추정 (피트니스 앱, 모션 캡처 등)
- **Object Detection** — 이미지/영상 내 객체 탐지
- **Image Segmentation** — 이미지에서 배경과 전경을 분리
- **Gesture Recognition** — 손동작 인식
- **Image Classification** — 이미지 분류

### Text 영역

- **Text Classification** — 텍스트 감성 분석 등
- **Text Embedding** — 텍스트 임베딩 생성

### Audio 영역

- **Audio Classification** — 소리 분류

### Generative AI 영역

- **LLM Inference** — 온디바이스에서 대규모 언어 모델 추론
- **Image Generation** — 이미지 생성

### 활용 사례

- **AR/VR 앱**: 얼굴 필터, 가상 메이크업, 핸드 트래킹 기반 인터랙션
- **피트니스/헬스케어**: 운동 자세 분석, 물리치료 보조
- **접근성**: 수어 인식, 제스처 기반 UI
- **보안**: 얼굴 인증, 동작 감지
- **콘텐츠 제작**: 실시간 배경 제거, 모션 캡처

### 간단한 Python 예시

```python
import mediapipe as mp
from mediapipe.tasks import python, vision

# 손 랜드마크 감지기 생성
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# 이미지에서 손 감지
image = mp.Image.create_from_file("image.jpg")
result = detector.detect(image)
```

### MediaPipe Studio

Google은 **MediaPipe Studio**라는 웹 기반 데모 환경도 제공하고 있어, 코드 작성 없이 브라우저에서 바로 각 솔루션을 테스트해볼 수 있습니다.

-----

## 3. 카메라 실시간 입력 처리

MediaPipe의 큰 강점 중 하나가 바로 **실시간 카메라 입력 처리**입니다.

### 입력 모드

MediaPipe는 세 가지 입력 모드를 지원합니다.

|모드             |설명                   |
|---------------|---------------------|
|**IMAGE**      |단일 정지 이미지 처리         |
|**VIDEO**      |녹화된 영상 파일의 프레임별 처리   |
|**LIVE_STREAM**|카메라에서 들어오는 실시간 스트림 처리|

`LIVE_STREAM` 모드가 카메라 실시간 입력을 처리하는 모드입니다.

### Python + OpenCV 예시 (웹캠 손 추적)

```python
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=2
) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR → RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 손 감지 처리
        results = hands.process(rgb_frame)
        
        # 감지된 손 랜드마크 그리기
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
        
        cv2.imshow('MediaPipe Hands', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

### 플랫폼별 카메라 연동

|플랫폼              |연동 방식                                                    |
|-----------------|---------------------------------------------------------|
|**Python**       |OpenCV(`cv2.VideoCapture`)로 웹캠 연결 후 프레임 단위로 전달           |
|**Android**      |CameraX API와 연동하여 카메라 프리뷰를 MediaPipe에 실시간 전달             |
|**iOS**          |AVFoundation 카메라 세션과 연동                                  |
|**웹(JavaScript)**|`navigator.mediaDevices.getUserMedia()`로 브라우저 카메라 접근 후 처리|

### 성능

일반적인 환경에서 **30fps 이상**의 실시간 처리가 가능하며, 모바일 디바이스에서도 GPU 가속을 활용해 부드럽게 동작합니다. 특히 손 추적, 얼굴 메시, 포즈 추정 같은 태스크는 실시간 카메라 입력에 최적화되어 있어 체감 지연이 거의 없습니다.

-----

## 4. Calculator Graph — 그래프 기반 파이프라인

MediaPipe의 **핵심 설계 철학**은 그래프 기반 파이프라인 아키텍처인 **Calculator Graph**입니다.

### 개념

MediaPipe의 파이프라인은 **방향성 비순환 그래프(DAG)** 형태로 구성됩니다.

```
[카메라 입력] → [전처리] → [추론] → [후처리] → [렌더링/출력]
                                         ↘ [외부 전달]
```

- 각 노드를 **Calculator**라고 부릅니다.
- 노드 간 데이터가 **Stream(패킷 스트림)**을 통해 흘러갑니다.

### 그래프 정의 방식 (ProtoBuf)

그래프는 `.pbtxt` (Protocol Buffer Text) 파일로 정의합니다.

```protobuf
# 예시: 카메라 → 전처리 → 추론 → 후처리 → 출력

input_stream: "input_video"

node {
  calculator: "ImageTransformationCalculator"  # 전처리 (리사이즈, 정규화 등)
  input_stream: "IMAGE:input_video"
  output_stream: "IMAGE:preprocessed_frame"
}

node {
  calculator: "InferenceCalculator"  # TFLite 모델 추론
  input_stream: "TENSOR:preprocessed_frame"
  output_stream: "TENSOR:raw_output"
  node_options: {
    [type.googleapis.com/mediapipe.InferenceCalculatorOptions] {
      model_path: "my_model.tflite"
      delegate { gpu {} }
    }
  }
}

node {
  calculator: "MyPostProcessCalculator"  # 후처리 (NMS, 좌표 변환 등)
  input_stream: "TENSOR:raw_output"
  output_stream: "DETECTIONS:detections"
}

node {
  calculator: "AnnotationOverlayCalculator"  # 화면에 결과 렌더링
  input_stream: "IMAGE:input_video"
  input_stream: "DETECTIONS:detections"
  output_stream: "IMAGE:output_video"
}

output_stream: "output_video"
```

### Calculator의 구조

각 Calculator(노드)는 C++로 작성하며, 기본 구조는 다음과 같습니다.

```cpp
class MyCustomCalculator : public CalculatorBase {
public:
  static absl::Status GetContract(CalculatorContract* cc) {
    // 입출력 스트림 타입 정의
    cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
    cc->Outputs().Tag("RESULT").Set<MyResult>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    // 초기화 (모델 로드, 리소스 할당 등)
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // 매 프레임마다 호출되는 핵심 로직
    auto& input = cc->Inputs().Tag("IMAGE").Get<ImageFrame>();
    
    // 처리 로직...
    
    cc->Outputs().Tag("RESULT").AddPacket(
        MakePacket<MyResult>(result).At(cc->InputTimestamp()));
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    // 정리 작업
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(MyCustomCalculator);
```

### 파이프라인 구성 예시

|단계        |사용 가능한 Calculator                                        |설명                            |
|----------|---------------------------------------------------------|------------------------------|
|카메라 실시간 입력|`CameraInputCalculator`, OpenCV 연동                       |웹캠/모바일 카메라 스트림                |
|영상 전처리    |`ImageTransformationCalculator`, `ColorConvertCalculator`|리사이즈, 색공간 변환, 정규화             |
|딥러닝 추론    |`InferenceCalculator`                                    |TFLite 모델 실행 (GPU/CPU 위임)     |
|결과 후처리    |커스텀 Calculator                                           |NMS, 좌표 변환, 필터링 등             |
|외부 전달     |커스텀 Calculator                                           |gRPC, MQTT, WebSocket, 파일 저장 등|

### 그래프의 강력한 기능들

#### 분기(Branch)

하나의 스트림을 여러 Calculator로 동시에 보낼 수 있습니다.

```
                  ↗ [얼굴 감지] → [얼굴 후처리]
[카메라] → [전처리]
                  ↘ [손 감지]  → [제스처 인식] → [외부 전달]
```

#### 동기화

여러 스트림의 타임스탬프를 자동으로 맞춰줍니다. 예를 들어 얼굴 결과와 손 결과를 같은 프레임 기준으로 합칠 수 있습니다.

#### 병렬 처리

독립적인 노드들은 자동으로 멀티스레드 병렬 실행됩니다.

#### 사이드 패킷(Side Packet)

모델 파일 경로, 설정값 등 한 번만 전달하면 되는 데이터를 별도로 주입할 수 있습니다.

### 주의할 점

Calculator Graph 기반 커스터마이징은 **C++ 레벨**에서 이루어지기 때문에 진입 장벽이 다소 있습니다. Python에서는 미리 만들어진 Solution(Hand, Face, Pose 등)을 쉽게 쓸 수 있지만, 완전히 새로운 그래프를 자유롭게 구성하려면 C++ 빌드 환경(Bazel)이 필요합니다.

Python 환경에서 유사한 파이프라인을 가볍게 구성하고 싶다면, MediaPipe의 기본 Solution들을 조합하면서 OpenCV와 함께 직접 파이프라인 로직을 작성하는 방식도 현실적인 대안입니다.