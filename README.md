# BomNae-SLR
---
### 이 연구는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 "수어 영상"을 활용하여 수행된 연구입니다. 본 연구에 활용된 데이터는 AI 허브(aihub.or.kr)에서 다운로드 받으실 수 있습니다.

프로젝트 개요 (Project Overview)

BomNae-SLR은 웹캠 영상을 통해 실시간으로 수어 동작을 인식하고, 이를 텍스트로 번역하는 시스템입니다. 사용자가 웹캠 앞에서 수어 동작을 하면, 시스템이 이를 분석하여 해당하는 단어를 화면에 보여줍니다.

주요 기능 (Key Features)

 - 실시간 수어 인식: 웹캠 영상을 실시간으로 처리하여 빠른 번역 결과를 제공합니다.
 - 웹 기반 인터페이스: 사용자가 별도의 프로그램 설치 없이 웹 브라우저에서 쉽게 이용할 수 있습니다.
 - 고성능 키포인트 추출: MediaPipe 라이브러리를 활용하여 영상에서 손과 신체의 주요 움직임을 정확하고 효율적으로 감지합니다.
 - 딥러닝 기반 번역: TensorFlow/Keras 기반의 딥러닝 모델을 사용하여 복잡한 수어 동작을 분류하고 번역합니다.

시스템 동작 방식 (How It Works)

 1. 영상 캡처 (프론트엔드): 사용자의 웹캠 영상을 웹 페이지(static/index.html)에서 실시간으로 캡처합니다.
 2. 데이터 전송: 캡처된 영상 프레임을 WebSocket을 통해 백엔드 서버(main.py)로 전송합니다.
 3. 키포인트 추출 (백엔드): 서버는 MediaPipe를 이용해 전송된 영상에서 양손과 신체의 3D 주요 좌표(keypoint)를 추출합니다.
 4. 모델 추론: 추출된 좌표 데이터를 일정 길이의 시퀀스로 만들어 수어 인식 딥러닝 모델(SLR_model.py)의 입력으로 사용합니다. 모델은 이 시퀀스를 분석하여 해당하는 단어를 예측합니다.
 5. 결과 표시: 예측된 단어는 다시 웹 페이지로 전송되어 사용자 화면에 나타납니다.

실행을 위한 요구 사항 (Requirements)

 - Python 3.8 이상
 - FastAPI
 - Uvicorn
 - OpenCV
 - TensorFlow
 - NumPy
 - MediaPipe
 - websockets

설치 방법 (Installation)

아래 명령어를 사용하여 필요한 라이브러리를 설치합니다.

 1 pip install fastapi "uvicorn[standard]" opencv-python numpy tensorflow mediapipe websockets

사용 방법 (Usage)

 1. 터미널에서 아래 명령어를 입력하여 백엔드 서버를 실행합니다.

  1     python main.py

 2. 웹 브라우저를 열고 http://localhost:8000 주소로 접속합니다.
 3. 브라우저에서 웹캠 접근 권한을 허용하면 애플리케이션이 동작을 시작하고, 수어 인식이 화면에 표시됩니다.
