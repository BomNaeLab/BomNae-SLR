import cv2
import numpy as np
from collections import deque
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
from mp_process import MediaPipeProcessor
from connection_manager import ConnectionManager
from reshape_data import reshape_data
from vote_system import vote_system
from collect_data import collect_data

# FastAPI 애플리케이션 초기화
app = FastAPI()

# 연결 관리 객체
manager = ConnectionManager()

# 상수 정의
MAX_FRAMES = 63
FRAME_SKIP = 6  # 새로운 예측을 위한 프레임 스킵
frame_counter = 0
# 데이터 버퍼 초기화
left_hand_data = deque(maxlen=MAX_FRAMES)
right_hand_data = deque(maxlen=MAX_FRAMES)
pose_data = deque(maxlen=MAX_FRAMES)
recent_results = deque(maxlen=20)
# Static 폴더에서 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_html():
    """HTML 파일 반환"""
    try:
        with open("static/index.html", "r") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Static HTML File Not Found</h1>", status_code=404)

async def handle_video_stream(websocket, mp_processor, frame_counter):
    """비디오 스트림 처리"""
    try:
        # 바이너리 데이터 수신
        data = await websocket.receive_bytes()

        # Blob 데이터를 numpy 배열로 디코딩
        np_img = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # MediaPipe 처리
        data = mp_processor.process_with_mediapipe(frame)
        lh_points = data['hand_left_keypoints_3d']
        rh_points = data['hand_right_keypoints_3d']
        p_points = data['pose_keypoints_3d']
        
        # 각 키포인트를 3D로 변환
        preFrameCoordP = np.array([[p_points[i], p_points[i + 1], p_points[i + 2]] for i in range(0, len(p_points), 3)], dtype=np.float32)
        preFrameCoordL = np.array([[lh_points[i], lh_points[i + 1], lh_points[i + 2]] for i in range(3, len(lh_points), 3)], dtype=np.float32)
        preFrameCoordR = np.array([[rh_points[i], rh_points[i + 1], rh_points[i + 2]] for i in range(3, len(rh_points), 3)], dtype=np.float32)
        # 포즈 랜드마크에서 특정 인덱스만 사용
        preFrameCoordP = preFrameCoordP[[i for i in range(19) if (0 <= i <= 7) or (17 <= i <= 18)]]

        # 랜드마크 값들을 모델 트레이닝에 적합한 모양으로 재배열.
        frameCoordL = preFrameCoordL.reshape(5, 4, 3).transpose(1, 0, 2)[::-1]
        frameCoordR = preFrameCoordR.reshape(5, 4, 3).transpose(1, 0, 2)[::-1]
        frameCoordP = preFrameCoordP
        
        # 데이터 버퍼에 추가
        left_hand_data.append(frameCoordL)
        right_hand_data.append(frameCoordR)
        pose_data.append(frameCoordP)
        frame_counter += 1  # 프레임 카운터 증가

        print(f'{frame_counter}')  # 출력

        # 63프레임 수집 후 예측
        if frame_counter >= MAX_FRAMES and len(left_hand_data):
            co_l, co_r, co_p = collect_data(left_hand_data, right_hand_data, pose_data)
            result = vote_system(co_l, co_r, co_p)
            if result:
                recent_results.appendleft(result)
                print(f"Sending result: {recent_results[0]}")  # 전송되는 데이터 확인
                await websocket.send_text(json.dumps(list(recent_results)))  # results를 JSON으로 변환해 보내기
                # 결과를 WebSocket에 보내기
            frame_counter -= FRAME_SKIP  # 프레임 스킵
            

        # 갱신된 frame_counter를 반환
        return frame_counter
    except Exception as e:
        print(f"Error processing video stream: {e}")
        await websocket.close()

@app.websocket("/ws/{client_id}")

async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket 엔드포인트"""
    await manager.connect(websocket, client_id=client_id)
    print(f"Client {client_id} connected.")

    mp_processor = MediaPipeProcessor() if client_id == "video" else None
    frame_counter = 0  # frame_counter 초기화

    try:
        while True:
            if client_id == "video":
                frame_counter = await handle_video_stream(websocket, mp_processor, frame_counter)
            elif client_id == "model_server":
                await handle_model_server(websocket)
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected.")
    except Exception as e:
        print(f"Error with client {client_id}: {e}")
    finally:
        if mp_processor:
            mp_processor.close()
        await manager.disconnect(websocket, client_id)

async def handle_model_server(websocket):
    """모델 서버 처리"""
    try:
        model_data = await websocket.receive_text()
        await manager.broadcast(model_data, target_client_id="model_server")
    except Exception as e:
        print(f"Error processing model server: {e}")
        await websocket.close()

async def main():
    """FastAPI 서버 실행"""
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, reload=False)
    server = uvicorn.Server(config)
    try:
        await server.serve()
    except asyncio.CancelledError:
        print("Server shutdown by asyncio task cancellation.")
    except KeyboardInterrupt:
        print("Server terminated by Ctrl+C.")
    finally:
        await server.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server terminated.")
    except Exception as e:
        print(f"Unhandled exception: {e}")