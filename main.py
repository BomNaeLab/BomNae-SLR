import base64
import numpy as np
import cv2
import mp_process as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from connection_manager import ConnectionManager  # 분리된 ConnectionManager 불러오기
import json


manager = ConnectionManager()
app = FastAPI()

# Static 폴더 내의 파일들을 서빙하도록 설정 (HTML 파일을 Static 폴더에 두기)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_html():
    with open("static/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id=client_id)
    try:
        while True:
            if client_id == "video":
                try:
                    # 바이너리 데이터 수신
                    data = await websocket.receive_bytes()

                    # Blob 데이터를 numpy 배열로 디코딩
                    np_img = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                    # MediaPipe 처리
                    landmarks = mp.process_with_mediapipe(frame)
                    l, r, p = mp.process_with_mediapipe_apart(frame)

                    # 처리된 좌표를 JSON으로 변환
                    l_json = json.dumps({"left_hand": l})
                    r_json = json.dumps({"right_hand": r})
                    p_json = json.dumps({"pose": p})
                    landmarks_json = json.dumps(landmarks)

                    # 결과를 해당 클라이언트 그룹으로 브로드캐스트
                    await manager.broadcast(landmarks_json, target_client_id="video")
                    await manager.broadcast(l_json, target_client_id="model_server")
                    await manager.broadcast(r_json, target_client_id="model_server")
                    await manager.broadcast(p_json, target_client_id="model_server")

                except  Exception as e:
                    print(f"Error processing video data: {e}")

            elif client_id == "model_server":
                try:
                    # Model Server의 경우 별도의 데이터 처리 로직 필요
                    # 여기에서 data를 수신하려면 별도로 정의해야 함
                    model_data = await websocket.receive_text()
                    await manager.broadcast(model_data, target_client_id="model_server")
                except Exception as e:
                    print(f"Error processing model server data: {e}")

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
        manager.disconnect(websocket, client_id)