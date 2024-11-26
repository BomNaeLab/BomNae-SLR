import asyncio
import websockets
import json
import mediapipe as mp
import cv2


async def send_coordinates():
    uri = "ws://localhost:8001/ws/coordinate_server"
    async with websockets.connect(uri) as websocket:
        while True:
            coordinates = {"x": 0.5, "y": 0.5, "z": 0.0}  # Mediapipe에서 얻은 좌표 예시
            serialized_data =json.dumps(coordinates)
            await websocket.send(serialized_data)
            response = await websocket.recv()
            print(f"Received from server: {response}")
            await asyncio.sleep(0.1)  # 좌표 전송 주기

asyncio.run(send_coordinates())
