import asyncio
import websockets
import json
import mediapipe as mp
import cv2


async def send_coordinates():
    uri = "ws://localhost:8001/ws/coordinate_server"
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv() 
            print(f"Received coordinates: {data}")

asyncio.run(send_coordinates())
