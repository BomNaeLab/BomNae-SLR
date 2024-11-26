import asyncio
import websockets
import json
from collections import deque
from reshape_data import reshape_data
from vote_system import vote_system  # vote_system 모듈 임포트

# 저장할 파일 경로
DATA_FILE = "/home/bomnaelab/fa_test/data.json"

# 각 좌표 데이터를 저장할 큐 (최대 63프레임 유지)
MAX_FRAMES = 63
left_hand_data = deque(maxlen=MAX_FRAMES)
right_hand_data = deque(maxlen=MAX_FRAMES)
pose_data = deque(maxlen=MAX_FRAMES)

# 데이터 저장 함수
async def save_data_to_file(left_hand_data, right_hand_data, pose_data, file_path):
    """
    좌표 데이터를 하나의 JSON 파일에 저장하는 함수
    """
    try:
        data = {
            "left_hand": list(left_hand_data),
            "right_hand": list(right_hand_data),
            "pose": list(pose_data)
        }

        # JSON 파일에 데이터를 기록
        with open(file_path, "w") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        # print(f"Data successfully saved to {file_path}.")  # 디버깅 로그
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")

# 좌표 처리 및 WebSocket 연결 함수
async def process_coordinates():
    uri = "ws://192.168.0.2:8000/ws/model_server"
    frame_counter = 0  # 프레임 카운터 추가

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected to WebSocket server.")
                while True:
                    try:
                        data = await websocket.recv()
                        coordinates = json.loads(data)

                        # 좌표 데이터를 분류하여 큐에 저장
                        if "left_hand" in coordinates:
                            left_hand_data.append(coordinates["left_hand"])
                        elif "right_hand" in coordinates:
                            right_hand_data.append(coordinates["right_hand"])
                        elif "pose" in coordinates:
                            pose_data.append(coordinates["pose"])
                        else:
                            print(f"Unknown data type: {coordinates}")
                            continue
                        frame_counter += 1
                        # 63프레임이 모였을 때 결과 도출
                        if frame_counter >= 63:
                            l,r,p=reshape_data()
                                
                            result = vote_system(l,r,p)  # vote_system 호출
                            if result:
                                print(f"Predicted result: {result}")
                            # 6프레임마다 새로운 예측
                            frame_counter = 0  # 프레임 카운터 리셋

                        # 하나의 JSON 파일에 데이터 저장
                        await save_data_to_file(left_hand_data, right_hand_data, pose_data, DATA_FILE)

                    except websockets.ConnectionClosedError as e:
                        print(f"WebSocket connection closed: {e}")
                        break
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                    except Exception as e:
                        print(f"Unexpected error: {e}")
        except websockets.exceptions.InvalidURI as e:
            print(f"Invalid WebSocket URI: {e}")
        except OSError as e:
            print(f"OS error: {e}")
        except Exception as e:
            print(f"Error connecting to WebSocket: {e}")

        print("Reconnecting in 5 seconds...")
        await asyncio.sleep(5)

# 실행
if __name__ == "__main__":
    try:
        asyncio.run(process_coordinates())
    except KeyboardInterrupt:
        print("Program terminated.")