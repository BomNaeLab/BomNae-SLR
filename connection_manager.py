from fastapi import WebSocket, WebSocketDisconnect

# class ConnectionManager:
#     def __init__(self):
#         self.active_connections: list[WebSocket] = []

#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         self.active_connections.append(websocket)

#     def disconnect(self, websocket: WebSocket):
#         self.active_connections.remove(websocket)

#     async def send_message(self, message: str, websocket: WebSocket):
#         await websocket.send_text(message)

#     async def broadcast(self, message: str):
#         for connection in self.active_connections:
#             await connection.send_text(message)
class ConnectionManager:
    def __init__(self):
        self.client_groups = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.client_groups:
            self.client_groups[client_id] = []
        self.client_groups[client_id].append(websocket)

    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.client_groups and websocket in self.client_groups[client_id]:
            self.client_groups[client_id].remove(websocket)
            if not self.client_groups[client_id]:
                del self.client_groups[client_id]

    async def broadcast(self, message: str, target_client_id: str = None):
        if target_client_id:
            # 특정 클라이언트 그룹에만 데이터 전송
            if target_client_id in self.client_groups:
                for connection in self.client_groups[target_client_id]:
                    await connection.send_text(message)
        else:
            # 모든 클라이언트에게 데이터 전송
            for group in self.client_groups.values():
                for connection in group:
                    await connection.send_text(message)