import reflex as rx
import json
import aiofiles
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional
from autogen_agentchat.messages import TextMessage

# External websocket endpoint and user input function
async def user_input_ws(websocket: WebSocket, client_token: str):
    await websocket.accept()
    active_connections[client_token] = websocket
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_json()
    except Exception as e:
        print(f"Websocket error: {str(e)}")
    finally:
        if client_token in active_connections:
            del active_connections[client_token]

async def _user_input(client_token: str, prompt: str, cancellation_token: Optional[any] = None) -> str:
    """External function to handle user input via websocket"""
    if client_token in active_connections:
        websocket = active_connections[client_token]
        try:
            data = await websocket.receive_json()
            return data["content"]
        except Exception as e:
            print(f"Error receiving input: {str(e)}")
    return ""

def setup(app: rx.App):
    app.api.websocket("/ws/user_input/{client_token}", user_input_ws)