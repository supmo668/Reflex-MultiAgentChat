"""The main Chat app."""

import reflex as rx
import logging
from fastapi import WebSocket, WebSocketDisconnect

from .components.agent_chat import chat, action_bar
from .websocket_handlers import input_ws_manager

# Configure logging
logger = logging.getLogger(__name__)

def index() -> rx.Component:
    """The main app."""
    return rx.vstack(
        rx.heading("AutoGen Chat", margin="1em"),
        chat(),
        action_bar(),
        background_color=rx.color("mauve", 1),
        color=rx.color("mauve", 12),
        min_height="100vh",
        align_items="stretch",
        spacing="0",
    )

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted in main route handler")
    
    try:
        while True:
            # Wait for a message
            data = await websocket.receive_json()
            logger.info(f"Received WebSocket message: {data}")
            
            # Check if the message contains an input request
            if "request_id" in data and "input" in data:
                request_id = data["request_id"]
                user_input = data["input"]
                
                logger.debug(f"Processing input for request_id: {request_id}")
                await input_ws_manager.process_message(request_id, user_input)
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected in main route handler")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        logger.exception("Detailed exception information:")


# Add state and page to the app.
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="violet",
    ),
)
app.add_page(index)
app.api.websocket("/ws/input", websocket_endpoint)