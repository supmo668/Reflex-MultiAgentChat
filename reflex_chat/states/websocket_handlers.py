import asyncio
from typing import Dict, Optional
import logging
from fastapi import WebSocket, WebSocketDisconnect

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class InputWebSocketManager:
    """Manages WebSocket connections for user input requests"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = InputWebSocketManager()
        return cls._instance
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, request_id: str):
        """Accept a WebSocket connection"""
        await websocket.accept()
        logger.info(f"[WS] WebSocket connection accepted for request {request_id}")
        
        async with self._lock:
            self.active_connections[request_id] = websocket
            # Create a future to be resolved when input is received
            self.pending_requests[request_id] = asyncio.Future()
            logger.debug(f"[WS] Created future for request {request_id} in connect method")
            logger.debug(f"[WS] Current pending requests: {list(self.pending_requests.keys())}")
        
        logger.debug(f"WebSocket connected for request {request_id}")
    
    async def disconnect(self, request_id: str) -> bool:
        """Disconnect a pending request.
        
        Args:
            request_id: The request ID to disconnect
            
        Returns:
            True if the request was found and disconnected, False otherwise
        """
        logger.info(f"[WS] Disconnecting request {request_id}")
        
        async with self._lock:
            if request_id in self.pending_requests:
                # Cancel the future
                future = self.pending_requests[request_id]
                if not future.done():
                    logger.debug(f"[WS] Cancelling future for request {request_id}")
                    future.cancel()
                    logger.debug(f"[WS] Future cancelled")
                else:
                    logger.debug(f"[WS] Future already done for request {request_id}")
                
                # Clean up
                logger.debug(f"[WS] Removing request {request_id} from pending_requests")
                del self.pending_requests[request_id]
                logger.debug(f"[WS] Request removed")
                
                # Also remove from active connections if present
                if request_id in self.active_connections:
                    logger.debug(f"[WS] Removing request {request_id} from active_connections")
                    del self.active_connections[request_id]
                    
                return True
            else:
                logger.warning(f"[WS] Request {request_id} not found in pending_requests")
                logger.debug(f"[WS] Current pending requests: {list(self.pending_requests.keys())}")
            
            return False
    
    async def process_message(self, request_id: str, message: str):
        """Process a message received from the WebSocket"""
        logger.info(f"[WS] Processing message for request {request_id}: {message[:50] if message else None}...")
        
        async with self._lock:
            if request_id in self.pending_requests:
                future = self.pending_requests[request_id]
                if not future.done():
                    logger.debug(f"[WS] Setting result for request {request_id}")
                    future.set_result(message)
                    logger.debug(f"[WS] Result set successfully")
                else:
                    logger.warning(f"[WS] Future already done for request {request_id}")
            else:
                logger.warning(f"[WS] No pending request found for {request_id}")
                logger.debug(f"[WS] Current pending requests: {list(self.pending_requests.keys())}")
    
    async def wait_for_input(self, request_id: str, timeout: float = 3600) -> Optional[str]:
        """Wait for input for a specific request ID.
        
        Args:
            request_id: The request ID to wait for
            timeout: Timeout in seconds
            
        Returns:
            The user input, or None if timed out or cancelled
        """
        logger.info(f"[WS] Waiting for input for request {request_id}")
        
        # Create a future to wait for the input if it doesn't exist
        async with self._lock:
            # Check if request already exists
            if request_id in self.pending_requests:
                logger.warning(f"[WS] Request ID {request_id} already exists in pending_requests")
                # Get the existing future
                future = self.pending_requests[request_id]
                if future.done():
                    logger.debug(f"[WS] Future already done for request {request_id}, creating new one")
                    future = asyncio.Future()
                    self.pending_requests[request_id] = future
            else:
                # Create a new future
                logger.debug(f"[WS] Creating new future for request {request_id}")
                future = asyncio.Future()
                self.pending_requests[request_id] = future
                
            logger.debug(f"[WS] Current pending requests: {list(self.pending_requests.keys())}")
        
        try:
            # Wait for the future to be resolved
            logger.debug(f"[WS] Starting wait_for on future for request {request_id}")
            result = await asyncio.wait_for(future, timeout)
            logger.info(f"[WS] Received input for request {request_id}: {result[:50] if result else None}...")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"[WS] Timed out waiting for input for request {request_id}")
            return None
        except asyncio.CancelledError:
            logger.warning(f"[WS] Cancelled waiting for input for request {request_id}")
            raise
        except Exception as e:
            logger.error(f"[WS] Error waiting for input for request {request_id}: {str(e)}")
            logger.exception("Detailed exception information:")
            raise
        finally:
            # Clean up
            async with self._lock:
                if request_id in self.pending_requests and self.pending_requests[request_id] == future:
                    logger.debug(f"[WS] Cleaning up request {request_id} in finally block")
                    del self.pending_requests[request_id]
                else:
                    logger.debug(f"[WS] Not cleaning up request {request_id} as future has changed")

# Create a singleton instance
input_ws_manager = InputWebSocketManager.get_instance()

async def handle_input_websocket(websocket: WebSocket, request_id: str):
    """Handle WebSocket connections for user input"""
    await input_ws_manager.connect(websocket, request_id)
    
    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_json()
            message = data.get("message", "")
            await input_ws_manager.process_message(request_id, message)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for request {request_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")
    finally:
        await input_ws_manager.disconnect(request_id)
