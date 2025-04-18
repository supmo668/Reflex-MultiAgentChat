import os
import json
import logging
from typing import Dict, List, Optional, Any, Literal, Tuple
import asyncio
from uuid import uuid4

import reflex as rx
import aiofiles

from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, UserInputRequestedEvent
from autogen_agentchat.teams._group_chat._events import GroupChatError, GroupChatPause

from autogen_core import CancellationToken

from ..websocket_handlers import input_ws_manager
from ..team_manager import TeamManager

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Constants
HISTORY_PATH = "team_history.json"

class QA(rx.Base):
    """A question and answer pair."""
    content: str
    source: str
    type: str = "TextMessage"
    metadata: Dict[str, Any] = {}


class ChatState(rx.State):
    """The chat state."""
    
    # The list of messages in the current chat
    messages: list[QA] = []
    
    # Whether we are processing a message
    processing: bool = False
    chat_ongoing: bool = False
    # The current message being typed
    current_message: str = ""
    
    # The team state as a dictionary
    team_state: dict = {}
    
    # Keep track of waiting for input state
    waiting_for_input: bool = False
    
    # Control whether the input box is enabled
    input_enabled: bool = True
    
    # WebSocket request tracking
    current_request_id: Optional[str] = None
    
    # Team instance ID to ensure we use the same team
    team_id: str = str(uuid4())

    async def get_team(self):
        """Get or create the team of agents using the TeamManager.
        
        Returns:
            A tuple of (team, termination) where team is the RoundRobinGroupChat instance
            and termination is the ExternalTermination instance for cancellation.
        """
        # Get the team from the manager
        team_manager = TeamManager.get_instance()
        
        # Define a callback to update UI state when messages are received
        async def update_state_callback(prompt: str, is_user_input: bool = False):
            if prompt.strip():
                async with self:
                    message_type = "system"
                    if is_user_input:
                        self.waiting_for_input = True
                        self.processing = False
                    
                    system_message = QA(
                        source="system",
                        content=prompt,
                        type=message_type
                    )
                    self.messages = self.messages + [system_message]
        
        # Get or create the team
        return await team_manager.get_team(self.team_id, update_state_callback)

    @rx.event(background=True)
    async def submit_message(self):
        """Submit a message to the team."""
        if not self.current_message:
            return
            
        # Get the message and clear input
        message = self.current_message
        async with self:
            self.current_message = ""
            self.processing = True  # Set processing flag
            self.input_enabled = False  # Disable input while processing
        
        # Get team manager instance
        team_manager = TeamManager.get_instance()
        
        # Check if a team is already waiting for input
        if self.team_id and await team_manager.is_team_waiting_for_input(self.team_id):
            logger.debug(f"Team {self.team_id} is waiting for input, submitting message: {message}")
            
            # Add user message to the chat first
            async with self:
                user_message = QA(
                    content=message,
                    source="user",
                    type="TextMessage"
                )
                self.messages = self.messages + [user_message]
                logger.debug(f"Added user message: {user_message}")
            
            # Submit the message to the waiting team
            success = await team_manager.submit_user_input(self.team_id, message)
            if success:
                logger.debug(f"Successfully submitted message to team {self.team_id}")
                return
            else:
                logger.error(f"Failed to submit message to team {self.team_id}")
                # Continue to start a new chat as fallback
        
        # If we reach here, either there's no ongoing chat or the team is not waiting for input
        # Start a new chat
        logger.debug(f"Starting new chat with message: {message}")
        async with self:
            user_message = QA(
                content=message,
                source="user",
                type="TextMessage"
            )
            self.messages = self.messages + [user_message]
            self.input_enabled = False  # Disable input while chat is processing
            logger.debug(f"Added user message before starting chat")
        yield ChatState.start_chat(message)

    @rx.event(background=True)
    async def start_chat(self, initial_message: Optional[str] = None):
        """Start or continue a chat conversation."""
        logger.debug("Starting chat")
        
        try:
            # Mark as processing
            async with self:
                self.processing = True
                self.chat_ongoing = True
            
            # Get or create team
            team, termination = await self.get_team()
            logger.debug("Got team instance")
            
            # Create initial message if provided
            request = initial_message if initial_message else "Let's talk about sci-fi movies"
            stream = team.run_stream(task=request)
            
            # Process events
            async for event in stream:
                if isinstance(event, TaskResult):
                    logger.debug("Skipping TaskResult message")
                    continue
                
                logger.debug(f"Got message: {event}")
                
                if isinstance(event, UserInputRequestedEvent):
                    logger.debug("UserInputRequestedEvent in stream")
                    continue
                    
                if hasattr(event, "source") and hasattr(event, "content"):
                    # Skip duplicate user messages - the first message in the stream will be the user's initial message
                    # which we've already added to the chat when submitting
                    if event.source == "user" and event.content == initial_message:
                        logger.debug(f"Skipping duplicate user message: {event.content[:30]}...")
                        continue
                        
                    # Check for "Enter your response" or similar prompts and enable input
                    if event.source == "system" and "Enter your response" in event.content:
                        logger.debug("Detected 'Enter your response' prompt, enabling input")
                        async with self:
                            self.input_enabled = True
                            continue  # Skip adding this message to the chat
                    # Extract metadata if available
                    metadata = {}
                    if hasattr(event, "metadata"):
                        metadata = event.metadata
                    if hasattr(event, "models_usage") and event.models_usage:
                        # Convert RequestUsage dataclass to a serializable dict
                        try:
                            if hasattr(event.models_usage, "__dict__"):
                                metadata["models_usage"] = {
                                    "prompt_tokens": getattr(event.models_usage, "prompt_tokens", 0),
                                    "completion_tokens": getattr(event.models_usage, "completion_tokens", 0),
                                    "total_tokens": getattr(event.models_usage, "total_tokens", 0)
                                }
                            elif hasattr(event.models_usage, "prompt_tokens"):
                                metadata["models_usage"] = {
                                    "prompt_tokens": event.models_usage.prompt_tokens,
                                    "completion_tokens": event.models_usage.completion_tokens,
                                    "total_tokens": getattr(event.models_usage, "total_tokens", 0)
                                }
                        except Exception as e:
                            logger.warning(f"Could not serialize models_usage: {e}")
                    
                    async with self:
                        agent_message = QA(
                            content=event.content,
                            source=event.source,
                            type=getattr(event, "type", "TextMessage"),
                            metadata=metadata
                        )
                        self.messages = self.messages + [agent_message]
                        logger.debug(f"Added agent message: {agent_message}")
                    
                    # Save history
                    try:
                        async with aiofiles.open(HISTORY_PATH, "w") as file:
                            await file.write(json.dumps([msg.dict() for msg in self.messages]))
                            logger.debug("Saved updated chat history")
                    except TypeError as e:
                        logger.error(f"Error saving chat history: {e}")
            
            # Save team state using TeamManager
            team_manager = TeamManager.get_instance()
            state = await team_manager.save_team_state(self.team_id)
            logger.debug("Got team state")
            
            if state:
                async with self:
                    self.team_state = state
                
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            async with self:
                # Add error message to chat
                error_message = QA(
                    content=f"Error: {str(e)}",
                    source="system",
                    type="error"
                )
                self.messages = self.messages + [error_message]
            
        finally:
            # Mark as finished
            async with self:
                self.processing = False
                self.chat_ongoing = False
                self.input_enabled = True
                
            # Clean up team using TeamManager
            team_manager = TeamManager.get_instance()
            await team_manager.remove_team(self.team_id)
            
            logger.debug("Finished chat")
    
    @rx.event(background=True)
    async def cancel_chat(self):
        """Cancel the current chat."""
        # Use TeamManager to cancel the team
        # This will also cancel any active WebSocket requests
        team_manager = TeamManager.get_instance()
        await team_manager.cancel_team(self.team_id)
            
        async with self:
            # Add cancellation message
            cancel_message = QA(
                content="Chat cancelled",
                source="system",
                type="system"
            )
            self.messages = self.messages + [cancel_message]
            
            # Reset state
            self.waiting_for_input = False
            self.processing = False
            self.chat_ongoing = False
            self.current_request_id = None
            self.input_enabled = True  # Re-enable input after cancellation

    @rx.event
    def on_message_input(self, value: str):
        """Handle message input changes.
        
        Args:
            value: The new message value.
        """
        self.current_message = value
    
    @rx.event(background=True)
    async def scroll_to_bottom(self):
        """Scroll the chat container to the bottom."""
        logger.debug("Scrolling chat container to bottom")
        try:
            yield rx.call_script(
                """
                const container = document.getElementById('chat-container');
                if (container) {
                    container.scrollTop = container.scrollHeight;
                    console.log('Scrolled chat container to bottom');
                } else {
                    console.warn('Chat container not found');
                }
                """
            )
            logger.debug("Scroll script executed")
        except Exception as e:
            logger.error(f"Error scrolling to bottom: {str(e)}")
