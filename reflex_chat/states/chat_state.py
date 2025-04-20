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

from .team_manager import TeamManager

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
    # The current message being typed
    current_message: str = ""
    
    # The team state as a dictionary
    team_state: dict = {}
    
    # Control whether the input box is enabled
    input_enabled: bool = True

    # WebSocket request tracking
    current_request_id: Optional[str] = None
    
    # Team instance ID to ensure we use the same team
    team_id: str = str(uuid4())
    
    # Helper method to reset team ID
    def _reset_team_id(self) -> str:
        """Reset the team ID to a new UUID and return it."""
        new_id = str(uuid4())
        self.team_id = new_id
        logger.debug(f"Reset team_id to new value: {new_id}")
        return new_id

    async def get_team(self):
        """Get or create the team of agents using the TeamManager.
        
        Returns:
            A tuple of (team, termination) where team is the RoundRobinGroupChat instance
            and termination is the ExternalTermination instance for cancellation.
        """
        # Get the team from the manager
        team_manager = TeamManager.get_instance()
        
        # Ensure we have a valid team_id
        if not self.team_id or len(self.team_id) < 5:  # Basic validation
            async with self:
                old_id = self.team_id
                self._reset_team_id()
                logger.debug(f"Invalid team_id in get_team: {old_id}, reset to: {self.team_id}")
        
        # Define a callback to update UI state when messages are received
        async def update_state_callback(prompt: str, is_user_input: bool = False):
            if prompt.strip():
                # Check for "Enter your response" prompts
                if "Enter your response" in prompt:
                    logger.debug("Detected 'Enter your response' prompt, enabling input without adding message")
                    async with self:
                        self.processing = False
                        self.input_enabled = True
                    return
                
                async with self:
                    message_type = "system"
                    if is_user_input:
                        self.processing = False
                    # Add system message to UI
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
        """Submit a message to the team.
        Don't add message here, add it in start_chat at `run_stream` iteration
        """
        logger.debug("=== SUBMIT_MESSAGE CALLED ===")
        if not self.current_message:
            logger.debug("No current message, returning early")
            return
            
        # Get the message and clear input
        message = self.current_message
        logger.debug(f"Processing message: '{message[:30]}...'")
        
        # Add user message to UI first to ensure it's visible
        async with self:
            self.current_message = ""
            self.processing = True  # Set processing flag
            self.input_enabled = False  # Disable input while processing
            logger.debug(f"State updated: processing={self.processing}, input_enabled={self.input_enabled}")
            
        # Get team manager instance
        team_manager = TeamManager.get_instance()
        
        # After cancellation, we should have a fresh team_id from _reset_team_id()
        # Log the current team_id for debugging
        logger.debug(f"Current team_id: {self.team_id}")
        
        # Always start a new chat after cancellation
        # We can detect this by checking if the team exists
        is_running = False
        try:
            is_running = await team_manager.is_team_running(self.team_id)
            logger.debug(f"Team is_running: {is_running}")
        except Exception as e:
            logger.error(f"Error checking if team is running: {str(e)}")
            is_running = False
        
        if not is_running:
            # Force start a new chat if we're after cancellation or no team is running
            # This ensures we always get a fresh start
            logger.debug(f"Starting new chat with message: '{message[:30]}...'")
            logger.debug("Yielding to start_chat event")
            try:
                # We need to ensure a fresh team_id
                async with self:
                    old_id = self.team_id
                    self._reset_team_id()
                    logger.debug(f"Reset team_id from {old_id} to {self.team_id} before starting chat")
                
                yield ChatState.start_chat(message)
                logger.debug("Successfully yielded to start_chat")
                return
            except Exception as e:
                logger.error(f"Error yielding to start_chat: {str(e)}")
                
        # If team is running but not waiting for input, it might be in an error state
        # Let's cancel it and start a new chat
        is_waiting = await team_manager.is_team_waiting_for_input(self.team_id)
        logger.debug(f"Team is_waiting: {is_waiting}")
        if not is_waiting:
            logger.debug(f"Team {self.team_id} is running but not waiting for input, cancelling it")
            await team_manager.cancel_team(self.team_id)
            # Generate a new team ID
            async with self:
                self._reset_team_id()
                logger.debug(f"Reset team_id to: {self.team_id}")
            # Start a new chat
            logger.debug(f"Starting new chat after cancelling non-responsive team")
            yield ChatState.start_chat(message)
            return
        
        # If we reach here, the team is running and waiting for input
        logger.debug(f"Team {self.team_id} is waiting for input, submitting message: {message[:30]}...")

        # Submit the message to the waiting team
        success = await team_manager.submit_user_input(self.team_id, message)
        if success:
            logger.debug(f"Successfully submitted message to team {self.team_id}")
            return
        else:
            logger.error(f"Failed to submit message to team {self.team_id}")
            # Continue to start a new chat as fallback
            logger.debug(f"Starting new chat after failed submission")
            yield ChatState.start_chat(message)

    @rx.event(background=True)
    async def start_chat(self, initial_message: str):
        """Start or continue a chat conversation."""
        logger.debug(f"=== START_CHAT CALLED with message: '{initial_message[:30]}...' ===")
        logger.debug(f"Current state: team_id={self.team_id}, processing={self.processing}, input_enabled={self.input_enabled}")
        
        # Force cleanup of any existing team with this ID to ensure a fresh start
        team_manager = TeamManager.get_instance()
        try:
            is_running = await team_manager.is_team_running(self.team_id)
            if is_running:
                logger.debug(f"Found existing team with ID {self.team_id}, removing it before starting new chat")
                await team_manager.remove_team(self.team_id)
        except Exception as e:
            logger.error(f"Error cleaning up existing team: {str(e)}")
        
        try:
            # Mark as processing
            async with self:
                self.processing = True
            
            # Ensure we have a valid team_id
            if not self.team_id or len(self.team_id) < 5:  # Basic validation
                async with self:
                    old_id = self.team_id
                    self._reset_team_id()
                    logger.debug(f"Invalid team_id: {old_id}, reset to: {self.team_id}")
            
            # Get or create team
            logger.debug(f"Getting team with ID: {self.team_id}")
            team, termination = await self.get_team()
            logger.debug(f"Got team instance with ID: {self.team_id}")
            
            # Create initial message if provided
            logger.debug(f"Running stream with initial message: '{initial_message[:30]}...'")
            stream = team.run_stream(task=initial_message)
            logger.debug("Stream created, starting to process events")
            
            # Process events
            event_count = 0
            async for event in stream:
                event_count += 1
                logger.debug(f"Processing event #{event_count}: {type(event).__name__}")
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

                        
                    # Check for "Enter your response" or similar prompts
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
                    
                    # No need to track seen messages anymore
                    
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
                self.input_enabled = True
                
            # Clean up team using TeamManager
            team_manager = TeamManager.get_instance()
            await team_manager.remove_team(self.team_id)
            
            logger.debug("Finished chat")
    
    @rx.event(background=True)
    async def cancel_chat(self):
        """Cancel the current chat.
        
        This will gracefully terminate the current chat using ExternalTermination
        and ensure the next message starts a new chat without any errors.
        """
        logger.debug("=== CANCEL_CHAT CALLED ===")
        logger.debug(f"Current state before cancellation: team_id={self.team_id}, processing={self.processing}, input_enabled={self.input_enabled}")
        
        # Use TeamManager to cancel the team
        team_manager = TeamManager.get_instance()
        
        # Store the old team ID for cancellation
        team_id_to_cancel = self.team_id
        
        # Generate a new team ID before cancellation to ensure a clean slate
        async with self:
            # Generate a new team ID for the next chat
            new_team_id = self._reset_team_id()
            logger.debug(f"Generated new team_id: {new_team_id} (old: {team_id_to_cancel})")
            
            # Reset UI state
            self.processing = False
            self.input_enabled = True
            
            # Clear current request ID
            self.current_request_id = None
            
        # Now cancel the old team - with enhanced error handling
        try:
            # First check if the team is running
            is_running = await team_manager.is_team_running(team_id_to_cancel)
            logger.debug(f"Team {team_id_to_cancel} is_running: {is_running}")
            
            if is_running:
                logger.debug(f"Cancelling team {team_id_to_cancel}")
                # Cancel the team - this should set the termination flag and remove the team
                await team_manager.cancel_team(team_id_to_cancel)
                
                # Wait a moment for cancellation to take effect
                await asyncio.sleep(0.5)
                
                # Double-check the team was actually removed
                still_running = await team_manager.is_team_running(team_id_to_cancel)
                if still_running:
                    logger.error(f"Team {team_id_to_cancel} still running after cancellation, forcing removal")
                    # Force remove the team
                    await team_manager.remove_team(team_id_to_cancel)
                    
                    # Final check
                    final_check = await team_manager.is_team_running(team_id_to_cancel)
                    logger.debug(f"Final check - team {team_id_to_cancel} still running: {final_check}")
            else:
                logger.debug(f"Team {team_id_to_cancel} not running, no need to cancel")
        except Exception as e:
            logger.error(f"Error during team cancellation: {str(e)}")
            # Even if there's an error, try to force remove the team
            try:
                await team_manager.remove_team(team_id_to_cancel)
                logger.debug(f"Force removed team {team_id_to_cancel} after error")
            except Exception as e2:
                logger.error(f"Error during forced team removal: {str(e2)}")
        
        # Final state check
        logger.debug(f"Chat cancelled and state reset. New team_id={self.team_id}, processing={self.processing}, input_enabled={self.input_enabled}")

    @rx.event(background=True)
    async def pause_chat(self):
        """Pause the current chat by saving its state and removing the team.
        
        This allows the user to start a new chat while preserving the state of the current one.
        """
        team_manager = TeamManager.get_instance()
        
        # Check if team is running before attempting to pause
        if await team_manager.is_team_running(self.team_id):
            # Save the team state
            state = await team_manager.save_team_state(self.team_id)
            
            # Store the state
            if state:
                async with self:
                    self.team_state = state
                    logger.debug(f"Team state saved: {len(state)} bytes")
            
            # Now remove the team to free up resources
            await team_manager.remove_team(self.team_id)
            
            # Add pause message
            async with self:
                pause_message = QA(
                    content="Chat paused. You can continue with a new conversation.",
                    source="system",
                    type="system"
                )
                self.messages = self.messages + [pause_message]
        
        # Reset processing state regardless of whether team was running
        async with self:
            self.processing = False
            self.current_request_id = None
            self.input_enabled = True  # Re-enable input after pausing
    
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
