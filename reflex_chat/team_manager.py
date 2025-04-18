"""Team manager for handling AutoGen teams outside of Reflex state."""

import os
import json
import logging
import asyncio
from uuid import uuid4
from typing import Dict, Optional, Tuple, Callable, Awaitable, Any, List

import yaml
import aiofiles
import reflex as rx

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import ExternalTermination
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient

from .websocket_handlers import input_ws_manager

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Constants
MODEL_CONFIG_PATH = "model_config.yaml"
STATE_PATH = "team_state.json"

class TeamManager:
    """Manages AutoGen teams outside of Reflex state."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = TeamManager()
        return cls._instance
    
    def __init__(self):
        """Initialize the team manager."""
        self._teams: Dict[str, Tuple[RoundRobinGroupChat, ExternalTermination]] = {}
        self._lock = asyncio.Lock()
        self._current_request_ids: Dict[str, Optional[str]] = {}
        self._messages: Dict[str, List[Any]] = {}
        self._callbacks: Dict[str, Callable] = {}
        
    async def get_user_input(self, team_id: str, prompt: str, cancellation_token: Optional[CancellationToken] = None) -> str:
        """Get human input through WebSocket.
        
        Args:
            team_id: The ID of the team requesting input
            prompt: The prompt to show to the user
            cancellation_token: Optional cancellation token
            
        Returns:
            The user input
            
        Raises:
            Exception: If no input is provided or timeout
            asyncio.CancelledError: If the input request is cancelled
        """
        try:
            logger.info(f"[TEAM {team_id}] Getting user input with prompt: {prompt[:50]}...")
            
            # Generate unique request ID
            request_id = str(uuid4())
            logger.debug(f"[TEAM {team_id}] Generated request ID: {request_id}")
            
            # Store the current request ID for this team
            self._current_request_ids[team_id] = request_id
            logger.debug(f"[TEAM {team_id}] Stored request ID in _current_request_ids")
            
            # Call the state update callback if provided
            if team_id in self._callbacks and self._callbacks[team_id]:
                logger.debug(f"[TEAM {team_id}] Calling state update callback")
                try:
                    await self._callbacks[team_id](prompt, True)
                    logger.debug(f"[TEAM {team_id}] State update callback completed successfully")
                except Exception as e:
                    logger.error(f"[TEAM {team_id}] Error in state update callback: {str(e)}")
            else:
                logger.debug(f"[TEAM {team_id}] No state update callback found")
                
            # Wait for input using WebSocket manager
            logger.debug(f"[TEAM {team_id}] Waiting for input via WebSocket with request ID: {request_id}")
            user_input = await input_ws_manager.wait_for_input(request_id)
            logger.debug(f"[TEAM {team_id}] wait_for_input returned: {user_input is not None}")
            
            # Check if input was received
            if not user_input:
                logger.error(f"[TEAM {team_id}] No input provided or timeout for request ID: {request_id}")
                raise Exception("No input provided or timeout")
                
            logger.info(f"[TEAM {team_id}] Received user input: {user_input[:50]}...")
            return user_input
            
        except asyncio.CancelledError:
            logger.warning(f"[TEAM {team_id}] Input request cancelled")
            raise
        except Exception as e:
            logger.error(f"[TEAM {team_id}] Error getting user input: {str(e)}")
            raise
        finally:
            # Clear the request ID
            if team_id in self._current_request_ids:
                logger.debug(f"[TEAM {team_id}] Clearing request ID {self._current_request_ids[team_id]}")
                self._current_request_ids[team_id] = None
    
    async def get_team(
        self, 
        team_id: str, 
        state_update_callback: Optional[Callable[[str, bool], Awaitable[None]]] = None
    ) -> Tuple[RoundRobinGroupChat, ExternalTermination]:
        """Get or create a team for the given ID.
        
        Args:
            team_id: Unique identifier for the team
            input_func: Function to get user input
            
        Returns:
            Tuple of (team, termination)
        """
        async with self._lock:
            # Check if team already exists
            if team_id in self._teams:
                logger.debug(f"Reusing existing team with ID {team_id}")
                return self._teams[team_id]
            
            logger.debug(f"Creating new team with ID {team_id}")
            
            # Create termination condition
            termination = ExternalTermination()
            
            # Load model config
            async with aiofiles.open(MODEL_CONFIG_PATH, "r") as file:
                model_config = yaml.safe_load(await file.read())
                logger.debug("Loaded model config")
            model_client = ChatCompletionClient.load_component(model_config)
            logger.debug("Created model client")
            
            # Create a wrapped input function that uses our get_user_input method
            # Termination is handled by the ExternalTermination condition
            async def wrapped_input_func(prompt: str, cancellation_token: Optional[CancellationToken] = None) -> str:
                try:
                    logger.debug(f"[TEAM {team_id}] wrapped_input_func called with prompt: {prompt[:50]}...")
                    # Call our internal get_user_input method
                    result = await self.get_user_input(team_id, prompt, cancellation_token)
                    logger.debug(f"[TEAM {team_id}] get_user_input completed successfully")
                    return result
                except asyncio.CancelledError:
                    logger.warning(f"[TEAM {team_id}] Input function cancelled")
                    # Set termination flag for graceful shutdown
                    termination.set()
                    raise
                except Exception as e:
                    logger.error(f"[TEAM {team_id}] Error in input function: {str(e)}")
                    # If any exception occurs, set termination flag for graceful shutdown
                    termination.set()
                    raise
            
            # Create agents
            assistant = AssistantAgent(
                name="assistant",
                model_client=model_client,
                system_message="You are a helpful assistant.",
            )
            
            yoda = AssistantAgent(
                name="yoda",
                model_client=model_client,
                system_message="Repeat the same message in the tone of Yoda.",
            )
            
            user_proxy = UserProxyAgent(
                name="user",
                input_func=wrapped_input_func,
            )
            
            # Create team with termination condition
            team = RoundRobinGroupChat(
                [assistant, yoda, user_proxy],
                termination_condition=termination
            )
            
            # Load state from file if it exists
            if os.path.exists(STATE_PATH):
                try:
                    async with aiofiles.open(STATE_PATH, "r") as file:
                        state = json.loads(await file.read())
                        await team.load_state(state)
                        logger.debug("Loaded team state from file")
                except Exception as e:
                    logger.error(f"Error loading team state: {e}")
            
            # Store the team and callback separately
            self._teams[team_id] = (team, termination)
            if state_update_callback:
                self._callbacks[team_id] = state_update_callback
            return self._teams[team_id]
    
    async def cancel_team(self, team_id: str) -> bool:
        """Cancel a team by ID.
        
        Args:
            team_id: ID of the team to cancel
            
        Returns:
            True if team was found and cancelled, False otherwise
        """
        async with self._lock:
            # Cancel any active WebSocket request
            if team_id in self._current_request_ids and self._current_request_ids[team_id]:
                request_id = self._current_request_ids[team_id]
                logger.debug(f"Cancelling WebSocket request {request_id} for team {team_id}")
                await input_ws_manager.disconnect(request_id)
                self._current_request_ids[team_id] = None
            
            # Set termination condition
            if team_id in self._teams:
                logger.debug(f"Cancelling team with ID {team_id}")
                _, termination = self._teams[team_id]
                termination.set()
                return True
            return False
    
    async def is_team_waiting_for_input(self, team_id: str) -> bool:
        """Check if a team is waiting for input.
        
        Args:
            team_id: The ID of the team to check
            
        Returns:
            True if the team is waiting for input, False otherwise
        """
        return team_id in self._current_request_ids and self._current_request_ids[team_id] is not None
        
    async def submit_user_input(self, team_id: str, user_input: str) -> bool:
        """Submit user input to a waiting team.
        
        Args:
            team_id: The ID of the team to submit input to
            user_input: The user input to submit
            
        Returns:
            True if the input was submitted successfully, False otherwise
        """
        if not self.is_team_waiting_for_input(team_id):
            logger.warning(f"[TEAM {team_id}] Team is not waiting for input")
            return False
            
        request_id = self._current_request_ids[team_id]
        if not request_id:
            logger.warning(f"[TEAM {team_id}] No request ID found")
            return False
            
        logger.debug(f"[TEAM {team_id}] Submitting user input for request ID: {request_id}")
        await input_ws_manager.process_message(request_id, user_input)
        return True
    
    async def remove_team(self, team_id: str) -> bool:
        """Remove a team by ID.
        
        Args:
            team_id: ID of the team to remove
            
        Returns:
            True if team was found and removed, False otherwise
        """
        async with self._lock:
            if team_id in self._teams:
                logger.debug(f"Removing team with ID {team_id}")
                del self._teams[team_id]
                
                # Clean up other resources
                if team_id in self._callbacks:
                    del self._callbacks[team_id]
                if team_id in self._current_request_ids:
                    del self._current_request_ids[team_id]
                    
                return True
            return False
    
    async def save_team_state(self, team_id: str) -> Optional[Dict[str, Any]]:
        """Save team state to a dictionary.
        
        Args:
            team_id: ID of the team to save
            
        Returns:
            Team state dictionary or None if team not found
        """
        async with self._lock:
            if team_id in self._teams:
                team, _ = self._teams[team_id]
                try:
                    state = await team.save_state()
                    return state
                except Exception as e:
                    logger.error(f"Error saving team state: {e}")
            return None
