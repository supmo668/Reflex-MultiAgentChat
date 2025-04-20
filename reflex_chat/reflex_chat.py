"""The main Chat app."""

import reflex as rx
import logging

from .components.agent_chat import chat, action_bar

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


# Add state and page to the app.
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="violet",
    ),
)
app.add_page(index)