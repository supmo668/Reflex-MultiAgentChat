import reflex as rx
from reflex_chat.states.chat_state import ChatState, QA

message_style = dict(
    display="inline-block",
    padding="1em",
    border_radius="8px",
    max_width=["30em", "30em", "50em", "50em", "50em", "50em"]
)

def message(qa: QA) -> rx.Component:
    """A single message in the chat.
    
    Args:
        qa: The message to display.
        
    Returns:
        A component displaying the message.
    """
    # Determine message style based on type and source
    bg_color = rx.cond(
        qa.type == "error",
        rx.color("red", 4),
        rx.cond(
            qa.type == "system",
            rx.color("blue", 4),
            rx.cond(
                qa.source == "user",
                rx.color("mauve", 4),
                rx.color("accent", 4)
            )
        )
    )
    
    text_color = rx.cond(
        qa.type == "error",
        rx.color("red", 12),
        rx.cond(
            qa.type == "system",
            rx.color("blue", 12),
            rx.cond(
                qa.source == "user",
                rx.color("mauve", 12),
                rx.color("accent", 12)
            )
        )
    )
    
    return rx.box(
        rx.box(
            rx.markdown(
                qa.content,
                background_color=bg_color,
                color=text_color,
                **message_style,
            ),
            text_align=rx.cond(
                qa.source == "user",
                "right",
                "left"
            ),
            margin_top="1em",
        ),
        width="100%",
    )

def chat() -> rx.Component:
    """The main chat component."""
    return rx.vstack(
        rx.box(
            rx.foreach(
                ChatState.messages,
                message
            ),
            width="100%",
            overflow_y="auto",
            id="chat-container",
        ),
        # Cancel button when chat is ongoing
        rx.cond(
            ChatState.chat_ongoing,
            rx.button(
                "Cancel Chat",
                on_click=ChatState.cancel_chat,
                color_scheme="red",
                margin="1em",
            ),
        ),
        py="8",
        flex="1",
        width="100%",
        max_width="50em",
        padding_x="4px",
        align_self="center",
        overflow="hidden",
        padding_bottom="5em",
        on_mount=ChatState.scroll_to_bottom,
    )

def action_bar() -> rx.Component:
    """The action bar for sending messages."""
    return rx.center(
        rx.vstack(
            rx.hstack(
                rx.input(
                    placeholder="Type something...",
                    value=ChatState.current_message,
                    on_change=ChatState.on_message_input,
                    width="100%",
                ),
                rx.button(
                    "Send",
                    on_click=ChatState.submit_message,
                    is_disabled=~ChatState.input_enabled,
                ),
                width="100%",
                max_width="50em",
                padding_x="4px",
            ),
            rx.text(
                "Powered by Reflex + AutoGen",
                text_align="center",
                font_size=".75em",
                color=rx.color("mauve", 10),
            ),
            align_items="center",
        ),
        position="sticky",
        bottom="0",
        left="0",
        padding_y="16px",
        backdrop_filter="auto",
        backdrop_blur="lg",
        border_top=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        align_items="stretch",
        width="100%",
    )
