from __future__ import annotations

from typing import TYPE_CHECKING
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from lime_python import Message as LimeMessage


def convert_message_from_blip(message: LimeMessage) -> HumanMessage:
    """Convert a lime protocol based Blip message to a LangChain message.

    Args:
        message: The lime protocol based message.

    Returns:
        The LangChain message.
    """
    content = message.content
    if isinstance(content, dict):
        if content['type'].startswith("image/"):
            content = [{"type": "image_url", "image_url": {"url": content["uri"]}}]
            if 'text' in content:
                content.append({"type": "text", "text": content["text"]})
            if 'title' in content:
                content.append({"type": "text", "text": content["title"]})
        else:
            content = str(content)
    return HumanMessage(content=content)
