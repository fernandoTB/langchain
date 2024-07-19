from __future__ import annotations
import json
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Type
from pydantic import Field
from pydantic.v1 import BaseModel
from langchain_community.utilities.blip import BlipClientWrapper
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool


class MessageMimetypes(str, Enum):
    """The lime mimetypes used by Blip"""
    PLAIN_TEXT: str = "text/plain"
    MEDIA_LINK: str = "application/vnd.lime.media-link+json"
    MENU: str = "application/vnd.lime.select+json"
    IMAGE: str = "image/jpeg"
    REPLY: str = "reply"
    BUTTON: str = "button"
    COLLECTION: str = "application/vnd.lime.collection+json"
    CONTAINER: str = "application/vnd.lime.container+json"


class CollectionPlainTextItem(BaseModel):
    """Model for a plain text item in a collection message."""
    type: Literal[MessageMimetypes.PLAIN_TEXT] = Field(
        default=MessageMimetypes.PLAIN_TEXT,
        description=f'The plain text content type, must always be: {MessageMimetypes.PLAIN_TEXT.value}.'
    )
    value: str = Field(description=f'The plain text content of the text item for collection message.')


class ImageContent(BaseModel):
    """Model for image content."""
    title: str
    text: str
    type: Literal[MessageMimetypes.IMAGE] = Field(
        default=MessageMimetypes.IMAGE,
        description=f'The image content type, must always be: {MessageMimetypes.IMAGE.value}.'
    )
    uri: str
    aspectRatio: Literal["1:1"]


class CollectionStaticImageItem(BaseModel):
    """Model for a static image item in a collection message."""
    type: Literal[MessageMimetypes.MEDIA_LINK] = Field(
        default=MessageMimetypes.MEDIA_LINK,
        description=f'The image item type for collection message, must always be: {MessageMimetypes.MEDIA_LINK.value}.'
    )
    value: ImageContent = Field(description=f'The image content of the image item for collection message.')


class MenuOption(BaseModel):
    """Model for a single menu option."""
    order: int = Field(description=f'The order number of the menu option.')
    text: str = Field(description=f'The text of the menu option.')


class MenuContent(BaseModel):
    """Model for the content of a menu."""
    text: str = Field(description=f'The text header of the menu.')
    options: list[MenuOption] = Field(description=f'The array of options of the menu.')


class CollectionMenuItem(BaseModel):
    """Model for a menu item in a collection message."""
    type: Literal[MessageMimetypes.MENU] = Field(
        default=MessageMimetypes.MENU,
        description=f'The menu item type, must always be {MessageMimetypes.MENU.value}'
    )
    value: MenuContent = Field(description=f'The content of the menu item.')


ContainerItem = Annotated[
    CollectionPlainTextItem | CollectionStaticImageItem | CollectionMenuItem,
    Field(discriminator='type')
]


class ContainerContent(BaseModel):
    """Model for the content of a container message."""
    itemType: Literal[MessageMimetypes.CONTAINER] = Field(
        default=MessageMimetypes.CONTAINER,
        description=f'The container array type, must always be {MessageMimetypes.CONTAINER.value}'
    )
    items: list[ContainerItem] = Field(description='The content array of the container.')


class LimeCollectionMessage(BaseModel):
    """Model for sending a collection type message with multiple content."""
    type: Literal[MessageMimetypes.COLLECTION] = Field(
        default=MessageMimetypes.COLLECTION,
        description=f'The collection message lime type, must always be {MessageMimetypes.COLLECTION.value}'
    )
    content: ContainerContent


class LimePlainTextMessage(BaseModel):
    """Model for sending a plain text type message."""
    type: Literal[MessageMimetypes.PLAIN_TEXT] = Field(
        default=MessageMimetypes.PLAIN_TEXT,
        description=f'The collection message lime type, must always be {MessageMimetypes.PLAIN_TEXT.value}'
    )
    content: str = Field(description='The text message content')


class LimeMediaLinkMessage(BaseModel):
    """Model for sending an image type message."""
    type: Literal[MessageMimetypes.MEDIA_LINK] = Field(
        default=MessageMimetypes.MEDIA_LINK,
        description=f'The collection message lime type, must always be {MessageMimetypes.MEDIA_LINK.value}'
    )
    content: ImageContent = Field(description='The image message content')


class BlipMessageSchema(BaseModel):
    """Schema for Blip messages."""
    message: LimeCollectionMessage | LimePlainTextMessage | LimeMediaLinkMessage = Field(
        description='The message to be sent to the user.',
        discriminator='type'
    )


class SendDynamicContentTool(BaseTool):
    """Tool for sending dynamic content as chat messages."""
    name: str = "send_dynamic_content"
    description: str = (
        "Send a chat message to user using dynamic content."
        "Useful for keeping good conversational experience, use it when you need to send formatted "
        "chat messages."
    )
    args_schema: Optional[Type[BaseModel]] = BlipMessageSchema
    client: BlipClientWrapper
    return_direct: bool = True

    def _run(
            self,
            message: LimeCollectionMessage,
            config: RunnableConfig,
            run_manager: Optional[CallbackManagerForToolRun]
    ) -> str:
        """
        Synchronously run the tool to send a message.

        Args:
            message (LimeCollectionMessage): The message to be sent.
            config (RunnableConfig): Configuration for the run.
            run_manager (Optional[CallbackManagerForToolRun]): Optional callback manager for the tool run.

        Returns:
            str: Result of the message sending operation.
        """
        try:
            from lime_python import Identity, Message as BlipMessage
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import lime_python python package. "
                "Please install it with `pip install lime-python`."
            )

        user_id = config['configurable'].get('blip_user_id')
        if user_id is None:
            raise ValueError(
                "Missing blip_user_id key in config['configurable'] "
                "When using via .invoke() or .stream(), pass in a config; "
                "e.g., chain.invoke(..., config={'configurable': {'blip_user_id': HERE}})"
            )

        self.client.send_message(
            message=BlipMessage.from_json({"to": user_id, **json.loads(message.json())})
        )
        return "message sent"

    async def _arun(
            self,
            message: LimeCollectionMessage,
            config: RunnableConfig,
            run_manager: Optional[CallbackManagerForToolRun]
    ) -> str:
        """
        Asynchronously run the tool to send a message.

        Args:
            message (LimeCollectionMessage): The message to be sent.
            config (RunnableConfig): Configuration for the run.
            run_manager (Optional[CallbackManagerForToolRun]): Optional callback manager for the tool run.

        Returns:
            str: Result of the message sending operation.
        """
        try:
            from lime_python import Identity, Message as BlipMessage
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import lime_python python package. "
                "Please install it with `pip install lime-python`."
            )

        user_id = config['configurable'].get('blip_user_id')
        if user_id is None:
            raise ValueError(
                "Missing blip_user_id key in config['configurable'] "
                "When using via .invoke() or .stream(), pass in a config; "
                "e.g., chain.invoke(..., config={'configurable': {'blip_user_id': HERE}})"
            )

        content = json.loads(message.json())
        content = {"to": user_id, **content}
        self.client.send_message(
            message=BlipMessage.from_json(content)
        )
        return "Message successfully sent !"
