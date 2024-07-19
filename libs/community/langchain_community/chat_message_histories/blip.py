from __future__ import annotations
import asyncio
import json
from typing import Any, Optional, TYPE_CHECKING, Union
from langchain_community.utilities.blip import BlipClientWrapper
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, convert_to_messages
from langchain_core.runnables import ConfigurableFieldSpec, Runnable, RunnableWithMessageHistory
from langchain_core.runnables.history import MessagesOrDictWithMessages

if TYPE_CHECKING:
    from lime_python import Identity


class BlipMessageHistory(BaseChatMessageHistory):
    """
    Chat message history stored using Blip User's Context variables.

    Attributes:
        client (BlipClientWrapper): The client wrapper for interacting with Blip.
        session_id (Identity): The session identifier for the user.
        variable_name (str): The name of the variable where chat history is stored.
    """

    def __init__(self, client: BlipClientWrapper, session_id: Identity, variable_name: str = "chat_history"):
        """
        Initialize the BlipMessageHistory.

        Args:
            client (BlipClientWrapper): The client wrapper for interacting with Blip.
            session_id (Identity): The session identifier for the user.
            variable_name (str, optional): The name of the variable where chat history is stored. Defaults to "chat_history".
        """
        self.client = client
        self.session_id = session_id
        self.variable_name = variable_name
        self.loop = asyncio.get_event_loop()

    @property
    def messages(self):
        """
        Retrieve all messages from Blip.

        Returns:
            list[BaseMessage]: List of chat messages.
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.run_until_complete(self.aget_messages())
        return self.get_messages()

    def get_messages(self) -> list[BaseMessage]:
        """
        Retrieve all messages from Blip synchronously.

        Returns:
            list[BaseMessage]: List of chat messages.
        """
        return asyncio.run_coroutine_threadsafe(self.aget_messages(), self.loop).result()

    async def aget_messages(self) -> list[BaseMessage]:
        """
        Retrieve all messages from Blip asynchronously.

        Returns:
            list[BaseMessage]: List of chat messages.
        """
        try:
            from lime_python import Command, CommandStatus, ReasonCode, CommandMethod
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import lime_python python package. "
                "Please install it with `pip install lime-python`."
            )
        response = await self.client.process_command_async(
            command=Command(
                method=CommandMethod.GET,
                uri=f"/contexts/{self.session_id}/{self.variable_name}"
            )
        )
        if (
                response.status == CommandStatus.FAILURE
                and response.reason["code"] == ReasonCode.COMMAND_RESOURCE_NOT_FOUND
        ):
            return []
        if response.status == CommandStatus.FAILURE:
            raise Exception(f"Failed requesting context variable {self.variable_name} for {self.session_id}")
        return convert_to_messages(json.loads(response.resource))

    def add_message(self, message: BaseMessage) -> None:
        """
        Add new message to the store.

        Args:
            message (BaseMessage): A BaseMessage object to store.
        """
        asyncio.run_coroutine_threadsafe(self.aadd_message(message), self.loop).result()

    async def aadd_message(self, message: BaseMessage) -> None:
        """
        Add a Message object to the store asynchronously.

        Args:
            message (BaseMessage): A BaseMessage object to store.
        """
        history = await self.aget_messages()
        history.append(message)
        await self.set_context_async(value=json.dumps([m.dict() for m in history]))

    async def set_context_async(self, value: str) -> None:
        """
        Set the context variable value asynchronously.

        Args:
            value (str): The value to set for the context variable.
        """
        try:
            from lime_python import Command, CommandStatus, CommandMethod
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import lime_python python package. "
                "Please install it with `pip install lime-python`."
            )
        response = await self.client.process_command_async(
            command=Command(
                method=CommandMethod.SET,
                uri=f"/contexts/{self.session_id}/{self.variable_name}",
                type_n="text/plain",
                resource=value,
            )
        )
        if response.status == CommandStatus.FAILURE:
            raise Exception(f"Failed updating context variable {self.variable_name} for {self.session_id}")
        return

    def clear(self) -> None:
        """
        Clear the chat history.
        """
        asyncio.run_coroutine_threadsafe(self.aclear(), self.loop).result()

    async def aclear(self) -> None:
        """
        Clear the chat history asynchronously.
        """
        await self.set_context_async(value=json.dumps([]))


class BlipRunnableWithMessageHistory(RunnableWithMessageHistory):
    """
    Runnable with message history that utilizes Blip for storing chat messages.

    Args:
        client (BlipClientWrapper): The client wrapper for interacting with Blip.
        runnable (Union[Runnable, LanguageModelLike]): The runnable or language model to use.
        input_messages_key (Optional[str], optional): The key for input messages. Defaults to 'input_message'.
        history_messages_key (Optional[str], optional): The key for history messages. Defaults to 'chat_history'.
        **kwargs (Any): Additional keyword arguments.
    """

    def __init__(
            self,
            client: BlipClientWrapper,
            runnable: Union[
                Runnable[
                    Union[MessagesOrDictWithMessages],
                    Union[str, BaseMessage, MessagesOrDictWithMessages],
                ],
                LanguageModelLike,
            ],
            *,
            input_messages_key: Optional[str] = 'input_message',
            history_messages_key: Optional[str] = 'chat_history',
            **kwargs: Any
    ):
        """
        Initialize the BlipRunnableWithMessageHistory.

        Args:
            client (BlipClientWrapper): The client wrapper for interacting with Blip.
            runnable (Union[Runnable, LanguageModelLike]): The runnable or language model to use.
            input_messages_key (Optional[str], optional): The key for input messages. Defaults to 'input_message'.
            history_messages_key (Optional[str], optional): The key for history messages. Defaults to 'chat_history'.
            **kwargs (Any): Additional keyword arguments.
        """

        def get_session_history(session_id: str):
            return BlipMessageHistory(client=client, session_id=session_id)

        super().__init__(
            runnable=runnable,
            get_session_history=get_session_history,
            input_messages_key=input_messages_key,
            history_messages_key=history_messages_key,
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="blip_user_id",
                    annotation=str,
                    name="Blip User ID",
                    description="Unique identifier for a blip user.",
                    default="",
                    is_shared=True
                )
            ],
            **kwargs
        )
