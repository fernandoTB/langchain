from __future__ import annotations
import asyncio
import logging
import os
import multiprocessing
from typing import Any
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from enum import Enum
from langchain_community.adapters.blip import convert_message_from_blip
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableSerializable

if TYPE_CHECKING:
    from lime_python import Identity, Message as BlipMessage, Command

IMPORT_ERROR_LIME = ImportError(
    "Could not import lime_python python package. "
    "Please install it with `pip install lime-python`.")
IMPORT_ERROR_BLIP_SDK = ImportError(
    "Could not import blip_sdk python package. "
    "Please install it with `pip install blip-sdk`.")
IMPORT_ERROR_LIME_WEBSOCKET = ImportError(
    "Could not import lime_transport_websocket python package. "
    "Please install it with `pip install lime-transport-websocket`.")

logger = logging.getLogger(__name__)


def get_append_message_runner(runner):
    """
    Get a runnable that appends new messages to the last messages.

    Args:
        runner (RunnableSerializable): The runner to be appended.

    Returns:
        Callable: A callable that appends outputs from new messages to the last messages.
    """
    runnable = RunnableParallel(last=RunnablePassthrough(), new=runner)

    def append_outputs(input, **kwargs: Any) -> Any:
        new = input['new']
        last = input['last']
        if not isinstance(last, list):
            last = [last]
        if not isinstance(new, list):
            new = [new]
        return last + new

    return runnable | append_outputs


def get_tool_runner(tool):
    """
    Get a tool runner that prepares tool inputs.

    Args:
        tool (RunnableSerializable): The tool to be prepared.

    Returns:
        Callable: A callable that prepares tool inputs.
    """

    def prepare_tool(input, config, **kwargs: Any) -> Any:
        return input.tool_calls

    return prepare_tool | tool.map()


class BlipMessageStatus(str, Enum):
    """
    The message status across a distributed system.

    PENDING meaning the message was not consumed yet by any worker.
    CONSUMED meaning the message was already consumed by other worker.
    """
    PENDING: str = "PENDING"
    CONSUMED: str = "CONSUMED"


class BlipDistributedMessageManager(ABC):
    """
    A message manager to avoid duplicate processing of messages from multiple connected clients.
    """

    @abstractmethod
    def fetch_message_status(self, message: BlipMessage) -> BlipMessageStatus:
        """
        Fetch the status of a message.

        Args:
            message (BlipMessage): The message to fetch the status for.

        Returns:
            BlipMessageStatus: The status of the message (PENDING or CONSUMED).
        """


class MemoryBlipDistributedMessageManager(BlipDistributedMessageManager):
    """
    A message manager implementation example using shared memory.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize the memory-based message manager.

        Args:
            max_size (int): The maximum size of the collected messages list.
        """
        self.lock = multiprocessing.Lock()
        self.manager = multiprocessing.Manager()
        self.collected_messages = self.manager.list()
        self.max_size = max_size

    def store_message_id(self, message_id: str):
        """
        Store a message ID in the collected messages list.

        Args:
            message_id (str): The message ID to store.
        """
        self.collected_messages.append(message_id)
        if len(self.collected_messages) >= self.max_size:
            self.collected_messages.remove(self.collected_messages[-1])

    def fetch_message_status(self, message: BlipMessage) -> BlipMessageStatus:
        """
        Fetch the message status from local shared memory.

        Args:
            message (BlipMessage): The message to fetch the status for.

        Returns:
            BlipMessageStatus: The status of the message (PENDING or CONSUMED).
        """
        with self.lock:
            if message.id not in self.collected_messages:
                self.store_message_id(message_id=message.id)
                return BlipMessageStatus.PENDING
            return BlipMessageStatus.CONSUMED


class BlipClientWrapper:
    def __init__(
            self,
            identifier: str = None,
            access_key: str = None,
            hostname: str = "ws.msging.net",
            is_trace_enabled: bool = False,
            distributed_message_manager: Optional[BlipDistributedMessageManager] = None
    ):
        """
        Initialize the Blip client wrapper.

        Args:
            identifier (str): The identifier for the Blip client.
            access_key (str): The access key for the Blip client.
            hostname (str): The hostname for the Blip client.
            is_trace_enabled (bool): Whether trace is enabled for the client.
            distributed_message_manager (Optional[BlipDistributedMessageManager]): The distributed message manager.
        """
        self.identifier = identifier or os.getenv("BLIP_IDENTIFIER")
        self.access_key = access_key or os.getenv("BLIP_ACCESS_KEY")
        self.hostname = hostname
        self.is_trace_enabled = is_trace_enabled
        self.distributed_message_manager = distributed_message_manager
        self.client = self.build_client()
        self.loop = asyncio.get_event_loop()

    def build_client(self):
        """
        Build and return the Blip client.

        Returns:
            Client: The built Blip client.
        """
        try:
            from blip_sdk import ClientBuilder, Client
        except (ImportError, ModuleNotFoundError):
            raise IMPORT_ERROR_BLIP_SDK
        try:
            from lime_transport_websocket import WebSocketTransport
        except (ImportError, ModuleNotFoundError):
            raise IMPORT_ERROR_LIME_WEBSOCKET
        try:
            from lime_python import Identity, Message as BlipMessage
        except (ImportError, ModuleNotFoundError):
            raise IMPORT_ERROR_LIME
        return (
            ClientBuilder()
            .with_hostname(self.hostname)
            .with_identifier(self.identifier)
            .with_access_key(self.access_key)
            .with_transport_factory(
                lambda: WebSocketTransport(is_trace_enabled=self.is_trace_enabled)
            )
            .build()
        )

    def send_message(self, message: BlipMessage):
        """
        Send a message using the Blip client.

        Args:
            message (BlipMessage): The message to send.

        Returns:
            Command: The result of the send message command.
        """
        return self.client.send_message(message=message)

    async def connect_async(self):
        """Connect to the Blip server asynchronously."""
        return await self.client.connect_async()

    async def process_command_async(self, command: Command):
        """
        Process a command asynchronously.

        Args:
            command (Command): The command to process.

        Returns:
            Command: The result of the processed command.
        """
        return await self.client.process_command_async(command=command)

    def callback(self, runnable: RunnableSerializable):
        """
        Create a callback to process received messages with a runnable.

        Args:
            runnable (RunnableSerializable): The runnable to process messages.

        Returns:
            Callable: The callback function to process messages.
        """

        async def process_message(message: BlipMessage):
            """
            Process received message with a runnable.

            Args:
                message (BlipMessage): The message to process.
            """
            try:
                from lime_python import Identity, Message
            except (ImportError, ModuleNotFoundError):
                raise IMPORT_ERROR_LIME
            user_id = Identity.parse_str(message.from_n)
            message = convert_message_from_blip(message=message)
            response = await runnable.ainvoke(
                input={"input_message": [message], "chat_history": []},
                config={"configurable": {"blip_user_id": user_id}}
            )
            if isinstance(response, AIMessage):
                if response.content and not response.tool_calls:
                    self.client.send_message(
                        message=Message(
                            type_n="text/plain",
                            content=response.content,
                            to=user_id
                        )
                    )

        return process_message

    def predicate(self, message_filter: Optional[Callable] = None):
        """
        Create a predicate to filter received messages.

        Args:
            message_filter (Optional[Callable]): An optional custom message filter.

        Returns:
            Callable: The predicate function to filter messages.
        """

        def message_predicate(message: BlipMessage) -> bool:
            """
            Filter received messages.

            Args:
                message (BlipMessage): The message to filter.

            Returns:
                bool: True if the message passes the filter, False otherwise.
            """
            if message.type_n == "application/vnd.lime.chatstate+json":
                return False
            if self.distributed_message_manager:
                status = self.distributed_message_manager.fetch_message_status(message=message)
                return status == BlipMessageStatus.PENDING
            if message_filter:
                return message_filter(message)
            return True

        return message_predicate

    def add_runnable(
            self,
            runnable: RunnableSerializable,
            message_filter: Optional[Callable] = None
    ):
        """
        Add a runnable to process messages that pass the filter.

        Args:
            runnable (RunnableSerializable): The runnable to process messages.
            message_filter (Optional[Callable]): An optional custom message filter.
        """
        try:
            from blip_sdk import Receiver
        except (ImportError, ModuleNotFoundError):
            raise IMPORT_ERROR_BLIP_SDK
        receiver = Receiver(
            predicate=self.predicate(message_filter),
            callback=self.callback(runnable)
        )
        self.client.add_message_receiver(receiver=receiver)

    def connect(self):
        """Start listening to messages asynchronously."""

        async def main():
            await self.client.connect_async()
            logger.info("Application started. Press Ctrl + c to stop.")

        self.loop.run_until_complete(main())

    def start(self):
        """Blocking start listening to messages."""
        self.connect()
        self.loop.run_forever()
