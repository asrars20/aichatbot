from typing import List, Type

from pydantic import Field
from steamship.agents.functional import FunctionsBasedAgent
from steamship.agents.llms.openai import ChatOpenAI
#from steamship.agents.mixins.transports.slack import (
#    SlackTransport,
#    SlackTransportConfig,
#)
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
#from steamship.agents.mixins.transports.telegram import (
#    TelegramTransport,
#    TelegramTransportConfig,
#)
from steamship.agents.schema import Tool
from steamship.agents.service.agent_service import AgentService
from steamship.agents.tools.question_answering import VectorSearchQATool
from steamship.invocable import Config
from steamship.invocable.mixins.blockifier_mixin import BlockifierMixin
from steamship.invocable.mixins.file_importer_mixin import FileImporterMixin
from steamship.invocable.mixins.indexer_mixin import IndexerMixin
from steamship.invocable.mixins.indexer_pipeline_mixin import IndexerPipelineMixin

SYSTEM_PROMPT = """You are Assistant, an assistant who answers questions about a class called Computer Graphics.
 
Who you are:
- You are a helpful robot.
- You were created by Steamship.
- You are kind, compassionate, optimistic robot.
- Your job is to answer questions about the course contents.

Access information about the course policies, course structure, and the instructor from the `syllabus` file to answer user questions.
Access information about the course content from the `lecture slides` file to answer user questions about the information covered in the class.
"""

class DocumentQAAgentService(AgentService):
    """DocumentQAService is an example AgentService that exposes:  # noqa: RST201

    - A few authenticated endpoints for learning PDF and YouTube documents:

         /index_url
        { url }

        /index_text
        { text }

    - An unauthenticated endpoint for answering questions about what it has learned

    This agent provides a starter project for special purpose QA agents that can answer questions about documents
    you provide.
    """
    
    USED_MIXIN_CLASSES = [
        IndexerPipelineMixin,
        FileImporterMixin,
        BlockifierMixin,
        IndexerMixin,
        SteamshipWidgetTransport,
        #TelegramTransport,
        #SlackTransport,
    ]
    """USED_MIXIN_CLASSES tells Steamship what additional HTTP endpoints to register on your AgentService."""

    class DocumentQAAgentServiceConfig(Config):
        """Pydantic definition of the user-settable Configuration of this Agent."""

        telegram_bot_token: str = Field(
            "", description="[Optional] Secret token for connecting to Telegram"
        )

    config: DocumentQAAgentServiceConfig
    """The configuration block that users who create an instance of this agent will provide."""

    tools: List[Tool]
    """The list of Tools that this agent is capable of using."""

    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return the Configuration class so that Steamship can auto-generate a web UI upon agent creation time."""
        return DocumentQAAgentService.DocumentQAAgentServiceConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Tools Setup
        # -----------

        # Tools can return text, audio, video, and images. They can store & retrieve information from vector DBs, and
        # they can be stateful -- using Key-Valued storage and conversation history.
        #
        # See https://docs.steamship.com for a full list of supported Tools.
        self.tools = [VectorSearchQATool()]

        # Agent Setup
        # ---------------------

        #self.set_default_agent(
        #    FunctionsBasedAgent(
        #        tools=self.tools,
        #        llm=ChatOpenAI(self.client),
        #    )
        #)
        
        
        self._agent = FunctionsBasedAgent(
                tools=self.tools,
                llm=ChatOpenAI(self.client),
            )
        
        self._agent.PROMPT = SYSTEM_PROMPT
        #self.....PROMPT = SYSTEM_PROMPT

        # Document QA Mixin Setup
        # -----------------------

        # This Mixin provides HTTP endpoints that coordinate the learning of documents.
        #
        # It adds the `/learn_url` endpoint which will:
        #    1) Download the provided URL (PDF, YouTube URL, etc)
        #    2) Convert that URL into text
        #    3) Store the text in a vector index
        #
        # That vector index is then available to the question answering tool, below.
        syllabus = IndexerPipelineMixin(self.client, self)
        lectures = IndexerPipelineMixin(self.client, self)
        syllabus.index_url("https://drive.google.com/file/d/1TX3WfLKYtGI4uIs7RWlL3mWacCIOCSG3/view", metadata={'name':'syllabus'})
        lectures.index_url("https://drive.google.com/file/d/11RfYZcCCVI_dQaHdvGKOEqsE-plhAlTs/view", metadata={'name':'lecture slides'})
        self.add_mixin(syllabus)
        self.add_mixin(lectures)
        #self.add_mixin(IndexerPipelineMixin(self.client, self).index_url('https://www.gutenberg.org/cache/epub/73052/pg73052-images.html'))

        # Communication Transport Setup
        # -----------------------------

        # Support Steamship's web client
        self.add_mixin(
            SteamshipWidgetTransport(
                client=self.client,
                agent_service=self,
            )
        )

        # Support Slack
        #self.add_mixin(
        #    SlackTransport(
        #        client=self.client,
        #        config=SlackTransportConfig(),
        #        agent_service=self,
        #    )
        #)

        # Support Telegram
        #self.add_mixin(
        #    TelegramTransport(
        #        client=self.client,
        #        config=TelegramTransportConfig(
        #            bot_token=self.config.telegram_bot_token
        #        ),
        #        agent_service=self,
        #    )
        #)
