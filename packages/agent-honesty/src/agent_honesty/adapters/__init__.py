"""
Truthify Framework Adapters
---------------------------
Drop-in integration layers for LangGraph, CrewAI, AutoGen, and LlamaIndex.
"""

from agent_honesty.adapters.autogen import TruthifyAgentInterceptor
from agent_honesty.adapters.base import BaseFrameworkAdapter, require_package
from agent_honesty.adapters.crewai import TruthifyCrewCallback, wrap_crew_tool
from agent_honesty.adapters.langgraph import TruthifyGraphEvaluator, TruthifyToolNode
from agent_honesty.adapters.llamaindex import (
    TruthifyLlamaAdapter,
    TruthifyLlamaToolWrapper,
    wrap_llama_tools,
)

__all__ = [
    "BaseFrameworkAdapter",
    "require_package",
    "TruthifyToolNode",
    "TruthifyGraphEvaluator",
    "TruthifyCrewCallback",
    "wrap_crew_tool",
    "TruthifyAgentInterceptor",
    "TruthifyLlamaAdapter",
    "TruthifyLlamaToolWrapper",
    "wrap_llama_tools",
]
