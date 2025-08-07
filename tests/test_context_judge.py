import sys
import os
from unittest.mock import MagicMock

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from langchain_core.runnables import Runnable
from tools.context_presence_judge import build_context_presence_tool

class MockLLM(Runnable):
    def __init__(self, response: str):
        from langchain_core.messages import AIMessage
        self.response_message = AIMessage(content=response)

    def invoke(self, input_dict: dict, config=None) -> MagicMock:
        # The invoke method is required by the Runnable interface
        return self.response_message

    def with_config(self, config: dict):
        # The tool calls .with_config(), so we return the object itself
        return self



def test_with_context_present():
    """
    Tests if the tool correctly identifies a message that has context.
    """
    # 1. Setup
    mock_llm = MockLLM(response="context_provided")
    context_judge_tool = build_context_presence_tool(mock_llm)

    # 2. Input
    test_message = "My car is making a weird rattling sound. What could be the cause?"
    
    # 3. Action
    result = context_judge_tool.func(test_message)

    # 4. Assert
    assert result == "context_provided"

def test_with_context_missing():
    """
    Tests if the tool correctly identifies a message that is a direct question.
    """
    # 1. Setup
    mock_llm = MockLLM(response="context_missing")
    context_judge_tool = build_context_presence_tool(mock_llm)

    # 2. Input
    test_message = "What is the capital of Mongolia?"

    # 3. Action
    result = context_judge_tool.func(test_message)

    # 4. Assert
    assert result == "context_missing"