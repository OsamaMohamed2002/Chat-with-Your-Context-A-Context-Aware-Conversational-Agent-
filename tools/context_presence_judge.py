from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLLM

from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import Runnable

def build_context_presence_tool(llm: BaseLLM) -> Tool:
    """
    Builds a LangChain Tool that uses an LLM to judge whether context is present in user input.
    """
    try:
        with open("prompts/context_judge_prompt.txt", "r") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        print("Error: prompts/context_judge_prompt.txt not found.")
        # Handle error appropriately, maybe raise it or use a default
        raise
        
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Set temperature to 0.0 for deterministic classification
    chain_llm = llm.with_config({"temperature": 0.0})
    
    
    # This ensures the chain's final output is a clean string.
    chain: Runnable = prompt | chain_llm | StrOutputParser()

    # This function cleans up the LLM's output string.
    def tool_function(text: str) -> str:
        """Helper function to invoke the chain and clean the output."""
        response = chain.invoke({"input": text})
        # Strip whitespace and quotes which LLMs sometimes add
        return response.strip().replace('"', '')

    return Tool.from_function(
        #  Pass the new wrapper function to the tool.
        func=tool_function,
        name="ContextPresenceJudge",
        description="Checks if context is present in the user's input. This is the first tool to use."
    )