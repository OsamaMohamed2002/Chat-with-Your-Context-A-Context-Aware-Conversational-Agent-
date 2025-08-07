from langchain_groq import ChatGroq
from tools.context_presence_judge import build_context_presence_tool
from tools.web_search_tool import build_web_search_tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.memory import ConversationBufferWindowMemory



def initialize_llm():
    """
    Initializes the LLM from Groq.
    Ensures the GROQ_API_KEY is set in the .env file.
    """
    try:
        # We use ChatGroq for chat-based models from the Groq API
        llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0 # A low temperature is good for tool-using agents
        )
        print("Groq LLM Initialized Successfully.")
        return llm
    except Exception as e:
        print(f"Failed to initialize Groq LLM. Please ensure your GROQ_API_KEY is set correctly in the .env file. Error: {e}")
        return None

def create_agent_chain():
    """
    Creates the final, polished chain with conditional routing and memory.
    This version includes refined prompts for better answer quality and a more natural tone.
    """
    print("Initializing final polished agent...")
    llm = initialize_llm()
    if not llm:
        return None, None

    # The Memory Object 
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)

    # The Context Judge Chain 
    judge_prompt = PromptTemplate.from_template(
        "Determine if the following user message includes background context or is just a direct question. "
        "Output only 'context_provided' or 'context_missing'.\n\nMessage: {input}\nClassification:"
    )
    context_judge_chain = judge_prompt | llm.with_config({"temperature": 0.0}) | StrOutputParser()

    #The Web Search Agent 
    search_prompt_template = """You are a helpful assistant. You have access to a web search tool. 
After using your tools, synthesize the findings and provide a direct, concise, and helpful final answer to the user.

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

New question: {input}
{agent_scratchpad}"""
    
    search_prompt = PromptTemplate.from_template(search_prompt_template)
    search_tools = [build_web_search_tool()]
    search_agent = create_react_agent(llm, search_tools, search_prompt)
    search_agent_executor = AgentExecutor(
        agent=search_agent, tools=search_tools, memory=memory, handle_parsing_errors=True, verbose=True
    ).with_config({"run_name": "WebSearchAgent"})
    search_chain = search_agent_executor | (lambda x: x["output"])

    # The Direct Answer Chain
    direct_answer_prompt_template = """You are a helpful assistant. Use the chat history and the user's question to provide a direct and helpful answer. 
Elaborate on the concepts using your own knowledge. Do not start your response by talking about the user's question or the conversation.

Chat History:
{chat_history}

User's Question: {input}

Helpful Answer:"""
    
    direct_answer_prompt = PromptTemplate.from_template(direct_answer_prompt_template)
    direct_answer_chain = (
        {
            "input": RunnablePassthrough(),
            "chat_history": lambda x: memory.load_memory_variables(x)["chat_history"]
        }
        | direct_answer_prompt 
        | llm 
        | StrOutputParser()
    )

    # The Router
    branch = RunnableBranch(
        (lambda x: "context_missing" in x["context_decision"].lower(), search_chain),
        direct_answer_chain
    )

    # The Full Chain 
    full_chain = {
        "context_decision": context_judge_chain,
        "input": RunnablePassthrough() 
    } | branch

    print("Polished agent initialized successfully.")
    return full_chain, memory
if __name__ == '__main__':
    agent = create_agent_chain()
    if agent:
        # Example 1: A question without context
        response1 = agent.run("What is LangChain used for?")
        print("\n--- Response 1 ---")
        print(response1)

        # Example 2: A question with context
        response2 = agent.run("LangChain is a framework for developing applications powered by LLMs. What company develops it?")
        print("\n--- Response 2 ---")
        print(response2)