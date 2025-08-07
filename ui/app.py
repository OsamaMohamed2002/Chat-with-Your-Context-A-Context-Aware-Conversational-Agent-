# In ui/app.py

import gradio as gr
from agent.agent_runner import create_agent_chain

# Initialize both the chain and memory
agent_chain, memory = create_agent_chain()

def chat_function(message, history):
    """
    The main function that manages the conversation state and interacts with the chain.
    """
    print(f"User message: {message}")
    if not agent_chain:
        return "Chain not initialized. Please check the console for errors."
    
    try:
        # The chain internally loads the history from the memory object where needed.
        response = agent_chain.invoke({"input": message})
        memory.save_context({"input": message}, {"output": response})

    except Exception as e:
        print(f"An error occurred during chain execution: {e}")
        response = "An error occurred while processing your request."

    print(f"Agent response: {response}")
    return response

def create_ui():
    """
    Creates the Gradio Chat UI.
    """
    print("Creating Gradio UI...")
    try:
        chat_interface = gr.ChatInterface(
            fn=chat_function,
            title="Chat with Your Context – A Context-Aware Conversational Agent 🚀",
            description="Ask me anything, or provide context first. I can search the web if needed!",
            
        )
        return chat_interface
    except Exception as e:
        print(f"Failed to create Gradio UI: {e}")
        return None