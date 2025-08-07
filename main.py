import os
from dotenv import load_dotenv

# --- LOAD ENVIRONMENT VARIABLES FIRST ---
# This must run before any other project modules are imported.
load_dotenv()


from ui.app import create_ui

def main():
    print("Creating Gradio UI...")
    demo = create_ui()
    if demo:
        demo.launch(share=True)
    else:
        print("Failed to create UI because the agent could not be initialized.")


if __name__ == "__main__":
    main()