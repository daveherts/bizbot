import argparse
import subprocess
import os

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Run BizBot in different modes.")
parser.add_argument("mode", choices=["gradio", "api"], help="Choose which interface to run.")
args = parser.parse_args()

# Launch Gradio UI
if args.mode == "gradio":
    print("Launching Gradio UI...")
    subprocess.run(["python3", "rag/gradio_ui.py"])  

# Launch FastAPI
elif args.mode == "api":
    print("Launching FastAPI server...")
    subprocess.run(["uvicorn", "rag.api.server:app", "--reload"])
