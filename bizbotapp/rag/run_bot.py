import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run BizBot in different modes.")
parser.add_argument("mode", choices=["gradio", "api"], help="Choose which interface to run.")
args = parser.parse_args()

if args.mode == "gradio":
    print("🚀 Launching Gradio UI...")
    subprocess.run(["python3", "gradio_ui.py"])

elif args.mode == "api":
    print("🚀 Launching FastAPI server...")
    subprocess.run(["uvicorn", "api.server:app", "--reload"])