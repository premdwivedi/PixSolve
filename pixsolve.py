import subprocess
import os


def run_app():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(cur_dir, "pixolve_aiml.py")
    subprocess.run(["streamlit", "run", app_path])