import os
import subprocess
import sys
venv_path = r"Y:\Faceon_Project\.venv\Lib\site-packages"
openvino_path = os.path.join(venv_path, "openvino")
os.environ["PYTHONPATH"] = venv_path
sys.path.append(venv_path)
mo_path = os.path.join(openvino_path, "tools", "mo", "mo.py")
input_model_path = r"C:\Users\USER\Downloads\model.onnx"
output_dir = r"C:\Users\USER\Downloads\IR"
subprocess.run(["python", mo_path, "--input_model", input_model_path, "--output_dir", output_dir], check=True)