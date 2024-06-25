import subprocess
import os

mo_path = r"Y:\Faceon_Project\.venv\Lib\site-packages\openvino\tools\mo\mo.py"
input_model_path = "C:\\Users\\USER\\Downloads\\vggface_saved_model.keras"
output_dir = "C:\\Users\\USER\\Downloads"

# Model Optimizer 실행
try:
    subprocess.run(["python", mo_path, "--input_model", input_model_path, "--output_dir", output_dir, "--framework", "tf"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error during model conversion: {e}")
