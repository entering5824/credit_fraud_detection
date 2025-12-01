# Cài đặt các thư viện cần thiết

import nbformat
from nbclient import NotebookClient
import os

notebook_path = r".\notebooks\01_EDA_and_Preprocessing.ipynb"
output_path = r".\notebooks\01_EDA_and_Preprocessing_executed.ipynb"

# Kiểm tra file tồn tại
if not os.path.exists(notebook_path):
    raise FileNotFoundError(f"Notebook {notebook_path} không tồn tại!")

# Đọc notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Tạo client để chạy notebook
client = NotebookClient(nb, timeout=600, kernel_name="python3")
executed_nb = client.execute()

# Lưu notebook đã chạy
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(executed_nb, f)

print(f"Notebook đã chạy xong, lưu tại: {output_path}")
