import os

# Danh sách thư mục cần bỏ qua
EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", ".idea", ".mypy_cache" ,"libs","userdata","logs"}

def print_tree(startpath, prefix=""):
    for item in sorted(os.listdir(startpath)):
        if item in EXCLUDE_DIRS:
            continue
        path = os.path.join(startpath, item)
        if os.path.isdir(path):
            print(f"{prefix}├── {item}/")
            print_tree(path, prefix + "│   ")
        else:
            print(f"{prefix}├── {item}")

# In cây thư mục từ thư mục hiện tại
print("project/")
print_tree(".")
