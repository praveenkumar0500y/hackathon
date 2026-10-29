import os
import shutil
import zipfile
from pathlib import Path

def create_project_zip():
    base_dir = Path(__file__).resolve().parent.parent
    zip_filename = "QML01_Bengaluru_Fleet_Dispatch.zip"
    
    # Destination 1: Workspace root
    root_zip_path = base_dir / zip_filename
    
    # Destination 2: Downloads folder
    user_home = Path.home()
    downloads_dir = user_home / "Downloads"
    downloads_zip_path = downloads_dir / zip_filename
    
    exclude_dirs = {
        ".git",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".venv",
        "venv",
        ".vite",
        "dist"
    }
    
    exclude_exts = {".pyc", ".pyo", ".zip"}

    print(f"Packaging project from: {base_dir}")
    print(f"Target: {root_zip_path}")
    
    with zipfile.ZipFile(root_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".git")]
            
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in exclude_exts or ext == ".log" or f in [".DS_Store", "Thumbs.db", ".env", ".env.local"]:
                    continue
                file_path = Path(root) / f
                if file_path == root_zip_path or file_path.name == zip_filename:
                    continue
                rel_path = file_path.relative_to(base_dir)
                zf.write(file_path, rel_path)

    # Copy to Downloads
    if downloads_dir.exists():
        shutil.copy2(root_zip_path, downloads_zip_path)
        print(f"Copied zip to Downloads: {downloads_zip_path}")
    
    size_mb = root_zip_path.stat().st_size / (1024 * 1024)
    print(f"Successfully packaged {zip_filename} ({size_mb:.2f} MB)")
    print(f"Workspace path: {root_zip_path}")
    print(f"Downloads path: {downloads_zip_path}")

if __name__ == "__main__":
    create_project_zip()
