# -*- coding: utf-8 -*-
"""
Packages the complete codebase into a clean, reproducible zip file for DA3 submission:
submission/BCSE209L_DA3_Code.zip
Excludes pycache, pytest caches, virtualenvs, and temporary files.
"""

import os
import zipfile

def make_zip():
    print("Packaging submission zip...")
    os.makedirs("submission", exist_ok=True)
    zip_path = "submission/BCSE209L_DA3_Code.zip"

    exclude_dirs = {
        '__pycache__', '.pytest_cache', '.git', '.gemini', 'submission',
        '.venv', 'venv', 'env', '.idea', '.vscode'
    }
    exclude_exts = {'.pyc', '.pyo', '.pyd', '.log'}

    count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            # Exclude forbidden directories in-place
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in exclude_exts:
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, ".")
                zipf.write(full_path, rel_path)
                count += 1

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Successfully packaged {count} files into {zip_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    make_zip()
