import os
import sys

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for path in [PROJECT_ROOT, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)
