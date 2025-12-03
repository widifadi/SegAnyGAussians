import os
import sys

# Add torch DLL directory to search path (Windows only)
try:
    import torch
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if sys.platform == 'win32' and os.path.exists(torch_lib_dir):
        os.add_dll_directory(torch_lib_dir)
except Exception:
    pass

from simple_knn._C import *
