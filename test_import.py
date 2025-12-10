
import sys
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
