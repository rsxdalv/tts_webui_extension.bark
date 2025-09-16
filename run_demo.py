import os
import sys

# apply torch patch
from tts_webui.utils.torch_load_patch import apply_torch_load_patch

apply_torch_load_patch()

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now run the demo
from extension_bark.demo import demo

# if __name__ == "__main__":
demo.launch()
