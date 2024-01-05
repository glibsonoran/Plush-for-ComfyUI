

import sys
#from .mng_json import json_manager
from .mng_json import json_manager

#Diagnostic version print to detect incompatible openai versions
import openai
print("Plush - Running on python installation:", sys.executable)
print("Plush - Current Openai Version: ", openai.__version__)

jmanager = json_manager()

if jmanager.update_config(False):
    print('Plush - config.json updated')
else:
    print('Plush - config.json no update')

__version__ ="1.10"
print('Plush - Version:', __version__)


#********************************************************************************
from . import style_prompt 

NODE_CLASS_MAPPINGS = style_prompt.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = style_prompt.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

