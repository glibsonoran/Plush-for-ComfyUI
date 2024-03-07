

import sys
#from .mng_json import json_manager
from .mng_json import json_manager

#Diagnostic version print to detect incompatible openai versions
import openai

print(f"Plush - Running on python installation: {sys.executable}, ver: {sys.version}")
print("Plush - Current Openai Version: ", openai.__version__)

jmanager = json_manager()
if jmanager.on_startup(False):
    jmanager.log_events("config.json was updated")
else:
    jmanager.log_events("config.json was not updated")

__version__ ="1.21.2"
print('Plush - Version:', __version__)


#********************************************************************************
from .style_prompt import NODE_CLASS_MAPPINGS as styClassMappings, NODE_DISPLAY_NAME_MAPPINGS as styDisplay
from .UtilNodes import NODE_CLASS_MAPPINGS as utilClassMappings, NODE_DISPLAY_NAME_MAPPINGS as utilDisplay
# ** unpacks the dicts into a new dict
NODE_CLASS_MAPPINGS = {**styClassMappings, **utilClassMappings}

NODE_DISPLAY_NAME_MAPPINGS = {**styDisplay, **utilDisplay}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

