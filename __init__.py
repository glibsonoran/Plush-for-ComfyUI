

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

__version__ ="1.20"
print('Plush - Version:', __version__)


#********************************************************************************
from .style_prompt import NODE_CLASS_MAPPINGS as styClassMappings, NODE_DISPLAY_NAME_MAPPINGS as styDisplay
from .UtilNodes import NODE_CLASS_MAPPINGS as utilClassMappings, NODE_DISPLAY_NAME_MAPPINGS as utilDisplay
# ** unpacks the dicts into a new dict
NODE_CLASS_MAPPINGS = {**styClassMappings, **utilClassMappings}

NODE_DISPLAY_NAME_MAPPINGS = {**styDisplay, **utilDisplay}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

