
#Standard Libraries
import sys

#Third Pary Libraries
#Diagnostic version print to detect incompatible openai versions
import openai

#Local Modules references
from .mng_json import json_manager




print(f"Plush - Running on python installation: {sys.executable}, ver: {sys.version}")
print("Plush - Current Openai Version: ", openai.__version__)

jmanager = json_manager()
if jmanager.on_startup(False):
    jmanager.log_events("config.json was updated")
else:
    jmanager.log_events("config.json was not updated")

__version__ ="1.22.1"
print('Plush - Version:', __version__)

WEB_DIRECTORY = "./web"

#********************************************************************************
from .style_prompt import NODE_CLASS_MAPPINGS as styClassMappings, NODE_DISPLAY_NAME_MAPPINGS as styDisplay
from .UtilNodes import NODE_CLASS_MAPPINGS as utilClassMappings, NODE_DISPLAY_NAME_MAPPINGS as utilDisplay
from .text_files import NODE_CLASS_MAPPINGS as textClassMappings, NODE_DISPLAY_NAME_MAPPINGS as textDisplay

NODE_CLASS_MAPPINGS = {**styClassMappings, **utilClassMappings, **textClassMappings}
NODE_DISPLAY_NAME_MAPPINGS = {**styDisplay, **utilDisplay, **textDisplay}

#NODE_CLASS_MAPPINGS = {**styClassMappings, **utilClassMappings, **testClassMappings}
#NODE_DISPLAY_NAME_MAPPINGS = {**styDisplay, **utilDisplay, **testDisplay}
WEB_DIRECTORY = "./Web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]
