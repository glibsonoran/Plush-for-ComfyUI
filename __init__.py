
import os
import json
import sys
#new
import shutil
import json

#Diagnostic version print to detect incompatible openai versions
import openai
print("Plush - Current Openai Version: ", openai.__version__)

__version__ ="0.995"


config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
backup_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bkup")
backup_config_path = os.path.join(backup_dir, "config.json")
#debug
#print(sys.path)
## Check if config file exists in ComfyUI_plush directory
if not os.path.isfile(config_file_path):
    # Check if there's a backup in the bkup directory
    if os.path.isfile(backup_config_path):
        # Copy the backup config.json to ComfyUI_plush directory
        shutil.copy(backup_config_path, config_file_path)
    else:
        pass

    #Try and open the JSON file and deal with a decode error
config_bad = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.bad")
try:
    with open(config_file_path, 'r') as config_file:
        config_data =  json.load(config_file)
    #if the config.json is corrupt, rename it config.bad and copy the backup in its place    
except json.JSONDecodeError as e:
    print(f"Error decoding JSON in, attempting to replace corrupt file {config_file_path}: {e}")
    os.rename(config_file_path, config_bad)
    shutil.copy(backup_config_path, config_file_path)



    """ config = {
                "autoUpdate": True,
                "branch": "dev",
                "openAI_API_Key": "sk-#########################################"
            }
            with open(config_file_path, "w") as f:
                json.dump(config, f, indent=4) """
#********************************************************************************
from . import style_prompt 

NODE_CLASS_MAPPINGS = style_prompt.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = style_prompt.NODE_DISPLAY_NAME_MAPPINGS
#from .style_prompt import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
#from sty import NODE_CLASS_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

