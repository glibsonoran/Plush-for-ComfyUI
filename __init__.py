#In ComfyUI_plush Lets puch this add to repostitory
from . import style_prompt 

NODE_CLASS_MAPPINGS = style_prompt.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = style_prompt.NODE_DISPLAY_NAME_MAPPINGS
#from .style_prompt import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
#from sty import NODE_CLASS_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']



""" import os
import json
import sys
#new
import shutil """


""" config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
backup_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bkup")
#debug
#print(sys.path)
## Check if config file exists in ComfyUI_plush directory
if not os.path.isfile(config_file_path):
    # Check if there's a backup in the bkup directory
    backup_config_path = os.path.join(backup_dir, "config.json")

    if os.path.isfile(backup_config_path):
        # Copy the backup config.json to ComfyUI_plush directory
        shutil.copy(backup_config_path, config_file_path)
    else:
        # Create config file in ComfyUI_plush directory if no backup exists
        pass """

""" config = {
            "autoUpdate": True,
            "branch": "dev",
            "openAI_API_Key": "sk-#########################################"
        }
        with open(config_file_path, "w") as f:
            json.dump(config, f, indent=4) """
#********************************************************************************
