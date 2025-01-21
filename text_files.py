# Modified from the original pysssss text_files.py
# All credit and kudos to the original author

import os
import folder_paths
from server import PromptServer
import glob
from aiohttp import web
from .mng_json import json_manager, TroubleSgltn
from pathlib import Path

def get_allowed_dirs():
    """
    Gets the allowed directories configuration using json_manager.
    Returns:
        dict: Dictionary of directory configurations from text_file_dirs.json
    """
    j_mngr = json_manager()
    user_dir = j_mngr.find_child_directory(j_mngr.script_dir, "user", create=True)
    config_path = j_mngr.append_filename_to_path(user_dir, "text_file_dirs.json")
    
    json_data = j_mngr.load_json(config_path, is_critical=True)
    if not json_data:
        msg = f"Failed to load configuration from: {config_path}"
        j_mngr.log_events(msg, TroubleSgltn.Severity.ERROR, is_trouble=True)
        raise FileNotFoundError(msg)
    return json_data

def get_valid_dirs():
    return get_allowed_dirs().keys()

def get_dir_from_name(name, j_mngr=None):
    """Get and validate directory path from configuration name"""
    if j_mngr is None:
        j_mngr = json_manager()
    
    dirs = get_allowed_dirs()
    if name not in dirs:
        msg = f"Directory '{name}' not found in configuration"
        j_mngr.log_events(msg, TroubleSgltn.Severity.ERROR, is_trouble=True)
        raise KeyError(msg)

    path = dirs[name]
    path = path.replace("$input", folder_paths.get_input_directory())
    path = path.replace("$output", folder_paths.get_output_directory())
    path = path.replace("$temp", folder_paths.get_temp_directory())
    
    return path

def is_child_dir(parent_path, child_path, j_mngr=None):
    """Verify child path is actually within parent path"""
    if j_mngr is None:
        j_mngr = json_manager()
        
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)
    is_child = os.path.commonpath([parent_path]) == os.path.commonpath([parent_path, child_path])
    
    if not is_child:
        msg = f"Security warning: Attempted access to path outside parent directory: {child_path}"
        j_mngr.log_events(msg, TroubleSgltn.Severity.WARNING, is_trouble=True)
        
    return is_child

def get_real_path(directory):
    directory = directory.replace("/**/", "/")
    directory = os.path.abspath(directory)
    directory = os.path.split(directory)[0]
    return directory

@PromptServer.instance.routes.get("/plush_for_comfy/text_file/{name}")
async def get_files(request):
    j_mngr = json_manager()
    name = request.match_info["name"]
    try:
        directory = get_dir_from_name(name, j_mngr)
        recursive = "/**/" in directory
        pre = get_real_path(directory)

        files = list(map(lambda t: os.path.relpath(t, pre),
                        glob.glob(directory, recursive=recursive)))

        if len(files) == 0:
            j_mngr.log_events(f"No files found in directory: {directory}", 
                            TroubleSgltn.Severity.INFO, 
                            is_trouble=True)
            files = ["[none]"]
            
        return web.json_response(files)
        
    except Exception as exc:
        j_mngr.log_events(f"Error listing files: {str(exc)}", 
                         TroubleSgltn.Severity.ERROR, 
                         is_trouble=True)
        raise

def get_file(root_dir, file, j_mngr=None):
    """Get and validate full file path"""
    if j_mngr is None:
        j_mngr = json_manager()
        
    if file == "[none]" or not file or not file.strip():
        msg = "No file selected"
        j_mngr.log_events(msg, TroubleSgltn.Severity.WARNING, is_trouble=True)
        raise ValueError(msg)

    root_dir = get_dir_from_name(root_dir, j_mngr)
    root_dir = get_real_path(root_dir)
    
    # Use pathlib for directory creation
    root_path = Path(root_dir)
    if not root_path.exists():
        j_mngr.log_events(f"Creating directory: {root_dir}")
        root_path.mkdir(parents=True, exist_ok=True)
        
    full_path = os.path.join(root_dir, file)

    if not is_child_dir(root_dir, full_path, j_mngr):
        msg = f"Security error: Attempted access to file outside root directory: {full_path}"
        j_mngr.log_events(msg, TroubleSgltn.Severity.ERROR, is_trouble=True)
        raise ReferenceError(msg)

    return full_path

class TextFileNode:
    """Base class for text file operations"""
    RETURN_TYPES = ("STRING", "STRING",)  # (content, troubles)
    RETURN_NAMES = ("saved_file", "troubleshooting")
    CATEGORY = "Plush/Utils"
    file = None

    @classmethod
    def VALIDATE_INPUTS(cls, root_dir, file, **kwargs):
        j_mngr = json_manager()
        if file == "[none]" or not file or not file.strip():
            return True
        try:
            get_file(root_dir, file, j_mngr)
            return True
        except Exception as exc:
            j_mngr.log_events(f"Input validation error: {str(exc)}", 
                            TroubleSgltn.Severity.ERROR, 
                            is_trouble=True)
            return False

    def load_text(self, **kwargs):
        j_mngr = json_manager()
        j_mngr.trbl.reset("Text File Operations")
        
        try:
            self.file = get_file(kwargs["root_dir"], kwargs["file"], j_mngr)
            content = j_mngr.read_file_contents(self.file, is_critical=False)
            if content is None:
                return ("", j_mngr.trbl.get_troubles())
                
            j_mngr.log_events(f"Successfully read file: {self.file}", is_trouble=True)
            return (content, j_mngr.trbl.get_troubles())
            
        except Exception as exc:
            j_mngr.log_events(f"Error reading file: {str(exc)}", 
                            TroubleSgltn.Severity.ERROR, 
                            is_trouble=True)
            return ("", j_mngr.trbl.get_troubles())

class LoadText(TextFileNode):
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if not cls.file:
            return False
        try:
            return os.path.getmtime(cls.file)
        except Exception:
            return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root_dir": (list(get_valid_dirs()), {}),
                "file": (["[none]"], {
                    "plush.binding": [{
                        "source": "root_dir",
                        "callback": [{
                            "type": "set",
                            "target": "$this.disabled",
                            "value": True
                        }, {
                            "type": "fetch",
                            "url": "/plush_for_comfy/text_file/{$source.value}",
                            "then": [{
                                "type": "set",
                                "target": "$this.options.values",
                                "value": "$result"
                            }, {
                                "type": "validate-combo"
                            }, {
                                "type": "set",
                                "target": "$this.disabled",
                                "value": False
                            }]
                        }],
                    }]
                })
            },
        }

    FUNCTION = "load_text"

class SaveText(TextFileNode):
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root_dir": (list(get_valid_dirs()), {}),
                "file": ("STRING", {"default": "file.txt"}),
                "append": (["append", "overwrite", "new only"], {}),
                "insert": ("BOOLEAN", {
                    "default": True, 
                    "label_on": "new line", 
                    "label_off": "none",
                    "plush.binding": [{
                        "source": "append",
                        "callback": [{
                            "type": "if",
                            "condition": [{
                                "left": "$source.value",
                                "op": "eq",
                                "right": '"append"'
                            }],
                            "true": [{
                                "type": "set",
                                "target": "$this.disabled",
                                "value": False
                            }],
                            "false": [{
                                "type": "set",
                                "target": "$this.disabled",
                                "value": True
                            }],
                        }]
                    }]
                }),
                "text": ("STRING", {"forceInput": True, "multiline": True})
            },
        }

    FUNCTION = "write_text"

    def write_text(self, **kwargs):
        j_mngr = json_manager()
        j_mngr.trbl.reset("Save Text Operations")
        
        try:
            self.file = get_file(kwargs["root_dir"], kwargs["file"], j_mngr)
            
            if kwargs["append"] == "new only" and os.path.exists(self.file):
                msg = f"File {self.file} already exists and 'new only' is selected"
                j_mngr.log_events(msg, TroubleSgltn.Severity.ERROR, is_trouble=True)
                raise FileExistsError(msg)
            
            text = kwargs["text"]
            if kwargs["append"] and kwargs["insert"]:
                text = f"\n{text}"
                
            success = j_mngr.write_string_to_file(
                text,
                self.file,
                is_critical=True,
                append=(kwargs["append"] == "append")
            )
            
            if success:
                j_mngr.log_events(f"Successfully wrote to file: {self.file}")
                return super().load_text(**kwargs)
                
            return ("", j_mngr.trbl.get_troubles())
            
        except Exception as exc:
            j_mngr.log_events(f"Error writing file: {str(exc)}", 
                         TroubleSgltn.Severity.ERROR, 
                         is_trouble=True)
            return ("", j_mngr.trbl.get_troubles())

NODE_CLASS_MAPPINGS = {
    "LoadText|plush": LoadText,
    "SaveText|plush": SaveText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadText|plush": "Load Saved Files 🧸",
    "SaveText|plush": "Save Files 🧸",
}