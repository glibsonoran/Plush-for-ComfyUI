
import json
import os
import shutil
from enum import Enum
import bisect
from datetime import datetime, timedelta
import re
import math
import time
from pathlib import Path
from typing import Optional, Any,  Union


class TroubleSgltn:
    """
    A Singleton class that acts as a central hub for log messages logged using json_manager.log_events().
    This class formats and stores these event messages until the reset() method is called, clearing the data and optionally
    creating a process header describing the method or class that's the origin of the logs that follow. 
    Nodes that use this class should initialize with TroubleSgltn.reset('my_process') at the top of the main method at the start of the run. 
    If you want a more granular listing of the processes being logged you can append a new process header using set_process_header.
    The node's main method can then query the .get_troubles() method at the end of the run to fetch all stored log messages 
    and present them to the user in the return tuple: 'return(result, TroubleSgltn.get_trouble()').
    """
    _instance = None

    class Severity(Enum):
        INFO = 1
        WARNING = 2
        ERROR = 3

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize any necessary attributes here
            cls._troubles = ""  # Example attribute for storing trouble messages
            cls._section_bullet = "\n\u27a4"
            cls._bullet = "\u2726"
            cls._new_lines = "\n"
            cls._header_stack = []
        return cls._instance
    
    def set_process_header(self, process_head:str="New Process")-> None:

        self._troubles += f'{self._new_lines}{self._section_bullet} Begin Log for: {process_head}:{self._new_lines}'
        self._header_stack.append(process_head)

    def pop_header(self)->bool:
        is_popped = False
        if self._header_stack:
            self._header_stack.pop()
            if self._header_stack:
                process_head = self._header_stack[-1]
                self._troubles += f'{self._new_lines}{self._section_bullet} Begin Log for: {process_head}:{self._new_lines}'
                is_popped = True
        
        return is_popped
           
            


    def log_trouble(self, message: str, severity: Severity) -> None:
        """
        Logs a trouble message with a specified severity, 
        and formats it for display.

        Args:
            message (str): The trouble message to log.
            severity (str): The severity level of the message.
        """
        # Example implementation; customize as needed
        trouble_message = f"{self._bullet} {severity.name}: {message}{self._new_lines}"
        self._troubles += trouble_message

    def reset(self, process_head:str='') -> None:
        """
        Resets the stored trouble messages.
        Sets log name header if value is passed
        """
        self._troubles = ""
        self._header_stack = []
        if process_head:
            self.set_process_header(process_head)


    def get_troubles(self) -> str:
        """
        Returns the stored trouble messages.

        Returns:
            str: The accumulated trouble messages.
        """
        return self._troubles if self._troubles else "No Troubles"


class helpSgltn:
    #Singleton class that contains help text for various nodes
    _instance = None

    def __new__(cls): 
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.get_file()
        return cls._instance
    
    def get_file(self):
        #Open help.json 
        j_mmgr = json_manager()
        help_file = j_mmgr.append_filename_to_path(j_mmgr.script_dir, 'help.json')
        help_data = j_mmgr.load_json(help_file, False)
        self._style_prompt_help = ""
        self._dalle_help = ''
        self._exif_wrangler_help = ''
        # Empty help text is not a critical issue for the app
        if not help_data:
            j_mmgr.log_e('Help data file is empty or missing.',
                         TroubleSgltn.Severity.ERROR)
            return
        #Get help text
        self._style_prompt_help = help_data.get('sp_help','')
        self._exif_wrangler_help = help_data.get('wrangler_help', '')
        self._dalle_help = help_data.get('dalle_help', '')

    @property
    def style_prompt_help(self)->str:
        return self._style_prompt_help
    
    @property
    def exif_wrangler_help(self)->str:
        return self._exif_wrangler_help
    
    @property
    def dalle_help(self)->str:
        return self._dalle_help
    

class json_manager:

    def __init__(self):

        self.trbl = TroubleSgltn()
        
        # Get the directory where the script is located
        # Public properties
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.script_parent, self.script_parent_dirname = self.findParent(self.script_dir)

        self.update_file = os.path.join(self.script_dir, 'update.json')
        self.config_file = os.path.join(self.script_dir, 'config.json')
        self.backup_dir = self.find_child_directory(self.script_dir, 'bkup', True, True)
        self.backup_config_path = os.path.join(self.backup_dir, 'config.json')
        self.comfy_dir = self.find_target_directory(self.script_dir, 'ComfyUI', True)
        self.customnodes_dir = self.find_child_directory(self.comfy_dir, 'custom_nodes', False, True)
        self.temp_dir = self.find_child_directory(self.script_dir, 'temp', True)
        self.log_dir = self.find_child_directory(self.script_dir, 'logs', True)
        self.log_file_name = "Plush-Events"
        #Private Properties
        self._config_bad = os.path.join(self.script_dir, 'config.bad')
        self._update_bad = os.path.join(self.script_dir, 'update.bad')


    def log_events(self, event: str, severity: TroubleSgltn.Severity = TroubleSgltn.Severity.INFO, is_trouble: bool = False,  
                   file_name: Union[str,None] = None, is_critical: bool=False) -> bool:    
        """
        Appends events with prepended timestamp to a text log file.
        Each event is written in a key/value pair format.
        Creates the file if it doesn't exist. Also, prints to console if specified.

        Args:
            event (str): The event information.
            severity (TroubleSgltn.Severity): An Enum indicating the severity of the issue
            file_name (str): The name of the log file. Defaults to self.log_file_name if None.
            is_trouble (bool): Whether to log the event in TroubleSgltn to be presented to the user
        Returns:
            bool: True if successful, False otherwise.
        """
        if file_name is None:
            file_name = self.log_file_name

        if is_trouble:
            self.trbl.log_trouble(event, severity)

        date_time = datetime.now()
        timestamp = date_time.strftime("%Y-%m-%d %I:%M:%S %p") #YYYY/MM/DD, 12 hour AM/PM

       #Create a dict of the log event
        log_event_data = {
            "timestamp": timestamp,
            "severity": severity.name,
            "event": event
        }
        #Conver the dict to a json string using json.dumps to handle invalid chars.
        log_event_json = self.convert_to_json_string(log_event_data, is_logger=True)
        if log_event_json is None:
            return False

        log_file_path = self.append_filename_to_path(self.log_dir, f"{file_name}.log", True)

        # Use the append_to_file utility to write the log event to the file
        success = self.append_to_file(log_event_json, log_file_path, is_critical, is_logger=True)

        return success



    def append_to_file  (self, data:str, file_path: Union[str,Path], is_critical:bool, is_logger:bool = False)->bool:   
        """
        Appends a text string to a file.
        Makes the file if it doesn't exist

        Args:
            data (str):  The text string to append to the file
            file_path (Union[str, Path]): The path and name of the file to which string will be appended
            is_critical (bool): If True raises exceptions for errors

        Returns:
            True if the append action is successful else False
        """
        file_path = Path(file_path)
        try:
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(data + '\n')
                return True
        except (IOError, OSError) as e:
            if not is_logger:
                self.log_events(f"Error writing data to file:  {file_path}: {e}", 
                                TroubleSgltn.Severity.ERROR,
                                True)
            if is_critical:
                raise
            return False


    def findParent(self, child:Union[Path, str], as_string: bool = True)->Union[tuple, None]:
        """
        Returns the parent directory path and parent directory name using pathlib.

        Args:
            child (Union[Path,str]): Path of the directory whose parent is to be found.
            as_string (bool): Determines if the return value will be a Path object or string

        Returns:
            tuple: A tuple containing the path to the parent directory and the parent directory name.
        """
        if child:
            child_path = Path(child).resolve()
            parent_path = child_path.parent
            parent_name = parent_path.name
        else:
            return None

        return (str(parent_path), parent_name) if as_string else (parent_path, parent_name)
    

    def find_target_directory(self, start_path: Union[Path, str], target_dir_name: str, as_string: bool = True) -> Union[Path, str, None]:
        """
        Walks up the directory structure from start_path to find a directory named target_dir_name.

        Args:
            start_path (Union [Path, str]): The starting directory path.
            target_dir_name (str): The target directory name to find.
            s_string (bool): Determines if the return value will be a Path object or string

        Returns:
            Path: A Path object or string of the found directory path, or None if not found.
        """
        current_path = Path(start_path).resolve()

        for parent in current_path.parents:
            if parent.name == target_dir_name:
                return str(parent) if as_string else parent

        return None
    
    def append_filename_to_path(self, directory_path: Union[Path, str], filename: str, as_string: bool = True) -> Union[Path, str]:
        """
        Appends a filename to the given directory path and returns the result as either a Path object or a string.

        Args:
            directory_path (Union[Path, str]): The path of the directory as a Path object or a string.
            filename (str): The filename to append.
            as_string (bool): If True, returns the path as a string.

        Returns:
            Union[Path, str]: The combined path including the filename, either as a Path object or a string.
        """
        combined_path = Path(directory_path) / filename
        return str(combined_path) if as_string else combined_path
    
    def delete_files_by_age(self, file_path: Union[str, Path], file_pattern: str, max_age_days: int=10,  is_critical: bool=False)->bool:
        """
        Deletes files in a specified directory that match a given pattern and are older than a specified age.

        Args:
        - file_path (Union[str, Path]): Path to the directory where files will be deleted.
        - file_pattern (str): The pattern to match files. Can include wildcards, e.g., '*.txt', '*.*', 'myfile.txt'.        
        - max_age_days (int): Maximum age of files to keep. Files older than this will be deleted.
        - is_critical (bool): Whether or not to raise file errors

        Returns:
            - (Bool): True if deletion is successful False if not
        """
        directory = Path(file_path)
        current_time = time.time()
        all_deletions_successful = True

        # Ensure the provided path is a directory
        if not directory.is_dir():
            self.log_events(f"'delete_file_by_age': The path {directory} is not a directory.",
                            TroubleSgltn.Severity.WARNING,
                            True)
            return False

        # Iterate over files matching the pattern
        for file in directory.glob(file_pattern):
            file_age_days = (current_time - file.stat().st_ctime) / (24 * 3600)

            if file_age_days > max_age_days:
                try:
                    file.unlink()
                    self.log_events(f"Deleted file: {file}",is_trouble=True)
                except Exception as e:
                    self.log_events(f"Error deleting file {file}: {e}",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
                    all_deletions_successful = False
                    if is_critical:
                        raise

        return all_deletions_successful
    
   
    def remove_log_entries_by_age(self, log_file_path, days_allowed):
        timestamp_format = "%Y-%m-%d %I:%M:%S %p"
        cutoff_time = datetime.now() - timedelta(days=days_allowed)
        deleted_count = 0
        updated_entries = []
        try:
            with open(log_file_path, "r", encoding='utf-8') as file:
                for line in file:
                    # Skip empty lines
                    if not line.strip():
                        continue
                    log_entry = json.loads(line)
                    entry_time = datetime.strptime(log_entry["timestamp"], timestamp_format)
                    if entry_time > cutoff_time:
                        updated_entries.append(json.dumps(log_entry))
                    else:
                        deleted_count += 1
        except Exception as e:
            self.log_events(f"Error reading log file: {log_file_path}: {e}",
                            TroubleSgltn.Severity.WARNING,
                            True)
            return None
        
        # Use write_string_to_file to write updated entries back
        updated_content = "\n".join(updated_entries) + '\n'
        success = self.write_string_to_file(updated_content, log_file_path)
        if not success:
            self.log_events(f"Failed to write updated log entries back to file: {log_file_path}",
                            TroubleSgltn.Severity.ERROR,
                            True)
            return None
        return(deleted_count)

    
    def generate_unique_filename(self, extension: str, base: str="")->str:
        """
        Generates a unique file name by incorporating Date and Time
        with a base and extension provided by the user

        Args
            extension (str) The file extension
            
            base (str)  The first part of the file name

        Returns:
            A unique filename with a unique numeric value prefaced by the base

        """
        # Get current date and time
        current_datetime = datetime.now()
        # Format the date and time in a specific format, e.g., YYYYMMDD_HHMMSS
        datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")
        # Append this string to your base file name
        unique_filename = f"{base}{datetime_str}.{extension}"
        return unique_filename

        

    def find_child_directory(self, parent: Union[str,Path], child: str, create: bool = False, as_string: bool = True) -> Union[Path, str]:
        """
        Finds a child directory within a given parent directory or optionally creates it if it doesn't exist.

        Args:
            parent (str): The starting directory path.
            child (str): The target child directory name to find or create.
            create (bool): If True, creates the child directory if it does not exist.
            as_string (bool): If True, returns the path as a string; False returns Path Object.

        Returns:
            Union[Path, str]: A Path object or string of the found or created child directory, or None if not found/created.
        """
        parent_path = Path(parent)
        child_path = parent_path / child

        if not child_path.is_dir() and create:
            try:
                child_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                self.log_events(f"Error creating directory {child_path}: {e}",
                                TroubleSgltn.Severity.WARNING,
                                True)
                return ""

        return str(child_path) if as_string else child_path


    # Load a file
    def load_json(self, ld_file: Union[str,Path], is_critical: bool=False):

        """
        Loads a JSON file.

        Args:
            ld_file (str): Path to the JSON file to be loaded.
            is_critical (bool): If True, raises exceptions for errors.

        Returns:
            dict or None: The loaded JSON data (dict) if valid, None otherwise.
        """
        try: 
            with open(ld_file, 'r', encoding='utf-8') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            self.log_events(f'JSON syntax error in: {ld_file}: {e}',
                            TroubleSgltn.Severity.WARNING,
                            True)
                        
        except FileNotFoundError:
            self.log_events(f"File not found: {ld_file}",
                            TroubleSgltn.Severity.WARNING,
                            True)
        except Exception as e:
            self.log_events(f"Plush - An unexpected error occurred while reading {ld_file}: {e}",
                            TroubleSgltn.Severity.WARNING,
                            True)
        
        if is_critical:
            raise
        
        return None

        
    def write_json(self, data: dict, file_path: str, is_critical: bool=False):
        """
        Writes a Python dictionary to a JSON file.

        Args:
            data (dict): The data to write to the JSON file.
            file_path (str): The path of the file to be written.
            is_critical (bool): If True raises exceptions for errors

        Returns:
            bool: True if write operation was successful, False otherwise.
        """
        try:
            with open(file_path, 'w',encoding='utf-8') as file:
                json.dump(data, file, indent=4)
            return True
        except TypeError as e:
            self.log_events(f"Plush - Data type not serializable in {file_path}: {e}",
                            TroubleSgltn.Severity.WARNING,
                            True)
        except Exception as e:
            self.log_events(f"Plush - An error occurred while writing to JSON: {file_path}: {e}",
                            TroubleSgltn.Severity.WARNING,
                            True)

        if is_critical:   
            raise

        return False
    

    def write_string_to_file(self, data: str, file_path: Union[str,Path], is_critical: bool=False)->bool:
        """
        Writes any string data to a file.  Including JSON strings.

        Args:
            data (str): The string to write to the file.
            file_path (str): The path and name of the file to write.
            is_critical (bool): If True, raises exceptions for errors

        Returns:
            bool: True if the write operation was successful, False otherwise.
        """
        if data:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(data)
                return True
            except (IOError, OSError) as e:
                self.log_events(f"Plush - An error occurred while writing to file: {file_path}: {e}",
                                TroubleSgltn.Severity.WARNING,
                                True)
                if is_critical:
                    raise
                return False
        else:
            return False
        

    def update_json_data(self, upd_data: dict, cfg_data: dict):

        """
        Update config_data with new key-value pairs from update_data.

        Args:
            config_data (dict): The original configuration data.
            update_data (dict): Data containing potential new keys and values.
        Note: 
            Handle the 'style' list differently from the rest of the fields
            it will have individual list items added or removed in alpha order.
            upd_data style[] items that start with '-' will be removed

        Returns:
            dict: The updated configuration data.
        """
        for key, value in upd_data.items():
            if key != 'style':
                cfg_data[key] = value

        if 'style' in upd_data:
            for item in upd_data['style']:
                if item.startswith('-'):
                    #Remove item (strip the "-" prefix before removing item[1:])
                    remove_item = item[1:]
                    if remove_item in cfg_data['style']:
                        cfg_data['style'].remove(remove_item)
                else:
                    #Add item(s) in alpha sort order
                    position = bisect.bisect_left(cfg_data['style'], item)
                    #check if the postion for the new item is not beyond eof, and that's it's not a duplicate
                    if position >= len(cfg_data['style']) or cfg_data['style'][position] != item:
                        cfg_data['style'].insert(position, item)


        return cfg_data
    
    
    
    def extract_from_dict(self, dict_data:dict, target:list)->dict:
        """
        A recursive method that extracts data from a dict by finding keys that meet the criteria in the target argument, and 
        returns them and their values in a new dict.  Duplicate keys have their values stored in a list under the key.  
        JSON strings are coerced to a dictionary object.
            Args:
                dict_data (dict):  The dictionary to be searched for matching values
                target (list):  A list of search values 

            Returns:
                A dictionary containg the dicts whose keys match the criteria and lists that either hold values from 
                duplicate keys, or had elements that matched the criteria
        """

        def find_it(data, search_key, new_dict):
                    
                    for k, v in data.items():
                        if k == search_key:

                            if k in new_dict: #key is a duplicate
                                if isinstance(new_dict[k], list): #if the new item is a list
                                    new_dict[k].append(v)#append the list with the dupe key
                                else:
                                    new_dict[k] = [new_dict[k], v] #Convert it to a list and append it
                            else:
                                new_dict[k] = v                            
                        
                        elif isinstance(v, str):
                            v = v.strip()
                            if v.startswith('{') or v.startswith('['):
                                try:
                                    parsed = json.loads(v)
                                    find_it(parsed, search_key, new_dict) 
                                except json.JSONDecodeError:
                                    #Whoops it's not a JSON string
                                    self.log_events(f"Attempt to convert string to dictionary failed, some data will be missing:  {v}",
                                                    TroubleSgltn.Severity.WARNING,
                                                    True)
                                    continue
                        elif isinstance(v, dict):
                             find_it(v, search_key, new_dict)
                        elif isinstance(v, list):                            
                            for i in v:                                
                                if i == search_key:                                  
                                    new_dict[k] = v
                                if isinstance(i, dict):
                                     find_it(i, search_key, new_dict)
              
        new_dict = {}
        local_source = dict_data

        if isinstance(target, list):
            for search_key in target:
                find_it(local_source, search_key, new_dict)
        else:
            self.log_events(f"'extract_from_dict', Incoming search terms were not a list object. Return empty dict.",
                            TroubleSgltn.Severity.WARNING,
                            True)
        return new_dict
    

    #**testing

    def extract_with_translation(self, dict_data: dict, translate_keys: dict, min_prompt_len:int=1, alpha_pct:float=0.0, filter_phrase:str ="") -> dict:
        """
        A recursive method that extracts and translates keys from a dict by finding keys that match those in the 
        translate_keys argument, and returns them with their values in a new dict using the friendly names. 
        Duplicate keys have their values stored in a list under the friendly name key. JSON strings are coerced to 
        dictionary objects.  Possible Prompts have to meet the additional criteria of having a min length limit and
        a max percent of numeric characters limit.
            Args:
                dict_data (dict): The dictionary to be searched.
                translate_keys (dict): A dictionary with original keys as keys and friendly names as values.
                min_prompt_len (int): The minimum length of a string to qualify as a Possible Prompt
                alpha_pct (float): The minimum percentage of alpha characters (+ comma) to qualify as a Possible Prompt
                filter_phrase (string): A string whose exact match must be present in order to qualify as a Possible Prompt
            Returns:
                A dictionary containing the translated keys and their values, with lists for duplicate keys or matched elements.
        """
        
        new_dict = {}

        def custom_sort(item):
            key = item[0]
            # Assign a high priority to 'Positive Prompts'
            if key == 'Possible Prompts':
                return (0, key)
            elif key == "Seed":
                return (1, key)
            elif key == 'Source File':
                return(3, key) #low priority for processing info
            elif key == 'Processing Application':
                return(3, key)
            # Normal priority for everything else
            return (2, key)
        

        def process_and_divide(friendly_name, value):
            # This function takes a string and performs mathematical division and APEX translation 
            #if the entire string matches the pattern
            pattern = re.compile(r'^(\d+)/(\d+)$')  # Anchors added to match the entire string
            match = pattern.match(value)
            if match:
                numerator, denominator = map(int, match.groups())  # Extract and convert the numbers
                if denominator != 0:  # Avoid division by zero
                    quotient = numerator / denominator
                    
                    # Apply specific logic for APEX values (e.g., Shutter Speed)
                    if "Shutter Speed" in friendly_name:
                        shutter_speed = 2**(-quotient)
                        if shutter_speed < 1:
                            # Convert to 1/x format for speeds faster than 1 second
                            reciprocal = round(1 / shutter_speed)
                            return f"1/{reciprocal} sec"
                        else:
                            # For 1 second or slower, simply round and add " sec"
                            return f"{round(shutter_speed,2)} sec"
                    elif "Aperture" in friendly_name:
                        # Example for Aperture, if needed, adjust formula accordingly
                        return f"F{math.sqrt(2**quotient):.2f}"
                    elif "Exposure Time" in friendly_name:
                        return(f"{str(round(quotient,5))} sec")
                    else:
                        # For other values, simply return the quotient rounded to two decimal places
                        return str(round(quotient, 2))
                else:
                    return "0"  # Return a default string representation for zero if denominator is zero
            else:
                return value  # Return the original value if no match is found

        
        def filter_prompt_items(items, min_prompt_len, alpha_pct, filter_phrase):
            """Recursively filter potential prompts based on length and numeric character ratio criteria, including in nested lists."""
            filtered_items = []
            filter_phrase_lower = filter_phrase.lower()

            def calculate_numeric_ratio(s: str) -> float:
                """Calculate the ratio of numeric characters in a string to the string's total length."""
                numeric_count = sum(c.isdigit() for c in s)
                return numeric_count / len(s) if s else 0
            
            def calculate_prompt_char_ratio(s: str) -> float:
                """Calculate the ratio of common prompt chars: alphabetical characters, spaces and commas to the string's total length."""
                alpha_space_count = sum(c.isalpha() or c.isspace() or c== ',' for c in s)
                return alpha_space_count / len(s) if s else 0


            def filter_recursive(item):
                if isinstance(item, str):
                    if len(item.strip()) >= min_prompt_len and calculate_prompt_char_ratio(item) >= alpha_pct and filter_phrase_lower in item.lower():
                        return item
                elif isinstance(item, list):
                    filtered_sublist = [filter_recursive(subitem) for subitem in item]
                    filtered_sublist = [subitem for subitem in filtered_sublist if subitem is not None]  # Remove None values
                    if filtered_sublist:
                        return filtered_sublist
                return None

            for item in items:
                result = filter_recursive(item)
                if result is not None:
                    filtered_items.append(result)

            return filtered_items


        def find_and_translate(data, translate_dict,friendly_name):
            try:
                if isinstance(data, dict):
                    for k, v in data.items():
                        if k in translate_dict:  # Key matches one we're looking for
                            friendly_name = translate_dict[k]  # Get the friendly name
                            
                            # Special handling for 'Possible Prompt' and Exif info
                            if friendly_name == 'Possible Prompts':
                                if isinstance(v, list):
                                    v = filter_prompt_items(v, min_prompt_len, alpha_pct, filter_phrase)
                                elif isinstance(v, str):
                                    # Wrap the string in a list to use the same filtering logic
                                    filtered_result = filter_prompt_items([v], min_prompt_len, alpha_pct, filter_phrase)
                                    v = filtered_result[0] if filtered_result else None  # Unwrap if not empty
                            elif "Exif" in k or "Xmp" in k:
                                # Apply division processing based on perform_math flag
                                if isinstance(v, str) and v.strip():
                                    v = process_and_divide(friendly_name, v)

                            #Append items if key is a duplicate
                            if friendly_name in new_dict:  # Key is a duplicate   
                                if isinstance(new_dict[friendly_name], list):
                                    new_dict[friendly_name].extend(v if isinstance(v, list) else [v])
                                else:
                                    new_dict[friendly_name] = [new_dict[friendly_name], v] if isinstance(v, list) else [new_dict[friendly_name]] + [v]
                            else:
                                if v:
                                    new_dict[friendly_name] = v

                        elif isinstance(v, str) and v:  # Check for nested JSON strings
                            if v.startswith('{') or v.startswith('['):
                                try:
                                    parsed = json.loads(v)
                                    find_and_translate(parsed, translate_dict, friendly_name)
                                except json.JSONDecodeError:
                                    self.log_events(f"JSON conversion failed: {v}", 
                                                    TroubleSgltn.Severity.WARNING,
                                                    True)
                        elif isinstance(v, dict) and v:  # Nested dict
                            find_and_translate(v, translate_dict, friendly_name)
                        elif isinstance(v, list) and v:  # List, could contain dicts
                            for i in v:
                                if isinstance(i, dict):
                                    find_and_translate(i, translate_dict, friendly_name)
                                elif isinstance(i,list):
                                    find_and_translate(i, translate_dict,friendly_name)
                        elif isinstance(v, tuple) and v:
                            processed_elements = []
                            for i in v:
                                if isinstance(i, dict):
                                    # Create a temporary dictionary to hold processed nested dictionaries
                                    temp_dict = {}
                                    find_and_translate(i, translate_dict, friendly_name)
                                    processed_elements.append(temp_dict)
                                elif k in translate_dict:
                                    # Directly append simple values within the tuple
                                    processed_elements.append(i)
                            # Append the processed elements to new_dict under the corresponding friendly name
                            # No need for a second check for k in translate_keys here
                            if k in translate_dict:
                                friendly_name = translate_dict[k]
                                self.log_events(f"Found tuple: {friendly_name}: {processed_elements}",
                                                is_trouble=True)
                                new_dict[friendly_name] = tuple(processed_elements)
                elif isinstance(data, list):
                    for item in data:
                        find_and_translate(item, translate_keys,friendly_name)  # Recursively handle items in lists
            except Exception as e:
                self.log_events(f"An unexpected error occurred during data translation {str(e)}",
                                    TroubleSgltn.Severity.ERROR,
                                    True)
                #End of find_and_translate

        if not isinstance(dict_data, dict) or not isinstance(translate_keys, dict):
            self.log_events("Improper data object passed to 'extract_with_translation', translation halted!",
                            TroubleSgltn.Severity.ERROR,
                            True)
            return(new_dict) #Return an empty dict if passed objects are invalid
        friendly_name = ""
        find_and_translate(dict_data, translate_keys,friendly_name)
        sorted_items = dict(sorted(new_dict.items(), key=custom_sort))
        return sorted_items


    def prep_formatted_file(self, parsed_dict):
        formatted_file = ""
        bullet = '➤ '  # Using a bullet point for headings \u27a4
        sub_bullet = '    • '  # Adding indentation before the sub_bullet for items \u2022
        sub_open_bracket = '['
        sub_close_bracket = ']'
        newlines = '\n'
        
        def process_item(key, item):
            """Flatten and format items from lists, tuples, or simple values, adding them to formatted_file."""
            if isinstance(item, (str, int, float)) and str(item).strip():
                if not 'Possible Prompts' in key:
                    return f"  {sub_open_bracket}{item}{sub_close_bracket}"
                else:
                    return f"{newlines}{sub_bullet}{item}{newlines}"
            elif isinstance(item, (list, tuple)):
                # Flatten nested lists/tuples and format their items
                return ''.join(process_item(key, subitem) for subitem in item)
            else:
                return ''  # Return an empty string for unsupported types or to skip processing
        
        for key, value in parsed_dict.items():
            formatted_file += f"{newlines}{newlines}{bullet} {key}:"
            formatted_file += process_item(key,value)

        return formatted_file.strip()  # Remove trailing newlines for cleaner output
    

    def remove_duplicates_from_keys(self, data_dict: dict, keys_to_check: list) -> None:
        """
        Removes duplicate entries from specified keys in the provided dictionary in-place.
        
        Args:
            data_dict (dict): The dictionary from which duplicates under specified keys should be removed.
            keys_to_check (list): A list of keys in the dictionary for which duplicates should be removed.
        """
        def flatten_list(nested_list):
            """Flatten a nested list of lists."""
            for item in nested_list:
                if isinstance(item, list):
                    yield from flatten_list(item)
                else:
                    yield item

        for key in keys_to_check:
            if key in data_dict and isinstance(data_dict[key], list):
                # Flatten the list if it contains nested lists and collect unique elements
                flattened_items = list(flatten_list(data_dict[key]))
                unique_items = []
                for item in flattened_items:
                    if item not in unique_items:
                        unique_items.append(item)
                # Replace the original list with the deduplicated and flattened list
                data_dict[key] = unique_items


    
    #**end testing
    def remove_keys_from_nested_json(self, dictionary: dict, keys_to_remove: list):
        """
        Removes key/value pairs from a nested dict/json that are extraneous to the task at hand
            based on the key values passed into keys_to_remove.  
            This method calls 'remove_keys_from_dict'
        Args:
            dictionary (dict): The json/dict from which k/v's will be removed
            keys_to_remove (list): A list of keys denoting which k/v's will be removed

        Return:  This is an in-place modification no explicit return value
        """
        keys_to_remove_set = set(keys_to_remove) 
        for key, value in list(dictionary.items()):
            if isinstance(value, str)and value.startswith('{') and value.endswith('}'):
                try:
                    # Attempt to deserialize the string value into a dictionary
                    nested_dict = json.loads(value)
                    self.remove_keys_from_dict(nested_dict, keys_to_remove_set)
                    # Serialize the dictionary back into a string
                    serialized_dict = self.convert_to_json_string(nested_dict)
                    if serialized_dict is not None:
                        dictionary[key] = serialized_dict
                except json.JSONDecodeError:
                    # Log JSON deserialization error
                    self.log_events(f"JSON deserialization error in 'remove_keys_from_nested_json': Key: {key}, Value: {value}",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
                except Exception as e:
                    # Log other unexpected errors
                    self.log_events(f"Unexpected error converting string data in 'remove_keys_from_nested_json': {e}",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
            elif isinstance(value, dict):
                self.remove_keys_from_dict(value, keys_to_remove_set)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self.remove_keys_from_dict(item, keys_to_remove_set)


    
    def remove_keys_from_dict(self, dict_data:dict, keys_to_remove:Union[list,set]):
        """
        Removes key/value pairs from a dict/json that are extraneous to the task at hand
            based on the key values passed into keys_to_remove
        Args:
            dict_data (dict): The json/dict from which k/v's will be removed
            keys_to_remove (list): A list of keys denoting which k/v's will be removed

        Return:  This is an in-place modification no explict return value
        """
        if isinstance(dict_data, dict):
            for key in list(dict_data.keys()):
                if key in keys_to_remove:
                    del dict_data[key]
                else:
                    self.remove_keys_from_dict(dict_data[key], keys_to_remove)
        elif isinstance(dict_data, list):
            for item in dict_data:
                self.remove_keys_from_dict(item, keys_to_remove)
    
    
    
    def convert_to_json_string(self, data, pretty=False, is_logger:bool = False) -> Optional[str]:
        """
        Converts a python serializable python object: dict, etc to a JSON string

        Args: 
            data: (Obj) A serializable python object suitable for conversion to a JSON string
            pretty: (bool) Make the json result more human readable

        Returns:
            json: in string format
            None: if conversion fails with an error
        """
        try:
            if pretty:
                jstring = json.dumps(data, indent=4, default=self.custom_serializer)
            else:
                jstring = json.dumps(data, default=self.custom_serializer)
            return jstring
        except TypeError as e:
            if not is_logger:
                self.log_events(f"Error converting object to JSON string{e}",
                                TroubleSgltn.Severity.WARNING,
                                True)

            return None
        
    def custom_serializer(self, obj)-> object:
        """
        Handles unserializable objects rejected by json.dumps.  Attempts to parse
        the object's data structures and convert byte code to text

        Args:
            obj (object): The unserializable object rejected by json.dumps

        Returns:
            Object: Object containing decoded byte code and other serializable data
        """

        def decode_integer_list(data):

            # Check if all data items are in the valid byte range
            if all(0 <= item <= 255 for item in data):
                byte_data = bytes(data)
                # ... existing decoding logic ...
            else:
                # Handle data with out-of-range values
                # For simplicity, converting each item to string, but you can adjust as needed
                self.log_events(f"Plush - Object has integer value that's out of bounds for byte conversion, 'decode_integer_list''")
                return [str(item) if not (0 <= item <= 255) else item for item in data]
            #Iterate through a list of Unicode Transformation Formats
            #testing each one against is_meaningful() until we find one that 
            #Produces readable text
            encodings = ['utf-8', 'utf-16be', 'utf-16le']

            # Check for common headers like ASCII or UNICODE and strip them        
            if byte_data.startswith(b'ASCII\x00\x00\x00'):
                byte_data = byte_data[8:]
            elif byte_data.startswith(b'UNICODE\x00\x00'):
                byte_data = byte_data[9:]
            else:
                # If no known header is present, return the original data
                #return data
                pass

            for encoding in encodings:
                try:
                    # Attempt to decode the byte data using the current encoding
                    
                    decoded_string = byte_data.decode(encoding, errors='replace')
                    if is_meaningful(decoded_string, data, encoding):
                        return decoded_string
                except UnicodeDecodeError:
                    # If a UnicodeDecodeError occurs, move to the next encoding
                    continue

            # If no encoding was successful, print an error and return the original data
            
            self.log_events(f"Plush - Unable to determine text format for object's.value: {data}")
            #return byte_data.hex()
            if len(str(data)) <=30:
                return data
            else:
                return "[Indecipherable and excessive length]"

            
        def is_meaningful(s, data, encoding):
            # This function assumes English Only
            null_count = s.count("\u0000")
            data_len = len(s)
            

            if data_len < 1:
                return False

            null_ratio = null_count/data_len
            ascii_ratio = sum(c < '\x80' for c in s) / data_len
            printable_ratio = sum(c.isprintable() for c in s) / data_len

            if len(s) <= 5:
                # For short strings, check if they are fully printable
                if printable_ratio < 1 or ascii_ratio < .8: #Adjust as needed
                    return False
            else:
                # Check for a high proportion of printable characters
                if ascii_ratio < 0.65 or printable_ratio < 0.55: #Adjust as needed
                    return False
                
            
            return True
        
        #First handle specifid object types.
        if "IFDRational" in str(type(obj)):
            return(str(obj))
        
        elif "exifread.utils.Ratio" in str(type(obj)):
            ratio_dict = {"numerator": obj.numerator, "denominator": obj.denominator}
            return ratio_dict
        
        elif "IfdTag" in str(type(obj)):
            # Process the values attribute
            processed_values= {}
            attribs = ["values","printable"]
            for attrib in attribs:
                data = getattr(obj, attrib)
                if isinstance(data, list) and all(isinstance(x, int) for x in data):
                    # Try decoding as encoded text
                    processed_values[attrib] = decode_integer_list(data)
                else:
                    processed_values[attrib] = data

            return ({
                "tag": obj.tag,
                "field_type": obj.field_type,
                "values": processed_values["values"],
                "printable": processed_values["printable"]
            })
        else:
            self.log_events(f"Unhandled Object in custom_serializer: {str(type(obj))}",
                            is_trouble=True)


        
        # lastly, handle lists and dicts that might contain out of range byte data
        if isinstance(obj, list):
            # Filter out values not in byte range and convert the rest to bytes
            byte_data = bytes(b for b in obj if 0 <= b <= 255)
            return byte_data.decode('utf-8', errors='replace')  # or use other decoding logic
        
        if isinstance(obj, dict):
            processed_dict = {}
            for key, value in obj.items():
                # Process each value in the dictionary
                if isinstance(value, (bytes, bytearray)):
                    processed_dict[key] = value.decode('utf-8', errors='replace')
                elif isinstance(value, int) and not 0 <= value <= 255:
                    # Handle integers not in byte range
                    processed_dict[key] = str(value)  # or some other handling
                else:
                    # Recursively process other values
                    processed_dict[key] = self.custom_serializer(value)
            return processed_dict
        
        else:
            self.log_events(f"Unhandled Object in custom_serializer: {str(type(obj))}",
                            is_trouble=True)
            return str(obj)
            

    def convert_from_json_string(self, json_string: str, is_critical:bool=False) -> Optional[Any]:
        """
        Converts a JSON string to a Python object (like a dict).

        Args: 
            json_string: (str) A JSON string to be converted into a Python object.

        Returns:
            Python object (like a dict) if conversion is successful.
            None if conversion fails with an error.
        """
        try:
            data = json.loads(json_string)
            return data
        except json.JSONDecodeError as e:
            self.log_events(f"Plush - Error converting JSON string to Python object: {e}", 
                            TroubleSgltn.Severity.WARNING,                           
                            True)

            if is_critical:
                raise
            return None
        


        
    # Warn then delete OpenAI API keys
    def _del_keys(self, d_data, d_file):

        """
        Initially replaces the key field value with a warning, then eventually it will delete the field.

        Args:
            d_data: a Python data file with the contents of the json file from which the 'key' field will be deleted

            d_file: a path to the file from which the 'key' field will be deleted

        """

        if 'key' in d_data:
            #del d_data['key']
            d_data['key'] = "STOP!!  You need to setup your OpenAI API key in an Environment Variable (see the ReadMe file).  This JSON will no longer be used for the API key."
        if 'sp_help' in d_data:
            del d_data['sp_help']

        self.write_json(d_data, d_file)



    def on_startup(self, keep_key: bool = True, max_log_age: int = 32):
        """
        Updates the configuration file by applying changes from 'update.json'

        Args:
            keep_key (bool): If True retains the 'key' entry in the config.json file
                            If False, removes it and relies on an Environment Variable for that value
                            Default is True for safety

        Returns:
            bool: True if an update is queued, represents a new version, and was successful
                False otherwise.
        """
                #pare down log file
        log_file = self.append_filename_to_path(self.log_dir,self.log_file_name + '.log')
        num_removed = self.remove_log_entries_by_age(log_file, max_log_age)
        if not num_removed == None:
            self.log_events(f'{num_removed} old entries were removed from the log file: {self.log_file_name}')
        else:
            if os.path.exists(log_file):
                    os.remove(log_file)
            self.log_events(f'Log file {self.log_file_name} was unable to be processed for old entries.  File was corrupt and was deleted',
                            TroubleSgltn.Severity.ERROR)
            

        # Check for config.json
        if not os.path.exists(self.config_file):

            # Check if there's a backup of config.json in the bkup directory
            if os.path.isfile(self.backup_config_path):
                # Copy the backup config.json to ComfyUI_plush directory
                shutil.copy(self.backup_config_path, self.config_file)
                self.log_events("Plush - missing config.json, getting backup file",
                                severity=TroubleSgltn.Severity.WARNING)
            else:
                # No config.json or backup
                self.log_events("Plush - Missing config.json and missing backup",
                                severity=TroubleSgltn.Severity.ERROR)
                raise FileNotFoundError("Plush - The config.json file is missing and there is no backup")
                #return False

        # load config.json and 
        # handle Config opening errors
        config_data = self.load_json(self.config_file)
        if config_data is None:
            #Restore from backup and retry
            self.log_events("Plush - Primary config.json was corrupt, attempting to use backup",
                            severity=TroubleSgltn.Severity.WARNING)

            #rename the corrupt config.json config.bad
            try:
                if os.path.exists(self._config_bad):
                    os.remove(self._config_bad)
                os.rename(self.config_file,self._config_bad)
            except PermissionError:
                self.log_events("Plush - Permission denied while renaming config.json")
            except os.error as e:
                self.log_events(f"Plush - Error renaming corrupt config.json: {e} ")

            if os.path.isfile(self.backup_config_path):
                shutil.copy(self.backup_config_path, self.config_file)
                try:
                    config_data = self.load_json(self.config_file, True)
                except Exception as e:
                    self.log_events(f"Plush - Error, invalid primary and backup config JSON file: {self.config_file} {e}",
                                    severity=TroubleSgltn.Severity.ERROR)
                    return False
            #if there's no backup config.json
            else:
                self.log_events("Plush - No backup available and a corrupted config.json",
                                severity=TroubleSgltn.Severity.ERROR)
                return False

        # Check for update.json
        if not os.path.exists(self.update_file):
            return False

        # Load update.json
        try:
            update_data = self.load_json(self.update_file, True)
        except json.JSONDecodeError as e:
            self.log_events(f"Plush - Error decoding JSON: {self.update_file}: {e}")
            os.rename(self.update_file, self._update_bad)
            return False
        except Exception as e:
            self.log_events(f" Unexpected error occurred while opening {self.update_file}: {e}",
                            severity=TroubleSgltn.Severity.ERROR)
            return False
        
        # Conditionally delete the "key" key/value pair in both the config.json and its backup file
        # once transition to env variable is complete
        if not keep_key:
            try:
                self.remove_keys_from_dict(config_data,remove_keys)
                self.remove_keys_from_dict(update_data, remove_keys)
            except Exception as e:
                self.log_events(f"Error while deleting keys or updating files to remove keys: {e}")


        # Version comparison and backup
        update_version = update_data.get('version', 0)
        config_version = config_data.get('version', 0)

        # Test if user has locked their config.json
        # If so exit with False
        if 'locked' in config_data and config_data['locked']:
            self.log_events("Plush - The configuration file is locked and cannot be updated")
            return False

        if config_version < update_version:

            # Conditionally delete the "key" key/value pair in both the config.json and its backup file
            # once transition to env variable is complete
            if not keep_key:
                remove_keys = ['key','sp_help']
                try:
                    self.remove_keys_from_dict(config_data,remove_keys)
                except Exception as e:
                    self.log_events(f"Error while deleting keys or updating files to remove keys: {e}")

            # Backup current config

            try: 
                shutil.copy(self.config_file, os.path.join(self.backup_dir, 'old_config.json'))
            except Exception as e:
                    self.log_events(f"Plush - Failed to create backup of old config.json file: {e}")

            # Add any new fields and their values from updata_data that weren't already in config_data
            # Also update any existing keys.
            config_data = self.update_json_data(update_data,config_data)

            # Write updated config back to file
            try: 
                self.write_json(config_data, self.config_file,True)
                self.log_events("Plush - Successfully updated config.json")
            except Exception as e:
                    self.log_events(f"Plush - Update of config.json failed on file write: {e}",
                                    severity=TroubleSgltn.Severity.ERROR)
                    return False
            
            #Update the config.json backup to the new data
            if not self.write_json(config_data, self.backup_config_path):
                self.log_events(f"Plush - Failed to update backup config.json on file write: {e}",
                                severity=TroubleSgltn.Severity.WARNING)

            #shutil.move(update_file, os.path.join(backup_dir, update_file_name))

            return True
        

        return False
