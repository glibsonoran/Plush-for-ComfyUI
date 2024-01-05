
import json
import os
import shutil
import bisect

class json_manager:

    def __init__(self):
        
        # Get the directory where the script is located
        # Public properties
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        self.update_file = os.path.join(self.script_dir, 'update.json')
        self.config_file = os.path.join(self.script_dir, 'config.json')
        self.backup_dir = os.path.join(self.script_dir, 'bkup')
        self.backup_config_path = os.path.join(self.backup_dir, 'config.json')
        #Private Properties
        self._config_bad = os.path.join(self.script_dir, 'config.bad')
        self._update_bad = os.path.join(self.script_dir, 'update.bad')

    # Load a file
    def load_json(self, ld_file: str, is_critical: bool=False):

        """
        Loads a JSON file.

        Args:
            ld_file (str): Path to the JSON file to be loaded.
            is_critical (bool): If True, raises exceptions for errors.

        Returns:
            dict or None: The loaded JSON data (dict) if valid, None otherwise.
        """
        try: 
            with open(ld_file, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            print(f"Plush - JSON syntax error in {ld_file}: {e}")
        except FileNotFoundError:
            print(f"Plush - File not found: {ld_file}")
        except Exception as e:
            print(f"Plush - An unexpected error occurred while reading {ld_file}: {e}")
        
        if is_critical:
            raise
        
        return None

        
    def write_json(self, data: dict, file_path: str, is_critical: bool=False):
        """
        Writes a Python dictionary to a JSON file.

        Args:
            data (dict): The data to write to the file.
            file_path (str): The path of the file to write to.
            is_critical (bool): If True raises exceptions for errors

        Returns:
            bool: True if write operation was successful, False otherwise.
        """
        try:
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            return True
        except TypeError as e:
            print(f"Plush - Data type not serializable in {file_path}: {e}")
        except Exception as e:
            print(f"Plush - An error occurred while writing to {file_path}: {e}")

        if is_critical:   
            raise

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

        
    # Warn then delete OpenAI API keys
    def _del_keys(self, d_data, d_file):

        """
        Initially replaces the key field value with a warning, then eventually it will delete the field.

        Args:
            d_data: a Python data file with the contents of the json file from which the 'key' field will be deleted

            d_file: a path to the file from the 'key' field will be deleted

        """

        if 'key' in d_data:
            #del d_data['key']
            d_data['key'] = "STOP!!  You need to setup your OpenAI API key in an Environment Variable (see the ReadMe file).  This JSON will no longer be used for the API key."

        self.write_json(d_data, d_file)



    def update_config(self, keep_key: bool = True):
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

        # Check for config.json
        if not os.path.exists(self.config_file):

            # Check if there's a backup of config.json in the bkup directory
            if os.path.isfile(self.backup_config_path):
                # Copy the backup config.json to ComfyUI_plush directory
                shutil.copy(self.backup_config_path, self.config_file)
                print("Plush missing config.json getting backup file")
            else:
                # No config.json or backup
                print("Plush - Missing config.json and missing backup")
                raise FileNotFoundError("Plush - he config.json file is missing and there is no backup")
                #return False

        # load config.json and 
        # handle Config opening errors
        config_data = self.load_json(self.config_file)
        if config_data is None:
            #Restore from backup and retry
            print("Plush - Primary config.json was corrupt, attempting to use backup")

            #rename the corrupt config.json config.bad
            try:
                if os.path.exists(self._config_bad):
                    os.remove(self._config_bad)
                os.rename(self.config_file,self._config_bad)
            except PermissionError:
                print("Plush - Permission denied while renaming config.json")
            except os.error as e:
                print(f"Plush - Error renaming corrupt config.json: {e} ")

            if os.path.isfile(self.backup_config_path):
                shutil.copy(self.backup_config_path, self.config_file)
                try:
                    config_data = self.load_json(self.config_file, True)
                except Exception as e:
                    print(f"Plush - Error, invalid primary and backup config JSON file: {self.config_file} {e}")
                    return False
            #if there's no backup config.json
            else:
                print("Plush - No backup available and a corrupted config.json")
                return False

        # Conditionally delete the "key" key/value pair in both the config.json and its backup file
        # once transition to env variable is complete
        if not keep_key:
            try:
                self._del_keys(config_data, self.config_file)
                bkup_data = self.load_json(self.backup_config_path)
                self._del_keys(bkup_data, self.backup_config_path)
            except Exception as e:
                print(f"Plush - Error while deleting keys or updating files to remove keys: {e}")


        # Check for update.json
        if not os.path.exists(self.update_file):
            return False

        # Load update.json
        try:
            update_data = self.load_json(self.update_file, True)
        except json.JSONDecodeError as e:
            print(f"Plush - Error decoding JSON: {self.update_file}: {e}")
            os.rename(self.update_file, self._update_bad)
            return False
        except Exception as e:
            print(f"Plush - Unexpected error occurred while opening {self.update_file}: {e}")
            return False


        # Version comparison and backup
        update_version = update_data.get('version', 0)
        config_version = config_data.get('version', 0)

        # Test if user has locked their config.json
        # If so exit with False
        if 'locked' in config_data and config_data['locked']:
            print("Plush - The configuration file is locked and cannot be updated")
            return False

        if config_version < update_version:
            # Backup current config
            if not os.path.exists(self.backup_dir):
                os.makedirs(self.backup_dir)

            try: 
                shutil.copy(self.config_file, os.path.join(self.backup_dir, 'old_config.json'))
            except Exception as e:
                    print(f"Plush - Failed to create backup of old config.json file: {e}")

            # Add any new fields and their values from updata_data that weren't already in config_data
            # Also update any existing keys.
            config_data = self.update_json_data(update_data,config_data)

            # Write updated config back to file
            try: 
                self.write_json(config_data, self.config_file,True)
                print("Plush - Successfully updated config.json")
            except Exception as e:
                    print(f"Plush - Update of config.json failed on file write: {e}")
                    return False
            
            #Update the config.json backup to the new data
            if not self.write_json(config_data, self.backup_config_path):
                print(f"Plush - Failed to update backup config.json on file write: {e}")

            #shutil.move(update_file, os.path.join(backup_dir, update_file_name))

            return True
        

        return False
