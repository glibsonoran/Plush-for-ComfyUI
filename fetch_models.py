
# ------------------------
# Standard Library Imports
# ------------------------
import os
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, Optional

# -------------------------
# Third-Party Library Imports
# -------------------------
import openai
from groq import Groq
#import google.generativeai as genai

# -----------------------
# Local Module Imports
# -----------------------
from .mng_json import json_manager, TroubleSgltn  # add .
from .utils import CommUtils

class RequestMode(Enum):
    OPENAI = 1
    OPENSOURCE = 2
    OOBABOOGA = 3
    CLAUDE = 4
    GROQ = 5
    GEMINI = 6
    OSSIMPLE = 7
    LMSTUDIO = 8
    OLLAMA = 9
    DALLE = 10
    OPENROUTER = 11
    REMOTE = 12

class ModelFetchStrategy(ABC):
 
    def __init__(self)->None:
        self.j_mngr = json_manager()
        self.utils = ModelUtils()

    @abstractmethod
    def fetch_models(self, key, api_obj, **kwargs):
        pass

    def extract_nested_value(self, data, key_path: str):
        """
        Extract nested values from a dict (or list of dicts) using a dot-delimited key path.
        Supports wildcards using '*' to iterate over all keys in a dict or all items in a list.
        
        For example, with key_path "models.text.*.name" it will:
        - Go to data["models"]
        - Then data["models"]["text"]
        - Then iterate over all values in that dict (because of '*')
        - Then extract each value's "name" key
    """
        keys = key_path.split(".")

        def _extract(obj, keys):
            if not keys:
                return obj

            current_key = keys[0]

            # Handle the wildcard: iterate over all elements of the current object.
            if current_key == "*":
                results = []
                if isinstance(obj, dict):
                    for val in obj.values():
                        extracted = _extract(val, keys[1:])
                        if extracted is not None:
                            results.append(extracted)
                elif isinstance(obj, list):
                    for item in obj:
                        extracted = _extract(item, keys[1:])
                        if extracted is not None:
                            results.append(extracted)
                return results

            # Not a wildcard: proceed normally.
            else:
                if isinstance(obj, dict):
                    value = obj.get(current_key)
                    if value is None:
                        return None
                    return _extract(value, keys[1:])
                elif isinstance(obj, list):
                    # When the current object is a list, apply the same key to every element.
                    results = []
                    for item in obj:
                        extracted = _extract(item, keys)
                        if extracted is not None:
                            results.append(extracted)
                    return results
                else:
                    # If it's not a dict or list, we can't traverse further.
                    return None

        return _extract(data, keys)

#*****************Containers****************************
#Create container for models that are generated in non-standard formats
class Model:
    def __init__(self, model_id):
        self.id = model_id

class ModelsContainer:
    def __init__(self, model_ids):
        self.data = [Model(model_id) for model_id in model_ids]


class ModelContainer:
    
    #A self-extracting container for model ids
    def __init__(self, models:list[str],request_mode:RequestMode=RequestMode.REMOTE)->None:
        self._models = models
        self._request_mode = request_mode

    def get_models(self, sort_it:bool=True, with_none:bool=True, include_filter: Optional[Iterable[str]] = None, exclude_filter: Optional[Iterable[str]]=None):
        """
        Retrieve the list of models with optional sorting, inclusion of 'none', 
        and filtering by inclusion or exclusion iterable strings (tuple or list).

        :param sort_it: If True, return the list sorted.
        :param with_none: If True, prepend the 'none' placeholder model.
        :param include_filter: Only include models containing any of these strings.
        :param exclude_filter: Exclude models containing any of these strings.
        :return: A list of filtered (and possibly sorted) model IDs.
        """

        models = ['none'] if with_none else []

        if include_filter and isinstance(include_filter, Iterable):
            filtered_models = [model for model in self._models 
                          if any(f.lower() in model.lower() for f in include_filter)
            ]
        else:
            filtered_models = self._models[:]

        if exclude_filter and isinstance(exclude_filter, Iterable):
            filtered_models = [model for model in filtered_models 
                          if all(f.lower() not in model.lower() for f in exclude_filter)
            ]

        models.extend(filtered_models)

        if sort_it:
            models.sort()

        return models
    
    @property
    def has_data(self)->bool:
        return bool(self._models)
    
    @property
    def request_mode(self)->RequestMode:
        return self._request_mode    

#***************End Containers*********************************


class FetchByProperty(ModelFetchStrategy):

    def fetch_models(self, key:str, api_obj, **kwargs):

        if not key:
            self.j_mngr.log_events("No OpenAI Key found.")
            return None

        api_obj.api_key = key
        #Get the model list 

        try:
            models = api_obj.models.list()  
        except Exception as e:
            self.j_mngr.log_events(f"openai Key is invalid or missing, unable to generate list of models. Error: {e}",
                                TroubleSgltn.Severity.WARNING,
                                True)
            return None
        

        return models

class FetchGemini(ModelFetchStrategy):

    def __init__(self)->None:
        super().__init__()  # Ensures common setup from ModelFetchStrategy
        self.comm = CommUtils()

    def fetch_models(self, key:str, api_obj, **kwargs) -> ModelContainer:
        model_list = []

        try:
            models = []  #Placeholder until gemini api is included
            #models = genai.list_models()
        except Exception as e:
            self.j_mngr.log_events(f"An exception occurred from the Gemini list_models method.  Exception: {e}",
                                   TroubleSgltn.Severity.ERROR)
            return ModelContainer(model_list, RequestMode.GEMINI)
        
        if not models:
            self.j_mngr.log_events("Gemini models method returned None or invalid object.", TroubleSgltn.Severity.ERROR)
            return ModelContainer(model_list, RequestMode.GEMINI)


        for model in models:
            # Use attribute access for model properties
            if hasattr(model, "supported_generation_methods") and "generateContent" in model.supported_generation_methods:
                # Fallback to 'Model Name Missing' if 'name' attribute doesn't exist
                model_name = getattr(model, "name", "Model-Name-Missing").removeprefix('models/')
                model_list.append(model_name)                

        if model_list:
            return ModelContainer(model_list, RequestMode.GEMINI)

        self.j_mngr.log_events("No 'content generation' models found in Google AI model list.", TroubleSgltn.Severity.WARNING)
        return ModelContainer(model_list, RequestMode.GEMINI)


class FetchByMethod(ModelFetchStrategy):

    def fetch_models(self, key:str, api_obj, **kwargs):

        if not key:
            self.j_mngr.log_events("No Groq Key found.")
            return None
        
        client = api_obj(api_key=key)

        try:
            model_list = client.models.list()
        except Exception as e:
            self.j_mngr.log_events(f"Groq Key is invalid or missing, unable to generate list of models. Error: {e}",
                                   TroubleSgltn.Severity.WARNING,
                                   True)
            return None
        return model_list   
    
class FetchOllama(ModelFetchStrategy):

    def __init__(self)->None:
        super().__init__()  # Ensures common setup from Request
        self.comm = CommUtils()

    def fetch_models(self, key:str, api_obj, **kwargs):
        """Parameters are ignored in this method and class as Ollama is a local app that has no
            imported api object and doesn't require a key.  Ollama is a local app
            that requires a model name be passed in the request."""
        
        url = self.utils.url_file("misc_urls.json", "ollama_url")
        t_response = self.comm.is_lm_server_up(url,1,2)
        if t_response:
            response = self.comm.get_data(url, retries=2)
        else:
            response = None

        model_list = []
        if response is None:
            return ModelsContainer(model_list)
        
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            self.j_mngr.log_events(f"Failed to decode Ollama models JSON file: {e}",
                                   TroubleSgltn.Severity.WARNING,
                                   True)
            return ModelsContainer(model_list)

        for model in data.get('models', []):
            model_list.append(model.get('name'))

        return ModelsContainer(model_list)


class FetchContainer(ModelFetchStrategy):

    def fetch_models(self, key:str, api_obj, **kwargs):
        return ModelsContainer(api_obj)


class FetchOptional(ModelFetchStrategy):

    def fetch_models(self, key:str, api_obj, **kwargs):
        """Parameters are ignored in this method and class as these model names exist in a
            local file named "optional_models.txt".  These model names are to used 
            for remote or local apps, other than Ollama, that require a file name to
            be passed.
        """
        model_list = []

        model_file = self.j_mngr.append_filename_to_path(self.j_mngr.script_dir, "opt_models.txt")

        if not os.path.exists(model_file):
            self.j_mngr.log_events("Optional Models file is missing.",
                                   TroubleSgltn.Severity.ERROR,
                                   True)
            return ModelsContainer(model_list)
        
        
        try:
            model_list = self.j_mngr.read_lines_of_file(model_file, is_critical=True) #Returns a list with any user entered model names
            return ModelsContainer(model_list)
        except Exception as e:
            self.j_mngr.log_events(f"Unable to read optional_models.txt file. Error: {e}",
                                   TroubleSgltn.Severity.ERROR,
                                   True)        
            return ModelsContainer(model_list)#empty model list
        
class FetchByURL(ModelFetchStrategy):
    def __init__(self) -> None:
        super().__init__()  # Ensures common setup from ModelFetchStrategy
        self.comm = CommUtils()
  

    def fetch_models(self, key: str, api_obj, **kwargs):
        url = kwargs.get('url', None)
        header = kwargs.get('header', {})
        remote_service = kwargs.get('service', "")
        requires_key = kwargs.get('key_req', True)       
        id_path = kwargs.get('id_path', 'data.*.id')  # Get the path to find model IDs directly from kwargs
        request_mode = kwargs.get('request_mode', RequestMode.REMOTE)        
        
        if header is None:
            header = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}"
            }
            
        # Initialize with empty ModelContainer
        model_list = ModelContainer([], request_mode)
        
        if url is None:
            self.j_mngr.log_events(f"No model fetch url provided for: {remote_service}.", TroubleSgltn.Severity.WARNING)
            return model_list
        
        if requires_key and not key:
            self.j_mngr.log_events(f"No key provided for model fetch from: {remote_service}.", TroubleSgltn.Severity.WARNING)
            return model_list
            
        response = self.comm.get_data(url, timeout=4, headers=header)
        self.j_mngr.log_events(f"Response: {response}")
        
        if response and response.status_code == 200:
            json_data = response.json()
            
            model_ids = []
            # If id_path is provided, use it to extract model IDs
            if id_path:
                model_ids = self.extract_nested_value(json_data, id_path)
                # If model_ids is a list of lists (due to wildcards), flatten it
                if model_ids and isinstance(model_ids[0], list):
                    model_ids = [item for sublist in model_ids for item in sublist]
            
            model_list = ModelContainer(model_ids, request_mode)
        else:
            response_code = response.status_code if response else 'N/A'
            self.j_mngr.log_events(f"Failed to retrieve {remote_service} models. Status code: {response_code}", 
                                    TroubleSgltn.Severity.ERROR, True)
        
        if not model_list.has_data:
            self.j_mngr.log_events(f"{remote_service} model list was empty.", TroubleSgltn.Severity.WARNING)
        
        return model_list


class FetchRemote(ModelFetchStrategy):
    def __init__(self) -> None:
        super().__init__()  # Ensures common setup from ModelFetchStrategy
        self.comm = CommUtils()
  

    def fetch_models(self, key: str, api_obj, **kwargs) -> ModelContainer:
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        model_list = []
        remote_service = kwargs.get('service', None)

        if remote_service is None:
            self.j_mngr.log_events("Remote model not defined in JSON.", TroubleSgltn.Severity.WARNING)
            return ModelContainer(model_list, RequestMode.REMOTE)

        url = self.utils.url_file("model_urls.json", f"{remote_service}.url")
        data_struc = self.utils.url_file("model_urls.json", f"{remote_service}.data_struc")

        # Use a default structure if not defined.
        if not data_struc:
            data_struc = "data.id"

        response = self.comm.get_data(url, timeout=4, headers=header)
        if response and response.status_code == 200:
            response_json = response.json()

            # Use our helper to extract model IDs according to the flexible dot notation.
            model_ids = self.extract_nested_value(response_json, data_struc)

            # If the extraction returns a list (or even nested lists), flatten it.
            if isinstance(model_ids, list):
                flat_ids = []
                for item in model_ids:
                    if isinstance(item, list):
                        flat_ids.extend(item)
                    else:
                        flat_ids.append(item)
                model_ids = flat_ids
            else:
                # Wrap a single value in a list.
                model_ids = [model_ids] if model_ids is not None else []

            # Replace any None values with a default string, if desired.
            for model_id in model_ids:
                model_list.append(model_id if model_id is not None else 'Unknown ID')

            if model_list:
                return ModelContainer(model_list, RequestMode.REMOTE)

            self.j_mngr.log_events(f"{remote_service} model list was empty.", TroubleSgltn.Severity.WARNING)
            return ModelContainer(model_list, RequestMode.REMOTE)
        response_code = response.status_code if response else 'N/A'
        self.j_mngr.log_events(f"Failed to retrieve {remote_service} models. Status code: {response_code}", TroubleSgltn.Severity.ERROR, True)
        return ModelContainer(model_list, RequestMode.REMOTE)


class FetchModels:
    def __init__(self):
        self.j_mngr = json_manager()
        self.strategy = None       

    def fetch_models(self, request_type:RequestMode, key: str="", api_obj:object=None, **kwargs):

        if request_type == RequestMode.OPENAI:
            api_obj = openai
            self.strategy = FetchByProperty()

        elif request_type == RequestMode.GROQ:
            api_obj = Groq
            self.strategy = FetchByMethod()

        elif request_type == RequestMode.CLAUDE:
            header = {
                "x-api-key": key,
                "anthropic-version": "2023-06-01"
            }

            kwargs = { "url": "https://api.anthropic.com/v1/models",
                    "header": header,
                    "service": "Anthropic",
                    "key_req": True,
                    "request_mode": request_type
            }
            self.strategy = FetchByURL()
            """
            if not api_obj:            
                api_obj = ['claude-3-haiku-20240307', 'claude-3-5-haiku-latest','claude-3-sonnet-20240229', 'claude-3-5-sonnet-20240620', 'claude-3-5-sonnet-latest', 'claude-3-opus-20240229']
            self.strategy = FetchContainer()
            """
        
        elif request_type == RequestMode.GEMINI:
            self.strategy = FetchGemini()

        elif request_type == RequestMode.OLLAMA:
            self.strategy = FetchOllama()

        elif request_type in (RequestMode.OPENSOURCE, RequestMode.OSSIMPLE):
            self.strategy = FetchOptional()

        elif request_type == RequestMode.REMOTE:
            self.strategy = FetchRemote()    

        if self.strategy:
            return self.strategy.fetch_models(key, api_obj, **kwargs)
        else:
            self.j_mngr.log_events("No Model fetch class specified",
                                   TroubleSgltn.Severity.WARNING,
                                   True)
     

class ModelUtils:
    def __init__(self) -> None:
        self.j_mngr = json_manager()
        

    def prep_models_list(self, models, sort_it: bool = False, filter_str: Optional[Iterable[str]] = None):
        # Start with 'none' here to prevent node error 'value not in list'
        prepped_models = ['none']

        if models is None or not hasattr(models, 'data') or not models.data:

            return prepped_models

        # Initialize filter_str to an empty tuple if it's None
        if filter_str is None:
            filter_str = ()

        # Include all models that contain any of the strings in filter_str
        filtered_models = [
            model.id for model in models.data
            if not filter_str or any(f.lower() in model.id.lower() for f in filter_str)
        ]

        prepped_models.extend(filtered_models)

        if sort_it:
            prepped_models.sort()

        return prepped_models
    
    def url_file(self, file_name: str, field_name: str) -> str:
        # Build the full path to your JSON file.
        url_file_name = self.j_mngr.append_filename_to_path(self.j_mngr.script_dir, file_name)
        
        # Load the JSON data.
        url_data = self.j_mngr.load_json(url_file_name)
        if not url_data:
            return ''
        
        # Split the field name on '.' to support nested keys.
        keys = field_name.split('.')
        value = url_data
        
        # Traverse the nested dictionaries.
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return ''  # Return empty if the key isn't found.
            else:
                return ''  # Return empty if the current value is not a dict.
        
        # Optionally, ensure that you're returning a string.
        return str(value) if value is not None else ''

    
    def x_url_file(self, file_name:str, field_name:str) -> str:
        url_file_name = self.j_mngr.append_filename_to_path(self.j_mngr.script_dir, file_name)
        url_data = self.j_mngr.load_json(url_file_name)
        if url_data:
            return url_data.get(field_name,'')
        return ''
    

