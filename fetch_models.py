
from abc import ABC, abstractmethod
from enum import Enum
from .mng_json import json_manager, TroubleSgltn #add .
from .utils import CommUtils
import openai
import os
import json
from groq import Groq
from typing import Iterable, Optional

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

class ModelFetchStrategy(ABC):
 
    def __init__(self)->None:
        self.j_mngr = json_manager()
        self.utils = ModelUtils()

    @abstractmethod
    def fetch_models(self, api_obj, key):
        pass


class FetchByProperty(ModelFetchStrategy):

    def fetch_models(self, api_obj, key:str):

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

class FetchGeminiModels(ModelFetchStrategy):

    def fetch_models(self, api_obj, key):

        
        api_obj.configure(api_key=key)

        try:
            models = api_obj.list_models()
        except Exception as e:
            self.j_mngr.log_events(f"Google gemini key is invalid or missing, unable to generate list of models. Error: {e}",
                                   TroubleSgltn.Severity.WARNING,
                                   True)
            return None
        
        model_list = []
        for mdl in models:
            if 'generateContent' in mdl.supported_generation_methods:
                parsed_model = mdl.name
                if parsed_model.startswith("models/"):
                    cleaned_model = parsed_model[len("models/"):]
                else:
                    cleaned_model = parsed_model

                model_list.append(cleaned_model)

        packaged_models = ModelsContainer(model_list)     

        return packaged_models

class FetchByMethod(ModelFetchStrategy):

    def fetch_models(self, api_obj, key:str):

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

    def fetch_models(self, api_obj, key):
        """Parameters are ignored in this method and class as Ollama is a local app that has no
            imported api object and doesn't require a key.  Ollama is a local app
            that requires a model name be passed in the request."""
        
        url = self.utils.url_file("urls.json", "ollama_url")
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
    
class FetchOptional(ModelFetchStrategy):

    def fetch_models(self, api_obj, key):
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



class FetchModels:
    def __init__(self):
        self.j_mngr = json_manager()
        self.strategy = None
        self.api_obj = None

    def fetch_models(self, request_type:RequestMode, key: str=""):

        if request_type == RequestMode.OPENAI:
            self.api_obj = openai
            self.strategy = FetchByProperty()

        elif request_type == RequestMode.GROQ:
            self.api_obj = Groq
            self.strategy = FetchByMethod()

        elif request_type == RequestMode.CLAUDE:
            model_names = ['claude-3-haiku-20240307',  'claude-3-5-haiku-latest', 'claude-3-sonnet-20240229', 'claude-3-5-sonnet-20240620', 'claude-3-5-sonnet-latest', 'claude-3-opus-20240229']
            return ModelsContainer(model_names)
        
        elif request_type == RequestMode.GEMINI:
            model_names = ['gemini-1.0-pro', 'gemini-1.0-pro-001', 'gemini-1.0-pro-latest', 'gemini-1.0-pro-vision-latest', 'gemini-1.5-pro-latest', 'gemini-pro', 'gemini-pro-vision']
            return ModelsContainer(model_names)
        
        elif request_type == RequestMode.OLLAMA:
            self.api_obj = None
            self.strategy = FetchOllama()

        elif request_type == RequestMode.OPENSOURCE or request_type == RequestMode.OSSIMPLE:
            self.api_obj = None
            self.strategy = FetchOptional()

        if self.strategy:

            return self.strategy.fetch_models(self.api_obj, key)
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
    
    def url_file(self, file_name:str, field_name:str) -> str:
        url_file_name = self.j_mngr.append_filename_to_path(self.j_mngr.script_dir, file_name)
        url_data = self.j_mngr.load_json(url_file_name)
        if url_data:
            return url_data.get(field_name,'')
        return ''
    
#Create container for models that are generated in non-standard formats
class Model:
    def __init__(self, model_id):
        self.id = model_id

class ModelsContainer:
    def __init__(self, model_ids):
        self.data = [Model(model_id) for model_id in model_ids]

class ModelContainer:
    def __init__(self, models:list[str])->None:
        self._models = models

    def get_models(self, sort_it:bool=True, with_none:bool=True, filter_str:str="",):

        models = ['none'] if with_none else []

        if filter_str:
            models.extend(model for model in self._models if filter_str.lower() in model.lower())
        else:
            models = self._models

        if sort_it:
            models.sort()

        return models
