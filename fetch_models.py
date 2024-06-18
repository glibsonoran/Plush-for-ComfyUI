
from abc import ABC, abstractmethod
from enum import Enum
from .mng_json import json_manager, TroubleSgltn #add .
import openai
from groq import Groq

class RequestMode(Enum):
    OPENAI = 1
    OPENSOURCE = 2
    OOBABOOGA = 3
    CLAUDE = 4
    GROQ = 5
    GEMINI = 6
    OSSIMPLE = 7

class ModelFetchStrategy(ABC):
 
    def __init__(self)->None:
        self.j_mngr = json_manager()

    @abstractmethod
    def fetch_models(self, api_obj, key):
        pass


class FetchByProperty(ModelFetchStrategy):

    def fetch_models(self, api_obj, key:str):

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

        client = api_obj(api_key=key)

        try:
            model_list = client.models.list()
        except Exception as e:
            self.j_mngr.log_events(f"Groq Key is invalid or missing, unable to generate list of models. Error: {e}",
                                   TroubleSgltn.Severity.WARNING,
                                   True)
            return None
        return model_list   

class FetchModels:
    def __init__(self):
        self.j_mngr = json_manager()
        self.strategy = None
        self.api_obj = None

    def fetch_models(self, request_type:RequestMode, key: str):

        if request_type == RequestMode.OPENAI:
            self.api_obj = openai
            self.strategy = FetchByProperty()

        elif request_type == RequestMode.GROQ:
            self.api_obj = Groq
            self.strategy = FetchByMethod()

        elif request_type == RequestMode.CLAUDE:
            model_names = ['claude-3-haiku-20240307', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229']
            return ModelsContainer(model_names)
        
        elif request_type == RequestMode.GEMINI:
            model_names = ['gemini-1.0-pro', 'gemini-1.0-pro-001', 'gemini-1.0-pro-latest', 'gemini-1.0-pro-vision-latest', 'gemini-1.5-pro-latest', 'gemini-pro', 'gemini-pro-vision']
            return ModelsContainer(model_names)

        if self.strategy and self.api_obj:
            return self.strategy.fetch_models(self.api_obj, key)
        else:
            self.j_mngr.log_events("Model fetch class or api object missing",
                                   TroubleSgltn.Severity.WARNING,
                                   True)
     

class ModelUtils:
    def __init__(self) -> None:
        self.j_mngr = json_manager()
        
        
    def prep_models_list(self, models, sort_it:bool=False, filter_str:str=""):
        #Start with 'None' here to prevent node error 'value not in list'
        prepped_models = ['none']

        if models is None or not hasattr(models, 'data') or not models.data:
            self.j_mngr.log_events("Models object is empty or malformed",
                                   TroubleSgltn.Severity.ERROR,
                                   True)
            return prepped_models
            
        filtered_models = [model.id for model in models.data if filter_str.lower() in model.id.lower()]

        prepped_models.extend(filtered_models)

        if sort_it:
            prepped_models.sort()
        
        return prepped_models 
    
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
        models.extend(model for model in self._models if filter_str.lower() in model.lower())

        if sort_it:
            models.sort()

        return models
    