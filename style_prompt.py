# ------------------------
# Standard Library Imports
# ------------------------
import os
from typing import Optional
from enum import Enum
from urllib.parse import urlparse

# -------------------------
# Third-Party Library Imports
# -------------------------
from PIL import Image, TiffImagePlugin, UnidentifiedImageError
import requests
from requests.adapters import HTTPAdapter, Retry
from openai import OpenAI
import anthropic
from aiohttp import web
#import google.generativeai as genai

# -----------------------
# Local Module Imports
# -----------------------
try:
    import folder_paths
    import nodes
    from server import PromptServer
    from .mng_json import json_manager, helpSgltn, TroubleSgltn
    from . import api_requests as rqst
    from .fetch_models import FetchModels, ModelUtils, RequestMode, ModelContainer
except ImportError:
    from mng_json import json_manager, helpSgltn, TroubleSgltn
    import api_requests as rqst
    from fetch_models import FetchModels, ModelUtils, RequestMode, ModelContainer


#pip install pillow
#pip install bytesio


#Enum for style_prompt user input modes
class InputMode(Enum):
    IMAGE_PROMPT = 1
    IMAGE_ONLY = 2
    PROMPT_ONLY = 3


#Get information from the config.json file
class cFigSingleton:
    _instance = None


    def __new__(cls): 
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._lm_client = None
            cls._anthropic_client = None
            cls._gemini_client = None
            cls._oai_client = None
            cls._lm_url = ""
            cls._lm_request_mode = None
            cls._lm_key = ""
            cls._custom_key = ""
            cls._custom_key_change = False #Flag only
            cls._groq_key = ""
            cls._claude_key = ""
            cls._gemini_key = ""
            cls._lm_models = None
            cls._groq_models = None
            cls._claude_models = None
            cls._gemini_models = None
            cls._ollama_models = None
            cls._optional_models = None
            cls._openrouter_models = None
            cls._written_url = ""
            cls.j_mngr = json_manager()
            cls._model_fetch = FetchModels()
            cls._model_prep = ModelUtils()
            cls._pyexiv2 = None
            cls._instance.get_file()

        return cls._instance
    
    
    def get_file(self):

        #Get script working directory
        #j_mngr = json_manager()

        # Error handling is in the load_json method
        # Errors will be raised since is_critical is set to True
        config_data =self.j_mngr.load_json(self.j_mngr.config_file, True)

        #Pyexiv2 seems to have trouble loading with some Python versions (it's misreading the vesrion number)
        #So I'll open it in a try block so as not to stop the whole suite from loading
        try:
            import pyexiv2
            self._pyexiv2 = pyexiv2
        except Exception as e:
            self._pyexiv2 = None
            self.j_mngr.log_events(f"The Pyexiv2 library failed to load with Error: {e} ",
                              TroubleSgltn.Severity.ERROR)

        #check if file is empty
        if not config_data:
            raise ValueError("Plush - Error: config.json contains no valid JSON data")
       
        # Try getting API key from Plush environment variable
        self._fig_key = os.getenv('OAI_KEY',"") or os.getenv('OPENAI_API_KEY',"")            
        
        if not self._fig_key:
            #Let user know some nodes will not function
            self.j_mngr.log_events("Open AI API key invalid or not found, some nodes will not be functional. See Read Me to install the key",
                              TroubleSgltn.Severity.WARNING)
        
        self._lm_key = os.getenv("LLM_KEY","") #Fetch the LLM_KEY if the user has created one
        self._groq_key = os.getenv("GROQ_API_KEY", "")
        self._claude_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._gemini_key = os.getenv("GEMINI_API_KEY", "")
            
        #Get user saved Open Source URL from the text file  
        #At this point all this does is pre-populate new instances of the node. 
        url_file =self.j_mngr.append_filename_to_path(self.j_mngr.script_dir, 'OpenSourceURL.txt')
        lm_url_list =self.j_mngr.read_lines_of_file(url_file)
        if lm_url_list:
            self._lm_url = lm_url_list[0] #This call will set up the client if the local LLM server is running
            self._written_url = self._lm_url
     
        self.figInstruction = config_data.get('instruction', "")
        self.figExample = config_data.get('example', "")
        self.figExample2 = config_data.get('example2', "")
        self.fig_n_example = config_data.get('n_example', "")
        self.fig_n_example2 = config_data.get('n_example2', "")
        self._use_examples = False
        self.figStyle = config_data.get('style', "")
        self.figImgInstruction = config_data.get('img_instruction', "")
        self.figImgPromptInstruction = config_data.get('img_prompt_instruction', "")
        self.fig_n_Instruction = config_data.get('n_instruction', "")
        self.fig_n_ImgPromptInstruction = config_data.get('n_img_prompt_instruction', "")
        self.fig_n_ImgInstruction = config_data.get('n_img_instruction', "")

        
        self._fig_gpt_models = []
        
        if self._fig_key:
            try:
                self._oai_client = OpenAI(api_key= self._fig_key)
            except Exception as e:
                self.j_mngr.log_events(f"Invalid or missing OpenAI API key.  Please note, keys must now be kept in an environment variable (see: ReadMe) {e}",
                                  severity=TroubleSgltn.Severity.ERROR)
                
        if self._claude_key:
            try:
                self._anthropic_client = anthropic.Anthropic(api_key = self._claude_key)
            except Exception as e:
                self.j_mngr.log_events(f"Invalid or missing Anthropic API key. Please note, keys must be kept in an environment variable.{e}",                                       
                                       severity=TroubleSgltn.Severity.ERROR)    
                
        if not self._gemini_key:
            try:
                #genai.configure(api_key=self._gemini_key)
                self._gemini_key = ""  #Placeholder until gemini api is ready
            except Exception as e:
                self.j_mngr.log_events(f"Invalid or missing Gemini API key. Please note, keys must be kept in an environment variable.{e}",                                       
                                       severity=TroubleSgltn.Severity.ERROR)    
                
                
                
        self._fig_gpt_models = self._model_fetch.fetch_models(RequestMode.OPENAI, self._fig_key)
        self._groq_models = self._model_fetch.fetch_models(RequestMode.GROQ, self._groq_key)
        self._claude_models = self._model_fetch.fetch_models(RequestMode.CLAUDE, self._claude_key,api_obj=self.anthropic_client)
        self._gemini_models = self._model_fetch.fetch_models(RequestMode.GEMINI, self._gemini_key)
        self._ollama_models =  self._model_fetch.fetch_models(RequestMode.OLLAMA, "")  
        self._optional_models = self._model_fetch.fetch_models(RequestMode.OPENSOURCE, "")    
   
    def get_chat_models(self, sort_it:bool=False, filter_str:tuple=())->list:
        return self._model_prep.prep_models_list(self._fig_gpt_models, sort_it, filter_str)      
      
    def get_groq_models(self):
        if self._groq_models is None:
            # Initialize with an empty container if none exists
            self._groq_models = ModelContainer([], RequestMode.GROQ)
        return self._groq_models    

    def get_claude_models(self):
        if self._claude_models is None:
            # Initialize with an empty container if none exists
            self._claude_models = ModelContainer([], RequestMode.CLAUDE)
        return self._claude_models

    def get_gemini_models(self)->ModelContainer:    

        if self._gemini_models is None:
            # Initialize with an empty container if none exists
            self._gemini_models = ModelContainer([], RequestMode.GEMINI)
        return self._gemini_models
    
    def get_ollama_models(self)->ModelContainer:

        if self._ollama_models is None:
            # Initialize with an empty container if none exists
            self._ollama_models = ModelContainer([], RequestMode.OLLAMA)
        return self._ollama_models  

    def get_optional_models(self, sort_it:bool=False, filter_str:tuple=())->list: 
        return self._model_prep.prep_models_list(self._optional_models, sort_it, filter_str)   
    
            
    def _set_llm_client(self, url:str, request_type:RequestMode=RequestMode.OPENSOURCE)-> bool:
        
        if not self.is_lm_server_up() or not url:
            self._lm_client = None
            self._lm_url = url
            self._lm_models = None
            self.j_mngr.log_events("Local LLM server is not running; aborting client setup.",
                          TroubleSgltn.Severity.WARNING,
                          True)
            return False
        
        lm_object = OpenAI
        key = "No key necessary" #Default value used in LLM front-ends that don't require a key
        #Use the requested API
        

        if request_type in (RequestMode.OOBABOOGA, RequestMode.OPENSOURCE):
            key = self._custom_key or self._lm_key  # key will be the first truthy value
            message = (
                "Setting Openai client with URL and key." 
                if key else 
                "Setting Openai client with URL, no key."
            )
            self.j_mngr.log_events(message, is_trouble=True)

        elif request_type == RequestMode.GROQ:
            key = self._groq_key
            if not key:
                self.j_mngr.log_events(
                    "Attempting to connect to Groq with no key",
                    TroubleSgltn.Severity.ERROR,
                    True
                )

        elif request_type == RequestMode.GEMINI:
            key = self._gemini_key
            if not key:
                self.j_mngr.log_events(
                    "Attempting to connect to Gemini with no key",
                    TroubleSgltn.Severity.ERROR,
                    True
                )

        else:
            key = self._groq_key
            self.j_mngr.log_events(
                "Setting Openai client with URL and Groq key.",
                is_trouble=True
    )


        
        try:
            self._lm_client = lm_object(base_url=url, api_key=key) 
            self._lm_url = url

        except Exception as e:
            self.j_mngr.log_events(f"Unable to create LLM client object using URL. Unable to communicate with LLM: {e}",
                            TroubleSgltn.Severity.ERROR,
                            True)
            return False

        return True

    @property
    def lm_client(self):
        return self._lm_client
    
    @property
    def lm_url(self):
        return self._lm_url
    
    def write_url(self, url:str) -> bool:
        # Save the current open source url for startup retrieval of models
        url_result = False
        if url and url != self._written_url:
            url_file = self.j_mngr.append_filename_to_path(self.j_mngr.script_dir, 'OpenSourceURL.txt')
            url_result = self.j_mngr.write_string_to_file(url, url_file)
            self._written_url = url
            self.j_mngr.log_events("Open source LLM URL saved to file.",
                                   TroubleSgltn.Severity.INFO,
                                   True)
        return url_result
    
    @lm_url.setter
    def lm_url(self, url: str):
        if url != self._lm_url or not self._lm_client or self._custom_key_change:  # Check if the new URL is different to avoid unnecessary operations
            
            self._lm_url = url
            self._lm_client = None
            if url:  # If the new URL is not empty, update the client
                self._set_llm_client(url, self._lm_request_mode)
    
    
    def is_lm_server_up(self):  #should be util in api_requests.py
        session = requests.Session()
        retries = Retry(total=2, backoff_factor=0, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        try:
            response = session.head(self._lm_url, timeout=4)  # Use HEAD to minimize data transfer            
            if 200 <= response.status_code <= 300:
                self.write_url(self._lm_url) #Save url to a text file
                self.j_mngr.log_events(f"Local LLM Server is running with status code: {response.status_code}",
                              TroubleSgltn.Severity.INFO,
                              True)
                return True
            else:
                self.write_url(self._lm_url) #Save url to a text file
                self.j_mngr.log_events(f"Server returned response code: {response.status_code}",
                                       TroubleSgltn.Severity.INFO,
                                       True)
                return True

        except requests.RequestException as e:            
            self.j_mngr.log_events(f"Local LLM Server is not running: {e}",
                              TroubleSgltn.Severity.WARNING,
                              True)
        return False  
            

    @property
    def use_examples(self)->bool:
        return self._use_examples

    @use_examples.setter        
    def use_examples(self, use_examples: bool):
        #Write, sets internal flag
        self._use_examples = use_examples    

    @property
    def lm_request_mode(self)->RequestMode:
        return self._lm_request_mode
    
    @lm_request_mode.setter
    def lm_request_mode(self, mode:RequestMode)-> None:
        self._lm_request_mode = mode
        
    @property
    def key(self)-> str:
        return self._fig_key
    
    @property
    def lm_key(self)-> str:
        return self._lm_key

    @property
    def custom_key(self)->str:
        return self._custom_key
    
    @custom_key.setter
    def custom_key(self, key:str)->None:
        #Inject key value from User-defined Env. Variable
        if key != self._custom_key:
            self._custom_key_change = True
            self._custom_key = key 
        else:
            self._custom_key_change = False


    
    @property
    def groq_key(self)->str:
        return self._groq_key
    
    @property
    def anthropic_key(self)->str:
        return self._claude_key
    
    @property
    def gemini_key(self)->str:
        return self._gemini_key

    @property
    def instruction(self):
        return self.figInstruction
    
    
    @property
    def example(self):
        if self._use_examples:
            return self.figExample
        return ""
    
    @property
    def example2(self):
        if self._use_examples:
            return self.figExample2
        return ""
    
    @property
    def n_Example(self):
        if self._use_examples:
            return self.fig_n_example
        return ""
    
    @property
    def n_example2(self):
        if self._use_examples:
            return self.fig_n_example2
        return ""

    @property
    def style(self):
        #make sure the designated default value is present in the list
        if "Photograph" not in self.figStyle:
            if not isinstance(self.figStyle, list):
                self.figStyle = []
            self.figStyle.append("Photograph")

        return self.figStyle
    
    @property
    def ImgInstruction(self):
        return self.figImgInstruction
    
    @property
    def ImgPromptInstruction(self):
        return self.figImgPromptInstruction
    
    @property
    def n_Instruction(self):
        return self.fig_n_Instruction
    
    @property
    def n_ImgPromptInstruction(self):
        return self.fig_n_ImgPromptInstruction
    
    @property
    def n_ImgInstruction(self):
        return self.fig_n_ImgInstruction
     

    @property
    def pyexiv2(self)-> Optional[object]:       
        return self._pyexiv2
    
    @property
    def anthropic_client(self)->Optional[object]:
        if self._claude_key:
            return self._anthropic_client
        return None
        
    @property
    def openaiClient(self)-> Optional[object]:
        if self._fig_key:
            return self._oai_client
        return None

#********************End Singleton*********************
#Module level functions to create binding endpoint and
#load drop down list values from file
def _load_env_var_list() -> list:

    # Inactive Experimental Code

    """
    Core function to load the environment variable list.
    Returns a list with either the env vars or ['error'] if something goes wrong.
    """
    j_mngr = json_manager()
    envvar_list = ['error']
    
    try:
        envvar_file = j_mngr.append_filename_to_path(j_mngr.script_dir, "user_envvar.txt")
        user_vars = j_mngr.read_lines_of_file(envvar_file, is_critical=True)
        if user_vars:
            envvar_list = user_vars
    except Exception as e:
        j_mngr.log_events(f"Unable to read 'user_envvar.txt' file. Error: {e}",
                         TroubleSgltn.Severity.ERROR,
                         True)
    
    return envvar_list


@PromptServer.instance.routes.get("/plush_for_comfyui/envvar_list")
async def handle_envvar_list_request(request):

    # Inactive Experimental Code

    j_mngr = json_manager()
    
    # Get any new variable from query params
    new_var = request.query.get('new_var')
    save_triggered = request.query.get('save_triggered', 'false') == 'true'
    
    if new_var and save_triggered:
        # Check if it exists in environment
        key = os.getenv(new_var, "missing")
        if key != "missing" and key:  # Valid env var
            # Read current list
            envvar_file = j_mngr.append_filename_to_path(j_mngr.script_dir, "user_envvar.txt")
            envvar_list = j_mngr.read_lines_of_file(envvar_file)
            
            # Add if not duplicate
            if new_var not in envvar_list:
                success = j_mngr.write_string_to_file(new_var + "\n", envvar_file, append=True)
                if success:
                    j_mngr.log_events("New environment variable saved.", is_trouble=True)
            else:
                j_mngr.log_events("Environment variable already exists in list.", 
                                TroubleSgltn.Severity.WARNING,
                                True)
        else:
            j_mngr.log_events(f"Environment Variable '{new_var}' not found in system.", 
                            TroubleSgltn.Severity.ERROR,
                            True)
    
    # Return updated list
    vars_list = _load_env_var_list()
    return web.json_response(vars_list)


class CustomKeyVar:

    #Active, pending success of experimental code

    def __init__(self):
        #instantiate Configuration and Help data classes
        self.cFig = cFigSingleton()
        self.help_data = helpSgltn()
        self.j_mngr = json_manager()
        self.trbl = TroubleSgltn()


    @classmethod
    def INPUT_TYPES(cls):# -> dict[str, dict[str, Any]]:

        j_mngr = json_manager()

        def get_envvar_list():
            envvar_list =['error']
            envvar_file = j_mngr.append_filename_to_path(j_mngr.script_dir,"user_envvar.txt")
            try:
                envvar_list = j_mngr.read_lines_of_file(envvar_file, is_critical=True) #Returns a list with any user entered env. variables
            except Exception as e:
                j_mngr.log_events(f"Unable to read 'user_envvar.txt' file. Error: {e}",
                                    TroubleSgltn.Severity.ERROR,
                                    True)   
                return envvar_list
            return envvar_list


        return {
            "required": {
                "Environment_Variable": (get_envvar_list(), {"default": "-New Env. Variable", "tooltip": "Choose to enter a new Env. Variable below, or choose an existing one from the list"}),
                "New_Env_Variable": ("STRING", {"multiline": False, "default": "", "tooltip": "Enter the name of a new Environment Variable that contains your API Key."}),                                 
            }
        } 

    RETURN_TYPES = ("KEY","STRING")
    RETURN_NAMES = ("Custom_ApiKey", "troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Prompt"  

    def gogo(self, Environment_Variable, New_Env_Variable)->tuple:

        self.trbl.reset("Custom API Key")

        env_var = ""
        if Environment_Variable == "-New Env. Variable":
            env_var = New_Env_Variable
            if not New_Env_Variable:
                self.j_mngr.log_events("You specified '-New Env Variable' but didn't provide one in the field.", 
                                       TroubleSgltn.Severity.ERROR,
                                       True)
        else:
            env_var = Environment_Variable

        key = os.getenv(env_var,"missing",) #Fetch the api Key from the User-def Env. Var.

        if key == "missing":
            self.j_mngr.log_events(f"Environment Variable missing or misspelled: {env_var}",
                TroubleSgltn.Severity.ERROR,
                True)
            return ("", self.trbl.get_troubles())
        if not key:
            self.j_mngr.log_events(f"Environment Variable not found: {env_var}",
                TroubleSgltn.Severity.ERROR,
                True)
            return ("", self.trbl.get_troubles()  )
        
        if Environment_Variable == "-New Env. Variable":
            envvar_file = self.j_mngr.append_filename_to_path(self.j_mngr.script_dir, "user_envvar.txt")
            envvar_list = self.j_mngr.read_lines_of_file(envvar_file)
            if New_Env_Variable not in envvar_list: # Prevent duplicates
                self.j_mngr.write_string_to_file(env_var + "\n", envvar_file, append=True)

        self.j_mngr.log_events("API Key succesfully retrieved.", is_trouble=True)

        return (key,self.trbl.get_troubles())



    
class AI_Chooser:
    def __init__(self):
        #instantiate Configuration and Help data classes
        self.cFig = cFigSingleton()
        self.help_data = helpSgltn()
        self.j_mngr = json_manager()
        self.trbl = TroubleSgltn()

    @staticmethod
    def select_request_mode(user_selection:str) -> RequestMode:
         
        mode_map = {
            "ChatGPT": RequestMode.OPENAI,
            "Groq": RequestMode.GROQ,
            "Anthropic": RequestMode.CLAUDE,
            "Gemini": RequestMode.GEMINI,
            "OpenAI API Connection (URL)": RequestMode.OPENSOURCE,
            "Direct Web Connection (URL)": RequestMode.OPENSOURCE,
            "LM_Studio (URL)": RequestMode.OPENSOURCE,
            "Ollama (URL)": RequestMode.OPENSOURCE,
            "Web Connection Simplified Data (URL)": RequestMode.OSSIMPLE,
            "Oobabooga API (URL)": RequestMode.OOBABOOGA
        }
        return mode_map.get(user_selection)
        

    @classmethod
    def INPUT_TYPES(cls):
        cFig=cFigSingleton()
        gptfilter = ("gpt","o1","o3","o4")
        #Floats have a problem, they go over the max value even when round and step are set, and the node fails.  So I set max a little over the expected input value
        return {
            "required": {
                "AI_Service": (["ChatGPT", "Groq", "Gemini","Anthropic"], {"default": "ChatGPT"}),
                "ChatGPT_model": (cFig.get_chat_models(True,gptfilter), {"default": ""}),
                "Groq_model": (cFig.get_groq_models().get_models(True), {"default": ""}), 
                "Google_Gemini_model": (cFig.get_gemini_models().get_models(True), {"default": ""}),
                "Anthropic_model": (cFig.get_claude_models().get_models(False, exclude_filter=("2.0", "2.1")), {"default": ""}),                                  
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        } 

    RETURN_TYPES = ("DICTIONARY",)
    RETURN_NAMES = ("AI_Selection",)

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Prompt"  

    def gogo(self, unique_id, AI_Service, ChatGPT_model, Groq_model, Google_Gemini_model, Anthropic_model):  

        ai_dict = {"service": AI_Chooser.select_request_mode(AI_Service), "model": None}

        if ai_dict['service'] == RequestMode.OPENAI:
            ai_dict['model'] = ChatGPT_model
        elif ai_dict['service'] == RequestMode.GROQ :
            ai_dict['model'] = Groq_model
        elif ai_dict['service'] == RequestMode.GEMINI:
            ai_dict['model'] = Google_Gemini_model
        elif ai_dict['service'] == RequestMode.CLAUDE:
            ai_dict['model'] = Anthropic_model

        return (ai_dict,)


class Enhancer:
#Build a creative prompt using a ChatGPT model    
   
    def __init__(self):
        #instantiate Configuration and Help data classes
        self.cFig = cFigSingleton()
        self.help_data = helpSgltn()
        self.j_mngr = json_manager()
        self.trbl = TroubleSgltn()
        self.ctx = rqst.request_context()


    def build_instruction(self, mode, style, prompt_style, elements, artist):
          #build the instruction from user input
        instruc = ""
        

        if prompt_style == "Narrative":
            if mode == InputMode.PROMPT_ONLY:
                if self.cFig.n_Instruction:
                    instruc = self.cFig.n_Instruction
                
            elif mode == InputMode.IMAGE_ONLY:
                if self.cFig.n_ImgInstruction:
                    instruc = self.cFig.n_ImgInstruction
                
            elif mode == InputMode.IMAGE_PROMPT:
                if self.cFig.n_ImgPromptInstruction:
                    instruc = self.cFig.n_ImgPromptInstruction

        else:      #Prompt_style is Tags
            if mode == InputMode.PROMPT_ONLY:
                if self.cFig.instruction:
                    instruc = self.cFig.instruction
                
            elif mode == InputMode.IMAGE_ONLY:
                if self.cFig.ImgInstruction:
                    instruc = self.cFig.ImgInstruction
                
            elif mode == InputMode.IMAGE_PROMPT:
                if self.cFig.ImgPromptInstruction:
                    instruc = self.cFig.ImgPromptInstruction

        if instruc.count("{}") >= 2:
            instruc = instruc.format(style, elements)
        elif instruc.count("{}") == 1:
            instruc = instruc.format(style)

        if artist >= 1:
            art_instruc = "  Include {} artist(s) who works in the specifed artistic style by placing the artists' name(s) at the end of the sentence prefaced by 'style of'."
            instruc += art_instruc.format(str(artist))

        return instruc

    def translateModelName(self, model: str)-> str:
        #Translate friendly model names to working model names
        #Not in use right now, but new models typically go through a period where there's 
        #no pointer value for the latest models.
        if model == "gpt-4 Turbo":
            model = "gpt-4-1106-preview"

        return model
    
    @staticmethod
    def undefined_to_none( sus_var):
        """
        Converts the string "undefined" to a None.

        Note: ComfyUI returns unconnected UI elements as "undefined"
        which causes problems when the node expects these to be handled as falsey
        Args:
            sus_var(any): The variable that might containt "undefined"
        Returns:
            None if the variable is set to the string "undefined" or unchanged (any) if not.
        """   
        return None if sus_var == "undefined" else sus_var
                
   
    @classmethod
    def INPUT_TYPES(cls):
        cFig=cFigSingleton()

        #Floats have a problem, they go over the max value even when round and step are set, and the node fails.  So I set max a little over the expected input value
        return {
            "required": {
                #"GPTmodel": (cFig.get_chat_models(True, 'gpt'),{"default": ""} ),
                "creative_latitude" : ("FLOAT", {"max": 1.201, "min": 0.1, "step": 0.1, "display": "number", "round": 0.1, "default": 0.7}),                  
                "tokens" : ("INT", {"max": 8000, "min": 20, "step": 10, "default": 500, "display": "number"}),                
                "style": (cFig.style,{"default": "Photograph"}),
                "artist" : ("INT", {"max": 3, "min": 0, "step": 1, "default": 1, "display": "number"}),
                "prompt_style": (["Tags", "Narrative"],{"default": "Tags"}),
                "max_elements" : ("INT", {"max": 25, "min": 3, "step": 1, "default": 10, "display": "number"}),
                "style_info" : ("BOOLEAN", {"default": False})                               
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {  
                "AI_Selection":("DICTIONARY", {"default": None}),
                "prompt": ("STRING",{"multiline": True, "default": ""}),          
                "image" : ("IMAGE", {"default": None})
            }
        } 

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING","STRING")
    RETURN_NAMES = ("AI_prompt", "AI_instruction","Style Info", "Help","troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Prompt"
 

    def gogo(self, creative_latitude, tokens, style, artist, prompt_style, max_elements, style_info, AI_Selection=None, prompt="", image=None, unique_id=None):

        if unique_id:
            self.trbl.reset('Style Prompt, Node #'+unique_id)
        else:
            self.trbl.reset('Style Prompt')
        
        _help = self.help_data.style_prompt_help
        CGPT_prompt = ""
        instruction = ""
        CGPT_styleInfo = ""

        if AI_Selection:       
            ais_model = AI_Selection['model']
        else:
            self.j_mngr.log_events("You must connect the  Plush AI_Chooser to the AI_Selection Input and choose an AI_Service and model to use",
                                   TroubleSgltn.Severity.ERROR,
                                   True)
            CGPT_prompt = "Plush AI_Chooser not connected to AI_Selection input, or missing input values"
            return(CGPT_prompt, instruction, CGPT_styleInfo, _help, self.trbl.get_troubles())

        # unconnected UI elements get passed in as the string "undefined" by ComfyUI
        image = self.undefined_to_none(image)
        prompt = self.undefined_to_none(prompt)
        #Translate any friendly model names    

        #build instruction based on user input
        mode = 0
        if image is not None and prompt:
            mode = InputMode.IMAGE_PROMPT
        elif image is not None:
            mode = InputMode.IMAGE_ONLY
        elif prompt:
            mode = InputMode.PROMPT_ONLY

        instruction = self.build_instruction(mode, style, prompt_style, max_elements, artist)  

        self.cFig.lm_request_mode = AI_Selection['service']

        if AI_Selection['service'] == RequestMode.OPENAI:
            self.ctx.request = rqst.oai_object_request()
        elif AI_Selection['service'] == RequestMode.GROQ:
            self.ctx.request = rqst.oai_object_request()
            # set the url so the function making the request will have a properly initialized object.               
            self.cFig.lm_url = "https://api.groq.com/openai/v1" # Ugh!  I've embedded a 'magic value' URL here for the OPENAI API Object because the GROQ API object looks flakey...
        elif AI_Selection['service'] == RequestMode.CLAUDE:
            self.ctx.request = rqst.claude_request()
        elif AI_Selection['service'] == RequestMode.GEMINI:
            self.ctx.request = rqst.oai_object_request()
            self.cFig.lm_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        if style_info:
            self.trbl.set_process_header("Art Style Info:")
            #User has request information about the art style.  GPT will provide it
            sty_prompt = f"Give an 150 word backgrounder on the art style: {style}.  Starting with describing what it is, include information about its history and which artists represent the style."

            kwargs = { "model": ais_model,
                "creative_latitude": creative_latitude,
                "tokens": tokens,
                "prompt": sty_prompt,
            }    
            CGPT_styleInfo = self.ctx.execute_request(**kwargs)
            self.trbl.pop_header()

        kwargs = { "model": ais_model,
            "creative_latitude": creative_latitude,
            "tokens": tokens,
            "prompt": prompt,
            "instruction": instruction,
            "image": image,
        }

        CGPT_prompt = self.ctx.execute_request(**kwargs)
    
        return (CGPT_prompt, instruction, CGPT_styleInfo, _help, self.trbl.get_troubles())


class addParameters:
    ##New##
    def __init__(self):
        #instantiate Configuration and Help data classes
        self.cFig = cFigSingleton()
        self.help_data = helpSgltn()
        self.j_mngr = json_manager()
        self.trbl = TroubleSgltn()

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "Parameters": ("STRING", {"multiline": True, "default": ""}),
                "Save_to_file": ("BOOLEAN", {"default": False}),
                "File_name": ("STRING", {"default": ""})
                                                          
            },
#           "optional": {
#              "text": ("STRING", {"default": None, "forceinput": True})
#         },

            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        
        } 
    
    RETURN_TYPES = ("LIST", "STRING", "STRING")
    RETURN_NAMES = ("Add_Parameter","Help","Troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Prompt"  

    def gogo(self, Parameters, Save_to_file, File_name:bool, unique_id=None):

        self.trbl.reset(f"Add Parameters, Node: {unique_id}")
        _help = self.help_data.add_params_help
        param_list = []

        #Create path and dir for saved .txt files
        write_dir = ''
        comfy_dir = self.j_mngr.comfy_dir
        if comfy_dir:
            output_dir = self.j_mngr.find_child_directory(comfy_dir,'output')
            if output_dir:
                write_dir = self.j_mngr.find_child_directory(output_dir, 'PlushFiles',True) #Setting argument to True means dir will be created if not present
                if not write_dir:
                    self.j_mngr.log_events('Unable to find or create PlushFiles directory. Unable to write files',
                                    TroubleSgltn.Severity.WARNING,
                                    True)
            else:
                self.j_mngr.log_events('Unable to find output directory, Unable to write files',
                                   TroubleSgltn.Severity.WARNING,
                                   True)
        else:
            self.j_mngr.log_events('Unable to find ComfyUI directory. Unable to write files.',
                                   TroubleSgltn.Severity.WARNING,
                                   True)

        if Save_to_file and write_dir:

            working_file_name = self.j_mngr.generate_unique_filename("txt", File_name + '_param_')
            working_file_path = self.j_mngr.append_filename_to_path(write_dir,working_file_name)
            self.j_mngr.write_string_to_file(Parameters,working_file_path)  
            self.j_mngr.log_events(f"Parameter file: '{working_file_name}' successfully written to directory [{write_dir}]",
                                   is_trouble=True)          


        if Parameters:
            template_dict = {"param": None, "value": None}
            param_list = self.j_mngr.positional_str_to_dict(Parameters,template_dict,"#","::")  #Parses the Parameters data and puts the result in a list of dicts
            if not param_list:
                self.j_mngr.log_events("Parameters not processed",
                                        TroubleSgltn.Severity.ERROR,
                                        True)
                return(param_list,_help,self.trbl.get_troubles())
        else:
            self.j_mngr.log_events("No paramter data provided.",
                                   TroubleSgltn.Severity.INFO,
                                   True)    
            return(param_list,_help,self.trbl.get_troubles())
        
        self.j_mngr.log_events(f"Parameters added: {str(param_list)}",
                               is_trouble=True)
        return(param_list,_help,self.trbl.get_troubles())


class addParams:
    ##Depricated##
    def __init__(self):
        #instantiate Configuration and Help data classes
        self.cFig = cFigSingleton()
        self.help_data = helpSgltn()
        self.j_mngr = json_manager()
        self.trbl = TroubleSgltn()

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "Parameter_type": (["none", "OpenAI JSON Format", "User Defined"], {"default": "none"}),
                "Param_Name": ("STRING", {"default": ""}),
                "Param_Value": ("STRING", {"multiline": True}),
                "Is_JSON": ("BOOLEAN", {"default": False})
                                                          
            },

            "hidden": {
                "unique_id": "UNIQUE_ID",
            },

            "optional": {
                "Add_Parameters": ("LIST", {"default": None, "forceInput": True})
            }
        
        } 
    
    RETURN_TYPES = ("LIST", "STRING", "STRING")
    RETURN_NAMES = ("Add_Parameter","Help","Troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Prompt"  

    def gogo(self, Parameter_type, Param_Name, Param_Value, Is_JSON: bool, Add_Parameters=None, unique_id=None):

        self.trbl.reset(f"Additional Parameters, Node: {unique_id}")
        _help = self.help_data.add_param_help 
        # Copy Add_Parameters or start fresh to avoid reference issues
        param_list = list(Add_Parameters) if isinstance(Add_Parameters, list) else []

            # Handle OpenAI JSON Format case
        if Parameter_type == "OpenAI JSON Format":
            f_json = {"type": "json_object"}
            param_list.append({"param": "response_format", "value": f_json})
            self.j_mngr.log_events(f"Param output: {str(param_list)}", is_trouble=True)
            return (param_list, _help, self.trbl.get_troubles())

        # Handle 'none' case
        if Parameter_type == "none":
            self.j_mngr.log_events(f"Param output: {str(param_list)}", is_trouble=True)
            return (param_list, _help, self.trbl.get_troubles())
        
        # Handle User Defined case
        if Parameter_type == "User Defined":
            if Param_Name and Param_Value:
                self.j_mngr.log_events(f"Processing User Defined parameter: {Param_Name}", is_trouble=True)
                temp_dict = {'param': Param_Name}
                
                # Attempt to parse as JSON if specified
                if Is_JSON:
                    p_json = self.j_mngr.convert_from_json_string(Param_Value)
                    if p_json:
                        temp_dict['value'] = p_json
                    else:
                        self.j_mngr.log_events(f"Invalid JSON presented to Additional Parameters. Node: {unique_id}",
                                            TroubleSgltn.Severity.ERROR,
                                            True)
                        self.j_mngr.log_events(f"Param output: {str(param_list)}", is_trouble=True)
                        return (param_list, _help, self.trbl.get_troubles())
                else:
                    # Infer type 
                    i_value = self.j_mngr.infer_type(Param_Value)
                    temp_dict['value'] = i_value

            else:
                self.j_mngr.log_events("Parameter Name or Value is missing. Parameter was not processed.",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
                self.j_mngr.log_events(f"Param output: {str(param_list)}", is_trouble=True)
                return (param_list, _help, self.trbl.get_troubles())
            
            if temp_dict:
                param_list.append(temp_dict)
                self.j_mngr.log_events(f"Param output: {str(param_list)}", is_trouble=True)
                
        return (param_list, _help, self.trbl.get_troubles())




class AdvPromptEnhancer:
    #Advance Prompt Enhancer: User entered Instruction, Prompt and Examples

    def __init__(self)-> None:
        self.cFig = cFigSingleton()
        self.help_data = helpSgltn()
        self.j_mngr = json_manager()
        self.trbl = TroubleSgltn()
        self.ctx = rqst.request_context()
        self.m_ttl = rqst.ollama_unload_request.ModelTTL #Enum

    def get_model(self, GPT_model, Groq_model, Gemini_model, Anthropic_model, Ollamm_model, Optional_model, connection_type)->str:
        
        if connection_type == "ChatGPT":
            return GPT_model
        
        if connection_type == "Groq":
            return Groq_model
        
        if connection_type == "Gemini":
            return Gemini_model

        if connection_type == "Anthropic":
            return Anthropic_model
        
        if "Ollama" in connection_type and Ollamm_model != "none":
            return Ollamm_model
        
        if Optional_model and Optional_model != "none":
            template = {"content": None}
            model_dlist = []
            model_dlist = self.j_mngr.insert_string_dict(Optional_model,template,"content","::")
            if len(model_dlist) > 1:
                return model_dlist[1]['content']

            return model_dlist[0]['content']

        return "none"        
    
    def model_ttl (self, selection:str)->rqst.ollama_unload_request.ModelTTL:
        """Translates user input from menu to appropriate enum value"""
        if selection == "Unload After Run":
            return self.m_ttl.KILL
        
        if selection == "Keep Alive Indefinitely":
            return self.m_ttl.INDEF
        
        return self.m_ttl.NOSET
    

    @classmethod
    def INPUT_TYPES(cls):
        cFig = cFigSingleton()
        gptfilter = ("gpt","o1", "o3","o4")

        return {
            "required": {
                "AI_service": (["ChatGPT", "Groq", "Anthropic","Gemini", "LM_Studio (URL)", "Ollama (URL)","OpenAI API Connection (URL)", "Direct Web Connection (URL)", "Web Connection Simplified Data (URL)", "Oobabooga API (URL)"], {"default": "Groq", "tooltip": "Choose connection type/service, connections ending with '(URL)' require a URL to be entered below"}),
                "ChatGPT_model": (cFig.get_chat_models(True,gptfilter), {"default": ""}),
                "Groq_model": (cFig.get_groq_models().get_models(True), {"default": ""}), 
                "Google_Gemini_model": (cFig.get_gemini_models().get_models(True, exclude_filter=("1.0","embedding","bison","imagen", "image-gen","learnlm","pro-vision", "aqa")), {"default": ""}),
                "Anthropic_model": (cFig.get_claude_models().get_models(False, exclude_filter=("2.0", "2.1")), {"default": ""}),
                "Ollama_model": (cFig.get_ollama_models().get_models(True), {"default": ""}), 
                "Ollama_model_unload": (["Unload After Run", "Keep Alive Indefinitely", "No Setting"], {"default": "No Setting", "tooltip": "Choose how long this model will stay loaded after completion"}),
                "Optional_model": (cFig.get_optional_models(True), {"default": "", "tooltip": "Enter these in the text file: 'opt_models.txt' in the Plush directory"}),                
                "creative_latitude" : ("FLOAT", {"max": 1.901, "min": 0.1, "step": 0.1, "display": "number", "round": 0.1, "default": 0.7, "tooltip": "temperature"}),                  
                "tokens" : ("INT", {"max": 20000, "min": 20, "step": 10, "default": 800, "display": "number"}), 
                "seed": ("INT", {"default": 9, "min": 0, "max": 0xffffffffffffffff}),
                "examples_delimiter":(["Pipe |", "Two newlines", "Two colons ::"], {"default": "Two newlines"}),
                "LLM_URL": ("STRING",{"default": "", "tooltip": "Enter the url for your service here when using connections that end with: (URL)"}),  #Removed "default": "cFig.lm_url"
                "Number_of_Tries": (["1","2","3","4","5","default"], {"default": "default"})            
                         
            },

            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {  
                "Instruction": ("STRING",{"multiline": True, "default": "", "forceInput": True}),
                "Examples_or_Context": ("STRING",{"multiline": True, "default": "", "forceInput": True}),
                "Prompt": ("STRING",{"multiline": True, "default": "", "forceInput": True}),
                "Add_Parameter": ("LIST", {"default": None, "forceInput": True}),
                "Custom_ApiKey":("KEY",{"default": "", "forceInput": True}),
                "image" : ("IMAGE", {"default": None})                          
                
            }
        } 

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("LLMprompt", "Context", "Help","Troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Prompt"

    def gogo(self, AI_service, ChatGPT_model, Groq_model, Google_Gemini_model, Anthropic_model, Ollama_model, Ollama_model_unload, Optional_model, creative_latitude, tokens, seed, examples_delimiter, 
              Number_of_Tries:str="", Add_Parameter=None, LLM_URL:str="", Instruction:str="", Prompt:str = "", Custom_ApiKey:str="", Examples_or_Context:str ="", image=None, unique_id=None):

        if unique_id:
            self.trbl.reset("Advanced Prompt Enhancer, Node #"+unique_id)
        else:
            self.trbl.reset("Advanced Prompt Enhancer")

        _help = self.help_data.adv_prompt_help


        # fix/validate the values of any unconnected inputs
        Instruction = Enhancer.undefined_to_none(Instruction)
        Prompt = Enhancer.undefined_to_none(Prompt)
        Examples = Enhancer.undefined_to_none(Examples_or_Context)
        LLM_URL = Enhancer.undefined_to_none(LLM_URL)
        image = Enhancer.undefined_to_none(image)
        if Custom_ApiKey is None or Custom_ApiKey == "undefined":
            Custom_ApiKey = ""  
        self.cFig.custom_key = Custom_ApiKey 
        
        if not isinstance(Add_Parameter, list):
            Add_Parameter = []

        remote_model = self.get_model(ChatGPT_model, Groq_model, Google_Gemini_model, Anthropic_model, Ollama_model, Optional_model, AI_service)
        model_ttl = self.model_ttl(Ollama_model_unload)
      
        if remote_model == "none":
            self.j_mngr.log_events("No model selected. If you're using a local desktop application, most will just use the loaded model.",
                                   TroubleSgltn.Severity.INFO,
                                   True)

        llm_result = "Unable to process request.  Make sure the local Open Source Server is running, and you've provided a valid URL.  If you're using a remote service (e.g.: ChaTGPT, Groq) make sure your key is valid, and a model is selected"

        #Create a list of dictionaries out of the user provided Examples_or_Context
        example_list = []    
        

        delimiter = None
        if examples_delimiter == "Two newlines":
            delimiter = "\n\n"
        elif examples_delimiter == "Two colons ::":
            delimiter = "::"
        elif examples_delimiter == "Pipe |":
            delimiter = "|"

        if Examples:
            example_list = self.j_mngr.build_context(Examples, delimiter)
        
        kwargs = { "model": remote_model,
                "creative_latitude": creative_latitude,
                "tokens": tokens,
                "seed": seed,
                "prompt": Prompt,
                "instruction": Instruction,
                "url": LLM_URL,
                "image": image,
                "example_list": example_list,
                "add_params": Add_Parameter,
                "tries": Number_of_Tries
        }
        context_output = ""
        if delimiter == "\n\n":
            ctx_delimiter = delimiter
        else:
            ctx_delimiter = "\n" + delimiter +"\n"

        context_output = (Examples + ctx_delimiter if Examples else "") + Prompt + ctx_delimiter

        if AI_service in ('OpenAI API Connection (URL)', 'Groq', 'Ollama (URL)', 'Gemini'):
    # Your code here: 
            
            unload_ctx = None #Initialize Model Unload request for use if Ollama request

            if AI_service == 'OpenAI API Connection (URL)':
                self.cFig.lm_request_mode = RequestMode.OPENSOURCE
            elif AI_service == "Groq":
                self.cFig.lm_request_mode = RequestMode.GROQ  
                LLM_URL = "https://api.groq.com/openai/v1" # Ugh!  I've embedded a 'magic value' URL here for the OPENAI API Object because the GROQ API object looks flakey...
            elif AI_service == "Ollama (URL)":
                self.cFig.lm_request_mode = RequestMode.OLLAMA
                if not LLM_URL:
                    LLM_URL = 'http://localhost:11434/v1'  #If URL unspecified
                unload_ctx = rqst.request_context()
                unload_ctx.request = rqst.ollama_unload_request()                
            elif AI_service == "Gemini":
                self.cFig.lm_request_mode = RequestMode.GEMINI
                LLM_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

            if not LLM_URL:
                self.j_mngr.log_events("'OpenAI API Connection (URL)' specified, but no URL provided or URL is invalid. Enter a valid URL",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
                return(llm_result,"", _help, self.trbl.get_troubles()) 

            # set the url so the function making the request will have a properly initialized object.               
            self.cFig.lm_url = LLM_URL 
            
            if not self.cFig.lm_client:
                self.j_mngr.log_events("Open Source LLM server is not running.  Aborting request.",
                                       TroubleSgltn.Severity.WARNING,
                                       True)
                return(llm_result,"", _help, self.trbl.get_troubles())

            llm_result = ""
            self.ctx.request = rqst.oai_object_request( )

            llm_result = self.ctx.execute_request(**kwargs)

            context_output += llm_result

            if unload_ctx: #If uload_ctx has been instantiated, execute the user's model unload setting
                unload_ctx.execute_request(model=remote_model, url=LLM_URL, model_TTL=model_ttl)

            return(llm_result, context_output, _help, self.trbl.get_troubles())
        

        if  AI_service == 'Anthropic':
 
            self.cFig.lm_request_mode = RequestMode.CLAUDE           
            claude_result = ""
            self.ctx.request = rqst.claude_request()

            claude_result = self.ctx.execute_request(**kwargs)

            context_output += claude_result

            return(claude_result, context_output, _help, self.trbl.get_troubles())

        
        if AI_service in ("Direct Web Connection (URL)", "LM_Studio (URL)"):

            if not LLM_URL:
                if AI_service == "LM_Studio (URL)":
                    LLM_URL = "http://localhost:1234/v1/chat/completions" #Set default URL if user doesn't provide one
                    kwargs['url'] = LLM_URL
                else: 
                    self.j_mngr.log_events("'Direct Web Connection (URL)' specified, but no URL provided or URL is invalid. Enter a valid URL",
                                        TroubleSgltn.Severity.WARNING,
                                        True)
                    return(llm_result, "", _help, self.trbl.get_troubles())  
            
            self.ctx.request = rqst.oai_web_request()

            if urlparse(LLM_URL).path == "/v1":
                self.j_mngr.log_events("Web connection urls should be either: '/v1/chat/completions' or (Olama) '/api/generate'. ",
                                       TroubleSgltn.Severity.WARNING,
                                       True)
            
            if AI_service == "LM_Studio (URL)":
                self.cFig.lm_request_mode = RequestMode.LMSTUDIO                    
            else:
                self.cFig.lm_request_mode = RequestMode.OPENSOURCE

            llm_result = self.ctx.execute_request(**kwargs)

            context_output += llm_result

            return(llm_result, context_output, _help, self.trbl.get_troubles())            
        
        if AI_service == "Web Connection Simplified Data (URL)":
            if not LLM_URL:
                self.j_mngr.log_events("'Web Connection Simplified Data (URL)' specified, but no URL provided or URL is invalid. Enter a valid URL",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
                return(llm_result, "", _help, self.trbl.get_troubles())  
            
            self.ctx.request = rqst.oai_web_request()
            self.cFig.lm_request_mode = RequestMode.OSSIMPLE

            llm_result = self.ctx.execute_request(**kwargs)

            context_output += llm_result

            return(llm_result,context_output, _help, self.trbl.get_troubles())  

        #Oobabooga via POST
        if AI_service == "Oobabooga API (URL)":
            if not LLM_URL:
                self.j_mngr.log_events("'Oobabooga API-URL' specified, but no URL provided or URL is invalid. Enter a valid URL",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
                return(llm_result,"", _help, self.trbl.get_troubles())  

            self.ctx.request = rqst.ooba_web_request()
            self.cFig.lm_request_mode = RequestMode.OOBABOOGA

            llm_result = self.ctx.execute_request(**kwargs)

            context_output += llm_result

            return(llm_result, context_output,  _help, self.trbl.get_troubles())   

        """
        if AI_service == "Gemini":
            self.cFig.lm_request_mode = RequestMode.GEMINI
            self.ctx.request = rqst.genaiRequest()
            kwargs['completion_mode'] = self.ctx.request.CompletionMode.TEXT.value

            llm_result = self.ctx.execute_request(**kwargs)      

            context_output += llm_result
            return(llm_result,context_output, _help, self.trbl.get_troubles()) 
        """


        #OpenAI ChatGPT request
        self.ctx.request = rqst.oai_object_request()
        self.cFig.lm_request_mode = RequestMode.OPENAI

        llm_result = self.ctx.execute_request(**kwargs)
        context_output += llm_result
       
        return(llm_result,context_output, _help, self.trbl.get_troubles())
    


class DalleImage:
#Accept a user prompt and parameters to produce a Dall_e generated image

    def __init__(self):
        self.cFig = cFigSingleton()
        self.help_data = helpSgltn()
        self.j_mngr = json_manager()
        self.trbl = TroubleSgltn()
        self.ctx = rqst.request_context()

        
    @classmethod
    def INPUT_TYPES(cls):
        #dall-e-2 API requires differnt input parameters as compared to dall-e-3, at this point I'll just use dall-e-3
        #                 "batch_size": ("INT", {"max": 8, "min": 1, "step": 1, "default": 1, "display": "number"})
        # Possible future implentation of batch_sizes greater than one.
        #                "image" : ("IMAGE", {"forceInput": True}),
        return {
            "required": {
                "GPTmodel": (["dall-e-3",], ),
                "prompt": ("STRING",{"multiline": True, "forceInput": True}), 
                "image_size": (["1792x1024", "1024x1792", "1024x1024"], {"default": "1024x1024"} ),              
                "image_quality": (["standard", "hd"], {"default": "hd"} ),
                "style": (["vivid", "natural"], {"default": "natural"} ),
                "batch_size": ("INT", {"max": 8, "min": 1, "step": 1, "default": 1, "display": "number"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "Number_of_Tries": (["1","2","3","4","5","default"], {"default": "default"})
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        } 

    RETURN_TYPES = ("IMAGE", "STRING","STRING","STRING" )
    RETURN_NAMES = ("image", "Dall_e_prompt","help","troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Image_Gen"

    def gogo(self, GPTmodel, prompt, image_size, image_quality, style, batch_size, seed, Number_of_Tries:str, unique_id=None):

        if unique_id:
            self.trbl.reset('Dall-e Image, Node #' + unique_id)
        else:
            self.trbl.reset('Dall-e Image Node')


        _help = self.help_data.dalle_help
        self.cFig.lm_request_mode = RequestMode.DALLE
        self.ctx.request = rqst.dall_e_request()
        kwargs = { "model": GPTmodel,
                "prompt": prompt,
                "image_size": image_size,
                "image_quality": image_quality,
                "style": style,
                "batch_size": batch_size,
                "tries": Number_of_Tries
        }
        batched_images, revised_prompt = self.ctx.execute_request(**kwargs)

        return (batched_images, revised_prompt, _help, self.trbl.get_troubles())
    

class GPTImage:
#Accept a user prompt and parameters to produce a GPT generated image

    def __init__(self):
        self.cFig = cFigSingleton()
        self.help_data = helpSgltn()
        self.j_mngr = json_manager()
        self.Ehc = Enhancer()
        self.trbl = TroubleSgltn()
        self.ctx = rqst.request_context()
        
    @classmethod
    def INPUT_TYPES(cls):
        #dall-e-2 API requires differnt input parameters as compared to dall-e-3, at this point I'll just use dall-e-3
        #                 "batch_size": ("INT", {"max": 8, "min": 1, "step": 1, "default": 1, "display": "number"})
        # Possible future implentation of batch_sizes greater than one.
        #                "image" : ("IMAGE", {"forceInput": True}),
        return {
            "required": {
                "GPTmodel": (["gpt-image-1"], ),
                "image_size": (["1024x1024","1536x1024","1024x1536"], {"default": "1024x1024"} ),              
                "image_quality": (["low","medium","high"], {"default": "high"} ),
                "batch_size": ("INT", {"max": 8, "min": 1, "step": 1, "default": 1, "display": "number"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "Number_of_Tries": (["1","2","3","4","5","default"], {"default": "default"})
            },
            "optional": {
                "image": ("IMAGE", {"default": None, "forceInput": True}), #Image to be edited
                "mask": ("MASK", {"default": None, "forceInput": True}), #Mask to be used for editing
                "prompt": ("STRING",{"multiline": True, "forceInput": True})                 
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        } 

    RETURN_TYPES = ("IMAGE", "STRING","STRING")
    RETURN_NAMES = ("image","help","troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Image_Gen"

    def gogo(self, GPTmodel, image_size, image_quality, batch_size, seed, Number_of_Tries:str, prompt="", image=None, mask=None, unique_id=None):

        if unique_id:
            self.trbl.reset('GPT Image, Node #' + unique_id)
        else:
            self.trbl.reset('GPT Image Node')

        _help = self.help_data.gpt_img_help


        if not prompt:
            self.j_mngr.log_events("No prompt provided.  Unable to process request.",
                                   TroubleSgltn.Severity.WARNING,
                                   True)
            return (rqst.gpt_image_request.blank_tensor, _help, self.trbl.get_troubles())
        
        image = Enhancer.undefined_to_none(image)
        mask = Enhancer.undefined_to_none(mask)

        kwargs = { "model": GPTmodel,
                "prompt": prompt,
                "image_size": image_size,
                "image_quality": image_quality,
                "batch_size": batch_size,
                "tries": Number_of_Tries
        }

        #These will be tensors, must be converted to PIL images of the same type with alpha channel, before being passed to the request
        if image is not None: 
            kwargs['image'] = image
            # If an image is provided, use Edit mode
            self.cFig.lm_request_mode = RequestMode.GPTIMAGEEDIT
            self.j_mngr.log_events("Image detected - using Edit mode", is_trouble=True)
            if mask is not None: #Don't allow mask if there's no image
                kwargs['mask'] = mask            
        else:
            # If no image is provided, use Generate mode
            self.cFig.lm_request_mode = RequestMode.GPTIMAGEGEN
            self.j_mngr.log_events("No image detected - using Generate mode", is_trouble=True)
            
        self.ctx.request = rqst.gpt_image_request()
        
        batched_images = self.ctx.execute_request(**kwargs)

        return (batched_images, _help, self.trbl.get_troubles())    
    

class ImagenImage:
#Accept a user prompt and parameters to produce a Dall_e generated image

    def __init__(self):
        self.cFig = cFigSingleton()
        self.help_data = helpSgltn()
        self.j_mngr = json_manager()
        self.trbl = TroubleSgltn()
        self.ctx = rqst.request_context()


    @classmethod
    def INPUT_TYPES(cls):

        cFig = cFigSingleton()

        return {
            "required": {
                "Image_model": (cFig.get_gemini_models().get_models(True, include_filter=('imagen','Imagen')),{"default":"none"}),                 
                "aspect_ratio" : (["1:1", "3:4", "4:3", "9:16", "16:9"], {"default": "1:1"}),
                "number_of_images": ("INT", {"max": 8, "min": 1, "step": 1, "default": 1, "display": "number"}),                
                "prompt": ("STRING",{"multiline": True, "forceInput": False}),                 
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "Custom_ApiKey":("KEY",{"default": "", "forceInput": True}),                          
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        } 

    RETURN_TYPES = ("IMAGE", "STRING","STRING" )
    RETURN_NAMES = ("image", "help","troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Image_Gen"

    def gogo(self, Image_model, prompt, aspect_ratio, number_of_images, seed, Custom_ApiKey=None, unique_id=None):   


        if unique_id:
            self.trbl.reset('Imagen Image, Node #' + unique_id)
        else:
            self.trbl.reset('Imagen Image Node')

        if Custom_ApiKey is None or Custom_ApiKey == "undefined":
            Custom_ApiKey = ""  
        self.cFig.custom_key = Custom_ApiKey 
             

        if Custom_ApiKey:
            self.cFig.lm_request_mode = RequestMode.OPENSOURCE
        else:
            self.cFig.lm_request_mode = RequestMode.GEMINI         

        _help = self.help_data.imagen_img_help

        self.ctx.request = rqst.ImagenRequest()

        kwargs = { "model": Image_model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "number_of_images": number_of_images
        }
        batched_images = self.ctx.execute_request(**kwargs) 

        return (batched_images, _help, self.trbl.get_troubles())
    

class GeminiImage:
#Accept a user prompt and parameters to produce a Gemini generated image

    def __init__(self):
        self.cFig = cFigSingleton()
        self.help_data = helpSgltn()
        self.j_mngr = json_manager()
        self.trbl = TroubleSgltn()
        self.ctx = rqst.request_context()


    @classmethod
    def INPUT_TYPES(cls):
        #dall-e-2 API requires differnt input parameters as compared to dall-e-3, at this point I'll just use dall-e-3
        #                 "batch_size": ("INT", {"max": 8, "min": 1, "step": 1, "default": 1, "display": "number"})
        # Possible future implentation of batch_sizes greater than one.
        #                "image" : ("IMAGE", {"forceInput": True}),
        cFig = cFigSingleton()
        return {
            "required": {
                "Image_model": (cFig.get_gemini_models().get_models(True, include_filter=("2.0","flash"), exclude_filter=("lite","pro","thinking","1.5")), {"default": "gemini-2.0-flash-exp"}),
                "creative_latitude" : ("FLOAT", {"max": 1.901, "min": 0.1, "step": 0.1, "display": "number", "round": 0.1, "default": 0.7, "tooltip": "temperature"}),                  
                "tokens" : ("INT", {"max": 20000, "min": 20, "step": 10, "default": 800, "display": "number"}),
                "prompt": ("STRING",{"multiline": True, "forceInput": False}),                 
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "Custom_ApiKey":("KEY",{"default": "", "forceInput": True}), 
                "Add_Parameter": ("LIST", {"default": None, "forceInput": True}),   
                "image" : ("IMAGE", {"default": None})                               
                            
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        } 

    RETURN_TYPES = ("IMAGE", "STRING", "STRING","STRING" )
    RETURN_NAMES = ("image", "text", "help", "troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Image_Gen"

    def gogo(self, Image_model, prompt, creative_latitude, tokens, seed, Custom_ApiKey=None, Add_Parameter=None, image=None, unique_id=None):   


        if unique_id:
            self.trbl.reset('Gemini Image, Node #' + unique_id)
        else:
            self.trbl.reset('Gemini Image Node')

        if Custom_ApiKey is None or Custom_ApiKey == "undefined":
            Custom_ApiKey = ""  
        self.cFig.custom_key = Custom_ApiKey 
             

        if Custom_ApiKey:
            self.cFig.lm_request_mode = RequestMode.OPENSOURCE
        else:
            self.cFig.lm_request_mode = RequestMode.GEMINI        
            
        if not isinstance(Add_Parameter, list):
            Add_Parameter = []

        _help = self.help_data.gemini_img_help 

        image = Enhancer.undefined_to_none(image)

        self.ctx.request = rqst.genaiRequest()
        image_tensor = rqst.genaiRequest.blank_tensor
        text_out = ""

        kwargs = { 
            "model": Image_model,
            "creative_latitude": creative_latitude,
            "tokens": tokens,
            "image": image,
            "prompt": prompt,
            "add_params": Add_Parameter,
            "completion_mode": rqst.genaiRequest.CompletionMode.TEXT_IMAGE.value
        }

        llm_result = self.ctx.execute_request(**kwargs)      

        if 'images' in llm_result and 'tensor' in llm_result['images']:
            # Extract just the tensor and store it where needed
            image_tensor = llm_result['images']['tensor']

        if 'text' in llm_result and llm_result['text']:
            text_out = llm_result['text']

        return(image_tensor, text_out, _help, self.trbl.get_troubles())  
    

class ImageInfoExtractor:

    def __init__(self):
        #self.Enh = Enhancer()
        self.j_mngr = json_manager()
        self.cFig = cFigSingleton()
        self.help_data = helpSgltn()
        self.trbl = TroubleSgltn()


    def sanitize_data(self, v):

        def contains_nonprintable(s):
            # Tests for the presence of disallowed non printable chars
            allowed_nonprintables = {'\n', '\r', '\t'}
            return any(c not in allowed_nonprintables and not c.isprintable() for c in s)

        if isinstance(v, bytes):
            # Attempt to decode byte data
            decoded_str = v.decode('utf-8', errors='replace')
            # Check if the result contains any non-allowed non-printable characters
            if not contains_nonprintable(decoded_str):
                return decoded_str
            return None
        elif isinstance(v, TiffImagePlugin.IFDRational):
            if v.denominator == 0:
                return None
            return float(v)
        elif isinstance(v, tuple):
            return tuple(self.sanitize_data(t) for t in v if self.sanitize_data(t) is not None)
        elif isinstance(v, dict):
            return {kk: self.sanitize_data(vv) for kk, vv in v.items() if self.sanitize_data(vv) is not None}
        elif isinstance(v, list):
            return [self.sanitize_data(item) for item in v if self.sanitize_data(item) is not None]
        else:
            return v


    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {                    
                    "write_to_file" : ("BOOLEAN", {"default": False}),
                    "Only_show_prompts": ("BOOLEAN", {"default": False}),
                    "file_prefix": ("STRING",{"default": "MetaData_"}),
                    "Min_prompt_len": ("INT", {"max": 2500, "min": 3, "step": 1, "default": 72, "display": "number"}),
                    "Alpha_Char_Pct": ("FLOAT", {"max": 1.001, "min": 0.01, "step": 0.01, "display": "number", "round": 0.01, "default": 0.90}), 
                    "Prompt_Filter_Term": ("STRING", {"multiline": False, "default": ""}),               
                    "image": (sorted(files), {"image_upload": True}),
                 },
                "hidden": {
                    "unique_id": "UNIQUE_ID",
                    
                }
        }
    
    CATEGORY = "Plush🧸/Utils"

    RETURN_TYPES = ("STRING","STRING","STRING")
    RETURN_NAMES = ("Image_info","help","troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = True

    def gogo(self, image, write_to_file, Only_show_prompts, file_prefix, Min_prompt_len, Alpha_Char_Pct,Prompt_Filter_Term, unique_id=None):

        if unique_id:
            self.trbl.reset('Exif Wrangler, Node# ' + unique_id) #Clears all trouble logs before a new run and passes the name of process to head the log lisiing
        else:
            self.trbl.reset('Exif Wrangler')

        _help = self.help_data.exif_wrangler_help #Establishes access to help files
        output = "Unable to process request"
        fatal = False
        exiv_comment = {}
        Show_headings = not Only_show_prompts

        #Make sure the pyexiv2 supporting library was able to load.  Otherwise exit gogo
        if not self.cFig.pyexiv2:
            self.j_mngr.log_events("Unable to load supporting library 'pyexiv2'.  This node is not functional.",
                                   TroubleSgltn.Severity.ERROR,
                                   True)
            return(output, _help, self.trbl.get_troubles())
        else:
            pyexiv2 = self.cFig.pyexiv2

        #Create path and dir for saved .txt files
        write_dir = ''
        comfy_dir = self.j_mngr.comfy_dir
        if comfy_dir:
            output_dir = self.j_mngr.find_child_directory(comfy_dir,'output')
            if output_dir:
                write_dir = self.j_mngr.find_child_directory(output_dir, 'PlushFiles',True) #Setting argument to True means dir will be created if not present
                if not write_dir:
                    self.j_mngr.log_events('Unable to find or create PlushFiles directory. Unable to write files',
                                    TroubleSgltn.Severity.WARNING,
                                    True)
            else:
                self.j_mngr.log_events('Unable to find output directory, Unable to write files',
                                   TroubleSgltn.Severity.WARNING,
                                   True)
        else:
            self.j_mngr.log_events('Unable to find ComfyUI directory. Unable to write files.',
                                   TroubleSgltn.Severity.WARNING,
                                   True)
        



        #Start potential separate method: def get_image_metadata()->Tuple
        #Get general meta-data and exif data and combine them
        image_path = folder_paths.get_annotated_filepath(image)
        try:
            with Image.open(image_path) as img:
                info = img.info
        except FileNotFoundError:
            self.j_mngr.log_events(f"Image file not found: {image_path}",
                                    TroubleSgltn.Severity.ERROR,
                                    True)
            fatal = True
        except PermissionError:
            self.j_mngr.log_events(f"Permission denied for image file: {image_path}",
                                    TroubleSgltn.Severity.ERROR,
                                    True)
            fatal = True
        except UnidentifiedImageError as e:
            self.j_mngr.log_events(f"Exif Wrangler was unable to identify image file: {image_path}; {e}",
                                    TroubleSgltn.Severity.ERROR,
                                    True)
            fatal = True
        except OSError as e:
            self.j_mngr.log_events(f"An Error occurred while opening the image file: {e}",
                                    TroubleSgltn.Severity.ERROR,
                                    True)
            fatal = True
        except ValueError:
            self.j_mngr.log_events(f"Invalid value for image path: {image_path}",
                                    TroubleSgltn.Severity.ERROR,
                                    True)
            fatal = True
        except MemoryError as e:
            self.j_mngr.log_events(f"Memory error occurred while processing the image: {e}",
                                    TroubleSgltn.Severity.ERROR,
                                    True)
            fatal = True
        except Exception as e:
            self.j_mngr.log_events(f"An unexpected error occurred: {e}",
                                    TroubleSgltn.Severity.ERROR,
                                    True)
            fatal = True

        if fatal:
            return(output,_help,self.trbl.get_troubles())
        
        img_file_name = os.path.basename(image_path)

        pyexiv2.set_log_level(4) #Mute log level
        try:
            with pyexiv2.Image(image_path) as exiv_img:
                exiv_exif = exiv_img.read_exif()
                exiv_iptc = exiv_img.read_iptc()
                exiv_xmp = exiv_img.read_xmp()
                exiv_comment = exiv_img.read_comment()
        except Exception as e:
            self.j_mngr.log_events(f'Unable to process Image file: {e}',
                                    TroubleSgltn.Severity.WARNING,
                                    True)
            output = "Unable to process image file."
            return(output, _help, self.trbl.get_troubles())

        if not exiv_comment:
            exiv_comment = {'comment':'null'}

        exiv_tag = {'processing_details':{
                         'ew_file': img_file_name, 
                         'path': image_path, 
                         'ew_id':'ComfyUI: Plush Exif Wrangler'
                    }
        }
        exiv_comm = {**exiv_comment, **exiv_tag}

        self.j_mngr.log_events(f"Evaluating image file: '{os.path.basename(image_path)}'",
                               is_trouble=True)

        # Sanitize and combine data             
        sanitized_exiv2 = {k: self.sanitize_data(v) for k, v in exiv_exif.items()} if exiv_exif else {}
        sanitized_xmp = {k: self.sanitize_data(v) for k, v in exiv_xmp.items()} if exiv_xmp else {}

        #extract the pertinent data subset from info
        extract_values = ['widgets_values','inputs']
        extracted_info = self.j_mngr.extract_from_dict(info, extract_values)
        

        working_meta_data = {**sanitized_xmp, **exiv_iptc,  **exiv_comm,**sanitized_exiv2, **extracted_info}
        #End potential separate method: get_image_metadata, Returns(meta_data, info, image_path)

        #Var to save a raw copy of working_meta_data for debug.
        #Leave as False except when in debug mode.
        debug_save = False

        #Print source dict working_meta_data or info to debug issues.
        if debug_save:
            debug_file_name = self.j_mngr.generate_unique_filename("json", 'debug_source_file')
            debug_file_path = self.j_mngr.append_filename_to_path(write_dir,debug_file_name)
            debug_json = self.j_mngr.convert_to_json_string(working_meta_data) #all data after first extraction
            #debug_json = self.j_mngr.convert_to_json_string(info) #Raw AI Gen data pre extraction, but w/o Exif info
            self.j_mngr.write_string_to_file(debug_json,debug_file_path,False)
        #Begin potential separate method def get_data_objects()->Tuple
        #process user intp to list

         
        exif_keys = {'Exif.Photo.UserComment': 'User Comment',
                        'Exif.Image.Make': 'Make',
                        'Exif.Image.Model': 'Model',
                        'Exif.Image.Orientation': 'Image Orientation',
                        'Exif.Photo.PixelXDimension': 'Pixel Width',
                        'Exif.Photo.PixelYDimension': 'Pixel Height',
                        'Exif.Photo.ISOSpeedRatings': 'ISO',
                        'Exif.Image.DateTime':'Created: Date_Time',
                        'Exif.GPSInfo.GPSLatitude': 'GPS Latitude',
                        'Exif.GPSInfo.GPSLatitudeRef': 'GPS Latitude Hemisphere',
                        'Exif.GPSInfo.GPSLongitude': 'GPS Longitude',
                        'Exif.GPSInfo.GPSLongitudeRef': 'GPS Longitude Hemisphere',
                        'Exif.GPSInfo.GPSAltitude': 'GPS Altitude',
                        'Exif.Photo.ShutterSpeedValue': 'Shutter Speed',
                        'Exif.Photo.ExposureTime': 'Exposure Time',
                        'Exif.Photo.Flash': "Flash",
                        'Exif.Photo.FocalLength': 'Lens Focal Length',
                        'Exif.Photo.FocalLengthIn35mmFilm': 'Lens 35mm Equiv. Focal Length',
                        'Exif.Photo.ApertureValue': 'Aperture',
                        'Exif.Photo.MaxApertureValue': 'Maximum Aperture',
                        'Exif.Photo.FNumber': 'f-stop',
                        'Exif.Image.Artist': 'Artist',
                        'Exif.Image.ExposureTime': 'Exposure Time',
                        'Exif.Image.MaxApertureValue': 'Camera Smallest Apeture',
                        'Exif.Image.MeteringMode': 'Metering Mode',
                        'Exif.Image.Flash': 'Flash',
                        'Exif.Image.FocalLength': 'Focal Length',
                        'Exif.Image.ExposureIndex': 'Exposure',
                        'Exif.Image.ImageDescription': 'Image Description',
                        'Xmp.OPMedia.IsHDRActive': 'HDR Active',
                        'Xmp.crs.UprightFocalLength35mm': '35mm Equiv Focal Length',
                        'Xmp.crs.LensProfileMatchKeyExifMake': 'Lens Make',
                        'Xmp.crs.LensProfileMatchKeyCameraModelName': 'Lens Model',
                        'Xmp.crs.CameraProfile': 'Camera Profile',
                        'Xmp.crs.WhiteBalance': 'White Balance',
                        'Xmp.xmp.CreateDate': 'Creation Date',
                        }
            

        #Testing translated extraction
        if Only_show_prompts:
            translate_keys = {'widgets_values': 'Possible Prompts',
                            'text': 'Possible Prompts'
                        }          
        else:
            translate_keys = {'widgets_values': 'Possible Prompts',
                            'text': 'Possible Prompts',
                            'steps': 'Steps',
                            'cfg': 'CFG',
                            'seed': 'Seed',
                            'noise_seed': 'Seed',
                            'ckpt_name': 'Models',
                            'resolution': 'Image Size',
                            'sampler_name': 'Sampler',
                            'scheduler': 'Scheduler',
                            'lora': 'Lora',
                            'denoise': 'Denoise',
                            'GPTmodel': 'OpenAI Model',
                            'image_size': 'Image Size',
                            'image_quality': 'Dall-e Image Quality',
                            'style': 'Style',
                            'batch_size': 'Batch Size',
                            'ew_file': 'Source File',
                            'ew_id': 'Processing Application'
                            }
        
        
        all_keys = {**exif_keys, **translate_keys}

        
    #End potential separate method get_data_objects returns(extract_values,discard_keys, translate_keys)
       
        working_dict = self.j_mngr.extract_with_translation(working_meta_data,all_keys,Min_prompt_len,Alpha_Char_Pct,Prompt_Filter_Term)
        #New step to remove possible candidate prompt duplicates
        dedupe_keys =['Possible Prompts',]
        self.j_mngr.remove_duplicates_from_keys(working_dict, dedupe_keys)

        output = self.j_mngr.prep_formatted_file(working_dict, Show_headings)

        if output: 
            if write_to_file and write_dir:
                working_file_name = self.j_mngr.generate_unique_filename("txt", file_prefix + '_ew_')
                working_file_path = self.j_mngr.append_filename_to_path(write_dir,working_file_name)
                self.j_mngr.write_string_to_file(output,working_file_path,False)
        else:
            output = "No metadata was found"
            self.j_mngr.log_events(f'No metadata was found in file: {os.path.basename(image_path)}',
                                    TroubleSgltn.Severity.INFO,
                                    True)

        return(output,_help,self.trbl.get_troubles())
               
      

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Enhancer": Enhancer,
    "AI Chooser": AI_Chooser,
    "AdvPromptEnhancer": AdvPromptEnhancer,
    "DalleImage": DalleImage,
    "Plush-Exif Wrangler" :ImageInfoExtractor,
    "Add Parameters": addParameters,
    "Custom API Key": CustomKeyVar,
    "Imagen Image": ImagenImage,
    "Gemini Image": GeminiImage,
    "GPT Image": GPTImage
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Enhancer": "Style Prompt🧸",
    "AI Chooser": "AI_Chooser🧸",
    "AdvPromptEnhancer": "Advanced Prompt Enhancer🧸",
    "DalleImage": "OAI Dall_e Image🧸",
    "Plush-Exif Wrangler": "Exif Wrangler🧸",
    "Add Parameters": "Add Parameters🧸",
    "Custom API Key": "Custom API Key🧸",
    "Imagen Image": "Imagen Image🧸",
    "Gemini Image": "Gemini Image🧸",
    "GPT Image": "GPT Image🧸"
}
