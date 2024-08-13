
import os
import base64
from io import BytesIO
from PIL import Image, ImageOps, TiffImagePlugin, UnidentifiedImageError
import folder_paths
import numpy as np
import torch
from typing import Optional
from enum import Enum
import requests
from requests.adapters import HTTPAdapter, Retry
from openai import OpenAI
import anthropic
from .mng_json import json_manager, helpSgltn, TroubleSgltn # add .
from . import api_requests as rqst
from .fetch_models import FetchModels, ModelUtils, RequestMode # add .



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
            cls._lm_url = ""
            cls._lm_request_mode = None
            cls._lm_key = ""
            cls._groq_key = ""
            cls._claude_key = ""
            cls._gemini_key = ""
            cls._lm_models = None
            cls._groq_models = None
            cls._claude_models = None
            cls._gemini_models = None
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
        self._groq_key = os.getenv("GROQ_API_KEY", "") or os.getenv("LLM_KEY", "")
        self._claude_key = os.getenv("ANTHROPIC_API_KEY", "") or os.getenv("LLM_KEY", "")
        self._gemini_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("LLM_KEY", "")
            
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
                self.figOAIClient = OpenAI(api_key= self._fig_key)
            except Exception as e:
                self.j_mngr.log_events(f"Invalid or missing OpenAI API key.  Please note, keys must now be kept in an environment variable (see: ReadMe) {e}",
                                  severity=TroubleSgltn.Severity.ERROR)
                
        if self._claude_key:
            try:
                self._anthropic_client = anthropic.Anthropic(api_key = self._claude_key)
            except Exception as e:
                self.j_mngr.log_events(f"Invalid or missing Anthropic API key. Please note, keys must be kept in an environment variable.{e}",                                       
                                       severity=TroubleSgltn.Severity.ERROR)     
                
                
        self._fig_gpt_models = self._model_fetch.fetch_models(RequestMode.OPENAI, self._fig_key)
        self._groq_models = self._model_fetch.fetch_models(RequestMode.GROQ, self._groq_key)
        self._claude_models = self._model_fetch.fetch_models(RequestMode.CLAUDE, self._claude_key)
        self._gemini_models = self._model_fetch.fetch_models(RequestMode.GEMINI, self._gemini_key)
                       
   
    def get_chat_models(self, sort_it:bool=False, filter_str:str="")->list:
        return self._model_prep.prep_models_list(self._fig_gpt_models, sort_it, filter_str)      
      
    def get_groq_models(self, sort_it:bool=False, filter_str:str=""):
        return self._model_prep.prep_models_list(self._groq_models, sort_it, filter_str)      

    def get_claude_models(self, sort_it:bool=False, filter_str:str="")->list:
        return self._model_prep.prep_models_list(self._claude_models, sort_it, filter_str)   

    def get_gemini_models(self, sort_it:bool=False, filter_str:str="")->list:       
        return self._model_prep.prep_models_list(self._gemini_models, sort_it, filter_str)            
        
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
            if not self._lm_key:
                self.j_mngr.log_events("Setting Openai client with URL, no key.",
                    is_trouble=True)
            else:
                key = self._lm_key
                self.j_mngr.log_events("Setting Openai client with URL and key.",
                    is_trouble=True)
        elif request_type == RequestMode.GROQ:
            if not self._groq_key:
                self.j_mngr.log_events("Attempting to connect to Groq with no key",
                                       TroubleSgltn.Severity.ERROR,
                                       True)
            else:
                key = self._groq_key
                self.j_mngr.log_events("Setting Openai client with URL and Groq key.",
                                       is_trouble=True)

        
        try:
            lm_client = lm_object(base_url=url, api_key=key) 
            self._lm_url = url
            self._lm_client = lm_client
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
        if url != self._lm_url or not self._lm_client:  # Check if the new URL is different to avoid unnecessary operations

            self._lm_url = url
            # Reset client and models only if a new URL is provided
            self._lm_client = None
            #self._lm_models = []
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
            return self.figOAIClient
        return None
    
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
            "LM_Studio": RequestMode.LMSTUDIO,
            "Local app (URL)": RequestMode.OPENSOURCE,
            "OpenAI compatible http POST": RequestMode.OPENSOURCE,
            "http POST Simplified Data": RequestMode.OSSIMPLE,
            "Oobabooga API-URL": RequestMode.OOBABOOGA
        }
        return mode_map.get(user_selection)
        

    @classmethod
    def INPUT_TYPES(cls):
        cFig=cFigSingleton()

        #Floats have a problem, they go over the max value even when round and step are set, and the node fails.  So I set max a little over the expected input value
        return {
            "required": {
                "AI_Service": (["ChatGPT", "Groq", "Anthropic"], {"default": "ChatGPT"}),
                "ChatGPT_model": (cFig.get_chat_models(True,'gpt'), {"default": ""}),
                "Groq_model": (cFig.get_groq_models(True), {"default": ""}), 
                "Anthropic_model": (cFig.get_claude_models(True), {"default": ""}),                                  
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        } 

    RETURN_TYPES = ("DICTIONARY",)
    RETURN_NAMES = ("AI_Selection",)

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush/Prompt"  

    def gogo(self, unique_id, AI_Service, ChatGPT_model, Groq_model, Anthropic_model):  

        ai_dict = {"service": AI_Chooser.select_request_mode(AI_Service), "model": None}

        if ai_dict['service'] == RequestMode.OPENAI and ChatGPT_model != "none":
            ai_dict['model'] = ChatGPT_model
        elif ai_dict['service'] == RequestMode.GROQ and Groq_model != "none":
            ai_dict['model'] = Groq_model
        elif ai_dict['service'] == RequestMode.CLAUDE and Anthropic_model != "none":
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

    CATEGORY = "Plush/Prompt"
 

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

        #Convert PyTorch.tensor to B64encoded image
        if isinstance(image, torch.Tensor):

            image = DalleImage.tensor_to_base64(image)

        #build instruction based on user input
        mode = 0
        if image and prompt:
            mode = InputMode.IMAGE_PROMPT
        elif image:
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
    
    

class AdvPromptEnhancer:
    #Advance Prompt Enhancer: User entered Instruction, Prompt and Examples

    def __init__(self)-> None:
        self.cFig = cFigSingleton()
        self.help_data = helpSgltn()
        self.j_mngr = json_manager()
        self.trbl = TroubleSgltn()
        self.ctx = rqst.request_context()

    def get_model(self, GPT_model, Groq_model, Anthropic_model, local_model, connection_type)->str:
        
        if connection_type == "ChatGPT":
            return GPT_model
        
        if connection_type == "Groq":
            return Groq_model

        if connection_type == "Anthropic":
            return Anthropic_model

        return local_model       
    

    @classmethod
    def INPUT_TYPES(cls):
        cFig = cFigSingleton()
        #open source models are too problematic to implement right now in an environment where you 
        #can't be sure if the local host server (open source) will be running, and where you can't
        #refresh the ui after the initial load.
        return {
            "required": {
                "AI_service": (["ChatGPT", "Groq", "Anthropic", "LM_Studio", "Local app (URL)", "OpenAI compatible http POST", "http POST Simplified Data", "Oobabooga API-URL"], {"default": "ChatGPT"}),
                "ChatGPT_model": (cFig.get_chat_models(True,'gpt'), {"default": ""}),
                "Groq_model": (cFig.get_groq_models(True), {"default": ""}), 
                "Anthropic_model": (cFig.get_claude_models(True), {"default": ""}), 
                "optional_local_model": ("STRING",{"default": "None"}),                 
                "creative_latitude" : ("FLOAT", {"max": 1.901, "min": 0.1, "step": 0.1, "display": "number", "round": 0.1, "default": 0.7}),                  
                "tokens" : ("INT", {"max": 8000, "min": 20, "step": 10, "default": 500, "display": "number"}), 
                "seed": ("INT", {"default": 9, "min": 0, "max": 0xffffffffffffffff}),
                "examples_delimiter":(["Pipe |", "Two newlines", "Two colons ::"], {"default": "Two newlines"}),
                "LLM_URL": ("STRING",{"default": cFig.lm_url})            
                         
            },

            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {  
                "Instruction": ("STRING",{"multiline": True, "default": "", "forceInput": True}),
                "Examples": ("STRING",{"multiline": True, "default": "", "forceInput": True}),
                "Prompt": ("STRING",{"multiline": True, "default": "", "forceInput": True}),
                "image" : ("IMAGE", {"default": None})                          
                
            }
        } 

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("LLMprompt", "Help","Troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush/Prompt"

    def gogo(self, AI_service, ChatGPT_model, Groq_model, Anthropic_model, optional_local_model, creative_latitude, tokens, seed, examples_delimiter, 
              LLM_URL:str="", Instruction:str="", Prompt:str = "", Examples:str ="",image=None, unique_id=None):

        if unique_id:
            self.trbl.reset("Advanced Prompt Enhancer, Node #"+unique_id)
        else:
            self.trbl.reset("Advanced Prompt Enhancer")

        _help = self.help_data.adv_prompt_help


        # set the value of unconnected inputs to None
        Instruction = Enhancer.undefined_to_none(Instruction)
        Prompt = Enhancer.undefined_to_none(Prompt)
        Examples = Enhancer.undefined_to_none(Examples)
        LLM_URL = Enhancer.undefined_to_none(LLM_URL)
        image = Enhancer.undefined_to_none(image)

        remote_model = self.get_model(ChatGPT_model, Groq_model, Anthropic_model, optional_local_model, AI_service)
      
        if remote_model == "None":
            self.j_mngr.log_events("No model selected. If you're using a Local application it will most likely use the loaded model.",
                                   TroubleSgltn.Severity.INFO,
                                   True)

        llm_result = "Unable to process request.  Make sure the local Open Source Server is running.  If you're using a remote service (e.g.: ChaTGPT, Groq) make sure your key is valid, and a model is selected"

        #Convert PyTorch.tensor to B64encoded image
        if isinstance(image, torch.Tensor):
            image = DalleImage.tensor_to_base64(image)

        #Create a list of dictionaries out of the user provided examples
        example_list = []    
        
        if Examples:
            delimiter ="|"
            if examples_delimiter == "Two newlines":
                delimiter = "\n\n"
            elif examples_delimiter == "Two colons ::":
                delimiter = "::"

            examples_template = {"role": "assistant", "content":None}
            example_list = self.j_mngr.insert_text_into_dict(Examples, examples_template, "content",delimiter)

        kwargs = { "model": remote_model,
                "creative_latitude": creative_latitude,
                "tokens": tokens,
                "seed": seed,
                "prompt": Prompt,
                "instruction": Instruction,
                "url": LLM_URL,
                "image": image,
                "example_list": example_list,
        }

        if  AI_service == 'Local app (URL)' or AI_service == "Groq": 
 
            if AI_service == 'Local app (URL)':
                self.cFig.lm_request_mode = RequestMode.OPENSOURCE
            elif AI_service == "Groq":
                self.cFig.lm_request_mode = RequestMode.GROQ  
                LLM_URL = "https://api.groq.com/openai/v1" # Ugh!  I've embedded a 'magic value' URL here for the OPENAI API Object because the GROQ API object looks flakey...

            if not LLM_URL:
                self.j_mngr.log_events("'Local app (URL)' specified, but no URL provided or URL is invalid. Enter a valid URL",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
                return("", _help, self.trbl.get_troubles()) 


            # set the url so the function making the request will have a properly initialized object.               
            self.cFig.lm_url = LLM_URL
            
            if not self.cFig.lm_client:
                self.j_mngr.log_events("Open Source LLM server is not running.  Aborting request.",
                                       TroubleSgltn.Severity.WARNING,
                                       True)
                return(llm_result, _help, self.trbl.get_troubles())

            llm_result = ""
            self.ctx.request = rqst.oai_object_request( )

            llm_result = self.ctx.execute_request(**kwargs)

            return(llm_result, _help, self.trbl.get_troubles())
        

        if  AI_service == 'Anthropic':
 
            self.cFig.lm_request_mode = RequestMode.CLAUDE           
            claude_result = ""
            self.ctx.request = rqst.claude_request()

            claude_result = self.ctx.execute_request(**kwargs)

            return(claude_result, _help, self.trbl.get_troubles())

        
        if AI_service == "OpenAI compatible http POST" or AI_service == "LM_Studio":
            if not LLM_URL:
                self.j_mngr.log_events("'OpenAI compatible http POST' specified, but no URL provided or URL is invalid. Enter a valid URL",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
                return(llm_result, _help, self.trbl.get_troubles())  
            
            self.ctx.request = rqst.oai_web_request()
            
            if AI_service == "LM_Studio":
                self.cFig.lm_request_mode = RequestMode.LMSTUDIO
            else:
                self.cFig.lm_request_mode = RequestMode.OPENSOURCE

            llm_result = self.ctx.execute_request(**kwargs)

            return(llm_result, _help, self.trbl.get_troubles())            
        
        if AI_service == "http POST Simplified Data":
            if not LLM_URL:
                self.j_mngr.log_events("'http POST Simplified Data' specified, but no URL provided or URL is invalid. Enter a valid URL",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
                return(llm_result, _help, self.trbl.get_troubles())  
            
            self.ctx.request = rqst.oai_web_request()
            self.cFig.lm_request_mode = RequestMode.OSSIMPLE

            llm_result = self.ctx.execute_request(**kwargs)

            return(llm_result, _help, self.trbl.get_troubles())  

        #Oobabooga via POST
        if AI_service == "Oobabooga API-URL":
            if not LLM_URL:
                self.j_mngr.log_events("'Oobabooga API-URL' specified, but no URL provided or URL is invalid. Enter a valid URL",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
                return(llm_result, _help, self.trbl.get_troubles())  

            self.ctx.request = rqst.ooba_web_request()
            self.cFig.lm_request_mode = RequestMode.OOBABOOGA

            llm_result = self.ctx.execute_request(**kwargs)

            return(llm_result, _help, self.trbl.get_troubles())            


        #OpenAI ChatGPT request
        self.ctx.request = rqst.oai_object_request()
        self.cFig.lm_request_mode = RequestMode.OPENAI

        llm_result = self.ctx.execute_request(**kwargs)
       
       
        return(llm_result, _help, self.trbl.get_troubles())
    


class DalleImage:
#Accept a user prompt and parameters to produce a Dall_e generated image

    def __init__(self):
        self.cFig = cFigSingleton()
        self.help_data = helpSgltn()
        self.j_mngr = json_manager()
        self.trbl = TroubleSgltn()
        self.ctx = rqst.request_context()

    @staticmethod    
    def b64_to_tensor( b64_image: str) -> tuple[torch.Tensor,torch.Tensor]:

        """
        Converts a base64-encoded image to a torch.Tensor.

        Note: ComfyUI expects the image tensor in the [N, H, W, C] format.  
        For example with the shape torch.Size([1, 1024, 1024, 3])

        Args:
            b64_image (str): The b64 image to convert.

        Returns:
            torch.Tensor: an image Tensor.
        """    
        j_mngr = json_manager()
        j_mngr.log_events("Converting b64 Image to Torch Tensor Image file",
                          is_trouble=True)    
        # Decode the base64 string
        image_data = base64.b64decode(b64_image)
        
        # Open the image with PIL and handle EXIF orientation
        image = Image.open(BytesIO(image_data))
        image = ImageOps.exif_transpose(image)
        
        # Convert to RGBA for potential alpha channel handling
        # Dalle doesn't provide an alpha channel, but this is here for
        # broad compatibility
        image = image.convert("RGBA")
        image_np = np.array(image).astype(np.float32) / 255.0  # Normalize

        # Split the image into RGB and Alpha channels
        rgb_np, alpha_np = image_np[..., :3], image_np[..., 3]
        
        # Convert RGB to PyTorch tensor and ensure it's in the [N, H, W, C] format
        tensor_image = torch.from_numpy(rgb_np).unsqueeze(0)  # Adds N dimension

        # Create mask based on the presence or absence of an alpha channel
        if image.mode == 'RGBA':
            mask = torch.from_numpy(alpha_np).unsqueeze(0).unsqueeze(0)  # Adds N and C dimensions
        else:  # Fallback if no alpha channel is present
            mask = torch.zeros((1, tensor_image.shape[2], tensor_image.shape[3]), dtype=torch.float32)  # [N, H, W]

        return tensor_image, mask
    

    @staticmethod
    def tensor_to_base64(tensor: torch.Tensor) -> str:
        """
        Converts a PyTorch tensor to a base64-encoded image.

        Note: ComfyUI provides the image tensor in the [N, H, W, C] format.  
        For example with the shape torch.Size([1, 1024, 1024, 3])

        Args:
            tensor (torch.Tensor): The image tensor to convert.

        Returns:
            str: Base64-encoded image string.
        """
        j_mngr = json_manager()
        j_mngr.log_events("Converting Torch Tensor image to b64 Image file",
                          is_trouble=True)
    # Convert tensor to PIL Image
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension if present
        pil_image = Image.fromarray((tensor.numpy() * 255).astype('uint8'))

        # Save PIL Image to a buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")  # Can change to JPEG if preferred
        buffer.seek(0)

        # Encode buffer to base64
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')

        return base64_image
    
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        } 

    RETURN_TYPES = ("IMAGE", "STRING","STRING","STRING" )
    RETURN_NAMES = ("image", "Dall_e_prompt","help","troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush/Image_Gen"

    def gogo(self, GPTmodel, prompt, image_size, image_quality, style, batch_size, seed, unique_id=None):

        if unique_id:
            self.trbl.reset('Dall-e Image, Node #' + unique_id)
        else:
            self.trbl.reset('Dall-e Image Node')

        _help = self.help_data.dalle_help
        self.ctx.request = rqst.dall_e_request()
        kwargs = { "model": GPTmodel,
                "prompt": prompt,
                "image_size": image_size,
                "image_quality": image_quality,
                "style": style,
                "batch_size": batch_size
        }
        batched_images, revised_prompt = self.ctx.execute_request(**kwargs)

        return (batched_images, revised_prompt, _help, self.trbl.get_troubles())
    

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
    
    CATEGORY = "Plush/Utils"

    RETURN_TYPES = ("STRING","STRING","STRING")
    RETURN_NAMES = ("Image_info","help","troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = True

    def gogo(self, image, write_to_file, file_prefix, Min_prompt_len, Alpha_Char_Pct,Prompt_Filter_Term, unique_id=None):

        if unique_id:
            self.trbl.reset('Exif Wrangler, Node# ' + unique_id) #Clears all trouble logs before a new run and passes the name of process to head the log lisiing
        else:
            self.trbl.reset('Exif Wrangler')

        _help = self.help_data.exif_wrangler_help #Establishes access to help files
        output = "Unable to process request"
        fatal = False
        exiv_comment = {}

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

        output = self.j_mngr.prep_formatted_file(working_dict)

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
    "Plush-Exif Wrangler" :ImageInfoExtractor
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Enhancer": "Style Prompt",
    "AI Chooser": "AI_Chooser",
    "AdvPromptEnhancer": "Advanced Prompt Enhancer",
    "DalleImage": "OAI Dall_e Image",
    "ImageInfoExtractor": "Exif Wrangler"
}
