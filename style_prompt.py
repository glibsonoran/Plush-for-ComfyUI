
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
import openai
from openai import OpenAI
from .mng_json import json_manager, helpSgltn, TroubleSgltn
from . import api_requests as rqst


#pip install pillow
#pip install bytesio

#Enum for style_prompt user input modes
class InputMode(Enum):
    IMAGE_PROMPT = 1
    IMAGE_ONLY = 2
    PROMPT_ONLY = 3

class RequestMode(Enum):
    OPENAI = 1
    OPENSOURCE = 2
    OOBABOOGA = 3
    CLAUDE = 4

#Get information from the config.json file
class cFigSingleton:
    _instance = None

    class APIObject(Enum):
        OPENAI = 1
        OTHER = 2

    def __new__(cls): 
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._lm_client = None
            cls._lm_url = ""
            cls._lm_key = ""
            cls._lm_client_type = None
            cls._lm_models = None
            cls._written_url = ""
            cls.j_mngr = json_manager()
            cls._pyexiv2 = None
            cls._instance.get_file()

        return cls._instance
    
    
    def get_file(self):

        def get_gpt_models(key:str):
            openai.api_key = key
            #Get the model list 
            try:
                models = openai.models.list()  
            except Exception as e:
                self.j_mngr.log_events(f"OpenAI API key is invalid or missing, unable to generate list of models. Error: {e}",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
                return None

            return models

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
                       
        self._fig_gpt_models = get_gpt_models(self._fig_key)
   
    def get_chat_models(self, sort_it:bool=False, filter_str:str="")->list:
        CGPT_models = []
        if self._fig_gpt_models and self._fig_gpt_models.data:
            if filter_str:
                filter_str = str(filter_str).lower()

                for model in self._fig_gpt_models.data:  
                    if filter_str in str(model.id).lower():
                        CGPT_models.append(model.id)  
        
                if sort_it:
                    CGPT_models = sorted(CGPT_models)
        else:
            CGPT_models.append('gpt-4-0125-preview')

        return CGPT_models
    
    def get_opensource_models(self, sort_it:bool=False):
        """
        Models are on hold right now.  There are too many issues dealing with a server that may
        or may not be running, excessively long connection attempts, and ui control list/saved value mismatches.
        """
        lm_models = []
        if self._lm_models and self._lm_models.data:

            for model in self._lm_models.data:
                lm_models.append(model.id)

            if sort_it:
                lm_models = sorted(lm_models)
        else:
            lm_models.append("Currently loaded model")
        return lm_models
    
    
    def set_llm_client(self, url:str, client_type:APIObject=APIObject.OPENAI, request_type:RequestMode=RequestMode.OPENSOURCE)-> bool:
        
        if not self.is_lm_server_up() or not url:
            self._lm_client = None
            self._lm_url = url
            self._lm_models = None
            self.j_mngr.log_events("Local LLM server is not running; aborting client setup.",
                          TroubleSgltn.Severity.WARNING,
                          True)
            return False
        
        lm_object = None
        key = "No key necessary" #Default value used in LLM front-ends that don't require a key
        #Use the requested API
        if client_type == self.APIObject.OPENAI: 
            lm_object = OpenAI 
        elif client_type == self.APIObject.OTHER: #Potential other Library/API objects
            pass 

        if request_type in (RequestMode.OOBABOOGA, RequestMode.OPENSOURCE):
            if not self._lm_key:
                self.j_mngr.log_events("Setting Openai client with URL, no key.",
                    is_trouble=True)
            else:
                key = self._lm_key
                self.j_mngr.log_events("Setting Openai client with URL and key.",
                    is_trouble=True)
        elif request_type == RequestMode.CLAUDE: # Will use CLAUDE_KEY env var
            pass
        
        try:
            lm_client = lm_object(base_url=url, api_key=key) 
            self._lm_url = url
            self._lm_client_type = client_type
            self._lm_client = lm_client
        except Exception as e:
            self.j_mngr.log_events(f"Unable to create LLM client object using URL. Unable to communicate with LLM: {e}",
                            TroubleSgltn.Severity.ERROR,
                            True)
            return False
        """
        Currently not active due to difficulties with connections times when server is not started.
        if self.is_lm_server_up():  
            try:
                self._lm_models = lm_client.models.list()
            except Exception as e:
                self.j_mngr.log_events(f"Unable to connect to retreive model list from Open Source LLM Manager: {e}.  Make sure your LLM Manager app is running and the url is correct.",
                                TroubleSgltn.Severity.ERROR,
                                True)
        else:
            self.j_mngr.log_events("Unable to retrieve open source models, LLM server is not running",
                              TroubleSgltn.Severity.WARNING,
                              True)
        """
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
    def lm_url(self, url: str, request_mode:RequestMode=RequestMode.OPENSOURCE):
        if url != self._lm_url or not self._lm_client:  # Check if the new URL is different to avoid unnecessary operations
            self._lm_url = url
            # Reset client and models only if a new URL is provided
            self._lm_client = None
            #self._lm_models = []
            if url:  # If the new URL is not empty, update the client
                self.set_llm_client(url, self.APIObject.OPENAI, request_mode)
    
    @property
    def lm_client_type(self)-> APIObject | None:
        return self._lm_client_type
    
    def is_lm_server_up(self):
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
    def key(self)-> str:
        return self._fig_key
    
    @property
    def lm_key(self)-> str:
        return self._lm_key

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
    def openaiClient(self)-> Optional[object]:
        if self._fig_key:
            return self.figOAIClient
        else:
            return None


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
                "GPTmodel": (cFig.get_chat_models(True, 'gpt'),{"default": "gpt-4-0125-preview"} ),
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
                "prompt": ("STRING",{"multiline": True, "default": ""}),          
                "image" : ("IMAGE", {"default": None})
            }
        } 

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING","STRING")
    RETURN_NAMES = ("CGPTprompt", "CGPTinstruction","Style Info", "Help","troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush/Prompt"
 

    def gogo(self, GPTmodel, creative_latitude, tokens, style, artist, prompt_style, max_elements, style_info, prompt="", image=None, unique_id=None):

        if unique_id:
            self.trbl.reset('Style Prompt, Node #'+unique_id)
        else:
            self.trbl.reset('Style Prompt')

        _help = self.help_data.style_prompt_help
        CGPT_prompt = ""
        instruction = ""
        CGPT_styleInfo = ""

        # unconnected UI elements get passed in as the string "undefined" by ComfyUI
        image = self.undefined_to_none(image)
        prompt = self.undefined_to_none(prompt)
        #Translate any friendly model names
        GPTmodel = self.translateModelName(GPTmodel)       

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

        self.ctx.request = rqst.oai_object_request()

        if style_info:
            self.trbl.set_process_header("Art Style Info:")
            #User has request information about the art style.  GPT will provide it
            sty_prompt = "Give an 150 word backgrounder on the art style: {}.  Starting with describing what it is, include information about its history and which artists represent the style."
            sty_prompt = sty_prompt.format(style)

            kwargs = { "model": GPTmodel,
                "creative_latitude": creative_latitude,
                "tokens": tokens,
                "prompt": sty_prompt,
                "request_type": RequestMode.OPENAI
            }    
            CGPT_styleInfo = self.ctx.execute_request(**kwargs)
            self.trbl.pop_header()

        kwargs = { "model": GPTmodel,
            "creative_latitude": creative_latitude,
            "tokens": tokens,
            "prompt": prompt,
            "instruction": instruction,
            "image": image,
            "request_type": RequestMode.OPENAI
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

    @staticmethod
    def join_punct(text: str, end_char:str=""):
        #Utility to create proper ending puctuation for text joins and ends
        text = text.rstrip()
        if text.endswith((':', ';', '-', ',', '.')):
            return ' '  # Add a space if special punctuation ends the text
        else:
            if end_char:
                return end_char + ' '  # Add the passed end character and space otherwise
            else:
                return ' ' #User's don't want the app to add a comma to their tags

    @staticmethod
    def enhanced_text_placement(generated_text:str, user_input:str, delimiter:str="|"):

        """
        Enhances text placement within a the generated text block based on user input, specified delimiters and markup.
        Text prefaced with "*" is placed at the beginning of the block, while text prefaced with "**"
        is placed in the middle, immediately following a period or comma. Unmarked text is added to the end
        of the block by default. This feature requires users to delimit each text segment intended for
        placement with a specified delimiter (default is a pipe '|'), regardless of its intended position.

        Args:
            generated_text (str): The existing text block generated by the LLM, where new text will be integrated.
            user_input (str): Delimited text input from the user containing potential markers for special placement.
            delimiter (str): The character used to separate different sections of the user input for specific placement.

        Returns:
            str: The updated text block with user input integrated at specified positions.

        """
            
            # Initialize default sections
        if not "*" in user_input:
            return generated_text + AdvPromptEnhancer.join_punct(generated_text,'.') + user_input.strip(', ')

        end_text, beginning_text, middle_text = '', '', ''
        
        # Split the input by pipe, which separates different sections
        sections = user_input.split(delimiter)
        
        for section in sections:
            # Strip leading/trailing whitespace and check for placement indicators
            section = section.strip(' \n')
            section_punct = AdvPromptEnhancer.join_punct(section)
            if section.startswith('**'):
                middle_text += section.lstrip('*') + section_punct
            elif section.startswith('*'):
                beginning_text += section.lstrip('*') + section_punct
            else:
                end_text += section + section_punct

        mid_punct = AdvPromptEnhancer.join_punct(middle_text)
        end_punct = AdvPromptEnhancer.join_punct(end_text)
        begin_punct = AdvPromptEnhancer.join_punct(beginning_text)

        # Integrate middle text based on punctuation logic in the generated_text
        commas = generated_text.count(',')
        periods = generated_text.count('.')
        punct_count = max(commas, periods)
        
        if middle_text:
            
            if punct_count == 0:
                end_text = end_punct.join([end_text, middle_text]) if end_text else middle_text
            elif punct_count <= 2:
                # Look for the first instance of either a comma or a period
                first_punctuation_index = len(generated_text)  # Default to the end of the string
                for char in [',', '.']:  # Check for both commas and periods
                    index = generated_text.find(char)
                    if 0 <= index < first_punctuation_index:  # Check if this punctuation occurs earlier
                        first_punctuation_index = index

                # Insert the middle text after the first punctuation found, if any
                if first_punctuation_index < len(generated_text):
                    insert_index = first_punctuation_index + 1  # Position right after the punctuation
                    generated_text = generated_text[:insert_index] + ' ' + middle_text + mid_punct + generated_text[insert_index:]
            else:
                # Insert at the midpoint punctuation
                target = punct_count // 2
                count = 0
                insert_index = 0
                for i, char in enumerate(generated_text):
                    if char in ',.':
                        count += 1
                        if count == target:
                            insert_index = i + 2  # After the punctuation and space
                            break
                generated_text = generated_text[:insert_index] + middle_text + mid_punct + generated_text[insert_index:]
        
        # Integrate beginning and end text
        if beginning_text:
            generated_text = beginning_text + begin_punct + generated_text
        if end_text:
            generated_text += AdvPromptEnhancer.join_punct(generated_text,'.') + end_text
        
        return generated_text.strip(', ')  # Ensure no leading or trailing commas 
    

    @classmethod
    def INPUT_TYPES(cls):
        cFig = cFigSingleton()
        #open source models are too problematic to implement right now in an environment where you 
        #can't be sure if the local host server (open source) will be running, and where you can't
        #refresh the ui after the initial load.
        return {
            "required": {
                
                "LLM": (["ChatGPT", "Other LLM via URL", "OpenAI compatible http POST", "Oobabooga API-URL"], {"default": "ChatGPT"}),
                "GPTmodel": (cFig.get_chat_models(True,'gpt'),{"default": "gpt-4-0125-preview"}),
                "creative_latitude" : ("FLOAT", {"max": 1.901, "min": 0.1, "step": 0.1, "display": "number", "round": 0.1, "default": 0.7}),                  
                "tokens" : ("INT", {"max": 8000, "min": 20, "step": 10, "default": 500, "display": "number"}), 
                "seed": ("INT", {"default": 9, "min": 0, "max": 0xffffffffffffffff}),
                "examples_delimiter":(["Pipe |", "Two newlines", "Two colons ::"], {"default": "Two newlines"}),
                "enhanced_tag_placement": ("BOOLEAN", {"default": False}),
                #"open_source_model": (cFig.get_opensource_models(True),),
                "LLM_URL": ("STRING",{"default": cFig.lm_url}),
                "Misc_tags": ("STRING", {"multiline": True})               
                         
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

    def gogo(self, LLM, GPTmodel, creative_latitude, tokens, seed, examples_delimiter, enhanced_tag_placement, LLM_URL:str="", Misc_tags:str="", Instruction:str="", Prompt:str = "", Examples:str ="",image=None, unique_id=None):

        if unique_id:
            self.trbl.reset("Advanced Prompt Enhancer, Node #"+unique_id)
        else:
            self.trbl.reset("Advanced Prompt Enhancer")

        _help = self.help_data.adv_prompt_help

        def misc_tags(output:str, enhanced_tag_placement)-> str:
            if enhanced_tag_placement:
                output = AdvPromptEnhancer.enhanced_text_placement(output, Misc_tags)               
            else:
                output  += AdvPromptEnhancer.join_punct(output, '.') + Misc_tags

            return output


        # set the value of unconnected inputs to None
        Instruction = Enhancer.undefined_to_none(Instruction)
        Prompt = Enhancer.undefined_to_none(Prompt)
        Examples = Enhancer.undefined_to_none(Examples)
        LLM_URL = Enhancer.undefined_to_none(LLM_URL)
        Misc_tags = Enhancer.undefined_to_none(Misc_tags)   
        image = Enhancer.undefined_to_none(image)

        llm_result = "Unable to process request.  Make sure the local Open Source Server is running."

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

        kwargs = { "model": GPTmodel,
                "creative_latitude": creative_latitude,
                "tokens": tokens,
                "prompt": Prompt,
                "instruction": Instruction,
                "url": LLM_URL,
                "image": image,
                "example_list": example_list,
        }

        if  LLM == 'Other LLM via URL' : 
 
            if not LLM_URL:
                self.j_mngr.log_events("'Other LLM via URL' specified, but no URL provided or URL is invalid. Enter a valid URL",
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
        
            kwargs['model'] = "LLM"
            kwargs['request_type'] = RequestMode.OPENSOURCE

            llm_result = ""
            self.ctx.request = rqst.oai_object_request( )

            llm_result = self.ctx.execute_request(**kwargs)

            if Misc_tags:
                llm_result = misc_tags(llm_result, enhanced_tag_placement)

            return(llm_result, _help, self.trbl.get_troubles())
        
        elif LLM == "OpenAI compatible http POST":
            if not LLM_URL:
                self.j_mngr.log_events("'OpenAI compatible http POST' specified, but no URL provided or URL is invalid. Enter a valid URL",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
                return(llm_result, _help, self.trbl.get_troubles())  
            
            self.ctx.request = rqst.oai_web_request()

            kwargs['model'] = "POST" #Value from a 'model' field would go here
            kwargs['request_type'] = RequestMode.OPENSOURCE

            llm_result = self.ctx.execute_request(**kwargs)

            if Misc_tags:
                llm_result = misc_tags(llm_result, enhanced_tag_placement)

            return(llm_result, _help, self.trbl.get_troubles())            

        #Oobabooga via POST
        elif LLM == "Oobabooga API-URL":
            if not LLM_URL:
                self.j_mngr.log_events("'Oobabooga API-URL' specified, but no URL provided or URL is invalid. Enter a valid URL",
                                    TroubleSgltn.Severity.WARNING,
                                    True)
                return(llm_result, _help, self.trbl.get_troubles())  

            self.ctx.request = rqst.ooba_web_request()

            kwargs['model'] = "oobabooga"
            kwargs['request_type'] = RequestMode.OOBABOOGA

            llm_result = self.ctx.execute_request(**kwargs)

            if Misc_tags:
                llm_result = misc_tags(llm_result, enhanced_tag_placement)

            return(llm_result, _help, self.trbl.get_troubles())            

        #OpenAI ChatGPT request
        self.ctx.request = rqst.oai_object_request( )

        kwargs['request_type'] = RequestMode.OPENAI

        llm_result = self.ctx.execute_request(**kwargs)
       
        if Misc_tags:
            llm_result = misc_tags(llm_result, enhanced_tag_placement)
       
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
    "AdvPromptEnhancer": AdvPromptEnhancer,
    "DalleImage": DalleImage,
    "Plush-Exif Wrangler" :ImageInfoExtractor
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Enhancer": "Style Prompt",
    "AdvPromptEnhancer": "Advanced Prompt Enhancer",
    "DalleImage": "OAI Dall_e Image",
    "ImageInfoExtractor": "Exif Wrangler"
}
