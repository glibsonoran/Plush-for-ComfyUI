from abc import ABC, abstractmethod
import torch
import time
import re
import requests
from urllib.parse import urlparse, urlunparse
import openai
import anthropic
from .mng_json import json_manager, TroubleSgltn
from .fetch_models import RequestMode
from .utils import ImageUtils 

class ImportedSgltn:
    """
    This class is temporary to prevent circular imports
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImportedSgltn, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized: #pylint: disable=access-member-before-definition
            self._initialized = True
            self._cfig = None
            self._dalle = None
            self._request_mode = None
            self.get_imports()

    def get_imports(self):
        # Guard against re-importing if already done
        if self._cfig is None or self._dalle is None:
            from .style_prompt import cFigSingleton, DalleImage
            self._cfig = cFigSingleton
            self._dalle = DalleImage
            self._request_mode = RequestMode


    @property
    def cfig(self):

        if self._cfig is None:
            self.get_imports()
        return self._cfig()

    @property
    def dalle(self):

        if self._dalle is None:
            self.get_imports()
        return self._dalle()
       

#Begin Strategy Pattern
class Request(ABC):

    def __init__(self):
        self.imps = ImportedSgltn()
        self.utils = request_utils()
        self.cFig = self.imps.cfig
        self.mode = RequestMode
        self.dalle = self.imps.dalle
        self.j_mngr = json_manager()

    @abstractmethod
    def request_completion(self, **kwargs) -> None:
        pass

class oai_object_request(Request): #Concrete class

 
    def request_completion(self, **kwargs):
        
        GPTmodel = kwargs.get('model')
        creative_latitude = kwargs.get('creative_latitude', 0.7)
        tokens = kwargs.get('tokens',500)
        prompt = kwargs.get('prompt', "")
        instruction = kwargs.get('instruction', "")
        file = kwargs.get('file',"")
        image = kwargs.get('image', None)
        example_list = kwargs.get('example_list', [])
        add_params = kwargs.get('add_params', None)

        request_type = self.cFig.lm_request_mode
        response = None
        CGPT_response = ""
        file += file.strip()
        client = None

        if request_type == self.mode.OPENSOURCE or request_type == self.mode.OLLAMA:
            if self.cFig.lm_url:
                self.j_mngr.log_events("Setting client to OpenAI Open Source LLM object",
                                    is_trouble=True)
                
                client = self.cFig.lm_client

                #Force the correct url path
                corrected_url = self.utils.validate_and_correct_url(self.cFig.lm_url,'/v1')
                client.base_url = corrected_url
            else:
                self.j_mngr.log_events("Open Source api object is not ready for use, no URL provided. Aborting",
                                  TroubleSgltn.Severity.WARNING,
                                    is_trouble=True)
                return CGPT_response
            
        if request_type == self.mode.GROQ:
            if self.cFig.lm_url:
                self.j_mngr.log_events("Setting client to OpenAI Groq LLM object",
                                    is_trouble=True)
                client = self.cFig.lm_client
            else:
                self.j_mngr.log_events("Groq OpenAI api object is not ready for use, no URL provided. Aborting",
                                  TroubleSgltn.Severity.WARNING,
                                    is_trouble=True)

        if request_type == self.mode.OPENAI:
            if self.cFig.key:
                self.j_mngr.log_events("Setting client to OpenAI ChatGPT object",
                                    is_trouble=True)
                client = self.cFig.openaiClient
            else:
                CGPT_response = "Invalid or missing OpenAI API key.  Keys must be stored in an environment variable (see: ReadMe). ChatGPT request aborted"
                self.j_mngr.log_events("Invalid or missing OpenAI API key.  Keys must be stored in an environment variable (see: ReadMe). ChatGPT request aborted",
                                  TroubleSgltn.Severity.WARNING,
                                    is_trouble=True)
                return CGPT_response



        if not client:
            if request_type ==  self.mode.OPENAI:
                self.j_mngr.log_events("Invalid or missing OpenAI API key.  Keys must be stored in an environment variable (see: ReadMe). ChatGPT request aborted",
                                        TroubleSgltn.Severity.ERROR,
                                        True)
                CGPT_response = "Invalid or missing OpenAI API key.  Keys must be stored in an environment variable (see: ReadMe). ChatGPT request aborted"
                
            else:
                self.j_mngr.log_events("LLM client not set.  Make sure local Server is running if using a local LLM front-end",
                                        TroubleSgltn.Severity.ERROR,
                                        True)
                CGPT_response = "Unable to process request, make sure local server is running"                
            return CGPT_response

        #there's an image
        if image:
            # Use the user's selected vision model if it's what was chosen,
            #otherwise use the last vision model in the list
            #If the user is using a local LLM they're on their own to make
            #the right model selection for handling an image

            if isinstance(image, torch.Tensor):  #just to be sure
                image = self.dalle.tensor_to_base64(image)
                
            if not isinstance(image,str):
                image = None
                self.j_mngr.log_events("Image file is invalid.  Image will be disregarded in the generated output.",
                                  TroubleSgltn.Severity.WARNING,
                                  True)

        messages = []

        #Use basic data structure if there is no image
        if not image:
            messages = self.utils.build_data_basic(prompt, example_list, instruction)
        else:
            messages = self.utils.build_data_multi(prompt, instruction, example_list, image)
            
        if not prompt and not image and not instruction and not example_list:
            # User has provided no prompt, file or image
            response = "Photograph of an stained empty box with 'NOTHING' printed on its side in bold letters, small flying moths, dingy, gloomy, dim light rundown warehouse"
            self.j_mngr.log_events("No instruction and no prompt were provided, the node was only able to provide a 'Box of Nothing'",
                              TroubleSgltn.Severity.WARNING,
                              True)
            return response

        params = {
        "model": GPTmodel,
        "messages": messages,
        "temperature": creative_latitude,
        "max_tokens": tokens
        }

        # Add the parameter if it exists 
        if add_params:
            add_keys =['param','value']
            self.j_mngr.append_params(params, add_params, add_keys)

        try:
            response = client.chat.completions.create(**params)

        except openai.APIConnectionError as e: # from httpx.
            self.j_mngr.log_events(f"Server connection error: {e.__cause__}",                                   
                                    TroubleSgltn.Severity.ERROR,
                                    True)
            if request_type == self.mode.OPENSOURCE:
                self.j_mngr.log_events(f"Local server is not responding to the URL: {self.cFig.lm_url}.  Make sure your LLM Manager/Front-end app is running and its local server is live.",
                                TroubleSgltn.Severity.WARNING,
                                True)
        except openai.RateLimitError as e:
            error_message = e.body.get('message', "No error message provided") if isinstance(e.body, dict) else str(e.body or "No error message provided")
            self.j_mngr.log_events(f"Server STATUS error {e.status_code}: {error_message}.",
                           TroubleSgltn.Severity.ERROR,
                           True)
        except openai.APIStatusError as e:
            error_message = e.body.get('message', "No error message provided") if isinstance(e.body, dict) else str(e.body or "No error message provided")
            self.j_mngr.log_events(f"Server STATUS error {e.status_code}: {error_message}.",
                           TroubleSgltn.Severity.ERROR,
                           True)
        except Exception as e:
            self.j_mngr.log_events(f"An unexpected server error occurred.: {e}",
                                TroubleSgltn.Severity.ERROR,
                                    True)


        if response and response.choices and 'error' not in response:
            rpt_model = ""
            rpt_usage = ""
            try:
                rpt_model = response.model
                rpt_usage = response.usage
            except Exception as e:
                self.j_mngr.log_events(f"Unable to report some completion information, error: {e}",
                                  TroubleSgltn.Severity.INFO,
                                  True)
            if rpt_model:    
                self.j_mngr.log_events(f"Using LLM: {rpt_model}",                                  
                               is_trouble=True)
            if rpt_usage:
                self.j_mngr.log_events(f"Tokens Used: {rpt_usage}",
                                  TroubleSgltn.Severity.INFO,
                                  True)
            CGPT_response = response.choices[0].message.content
            CGPT_response = self.utils.clean_response_text(CGPT_response)
        else:
            err_mess = getattr(response, 'error', "Error message missing")

            CGPT_response = "Server was unable to process the request"
            self.j_mngr.log_events(f"Server was unable to process this request. Error: {err_mess}",
                                TroubleSgltn.Severity.ERROR,
                                True)
        return CGPT_response

class oai_web_request(Request):


    def request_completion(self, **kwargs):

        """
        Uses the incoming arguments to construct a JSON that contains the request for an LLM response.
        Accesses an LLM via an http POST.
        Sends the request via http. Handles the OpenAI return object and extacts the model and the response from it.

        Args:
            GPTmodel (str):  The ChatGPT model to use in processing the request. Alternately this serves as a flag that the function will processing open source LLM data (GPTmodel = "LLM")
            creative_latitude (float): A number setting the 'temperature' of the LLM
            tokens (int): A number indicating the max number of tokens used to process the request and response
            url (str): The url for the server the information is being sent to
            request_:type (Enum): Specifies whether the function will be using a ChatGPT configured api object or an third party/url configured api object.
            prompt (str): The users' request to action by the LLM
            instruction (str): Text describing the conditions and specific requirements of the return value
            image (b64 JSON/str): An image to be evaluated by the LLM in the context of the instruction

        Return:
            A string consisting of the LLM's response to the instruction and prompt in the context of any image and/or file
        """
        GPTmodel = kwargs.get('model', "")
        creative_latitude = kwargs.get('creative_latitude', 0.7)
        url = kwargs.get('url',None)
        tokens = kwargs.get('tokens', 500)
        image = kwargs.get('image', None)
        prompt = kwargs.get('prompt', None)
        instruction = kwargs.get('instruction', "")
        example_list = kwargs.get('example_list', [])
        add_params = kwargs.get('add_params', None)



        request_type = self.cFig.lm_request_mode

        response = None
        CGPT_response = ""    

        self.cFig.lm_url = url
        if not self.cFig.is_lm_server_up:
            self.j_mngr.log_events("Local or remote server is not responding, may be unable to send data.",
                                   TroubleSgltn.Severity.WARNING,
                                   True)

        #if there's an image here
        if image and request_type == self.mode.OSSIMPLE:
            self.j_mngr.log_events("The AI Service using 'Simplfied Data' can't process an image. The image will be disregarded in generated output.",
                                    TroubleSgltn.Severity.INFO,
                                    True)
            image = None

        if image:
            #The user is on their own to make
            #the right model selection for handling an image

            if isinstance(image, torch.Tensor):  #just to be sure
                image = self.dalle.tensor_to_base64(image)
                
            if not isinstance(image,str):
                image = None
                self.j_mngr.log_events("Image file is invalid.  Image will be disregarded in the generated output.",
                                  TroubleSgltn.Severity.WARNING,
                                  True)

        key = ""
        if request_type == self.mode.OPENAI:
            key =  self.cFig.key
        elif request_type == self.mode.OPENSOURCE or request_type == self.mode.LMSTUDIO:
            key = self.cFig.lm_key
        elif request_type == self.mode.GROQ:
            key = self.cFig.groq_key
        else:
            self.j_mngr.log_events("No LLM key value found",
                                   TroubleSgltn.Severity.WARNING,
                                   True)
                
        headers = self.utils.build_web_header(key) 

        if request_type == self.mode.OSSIMPLE or not image:
            messages = self.utils.build_data_basic(prompt, example_list, instruction) #Some apps can't handle an embedded list of role:user dicts
            self.j_mngr.log_events("Using Basic data structure",
                                   TroubleSgltn.Severity.INFO,
                                   True)
        else:
            messages = self.utils.build_data_multi(prompt,instruction,example_list, image)
            self.j_mngr.log_events("Using Complex data structure",
                                   TroubleSgltn.Severity.INFO,
                                   True)

        params = {
        "model": GPTmodel,
        "messages": messages,
        "temperature": creative_latitude,
        "max_tokens": tokens
        }   

        if add_params:
            add_keys =['param','value']
            self.j_mngr.append_params(params, add_params, add_keys)


        post_success = False
        response_json = ""
        #payload = {**params}
        try:
            response = requests.post(url, headers=headers, json=params, timeout=(12,120))
            
            if response.status_code in range(200, 300):
                response_json = response.json()
                if response_json and not 'error' in response_json:
                    CGPT_response = self.utils.clean_response_text(response_json['choices'][0]['message']['content'] )
                    post_success = True
                else:
                    error_message = response_json.get('error', 'Unknown error')
                    self.j_mngr.log_events(f"Server was unable to process the response. Error: {error_message}",
                                    TroubleSgltn.Severity.ERROR,
                                    True)
            else:
                CGPT_response = 'Server was unable to process this request'
                self.j_mngr.log_events(f"Server was unable to process the request.  Status: {response.status_code}: {response.text}",
                                    TroubleSgltn.Severity.ERROR,
                                    True)
                
        except Exception as e:
            self.j_mngr.log_events(f"Unable to send data to server.  Error: {e}",
                              TroubleSgltn.Severity.ERROR,
                              True)
        if post_success:   
            try:
                rpt_model = response_json['model']
                rpt_usage = response_json['usage']
                if rpt_model:    
                    self.j_mngr.log_events(f"Using LLM: {rpt_model}",                                  
                                is_trouble=True)
                if rpt_usage:
                    self.j_mngr.log_events(f"Tokens Used: {rpt_usage}",
                                        is_trouble=True)

            except Exception as e:
                self.j_mngr.log_events(f"Unable to report some completion information: model, usage.  Error: {e}",
                                    TroubleSgltn.Severity.INFO,
                                    True)    

        return CGPT_response


class ooba_web_request(Request):


    def request_completion(self, **kwargs):

        """
        Accesses an OpenAI API client and uses the incoming arguments to construct a JSON that contains the request for an LLM response.
        Sends the request via the client. Handles the OpenAI return object and extacts the model and the response from it.

        Args:
            GPTmodel (str):  The ChatGPT model to use in processing the request. Alternately this serves as a flag that the function will processing open source LLM data (GPTmodel = "LLM")
            creative_latitude (float): A number setting the 'temperature' of the LLM
            tokens (int): A number indicating the max number of tokens used to process the request and response
            url (str): The url for the server the information is being sent to
            request_:type (Enum): Specifies whether the function will be using a ChatGPT configured api object or an third party/url configured api object.
            prompt (str): The users' request to action by the LLM
            instruction (str): Text describing the conditions and specific requirements of the return value
            image (b64 JSON/str): An image to be evaluated by the LLM in the context of the instruction

        Return:
            A string consisting of the LLM's response to the instruction and prompt in the context of any image and/or file
        """
        GPTmodel = kwargs.get('model', "")
        creative_latitude = kwargs.get('creative_latitude', 0.7)
        url = kwargs.get('url',None)
        tokens = kwargs.get('tokens', 500)
        image = kwargs.get('image', None)
        prompt = kwargs.get('prompt', None)
        instruction = kwargs.get('instruction', "")
        example_list = kwargs.get('example_list', [])
        request_type = self.cFig.lm_request_mode
        add_params = kwargs.get('add_params', None)


        response = None
        CGPT_response = ""    

        url = self.utils.validate_and_correct_url(url) #validate v1/chat/completions path

        self.cFig.lm_url = url
        if not self.cFig.is_lm_server_up:
            self.j_mngr.log_events("Local server is not responding, may be unable to send data.",
                                   TroubleSgltn.Severity.WARNING,
                                   True)

        #image code is here, but right now none of the tested LLM front ends can handle them 
        #when using an http POST
        if image:
            image = None
            self.j_mngr.log_events('Images not supported in this mode at this time.  Image not transmitted',
                              TroubleSgltn.Severity.WARNING,
                              True)   

        key = ""
        if request_type == self.mode.OPENAI:
            key =  self.cFig.key
        else:
            key = self.cFig.lm_key
                
        headers = self.utils.build_web_header(key) 

        #messages = self.utils.build_data_basic(prompt, example_list, instruction)
        
        messages = self.utils.build_data_ooba(prompt, example_list, instruction)

        if request_type == self.mode.OOBABOOGA:
            self.j_mngr.log_events(f"Processing Oobabooga http: POST request with url: {url}",
                              is_trouble=True)
            params = {
            "model": GPTmodel,
            "messages": messages,
            "temperature": creative_latitude,
            "max_tokens": tokens,
            "user_bio": "",
            "user_name": ""
            }
        else:
            params = {
            "model": GPTmodel,
            "messages": messages,
            "temperature": creative_latitude,
            "max_tokens": tokens
            }    

        # Add the parameter if it exists 
        if add_params:
            add_keys =['param','value']
            self.j_mngr.append_params(params, add_params, add_keys)

        post_success = False
        response_json = ""
        #payload = {**params}
        try:
            response = requests.post(url, headers=headers, json=params, timeout=(12,120))
            
            if response.status_code in range(200, 300):
                response_json = response.json()
                if response_json and not 'error' in response_json:
                    CGPT_response = self.utils.clean_response_text(response_json['choices'][0]['message']['content'] )
                    post_success = True
                else:
                    error_message = response_json.get('error', 'Unknown error')
                    self.j_mngr.log_events(f"Server was unable to process the response. Error: {error_message}",
                                    TroubleSgltn.Severity.ERROR,
                                    True)
            else:
                CGPT_response = 'Server was unable to process this request'
                self.j_mngr.log_events(f"Server was unable to process the request.  Status: {response.status_code}: {response.text}",
                                    TroubleSgltn.Severity.ERROR,
                                    True)
                
        except Exception as e:
            self.j_mngr.log_events(f"Unable to send data to server.  Error: {e}",
                              TroubleSgltn.Severity.ERROR,
                              True)
        if post_success:   
            try:
                rpt_model = response_json['model']
                rpt_usage = response_json['usage']
                if rpt_model:    
                    self.j_mngr.log_events(f"Using LLM: {rpt_model}",                                  
                                is_trouble=True)
                if rpt_usage:
                    self.j_mngr.log_events(f"Tokens Used: {rpt_usage}",
                                        is_trouble=True)

            except Exception as e:
                self.j_mngr.log_events(f"Unable to report some completion information: model, usage.  Error: {e}",
                                    TroubleSgltn.Severity.INFO,
                                    True)    

        return CGPT_response
    

class claude_request(Request):

    def request_completion(self, **kwargs):
        
        claude_model = kwargs.get('model')
        creative_latitude = kwargs.get('creative_latitude', 0.7)
        tokens = kwargs.get('tokens',500)
        prompt = kwargs.get('prompt', "")
        instruction = kwargs.get('instruction', "")
        file = kwargs.get('file',"")
        image = kwargs.get('image', None)
        example_list = kwargs.get('example_list', [])
        add_params = kwargs.get('add_params', None)


        request_type = self.cFig.lm_request_mode
        
        response = None
        claude_response = ""
        file += file.strip()
        client = None

        if request_type == self.mode.CLAUDE:
            client = self.cFig.anthropic_client


        if not client:
            if request_type ==  self.mode.CLAUDE:
                self.j_mngr.log_events("Invalid or missing anthropic API key (Claude).  Keys must be stored in an environment variable (see: ReadMe). Claude request aborted",
                                        TroubleSgltn.Severity.ERROR,
                                        True)
                claude_response = "Invalid or missing anthropic API key.  Keys must be stored in an environment variable (see: ReadMe). Claude request aborted"           
            return claude_response

        #there's an image
        if image:
            # Use the user's selected vision model if it's what was chosen,
            #otherwise use the last vision model in the list
            #If the user is using a local LLM they're on their own to make
            #the right model selection for handling an image

            if isinstance(image, torch.Tensor):  #just to be sure
                image = self.dalle.tensor_to_base64(image)
                
            if not isinstance(image,str):
                image = None
                self.j_mngr.log_events("Image file is invalid.  Image will be disregarded in the generated output.",
                                  TroubleSgltn.Severity.WARNING,
                                  True)

        messages = []

        messages = self.utils.build_data_claude(prompt, example_list, image)
            
        if not prompt and not image and not instruction and not example_list:
            # User has provided no prompt, file or image
            claude_response = "Photograph of an stained empty box with 'NOTHING' printed on its side in bold letters, small flying moths, dingy, gloomy, dim light rundown warehouse"
            self.j_mngr.log_events("No instruction and no prompt were provided, the node was only able to provide a 'Box of Nothing'",
                              TroubleSgltn.Severity.WARNING,
                              True)
            return claude_response

        params = {
        "model": claude_model,
        "messages": messages,
        "temperature": creative_latitude,
        "system": instruction,
        "max_tokens": tokens
        }

        # Add the parameter if it exists 
        if add_params:
            add_keys =['param','value']
            self.j_mngr.append_params(params, add_params, add_keys)   

        try:
            response = client.messages.create(**params)

        except anthropic.AuthenticationError as e:
            self.j_mngr.log_events(f"Authentication error: {request_utils.parse_anthropic_error(e)}", 
                                   TroubleSgltn.Severity.ERROR, 
                                   True)
        except anthropic.PermissionDeniedError as e:
            self.j_mngr.log_events(f"Permission denied error: {request_utils.parse_anthropic_error(e)}", 
                                   TroubleSgltn.Severity.ERROR, 
                                   True)
        except anthropic.NotFoundError as e:
            self.j_mngr.log_events(f"Not found error: {request_utils.parse_anthropic_error(e)}", 
                                   TroubleSgltn.Severity.ERROR, 
                                   True)
        except anthropic.RateLimitError as e:
            self.j_mngr.log_events(f"Rate limit exceeded error: {request_utils.parse_anthropic_error(e)}", 
                                   TroubleSgltn.Severity.WARNING, 
                                   True)
        except anthropic.BadRequestError as e:
            self.j_mngr.log_events(f"Bad request error: {request_utils.parse_anthropic_error(e)}", 
                                   TroubleSgltn.Severity.ERROR, 
                                   True)
        except anthropic.InternalServerError as e:
            self.j_mngr.log_events(f"Internal server error: {request_utils.parse_anthropic_error(e)}", 
                                   TroubleSgltn.Severity.ERROR, 
                                   True)
        except Exception as e:
            self.j_mngr.log_events(f"Unexpected error: {request_utils.parse_anthropic_error(e)}", 
                                   TroubleSgltn.Severity.ERROR, 
                                   True)


        if response and 'error' not in response:
            rpt_model = ""
            try:
                rpt_model = response.model
                rpt_usage = response.usage

                if rpt_model:    
                    self.j_mngr.log_events(f"Using LLM: {rpt_model}",                                  
                                is_trouble=True)
                if rpt_usage:
                    self.j_mngr.log_events(f"Tokens Used: {rpt_usage}",
                                    TroubleSgltn.Severity.INFO,
                                    True)              
            except Exception as e:
                self.j_mngr.log_events(f"Unable to report some completion information, error: {e}",
                                  TroubleSgltn.Severity.INFO,
                                  True) 
            try:               
                claude_response = response.content[0].text
            except (IndexError, AttributeError):
                claude_response = "No data was returned"
                self.j_mngr.log_events("Claude response was not valid data",
                                       TroubleSgltn.Severity.WARNING,
                                       True)
            claude_response = self.utils.clean_response_text(claude_response)
        else:
            claude_response = "Server was unable to process the request"
            self.j_mngr.log_events('Server was unable to process this request.',
                                TroubleSgltn.Severity.ERROR,
                                True)
        return claude_response

class dall_e_request(Request):

    def __init__(self):
        super().__init__()  # Ensures common setup from Request
        self.trbl = TroubleSgltn() 
        self.iu = ImageUtils()

    def request_completion(self, **kwargs)->tuple[torch.Tensor, str]:

        GPTmodel = kwargs.get('model')
        prompt = kwargs.get('prompt')
        image_size = kwargs.get('image_size')    
        image_quality = kwargs.get('image_quality')
        style = kwargs.get('style')
        batch_size = kwargs.get('batch_size', 1)

        self.trbl.set_process_header('Dall-e Request')

        batched_images = torch.zeros(1, 1024, 1024, 3, dtype=torch.float32)
        revised_prompt = "Image and mask could not be created"  # Default prompt message
        
        if not self.cFig.openaiClient:
            self.j_mngr.log_events("OpenAI API key is missing or invalid.  Key must be stored in an enviroment variable (see ReadMe).  This node is not functional.",
                                   TroubleSgltn.Severity.WARNING,
                                   True)
            return(batched_images, revised_prompt)
                
        client = self.cFig.openaiClient 
        
        
        self.j_mngr.log_events(f"Talking to Dalle model: {GPTmodel}",
                               is_trouble=True)

        have_rev_prompt = False   
        images_list = []

        for _ in range(batch_size):
            try:

                response = client.images.generate(
                    model = GPTmodel,
                    prompt = prompt, 
                    size = image_size,
                    quality = image_quality,
                    style = style,
                    n=1,
                    response_format = "b64_json",
            )
 
            # Get the revised_prompt
                if response and not 'error' in response:
                    if not have_rev_prompt:
                        revised_prompt = response.data[0].revised_prompt
                        have_rev_prompt = True
                    #Convert the b64 json to a pytorch tensor
                    b64Json = response.data[0].b64_json
                    if b64Json:
                        png_image, _ = self.dalle.b64_to_tensor(b64Json)
                        images_list.append(png_image)
                    else:
                        self.j_mngr.log_events(f"Dalle-e could not process an image in your batch of: {batch_size} ",
                                            TroubleSgltn.Severity.WARNING,
                                            True)  
                    
                else:
                    self.j_mngr.log_events(f"Dalle-e could not process an image in your batch of: {batch_size} ",
                                        TroubleSgltn.Severity.WARNING,
                                        True)   
            except openai.APIConnectionError as e: 
                self.j_mngr.log_events(f"ChatGPT server connection error in an image in your batch of {batch_size} Error: {e.__cause__}",
                                        TroubleSgltn.Severity.ERROR,
                                        True)
            except openai.RateLimitError as e:
                self.j_mngr.log_events(f"ChatGPT RATE LIMIT error in an image in your batch of {batch_size} Error: {e}: {e.response}",
                                        TroubleSgltn.Severity.ERROR,
                                        True)
                time.sleep(0.5)
            except openai.APIStatusError as e:
                self.j_mngr.log_events(f"ChatGPT STATUS error in an image in your batch of {batch_size}; Error: {e.status_code}:{e.response}",
                                        TroubleSgltn.Severity.ERROR,
                                        True)
            except Exception as e:
                self.j_mngr.log_events(f"An unexpected error in an image in your batch of {batch_size}; Error:{e}",
                                        TroubleSgltn.Severity.ERROR,
                                        True)
                
                
        if images_list:
            count = len(images_list)
            self.j_mngr.log_events(f'{count} images were processed successfully in your batch of: {batch_size}',
                                   is_trouble=True)
            
            batched_images = torch.cat(images_list, dim=0)
        else:
            self.j_mngr.log_events(f'No images were processed in your batch of: {batch_size}',
                                   TroubleSgltn.Severity.WARNING,
                                   is_trouble=True)
        self.trbl.pop_header()
        return(batched_images, revised_prompt)
    
    def modify_image(self, client, model, image_bytes, prompt, image_size):
        """This is an unused stub to be used if Dall-e-3 ever implements image to image edits"""
        image_bytes.seek(0)  # Ensure the buffer is at the beginning

        response = client.images.edit(
        model=model,
        image=image_bytes,
        prompt=prompt,
        n=1,
        size=image_size,
        response_format = "b64_json"
        )

        return response

class request_context:
    def __init__(self)-> None:
        self._request = None
        self.j_mngr = json_manager()

    @property
    def request(self)-> Request:
        return self._request

    @request.setter
    def request(self, request:Request)-> None:
        self._request = request

    def execute_request(self, **kwargs):
        if self._request is not None:
            return self._request.request_completion(**kwargs)
        
        self.j_mngr.log_events("No request strategy object was set",
                               TroubleSgltn.Severity.ERROR,
                               True)
        return None
    
class request_utils:

    def __init__(self)-> None:
        self.j_mngr = json_manager()
        self.mode = RequestMode

    def build_data_multi(self, prompt:str, instruction:str="", examples:list=None, image:str=None):
        """
        Builds a list of message dicts, aggregating 'role:user' content into a list under 'content' key.
        - image: Base64-encoded string or None. If string, included as 'image_url' type content.
        - prompt: String to be included as 'text' type content under 'user' role.
        - examples: List of additional example dicts to be included.
        - instruction: Instruction string to be included under 'system' role.
        """
        messages = []
        user_role = {"role": "user", "content": None}
        user_content = []

        if instruction:
            messages.append({"role": "system", "content": instruction})         

        if examples:
            messages.extend(examples)

        if prompt:
            user_content.append({"type": "text", "text": prompt})

        processed_image = self.process_image(image)
        if processed_image:
            user_content.append(processed_image)
        

        if user_content:
            user_role['content'] = user_content   
            messages.append(user_role)

        return messages
    
    def build_data_basic(self, prompt:str, examples:list=None, instruction:str=""):
        """
        Builds a list of message dicts, presenting each 'role:user' item in its own dict.
        - prompt: String to be included as 'text' type content under 'user' role.
        - examples: List of additional example dicts to be included.
        - instruction: Instruction string to be included under 'system' role.
        """

        messages = []

        if instruction:
            messages.append({"role": "system", "content": instruction})

        if examples:
            messages.extend(examples)     
        
        if prompt:
            messages.append({"role": "user", "content": prompt})


        return messages
    
    def build_data_ooba(self, prompt:str, examples:list=None, instruction:str="")-> list:
        """
        Builds a list of message dicts, presenting each 'role:user' item in its own dict.
        Since Oobabooga's system message is broken it includes it in the prompt
        - prompt: String to be included as 'text' type content under 'user' role.
        - examples: List of additional example dicts to be included.
        - instruction: Instruction string to be included under 'system' role.
        """

        messages = []

        ooba_prompt = ""

        if instruction:
            ooba_prompt += f"INSTRUCTION: {instruction}\n\n"

        if prompt:
            ooba_prompt += f"PROMPT: {prompt}"


        if examples:
            messages.extend(examples)
        
        if ooba_prompt:
            messages.append({"role": "user", "content": ooba_prompt.strip()})

        return messages 
    
    def build_data_claude(self, prompt:str, examples:list=None, image:str=None)-> list:
        """
        Builds a list of message dicts, aggregating 'role:user' content into a list under 'content' key.
        - image: Base64-encoded string or None. If string, included as 'image_url' type content.
        - prompt: String to be included as 'text' type content under 'user' role.
        - examples: List of additional example dicts to be included.
        """
        messages = []
        user_role = {"role": "user", "content": None}
        user_content = []

        if examples:
            messages.extend(examples)


        processed_image = self.process_image(image,RequestMode.CLAUDE)
        if processed_image:
            user_content.append(processed_image)
        
        if prompt:
            user_content.append({"type": "text", "text": prompt})

        if user_content:     
            user_role['content'] = user_content    
            messages.append(user_role)

        return messages


    def process_image(self, image: str, request_type:RequestMode=RequestMode.OPENAI) :
        if not image:
            return None
        
        if isinstance(image, str):
            if request_type == self.mode.CLAUDE:
                return {
                "type": "image",
                "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image
                }
            }

            return {"type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                    }                  
                    }

        self.j_mngr.log_events("Image file is invalid.", TroubleSgltn.Severity.WARNING, True)
        return None
    
    def build_web_header(self, key:str=""):
        if key:
            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}" 
            }
        else:
            headers = {
                "Content-Type": "application/json"
            }    

        return headers    

    def validate_and_correct_url(self, user_url:str, required_path:str='/v1/chat/completions'):
        """
        Takes the user's url and make sure it has the correct path for the connection
        args:
            user_url (str): The url to be validated and corrected if necessary
            required_path (str):  The correct path
        return:            
            A string with either the original url if it was correct or the corrected url if it wasn't
        """
        corrected_url = ""
        parsed_url = urlparse(user_url)
                
        # Check if the path is the required_path
        if not parsed_url.path == required_path:
            corrected_url = urlunparse((parsed_url.scheme,
                                        parsed_url.netloc,
                                        required_path,
                                        '',
                                        '',
                                        ''))

        else:
            corrected_url = user_url
            
        self.j_mngr.log_events(f"URL was validated and is being presented as: {corrected_url}",
                            TroubleSgltn.Severity.INFO,
                            True)

        return corrected_url   
    
    def clean_response_text(self, text: str)-> str:
        # Replace multiple newlines or carriage returns with a single one
        cleaned_text = re.sub(r'\n+', '\n', text).strip()
        return cleaned_text
    
    @staticmethod
    def parse_anthropic_error(e):
        """
        Parses error information from an exception object.

        Args:
            e (Exception): The exception from which to parse the error information.

        Returns:
            str: A user-friendly error message.
        """
        # Default error message
        default_message = "An unknown error occurred"

        # Check if the exception has a response attribute and it can be converted to JSON
        if hasattr(e, 'response') and callable(getattr(e.response, 'json', None)):
            try:
                error_details = e.response.json()
                # Navigate through the nested dictionary safely
                return error_details.get('error', {}).get('message', default_message)
            except ValueError:
                # JSON decoding failed
                return f"Failed to decode JSON from response: {e.response.text}"
            except Exception as ex:
                # Catch-all for any other issues that may arise
                return f"Error processing the error response: {str(ex)}"
        elif hasattr(e, 'message'):
            return e.message
        else:
            return str(e)  

    
    