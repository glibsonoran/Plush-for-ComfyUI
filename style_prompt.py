
import openai
from openai import OpenAI
import os
import json
import base64
from io import BytesIO
from PIL import Image 
import numpy as np
import torch
import sys
import shutil

#debug
print(f"Python system path {sys.path}")
#pip install pillow
#pip install bytesio


#Get information from the config.json file
class cFigSingleton:
    _instance = None

    def __new__(cls): 
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.get_file()
        return cls._instance
    

    
    def get_file(self):
        #Get script working directory
        
        NODE_FILE = os.path.abspath(__file__)
        #Get script directory (the location of config.json)
        SUITE_DIR = os.path.dirname(NODE_FILE)
        PARENT_DIR = os.path.normpath(os.path.join(SUITE_DIR, '..'))     
        CUR_DIR = os.path.basename(SUITE_DIR)      
        #If we're running as a package in ComfyUI_plus, use the parent directory for config.json. 
        #Otherwise use the current (SUITE_DIR) directory.
        if CUR_DIR == 'src':
            config_file_path = os.path.join(PARENT_DIR, 'config.json') 
        else:
            config_file_path = os.path.join(SUITE_DIR, 'config.json')

        #print("Enhancer json path:" + config_file_path)

         #Open and read config.json       

        try:
            with open(config_file_path, 'r') as config_file:
                #check if file is empty
                if os.stat(config_file_path).st_size == 0:
                    raise ValueError("Error: config.json is empty")
                
                config_data =  json.load(config_file)
                if not config_data:
                    raise ValueError("Error: config.json contains no valid JSON data")
        except FileNotFoundError:
            raise FileNotFoundError(f"Error config.json not found in {config_file_path}.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {config_file_path}: {e}")
            raise
        
        #set property variables
        self.figKey = config_data['key']
        self.figInstruction = config_data['instruction']
        self.figExample = config_data['example']
        self.figStyle = config_data['style']

    @property
    def key(self):
        return self.figKey

    @property
    def instruction(self):
        return self.figInstruction
    
    @property
    def example(self):
        return self.figExample
    
    @property
    def style(self):
        #make sure the designated default value is present in the list
        if "Photograph" not in self.figStyle:
            self.figStyle.append("Photograph")

        return self.figStyle


class Enhancer:
#Build a creative prompt using a ChatGPT model    
   
    def __init__(self):
        self.eFig = cFigSingleton()
        
    def build_instruction(self, style, elements, artist):
          #build the prompt from user input
        if self.eFig.instruction.count("{}") >= 2:
            instruc = self.eFig.instruction.format(style, elements)
        elif self.eFig.instruction.count("{}") == 1:
            instruc - self.eFig.instruction.format(style)
        else:
            instruc = self.eFig.instruction

        if artist:
            instruc = instruc + "  Include an artist who works in the specifed artistic style by placing the artist's name at the end of the sentence prefaced by 'style of'.  "
        return(instruc)
    

    def cgptRequest(self, GPTmodel, client, creative_latitude, tokens, prompt, instruction="", example=""):
        #Requet a prompt or backgrounder from ChatGPT
        print(f"Talking to model: {GPTmodel}")
        try:
            #call the ChatGPT API with the user selections, instruction, example and prompt
            chat_completion = client.chat.completions.create(
                model = GPTmodel,
                temperature = creative_latitude,
                max_tokens = tokens,
                stream = False,
            messages = [
                {"role": "system", "content": instruction},
                {"role": "assistant", "content": example},
                {"role": "user", "content": prompt,},
                ],
            )
        except openai.APIConnectionError as e:
            print("Server connection error: {e.__cause__}")  # from httpx.
            raise
        except openai.RateLimitError as e:
            print(f"OpenAI RATE LIMIT error {e.status_code}: (e.response)")
            raise
        except openai.APIStatusError as e:
            print(f"OpenAI STATUS error {e.status_code}: (e.response)")
            raise
        except openai.BadRequestError as e:
            print(f"OpenAI BAD REQUEST error {e.status_code}: (e.response)")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise   

        #First of choices [0] content is the generated prompt 
        CGPT_prompt = chat_completion.choices[0].message.content
        
        return(CGPT_prompt)
        
    
    @classmethod
    def INPUT_TYPES(cls):
        iFig=cFigSingleton()

        return {
            "required": {
                "GPTmodel": (["gpt-3.5-turbo","gpt-4"],{"default": "gpt-4"} ),
                "creative_latitude" : ("FLOAT", {"max": 1.2, "min": 0.1, "step": 0.1, "display": "number", "default": 0.7}),
                "tokens" : ("INT", {"max": 8000, "min": 50, "step": 10, "default": 2000, "display": "number"}),
                "prompt": ("STRING",{"multiline": True, "forceInput": True}),
                "example" : ("STRING", {"forceInput": True, "multiline": True}),
                "style": (iFig.style,{"default": "Photograph"}),
                "artist" : ("BOOLEAN", {"default": True}),
                "max_elements" : ("INT", {"max": 20, "min": 3, "step": 1, "default": 10, "display": "number"}),
                "style_info" : ("BOOLEAN", {"default": False})                
            },
        } 

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("CGPTprompt", "CGPTinstruction","Style Info")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush"
 

    def gogo(self, GPTmodel, creative_latitude, tokens, example, prompt, style, artist, max_elements, style_info):
        
        #If no example text was provided by the user, use my default
        if not example:
            example = self.eFig.example
        
        client = OpenAI(
        api_key= self.eFig.key
        )

        enH = Enhancer()
        #build instruction based on user input
        instruction = enH.build_instruction(style, max_elements, artist)  

        CGPT_styleInfo = ""

        if style_info:
            #User has request information about the art style.  GPT will provide it
            sty_prompt = "Give an 150 word backgrounder on the art style: {}.  Starting with describing what it is, include information about its history and which artists represent the style."
            sty_prompt = sty_prompt.format(style)
 
            CGPT_styleInfo = self.cgptRequest(GPTmodel, client, creative_latitude, tokens, sty_prompt )

        CGPT_prompt = self.cgptRequest(GPTmodel, client, creative_latitude, tokens, prompt, instruction, example)

        
       # *****************************************
       # print("Tokens: " + str(tokens))
       # print("Creativity: " + str(creative_latitude))
        #print("Prompt: " + CGPT_prompt)
        #print("Instruction: " + instruction)
    
        return (CGPT_prompt, instruction, CGPT_styleInfo)

 

#debug testing 
""" Enh = Enhancer()
Enh.INPUT_TYPES()
test_resp = Enh.gogo("gpt-4", 0.7, 2000, "", "A beautiful eagle soaring above a verdant forest", "Biomorphic Abstraction", True, 10,True)
print (test_resp[2]) """

class DalleImage:
#Accept a user prompt and parameters to produce a Dall_e generated image

    def __init__(self):
        self.eFig = cFigSingleton()

    @classmethod
    def INPUT_TYPES(cls):
        #dall-e-2 API requires differnt input parameters as compared to dall-e-3, at this point I'll just use dall-e-3
        return {
            "required": {
                "GPTmodel": (["dall-e-3",], ),
                "prompt": ("STRING",{"multiline": True, "forceInput": True}), 
                "image_size": (["1792x1024", "1024x1792", "1024x1024"], {"default": "1024x1024"} ),              
                "image_quality": (["standard", "hd"], {"default": "hd"} ),
                "style": (["vivid", "natural"], {"default": "natural"} )
            },
        } 

    RETURN_TYPES = ("IMAGE", "MASK", "STRING" )
    RETURN_NAMES = ("image", "mask", "Dall_e_prompt")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush"

    def gogo(self, GPTmodel, prompt, image_size, image_quality, style):
        
        
        client = OpenAI(
        api_key= self.eFig.key
        )

        
        print(f"Talking to Dalle model: {GPTmodel}")
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
        except openai.APIConnectionError as e:
            print("Server connection error: {e.__cause__}")  # from httpx.
            raise
        except openai.RateLimitError as e:
            print(f"OpenAI RATE LIMIT error {e.status_code}: (e.response)")
            raise
        except openai.APIStatusError as e:
            print(f"OpenAI STATUS error {e.status_code}: (e.response)")
            raise
        except openai.BadRequestError as e:
            print(f"OpenAI BAD REQUEST error {e.status_code}: (e.response)")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

 
      #Covert b64 JSON to .png file
        revised_prompt = response.data[0].revised_prompt
        b64Json = response.data[0].b64_json
        png_data = base64.b64decode(b64Json)
        
        png_image = Image.open(BytesIO(png_data))
       
        #This saves the .png as a file
        #os.makedirs("test_image", exist_ok=True)
        #png_image.save("test_image/testDall_eImage2.png")
        #*************************************************

        # Convert the image to RGB format
        png_image = png_image.convert("RGB")
        png_image = np.array(png_image)
        png_image = png_image.astype(np.float32) / 255.0

        # Convert the image to a PyTorch tensor
        #png_image = np.transpose(png_image, (2, 0, 1))
        png_image = torch.from_numpy(png_image).unsqueeze(0)
        #.permute(0, 1, 2, 3)

        # Check if the image has an alpha channel
    
        if png_image.shape[3] == 4:  # Assuming channels are in the third dimension
            # Extract and normalize the alpha channel
            mask = png_image[:, 3, :, :].clone() / 255.0
            mask = 1. - mask
        else:
            # Create a default mask if there's no alpha channel
            mask = torch.zeros_like(png_image[:, 0, :, :], dtype=torch.float32)
        
        #png_image = png_image * 0.9529

        png_image = png_image.float()

        mask = mask.float()
        #Image.fromarray(np.clip(255. * png_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        #may need this if sizing doesn't work
        #resized_image = transforms.functional.resize(png_image, (1, 3, 1024, 1024))
        
        return (png_image, mask.unsqueeze(0), revised_prompt)
    

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Enhancer": Enhancer,
    "DalleImage": DalleImage
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Enhancer": "Style Prompt",
    "DalleImage": "OAI Dall_e Image"
}
    


#debug testing    
""" Di = DalleImage()
ddict = Di.INPUT_TYPES()
tst = []
tst = Di.gogo("dall-e-3", "A woman standing by a flowing river", "1024x1024", "hd", "natural")
myname = tst[0].names  """