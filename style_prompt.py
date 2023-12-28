
import openai
from openai import OpenAI
import os
import base64
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import torch
import sys
from .mng_json import json_manager

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
        j_mngr = json_manager()

        # Error handling is in the load_json method
        # Errors will be raised since is_critical is set to True
        config_data = j_mngr.load_json(j_mngr.config_file, True)

        #check if file is empty
        if not config_data:
            raise ValueError("Plush - Error: config.json contains no valid JSON data")
        
        #set property variables
        # Try getting API key from Plush environment variable
        self.figKey = os.getenv('OAI_KEY')
        # Try the openAI recommended Env Variable.
        if not self.figKey:
            self.figKey = os.getenv("OPENAI_API_KEY")
        # Temporary: Lastly get the API key from config.json
        if not self.figKey:
            self.figKey = config_data['key']
        # Final check to ensure an API key is set
        if not self.figKey:
            raise ValueError("Plush - Error: OpenAI API key not found. Please set it as an environment variable (See the Plush ReadMe).")
     
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
          #build the instruction from user input
        if self.eFig.instruction.count("{}") >= 2:
            instruc = self.eFig.instruction.format(style, elements)
        elif self.eFig.instruction.count("{}") == 1:
            instruc - self.eFig.instruction.format(style)
        else:
            instruc = self.eFig.instruction

        if artist >= 1:
            art_instruc = "  Include {} artist(s) who works in the specifed artistic style by placing the artists' name(s) at the end of the sentence prefaced by 'style of'."
            instruc = instruc + art_instruc.format(str(artist))

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

        #Floats have a problem, they go over the max value even when round and step are set, and the node fails.  So I set max a little over the expected input value
        return {
            "required": {
                "GPTmodel": (["gpt-3.5-turbo","gpt-4","gpt-4-1106-preview"],{"default": "gpt-4"} ),
                "creative_latitude" : ("FLOAT", {"max": 1.201, "min": 0.1, "step": 0.1, "display": "number", "round": 0.1, "default": 0.7}),
                "tokens" : ("INT", {"max": 8000, "min": 50, "step": 10, "default": 2000, "display": "number"}),
                "prompt": ("STRING",{"multiline": True, "forceInput": True}),
                "example" : ("STRING", {"forceInput": True, "multiline": True}),
                "style": (iFig.style,{"default": "Photograph"}),
                "artist" : ("INT", {"max": 3, "min": 0, "step": 1, "default": 1, "display": "number"}),
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
        CGPT_styleInfo = None
        enH = Enhancer()
        #build instruction based on user input
        instruction = enH.build_instruction(style, max_elements, artist)  

        if style_info:
            #User has request information about the art style.  GPT will provide it
            sty_prompt = "Give an 150 word backgrounder on the art style: {}.  Starting with describing what it is, include information about its history and which artists represent the style."
            sty_prompt = sty_prompt.format(style)
 
            CGPT_styleInfo = self.cgptRequest(GPTmodel, client, creative_latitude, tokens, sty_prompt )

        CGPT_prompt = self.cgptRequest(GPTmodel, client, creative_latitude, tokens, prompt, instruction, example)

    
        return (CGPT_prompt, instruction, CGPT_styleInfo)

 

#debug testing 
""" Enh = Enhancer()
Enh.INPUT_TYPES()
test_resp = Enh.gogo("gpt-4", 0.7, 2000, "", "A beautiful eagle soaring above a verdant forest", "Biomorphic Abstraction", 3, 10,True)
print (test_resp[2]) """

class DalleImage:
#Accept a user prompt and parameters to produce a Dall_e generated image

    def __init__(self):
        self.eFig = cFigSingleton()

        
    def b64_to_tensor(self, b64_image: str) -> torch.Tensor:

        """
        Converts a base64-encoded image to a torch.Tensor.

        Note: ComfyUI expects the image tensor in the [N, H, W, C] format.  
        For example with the shape torch.Size([1, 1024, 1024, 3])

        Args:
            b64_image (str): The b64 image to convert.

        Returns:
            torch.Tensor: an image Tensor.
        """        
        # Decode the base64 string
        image_data = base64.b64decode(b64_image)
        
        # Open the image with PIL and handle EXIF orientation
        image = Image.open(BytesIO(image_data))
        image = ImageOps.exif_transpose(image)
        
        # Convert to RGB and normalize
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Convert to PyTorch tensor
        tensor_image = torch.from_numpy(image_np)

        # Check shape and permute if necessary
        #if tensor_image.shape[-1] in [3, 4]:
            #tensor_image = tensor_image.permute(2, 0, 1)  # Convert to [C, H, W]  
  
        # Create a mask if there's an alpha channel
        if tensor_image.ndim == 3:  # If the tensor is [C, H, W]
            mask = torch.zeros_like(tensor_image[0, :, :], dtype=torch.float32)
        elif tensor_image.ndim == 4:  # If the tensor is [N, C, H, W]
            mask = torch.zeros_like(tensor_image[0, 0, :, :], dtype=torch.float32)

        if tensor_image.shape[1] == 4:  # Assuming channels are in the first dimension after unsqueeze
            mask = 1.0 - tensor_image[:, 3, :, :] / 255.0
        
        tensor_image = tensor_image.float()
        mask = mask.float()

        return tensor_image.unsqueeze(0), mask
    

    
    def tensor_to_base64(self, tensor: torch.Tensor) -> str:
        """
        Converts a PyTorch tensor to a base64-encoded image.

        Note: ComfyUI provides the image tensor in the [N, H, W, C] format.  
        For example with the shape torch.Size([1, 1024, 1024, 3])

        Args:
            tensor (torch.Tensor): The image tensor to convert.

        Returns:
            str: Base64-encoded image string.
        """
    # Convert tensor to PIL Image
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension if present
        pil_image = Image.fromarray((tensor.numpy() * 255).astype('uint8'))

        # Save PIL Image to a buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")  # Can change to PNG if preferred
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

 
      # Get the revised_prompt
        revised_prompt = response.data[0].revised_prompt

        #Convert the b64 json to a pytorch tensor

        b64Json = response.data[0].b64_json

        png_image, mask = self.b64_to_tensor(b64Json)
        

        
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