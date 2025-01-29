#Standard Libraries
import random

#Third Party Libraries
import torch

#Local Module references
from .mng_json import json_manager, helpSgltn, TroubleSgltn
from .fetch_models import FetchModels, RequestMode
from .style_prompt import cFigSingleton


class AnyType(str):
    """A special type that can be connected to any other types. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")


class Tagger:
    def __init__(self)-> None:
        self.trbl = TroubleSgltn()
        self.j_mngr = json_manager()
        self.help_data = helpSgltn()

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
    def enhanced_text_placement(generated_text:str, b_tags:str="", m_tags:str="", e_tags:str="", pref_periods:bool=False):

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
        b_tags.strip()
        m_tags.strip()
        e_tags.strip()

        if not b_tags and not m_tags and not e_tags:
            return generated_text 
        
        end_text, beginning_text, middle_text = '', '', ''
        beginning_text = b_tags
        middle_text = m_tags
        end_text = e_tags
    

        mid_punct = Tagger.join_punct(middle_text)
        end_punct = Tagger.join_punct(end_text)
        begin_punct = Tagger.join_punct(beginning_text)

        # Integrate middle text based on punctuation logic in the generated_text
        commas = generated_text.count(',')
        periods = generated_text.count('.')
        #punct_count = max(commas, periods)
        search_punct = []
        if pref_periods and periods > 1:
            punct_count = periods
            search_punct = ["."]
        else:
            punct_count = commas + periods
            search_punct = [",", "."]
        
        if middle_text:
            
            if punct_count == 0:
                end_text = end_punct.join([end_text, middle_text]) if end_text else middle_text
            elif punct_count <= 2:
                # Look for the first instance of either a comma or a period
                first_punctuation_index = len(generated_text)  # Default to the end of the string
                for char in search_punct:  # Check for both commas and periods
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
                    if char in search_punct:
                        count += 1
                        if count == target:
                            insert_index = i + 2  # After the punctuation and space
                            break
                generated_text = generated_text[:insert_index] + middle_text + mid_punct + generated_text[insert_index:]
        
        # Integrate beginning and end text
        if beginning_text:
            generated_text = beginning_text + begin_punct + generated_text
        if end_text:
            generated_text += Tagger.join_punct(generated_text) + end_text
        
        return generated_text.strip(', ')  # Ensure no leading or trailing commas 
    


    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "Beginning_tags": ("STRING", {"multiline": True}),
                "Middle_tags": ("STRING", {"multiline": True}),
                "Prefer_middle_tag_after_period": ("BOOLEAN", {"default": True}),               
                "End_tags": ("STRING", {"multiline": True})                                      
            },
            "optional": {
                "text": ("STRING", {"forceInput": True, "multiline": True})
            },

            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        } 

    RETURN_TYPES = ("STRING","STRING","STRING")
    RETURN_NAMES = ("tagged_text", "help","troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Utils"

    def gogo(self, unique_id, text, Beginning_tags, Middle_tags, End_tags, Prefer_middle_tag_after_period)-> tuple:
        _help = self.help_data.tagger_help
        if unique_id:
            self.trbl.reset('Tagger, Node #' + unique_id)
        else:
            self.trbl.reset('Tagger')

        if Middle_tags:
            if Prefer_middle_tag_after_period:
                message = "Prefer middle tags to be inserted after a period."
            else:
                message = "Allow middle tags to be inserted after either a period or a comma."

            self.j_mngr.log_events(message,
                                   is_trouble=True)

        output = Tagger.enhanced_text_placement(text, Beginning_tags, Middle_tags, End_tags,Prefer_middle_tag_after_period)

        self.j_mngr.log_events("Inserting tags into text block.",
                               is_trouble=True)
                               
        return(output, _help, self.trbl.get_troubles())
    

class randomOut:

    def __init__(self)-> None:
        self.trbl = TroubleSgltn()
        self.j_mngr = json_manager()
        self.help_data = helpSgltn()

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "Text": ("STRING", {"multiline": True, "forceInput": True}),
                "randomized_outputs": (['1','2','3','4','5'], {"default": "2"}),
                "seed": ("INT", {"default": 9, "min": 0, "max": 0xffffffffffffffff})
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        } 
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING")
    RETURN_NAMES = ("Text_1", "Text_2","Text_3","Text_4","Text_5")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Utils"

    def gogo(self, Text, unique_id, seed, randomized_outputs=0):
        random.seed(seed)
        rnd_out = int(randomized_outputs)
        int_rnd = random.randint(1, rnd_out)
        outputs = ["","","","",""]
        if 1 <= int_rnd <= 5:
            outputs[int_rnd-1] = Text
        out_tuple = tuple(outputs[:5])

        return out_tuple



class randomImgOut:

    def __init__(self)-> None:
        self.trbl = TroubleSgltn()
        self.j_mngr = json_manager()
        self.help_data = helpSgltn()

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "Image": ("IMAGE", {"multiline": True, "forceInput": True}),
                "randomized_outputs": (['1','2','3','4','5'], {"default": "2"}),
                "seed": ("INT", {"default": 9, "min": 0, "max": 0xffffffffffffffff})
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        } 
    
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE","IMAGE","IMAGE")
    RETURN_NAMES = ("Image_1", "Image_2","Image_3","Image_4","Image_5")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Utils"

    def gogo(self, Image, unique_id, seed, randomized_outputs=0):
        random.seed(seed)
        rnd_out = int(randomized_outputs)
        int_rnd = random.randint(1, rnd_out)
        dummy_tensor = torch.zeros(1, 8, 8, 3, dtype=torch.float32)
        outputs = [dummy_tensor,dummy_tensor,dummy_tensor,dummy_tensor,dummy_tensor]
        if 1 <= int_rnd <= 5:
            outputs[int_rnd-1] = Image
        out_tuple = tuple(outputs[:5])

        return out_tuple
    

class mixer:

    def __init__(self)-> None:
        self.trbl = TroubleSgltn()
        self.j_mngr = json_manager()
        self.help_data = helpSgltn()

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "optional": {
                "Text_1": ("STRING", {"multiline": True, "forceInput": True}),
                "Text_2": ("STRING", {"multiline": True, "forceInput": True}),
                "Text_3": ("STRING", {"multiline": True, "forceInput": True}),
                "Text_4": ("STRING", {"multiline": True, "forceInput": True}),
                "Text_5": ("STRING", {"multiline": True, "forceInput": True}),

                "seed": ("INT", {"default": 9, "min": 0, "max": 0xffffffffffffffff})
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        } 
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING")
    RETURN_NAMES = ("Output_1", "Output_2","Output_3","Output_4","Output_5")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Utils"

    def gogo(self, unique_id, seed, Text_1:str="", Text_2:str="", Text_3:str="", Text_4:str="", Text_5:str=""):

        valid_input = [text for text in [Text_1, Text_2, Text_3, Text_4, Text_5] if text] #Fill the list only with actual input data
        random.seed(seed)        
        random.shuffle(valid_input) #randomize list order

        padding = 5 - len(valid_input)
        valid_input.extend([""] * padding) #pad the outputs with no valid input data with empty strings
        out_tuple = tuple(valid_input[:5])

        return out_tuple    

class imgMixer:

    def __init__(self)-> None:
        self.trbl = TroubleSgltn()
        self.j_mngr = json_manager()
        self.help_data = helpSgltn()

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "optional": {
                "Image_1": ("IMAGE", {"multiline": True, "forceInput": True}),
                "Image_2": ("IMAGE", {"multiline": True, "forceInput": True}),
                "Image_3": ("IMAGE", {"multiline": True, "forceInput": True}),
                "Image_4": ("IMAGE", {"multiline": True, "forceInput": True}),
                "Image_5": ("IMAGE", {"multiline": True, "forceInput": True}),

                "seed": ("INT", {"default": 9, "min": 0, "max": 0xffffffffffffffff})
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        } 
    
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE","IMAGE","IMAGE")
    RETURN_NAMES = ("Output_1", "Output_2","Output_3","Output_4","Output_5")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Utils"

    def gogo(self, unique_id, seed, Image_1:str="", Image_2:str="", Image_3:str="", Image_4:str="", Image_5:str=""):

        valid_input = [img for img in [Image_1, Image_2, Image_3, Image_4, Image_5] if isinstance(img, torch.Tensor) and img.numel() > 0] #Fill the list only with actual input data
        random.seed(seed)        
        random.shuffle(valid_input) #randomize list order
        dummy_tensor = torch.zeros(1, 8, 8, 3, dtype=torch.float32)
        padding = 5 - len(valid_input)
        valid_input.extend([dummy_tensor] * padding) #pad the outputs with no valid input data with tiny black tensors
        out_tuple = tuple(valid_input[:5])

        return out_tuple    


class removeText:
    def __init__(self):
        self.trbl = TroubleSgltn()
        self.j_mngr = json_manager()
        self.help_data = helpSgltn()

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "Opening_tag": ("STRING", {"multiline": False, "tooltip": "Enter the character(s) that start the text block you want to remove"}),
                "Closing_tag": ("STRING", {"multiline": False, "tooltip": "Enter the character(s) that end the text block you want to remove"}), 
                "Open_tag_instance": ("INT", {"default": 1, "min": 1, "max": 550, "tooltip": "Enter which instance of the Opening_tag you want to use"}), 
                "Close_tag_instance": ("INT", {"default": 1, "min": 1, "max": 550, "tooltip": "Enter which instance of the Closing_tag you want to use"}),                              
                "Remove_tags": ("BOOLEAN", {"default": True}),
                "Pass_Through_on_error": ("BOOLEAN", {"default": True})
            },
            "optional" :{
                "Text":("STRING",{"default": "", "forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        } 
    
    RETURN_TYPES = ("STRING","STRING", "STRING")
    RETURN_NAMES = ("Text" ,"Removed_text" ,"Troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Utils"

    def gogo (self, Text, Opening_tag, Closing_tag, unique_id, Open_tag_instance, Close_tag_instance, Pass_Through_on_error:bool, Remove_tags:bool)->tuple:

        self.trbl.reset(f"Remove Text, Node: {unique_id}")

        clean_text, removed_text = self.j_mngr.remove_text(Text, Opening_tag, Closing_tag, Open_tag_instance, Close_tag_instance, Remove_tags)

        if clean_text:
            self.j_mngr.log_events("Sucessfully removed text", is_trouble=True)
        elif Pass_Through_on_error:
            self.j_mngr.log_events("Removal Failed, but original text was passed through",
                                   TroubleSgltn.Severity.WARNING,
                                   True)
            clean_text = Text

        return(clean_text, removed_text, self.trbl.get_troubles())




class typeConvert:

    def __init__(self)-> None:
        self.trbl = TroubleSgltn()
        self.j_mngr = json_manager()
        self.help_data = helpSgltn()

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "optional": {
                "Text": ("STRING", {"multiline": True, "forceInput": True}),
                "Cross_reference_types": ("BOOLEAN", {"default": True})
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        } 
    
    RETURN_TYPES = ("STRING","FLOAT","INT","BOOLEAN","LIST","DICTIONARY", "STRING", "STRING")
    RETURN_NAMES = ("Text", "Float","Integer","Boolean","List", "JSON/Dict", "help","Troubleshooting")

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Utils"

    def gogo(self, unique_id, Cross_reference_types:bool=False,  Text:str=""):

        self.trbl.reset(f"Type Converter, Node: {unique_id}")
        _help = self.help_data.type_convert_help
        out_list = [Text, None, None, None, None, None]
        cxt = Cross_reference_types

        # Log the input string received
        self.j_mngr.log_events(f"Received input text: {Text}", is_trouble=True)

        # Infer the data type of the input text
        inferred_value = self.j_mngr.infer_type(Text)
        inferred_type = type(inferred_value)

        # Log the inferred type
        self.j_mngr.log_events(f"Primary Inferred data type: {inferred_type}", is_trouble=True)

        # Place the inferred value in the correct position based on its type
        # Handle float and integer conversions
        if isinstance(inferred_value, float):
            out_list[1] = inferred_value  # Store the float value
            if cxt:
                out_list[2] = round(inferred_value)  # Store the rounded integer equivalent

        elif isinstance(inferred_value, int):
            if isinstance(inferred_value, bool): #If it's also a boolean convert to an int
                out_list[2] = int(inferred_value) 
            else:
                out_list[2] = inferred_value  # Store the integer value
            if cxt:
                out_list[1] = float(inferred_value)  # Store the float equivalent (int.0)

            # Convert integer to boolean if it's 0 or 1
            if inferred_value == 0 and cxt:
                out_list[3] = False  # int 0 -> False
            elif inferred_value == 1 and cxt:
                out_list[3] = True  # int 1 -> True

        elif isinstance(inferred_value, bool):
            out_list[3] = inferred_value  # Store the boolean value
            if cxt:
                out_list[2] = int(inferred_value)  # Store the integer equivalent (1 for True, 0 for False)

        elif isinstance(inferred_value, list):
            out_list[4] = inferred_value

        elif isinstance(inferred_value, dict):
            out_list[5] = inferred_value
        
        else:
            self.j_mngr.log_events("No data type conversion performed. Output as string.", 
                                   TroubleSgltn.Severity.WARNING,
                                   True)

            # Return the tuple of outputs
            
        out_tuple = tuple(out_list) + (_help, self.trbl.get_troubles())

        return out_tuple    
    


class OpenRouterModels:

    def __init__(self):
        self.trbl = TroubleSgltn()
        self.j_mngr = json_manager()
        self.help_data = helpSgltn()
        self.ftch = FetchModels()
        self.cFig = cFigSingleton()
        self._model_container = None #Cached object per node session

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "Service_Name": (["OpenRouter"], {"default": "OpenRouter", "tooltip": "Select the Service you want to load models from."}),
                "Output_to": (["Test_Output","Optional Models List"], {"default": "Test_Output", "tooltip": "Select the drop-down list data file or the test output and send the model list to it."}),
                "Include_Filter": ("STRING", {"multiline": True, "tooltip": "Enter criteria for including models in the list. Separate items with a comma(s)"}),
                "Exclude_Filter": ("STRING", {"multiline": True, "tooltip": "Enter criteria for excluding models from the list. Separate items with a comma(s)"}),
                "Remove_Prior_Sevice_Name_Entries": ("BOOLEAN", {"default": True, "tooltip": "Remove entries in the text file that include the current Service Name"}),
                "Sort_Models": ("BOOLEAN", {"default": True})
            },
            "optional" :{
                "Custom_ApiKey":("KEY",{"default": "", "forceInput": True}),
            }
        } 
    
    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("Test_Output","Troubleshooting")
    
    FUNCTION = "gogo"

    OUTPUT_NODE = True

    CATEGORY = "Plush🧸/Utils"

    def gogo(self, Service_Name, Output_to, Custom_ApiKey:str="", Include_Filter:str="", Exclude_Filter:str="",Remove_Prior_Sevice_Name_Entries:bool=True, Sort_Models:bool=True):

        self.trbl.reset("OpenRouter Models")

        output = ""
        dest_file = ""
        if Output_to == "Optional Models List":
            dest_file = "opt_models.txt"

        if not Custom_ApiKey:
            self.j_mngr.log_events("You must attach the Plush 'Custom API Key' node and enter a valid Env. Variable name. Error: No API Key provided",
                                   TroubleSgltn.Severity.ERROR,
                                   True)
            return(output,self.trbl.get_troubles(),)
        
        if Include_Filter:
            Include_Filter = self.j_mngr.create_iterable(Include_Filter)
        if Exclude_Filter:
            Exclude_Filter = self.j_mngr.create_iterable(Exclude_Filter)

        if not self._model_container or not self._model_container.has_data:  
            self.j_mngr.log_events("Fetching models names from Open Router", is_trouble=True)
            self._model_container = self.ftch.fetch_models(request_type=RequestMode.OPENROUTER, key=Custom_ApiKey)

            # Early exit if no models were fetched
            if not self._model_container.has_data:
                self.j_mngr.log_events(f"No models were returned from service: {Service_Name}",
                                        TroubleSgltn.Severity.WARNING,
                                        is_trouble=True)
                return (output, self.trbl.get_troubles(),)
            
        else:  #If model_container has data, reuse it
            self.j_mngr.log_events("Using cached model list", is_trouble=True)
 
        model_list = self._model_container.get_models(sort_it=Sort_Models, include_filter=Include_Filter, exclude_filter=Exclude_Filter, with_none=False)

        if not model_list:  # No models met the filter criteria
            self.j_mngr.log_events("Filtered model list is empty; no output.", 
                               TroubleSgltn.Severity.INFO, is_trouble=True)
            return (output, self.trbl.get_troubles(),)

            # Prepend service name
        model_list = [f"{Service_Name} :: {model}" for model in model_list]
        model_count = len(model_list)

        self.j_mngr.log_events(f"Sending model list to Output/File: {Output_to}", is_trouble=True)

        if not Output_to == "Test_Output":
            write_file = self.j_mngr.append_filename_to_path(self.j_mngr.script_dir, dest_file)
            output = f"Output sent to file: {Output_to}"

            if Remove_Prior_Sevice_Name_Entries:
                self.j_mngr.log_events(f"Removing existing lines that include the text: {Service_Name}",is_trouble=True )
                self.j_mngr.remove_lines_by_criteria(file_path=write_file, delete_criteria=Service_Name)

            self.j_mngr.write_list_to_file(model_list, write_file, append=True)
            self.j_mngr.log_events(f"{model_count} models were written to: {write_file}", is_trouble=True)

        else: #User is sending output to the node's output rather than a file
            self.j_mngr.log_events(f"{model_count} models were output.", is_trouble=True)

            output = "\n".join(model_list)

        return(output,self.trbl.get_troubles(),)
 

class mulTextSwitch:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "active_input": ("INT", {"max": 3, "min": 1, "step": 1, "default": 1, "display": "number"})
            },
            "optional": {
                "Input_1": ("STRING", {"multiline": True, "forceInput": True}),
                "Input_2": ("STRING", {"multiline": True, "forceInput": True}),
                "Input_3": ("STRING", {"multiline": True, "forceInput": True}),
            }
        } 
    
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("Multiline Text", )

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Utils"

    def gogo(self, active_input, Input_1=None, Input_2=None, Input_3=None):

        ret_text = ""

        if active_input == 1 and Input_1:
            ret_text = Input_1
        elif active_input == 2 and Input_2:
            ret_text = Input_2
        elif active_input ==3 and Input_3:
            ret_text = Input_3

        if not ret_text:
            raise Exception ("Missing text input, check selction")

        return (ret_text, )
    


class ImgTextSwitch:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "active_input": ("INT", {"max": 3, "min": 1, "step": 1, "default": 1, "display": "number"})
            },
            "optional": {
                "Text_1": ("STRING", {"multiline": True, "forceInput": True}),
                "Image_1" : ("IMAGE", {"default": None}),
                "Text_2": ("STRING", {"multiline": True, "forceInput": True}),
                "Image_2" : ("IMAGE", {"default": None}),
                "Text_3": ("STRING", {"multiline": True, "forceInput": True}),
                "Image_3" : ("IMAGE", {"default": None})
            }
        } 
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("Multiline Text","Image" )

    FUNCTION = "gogo"

    OUTPUT_NODE = False

    CATEGORY = "Plush🧸/Utils"

    def gogo(self, active_input, Text_1=None, Image_1=None, Text_2=None, Image_2=None, Text_3=None, Image_3=None):

        ret_text = ""
        ret_img = None

        if active_input == 1:
            ret_text = Text_1
            ret_img = Image_1
        elif active_input == 2:
            ret_text = Text_2
            ret_img = Image_2
        elif active_input ==3:
            ret_text = Text_3
            ret_img = Image_3

        if not ret_text and not ret_img:
            raise Exception ("Missing text and image input, check selction")

        return (ret_text, ret_img)

class jsonParse:
    def __init__(self)-> None:
        self.trbl = TroubleSgltn()
        self.j_mngr = json_manager()
        self.help_data = helpSgltn()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key_1": ("STRING", {"default": "", "tooltip": "JSON key whose value will be output"}),
                "key_2": ("STRING", {"default": "", "tooltip": "JSON key whose value will be output"}),
                "key_3": ("STRING", {"default": "", "tooltip": "JSON key whose value will be output"}),
                "key_4": ("STRING", {"default": "", "tooltip": "JSON key whose value will be output"}),
                "key_5": ("STRING", {"default": "", "tooltip": "JSON key whose value will be output"})
            },
            "optional": {
                "json_string": ("STRING",{"multiline": True, "default": "", "forceInput": True}),
            }
        }

    FUNCTION = "gogo"
    RETURN_NAMES = ("string_1", "string_2", "string_3", "string_4", "string_5", "JSON_Obj", "help", "Troubleshooting")
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "DICTIONARY","STRING","STRING")
    CATEGORY = "Plush🧸/Utils"

    def gogo(self, json_string: str, key_1: str, key_2: str, key_3: str, key_4: str, key_5: str,):
        self.trbl.reset("Extract JSON")
        _help = self.help_data.extract_json_help
        s_json = json_string.strip()

        #Some nodes strip off curly braces.
        if not s_json.startswith('{'):
            s_json = '{' + s_json
        if not s_json.endswith('}'):
            s_json = s_json + '}'

        # Create a list and exclude empty strings
        key_list = [var for var in [key_1, key_2, key_3, key_4, key_5] if var]

        p_json = self.j_mngr.convert_from_json_string(s_json)
        if p_json is None:
            self.j_mngr.log_events("Invalid JSON presented in JSON parse node.", TroubleSgltn.Severity.ERROR, True)
            return "", "", "", "", "", {},_help,self.trbl.get_troubles()

        if isinstance(p_json, (list, dict)):
            self.j_mngr.log_events("Extracting JSON key values.",is_trouble=True)
            p_json = self.j_mngr.extract_list_of_dicts(p_json, key_list)
        else:
            self.j_mngr.log_events("Invalid JSON presented in JSON parse node.",
                                    TroubleSgltn.Severity.ERROR,
                                    True)
            return "", "", "", "", "", {},_help,self.trbl.get_troubles() 

        # Extract values for each key and ensure output format
        values = {k: str(p_json.get(k, "")) for k in key_list}

        return (*[values.get(k, "") for k in [key_1, key_2, key_3, key_4, key_5]], p_json,_help,self.trbl.get_troubles())
    
class TextAny:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Text": ("STRING", {"default": '', "multiline": True}),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "gogo"
    CATEGORY = "Plush🧸/Utils"

    def gogo(self, Text):
        #Connects to any data type
        return (Text,)

    

class ShowInfo_md:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "md_text": ("STRING", {"multiline": True, "forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("String",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "Plush🧸/Utils"

    def notify(self, md_text, unique_id=None, extra_pnginfo=None):
        text = md_text
        if unique_id and extra_pnginfo and "workflow" in extra_pnginfo[0]:
            workflow = extra_pnginfo[0]["workflow"]
            node = next(
                (x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]), None
            )
            if node:
                node["widgets_values"] = [text]
        
        # Ensure that 'text' is a string
        if isinstance(text, list):
            text = ''.join(text)
        elif not isinstance(text, str):
            text = str(text)
        

        return {"ui": {"text": text}, "result": (text,)}
    



    
NODE_CLASS_MAPPINGS = {
    "mulTextSwitch": mulTextSwitch,
    "ImgTextSwitch": ImgTextSwitch,
    "Tagger": Tagger,
    "ParseJSON": jsonParse,
    "Random Output": randomOut,
    "Random Mixer": mixer,
    "Type Converter": typeConvert,
    "Image Mixer": imgMixer,
    "Random Image Output": randomImgOut,
    "Load Remote Models": OpenRouterModels,
    "Text (Any)": TextAny,
    "Remove Text": removeText


}

NODE_DISPLAY_NAME_MAPPINGS = {
"mulTextSwitch": "MultiLine Text Switch🧸",
"ImgTextSwitch": "Image & Text Switch🧸",
"Tagger": "Tagger🧸",
"ParseJSON": "Extract JSON data🧸",
"Random Output": "Random Output🧸",
"Random Mixer": "Random Mixer🧸",
"Type Converter": "Plush - Type Converter🧸",
"Image Mixer": "Image Mixer🧸",
"Random Image Output": "Random Image Output🧸",
"Load Remote Models": "Load Remote Models🧸",
"Text (Any)": "Text (Any input)🧸",
"Remove Text": "Remove Text Block🧸"

}
