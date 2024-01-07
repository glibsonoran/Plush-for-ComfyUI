


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

    CATEGORY = "Plush/Utils"

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

    CATEGORY = "Plush/Utils"

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
    
NODE_CLASS_MAPPINGS = {
    "mulTextSwitch": mulTextSwitch,
    "ImgTextSwitch": ImgTextSwitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
"mulTextSwitch": "MultiLine Text Switch",
"ImgTextSwitch": "Image & Text Switch"
}