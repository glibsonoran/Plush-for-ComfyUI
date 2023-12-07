# Plush-for-ComfyUI
****
### Plush contains two OpenAI enabled nodes: 


**Style Prompt**: Takes your prompt and the art style you specify and generates a prompt from ChatGPT3 or 4 that Stable Diffusion can use to generate an image in that style.


**OAI Dall_e 3**:  Takes your prompt and parameters and produces a Dall_e3 image in ComfyUI.
*****
### Installation:

Follow the link to the [Plush for ComfyUI Github page](https://github.com/glibsonoran/Plush-for-ComfyUI "Plush Github Page").  

Click on the green Code button at the top right of the page.  When the tab drops down, click to the right of the url to copy it.

![alt text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/c9277b63-7307-4fbd-86e6-b772db4165af "Copy Url")

Then navigate, in the command window on your computer, to the ComfyUI/custom_nodes folder and type:
 >git clone https://github.com/glibsonoran/Plush-for-ComfyUI.git.

You should then have a new folder ComfyUI/custom_nodes/Plush-for-ComfyUI.
****
### Requirements: 

* You’ll need to have ComfyUI installed and it’s recommended that you have the Base and Refiner SDXL models as those are the models this node was designed for and tested on.  The Style Prompt node relies on having a model that has a broad set of images that have been carefully labeled with art style and artist.  I think the SDXL base and refiner are best suited to this.

* You’ll need a valid API key from OpenAI, which requires a paid account.  Generate the key from their website.

* Plush requires the OpenAI Python library version 1.3.5 or later.  If you have used earlier nodes that communicate with ChatGPT you may have an early version of this library in which case you’ll need to upgrade it.  You can upgrade by typing the command:   **>pip install openai --upgrade**  in a directory *where it will apply to the installation of Python that ComfyUI is using.*  

* Be aware that the new OpenAI API is not backward compatible and apps that use the older library may break after this upgrade.
****

### Usage:
 
Before you launch ComfUI, you’ll need to give Plush access to your OpenAI API key.  Navigate to the ComfyUI/custom_nodes/Plush-for-ComfyUI directory and open the configuration JSON:  config.json. You’ll see the word “key” and next to it a placeholder for you API key “jk-######”.  Replace the placeholder with your API key and close the file.  

There’s another config.json in the ComfyUI/custom_nodes/Plush-for-ComfyUI/bkup directory.  This file gets copied to the main directory if the main config.json is missing or corrupted.  It’s a good idea to enter your API key in this file too.

Style Prompt:


Inputs:

prompt:  Your prompt, it doesn’t need to be wordy or complex, simpler prompts work better.


example (optional):  A text example of how you want ChatGPT’s prompt to look.  There’s a default example in Style Prompt that works well, you can override it if you like by using this input.  Examples are for writing style, it doesn’t matter if they pertain to the same subject as your prompt.


Outputs: 

CGPTprompt:  The prompt ChatGPT generates for your image, this should connect to the CLIP node. Alternatively you can have a text display node either in-line between Style Prompt and the CLIP node, or as a separate branch off this output.  In either case a text display node will show you the ChatGPT generated prompt.

CGPTInstruction (optional): This will show you the instruction that was sent to ChatGPT along with the prompt.  The instruction tells ChatGPT how to treat the prompt.  It’s pretty much the same every time so typically it’s not worth hooking up after you’ve seen a couple.

Style Info (optional):  If the style_info UI control is set to “true”, this will output a brief backgrounder on the art style you’ve chosen:  What it is, its history and the names of some artists who have been influential in that style.  This will require connecting it to a text display box if you’re going to use it.


UI inputs:

GPTModel (default gpt-4):  The ChatGPT model that’s going to generate the prompt. GPT-4 works better than GPT-3.5 turbo, but 3.5 is slightly cheaper to use.

Creative_lattitude (default 0.7):  This is very similar to cfg in the KSampler.  It’s how much freedom the AI model has to creatively interpret your prompt, example and instruction.  Small numbers make the model stick closely to your input, larger ones give it more freedom to improvise.  The actual range is from 0.1 to 2.0, but I’ve found that anything above 1.1 or 1.2 is just disjointed word salad. So I’ve limited the range to 1.2, and even then I don’t go above 0.9.

Tokens (default 2000):  A limit on how many tokens ChatGPT can use in providing your prompt.  Paid use of the API is based on the number of tokens used.  This isn’t how many ChatGPT will use, it’s a limit on how many it can use.  If you want to strictly control costs you can play around with the minimum number of tokens that will get you a good prompt.  I just leave it at 2000.

Style (default Photograph):  This is the heart of Style Prompt.  I’ve included a list of dozens of art styles to choose from and my instructions tell ChatGPT to build the prompt in a way that pertains to the chosen style.  It’s ChatGPT’s interpretation of the art style, knowledge of artists that work in that style, and what descriptive elements best relate to that style that makes the node effective at depicting the various styles.

Artist (default True): Whether to include a “style of” statement with the name of an artist that exemplifies the style you’ve chosen.  Style Prompt is better at depicting the chosen style if this is set to True.

Max_elements (default 10):  The maximum number of descriptive elements for ChatGPT to include in its generated prompt.  Stable Diffusion gives the highest weighting to text at the beginning of the prompt, and the weighting falls off from there.  There’s definitely a point where long wordy SD prompts result in diminishing returns.  The range here is from 3 to 20.  I think 6 to 10 works about the best.

Style_info (default false):  If this is set to true, Style Prompt will send a second request to ChatGPT to provide a description of the chosen style, historical information about it, and information on some of the most influential artists in that style.  

Examples:


Style: Fish Eye Lens Photograph


Style: High Key Photograph


Style: Semi Realistic Cyberpunk


Style: Low Key Photograph


Style: Art Deco


Style: Origami


Style: Fashion Art


OAI Dall_e Image



I’m not going to go into detail about this node.  The main thing is that it takes your prompt and outputs an image.  Right now it’s only setup to use dall_e3 as the required input values are too different for me to include dall_e2.  3 produces better images so I just didn’t think accommodating 2 was worth it. 

You should be aware that in the API implementation Dall_e completely rewrites your prompt in an attempt to control misuse.  The text of that rewritten prompt is what is produced by the Dall_e_prompt output in this node. This can create some odd results, and some prompts will generate a ComfyUI error as Dall_e reports that the prompt violates their policies.  This can happen even with very benign subject matter.  Right now I think the Dall_e engine built into the Edge browser gives better results than the API, but every once in a while this will produce a winner.


