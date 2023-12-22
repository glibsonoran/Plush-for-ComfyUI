# Plush-for-ComfyUI
****
### Plush contains two OpenAI enabled nodes: 


**Style Prompt**: Takes your prompt and the art style you specify and generates a prompt from ChatGPT3 or 4 that Stable Diffusion can use to generate an image in that style.


**OAI Dall_e 3**:  Takes your prompt and parameters and produces a Dall_e3 image in ComfyUI.
*****
### Installation:

Install through the ComfyUI manager:  
  *  Start the Manager.
  *  Click on Install Custom Nodes
  *  Search for "Plush"
  *  Find "Plush-for-ComfyUI"
  *  Click Install.
************
Manual install:

Follow the link to the [Plush for ComfyUI Github page](https://github.com/glibsonoran/Plush-for-ComfyUI "Plush Github Page") if you're not already here.  

Click on the green Code button at the top right of the page.  When the tab drops down, click to the right of the url to copy it.

![alt text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/c9277b63-7307-4fbd-86e6-b772db4165af "Copy Url")

Then navigate, in the command window on your computer, to the **ComfyUI/custom_nodes** folder and enter the command by typing *git clone* and pasting the url you copied after it:
```
git clone https://github.com/glibsonoran/Plush-for-ComfyUI.git.

cd Plush-for-ComfyUI/

python -m pip install -r requirements.txt
```
>
****
### Requirements: 
##  Your OpenAI API Key:
* You’ll need a valid API key from OpenAI, which requires a paid account.  Generate the key from their website.
   *  **You should set a reasonable $dollar limit on the usage of your OpenAI API key to prevent a large bill if the key is compromised.**  You can do this in the account setting at the OpenAI website.
   ********
     ![DollarLimitCGPT](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/d26fd380-b3ee-4aee-bf02-393f7485fb50)
  *********
   *  **Installation and usage of Plush-for-ComfyUI constitutes your acceptance of responsibility for any losses due to a comprimised key.**  Plush-for-Comfy uses the OpenAI recommended security for storing your key (an Environment Variable) for your safety.
   *  You'll need to create a new Environment Variable to store the API key called: "OAI_KEY".  See the following instructions on creating the Environment Variable and setting its value        to your API key:

##  Use Environment Variables in place of your API key

An environment variable is a variable that is set on your operating system, rather than within your application. It consists of a name and value.We recommend that you set the name of the variable to OAI_KEY.  Note that after you set your Enviroment Variable, you will have to **reboot your machine** in order for it to take effect.
##  Windows Set-up

**Option 1**: Set your ‘OAI_KEY’ Environment Variable via the cmd prompt

Run the following in the cmd prompt, replacing <yourkey> with your API key:

```setx OAI_KEY “(your key)"```

You can validate that this variable has been set by opening a new cmd prompt window and typing in 

```echo %OAI_KEY%```

**Option 2**: Set your ‘OAI_KEY’ Environment Variable through the Control Panel

1. Open System properties by right clicking the windows startup button and selecting "System". Then select Advanced system settings

2. Select Environment Variables...

3. Select New… from the User variables section(top). Add your name/key value pair ('OAI_KEY/'jk-####'), replacing (yourkey) with your API key.

Variable name: OAI_KEY
Variable value: (yourkey)


##  Linux / MacOS Set-up

**Option 1**: Set your ‘OAI_KEY’ Environment Variable using zsh

1. Run the following command in your terminal, replacing yourkey with your API key. 

```echo "export OAI_KEY='yourkey'" >> ~/.zshrc```

2. Update the shell with the new variable:

```source ~/.zshrc```

3. Confirm that you have set your environment variable using the following command. 

```echo $OPENAI_API_KEY```

The value of your API key will be the resulting output.


**Option 2**: Set your ‘OAI_KEY’ Environment Variable using bash

Follow the directions in Option 1, replacing .zshrc with .bash_profile.

###  NOTE: Initially Plush-for-ComfyUI used the config.json file to store the OpenAI API key.  I will be transitioning all installations over to Enviroment Variable only with a deadline sometime in Jan. 2024.

 You’re all set! Now Plush can load your key when you startup ComfyUI.

******************************
###  More Requirements:

* You’ll need to have ComfyUI installed and it’s recommended that you have the Base and Refiner SDXL models as those are the models this node was designed for and tested on.  The Style Prompt node relies on having a model that has a broad set of images that have been carefully labeled with art style and artist.  I think the SDXL base and refiner are best suited to this.

* Plush requires the OpenAI Python library version 1.3.5 or later.  This should be handled by the "requirements.txt" file included in this package. If you have used earlier nodes that communicate with ChatGPT you may have an early version of this library.  If for some reason installing *Plush* doesn't upgrade this library, you can upgrade it manually by typing the command:
* >pip install openai --upgrade

  in a directory or virtual environment *where it will be applied to the installation of Python that ComfyUI is using.*  

* Be aware that in some instances the new OpenAI API is not backward compatible and apps that use the older library may break after this upgrade.
****

### Usage:

I reccommend starting off using Style Prompt with a full SDXL Base and Refiner model, these models have the depth and labeling of art styles and artists that works well with this node.  You'll find a Workflow image in the **custom_nodes/Plush-for-ComfyUI/Example_workflows** directory.  If you want a quick setup, drag this image directly onto your ComfyUI workspace in your browser, it will automatically load the graph.  Style Prompt doesn't work well with quick print/turbo workflows like LCM that rely on low cfg values.  Stable Diffusion has to implement the whole (or most) of a fairly detailed prompt in order to get the right style effect, and these workflows just don't pick everything up.  At least initially I recommend you use the more basic SDXL workflows and models

***
![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/32debc23-3da0-4a65-a3f2-6cf803bde741 "Style Prompt Node")

#### Style Prompt:


**Inputs**:

*prompt*:  Your prompt, it doesn’t need to be wordy or complex, simpler prompts work better.


*example (optional)*:  A text example of how you want ChatGPT’s prompt to look.  There’s a default example in Style Prompt that works well, but you can override it if you like by using this input.  Examples are mostly for writing style, it doesn’t matter if they pertain to the same subject as your prompt.


**Outputs**: 

*CGPTprompt*:  The prompt ChatGPT generates for your image, this should connect to the CLIP node. Alternatively you can have a text display node either in-line between Style Prompt and the CLIP node, or as a separate branch off this output.  In either case a text display node will show you the ChatGPT generated prompt.

*CGPTInstruction (optional)*: This will show you the instruction that was sent to ChatGPT along with the prompt.  The instruction tells ChatGPT how to treat the prompt.  It’s pretty much the same every time so typically it’s not worth hooking up after you’ve seen a couple.

*Style Info (optional)*:  If the style_info UI control is set to “true”, this will output a brief backgrounder on the art style you’ve chosen:  This will display important characteristics of the style, its history and the names of some artists who have been influential in that style.  This will require connecting it to a text display box if you’re going to use it.


**UI inputs**:

*GPTModel (default gpt-4)*:  The ChatGPT model that’s going to generate the prompt. GPT-4 works better than GPT-3.5 turbo, but 3.5 is slightly cheaper to use.  The new GPT-4Turbo is now included as: "gpt-4-1106-preview"

*Creative_lattitude (default 0.7)*:  This is very similar to cfg in the KSampler.  It’s how much freedom the AI model has to creatively interpret your prompt, example and instruction.  Small numbers make the model stick closely to your input, larger ones give it more freedom to improvise.  The actual range is from 0.1 to 2.0, but I’ve found that anything above 1.1 or 1.2 is just disjointed word salad. So I’ve limited the range to 1.2, and even then I don’t go above 0.9.

*Tokens (default 2000)*:  A limit on how many tokens ChatGPT can use in providing your prompt.  Paid use of the API is based on the number of tokens used.  This isn’t how many ChatGPT *will* use, it’s a limit on how many it *can* use.  If you want to strictly control costs you can play around with the minimum number of tokens that will get you a good prompt.  I just leave it at 2000.

*Style (default Photograph)*:  This is the heart of Style Prompt.  I’ve included a list of dozens of art styles to choose from and my instructions tell ChatGPT to build the prompt in a way that pertains to the chosen style.  It’s ChatGPT’s interpretation of the art style, knowledge of artists that work in that style, and what descriptive elements best relate to that style that makes the node effective at depicting the various styles.

*Artist (default 1, range: 0 - 3)*: Whether to include a “style of” statement with the name of 1 to 3 artist(s) that exemplify the style you’ve chosen.  Style Prompt is better at depicting the chosen style if this is set to 1 or greater.  If you don't want to include an artist, set this to 0.

*Max_elements (default 10)*:  The maximum number of descriptive elements for ChatGPT to include in its generated prompt.  Stable Diffusion gives the highest weighting to text at the beginning of the prompt, and the weighting falls off from there.  There’s definitely a point where long wordy SD prompts result in diminishing returns.  This input lets you limit the length of your prompt.  The range here is from 3 to 20.  I think 6 to 10 works about the best.

*Style_info (default false)*:  If this is set to true, Style Prompt will send a second request to ChatGPT to provide a description of the chosen style, historical information about it, and information on some of the most influential artists in that style.  



### Examples:
![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/aec4ce84-e5a8-4a43-966c-23b65262fb18 "Fish Eye Lens Photograph")

Style: Fish Eye Lens Photograph

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/cdadbd7d-9f48-4f7b-bafa-c9ce69b0f0ea "High Key Photograph")

Style: High Key Photograph

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/0c59872e-9c6b-489c-88b7-9e379a21e40a "Semi Realistic Cyberpunk")

Style: Semi Realistic Cyberpunk

![Alt Text1](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/15d48cf0-32de-4b82-932b-365094e36254 "Low Key Photograph")

Style: Low Key Photograph

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/e0246439-7c56-4498-ab4b-2358baa96bc0 "Art Deco")

Style: Art Deco

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/e51f9c1f-07a0-4410-8e09-c04019c2714d "Origami Art")

Style: Origami

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/cee0724c-cabd-425c-b3df-1361f37285f9 "Fashion Art")

Style: Fashion Art

****
### OAI Dall_e Image

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/03ecaa31-6a2c-4426-baa1-5dad5b41b36e "OAI Dall_e Node")



I’m not going to go into detail about this node.  The main thing is that it takes your prompt and outputs an image.  Right now it’s only setup to use dall_e3 as the required input values are too different for me to include dall_e2.  Dalle_e3 produces better images so I just didn’t think accommodating Dall_e2 was worth it. 

You should be aware that in the API implementation Dall_e completely rewrites your prompt in an attempt to control misuse.  The text of that rewritten prompt is what is produced by the Dall_e_prompt output in this node. This can create some odd results, and some prompts will generate a ComfyUI error as Dall_e reports that the prompt violates their policies.  This can happen even with very benign subject matter.  Right now I think the Dall_e engine built into the Edge browser gives better results than the API, but every once in a while this will produce a winner.


