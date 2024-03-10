# Plush-for-ComfyUI
****
[**Click here if you need to setup your OpenAI key in an Plush compatible evironment variable**](#requirements)

[**Prompt and Image examples from the Style_Prompt and Style_Prompt + OAI Dall-e3 node(s)**](#examples)
****
### Updates:
3/7/2023 @11:00PM PST *Version 1.21.3*
*    **New node: Advanced Prompt Enhancer, can use open-source LLM's**:  Uses Open-Source LLM (via an front-end app like *LM Studio*); or ChatGPT/ChatGPT-Vision to produce a prompt or other generated text from your: Instruction, Prompt, Example(s), Image or any combination of these. The open-source connection works through the OpenAI API without requiring an OpenAI account or key.  This provides a connection to LLM front-ends like *LM Studio*.  So far this has just been tested with *LM Studio* but should work with any LLM front-end that can accept input from the OpenAI API Object.  Using the ChatGPT models (including GPT-vision models) does require an API key. See the help file, available from the node's *help* output, for more details.
*    **This node was largely spec'd by Alessandro Perilli, click to see his [amazing ComfyUI all encompassing 'AP Workflow'](https://perilli.com/ai/comfyui/)**.
********************
2/19/24 @12:20PM PST *Version 1.20.3*
*   **OAI Dall-e3 node can now create batches of up to 8 images**
*   **OAI Dalle-e3 node now has a mock 'seed' value**:  While the seed value does not affect a latent or the image, it does allow the Dall-e node to run automatically with every Queue if set to: "randomize" or "increment/decrement".  If you want to have the default behavior where it only runs once per prompt, or if its setting are changed, set seed to 'fixed'.
********************
2/13/24 @4:40PM PST *Version 1.20*
*   **A new node that doesn't require the OpenAI API key, Plush Exif Wrangler:**  Exif Wrangler will extract Exif and/or AI generation workflow metadata from .jpg (.jpeg) and .png images.  .jpg photographs can be queried for their camera settings.  ComfyUI's .png files will yield certain values from their workflow including the prompt, model, seed etc.  Images from other AI generators may or may not yield data depending on where they store their metadata. For instance Auto 1111 .jpg's will yield their workflow information that's stored in their Exif comment.
*   **Exif Wrangler offers the option to save your AI Generation or Exif information to a file:** The file is stored in *..ComfyUI/output/PlushFiles*, the directory is created when you save your first Exif file.
*   **The Exif Wrangler node can be used without an OpenAI API key.**
*   **Plush nodes now include a *troubleshooting* output:**  This output will display INFO/WARNING/ERROR data that's caputured by Plush's logging function in a text display node.  Logging begins when you press the *Queue* button and pertains only to the individual Plush node.
*   **Plush now creates and maintains a log file:** *Plush-Events.log*.  It's in the *..ComfyUI/custom_nodes/Plush-for-ComfyUI/logs* directory.  The directory will be created when you first run this version.
*   **A new set of example workflows replaces the older ones in the Example Workflows directory, plus the addition of an example workflow for Exif Wrangler.**  
*******************
1/21/24 @7:09PM PST *Revert some of the changes in Version 1.16*
*  **The addition of 2 sets of examples to facilitate "few shot" learning was too confusing for ChatGPT, I had to revert back to no examples.**  Few shot learning consists of providing the LLM an instruction and several examples of the desired response.  But style prompt's instruction is too complex to mix with examples. When I tried that ChatGPT completely lost the plot. So this has been reverted to no examples.  
*******************
1/16/24 @1:00PM PST *Version 1.16*
*  **Version 1.16, Fixes to unconnected inputs and the "undefined" values they generate, and an additional set of examples to send in the prompt request to ChatGPT.  This will facilitate "few shot" learning for generating prompts**
*  **A new example workflow has been addded**: *StylePromptBaseOnly.png* in the *Example_Workflows* directory, it's a StylePrompt workflow that uses one KSampler, no Refiner.
********************
1/8/24 @6:00pm PST *Version 1.15*
*  **Version 1.15, adds a new UI field: 'prompt_style' and a 'Help' output to the style_prompt node**
* **prompt_style**: lets you choose between:
    * **Narrative**: A prompt style that is long form creative writing with grammatically correct sentences.  This is the preferred form for Dall_e
    * **Tags**: A prompt style that is terse, a stripped down list of visual elements without conjunctions or grammatical phrasing.  This is the preferred form for Stable Diffusion and Midjourney.
*  **Help**: Attach a text input display node to get explanations of the various UI fields in style-prompt  
*********************
1/7/24 @4:07 PST
*  **Two new switch utility nodes added** Both switches allow you to use text primitives in their multiline configuration.  One is a 3 => 1 output switch for multiline text, the other a 3 => 1 output for multiline text and image.
*********
1/5/24 @12:02pm PST: *Version 1.10*
*  **Plush-for-ComfyUI will no longer load your API key from the .json file**  You must now store your OpenAI API key in an environment variable.  [See instructions below:](#use-environment-variables-in-place-of-your-api-key)
*  **A new example workflow .png has been added to the "Example Workflows" directory.** This workflow reflects the new features in the Style Prompt node.
**************
12/29/23 @4:24pm PST:
*  **New feature:**  Plush-for-ComfyUI style_prompt can now use image files to generate text prompts.  Image files can be used alone, or with a text prompt.  ChatGPT will interpret the image or image + prompt and generate a text prompt based on its evaluation of the input.  This is not exif extraction, it will not recreate the original prompt that produced the image, it's ChatGPT's interpretation of the image or image + prompt.
***************
***************
### Plush contains three OpenAI enabled nodes.  It also contains nodes that don't require OpenAI's API: Two multiline switches, the Advanced Prompt Enhancer and an Exif/AI metadata (prompt, seed, model, etc) extractor:

**Style Prompt**: Takes your: Text prompt, your image, or your text prompt and image, and the art style you specify and generates a prompt from ChatGPT3 or 4 that Stable Diffusion and/or Dall-e can use to generate an image in that style.

**Advanced Prompt Enhancer**: Take your: Prompt, Instruction, image, Examples and generates text output which can be a prompt or other output (e.g. caption).  This node can be used with certain Open Source LLM front-ends (e.g. LM Studio) or with ChatGPT.

**OAI Dall_e 3**:  Takes your prompt and parameters and produces a Dall_e3 image in ComfyUI.

**Switch Nodes**:  Allows you to handle multiple multiline text inputs

**Exif Wrangler**: Extracts Exif and/or AI generation workflow metadata from .jpg (.jpeg) and .png images.
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
##  Your OpenAI API Key (Not required for Exif Wrangler, switch nodes, or Advanced Prompt Enhancer when used with open-source LLM's):
* For the Style Prompt and Dall-e nodes, you’ll need a valid API key from OpenAI. For Advanced Prompt Enhancer, you'll need a valid OpenAI API key if you're going to use it with ChatGPT models,  if you're only using it with open-source LLM's, you won't need one. The OpenAI API key requires a paid account.  Generate the key from their website.
   *  **You should set a reasonable $dollar limit on the usage of your OpenAI API key to prevent a large bill if the key is compromised.**  You can do this in the account setting at the OpenAI website.
   ********
     ![DollarLimitCGPT](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/d26fd380-b3ee-4aee-bf02-393f7485fb50)
  *********
   *  **Installation and usage of Plush-for-ComfyUI constitutes your acceptance of responsibility for any losses due to a compromised key.**  Plush-for-Comfy uses the OpenAI recommended security for storing your key (an Environment Variable) for your safety.

   *  You can choose to create a new Environment Variable specific to Plush called: 'OAI_KEY' and store the API key there, or if you prefer, you can use the OpenAI standard environment variable: 'OPENAI_API_KEY'.

   *  Plush looks for the 'OAI_KEY' variable first, if it's not there it looks for 'OPENAI_API_KEY'.  Using the 'OAI_KEY' variable will allow you to generate a separate key for Plush and track those costs separately if your other OpenAI API apps are using the standard variable.  Either way you'll need to have at least one of the two enviroment variables defined with a valid active key.

   *  If you need to make a new environment variable, see the following instructions on how to create it and set its value to your API key:

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

```echo $OAI_KEY```

The value of your API key will be the resulting output.


**Option 2**: Set your ‘OAI_KEY’ Environment Variable using bash

Follow the directions in Option 1, replacing .zshrc with .bash_profile.

 You’re all set! Now Plush can load your key when you startup ComfyUI.

******************************
###  More Requirements:

* You’ll need to have ComfyUI installed and it’s recommended that you have the Base and Refiner SDXL models as those are the models this node was designed for and tested on, it also seems to work really well with the new [OpenDalle model](https://huggingface.co/dataautogpt3/OpenDalleV1.1).  The Style Prompt node relies on having a model that has a broad set of images that have been carefully labeled with art style and artist.  I think the SDXL base and refiner are best suited to this.

* Plush requires the OpenAI Python library version 1.3.5 or later.  This should be handled by the "requirements.txt" file included in this package. If you have used earlier nodes that communicate with ChatGPT you may have an early version of this library.  If for some reason installing *Plush* doesn't upgrade this library, you can upgrade it manually by typing the command:
  
*  ```pip install --upgrade openai```

  in a directory or virtual environment *where it will be applied to the installation of Python that ComfyUI is using.*  

* Be aware that in some instances the new OpenAI API is not backward compatible and apps that use the older library may break after this upgrade.
****

### Usage:

I reccommend starting off using Style Prompt with a full SDXL Base and Refiner model, these models have the depth and labeling of art styles and artists that works well with this node.  You'll find a Workflow image in the **custom_nodes/Plush-for-ComfyUI/Example_workflows** directory.  If you want a quick setup, drag this image directly onto your ComfyUI workspace in your browser, it will automatically load the graph.  The new [OpenDalle model](https://huggingface.co/dataautogpt3/OpenDalleV1.1) model is also reccomended. Style Prompt doesn't work well with quick print/turbo workflows like LCM that rely on low cfg values.  Stable Diffusion has to implement the whole (or most) of a fairly detailed prompt in order to get the right style effect, and these workflows just don't pick everything up.  At least initially I recommend you use the more basic SDXL workflows and models

New to Style Prompt is the ability to interpret images and convert them into Stable Diffusion prompts using the new ChatGPT vision model. *You will be using the "gpt-4-vision-preview" model if you decide to use an image in your input, regardless of your GPTmodel selection.  It's the only model that can handle image input*.  

You can use this feature to:
* get prompt ideas from an image you like 
* To iterate on an image theme by selecting differnt art styles to apply to the image intepretation 
* By adding your own text prompt to the image to compose an image similar to your input image but with added visual elements.
* Or, of course, any other creative process you can think of

***
![StylePrompt](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/fc601470-5b4a-4332-ae55-434cd7b76a0d)
=======

#### Style Prompt:

**Inputs**:

*prompt*:  Your prompt, it doesn’t need to be wordy or complex, simpler prompts work better.

*image (optional)*:  Attach a "load image" or other node with an image output here.  The image will be interpreted by ChatGPT and formulated into a prompt for Stable Diffusion.  You can include an image alone, or an image + prompt. In the latter case both the prompt and image will be interprted by ChatGPT. When an image is included for interpretation, Style Prompt will automatically use the OpenAI "Vision" model (gpt-4-vision-preview) instead of the model selected in the "GPTmodel" field. This is because it's the only ChatGPT model that will accept image input.

*example (optional)*:  A text example of how you want ChatGPT’s prompt to look.  There’s a default example in Style Prompt that works well, but you can override it if you like by using this input.  Examples are mostly for writing style, it doesn’t matter if they pertain to the same subject as your prompt.
***********************

**Outputs**: 

*CGPTprompt*:  The prompt ChatGPT generates for your image, this should connect to the CLIP node. Alternatively you can have a text display node either in-line between Style Prompt and the CLIP node, or as a separate branch off this output.  In either case a text display node will show you the ChatGPT generated prompt.

*CGPTInstruction (optional)*: This will show you the instruction that was sent to ChatGPT along with the prompt.  The instruction tells ChatGPT how to treat the prompt.  It’s pretty much the same every time so typically it’s not worth hooking up after you’ve seen a couple.

*Style Info (optional)*:  If the style_info UI control is set to “true”, this will output a brief backgrounder on the art style you’ve chosen:  This will display important characteristics of the style, its history and the names of some artists who have been influential in that style.  This will require connecting it to a text display box if you’re going to use it.

*Help*:  Hook up a text display node to this output and press the Queue button to see a brief help file that explains the functions of the UI Input elements.
****************

**UI inputs**:

*GPTModel (default gpt-4)*:  The ChatGPT model that’s going to generate the prompt. GPT-4 works better than GPT-3.5 turbo, but 3.5 is slightly cheaper to use.  The new GPT-4Turbo is now included.

*Creative_lattitude (default 0.7)*:  This is very similar to cfg in the KSampler.  It’s how much freedom the AI model has to creatively interpret your prompt, example and instruction.  Small numbers make the model stick closely to your input, larger ones give it more freedom to improvise.  The actual range is from 0.1 to 2.0, but I’ve found that anything above 1.1 or 1.2 is just disjointed word salad. So I’ve limited the range to 1.2, and even then I don’t go above 0.9.

*Tokens (default 500)*:  A limit on how many tokens ChatGPT can use in providing your prompt.  Paid use of the API is based on the number of tokens used.  This isn’t how many ChatGPT *will* use, it’s a limit on how many it *can* use.  If you want to strictly control costs you can play around with the minimum number of tokens that will get you a good prompt.  I just leave it at 500.

*Style (default Photograph)*:  This is the heart of Style Prompt.  I’ve included a list of dozens of art styles to choose from and my instructions tell ChatGPT to build the prompt in a way that pertains to the chosen style.  It’s ChatGPT’s interpretation of the art style, knowledge of artists that work in that style, and what descriptive elements best relate to that style that makes the node effective at depicting the various styles.

*Artist (default 1, range: 0 - 3)*: Whether to include a “style of” statement with the name of 1 to 3 artist(s) that exemplify the style you’ve chosen.  Style Prompt is better at depicting the chosen style if this is set to 1 or greater.  If you don't want to include an artist, set this to 0.

*prompt_style (default, Tags)*:  Let's you choose between two types of prompts: **Narrative**: A prompt style that is long form creative writing with grammatically correct sentences.  This is the preferred form for Dall_e. **Tags**: A prompt style that is terse, a stripped down list of visual elements without conjunctions or grammatical phrasing.  This is the preferred form for Stable Diffusion and Midjourney. 

*Max_elements (default 10)*:  The maximum number of descriptive elements for ChatGPT to include in its generated prompt.  Stable Diffusion gives the highest weighting to text at the beginning of the prompt, and the weighting falls off from there.  There’s definitely a point where long wordy SD prompts result in diminishing returns.  This input lets you limit the length of your prompt.  The range here is from 3 to 25.  I think 6 to 10 works about the best.

*Style_info (default false)*:  If this is set to true, Style Prompt will send a second request to ChatGPT to provide a description of the chosen style, historical information about it, and information on some of the most influential artists in that style.  



### Examples:
![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/aec4ce84-e5a8-4a43-966c-23b65262fb18 "Fish Eye Lens Photograph")

Prompt: Fish-Eye lens Photograph of a joyful young woman on a bustling downtown street, her smile amplified by the distorted perspective, skyscrapers curving around her in a surreal fishbowl effect, their windows reflecting the radiant midday sun, the surrounding crowd and traffic appearing as miniature figures in the margins, parked cars stretched and skewed into bizarre shapes, the blue sky overhead warped into a swirling dome, style of Justin Quinnell.

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/cdadbd7d-9f48-4f7b-bafa-c9ce69b0f0ea "High Key Photograph")

Prompt: High Key Photography of a noir-era actress, vividly red lipstick, sparkling diamond jewelry, soft-focus background, luxurious fur stole, pearlescent lighting effects, dramatic high-contrast shadows, and mirrored reflections, style of Terry O'Neill.

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/d12a36c7-6fc8-4d36-997e-f5304855fe16 "Digital Art")

Prompt: Digital Art, female portrait, abstract elements, red hair, polka dots, vibrant colors, contrast, geometric shapes, surrealism, bold makeup, dripping paint effect, large eyes, stylized features, style of Patrice Murciano, style of Aya Kato.

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/2a4954b8-6ea7-44b5-a6e3-67baf821d227 "Fantasy Art")

Prompt: Fantasy Art of a radiant young woman, her eyes glowing with an ethereal light, clad in a cloak of starlight, amidst a sprawling urban jungle, buildings bathed in the soft hues of twilight, with the stylized graffiti murals pulsating with arcane energy, under the watchful gaze of celestial constellations, style of Yoshitaka Amano.

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/27f2bb3d-fb1d-47de-83c6-94a53e3ebca1 "Chiaroscuro Art")

(Dall-e3 node) Prompt: Chiaroscuro Art: female warrior, profile view, low key lighting, contrast of shadow and light, detailed battle dress, animal skins, flowing black hair, blood smeared face, victory shine, defiant wind, exhalation pose, dark stormy sky, distant lightning, triumphant spear thrust, high heels, leather wristbands, feathered necklace, steel breastplate, war paint stripes, rusted spear, cracked shield, muddy battlefield, fallen enemies, style of Mario Testino. 

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/b103d32e-8ffe-4aa9-bb25-355179263b88 "Low Key Photograph")

Prompt: Low Key Photograph of a young woman, her features highlighted by a single, dramatic light source, cradling a small dog in her arms, the dog's coat a play of shadow and sheen, against a backdrop of deep, impenetrable shadows, the surrounding space filled with soft whispers of darkness, the barest hint of a window barely discernible in the background, the light creating a stark contrast between subject and surrounding, style of Bill Henson.

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/33c2f9e1-6c10-4d40-9782-dfcde2432564 "High Key Photograph")

(Dall-e3 node) Prompt: High Key Photograph of a sun-bleached Sonoran desert landscape, the towering silhouette of a saguaro cactus against a bright, cloudless sky, crescent shaped sand dunes under harsh midday illumination, distant mountains reduced to ghostly outlines, the play of light and shadow accentuating the textures of the desert, each grain of sand aglow, all bathed in an intense, blinding light, style of Michael Frye

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/615ca38c-ecc0-4fd0-8897-ba87bde2fa9b "Fashion Sketch")

Prompt: Fashion Sketch of a statuesque model draped in a flowing grey dress, adorned with vibrant yellow accents, posed against a minimalist white background, with sharp, angular lines defining her silhouette, a dramatic contrast of shadow and light to highlight the fabric's texture, her gaze focused and intense, emanating an air of sophistication, her enigmatic smile hinting at a story untold, and a yellow hat as the final touch, style of Hayden Williams.

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/88ad1981-82ea-45e7-9bd7-e5220f753b93 "Biomorphic Abstraction")

Prompt: Biomorphic Abstraction, surreal portrait, female figure, high-contrast, oversized eyes, glossy lips, polychromatic splashes, geometric shapes, dripping paint, monochromatic background, stylized features, sharp shadows, dynamic composition, style of Kandinsky, style of Joan Miró.

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/78478edd-890b-4df9-864c-258c3f3bd96a "Long Exposure Photograph")

(Dall-e3 node) Prompt: Long Exposure Photograph capturing a solitary blue sailboat with its sails fully unfurled, gliding over a smooth, glass-like sea under the ethereal glow of a full moon. The image is framed to emphasize the stark contrast between the deep, velvety blues of the night sea and the subtle, shimmering silver path created by the moonlight. The sailboat is positioned slightly off-center, sailing towards the right, inviting the viewer's gaze to follow its journey. The surrounding darkness envelops the scene, with the moon's reflection acting as the main source of illumination, creating a serene yet mysterious atmosphere. The composition is minimalist, focusing on the interplay of light and shadow, the texture of the sailboat against the liquid mirror of the sea, and the infinite horizon merging sea and sky into one. Style of Michael Kenna.

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/07f7618f-0639-4622-92d1-0e3a3dd980f0 "Art Deco")

Prompt: Art Deco of a poised young woman in a sleek, geometrically patterned dress, her sharp silhouette highlighted against a jeweled sunset, standing on the crest of a manicured grassy hill, her eyes glinting with reflected urban skylines composed of streamlined skyscrapers in the distance, her hands softly clutching an elegant sequined clutch and a feathered hat delicately perched on her bobbed hair, all under the radiant glow of a large, low-hanging moon, style of Tamara De Lempicka.

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/01bdfaaf-0c9b-452a-b67d-5b1379308b1a "Zulu Urban Art")

Prompt: Zulu Urban Art, detailed female portrait, half-shaved head with blonde hair, geometric patterns, bold contrasts, abstract shapes, vibrant colors, dripping paint, surreal composition, expressive eyes, red lips, polka dots, modern fashion, style of Kobra, Shepard Fairey.

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/e51f9c1f-07a0-4410-8e09-c04019c2714d "Origami Art")

Prompt: Origami of a poised young woman crafted from intricate, emerald-green folds standing tall on a textured, grassy hill, with a meticulously folded skyline of a bustling city in the distance, under a sweeping blue paper sky, the sun casting long, dramatic shadows, all bathed in soft, warm light, style of Robert J. Lang.

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/cee0724c-cabd-425c-b3df-1361f37285f9 "Fashion Art")

Prompt: Fashion Art of a sophisticated young woman standing central, adorned in an avant-garde, voluminous tulle gown cascading to the grassy hill under her feet, an ornate oversized feather hat on her head, peering into the distance with a mysterious, melancholic gaze, her body illuminated by the glowing moon, the sprawling city skyline serving as a contrasting backdrop, in the strikingly dramatic style of Alexander McQueen.


![SPImagePlusPrompt](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/9e960755-168e-4c4d-959a-e9d5e1cedb41 "Photograph, Image Plus Prompt")

Style: Photograh, Example of Image +Prompt Input

=======

****
### OAI Dall_e Image

![Alt Text](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/03ecaa31-6a2c-4426-baa1-5dad5b41b36e "OAI Dall_e Node")



I’m not going to go into detail about this node.  The main thing is that it takes your prompt and outputs an image.  Right now it’s only setup to use dall_e3 as the required input values are too different for me to include dall_e2.  Dalle_e3 produces better images so I just didn’t think accommodating Dall_e2 was worth it. 

You should be aware that in the API implementation Dall_e completely rewrites your prompt in an attempt to control misuse.  The text of that rewritten prompt is what is produced by the Dall_e_prompt output in this node. This can create some odd results, and some prompts will generate a ComfyUI error as Dall_e reports that the prompt violates their policies.  This can happen even with very benign subject matter.  Right now I think the Dall_e engine built into the Edge browser gives better results than the API, but every once in a while this will produce a winner.


=======



