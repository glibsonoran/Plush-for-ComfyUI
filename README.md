# Plush-for-ComfyUI
****
[**How to use Advanced Prompt Enhancer with OpenRouter**](How to connect to OpenRouter) and other remote AI services that are not preconfigured.
**You don't need a paid key to use Advanced Prompt Enhancer with Groq or open source LLM's.  You can use them for free (Groq requires a free key).  But if you want to use it with ChatGPT or Claude3 then:**
[**Click here if you need to setup your API key or an optional Open-source key in an Plush compatible evironment variable**](#requirements)

[**Prompt and Image examples from the Style_Prompt and Style_Prompt + OAI Dall-e3 node(s)**](#examples)
****
### Updates:
10/20/2024 @5:04pm PST *Version 1.21.19*
*   **New Utility nodes:**
    *   Type Converter:  Converts various data types from a string.  Also can cross-reference equivalent data type
    *   Image Mixer and Random Image Output: Two randomizing nodes that can randomly change the order of images or randomly present an image to another node from a group of images.
*   **Advance Prompt Enhancer (APE), Context Output**
    *   Allows chaining of successive prompt/response interactions between APE nodes, allow each successive node to be aware of earlier node's interactions i.e. Nodes linked in this manner share their context.
*   **Advanced Prompt Enhancer, the old Parameters node has been removed from the Suite**
*   **Help.json**
    *   Updated to reflect changes detailed above.
*************

10/16/2024 @9:24pm PST *Version 1.21.18.1*
*   **New node *Add Parameters*:**
    *   I've added another version of the node to add parameters called: *Add Parameters*.  This node takes text entered into a text area.  See the Example Workflow: .../Plush-for_ComfyUI/Example_Workflows/How_to_use_addParameters.png file for more details.
*   **New nodes: Random Output & Random Mixer**
    *   These nodes allow you to send output to different inputs randomly.  See the Example Workflow: .../Plush-for_ComfyUI/Example_Workflows/How_to_use_RandomizedUtilityNodes.png for more information.
*   **Help.json**
    *   Updated to reflect changes detailed above.
*************

10/14/2024 @5:28pm PST *Version 1.21.18*
*   **New node *Additional Parameter*:**
    *   This node will allow you to add inference parameters to Advanced Prompt Enhancer (APE) that don't appear in the UI.  Parameters like *top_p* and *response_format* can allow you to have more control over the inference process.  These nodes can be daisy chained which will allow you add mulitiple parameters per run.  Find a list of [OpenAI parameters](https://platform.openai.com/docs/api-reference/chat/create) and their information at the link.  You should check the documentation of whichever service and model you're using to make sure the parameters you want to use can be applied.  You can find a Workflow Example of how to use this new node and the *Extract JSON data* node mentioned below here: `.../custom_nodes/Plush-for-ComfyUI/Example_Workflows/How_to_use_additionalParameters.png`
*   **New node *Extract JSON data:**
    *   This node allows you extract data from a string JSON by specifying the keys you want to query.  It can be used in tandem with the new Additional Parameter node that adds: *response_format* or the Additional Parameter node's menu item: *OpenAI JSON Format* to force JSON output from your model.  This output can then be queried using the *Extract JSON data* node. You can find a Workflow Example of how to use this new node here: `.../custom_nodes/Plush-for-ComfyUI/Example_Workflows/How_to_use_additionalParameters.png`
*   **Help.json**
    *   Updated to reflect changes detailed above.
*************
9/28/2024 @12:03pm PST *Version 1.21.16*
*   **Update to Advanced Prompt Enhancer, bug fix user-entered models are now in the `opt_models.txt` file:**
    *   The file: *optional_models.txt* was being overwritten when updating the installation using the ComfyUI Manager, although it stayed intact when being updated by a standard git pull.  Since most people update using the manager, I've decided to use an untracked file: `opt_models.txt` that will now hold your user-entered model names.
    *   The new file `opt_models.txt` in the */Plush-for-ComfyUI* directory is created during first startup after updating or installing the Plush node suite.  It will not appear until after your first restarted of ComfyUI.
    *   The new file `opt_models.txt` has a comment header that explains how to enter the model names, and the Advanced Prompt Enhancer help output has information on the new field.
    *   The models extracted from this file and listed in the *Optional_model* field pull down will only be applied when using *AI_Services* that end in *(URL)*.  
*   **Help.json**
    *   Updated to reflect changes detailed above.
*************
9/26/2024 @9:31pm PST *Version 1.21.15*
*   **Update to Advanced Prompt Enhancer:**
    *   Added a new field: *Optional_model*.  The model names in this drop-down field are extacted from a user editable text file: `custom_nodes/Plush-for-ComfyUI/optional_models.txt`.  This is for AI Services, local or remote, that require a model name to be provided as part of the inference request.
    *   The file *optional_models.txt* has a comment header that explains how to enter the model names, and the Advanced Prompt Enhancer help output has information on the new field.
    *   The models listed in *Optional_model* field pull down will only be applied when useing *AI_Services* that end in *(URL)*.
    *   Fixed various bugs in Advanced Prompt Enhancer code.
*   **Help.json**
    *   Updated to reflect changes detailed above.
*************
9/18/2024 @2:19pm PST *Version 1.21.14*
*   **Update to Advanced Prompt Enhancer:**
    *   Fixed *Examples* input, which is now renamed *Examples_or_Context*. This input will now automatically parse delimited example or context input into the proper roles (User or Assistant) and present them to the LLM as required (use the node's *help* output for more details).  This allows you to provide examples of how you want your output to look, or to use *Few Shot Prompting*, or to keep context by including earlier input.
    *   Advanced Prompt Enhancer's model filter will now include OpenAI's new "o" series models if you qualify to use them in your API account: e.g. "o1".
    *   There's now a dedicated *Ollama (URL)* menu item in the *AI_Services* menu.
*   **Example_Workflows**
    *   Fixed some older examples, updated them to use the revised Advanced Prompt Enhancer.
    *   Added a new Example: *How_To_Use_Examples* to illustrate how to use the newly fixed and renamed *Examples_or_Context* input in Advanced Prompt Enhancer.
*   **Help.json**
    *   Updated to reflect changes detailed above.
*************
8/31/2024 @3:56pm PST *Version 1.21.13*
*   **Update to Advanced Prompt Enhancer:**
*   Added *Ollama_model* field.  If you're using Ollama with Advanced Prompt Enhancer you'll now be able to select your model from a drop down. In order for this to work correctly you'll have to have loaded Ollama with the models you intend to use before starting ComfyUI. (Display the node's *help* output for more information.) 
*************
8/13/2024 @4:58pm PST *Version 1.21.12*
*   **Update to Advanced Prompt Enhancer:**
*   Added *optional_local_model* field.  If your local LLM front end requires you to pass a model name (most don't), you can put it here.  This is primarily for *Ollama* users. 
*************
7/10/2024 @11:45am PST *No version change*  
*    **New Example Workflows Demonstrating Agentic Systems:**
*    New workflow: *AgentsExample.png* is a simple Refinement Cascade.  It takes a prompt and evaluates it as to how closely it followed the Instruction and then revises it to adhere to the Instruction more closely.
*    New workflow: *Agent-ImageEvaluator.png* Evaluates an image in regard to how closely it followed the prompt. It then produces a list of items to improve Prompt Adherence.
*    New workflow: *Mixture of Agents.png*  Uses a matrix of individual AI agents that collaborate and produce a consensus output to improve accuracy.
*************
7/6/2024 @1:37pm PST *Version 1.21.11*
*   **Updates to Advanced Prompt Enhancer and Style Prompt nodes**
*   Advanced Prompt Enhancer: Added *LM Studio* selection item to *AI Services* list.  This is an http: POST connection so you can use the chat completions (e.g.: `http://localhost:1234/v1/chat/completions`) url that appears on LM Studio's server screen when you first start the server.
*   Style Prompt: Can now use remote services and models from: ChatGPT, Grog and Anthropic, whereas before it could only use ChatGPT. As a result it now requires you to connect the new *AI Chooser* node in order to select the service and model you want to use.  The Example Workflows for Style Prompt have been updated to reflect the use of this new node.
**************
5/23/24 @08:16am PST *Version 1.21.10*
*   **Minor Update to Advanced Prompt Enhancer and Tagger**
*   Fixed data order issue to accomodate the changes to LM Studio's API
*   Revised Tagger so it doesn't add a period to the end of the processed text block
*****************
5/14/24 @04:27pm PST *Version 1.21.9*
*    **Minor Update to Advanced Prompt Enhancer and Style Prompt**
*    Added new AI Service: *http POST Simplified Data*.  This is an http POST type connection and requires the `v1/chat/completions` path in the url.  It can be used for local applications that can't handle the OpenAI standard nested data structure.  Instead this uses a flatter simplified data structure.  If you're unable to connect to your local application with the other AI Service methods,  especially if you're getting a server processing error in the 500 range,  this might work for you.
*    Removed the automatic selection of a vision capable model for ChatGPT when you pass it an image.  Both Style Prompt and Advanced Prompt Enhancer will now simply apply whatever model you select.  There are now three vision capable models in the ChatGPT model list and at this point it's better to let you select the one you want to use.  The model will now just report that it can't process your data if you try to send an image to a model that isn't vision capable.
***************
4/26/2024 @11:47am PST *Version 1.21.8*
*  **Advanced Prompt Enhancer, now supports Anthropic (Claude) and Groq connections'** 
   *    **Grog** is a free service that provides a remote inferencing platform for the latest high quality open-source models including the new *Llama 3* models (*llama3-70b* & *llama3-8b*) and *Mixtral-8x7b*.
   *    **Anthropic** is a paid service that provides a remote inferencing service for its 3 *Claude* models, which are considered to be roughly on par with *ChatGPT* models.
   *    Both services require that you obtain an API key from their website.  Plush requires the API key to be stored in an enviroment variable [(see instructions here)](#requirements)
*  **New node: 'Tagger'**   This node allows you to add tags to any text block.  Tags can be added to the beginning, middle or end of the text block.  This facilitates adding lora, weighted text, or other prompt specific tags to prompts generated by AI, but can also be used to tag any text output.
******************
4/14/2024 @02:01pm PST *Version 1.21.7*
*  **Advanced Prompt Enhancer, new connection type: 'OpenAI compatible http POST'**  A connection type that uses a web POST connection rather than the OpenAI API Object to connect to the LLM front-end's local server.
*  **Oobabooga API connection:**   This connection now automatically formats the URL to include the `/v1/chat/completions` path.  This connection now includes both the Instruction (role:system) and Prompt (role:user) in the user submission to get around the problem of Oobabooga ignoring the system instruction.
******************
3/24/2024 @12:16pm PST *Version 1.21.6*
*  **Advanced Prompt Enhancer, new connection type:** The Oobabooga Textgen Webui API has been broken for a couple of weeks which has been resulting in a 'None Type' errors.  To get around this issue I've added a new Oobabooga connection type *Oobabooga API-URL* that uses an http POST connection rather than the OpenAI API Object.  Select this at the *LLM* selection field and provide a url that includes the path: `/chat/completions`. For example a url for this type of connection would look like: `http://127.0.0.1:5000/v1/chat/completions`.  However when using this method of connection with Oobabooga TG it seems to only see the prompt and not the instruction or examples.
*  **You can use a key with open source LLM products:** You can define an key in an environment variable named: `LLM_KEY` if you want to use a key with an LLM front-end, API or other product.  While these products are usually free, some use keys for security and privacy.  If you want to use a key just create the env var with your key and it will automatically be applied to any connection with LLM products other than ChatGPT.  If you have an OpenAI ChatGPT key in its own env var this will be unaffected, it will be used separately when you choose the ChatGPT connection type.
******************
3/19/2024 @2:36PM PST *Version 1.21.5*
*  **Advanced Prompt Enhancer can now provide an image file as input to generate text from open source LLM's that have 'vision' capability.**  Many open source LLM front-ends are now incorporating 'vision' capable models that can interpret image files (e.g. Koboldcpp's latest update). Advanced Prompt Enhancer can now send image files to these open source models to include in the inferencing process.  You can provide instructions to the model through the 'Instruction' input as to how the model should interpret the image, and you can add additional elements to be included in the output through the 'prompt' input.
*  **Exif Wrangler will now extract GPS location data when available from JPG photographs.** 
******************
3/11/2024 @8:38PM PST *Version 1.21.4*
* **Bug fix for users that don't have an OpenAI API key**: Advanced Prompt Enhancer was throwing an error because it couldn't create its list of models unless the user had a valid paid OpenAI API key.  This error has been fixed and users without a key should be able to use their Open Source LLM's without issue.
******************
3/7/2024 @11:00PM PST *Version 1.21.3*
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
##  Your OpenAI API or Open Source Key [optional] (Not required for Exif Wrangler, switch nodes, or Advanced Prompt Enhancer when used with open-source LLM's):
* For the Style Prompt and Dall-e nodes, you’ll need a valid API key from OpenAI.
* For Advanced Prompt Enhancer, you'll need a valid API key if you're going to use it with ChatGPT, Anthropic or Groq models,  if you're only using it with open-source LLM's, you won't need one.
* Some Open-source products use a free key for security and privacy so you have the option to create one if you want.  Most of these products don't use a key, so don't worry if you don't have one.
* The OpenAI API & Anthropic keys require a paid account, if you want to use an Open-source key they are typically free.  The Groq API key is free also.   Generate the key from their website.
  
  **The follwing table lists the Enviroment Variables that Plush recognizes and how the API keys they contain are applied.**

  | Enviroment Variable | Anthropic | Groq | OpenAI ChatGPT | Open Source (e.g. Tabby API) |
  | :------ | :------: | :------: | :------: | :------: |
  | `OAI_KEY` |      |      |   **X**   |      |
  | `OPENAI_API_KEY` |      |      |   **X**   |      |
  | `LLM_KEY` |      |      |      |   **X**   |
  | `GROQ_API_KEY` |      |   **X**   |      |      |
  | `ANTHROPIC_API_KEY` |   **X**   |      |      |      |
  -------------------------------------
   *  **You should set a reasonable $dollar limit on the usage of your OpenAI API key to prevent a large bill if the key is compromised.**  You can do this in the account setting at the OpenAI website.
     
     ![DollarLimitCGPT](https://github.com/glibsonoran/Plush-for-ComfyUI/assets/31249593/d26fd380-b3ee-4aee-bf02-393f7485fb50)
  *********
   *  **Installation and usage of Plush-for-ComfyUI constitutes your acceptance of responsibility for any losses due to a compromised key.**  Plush-for-Comfy uses the OpenAI recommended security for storing your key (an Environment Variable) for your safety.

   *  With regard to ChatGPT, you can choose to create a new Environment Variable specific to Plush called: `OAI_KEY` and store the API key there, or if you prefer, you can use the OpenAI standard environment variable: `OPENAI_API_KEY`.
     
   *  Optionally you can create a key for Open-source products in the Environment Variable `LLM_KEY`.  While Open-source products are generally free to use, some use a key for security and privacy. 

   *  Plush looks for the 'OAI_KEY' variable first, if it's not there it looks for 'OPENAI_API_KEY'.  Using the 'OAI_KEY' variable will allow you to generate a separate key for Plush and track those costs separately if your other OpenAI API apps are using the standard variable.  Either way you'll need to have at least one of the two enviroment variables defined with a valid active key if you want to use ChatGPT as an inference engine.  For Open-source products, once you populate 'LLM_KEY' with your key value, it will automatically be applied to all non-ChatGPT connections.  Enviroment Variables for other supported AI services are shown in the table above.

   *  If you need to make a new environment variable, see the following instructions on how to create it and set its value to your API key:

##  How to Setup Your Environment Variables

An environment variable is a variable that is set on your operating system, rather than within your application. It consists of a name and value. For a paid ChatGPT key you can set the name of the variable to: `OAI_KEY` or `OPENAI_API_KEY`. If you're using an Open-source product that requires or can use a key (most do not), use the environment variable: `LLM_KEY`. Refer to the table above for other services. The example below only refers to 'OAI_KEY' but you can substitute the environment variable name that applies to you per the [table](https://github.com/glibsonoran/Plush-for-ComfyUI/blob/main/README.md#your-openai-api-or-open-source-key-optional-not-required-for-exif-wrangler-switch-nodes-or-advanced-prompt-enhancer-when-used-with-open-source-llms) above. 

Note that after you set your Enviroment Variable, you will have to **reboot your machine** in order for it to take effect.
##  Windows Set-up

**Option 1**: Set your ‘OAI_KEY’ Environment Variable via the cmd prompt with admin privileges.


Run the following in the cmd prompt, replacing <yourkey> with your API key:

`setx OAI_KEY (your key)`

You can validate that this variable has been set by opening a new cmd prompt window and typing in 

`echo %OAI_KEY%`

**Option 2**: Set your ‘OAI_KEY’ Environment Variable through the Control Panel

1. Open System properties by right clicking the windows startup button and selecting "System". Then select Advanced system settings

2. Select Environment Variables...

3. Select New… from the User variables section(top). Add your name/key value pair ('OAI_KEY/'jk-####'), replacing (yourkey) with your API key.

Variable name: OAI_KEY
Variable value: (yourkey)

In either case if you're having trouble and getting an invalid key response, per the instructions above, please try rebooting your machine.


##  Linux / MacOS Set-up

**Option 1**: Set your ‘OAI_KEY’ Environment Variable using zsh

1. Run the following command in your terminal, replacing yourkey with your API key. 

`echo "export OAI_KEY=(yourkey)" >> ~/.zshrc`

2. Update the shell with the new variable:

`source ~/.zshrc`

3. Confirm that you have set your environment variable using the following command. 

`echo $OAI_KEY`

The value of your API key will be the resulting output.


**Option 2**: Set your ‘OAI_KEY’ Environment Variable using bash

Follow the directions in Option 1, replacing .zshrc with .bash_profile.

 You’re all set! Now Plush can load your key when you startup ComfyUI.

******************************
## How to connect to OpenRouter 

You can connect to remote AI serivces that are not preconfigured in Advanced Prompt Enhancer (APE) by following the steps below:

1) Obtain an API key from the service you want to use, you many have to pay for this.  

2) If you know how to create environment variables, create one named: `LLM_KEY` and enter your API key.  If you don't know how to create an enviroment variable there are instructions [here](#requirements)

3) Open the text file: `.../ComfyUI/custom nodes/Plush-for-ComfyUI/opt_models.txt` Follow the instructions in the comment header and enter the names of the AI models you want to use.  Make sure you use the exact names the service requires for their API. They should have a web page that shows these names.  Save the text file.

4) Start ComfyUI.  In the APE node you can setup your connection to the service two different ways: 

    - By choosing: *OpenAI API Connection (URL)* in the AI_service pull down
    - By choosing: *Direct Web Connection (URL)* in the AI_service pull down

5) Select the model you want to use in the *Optional_models* pull down, these will be the models you entered in the text file in step 3.

6) Enter the url for the site you want to connect to in the 'LLM_URL' field.  The *OpenAI API Connection* method will require a url that has a '/v1' path.  The *Direct Web Connection* method will require a url that has a '/v1/chat/completions' path.  The following are examples for OpenRouter:

    - **OpenAI API Connection:** LLM_URL = 'https://openrouter.ai/api/v1'
    - **Direct Web Connection:** LLM_URL = 'https://openrouter.ai/api/v1/chat/completions' 

7) Connect a *ShowText|pysssss* node to the *troublshooting* output of the APE node, then go ahead and run your workflow.  If you have any issues the troubleshooting output should help you diagnose the problem.

************************************
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



