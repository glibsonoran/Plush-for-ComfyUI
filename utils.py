# =======================
#  Standard Libraries
# =======================
from enum import Enum
from typing import Optional, Dict, Any, Sequence
from io import BytesIO
import base64
import subprocess
import sys
import importlib
import importlib.metadata

# =======================
#  Third-Party Libraries
# =======================
import requests
from requests.adapters import HTTPAdapter, Retry
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import numpy as np

# =======================
#  Local Modules
# =======================
try:
    from .mng_json import json_manager, TroubleSgltn
except ImportError:
    from mng_json import json_manager, TroubleSgltn

class ImageFormat(Enum):
    B64_IMAGE = "b64-image"  # Base64 encoded image (JPEG/PNG)
    BYTE_IMAGE = "byte-image"  # Raw byte image (JPEG/PNG)
    UNKNOWN = "unknown"  # Neither base64 nor raw image

class CommUtils:
    def __init__(self)->None:
        self.j_mngr = json_manager()        

    def is_lm_server_up(self, endpoint:str, comm_retries:int=2, timeout:int=4):  #should be util in api_requests.py
        session = requests.Session()
        retries = Retry(total=comm_retries, backoff_factor=0, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        try:
            response = session.head(endpoint, timeout=timeout)  # Use HEAD to minimize data transfer            
            if 200 <= response.status_code <= 300:
                self.write_url(endpoint) #Save url to a text file
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
    
    def get_data(self, endpoint:str="", timeout:int=8, retries:int=1, data_type:str="", headers:Optional[Dict[str,str]]=None )-> requests.Response | None:

        """
        Sends a GET request to the specified endpoint with configurable timeout, retry logic, and headers.

        Parameters:
            endpoint (str): The API endpoint URL to send the GET request to.
            timeout (int): The maximum number of seconds to wait for a response before timing out. Default is 8.
            retries (int): The number of times to retry the request in case of failure due to certain HTTP errors (500, 502, 503, 504). Default is 1.
            data_type (str): A descriptive label for the type of data being fetched, used for logging purposes.
            headers (Optional[Dict[str, str]]): A dictionary of additional HTTP headers to include in the request.

        Returns:
            requests.Response | None: The response object if the request is successful, or None if an error occurs.

        Raises:
            requests.RequestException: Handled internally. Logs an error message and returns None if a request failure occurs.

        Notes:
            - Implements automatic retry logic for transient server errors.
            - Logs a warning if the request fails, including the HTTP status code and error details.
        """


        session = requests.Session()
        gretries = Retry(total=retries, backoff_factor=0, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=gretries))
        stat_code = 0
        try:
            response = session.get(endpoint, timeout=timeout, headers=headers,)
            stat_code = response.status_code
            response.raise_for_status()  # Raises an HTTPError if the response status code indicates an error
            return response

        except requests.RequestException as e:
            self.j_mngr.log_events(f"Unable to fetch data for: {data_type}.  Server returned code: {stat_code}. Error: {e} ",
            TroubleSgltn.Severity.WARNING,
            True)
            return None
        

    def post_data(
        self,
        endpoint: str = "",
        timeout: int = 8,
        retries: int = 1,
        data_type: str = "",
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        show_errors: bool = True
        ) -> requests.Response | None:
        """
        Sends a POST request to the specified endpoint, supporting both JSON and form-encoded data.

        Parameters:
            endpoint (str): The API endpoint URL.
            timeout (int): Timeout duration in seconds (default: 8).
            retries (int): Number of retries on failure (default: 1).
            data_type (str): A label for logging purposes.
            json (Optional[Dict[str, Any]]): JSON payload (application/json).
            data (Optional[Dict[str, Any]]): Form-encoded data (application/x-www-form-urlencoded).
            headers (Optional[Dict[str, str]]): Additional headers.

        Returns:
            requests.Response | None: The response object if successful, otherwise None.

        Notes:
            - Implements automatic retry logic for transient server errors.
            - Logs a warning if the request fails, including the HTTP status code and error details.
        """
        session = requests.Session()
        gretries = Retry(total=retries, backoff_factor=0, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=gretries))

        stat_code = 0
        try:
            response = session.post(endpoint, timeout=timeout, headers=headers, json=json if json else None, data=data if data else None)
            stat_code = response.status_code
            response.raise_for_status()  # Raises an HTTPError if the response status code indicates an error
            return response

        except requests.RequestException as e:
            self.j_mngr.log_events(
                f"Unable to post data for: {data_type}. Server returned code: {stat_code}. Error: {e}",
                TroubleSgltn.Severity.WARNING,
                show_errors
            )
            return None
        
    def write_url(self, url:str) -> bool:
        # Save the current open source url for startup retrieval of models

        url_file = self.j_mngr.append_filename_to_path(self.j_mngr.script_dir, 'OpenSourceURL.txt')
        url_result = self.j_mngr.write_string_to_file(url, url_file)
        self._written_url = url
        self.j_mngr.log_events("Open source LLM URL saved to file.",
                                TroubleSgltn.Severity.INFO,
                                True)
        return url_result


class ImageUtils:
    def __init__(self):
        self.j_mngr = json_manager()
        #self.trbl = TroubleSgltn()

    def detect_image_format(self, image_data):
        """
        Detect whether the content is a base64-encoded image or raw byte array (image).
        
        Args:
            image_data (str or bytes): The image data to check.

        Returns:
            ImageFormat: Enum indicating 'b64-image', 'byte-image', or 'unknown'.
        """
        # Check if the content is a base64-encoded string
        if isinstance(image_data, str):
            try:
                # Attempt to decode the base64 string
                base64.b64decode(image_data, validate=True)
                return ImageFormat.B64_IMAGE
            except (ValueError, base64.binascii.Error) as e:
                self.j_mngr.log_events(f"Unable to decode encoded image string. Error: {e}",
                                       TroubleSgltn.Severity.ERROR,
                                       True)
                return ImageFormat.UNKNOWN  # Not a valid base64 string
        
        # Check if it's already raw bytes (regardless of format)
        elif isinstance(image_data, bytes):
            return ImageFormat.BYTE_IMAGE
        
        self.j_mngr.log_events("Image is in an Unknown format.  Unable to process image.",
                               TroubleSgltn.Severity.ERROR,
                               True)
        
        return ImageFormat.UNKNOWN


    def b64_to_tensor(self, b64_image: str) -> tuple[torch.Tensor,torch.Tensor]:

        """
        Converts a base64-encoded image to a torch.Tensor.

        Note: ComfyUI expects the image tensor in the [N, H, W, C] format.  
        For example with the shape torch.Size([1, 1024, 1024, 3])

        Args:
            b64_image (str): The b64 image to convert.

        Returns:
            torch.Tensor: an image Tensor.
        """    
        self.j_mngr.log_events("Converting b64 Image to Torch Tensor Image file",
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
        self.j_mngr.log_events("Converting Torch Tensor image to b64 Image file",
                          is_trouble=True)
        
        if tensor.is_cuda:  # Check if the tensor is on GPU
            tensor = tensor.cpu()  # Move tensor to CPU

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
    

    def tensor_to_bytes(self, tensor: torch.Tensor, ensure_alpha: bool = False, file_name:str ='image.png') -> BytesIO:
        """
        Converts a PyTorch tensor to a bytes object.

        Args:
            tensor (torch.Tensor): The image tensor to convert.
            ensure_alpha (bool): If True, ensures the image is in RGBA format.
            file_name (str): The name attribute of the byte file (for identification purposes).

        Returns:
            BytesIO: BytesIO object containing the image data.
        """

        if tensor.is_cuda:  # Check if the tensor is on GPU
            tensor = tensor.cpu()  # Move tensor to CPU

        # Convert tensor to PIL Image
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension if present

        # Convert tensor to numpy array and scale values properly
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
            # Assume values are in [0, 1] range
            img_array = (tensor.numpy() * 255).astype('uint8')
        else:
            # Already in uint8 format
            img_array = tensor.numpy()
        
        pil_image = Image.fromarray(img_array)

        if ensure_alpha:
            # Ensure RGBA format
            if pil_image.mode == "L":  # If grayscale, use the grayscale values as alpha
                pil_image = pil_image.convert("RGBA")
                pil_image.putalpha(pil_image)  # Use the grayscale values as the alpha channel
            else:
                pil_image = pil_image.convert("RGBA")  # Convert to RGBA if not already

        # Save PIL Image to a buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")  # Can change to JPEG if preferred
        buffer.seek(0)
        buffer.name = file_name  # Set the name of the byte file for easier identification

        return buffer   
     
    def bytes_to_tensor(self, image_data: bytes) -> torch.Tensor:
        """
        Converts binary image data (bytes) to a torch.Tensor in [N, H, W, C] format.
        Handles JPEG, PNG, and other formats.
        
        Args:
            image_data (bytes): The raw image bytes (binary data).
        
        Returns:
            torch.Tensor: The image tensor.
        """

        # Load the image data from bytes into a PIL Image
        image = Image.open(BytesIO(image_data))
        
        # Convert the image to RGBA format (or RGB if you prefer)
        image = image.convert("RGBA")
        
        # Convert the PIL image to a NumPy array and normalize pixel values
        image_np = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Split the image into RGB and Alpha channels
        rgb_np = image_np[..., :3]  # Extract RGB channels
        
        # Convert NumPy array to PyTorch tensor and ensure it's on CPU
        tensor_image = torch.from_numpy(rgb_np)
        if tensor_image.is_cuda:
            tensor_image = tensor_image.cpu()

        # Add batch dimension [N, H, W, C]
        tensor_image = tensor_image.unsqueeze(0)   
        
        return tensor_image
    
    def produce_images(self, response, response_key='data', field_name='b64_json', field2_name=""):
        """
        Processes an API response to extract base64-encoded images and convert them into PyTorch tensors.

        This function is designed to handle API responses with either shallow or nested JSON structures. 
        It extracts base64-encoded images or raw image byte data from the response, converts them into 
        PyTorch tensors, and concatenates them into a single tensor if multiple images are found.

        Args:
            response (dict or object): The API response, either a dictionary or an object (like DALL-E's ImagesResponse).
            response_key (str): The key in the response dictionary or object attribute that contains the list of items 
                                (default is 'data').
            field_name (str): The key used to access the base64-encoded image data in each item 
                            of the response (default is 'b64_json').
            field2_name (str, optional): An optional key for accessing a nested structure. If provided, 
                                        this key will be used to access a nested dictionary inside each 
                                        item before attempting to extract 'field_name'. Default is "" 
                                        (i.e., no nested structure).

        Returns:
            torch.Tensor or None: Returns a PyTorch tensor containing all the processed images. 
                                If no images are found, None is returned.
        """
        image_list = []

        # Function to extract and process images
        def extract_and_process_images(items):
            for index, item in enumerate(items):
                # Access nested field or image data
                if field2_name:
                    b64_image = item.get(field2_name, {}).get(field_name, None) if isinstance(item, dict) else getattr(item, field2_name, {}).get(field_name, None)
                else:
                    b64_image = item.get(field_name, None) if isinstance(item, dict) else getattr(item, field_name, None)
                    
                if b64_image:
                    image_format = ImageUtils.detect_image_format(self, b64_image)
                    if image_format == ImageFormat.B64_IMAGE:
                        image_tensor, _ = ImageUtils.b64_to_tensor(self, b64_image)
                        image_list.append(image_tensor)
                    elif image_format == ImageFormat.BYTE_IMAGE:
                        image_list.append(ImageUtils.bytes_to_tensor(self, b64_image))
                else:
                    self.j_mngr.log_events(f"No image found at index {index}")

        # Check if response is a dictionary
        if isinstance(response, dict):
            if response_key in response and isinstance(response[response_key], list):
                extract_and_process_images(response[response_key])
            else:
                self.j_mngr.log_events(f"No images found in the response under key '{response_key}'")

        # Check if response is an object with the attribute
        elif hasattr(response, response_key):
            items = getattr(response, response_key)
            if isinstance(items, list):
                extract_and_process_images(items)
            else:
                self.j_mngr.log_events(f"No images found in the response under key '{response_key}'")

        if image_list:
            if len(image_list) > 1:
                return torch.cat(image_list)
            else:
                return image_list[0].unsqueeze(0)

        self.j_mngr.log_events(f"No images found in the response under key '{response_key}'")
        return None
    
    def extract_batch_size(self, tensor: torch.Tensor) -> int:
        """
        Extracts the batch size (N) from a PyTorch tensor.
        
        Args:
            tensor (torch.Tensor): The image tensor in [N, H, W, C] format.

        Returns:
            int: The number of images (batch size).
        """

        if tensor.ndim != 4:
            raise ValueError(f"Expected a 4D tensor [N, H, W, C], but got shape {tensor.shape}")
        
        return tensor.shape[0]  # Return batch size (N)
        
    def pad_and_cat_images(self, image_list, dim=2, pad_value=0.0):
        """
        Pads image tensors (in [N, H, W, C] format) to the same height and width, then concatenates.

        Args:
            image_list (list of torch.Tensor): List of image tensors [N, H, W, C].
            dim (int): Dimension along which to concatenate (1=vertical, 2=horizontal).
            pad_value (float): Padding pixel value (default=0.0 for black padding).

        Returns:
            torch.Tensor: Concatenated image tensor.
        """
        # Ensure all images have the same batch size (N)
        assert all(img.shape[0] == image_list[0].shape[0] for img in image_list), "All images must have the same batch size"

        # Determine max height and width
        max_height = max(img.shape[1] for img in image_list)
        max_width = max(img.shape[2] for img in image_list)

        # Pad each image to match the max dimensions
        padded_images = []
        for img in image_list:
            _, h, w, _ = img.shape  # Ignore N and C dimensions
            pad_h = max_height - h
            pad_w = max_width - w

            # Padding format: (left, right, top, bottom)
            padded_img = F.pad(img, (0, 0, 0, pad_w, 0, pad_h), value=pad_value)
            padded_images.append(padded_img)

        return torch.cat(padded_images, dim=dim)
    
    def pad_images_to_batch(self, image_list, pad_value=0.0):
        """
        Pads a list of images ([N, H, W, C]) to match the largest image size, keeping batch format.
        Logs the original image sizes before padding.

        Args:
            image_list (list of torch.Tensor): List of image tensors [N, H, W, C].
            pad_value (float): Padding pixel value (default=0.0 for black padding).

        Returns:
            torch.Tensor: A properly padded batch tensor [N, max_H, max_W, C].
        """
        # Ensure batch size (N) is consistent
        assert all(img.shape[0] == image_list[0].shape[0] for img in image_list), "All images must have the same batch size"

        # Move images to CPU if needed
        image_list = [img.cpu() if img.is_cuda else img for img in image_list]        

        # Capture original sizes for logging
        image_size_list = [(img.shape[1], img.shape[2]) for img in image_list]  # [(H, W), ...]
        self.j_mngr.log_events(f"Padding and batching images of sizes: {image_size_list}", is_trouble=True)

        # Determine max height and width
        max_height = max(h for h, _ in image_size_list)
        max_width = max(w for _, w in image_size_list)

        # Pad each image to the max dimensions
        padded_images = []
        for img in image_list:
            _, h, w, _ = img.shape  # Ignore N and C dimensions
            pad_h = max_height - h
            pad_w = max_width - w

            # Corrected padding order
            padded_img = F.pad(img, (0, 0, 0, pad_w, 0, pad_h), value=pad_value)

            # **Check if the padded image has an unexpected shape**
            if padded_img.shape[1] == 1:
                self.j_mngr.log_events(f"WARNING: Padded image has unexpected shape {padded_img.shape}, squeezing!", is_trouble=True)
                padded_img = padded_img.squeeze(1)  # Remove any accidental extra dimension

            padded_images.append(padded_img)


        # Use torch.cat() instead of torch.stack()
        batch_tensor = torch.cat(padded_images, dim=0)  # Shape: [N, max_H, max_W, C]

        # Debugging: Log final batch shape before returning
        self.j_mngr.log_events(f"Final batch shape after cat: {batch_tensor.shape}", is_trouble=True)

        return batch_tensor
                

class PythonUtils:

    def __init__(self)->None:
        self.j_mngr = json_manager() 


    def get_library_version(self, library_name: str) -> str:
        """
        Returns the version number of the specified library in the current active Python environment if installed,
        and logs the result.

        Args:
            library_name (str): The name of the library to check.

        Returns:
            str: Version number if found, or a message indicating not installed.
        """
        try:
            version = importlib.metadata.version(library_name)
            self.j_mngr.log_events(
                f"[LibVersion] {library_name} version found: {version}",
                severity="INFO",
                is_trouble=False
            )
            return version

        except importlib.metadata.PackageNotFoundError:
            self.j_mngr.log_events(
                f"[LibVersion] {library_name} is not installed.",
                severity="WARNING",
                is_trouble=False
            )

        except Exception as e:
            self.j_mngr.log_events(
                f"[LibVersion] Error checking version for {library_name}: {e}",
                severity="ERROR",
                is_trouble=False
            )
        return ""       


    def run_python_module_commands(self, command_lists: list[Sequence[str]], label: str = "ScriptBatch"):
        """
        Runs one or more module commands using the current Python environment's interpreter.

        Args:
            command_lists (list[Sequence[str]]): A list of argument lists or tuples. Each will be run as:
                [sys.executable, "-m", *args]
            label (str): A string label to prefix log messages.
        """
        for args in command_lists:
            full_cmd = [sys.executable, "-m"] + args
            try:
                subprocess.run(full_cmd, check=True)
                self.j_mngr.log_events(
                    f"[{label}] Successfully ran: {' '.join(full_cmd)}",
                    TroubleSgltn.Severity.INFO,
                    False
                )
            except subprocess.CalledProcessError as e:
                self.j_mngr.log_events(
                    f"[{label}] Failed command: {' '.join(full_cmd)}\nError: {e}",
                    TroubleSgltn.Severity.WARNING,
                    False
                )
            except Exception as e:
                self.j_mngr.log_events(
                    f"[{label}] Unexpected error running: {' '.join(full_cmd)}\n{type(e).__name__}: {e}",
                    TroubleSgltn.Severity.WARNING,
                    False
                )
