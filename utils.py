# =======================
#  Standard Libraries
# =======================
from enum import Enum
from typing import Optional, Dict, Any
from io import BytesIO
import base64

# =======================
#  Third-Party Libraries
# =======================
import requests
from requests.adapters import HTTPAdapter, Retry
from PIL import Image, ImageOps
import torch
import numpy as np

# =======================
#  Local Modules
# =======================
from .mng_json import json_manager, TroubleSgltn

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
    

    def tensor_to_bytes(self, tensor: torch.Tensor) -> BytesIO:
        """
        Converts a PyTorch tensor to a bytes object.

        Args:
            tensor (torch.Tensor): The image tensor to convert.

        Returns:
            BytesIO: BytesIO object containing the image data.
        """

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
        
        # Convert RGB NumPy array to PyTorch tensor
        tensor_image = torch.from_numpy(rgb_np).unsqueeze(0)  # Add batch dimension [N, H, W, C]
        
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
           