# Standard library
from abc import ABC, abstractmethod
import time
import json
import re
from enum import Enum
from typing import Callable, Any, Optional, Type, Union, List, Tuple
from urllib.parse import urlparse, urlunparse

# Third-party libraries
import torch
import requests
import openai
import anthropic

# Local modules
from .mng_json import json_manager, TroubleSgltn
from .fetch_models import RequestMode
from .utils import ImageUtils


class ImportedSgltn:
    """
    This class is temporary to prevent circular imports between style_prompt
    and api_requests modules.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImportedSgltn, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized: #pylint: disable=access-member-before-definition
            self._initialized = True
            self._cfig = None
            self._dalle = None
            self.get_imports()

    def get_imports(self):
        """Import and initialize singleton instances from style_prompt"""
        # Guard against re-importing if already done
        if self._cfig is None or self._dalle is None:
            from .style_prompt import cFigSingleton, DalleImage
            self._cfig = cFigSingleton
            self._dalle = DalleImage


    @property
    def cfig(self):
        """Returns the cFigSingleton instance"""
        if self._cfig is None:
            self.get_imports()
        return self._cfig()

    @property
    def dalle(self):
        """Returns the DALLE instance"""
        if self._dalle is None:
            self.get_imports()
        return self._dalle()
       
class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        exponential_base: float = 2.0,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        retryable_http_status_codes: Optional[List[int]] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_exceptions = retryable_exceptions
        self.retryable_http_status_codes = retryable_http_status_codes or [
            408,  # Request Timeout
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504   # Gateway Timeout
        ]


class ErrorParser:
    """Extracts standardized error information from various API responses"""
    
    @staticmethod
    def get_error_code(response: Any) -> Optional[int]:
        """
        Extracts error code from various response formats.
        Returns error code if found, None otherwise.
        """
        # Handle HTTP Response objects
        if isinstance(response, requests.Response):
            return response.status_code

        # OpenAI-style errors (and compatible services like OpenRouter)
        if hasattr(response, 'error'):
            error = response.error
            if isinstance(error, dict):
                # Direct error code
                if 'code' in error and isinstance(error['code'], int):
                    return error['code']

                if 'status' in error and isinstance(error['status'], int):
                    return error['status']

                if 'status_code' in error and isinstance(error['status_code'], int):
                    return error['status_code']
                
                # Nested in metadata (like OpenRouter/Google)
                metadata = error.get('metadata', {})
                if metadata and isinstance(metadata.get('raw'), str):
                    try:
                        raw_error = json.loads(metadata['raw'])
                        code = raw_error.get('error', {}).get('code')
                        if isinstance(code, int):
                            return code
                    except (json.JSONDecodeError, AttributeError):
                        pass

        # Anthropic-style responses
        if hasattr(response, 'status_code'):
            return response.status_code

        # Handle raw JSON responses (some services return direct JSON)
        if isinstance(response, dict):
            # Try common error code paths
            paths = [
                ['error', 'code'],
                ['error', 'status_code'],
                ['error', 'status'],
                ['code'],
                ['status_code'],
                ['status']
            ]
            for path in paths:
                value = response
                for key in path:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = None
                        break
                if isinstance(value, int):
                    return value

        return None

class RetryHandler:
    """Handles retry logic for API calls"""
    def __init__(self, config: RetryConfig, logger: Any):
        self.config = config
        self.logger = logger
        self.error_parser = ErrorParser()

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff"""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        return delay

    def should_retry(self, response: Any) -> bool:
        """Determine if the response is retryable"""
        error_code = self.error_parser.get_error_code(response)
        
        if error_code:
            # Check if it's a retryable code
            return error_code in self.config.retryable_http_status_codes 

        # Handle standard exceptions
        if isinstance(response, Exception) and self.config.retryable_exceptions:
            return any(isinstance(response, exc) for exc in self.config.retryable_exceptions)
        
        return False

    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        last_error_info = None  # Track the last error information
        self.logger.log_events(f"Maximum tries set to: {self.config.max_retries}",
                               is_trouble=True)
        
        for attempt in range(self.config.max_retries):
            try:
                response = func(*args, **kwargs)

                # For HTTP responses
                if isinstance(response, requests.Response):
                    try:
                        response_json = response.json()
                        if 'error' in response_json:
                            error_code = self.error_parser.get_error_code(response_json)
                            if error_code in self.config.retryable_http_status_codes:
                                last_error_info = response_json['error']  # Store error info
                                delay = self.calculate_delay(attempt)
                                
                                self.logger.log_events(
                                    f"Retryable error detected in response content ({error_code}), "
                                    f"retrying in {delay:.2f} seconds...",
                                    TroubleSgltn.Severity.WARNING,
                                    True
                                )
                                time.sleep(delay)
                                continue
                    except ValueError:
                        pass

                    # Then check status codes
                    if 200 <= response.status_code < 300:
                        return response
                    elif self.should_retry(response):
                        last_error_info = {'status': response.status_code, 'text': response.text}
                        delay = self.calculate_delay(attempt)
                        self.logger.log_events(
                            f"Rate limit or server error {response.status_code}, "
                            f"retrying in {delay:.2f} seconds...",
                            TroubleSgltn.Severity.WARNING,
                            True
                        )
                        time.sleep(delay)
                        continue
                    else:
                        return response

                # For OpenAI/API responses with embedded errors
                error_code = self.error_parser.get_error_code(response)
                if error_code and error_code in self.config.retryable_http_status_codes:
                    last_error_info = response.error if hasattr(response, 'error') else str(response)
                    delay = self.calculate_delay(attempt)
                    self.logger.log_events(
                        f"Rate limit or error detected in API response ({error_code}), "
                        f"retrying in {delay:.2f} seconds...",
                        TroubleSgltn.Severity.WARNING,
                        True
                    )
                    time.sleep(delay)
                    continue

                return response

            except Exception as e:
                last_exception = e
                last_error_info = str(e)  # Store exception info
                
                if not self.should_retry(e):
                    self.logger.log_events(
                        f"Non-retryable error occurred: {str(e)}",
                        TroubleSgltn.Severity.ERROR,
                        True
                    )
                    raise

                delay = self.calculate_delay(attempt)
                self.logger.log_events(
                    f"Attempt {attempt + 1}/{self.config.max_retries} failed. "
                    f"Retrying in {delay:.2f} seconds. Error: {str(e)}",
                    TroubleSgltn.Severity.WARNING,
                    True
                )
                time.sleep(delay)

        # Create a meaningful exception with the last error information
        error_message = f"Maximum retry attempts ({self.config.max_retries}) exceeded. "
        if last_error_info:
            error_message += f"Last error: {last_error_info}"
        
        # Raise the original exception if we have one, otherwise raise a RuntimeError
        if last_exception:
            raise last_exception
        raise RuntimeError(error_message)

class RetryConfigFactory:
    """Factory for creating retry configurations based on request type"""
    
    @staticmethod
    def create_config(request_type: RequestMode) -> RetryConfig:
        web_exceptions = [
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException,
            ConnectionError,
            TimeoutError
        ]
        
        api_exceptions = [
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APIStatusError
        ]

        anthropic_exceptions = [
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.APIStatusError,
            anthropic.APIError
        ]

        configs = {
            RequestMode.OPENAI: RetryConfig(
                max_retries=3,
                base_delay=1.0,
                max_delay=10.0,
                retryable_exceptions=api_exceptions
            ),
            RequestMode.CLAUDE: RetryConfig(
                max_retries=2,
                base_delay=2.0,
                max_delay=8.0,
                retryable_exceptions=anthropic_exceptions
            ),
            RequestMode.OPENSOURCE: RetryConfig(
                max_retries=3,
                base_delay=1.0,
                max_delay=8.0,
                retryable_exceptions=web_exceptions,
                retryable_http_status_codes=[408, 429, 500, 502, 503, 504]
            ),
            RequestMode.OSSIMPLE: RetryConfig(
                max_retries=3,
                base_delay=1.0,
                max_delay=8.0,
                retryable_exceptions=web_exceptions,
                retryable_http_status_codes=[408, 429, 500, 502, 503, 504]
            ),
            RequestMode.LMSTUDIO: RetryConfig(
                max_retries=2,
                base_delay=0.5,
                max_delay=4.0,
                retryable_exceptions=web_exceptions,
                retryable_http_status_codes=[408, 429, 500, 502, 503, 504]
            ),
            RequestMode.GROQ: RetryConfig(
                max_retries=3,
                base_delay=1.0,
                max_delay=6.0,
                retryable_exceptions=api_exceptions
            ),
            RequestMode.OOBABOOGA: RetryConfig(
                max_retries=2,
                base_delay=1.0,
                max_delay=6.0,
                retryable_exceptions=web_exceptions,
                retryable_http_status_codes=[408, 429, 500, 502, 503, 504]
            ),
            # DALL-E specific configuration
            RequestMode.DALLE: RetryConfig(
                max_retries=3,
                base_delay=2.0,
                max_delay=15.0,
                retryable_http_status_codes=[400,429],
                retryable_exceptions=[
                    openai.APIConnectionError,
                    openai.RateLimitError,
                    openai.APIStatusError
                ]
            )
        }
        return configs.get(request_type, RetryConfig())
    

class Request(ABC):
    """Abstract base class for all request types"""

    class RequestType(Enum):
        COMPLETION = "completion"
        POST = "post"
        IMAGE = "image"
        ANTHROPIC = "claude"

    def __init__(self):
        self.imps = ImportedSgltn()
        self.utils = request_utils()
        self.cFig = self.imps.cfig
        self.mode = RequestMode
        self.dalle = self.imps.dalle
        self.j_mngr = json_manager()
        
        # Initialize retry configuration and handler
        retry_config = RetryConfigFactory.create_config(self.cFig.lm_request_mode)
        self.retry_handler = RetryHandler(retry_config, self.j_mngr)

    def _initialize_retry_handler(self, **kwargs):
        """Initialize retry handler with optional override from kwargs"""
        # Get base configuration
        retry_config = RetryConfigFactory.create_config(self.cFig.lm_request_mode)
        
        # Override max_retries if provided in kwargs otherwise use default
        if 'tries' in kwargs and kwargs['tries']:
            tries = kwargs['tries']
            if isinstance(tries, str) and tries != "default":
                retry_config.max_retries = int(tries)
               
        self.retry_handler = RetryHandler(retry_config, self.j_mngr)

    def _make_request(self, request_type: RequestType, *args) -> Any:
        """Unified request method handling different request types"""
        if request_type == self.RequestType.COMPLETION:
            client, params = args
            return client.chat.completions.create(**params)
        
        elif request_type == self.RequestType.ANTHROPIC:
            client, params = args
            return client.messages.create(**params)
        
        elif request_type == self.RequestType.POST:
            url, headers, params = args
            return requests.post(url, headers=headers, json=params, timeout=(12, 120))
        
        elif request_type == self.RequestType.IMAGE:
            client, params = args
            return client.images.generate(**params)
        
        else:
            raise ValueError(f"Unsupported request type: {request_type}")        

    @abstractmethod
    def request_completion(self, **kwargs) -> Any:
        pass

    def _process_image(self, image: Optional[Union[str, torch.Tensor]]) -> Optional[str]:
        """Common image processing logic"""
        if not image:
            return None

        if isinstance(image, torch.Tensor):
            image = self.dalle.tensor_to_base64(image)
            
        if not isinstance(image, str):
            self.j_mngr.log_events(
                "Image file is invalid. Image will be disregarded in the generated output.",
                TroubleSgltn.Severity.WARNING,
                True
            )
            return None
            
        return image

    def _log_completion_metrics(self, response: Any, response_type: str = "standard"):
        """Common logging for completion metrics"""
        try:
            if response_type == "standard":
                if getattr(response, 'model', None):
                    self.j_mngr.log_events(
                        f"Using LLM: {response.model}", 
                        is_trouble=True
                    )
                if getattr(response, 'usage', None):
                    self.j_mngr.log_events(
                        f"Tokens Used: {response.usage}",
                        TroubleSgltn.Severity.INFO,
                        True
                    )
            elif response_type == "json":
                if response.get('model'):
                    self.j_mngr.log_events(
                        f"Using LLM: {response['model']}", 
                        is_trouble=True
                    )
                if response.get('usage'):
                    self.j_mngr.log_events(
                        f"Tokens Used: {response['usage']}",
                        TroubleSgltn.Severity.INFO,
                        True
                    )
        except Exception as e:
            self.j_mngr.log_events(
                f"Unable to report completion metrics: {e}",
                TroubleSgltn.Severity.INFO,
                True
            )

class oai_object_request(Request):
    """Concrete class for OpenAI API object-based requests"""
    
    # def _make_completion_request(self, client, params):
    #     """Wrapped completion request for retry handling"""
    #     return client.chat.completions.create(**params)

    def _get_client(self) -> Optional[Any]:
        """Get appropriate client based on request type"""
        request_type = self.cFig.lm_request_mode
        client = None
        error_message = None

        if request_type in [self.mode.OPENSOURCE, self.mode.OLLAMA]:
            if self.cFig.lm_url:
                self.j_mngr.log_events(
                    "Setting client to OpenAI Open Source LLM object",
                    is_trouble=True
                )
                client = self.cFig.lm_client
            else:
                error_message = "Open Source api object is not ready for use, no URL provided."
                
        elif request_type == self.mode.GROQ:
            if self.cFig.lm_url:
                self.j_mngr.log_events(
                    "Setting client to OpenAI Groq LLM object",
                    is_trouble=True
                )
                client = self.cFig.lm_client
            else:
                error_message = "Groq OpenAI api object is not ready for use, no URL provided."
                
        elif request_type == self.mode.OPENAI:
            if self.cFig.key:
                self.j_mngr.log_events(
                    "Setting client to OpenAI ChatGPT object",
                    is_trouble=True
                )
                client = self.cFig.openaiClient
            else:
                error_message = "Invalid or missing OpenAI API key. Keys must be stored in an environment variable."

        if error_message:
            self.j_mngr.log_events(
                error_message,
                TroubleSgltn.Severity.WARNING,
                True
            )

        return client

    def request_completion(self, **kwargs) -> str:
        """Execute completion request with retry handling"""
        GPTmodel = kwargs.get('model')
        creative_latitude = kwargs.get('creative_latitude', 0.7)
        tokens = kwargs.get('tokens', 500)
        prompt = kwargs.get('prompt', "")
        instruction = kwargs.get('instruction', "")
        #file = kwargs.get('file', "").strip()
        image = kwargs.get('image', None)
        example_list = kwargs.get('example_list', [])
        add_params = kwargs.get('add_params', None)

        CGPT_response = ""
        client = self._get_client()
        self._initialize_retry_handler(**kwargs)

        if not client:
            return "Unable to process request, client initialization failed"

        # Process image if present
        image = self._process_image(image)

        # Build messages based on presence of image
        if not image:
            messages = self.utils.build_data_basic(prompt, example_list, instruction)
        else:
            messages = self.utils.build_data_multi(prompt, instruction, example_list, image)

        # Handle empty input case
        if not any([prompt, image, instruction, example_list]):
            return "Photograph of a stained empty box with 'NOTHING' printed on its side in bold letters"

        params = {
            "model": GPTmodel,
            "messages": messages,
            "temperature": creative_latitude,
            "max_tokens": tokens  
        }

        # Certain models have parameter restrictions     
        params = self.utils.model_param_adjust(params, GPTmodel)
            
        if add_params:
            self.j_mngr.append_params(params, add_params, ['param', 'value'])

        try:
            response = self.retry_handler.execute_with_retry(
                self._make_request,
                self.RequestType.COMPLETION,
                client,
                params
            )  #_make_request is passed as a wrapped function, the arguments that follow are passed into
               #args which is unpacked as a tuple in _make_request()

            if response and response.choices and 'error' not in response:
                self._log_completion_metrics(response)
                CGPT_response = self.utils.clean_response_text(
                    response.choices[0].message.content
                )
            else:
                err_mess = getattr(response, 'error', "Error message missing")
                self.j_mngr.log_events(
                    f"Server was unable to process this request. Error: {err_mess}",
                    TroubleSgltn.Severity.ERROR,
                    True
                )
                CGPT_response = "Server was unable to process the request"

        except Exception as e:
            self.j_mngr.log_events(
                f"Request failed: {str(e)}",
                TroubleSgltn.Severity.ERROR,
                True
            )
            CGPT_response = "Server was unable to process the request"

        return CGPT_response

class claude_request(Request):
    """Concrete class for Claude/Anthropic API requests"""
    
    def request_completion(self, **kwargs) -> str:
        claude_model = kwargs.get('model')
        creative_latitude = kwargs.get('creative_latitude', 0.7)
        tokens = kwargs.get('tokens', 500)
        prompt = kwargs.get('prompt', "")
        instruction = kwargs.get('instruction', "")
        image = kwargs.get('image', None)
        example_list = kwargs.get('example_list', [])
        add_params = kwargs.get('add_params', None)

        claude_response = ""
        client = self.cFig.anthropic_client
        self._initialize_retry_handler(**kwargs)

        if not client:
            self.j_mngr.log_events(
                "Invalid or missing Anthropic API key. Keys must be stored in an environment variable.",
                TroubleSgltn.Severity.ERROR,
                True
            )
            return "Invalid or missing Anthropic API key"

        # Process image if present
        image = self._process_image(image)

        # Build messages
        messages = self.utils.build_data_claude(prompt, example_list, image)

        # Handle empty input case
        if not any([prompt, image, instruction, example_list]):
            return "Empty request, no input provided"

        # Prepare request parameters
        params = {
            "model": claude_model,
            "messages": messages,
            "temperature": creative_latitude,
            "system": instruction,
            "max_tokens": tokens
        }

        if add_params:
            self.j_mngr.append_params(params, add_params, ['param', 'value'])

        try:
            response = self.retry_handler.execute_with_retry(
                self._make_request,
                self.RequestType.ANTHROPIC,
                client,
                params
            )

            if response and 'error' not in response:
                self._log_completion_metrics(response)
                try:
                    claude_response = response.content[0].text
                    claude_response = self.utils.clean_response_text(claude_response)
                except (IndexError, AttributeError):
                    claude_response = "No valid data was returned"
                    self.j_mngr.log_events(
                        "Claude response was not valid data",
                        TroubleSgltn.Severity.WARNING,
                        True
                    )
            else:
                claude_response = "Server was unable to process the request"
                self.j_mngr.log_events(
                    'Server was unable to process this request.',
                    TroubleSgltn.Severity.ERROR,
                    True
                )

        except Exception as e:
            error_msg = self.utils.parse_anthropic_error(e)
            self.j_mngr.log_events(
                f"Request failed: {error_msg}",
                TroubleSgltn.Severity.ERROR,
                True
            )
            claude_response = "Server was unable to process the request"

        return claude_response
    

class oai_web_request(Request):
    """Concrete class for OpenAI-compatible web requests"""

    def request_completion(self, **kwargs) -> str:
        GPTmodel = kwargs.get('model', "")
        creative_latitude = kwargs.get('creative_latitude', 0.7)
        url = kwargs.get('url', None)
        tokens = kwargs.get('tokens', 500)
        image = kwargs.get('image', None)
        prompt = kwargs.get('prompt', None)
        instruction = kwargs.get('instruction', "")
        example_list = kwargs.get('example_list', [])
        add_params = kwargs.get('add_params', None)

        CGPT_response = ""
        request_type = self.cFig.lm_request_mode
        self._initialize_retry_handler(**kwargs)

        # URL setup and validation
        self.cFig.lm_url = url
        if not self.cFig.is_lm_server_up:
            self.j_mngr.log_events(
                "Local or remote server is not responding, may be unable to send data.",
                TroubleSgltn.Severity.WARNING,
                True
            )

        # Process image if present
        if image and request_type == self.mode.OSSIMPLE:
            self.j_mngr.log_events(
                "The AI Service using 'Simplified Data' can't process an image. The image will be disregarded in generated output.",
                TroubleSgltn.Severity.INFO,
                True
            )
            image = None
        else:
            image = self._process_image(image)

        # Get appropriate key for request type
        key = self._get_key_for_request_type(request_type)
        headers = self.utils.build_web_header(key)

        # Build message structure
        if request_type == self.mode.OSSIMPLE or not image:
            messages = self.utils.build_data_basic(prompt, example_list, instruction)
            self.j_mngr.log_events(
                "Using Basic data structure",
                TroubleSgltn.Severity.INFO,
                True
            )
        else:
            messages = self.utils.build_data_multi(prompt, instruction, example_list, image)
            self.j_mngr.log_events(
                "Using Complex data structure",
                TroubleSgltn.Severity.INFO,
                True
            )

        params = {
            "model": GPTmodel,
            "messages": messages,
            "temperature": creative_latitude,
            "max_tokens": tokens  
        }

        # Certain models have parameter restrictions        
        params = self.utils.model_param_adjust(params, GPTmodel)

            
        if add_params:
            self.j_mngr.append_params(params, add_params, ['param', 'value'])

        try:
            response = self.retry_handler.execute_with_retry(
                self._make_request,
                self.RequestType.POST,
                url,
                headers,
                params
            )

            if response.status_code in range(200, 300):
                response_json = response.json()
                if response_json and 'error' not in response_json:
                    CGPT_response = self.utils.clean_response_text(
                        response_json['choices'][0]['message']['content']
                    )
                    self._log_completion_metrics(response_json, "json")
                else:
                    error_message = response_json.get('error', 'Unknown error')
                    self.j_mngr.log_events(
                        f"Server error in response: {error_message}",
                        TroubleSgltn.Severity.ERROR,
                        True
                    )
                    CGPT_response = "Server was unable to process the request"
            else:
                self.j_mngr.log_events(
                    f"Server error status: {response.status_code}: {response.text}",
                    TroubleSgltn.Severity.ERROR,
                    True
                )
                CGPT_response = "Server was unable to process the request"

        except Exception as e:
            self.j_mngr.log_events(
                f"Request failed: {str(e)}",
                TroubleSgltn.Severity.ERROR,
                True
            )
            CGPT_response = "Server was unable to process the request"

        return CGPT_response

    def _get_key_for_request_type(self, request_type: RequestMode) -> str:
        """Get appropriate key based on request type"""
        if request_type == self.mode.OPENAI:
            return self.cFig.key
        elif request_type in [self.mode.OPENSOURCE, self.mode.LMSTUDIO]:
            key = self.cFig.custom_key or self.cFig.lm_key #Will populate with first 'truthy' value
            return key
        elif request_type == self.mode.GROQ:
            return self.cFig.groq_key
        return ""

class ooba_web_request(Request):
    """Concrete class for Oobabooga web requests"""

    def request_completion(self, **kwargs) -> str:
        GPTmodel = kwargs.get('model', "")
        creative_latitude = kwargs.get('creative_latitude', 0.7)
        url = kwargs.get('url', None)
        tokens = kwargs.get('tokens', 500)
        prompt = kwargs.get('prompt', None)
        instruction = kwargs.get('instruction', "")
        example_list = kwargs.get('example_list', [])
        add_params = kwargs.get('add_params', None)

        CGPT_response = ""
        request_type = self.cFig.lm_request_mode
        self._initialize_retry_handler(**kwargs)

        # URL setup and validation
        url = self.utils.validate_and_correct_url(url)
        self.cFig.lm_url = url

        if not self.cFig.is_lm_server_up:
            self.j_mngr.log_events(
                "Local server is not responding, may be unable to send data.",
                TroubleSgltn.Severity.WARNING,
                True
            )

        # Get appropriate key
        key = self.cFig.key if request_type == self.mode.OPENAI else self.cFig.lm_key
        headers = self.utils.build_web_header(key)

        # Build messages with Oobabooga-specific format
        messages = self.utils.build_data_ooba(prompt, example_list, instruction)

        # Prepare request parameters
        params = {
            "model": GPTmodel,
            "messages": messages,
            "temperature": creative_latitude,
            "max_tokens": tokens,
        }

        # Add Oobabooga-specific parameters
        if request_type == self.mode.OOBABOOGA:
            self.j_mngr.log_events(
                f"Processing Oobabooga http: POST request with url: {url}",
                is_trouble=True
            )
            params.update({
                "user_bio": "",
                "user_name": ""
            })

        if add_params:
            self.j_mngr.append_params(params, add_params, ['param', 'value'])

        try:
            response = self.retry_handler.execute_with_retry(
                self._make_request,
                self.RequestType.POST,
                url,
                headers,
                params
            )

            if response.status_code in range(200, 300):
                response_json = response.json()
                if response_json and 'error' not in response_json:
                    CGPT_response = self.utils.clean_response_text(
                        response_json['choices'][0]['message']['content']
                    )
                    self._log_completion_metrics(response_json, "json")
                else:
                    error_message = response_json.get('error', 'Unknown error')
                    self.j_mngr.log_events(
                        f"Server error in response: {error_message}",
                        TroubleSgltn.Severity.ERROR,
                        True
                    )
            else:
                CGPT_response = "Server was unable to process the request"
                self.j_mngr.log_events(
                    f"Server error status: {response.status_code}: {response.text}",
                    TroubleSgltn.Severity.ERROR,
                    True
                )

        except Exception as e:
            self.j_mngr.log_events(
                f"Request failed: {str(e)}",
                TroubleSgltn.Severity.ERROR,
                True
            )
            CGPT_response = "Server was unable to process the request"

        return CGPT_response

class dall_e_request(Request):
    """Concrete class for DALL-E image generation requests"""

    def __init__(self):
        super().__init__()
        self.trbl = TroubleSgltn()
        self.iu = ImageUtils()
        # Override with DALL-E specific retry config
        retry_config = RetryConfigFactory.create_config(self.cFig.lm_request_mode)
        self.retry_handler = RetryHandler(retry_config, self.j_mngr)


    def request_completion(self, **kwargs) -> Tuple[torch.Tensor, str]:
        GPTmodel = kwargs.get('model')
        prompt = kwargs.get('prompt')
        image_size = kwargs.get('image_size')
        image_quality = kwargs.get('image_quality')
        style = kwargs.get('style')
        batch_size = kwargs.get('batch_size', 1)

        self.trbl.set_process_header('Dall-e Request')
        batched_images = torch.zeros(1, 1024, 1024, 3, dtype=torch.float32)
        revised_prompt = "Image and mask could not be created"

        client = self.cFig.openaiClient
        self._initialize_retry_handler(**kwargs)

        if not client:
            self.j_mngr.log_events(
                "OpenAI API key is missing or invalid. Key must be stored in an environment variable.",
                TroubleSgltn.Severity.WARNING,
                True
            )
            return batched_images, revised_prompt

        self.j_mngr.log_events(
            f"Talking to Dalle model: {GPTmodel}",
            is_trouble=True
        )

        images_list = []
        have_rev_prompt = False

        for _ in range(batch_size):
            params = {
                "model": GPTmodel,
                "prompt": prompt,
                "size": image_size,
                "quality": image_quality,
                "style": style,
                "n": 1,
                "response_format": "b64_json"
            }

            try:
                response = self.retry_handler.execute_with_retry(
                    self._make_request,
                    self.RequestType.IMAGE,
                    client,
                    params
                )

                if response and 'error' not in response:
                    if not have_rev_prompt:
                        revised_prompt = response.data[0].revised_prompt
                        have_rev_prompt = True

                    b64Json = response.data[0].b64_json
                    if b64Json:
                        png_image, _ = self.dalle.b64_to_tensor(b64Json)
                        images_list.append(png_image)
                    else:
                        self.j_mngr.log_events(
                            f"Dalle-e could not process an image in your batch of: {batch_size}",
                            TroubleSgltn.Severity.WARNING,
                            True
                        )

            except Exception as e:
                self.j_mngr.log_events(
                    f"Failed to generate image {_ + 1}/{batch_size}: {str(e)}",
                    TroubleSgltn.Severity.ERROR,
                    True
                )

        if images_list:
            count = len(images_list)
            self.j_mngr.log_events(
                f'{count} images were processed successfully in your batch of: {batch_size}',
                is_trouble=True
            )
            batched_images = torch.cat(images_list, dim=0)
        else:
            self.j_mngr.log_events(
                f'No images were processed in your batch of: {batch_size}',
                TroubleSgltn.Severity.WARNING,
                is_trouble=True
            )

        self.trbl.pop_header()
        return batched_images, revised_prompt


class ollama_unload_request(Request):
    """Concrete class for model unload requests"""

    class ModelTTL(Enum):
        KILL = 0
        INDEF = -1
        NOSET = "no_setting"

    def __init__(self):
        super().__init__()
        self.trbl = TroubleSgltn()

    def request_completion(self, **kwargs) -> bool:

        req_mode = self.cFig.lm_request_mode

        keep_alive = kwargs.get('model_TTL', self.ModelTTL.NOSET)
        if not isinstance(keep_alive, self.ModelTTL):
            self.j_mngr.log_events("Invalid `model_TTL` value provided.", 
                                    TroubleSgltn.Severity.WARNING, 
                                    True
            )
            return False        

        if keep_alive == self.ModelTTL.NOSET: #Don't change the current TTL setting
            return True
        
        self.trbl.set_process_header("Ollama Unload Model Setting")
        
        if req_mode not in {RequestMode.OLLAMA, RequestMode.OPENSOURCE}:
            self.j_mngr.log_events("Model Unloading does not work with this AI Service type.",
                                   TroubleSgltn.Severity.WARNING,
                                   True)
            return False        

        model = kwargs.get('model')

        if not model:
            self.j_mngr.log_events("No model specified for unload", 
                                    TroubleSgltn.Severity.WARNING,
                                    True)
            return False
        
        llm_url = kwargs.get('url', 'http://localhost:11434')  # Get URL or use default           
        # replace the URL path with Ollama's native endpoint
        base_url = self.utils.validate_and_correct_url(llm_url, '/api/generate')
        headers = self.utils.build_web_header()

        params = {
            "model": model,
            "keep_alive": keep_alive.value
        }
        try:
            self.j_mngr.log_events(f"Attempting to set model TTL using URL: {base_url}", is_trouble=True)
            response = requests.post(base_url, headers=headers, json=params, timeout=5)

        except requests.RequestException as e:
            self.j_mngr.log_events(f"Model unload request failed: {e.__class__.__name__}: {str(e)}", 
                                    TroubleSgltn.Severity.WARNING, 
                                    True)
            return False
        
        response_text = response.text if response.text else "None Provided"

        if response.status_code == 200:
            self.j_mngr.log_events(f"Model unload setting successful.  Response: {response_text}", is_trouble=True)
            return True
        
        self.j_mngr.log_events(f"Model unload failed with status: {response.status_code}, Response: {response_text}", 
                            TroubleSgltn.Severity.WARNING,
                            True)
        return False
    
    
class request_context:
    def __init__(self)-> None:
        self._request = None
        self.j_mngr = json_manager()

    @property
    def request(self)-> Request:
        return self._request

    @request.setter
    def request(self, request:Request)-> None:
        self._request = request

    def execute_request(self, **kwargs):
        if self._request is not None:
            return self._request.request_completion(**kwargs)
        
        self.j_mngr.log_events("No request strategy object was set",
                               TroubleSgltn.Severity.ERROR,
                               True)
        return None
    
class request_utils:

    def __init__(self)-> None:
        self.j_mngr = json_manager()
        self.mode = RequestMode


    def model_param_adjust(self, params: dict, model: str) -> dict:
        adj_models = ['o1', 'o3']
        if any(m in model for m in adj_models):
            self.j_mngr.log_events(
                "The 'o' models have parameter restrictions. Removing 'max_tokens' and setting 'temperature' to 1",
                TroubleSgltn.Severity.INFO,
                True
            )
            
            # Handle temperature parameter
            if 'temperature' in params:
                params['temperature'] = 1
            
            # Handle max_tokens parameter
            if 'max_tokens' in params:
                #params['max_completion_tokens'] = params.pop('max_tokens')
                params.pop('max_tokens', None)
                
            return params
        
        return params  



    def build_data_multi(self, prompt:str, instruction:str="", examples:list=None, image:str=None):
        """
        Builds a list of message dicts, aggregating 'role:user' content into a list under 'content' key.
        - image: Base64-encoded string or None. If string, included as 'image_url' type content.
        - prompt: String to be included as 'text' type content under 'user' role.
        - examples: List of additional example dicts to be included.
        - instruction: Instruction string to be included under 'system' role.
        """
        messages = []
        user_role = {"role": "user", "content": None}
        user_content = []

        if instruction:
            messages.append({"role": "system", "content": instruction})         

        if examples:
            messages.extend(examples)

        if prompt:
            user_content.append({"type": "text", "text": prompt})

        processed_image = self.process_image(image)
        if processed_image:
            user_content.append(processed_image)
        

        if user_content:
            user_role['content'] = user_content   
            messages.append(user_role)

        return messages
    
    def build_data_basic(self, prompt:str, examples:list=None, instruction:str=""):
        """
        Builds a list of message dicts, presenting each 'role:user' item in its own dict.
        - prompt: String to be included as 'text' type content under 'user' role.
        - examples: List of additional example dicts to be included.
        - instruction: Instruction string to be included under 'system' role.
        """

        messages = []

        if instruction:
            messages.append({"role": "system", "content": instruction})

        if examples:
            messages.extend(examples)     
        
        if prompt:
            messages.append({"role": "user", "content": prompt})


        return messages
    
    def build_data_ooba(self, prompt:str, examples:list=None, instruction:str="")-> list:
        """
        Builds a list of message dicts, presenting each 'role:user' item in its own dict.
        Since Oobabooga's system message is broken it includes it in the prompt
        - prompt: String to be included as 'text' type content under 'user' role.
        - examples: List of additional example dicts to be included.
        - instruction: Instruction string to be included under 'system' role.
        """

        messages = []

        ooba_prompt = ""

        if instruction:
            ooba_prompt += f"INSTRUCTION: {instruction}\n\n"

        if prompt:
            ooba_prompt += f"PROMPT: {prompt}"


        if examples:
            messages.extend(examples)
        
        if ooba_prompt:
            messages.append({"role": "user", "content": ooba_prompt.strip()})

        return messages 
    
    def build_data_claude(self, prompt:str, examples:list=None, image:str=None)-> list:
        """
        Builds a list of message dicts, aggregating 'role:user' content into a list under 'content' key.
        - image: Base64-encoded string or None. If string, included as 'image_url' type content.
        - prompt: String to be included as 'text' type content under 'user' role.
        - examples: List of additional example dicts to be included.
        """
        messages = []
        user_role = {"role": "user", "content": None}
        user_content = []

        if examples:
            messages.extend(examples)


        processed_image = self.process_image(image,RequestMode.CLAUDE)
        if processed_image:
            user_content.append(processed_image)
        
        if prompt:
            user_content.append({"type": "text", "text": prompt})

        if user_content:     
            user_role['content'] = user_content    
            messages.append(user_role)

        return messages


    def process_image(self, image: str, request_type:RequestMode=RequestMode.OPENAI) :
        if not image:
            return None
        
        if isinstance(image, str):
            if request_type == self.mode.CLAUDE:
                return {
                "type": "image",
                "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image
                }
            }

            return {"type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                    }                  
                    }

        self.j_mngr.log_events("Image file is invalid.", TroubleSgltn.Severity.WARNING, True)
        return None
    
    def build_web_header(self, key:str=""):
        if key:
            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}" 
            }
        else:
            headers = {
                "Content-Type": "application/json"
            }    

        return headers    

    def validate_and_correct_url(self, user_url:str, required_path:str='/v1/chat/completions'):
        """
        Takes the user's url and make sure it has the correct path for the connection
        args:
            user_url (str): The url to be validated and corrected if necessary
            required_path (str):  The correct path
        return:            
            A string with either the original url if it was correct or the corrected url if it wasn't
        """
        corrected_url = ""
        parsed_url = urlparse(user_url)
                
        # Check if the path is the required_path
        if not parsed_url.path == required_path:
            corrected_url = urlunparse((parsed_url.scheme,
                                        parsed_url.netloc,
                                        required_path,
                                        '',
                                        '',
                                        ''))

        else:
            corrected_url = user_url
            
        self.j_mngr.log_events(f"URL was validated and is being presented as: {corrected_url}",
                            TroubleSgltn.Severity.INFO,
                            True)

        return corrected_url   
    
    def clean_response_text(self, text: str)-> str:
        # Replace multiple newlines or carriage returns with a single one
        cleaned_text = re.sub(r'\n+', '\n', text).strip()
        return cleaned_text
    
    @staticmethod
    def parse_anthropic_error(e):
        """
        Parses error information from an exception object.

        Args:
            e (Exception): The exception from which to parse the error information.

        Returns:
            str: A user-friendly error message.
        """
        # Default error message
        default_message = "An unknown error occurred"

        # Check if the exception has a response attribute and it can be converted to JSON
        if hasattr(e, 'response') and callable(getattr(e.response, 'json', None)):
            try:
                error_details = e.response.json()
                # Navigate through the nested dictionary safely
                return error_details.get('error', {}).get('message', default_message)
            except ValueError:
                # JSON decoding failed
                return f"Failed to decode JSON from response: {e.response.text}"
            except Exception as ex:
                # Catch-all for any other issues that may arise
                return f"Error processing the error response: {str(ex)}"
        elif hasattr(e, 'message'):
            return e.message
        else:
            return str(e)
