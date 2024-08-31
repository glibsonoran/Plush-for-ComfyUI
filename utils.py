import requests   
from requests.adapters import HTTPAdapter, Retry 
from .mng_json import json_manager, TroubleSgltn 

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
    
    def get_data(self, endpoint:str="", timeout:int=8, retries:int=1, data_type:str="" )-> requests.Response | None:
        session = requests.Session()
        gretries = Retry(total=retries, backoff_factor=0, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=gretries))
        stat_code = 0
        try:
            response = session.get(endpoint, timeout=timeout)
            stat_code = response.status_code
            response.raise_for_status()  # Raises an HTTPError if the response status code indicates an error
            return response

        except requests.RequestException as e:
            self.j_mngr.log_events(f"Unable to fetch data for: {data_type}.  Server returned code: {stat_code}. Error: {e} ",
            TroubleSgltn.Severity.WARNING,
            True)
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
