import os

_TEST_ROOT = os.path.dirname(__file__)

_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)

_PATH_RAW_DATA = os.path.join(_PROJECT_ROOT, "data", "raw") 
_PATH_PROCESSED_DATA = os.path.join(_PROJECT_ROOT, "data", "processed") 
_PATH_MODELS = os.path.join(_PROJECT_ROOT, "models")  
_PATH_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "config.yaml")  
