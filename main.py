import json
import logging
import time

# Start of execution time
start_time = time.time()

# Setup logger
logging.basicConfig( level = logging.INFO, format = "[INFO] %(message)s" ) #configure logging
logger = logging.getLogger(__name__) #return a logger with the specified name

# Features Configurations
with open( "./utils/config.json", "r" ) as file:
    config = json.load( file )


