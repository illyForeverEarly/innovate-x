import argparse
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

# Function to parse arguments
def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument( "-p", "--prototxt", required = False,
                    help = "path to Caffe 'deploy' prototxt file" )
    ap.add_argument( "-m", "--model", required = True, 
                    help = "path to Caffe pre-trained model" )
    ap.add_argument( "-i", "--input", type = str,
                    help = "path to optional input video file" )
    ap.add_argument( "-o", "--output", type = str,
                    help = "path to optional output video file" )
    # Confidence (by default) = 0.4
    ap.add_argument( "-c", "--confidence", type = float, default = 0.4,
                    help = "minimum probability to filter weak detections" )
    ap.add_argument( "-s", "--skip-frames", type = int, default = 30,
                    help = "number of skip frames between detections" )
    args = vars( ap.parse_args() ) # args is a dictionary where arguments are keys
    return args



