import argparse
import csv
from itertools import zip_longest
import json
import logging
import time

from utils.mailer import Mailer

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

# Function to send email alerts
def send_mail():
    Mailer().send( config["Email_Receive"] )

# Function to log the counting data
def log_data( move_in, in_time, move_out, out_time ):
    data = [ move_in, in_time, move_out, out_time ]
    # Cartesian product, returns tuples
    export_data = zip_longest( *data, fillvalue = '' )

    with open( "./utils/data/logs/counting_data.csv", "w", newline = '' ) as myfile:
        # Write export data to counting_data.csv
        wr = csv.writer( myfile, quoting = csv.QUOTE_ALL )
        if myfile.tell() == 0: # If header rows do NOT exist
            wr.writerow( ("Move In", "In Time", "Move Out", "Out Time") )
            wr.writerows( export_data )

# Actual function
def people_counter():
    args = parse_arguments() # Asks for CLI arguments, args is dictionary
    # List of class labels MobileNet SSD was trained to detect
    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]
    # Load model from disk (assigns model to net)
    net = cv2.dnn.readNetFromCaffe( args[ "prototxt" ], args[ "model" ] )


