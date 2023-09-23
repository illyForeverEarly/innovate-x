import datetime
import threading
import dlib
import imutils
from concurrent.futures import thread
from imutils.video import FPS
import cv2
from imutils.video import VideoStream
import argparse
import csv
from itertools import zip_longest
import json
import logging
import time

import numpy as np
from tracker.centroidtracker import CentroidTracker

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

    # If there was no video argument, then IP-camera
    if not args.get( "input", False ):
        logger.info( "Starting the live stream..." )
        vs = VideoStream( config["url"] ).start() # Starts live stream of given url
        time.sleep(2.0)

    # Otherwise, video file
    else:
        logger.info( "Starting the video..." )
        vs = cv2.VideoCapture( args["input"] )

    # Initialize the video writer
    writer = None

    # Initialize frame dimensions
    W = None # Width
    H = None # Height

    # Create a Centroid Tracker object
    ct = CentroidTracker( maxDissappeared = 40, maxDistance = 50 )
    trackers = [] # Store dlib correlation filters
    trackableObjects = {} # Dictionary to map unique object IDs to Trackable Object

    # Total number of frames processed so far
    totalFrames = 0
    # Total number of objects moving left or right
    totalLeft = 0
    totalRight = 0

    # Initialize lists to store counting data
    total = []
    move_out = []
    move_in = []
    out_time = []
    in_time = []

    # Start FPS
    fps = FPS().start()

    # Concurrency
    if config["Thread"]:
        vs = thread.ThreadingClass( config["url"] )

    # Loop over frames from video stream
    while True:
        # Handle if we're reading from VideoCapture or VideoStream
        frame = vs.read()
        frame = frame[1] if args.get( "input", False ) else frame

        # If video and we didn't get the frame, we reached the end of the video
        if args["input"] is not None and frame is None:
            break

        # Resize frame
        frame = imutils.resize( frame, width = 500 )
        # Convert from BGR to RGB for dlib
        rgb = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB )

        # Set frame dimensions if empty
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # If writing video to disk, initialize writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter( args["output"], fourcc, 30, (W,H), True )

        # Current status
        status = "Waiting"
        # Returned list of bounding box rectangles by object detector or correlation tracker
        rects = []

        # More computationally expensive object detection
        if totalFrames % args["skip_frames"] == 0:
            # Set status
            status = "Detecting"
            trackers = [] # Initialize new set of object trackers

            blob = cv2.dnn.blobFromImage( frame, 0.007843, (W,H), 127.5 ) # Convert frame to blob
            net.setInput( blob ) # Pass blob through model
            detections = net.forward() # Get detections

            # Loop over detections
            for i in np.arange( 0, detections.shape[2] ):
                confidence = detections[0, 0, i, 2] # Extract confidence

                if confidence > args["confidence"]: # Filter weak connections
                    # Extract index of class label
                    idx = int( detections[0, 0, i, 1] )

                    # If not a person - ignore
                    if CLASSES[ idx ] != "person":
                        continue

                    # Compute (x,y) of bounding box
                    box = detections[ 0, 0, i, 3:7 ] * np.array( [W, H, W, H] )
                    (startX, startY, endX, endY) = box.astype("int")

                    # Construct dlib rectangle object
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle( startX, startY, endX, endY )
                    # Start dlib filtering
                    tracker.start_track( rgb, rect )

                    # Add to list of trackers
                    trackers.append( tracker )

        # Otherwise, use object trackers than detectors
        else:
            # Loop over trackers
            for tracker in trackers:
                # Set status
                status = "Tracking"

                # Update tracker
                tracker.update( rgb )
                pos = tracker.get_position() # Get tracker position

                # Unpack position
                startX = int( pos.left() )
                startY = int( pos.top() )
                endX = int( pos.right() )
                endY = int( pos.bottom() )

                # Add to rectangles list
                rects.append( (startX, startY, endX, endY) )

        # Draw vertical line
        cv2.line( frame, (W // 2, 0), ( W // 2, H ), (0, 0, 0), 3 )
        cv2.putText( frame, "-Prediction border - Entrance-", (10, H - ( (i * 20) + 200 )),
                    cv2.FONT_HERSHEY_TRIPLEX , 0.5, (255, 255, 255), 1 )
        
        # Use centroid tracker to associate old and new object centroids
        objects = ct.update( rects )

        # Loop over tracked objects
        for ( objectID, centroid ) in objects.items():
            # Check if trackable object exists for current objectID
            to = trackableObjects.get( objectID, None )

            # If they don't exist - create them
            if to is None:
                to = trackableObjects( objectID, centroid )

            # Otherwise, they do exist
            else:
                # Determine direction
                x = [ c[0] for c in to.centroids ]
                direction = centroid[0] - np.mean(x)
                to.centroids.append( centroid )

                # Check if the object has been counted
                if not to.counted:
                    # If direction < 0 (moving left)
                    # AND centroid is to the left of the line - 
                    # count the object
                    if direction < 0 and centroid[0] < W // 2:
                        totalLeft += 1
                        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        move_out.append( totalLeft )
                        out_time.append( date_time )
                        to.counted = True

                    # If direction > 0 (moving right)
                    # AND centroid is to the right of the line - 
                    # count the object
                    elif direction > 0 and centroid[0] > W // 2:
                        totalRight += 1
                        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        move_in.append( totalRight )
                        out_time.append( date_time )
                        to.counted = True

                        # If people limit exceeds Threshold, send email alert
                        if sum( total ) >= config[ "Threshold" ]:
                            cv2.putText( frame, "-ALERT: A lot of people-", (10, frame.shape[0] - 80),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2 )
                            if config[ "ALERT" ]:
                                logger.info( "Sending email alert..." )
                                email_thread = threading.Thread( target = send_mail )
                                email_thread.daemon = True
                                email_thread.start()
                                logger.info( "Alert sent!" )
                        to.counted = True

                        # Total inside:
                        total = []
                        total.append( len(move_in) - len(move_out) )

            # Store trackable objects in dictionary
            trackableObjects[ objectID ] = to

            # Draw ID and centroid of object
            text = "ID: {}".format( objectID )
            cv2.putText( frame, text, ( centroid[0] + 10, centroid[1] + 10 ),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), -1 )
            cv2.circle( frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1 )

        # Construct tuple of info to display
        info_status = [
            ("Exit: ", totalLeft),
            ("Enter: ", totalRight),
            ("Status: ", status)
        ]

        info_total = [
            ("Total people inside", ", ".join( map( str, total ) ) ),
        ]

        # Display output
        for ( i, (k, v) ) in enumerate( info_status ):
            text = "{}{}".format( k, v )
            cv2.putText( frame, text, ( 10, H - ( ( i * 20 ) + 20 ) ),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 2 )
        
        for ( i, (k, v) ) in enumerate( info_total ):
            text = "{}{}".format( k, v )
            cv2.putText( frame, text, ( 265, H - ( ( i * 20 ) + 60 ) ),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 2 )

        # Log to save counting data
        if config["Log"]:
            log_data( move_in, in_time, move_out, out_time )

        # Check if we need to write frame to disk
        if writer is not None:
            writer.write( frame )

        # Show output frame
        cv2.imshow("Live", frame)
        key = cv2.waitKey(1) & 0xFF
        # If 'q' is pressed, escape the loop
        if key == ord("q"):
            break

        # Increment total frames processed thus far
        # Update FPS counter
        totalFrames += 1
        fps.update()

        # Create timer
        if config["Timer"]:
            # Auto-timer to stop live-stream (8 hours)
            end_time = time.time()
            num_seconds = ( end_time - start_time )
            if num_seconds > 28800:
                break

        
