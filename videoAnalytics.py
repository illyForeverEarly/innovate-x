import cv2
import tensorflow as tf
import numpy as np
import keyboard

# Load SSD MobileNet
model = tf.saved_model.load("ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model")

# Camera capture
cap = cv2.VideoCapture(0) #default TODO: download video test

while True:
    # Capture frame
    ret, frame = cap.read()

    resized_frame = cv2.resize(frame, (640, 480))
    input_tensor = np.expand_dims(resized_frame, axis=0)

    # Object detection
    detections = model(input_tensor)

    # Process detections
    num_people = len(detections)

    # TODO: store/analyze data
    print(num_people)

    # Display
    cv2.imshow('Live', frame)
    cv2.waitKey(1)

    # 'q' to exit
    if keyboard.is_pressed("q"):
        break

# Release camera
cap.release
# Close OpenCV windows
cv2.destroyAllWindows()