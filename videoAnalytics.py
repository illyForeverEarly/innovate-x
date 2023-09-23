import cv2
import tensorflow as tf

# Load SSD MobileNet
model = tf.saved_model.load("./ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8(1)/")

# Camera capture
cap = cv2.VideoCapture(0) #default TODO: download video test

while True:
    # Capture frame
    ret, frame = cap.read()

    # Object detection
    detections = model(frame)

    # Process detections
    num_people = len(detections)

    # TODO: store/analyze data

    # Display
    cv2.imshow('Live', frame)

    # 'q' to exit
    if cv2.waitkey(1) & 0xFF == ord('q'):
        break

# Release camera
cap.release
# Close OpenCV windows
cv2.destroyAllWindows()