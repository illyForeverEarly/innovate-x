import queue
import threading
import cv2

# Create threading class
class ThreadingClass:
  
  def __init__(self, name):
    self.cap = cv2.VideoCapture( name )
	# Define empty queue
    self.q = queue.Queue()
    # Define thread
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # Read the frames as they are available
  # Removes OpenCV's internal buffer
  # Reduces the frame lag
  def _reader(self):
    while True:
      ret, frame = self.cap.read() # Read frames
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()
        except queue.Empty:
          pass
      self.q.put(frame) # Store frames in queue (not buffer)

  def read(self):
    return self.q.get() # Fetch frames from queue one by one

  def release(self):
    return self.cap.release() # Release the hw resource
