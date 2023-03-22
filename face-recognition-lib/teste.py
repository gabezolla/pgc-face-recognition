from imutils.video import VideoStream
import cv2

# initialize the VideoStream object with the default webcam
vs = VideoStream(usePiCamera=True).start()

# loop over frames from the video stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    cv2.imshow("Frame", frame)

    # if the 'q' key is pressed, stop the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.stop()
cv2.destroyAllWindows()
