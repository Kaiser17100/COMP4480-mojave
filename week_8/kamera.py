import cv2
import os
import time
import threading


def enable_gazebo_camera():
    """Waits 2 seconds, then tells Gazebo to start streaming."""
    print("[Gazebo] Waiting 2 seconds for OpenCV to get ready...")
    time.sleep(2.0)
    print("[Gazebo] Sending 'enable' signal to Gazebo...")
    topic = '/world/runway/model/uav_1/link/base_link/sensor/nose_camera/image/enable_streaming'
    os.system(f'gz topic -t {topic} -m gz.msgs.Boolean -p "data: 1"')


def main():
    # 1. Start the delayed Gazebo trigger in the background
    threading.Thread(target=enable_gazebo_camera, daemon=True).start()

    # 2. Immediately open OpenCV to listen on port 5600
    print("[OpenCV] Opening UDP port 5600 and waiting for stream...")
    pipeline = "udpsrc port=5600 ! application/x-rtp, payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink drop=true sync=false"

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[OpenCV] ERROR: Failed to open GStreamer pipeline. Check your OpenCV installation.")
        return

    print("[OpenCV] Port opened successfully! Waiting for frames (Window will pop up shortly)...")
    print("[OpenCV] Press 'q' on your keyboard while the video window is focused to quit.")

    # 3. Read and display the frames
    while True:
        ret, frame = cap.read()

        if not ret:
            # If Gazebo hasn't sent frames yet, just loop and wait
            continue

        # Display the frame
        cv2.imshow("Gazebo Camera - Manual Test", frame)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing video stream...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()