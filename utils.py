import cv2
import matplotlib.pyplot as plt

def infer_test_dataset(data_path):

    cap = cv2.VideoCapture(data_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while True:
        if input("Type n to stop, press enter to ocntinue: ") == "n":
            break

        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_rgb = frame[:, :, ::-1]  # Convert BGR to RGB

        plt.imshow(frame_rgb)
        plt.title(f"Prediction for Frame {frame_count}")
        plt.show(block=True) #block = True ensures that the next frame is not displayed until the current one is closed.
        frame_count += 1

    cap.release()
