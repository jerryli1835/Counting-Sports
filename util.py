import cv2
import os


# the buffer list for storing the data in time windows
class BufferList:
    def __init__(self, buffer_time, default_value=0):
        self.buffer = [default_value for _ in range(buffer_time)]

    def push(self, value):
        self.buffer.pop(0)
        self.buffer.append(value)

    def max(self):
        return max(self.buffer)

    def min(self):
        buffer = [value for value in self.buffer if value]
        if buffer:
            return min(buffer)
        return 0


# function for compiling the iamges to a video
def create_video_from_images(image_folder, video_path, fps=25):
    # Get a list of all images in the folder sorted by name

    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg") or img.endswith(".png")]

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', 'X264', etc.
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

        # Write the frame to the video
        video.write(frame)

    # Release the video writer object
    video.release()
