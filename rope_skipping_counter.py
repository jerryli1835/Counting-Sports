import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import os
import shutil
import time
from rtmlib import draw_skeleton, RTMPose, Body
from util import BufferList


# main class of implementing the rope skipping counter
class RopeCounter:

    def __init__(self, file_name, activate_coef, buffer_time):
        # landmark indices for hip
        self.selected_landmarks_1 = [23, 24]
        self.selected_landmarks_rtmlib_1 = [11, 12]
        # landmark indices for shoulder
        self.selected_landmarks_2 = [11, 12]
        self.selected_landmarks_rtmlib_2 = [5, 6]
        # some default values
        self.cy_max = 100
        self.cy_min = 100
        # the flag is 150 if the person is currently in up pose; 250 if in down pose
        self.flip_flag = 250
        self.prev_flip_flag = 250

        # time window length: if the video is 30 fps, the loading time is 1 second for buffer_time=30
        self.buffer_time = buffer_time
        # video file name
        self.file_name = file_name
        # sensitivity of detection of jumping
        self.activate_coef = activate_coef

        # constructing the buffer lists
        self.center_y = BufferList(buffer_time)
        self.center_y_up = BufferList(buffer_time)
        self.center_y_down = BufferList(buffer_time)
        self.center_y_pref_flip = BufferList(buffer_time)
        self.center_y_flip = BufferList(buffer_time)
        self.landmark_1_buffer = BufferList(buffer_time)
        self.landmark_2_buffer = BufferList(buffer_time)
        self.shoulder_hip_buffer = BufferList(buffer_time)
        self.bound_switch_up = BufferList(buffer_time)
        self.bound_switch_down = BufferList(buffer_time)
        self.critirion_1 = BufferList(buffer_time)

    def count_plot(self, display=True):
        """
        Method for implementing the counting and live displaying the processed video and data,
        using mediapipe as model.
        Save the processed video in the SportsCounter/videos folder.
        """
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        # some values for initialisation
        cy_max = self.cy_max
        cy_min = self.cy_min
        flip_flag = self.flip_flag
        prev_flip_flag = self.prev_flip_flag
        count = 0
        frame_num = 0

        # For webcam input:
        cap = cv2.VideoCapture(self.file_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.file_name.replace(".mp4", "_output.mp4"),
            fourcc,
            20.0,
            (int(cap.get(3)), int(cap.get(4))),
        )
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                image_height, image_width, _ = image.shape

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                    # x, y coordinates for hip
                    landmarks_1 = [
                        (lm.x * image_width, lm.y * image_height)
                        for i, lm in enumerate(results.pose_landmarks.landmark)
                        if i in self.selected_landmarks_1
                    ]
                    # mean coordinates for hip
                    cx = int(np.mean([x[0] for x in landmarks_1]))
                    cy_1 = int(np.mean([x[1] for x in landmarks_1]))
                    self.landmark_1_buffer.push(cy_1)

                    # x, y coordinates for shoulder
                    landmarks_2 = [
                        (lm.x * image_width, lm.y * image_height)
                        for i, lm in enumerate(results.pose_landmarks.landmark)
                        if i in self.selected_landmarks_2
                    ]
                    # mean coordinates for shoulder
                    cy_2 = int(np.mean([x[1] for x in landmarks_2]))
                    self.landmark_2_buffer.push(cy_2)
                    cy_shoulder_hip = cy_1 - cy_2
                    self.shoulder_hip_buffer.push(cy_shoulder_hip)

                else:
                    cx = 0
                    cy_1 = 0
                    cy_shoulder_hip = 0

                # center coordinate of hip position
                cy = int((cy_1 + self.center_y.buffer[-1]) / 2)
                # set data
                self.center_y.push(cy)

                # upper bound for center coordinate of hip position
                cy_max = 0.5 * cy_max + 0.5 * self.center_y.max()
                self.center_y_up.push(cy_max)

                # lower bound for center coordinate of hip position
                cy_min = 0.5 * cy_min + 0.5 * self.center_y.min()
                self.center_y_down.push(cy_min)

                # store the previous flip flag
                prev_flip_flag = flip_flag
                self.center_y_pref_flip.push(prev_flip_flag)

                # the distance between the upper bound and the lower bound
                dy = cy_max - cy_min

                # two criteria for detecting a sub jump process
                self.bound_switch_up.push(cy_min + 0.35 * dy)
                self.bound_switch_down.push(cy_max - 0.55 * dy)

                # detect whether the jump amplitude is large enough for the person
                self.critirion_1.push(dy - self.activate_coef * cy_shoulder_hip)
                if dy > self.activate_coef * cy_shoulder_hip:
                    # current is up pose
                    if cy > cy_max - 0.55 * dy and flip_flag == 150:
                        flip_flag = 250
                    # current is down pose
                    if 0 < cy < cy_min + 0.35 * dy and flip_flag == 250:
                        flip_flag = 150

                # store the current flip flag
                self.center_y_flip.push(flip_flag)

                if prev_flip_flag < flip_flag:
                    # print(0.9 * cy_max, cy_max, cy_max-cy_min, (0.1 * cy_max) / (cy_max-cy_min))
                    # update the count if there is a period of jump
                    count = count + 1

                # plot and display the results
                if frame_num <= self.buffer_time:
                    cv2.putText(
                        image,
                        "Loading the buffer",
                        (int(image_width * 0.6), int(image_height * 0.4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3,
                    )
                else:
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(
                        image,
                        "centroid",
                        (cx - 25, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                    cv2.putText(
                        image,
                        "count = " + str(count),
                        (int(image_width * 0.6), int(image_height * 0.4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        3,
                    )
                if display:
                    plt.clf()
                    plt.plot(self.center_y.buffer, label="center_y")
                    plt.plot(self.center_y_up.buffer, label="center_y_up")
                    plt.plot(self.center_y_down.buffer, label="center_y_down")
                    plt.plot(self.center_y_pref_flip.buffer, label="center_y_pref_flip")
                    plt.plot(self.center_y_flip.buffer, label="center_y_flip")
                    plt.plot(self.landmark_1_buffer.buffer, label="landmark_1")
                    plt.plot(self.landmark_2_buffer.buffer, label="landmark_2")
                    # plt.plot(shoulder_hip_buffer.buffer, label="shoulder_hip")
                    plt.plot(self.bound_switch_up.buffer, label="switch_up")
                    plt.plot(self.bound_switch_down.buffer, label="switch_down")
                    plt.plot(self.critirion_1.buffer, label="critirion_1")

                    plt.gca().invert_yaxis()

                    plt.legend(loc="upper right")
                    plt.pause(0.1)

                frame_num += 1

                # display.
                if display:
                    cv2.imshow("MediaPipe Pose", image)

                out.write(image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def count_plot_rtmlib(self, pose='local_rtmpose', mode='tiny', display=True):
        """
        Method for implementing the counting and live displaying the processed video and data,
        using rtmlib as model. Save the processed video in the SportsCounter/videos folder.
        """
        # set some configurations
        device = 'cpu'
        backend = 'onnxruntime'
        openpose_skeleton = False

        # get the model
        pose_model = Body(pose=pose, to_openpose=openpose_skeleton, mode=mode, backend=backend, device=device)

        # some values for initialisation
        cy_max = self.cy_max
        cy_min = self.cy_min
        flip_flag = self.flip_flag
        prev_flip_flag = self.prev_flip_flag
        count = 0
        frame_num = 1

        # For webcam input:
        cap = cv2.VideoCapture(self.file_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.file_name.replace(".mp4", "_output.mp4"),
            fourcc,
            20.0,
            (int(cap.get(3)), int(cap.get(4))),
        )

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image_height, image_width, _ = image.shape

            # inference the pose
            keypoints, scores = pose_model(image)

            # filter results (getting the one closest to the center of the image horizontally)
            if keypoints.shape[0] > 1:
                coords_mat = np.zeros((keypoints.shape[0], 4))
                for q in range(keypoints.shape[0]):
                    coords_mat[q, 0] = keypoints[q, 5, 0] - image_width/2
                    coords_mat[q, 1] = keypoints[q, 6, 0] - image_width/2
                    coords_mat[q, 2] = keypoints[q, 11, 0] - image_width/2
                    coords_mat[q, 3] = keypoints[q, 12, 0] - image_width/2
                mean_coords = np.mean(coords_mat, axis=1)
                mean_coords_squared = mean_coords ** 2
                max_mean_row_index = np.argmin(mean_coords_squared)
                keypoints = keypoints[max_mean_row_index:max_mean_row_index+1, :, :]

            img_show = image.copy()

            # visualise the pose predictions
            img_show = draw_skeleton(img_show,
                                     keypoints,
                                     scores,
                                     openpose_skeleton=openpose_skeleton,
                                     kpt_thr=0.3,
                                     line_width=2)

            if keypoints is not None:
                # x, y coordinates for hip
                list_1_tmp = self.selected_landmarks_rtmlib_1
                landmarks_1 = [(keypoints[0, list_1_tmp[0], 0], keypoints[0, list_1_tmp[0], 1]),
                               (keypoints[0, list_1_tmp[1], 0], keypoints[0, list_1_tmp[1], 1])]
                # mean coordinates for hip
                cx = int(np.mean([x[0] for x in landmarks_1]))
                cy_1 = int(np.mean([x[1] for x in landmarks_1]))
                self.landmark_1_buffer.push(cy_1)

                # x, y coordinates for shoulder
                list_2_tmp = self.selected_landmarks_rtmlib_2
                landmarks_2 = [(keypoints[0, list_2_tmp[0], 0], keypoints[0, list_2_tmp[0], 1]),
                               (keypoints[0, list_2_tmp[1], 0], keypoints[0, list_2_tmp[1], 1])]
                # mean coordinates for shoulder
                cy_2 = int(np.mean([x[1] for x in landmarks_2]))
                self.landmark_2_buffer.push(cy_2)
                cy_shoulder_hip = cy_1 - cy_2
                self.shoulder_hip_buffer.push(cy_shoulder_hip)

            else:
                cx = 0
                cy_1 = 0
                cy_shoulder_hip = 0

            # center coordinate of hip position
            cy = int((cy_1 + self.center_y.buffer[-1]) / 2)
            # set data
            self.center_y.push(cy)

            # upper bound for center coordinate of hip position
            cy_max = 0.5 * cy_max + 0.5 * self.center_y.max()
            self.center_y_up.push(cy_max)

            # lower bound for center coordinate of hip position
            cy_min = 0.5 * cy_min + 0.5 * self.center_y.min()
            self.center_y_down.push(cy_min)

            # store the previous flip flag
            prev_flip_flag = flip_flag
            self.center_y_pref_flip.push(prev_flip_flag)

            # the distance between the upper bound and the lower bound
            dy = cy_max - cy_min

            # two criteria for detecting a sub jump process
            self.bound_switch_up.push(cy_min + 0.35 * dy)
            self.bound_switch_down.push(cy_max - 0.55 * dy)

            # detect whether the jump amplitude is large enough for the person
            self.critirion_1.push(dy - self.activate_coef * cy_shoulder_hip)
            if dy > self.activate_coef * cy_shoulder_hip:
                # current is up pose
                if cy > cy_max - 0.55 * dy and flip_flag == 150:
                    flip_flag = 250
                # current is down pose
                if 0 < cy < cy_min + 0.35 * dy and flip_flag == 250:
                    flip_flag = 150

            # store the current flip flag
            self.center_y_flip.push(flip_flag)

            if prev_flip_flag < flip_flag:
                # print(0.9 * cy_max, cy_max, cy_max-cy_min, (0.1 * cy_max) / (cy_max-cy_min))
                # update the count if there is a period of jump
                count = count + 1

            # plot and display the results
            if frame_num <= self.buffer_time:
                cv2.putText(
                    img_show,
                    "Loading the buffer",
                    (int(image_width * 0.6), int(image_height * 0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                )
            else:
                cv2.circle(img_show, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    img_show,
                    "centroid",
                    (cx - 25, cy - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                cv2.putText(
                    img_show,
                    "count = " + str(count),
                    (int(image_width * 0.6), int(image_height * 0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                )

            if display:
                plt.clf()
                plt.plot(self.center_y.buffer, label="center_y")
                plt.plot(self.center_y_up.buffer, label="center_y_up")
                plt.plot(self.center_y_down.buffer, label="center_y_down")
                plt.plot(self.center_y_pref_flip.buffer, label="center_y_pref_flip")
                plt.plot(self.center_y_flip.buffer, label="center_y_flip")
                plt.plot(self.landmark_1_buffer.buffer, label="landmark_1")
                plt.plot(self.landmark_2_buffer.buffer, label="landmark_2")
                # plt.plot(shoulder_hip_buffer.buffer, label="shoulder_hip")
                plt.plot(self.bound_switch_up.buffer, label="switch_up")
                plt.plot(self.bound_switch_down.buffer, label="switch_down")
                plt.plot(self.critirion_1.buffer, label="critirion_1")

                plt.gca().invert_yaxis()

                plt.legend(loc="upper right")
                plt.pause(0.1)

            frame_num += 1

            # display.
            if display:
                cv2.imshow("rtmlib pose", img_show)
            out.write(img_show)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def plot_image_save(self, save_folder_path):
        """
        Method for implementing the counting using mediapipe as model,
        and save the processed images along with the data evolution of landmarks in the SportsCounter/outputs folder.

        Input:
        save_folder_path (str): the target folder to save all the processed images.
        """
        # Check if the directory exists and delete it if it does
        if os.path.exists(save_folder_path):
            shutil.rmtree(save_folder_path)
        os.makedirs(save_folder_path, exist_ok=True)

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        # some values for initialisation
        cy_max = self.cy_max
        cy_min = self.cy_min
        flip_flag = self.flip_flag
        prev_flip_flag = self.prev_flip_flag
        count = 0
        frame_num = 0

        # For webcam input:
        cap = cv2.VideoCapture(self.file_name)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                last_six_digits = str(frame_num).zfill(6)
                file_name = f'{last_six_digits}.jpg'
                save_file_path = os.path.join(save_folder_path, file_name)

                image_height, image_width, _ = image.shape

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                    # x, y coordinates for hip
                    landmarks_1 = [
                        (lm.x * image_width, lm.y * image_height)
                        for i, lm in enumerate(results.pose_landmarks.landmark)
                        if i in self.selected_landmarks_1
                    ]
                    # mean coordinates for hip
                    cx = int(np.mean([x[0] for x in landmarks_1]))
                    cy_1 = int(np.mean([x[1] for x in landmarks_1]))
                    self.landmark_1_buffer.push(cy_1)

                    # x, y coordinates for shoulder
                    landmarks_2 = [
                        (lm.x * image_width, lm.y * image_height)
                        for i, lm in enumerate(results.pose_landmarks.landmark)
                        if i in self.selected_landmarks_2
                    ]
                    # mean coordinates for shoulder
                    cy_2 = int(np.mean([x[1] for x in landmarks_2]))
                    self.landmark_2_buffer.push(cy_2)
                    cy_shoulder_hip = cy_1 - cy_2
                    self.shoulder_hip_buffer.push(cy_shoulder_hip)
                else:
                    cx = 0
                    cy_1 = 0
                    cy_shoulder_hip = 0

                # center coordinate of hip position
                cy = int((cy_1 + self.center_y.buffer[-1]) / 2)
                # set data
                self.center_y.push(cy)

                # upper bound for center coordinate of hip position
                cy_max = 0.5 * cy_max + 0.5 * self.center_y.max()
                self.center_y_up.push(cy_max)

                # lower bound for center coordinate of hip position
                cy_min = 0.5 * cy_min + 0.5 * self.center_y.min()
                self.center_y_down.push(cy_min)

                # store the previous flip flag
                prev_flip_flag = flip_flag
                self.center_y_pref_flip.push(prev_flip_flag)

                # the distance between the upper bound and the lower bound
                dy = cy_max - cy_min

                # two criteria for detecting a sub jump process
                self.bound_switch_up.push(cy_min + 0.35 * dy)
                self.bound_switch_down.push(cy_max - 0.55 * dy)

                # detect whether the jump amplitude is large enough for the person
                self.critirion_1.push(dy - self.activate_coef * cy_shoulder_hip)
                if dy > self.activate_coef * cy_shoulder_hip:
                    # current is up pose
                    if cy > cy_max - 0.55 * dy and flip_flag == 150:
                        flip_flag = 250
                    # current is down pose
                    if 0 < cy < cy_min + 0.35 * dy and flip_flag == 250:
                        flip_flag = 150

                # store the current flip flag
                self.center_y_flip.push(flip_flag)

                if prev_flip_flag < flip_flag:
                    # print(0.9 * cy_max, cy_max, cy_max-cy_min, (0.1 * cy_max) / (cy_max-cy_min))
                    # update the count if there is a period of jump
                    count = count + 1

                # plot and display the results
                if frame_num <= self.buffer_time:
                    cv2.putText(
                        image,
                        "Loading the buffer",
                        (int(image_width * 0.6), int(image_height * 0.4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3,
                    )
                else:
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(
                        image,
                        "centroid",
                        (cx - 25, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                    cv2.putText(
                        image,
                        "count = " + str(count),
                        (int(image_width * 0.6), int(image_height * 0.4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        3,
                    )

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                ax1.plot(self.center_y.buffer, label="center_y")
                ax1.plot(self.center_y_up.buffer, label="center_y_up")
                ax1.plot(self.center_y_down.buffer, label="center_y_down")
                ax1.plot(self.center_y_pref_flip.buffer, label="center_y_pref_flip")
                ax1.plot(self.center_y_flip.buffer, label="center_y_flip")
                ax1.plot(self.landmark_1_buffer.buffer, label="landmark_1")
                ax1.plot(self.landmark_2_buffer.buffer, label="landmark_2")
                # ax1.plot(shoulder_hip_buffer.buffer, label="shoulder_hip")
                ax1.plot(self.bound_switch_up.buffer, label="switch_up")
                ax1.plot(self.bound_switch_down.buffer, label="switch_down")
                ax1.plot(self.critirion_1.buffer, label="critirion_1")

                # ax1.invert_yaxis()
                ax1.set_ylim(image_height, 0)
                ax1.legend(loc="upper right")
                ax1.set_title("Data Evolution")

                # Display the video frame with potential overlays
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax2.imshow(image_rgb)
                ax2.set_xlim(0, image_width)
                ax2.set_ylim(image_height, 0)
                ax2.set_title("Video Frames and Skipping Count")

                plt.tight_layout()
                plt.savefig(save_file_path, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                plt.pause(0.1)

                frame_num += 1

                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

    def plot_image_save_rtmlib(self, save_folder_path, pose='local_rtmpose', mode='tiny'):
        """
        Method for implementing the counting using rtmlib as model,
        and save the processed images along with the data evolution of landmarks in the SportsCounter/outputs folder.

        Input:
        save_folder_path (str): the target folder to save all the processed images.
        pose (str): The model type for tracking pose (Options: 'local_rtmo': use the rtmo models stored in models folder,
            'rtmo': use the rtmo models by downloading online, '
            local_rtmpose': use rtmpose models stored in models folder,
            None: use rtmpose models by downloading online).
        mode (str): the corresponding modes for each pose option (check Body in /SportsCounter/rtmlib/tools/solution/body.py).
        """
        # Check if the directory exists and delete it if it does
        if os.path.exists(save_folder_path):
            shutil.rmtree(save_folder_path)
        os.makedirs(save_folder_path, exist_ok=True)

        # set some configurations
        device = 'cpu'
        backend = 'onnxruntime'
        openpose_skeleton = False
        # get the model
        pose_model = Body(pose=pose, to_openpose=openpose_skeleton, mode=mode, backend=backend, device=device)

        # some values for initialisation
        cy_max = self.cy_max
        cy_min = self.cy_min
        flip_flag = self.flip_flag
        prev_flip_flag = self.prev_flip_flag
        count = 0
        frame_num = 1

        # For webcam input:
        cap = cv2.VideoCapture(self.file_name)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            last_six_digits = str(frame_num).zfill(6)
            file_name = f'{last_six_digits}.jpg'
            save_file_path = os.path.join(save_folder_path, file_name)

            image_height, image_width, _ = image.shape

            # inference the pose
            keypoints, scores = pose_model(image)

            # filter results (getting the one closest to the center of the image horizontally)
            if keypoints.shape[0] > 1:
                coords_mat = np.zeros((keypoints.shape[0], 4))
                for q in range(keypoints.shape[0]):
                    coords_mat[q, 0] = keypoints[q, 5, 0] - image_width/2
                    coords_mat[q, 1] = keypoints[q, 6, 0] - image_width/2
                    coords_mat[q, 2] = keypoints[q, 11, 0] - image_width/2
                    coords_mat[q, 3] = keypoints[q, 12, 0] - image_width/2
                mean_coords = np.mean(coords_mat, axis=1)
                mean_coords_squared = mean_coords ** 2
                max_mean_row_index = np.argmin(mean_coords_squared)
                keypoints = keypoints[max_mean_row_index:max_mean_row_index+1, :, :]

            img_show = image.copy()

            # visualise the pose predictions
            img_show = draw_skeleton(img_show,
                                     keypoints,
                                     scores,
                                     openpose_skeleton=openpose_skeleton,
                                     kpt_thr=0.3,
                                     line_width=2)

            # x, y coordinates for hip
            list_1_tmp = self.selected_landmarks_rtmlib_1
            landmarks_1 = [(keypoints[0, list_1_tmp[0], 0], keypoints[0, list_1_tmp[0], 1]),
                           (keypoints[0, list_1_tmp[1], 0], keypoints[0, list_1_tmp[1], 1])]
            # mean coordinates for hip
            cx = int(np.mean([x[0] for x in landmarks_1]))
            cy_1 = int(np.mean([x[1] for x in landmarks_1]))
            self.landmark_1_buffer.push(cy_1)

            # x, y coordinates for shoulder
            list_2_tmp = self.selected_landmarks_rtmlib_2
            landmarks_2 = [(keypoints[0, list_2_tmp[0], 0], keypoints[0, list_2_tmp[0], 1]),
                           (keypoints[0, list_2_tmp[1], 0], keypoints[0, list_2_tmp[1], 1])]
            # mean coordinates for shoulder
            cy_2 = int(np.mean([x[1] for x in landmarks_2]))
            self.landmark_2_buffer.push(cy_2)
            cy_shoulder_hip = cy_1 - cy_2
            self.shoulder_hip_buffer.push(cy_shoulder_hip)

            # center coordinate of hip position
            cy = int((cy_1 + self.center_y.buffer[-1]) / 2)
            # set data
            self.center_y.push(cy)

            # upper bound for center coordinate of hip position
            cy_max = 0.5 * cy_max + 0.5 * self.center_y.max()
            self.center_y_up.push(cy_max)

            # lower bound for center coordinate of hip position
            cy_min = 0.5 * cy_min + 0.5 * self.center_y.min()
            self.center_y_down.push(cy_min)

            # store the previous flip flag
            prev_flip_flag = flip_flag
            self.center_y_pref_flip.push(prev_flip_flag)

            # the distance between the upper bound and the lower bound
            dy = cy_max - cy_min

            # two criteria for detecting a sub jump process
            self.bound_switch_up.push(cy_min + 0.35 * dy)
            self.bound_switch_down.push(cy_max - 0.55 * dy)

            # detect whether the jump amplitude is large enough for the person
            self.critirion_1.push(dy - self.activate_coef * cy_shoulder_hip)
            if dy > self.activate_coef * cy_shoulder_hip:
                # current is up pose
                if cy > cy_max - 0.55 * dy and flip_flag == 150:
                    flip_flag = 250
                # current is down pose
                if 0 < cy < cy_min + 0.35 * dy and flip_flag == 250:
                    flip_flag = 150

            # store the current flip flag
            self.center_y_flip.push(flip_flag)

            if prev_flip_flag < flip_flag:
                # print(0.9 * cy_max, cy_max, cy_max-cy_min, (0.1 * cy_max) / (cy_max-cy_min))
                # update the count if there is a period of jump
                count = count + 1

            # plot and display the results
            if frame_num <= self.buffer_time:
                cv2.putText(
                    img_show,
                    "Loading the buffer",
                    (int(image_width * 0.6), int(image_height * 0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                )
            else:
                cv2.circle(img_show, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    img_show,
                    "centroid",
                    (cx - 25, cy - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                cv2.putText(
                    img_show,
                    "count = " + str(count),
                    (int(image_width * 0.6), int(image_height * 0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax1.plot(self.center_y.buffer, label="center_y")
            ax1.plot(self.center_y_up.buffer, label="center_y_up")
            ax1.plot(self.center_y_down.buffer, label="center_y_down")
            ax1.plot(self.center_y_pref_flip.buffer, label="center_y_pref_flip")
            ax1.plot(self.center_y_flip.buffer, label="center_y_flip")
            ax1.plot(self.landmark_1_buffer.buffer, label="landmark_1")
            ax1.plot(self.landmark_2_buffer.buffer, label="landmark_2")
            # ax1.plot(shoulder_hip_buffer.buffer, label="shoulder_hip")
            ax1.plot(self.bound_switch_up.buffer, label="switch_up")
            ax1.plot(self.bound_switch_down.buffer, label="switch_down")
            ax1.plot(self.critirion_1.buffer, label="critirion_1")

            # ax1.invert_yaxis()
            ax1.set_ylim(image_height, 0)
            ax1.legend(loc="upper right")
            ax1.set_title("Data Evolution")

            # Display the video frame with potential overlays
            image_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
            ax2.imshow(image_rgb)
            ax2.set_xlim(0, image_width)
            ax2.set_ylim(image_height, 0)
            ax2.set_title("Video Frames and Skipping Count")

            plt.tight_layout()
            plt.savefig(save_file_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            plt.pause(0.1)

            frame_num += 1

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def plot_image_save_AllLandmarks(self, save_folder_path, index_list):
        """
        Method for drawing posture predictions using mediapipe as model,
        and save the processed images along with data evolution of landmarks chosen in the SportsCounter/outputs folder.

        Input:
        save_folder_path (str): the target folder to save all the processed images.
        index_list (List): the list of indices corresponding to the keypoints one would like to examine.
        """
        # Check if the directory exists and delete it if it does
        if os.path.exists(save_folder_path):
            shutil.rmtree(save_folder_path)
        os.makedirs(save_folder_path, exist_ok=True)

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        list_buffer = [BufferList(self.buffer_time) for _ in range(len(index_list))]

        # some values for initialisation
        frame_num = 0

        # For webcam input:
        cap = cv2.VideoCapture(self.file_name)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                last_six_digits = str(frame_num).zfill(6)
                file_name = f'{last_six_digits}.jpg'
                save_file_path = os.path.join(save_folder_path, file_name)

                image_height, image_width, _ = image.shape

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                    for j in range(len(index_list)):
                        landmarks_tmp = [
                            (lm.x * image_width, lm.y * image_height)
                            for i, lm in enumerate(results.pose_landmarks.landmark)
                            if i == index_list[j]
                        ]
                        list_buffer[j].push(landmarks_tmp[0][1])

                # plot and display the results

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                for j in range(len(index_list)):
                    ax1.plot(list_buffer[j].buffer, label=f"landmark_{index_list[j]}")

                # ax1.invert_yaxis()
                ax1.set_ylim(image_height, 0)
                ax1.legend(loc="upper right")
                ax1.set_title("Data Evolution")

                # Display the video frame with potential overlays
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax2.imshow(image_rgb)
                ax2.imshow(image)
                ax2.set_xlim(0, image_width)
                ax2.set_ylim(image_height, 0)
                ax2.set_title("Video Frames")

                plt.tight_layout()
                plt.savefig(save_file_path, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                plt.pause(0.1)

                frame_num += 1

                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

    def plot_image_save_AllLandmarks_rtmlib(self, save_folder_path, index_list, pose='local_rtmpose', mode='tiny'):
        """
        Method for drawing posture predictions using rtmlib as model,
        and save the processed images along with data evolution of landmarks chosen in the SportsCounter/outputs folder.

        Input:
        save_folder_path (str): the target folder to save all the processed images.
        pose (str): The model type for tracking pose (Options: 'local_rtmo': use the rtmo models stored in models folder.
        index_list (List): the list of indices corresponding to the keypoints one would like to examine.
        pose (str): The model type for tracking pose (Options: 'local_rtmo': use the rtmo models stored in models folder,
            'rtmo': use the rtmo models by downloading online, '
            local_rtmpose': use rtmpose models stored in models folder,
            None: use rtmpose models by downloading online).
        mode (str): the corresponding modes for each pose option (check Body in /SportsCounter/rtmlib/tools/solution/body.py).
        """
        device = 'cpu'
        backend = 'onnxruntime'
        openpose_skeleton = False
        pose_model = Body(pose=pose, to_openpose=openpose_skeleton, mode=mode, backend=backend, device=device)

        # Check if the directory exists and delete it if it does
        if os.path.exists(save_folder_path):
            shutil.rmtree(save_folder_path)
        os.makedirs(save_folder_path, exist_ok=True)

        list_buffer = [BufferList(self.buffer_time) for _ in range(len(index_list))]

        # some values for initialisation
        frame_num = 0

        # For webcam input:
        cap = cv2.VideoCapture(self.file_name)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            s = time.time()
            last_six_digits = str(frame_num).zfill(6)
            file_name = f'{last_six_digits}.jpg'
            save_file_path = os.path.join(save_folder_path, file_name)

            image_height, image_width, _ = image.shape

            keypoints, scores = pose_model(image)
            det_time = time.time() - s
            print(f'Frame {frame_num}: detection time = {det_time:.3f} seconds')

            if keypoints.shape[0] > 1:
                coords_mat = np.zeros((keypoints.shape[0], 4))
                for q in range(keypoints.shape[0]):
                    coords_mat[q, 0] = keypoints[q, 5, 0] - image_width/2
                    coords_mat[q, 1] = keypoints[q, 6, 0] - image_width/2
                    coords_mat[q, 2] = keypoints[q, 11, 0] - image_width/2
                    coords_mat[q, 3] = keypoints[q, 12, 0] - image_width/2
                mean_coords = np.mean(coords_mat, axis=1)
                mean_coords_squared = mean_coords ** 2
                max_mean_row_index = np.argmin(mean_coords_squared)
                keypoints = keypoints[max_mean_row_index:max_mean_row_index+1, :, :]

            img_show = image.copy()

            img_show = draw_skeleton(img_show,
                                     keypoints,
                                     scores,
                                     openpose_skeleton=openpose_skeleton,
                                     kpt_thr=0.3,
                                     line_width=2)

            # img_show = cv2.resize(img_show, (image_width, image_height))

            for j in range(len(index_list)):
                list_buffer[j].push(keypoints[0, index_list[j], 1])

            # plot and display the results

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            for j in range(len(index_list)):
                ax1.plot(list_buffer[j].buffer, label=f"landmark_{index_list[j]}")

            # ax1.invert_yaxis()
            ax1.set_ylim(image_height, 0)
            ax1.legend(loc="upper right")
            ax1.set_title("Data Evolution")

            # Display the video frame with potential overlays
            image_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
            ax2.imshow(image_rgb)
            ax2.imshow(img_show)
            ax2.set_xlim(0, image_width)
            ax2.set_ylim(image_height, 0)
            ax2.set_title("Video Frames")

            plt.tight_layout()
            plt.savefig(save_file_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            plt.pause(0.1)

            frame_num += 1

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
