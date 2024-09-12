# Counting-Sports
This is the repository implementing sports counter for rope skipping, situp, and pullup.


# Installation:
After cloning the repository, run:
```bash
cd SportsCounter
pip install -r requirements.txt
pip install -e .
```

# The organisation of the repository:
The "models" folder contains all the ONNX model files utilised by rtmlib. The "output" folder contains sub folders for image frames of different videos. The "rtmlib" folder contains all utiltiies of the [rtmlib repository](https://github.com/Tau-J/rtmlib/tree/main) for pose predictions. The "videos" folder contains all videos for analysis.

# Implementation details:
## Methodology:
The SportsCounter uses human pose detection frameworks to obtain landmark information of the human in picture frames, and analyse the data evolution over time of key parts of the body to perform counting. Now the repository supports three sports: rope skipping, situp and pullup. 
Three files are used to construct the designated class for the sport (such as RopeCounter class in rope_skipping_counter.py), and a range of methods are implemented for counting and plotting utilities. 

## Pose detection frameworks:
### Mediapipe:
MediaPipe is developed by Google, which provides a suite of libraries and tools for you to quickly apply artificial intelligence (AI) and machine learning (ML) techniques in your applications ([Developer manual](https://ai.google.dev/edge/mediapipe/solutions/guide); [GitHub](https://github.com/google-ai-edge/mediapipe)). We use the [Pose Landmark Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) model in mediapipe solutions. Given a picture frame, the model will predict the coordinates of different sections of the body (the model will only choose one person if multiple people are present). The indices of different parts of bodies can be checked on [their webpage](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker). 

### rtmlib:
[Rtmlib](https://github.com/Tau-J/rtmlib/tree/main) is a project under [MMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x) for multi-person body pose detection. Two major methods are implemented in this framework: RTMO and RTMPose. RTMO is a one-stage model which contains both object detection pose prediction, and RTMPose is puely for pose detection and needs to be combined with a detection model such as yolox. Given a picture frame, the model will predict the coordinates of different sections of the body for all people in the frame, and indices of different parts of bodies can be checked in [their repository](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose). 

## Rope skipping:
The core mechanism of counting relies on monitoring the position of hip. When the hip is moving regularly in periods on the vertical axis, a count is added if a peroid is finished (starting from a downward position, crossing the criterion for switching to upward position and then crossing the criterion for switching to downward position).

Buffer lists are created for a certain preset time window, and data are updated to the buffer lists to track the movements of the body. However, although a default value could be set, the buffer lists need to be first completely filled with real data from predictions to be useful for analysis. Hence, there exists a loading time for the buffer. If the video is 30 fps, then a buffer list of length 30 is 1 second of loading time.

To identify a valid skipping, some criteria are also defined to activate the counting. For example, the activate_coef argument in the RopeCounter class is set, so that skipping is valid only when the amplitude of hip movements is larger than activate_coef*(distance between hip and shoulder), where the amplitude is computed by tracking the upper and lower bounds of hip positions over time.

A sequence of methods are implemented for counting and plotting:
- count_plot: implementing the counting and live displaying the processed video and data, using mediapipe as model. Save the processed video in the SportsCounter/videos folder.
- count_plot_rtmlib: implementing the counting and live displaying the processed video and data, using rtmlib as model. Save the processed video in the SportsCounter/videos folder.
- plot_image_save: implementing the counting using mediapipe as model, and saving the processed images along with the data evolution of landmarks in the SportsCounter/outputs folder.
- plot_image_save_rtmlib: implementing the counting using rtmlib as model, and saving the processed images along with the data evolution of landmarks in the SportsCounter/outputs folder.
- plot_image_save_AllLandmarks: drawing posture predictions using mediapipe as model, and saving the processed images along with data evolution of landmarks chosen in the SportsCounter/outputs folder.
- plot_image_save_AllLandmarks_rtmlib: drawing posture predictions using rtmlib as model, and saving the processed images along with data evolution of landmarks chosen in the SportsCounter/outputs folder.

## Situp: 
The core mechanism of counting relies on monitoring the position of shoulder. When the shoulder is moving regularly in periods on the vertical axis, a count is added if down pose is switched to up pose indicated by the flag (starting from a downward position, crossing the criterion for switching to upward position).

To identify a valid situp, some criteria are also defined to activate the counting. For example, the activate_coef argument in the SitupCounter class is set, so that situp is valid only when the amplitude of shoulder movements is larger than activate_coef*(distance between hip and shoulder), where the amplitude is computed by tracking the upper and lower bounds of shoulder positions over time. This could also be used to classify the quality of a situp based on the ratios of amplitude and distance between hip and shoulder. The leg posture is also monitored, that if the leg is flat on the floor, the message will be displayed on the image.

The for counting and plotting are similar to that of rope skipping.

## Pullup: 
The core mechanism of counting relies on monitoring the position of shoulder. When the shoulder is moving regularly in periods on the vertical axis, a count is added if down pose is switched to up pose indicated by the flag (starting from a downward position, crossing the criterion for switching to upward position).

To identify a valid pullup, some criteria are also defined to activate the counting. For example, the pullup is valid only when the amplitude of shoulder movements is larger than the distance between elbow and hand, and the upperbound of the distince between hand and feet should be larger than 2.5*distance between hip and shoulder. This could ensure that the extent of the person's pullup is good enough and that the person is actually hanging on the bar. This could also be used to classify the quality of a pullup based on these distances. 

The for counting and plotting are similar to that of rope skipping.

## Demo:
A demo could be run in SportsCounter/experiment.py, where instances of three classes are constructed, and the counting procedure is implemented.

## Gradio deployment:
Gradio interfaces for each sport are ready for deployment in gradio_SportsCounter.py (demo_rope, demo_situp and demo_pullup), where one can input the video and set the parameters to get the counted video output. 

The "method" keyword argument and associated methods ("method" keyword argument in wrapper function to associated methods in class):

'mediapipe': count_plot; 'rtmlib': count_plot_rtmlib; 'mediapipe_data': plot_image_save; 'rtmlib_data': plot_image_save_rtmlib

