from util import create_video_from_images
from rope_skipping_counter import RopeCounter
from situp_counter import SitupCounter
from pullup_counter import PullupCounter


# experiments for rope skipping
first_trail = RopeCounter('videos/rope_4.mp4', 0.2, 30)
first_trail.count_plot()
first_trail.count_plot_rtmlib()
first_trail.plot_image_save('outputs/rope_4')
first_trail.plot_image_save_rtmlib('outputs/rope_4_rtmlib')
first_trail.plot_image_save_AllLandmarks('outputs/rope_4_landmarks', [0, 1, 11, 12, 13, 14, 15, 15, 23, 24, 25, 26, 27, 28, 31, 32])
first_trail.plot_image_save_AllLandmarks_rtmlib('outputs/rope_4_landmarks_rtmlib', [5, 6, 9, 10, 11, 12])

image_folder_1 = '/Users/lizhengyang/KNQ_codes/SportsCounter/outputs/rope_4_rtmlib'
video_path_1 = '/Users/lizhengyang/KNQ_codes/SportsCounter/outputs/rope_4_rtmlib_output.mp4'
create_video_from_images(image_folder_1, video_path_1)

# experiments for situp
situp_trail2 = SitupCounter('videos/situp3.mp4', 0.6, 30)
situp_trail2.count_plot()
situp_trail2.count_plot_rtmlib()
situp_trail2.plot_image_save('outputs/situp3')
situp_trail2.plot_image_save_rtmlib('outputs/situp3_rtmlib')

image_folder_2 = '/Users/lizhengyang/KNQ_codes/SportsCounter/outputs/situp3'
video_path_2 = '/Users/lizhengyang/KNQ_codes/SportsCounter/outputs/situp3output.mp4'
create_video_from_images(image_folder_2, video_path_2)

# experiments for pullup
pullup_trail1 = PullupCounter('videos/pullup2.mp4', 0.2, 30)
pullup_trail1.count_plot()
pullup_trail1.count_plot_rtmlib()
pullup_trail1.plot_image_save('outputs/pullup2')
pullup_trail1.plot_image_save_rtmlib('outputs/pullup2_rtmlib')

pullup_trail1.plot_image_save_AllLandmarks('outputs/pullup2_landmarks', [0, 1, 11, 12, 13, 14, 15, 15, 23, 24, 25, 26, 27, 28, 31, 32])
pullup_trail1.plot_image_save_AllLandmarks_rtmlib('outputs/pullup2_landmarks_rtmlib', [5, 6, 9, 10, 11, 12])

image_folder_3 = '/Users/lizhengyang/KNQ_codes/SportsCounter/outputs/pullup2'
video_path_3 = '/Users/lizhengyang/KNQ_codes/SportsCounter/outputs/pullup2_output.mp4'
create_video_from_images(image_folder_3, video_path_3)

# experiments for genreal pose landmarks
general_trail1 = PullupCounter('videos/toby_general.mp4', 0.2, 30)
general_trail1.plot_image_save_AllLandmarks('outputs/general_landmarks', [0, 1, 11, 12, 13, 14, 15, 15, 23, 24, 25, 26, 27, 28, 31, 32])
general_trail1.plot_image_save_AllLandmarks_rtmlib('outputs/general_landmarks_rtmlib', [5, 6, 9, 10, 11, 12])

image_folder_4 = '/Users/lizhengyang/KNQ_codes/SportsCounter/outputs/general_landmarks_rtmlib'
video_path_4 = '/Users/lizhengyang/KNQ_codes/SportsCounter/outputs/general_landmarks_rtmlib.mp4'
create_video_from_images(image_folder_4, video_path_4)
