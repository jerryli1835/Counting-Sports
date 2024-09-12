from util import create_video_from_images
from rope_skipping_counter import RopeCounter
from situp_counter import SitupCounter
from pullup_counter import PullupCounter
import os
import gradio as gr


def count_rope_skipping(input_path, activate_coef=0.2, time_buffer=30, method='rtmlib'):
    counter = RopeCounter(input_path, activate_coef, time_buffer)
    if method == 'mediapipe':
        counter.count_plot(display=False)
        dot_index = input_path.rfind('.')
        output_path = input_path[:dot_index] + "_output" + input_path[dot_index:]
    elif method == 'rtmlib':
        counter.count_plot_rtmlib(display=False)
        dot_index = input_path.rfind('.')
        output_path = input_path[:dot_index] + "_output" + input_path[dot_index:]
    elif method == 'mediapipe_data':
        current_directory = os.getcwd()
        output_image_dir = os.path.join(current_directory, "rope_gradio")
        output_path = os.path.join(current_directory, "output_rope.mp4")
        counter.plot_image_save(output_image_dir)
        create_video_from_images(output_image_dir, output_path)
    elif method == 'rtmlib_data':
        current_directory = os.getcwd()
        output_image_dir = os.path.join(current_directory, "rope_gradio")
        output_path = os.path.join(current_directory, "output_rope.mp4")
        counter.plot_image_save_rtmlib(output_image_dir)
        create_video_from_images(output_image_dir, output_path)
    else:
        raise ValueError(f"Method '{method}' is not supported. The supported methods are 'mediapipe', 'rtmlib', 'mediapipe_data' and 'rtmlib_data'.")

    return output_path


demo_rope = gr.Interface(
                         fn=count_rope_skipping,
                         inputs=["video", gr.Slider(minimum=0, maximum=1), gr.Slider(minimum=0, maximum=100), "text"],
                         outputs=["video"])

demo_rope.launch(share=True)


def count_situp(input_path, activate_coef=0.6, time_buffer=30, method='rtmlib'):
    counter = SitupCounter(input_path, activate_coef, time_buffer)
    if method == 'mediapipe':
        counter.count_plot(display=False)
        dot_index = input_path.rfind('.')
        output_path = input_path[:dot_index] + "_output" + input_path[dot_index:]
    elif method == 'rtmlib':
        counter.count_plot_rtmlib(display=False)
        dot_index = input_path.rfind('.')
        output_path = input_path[:dot_index] + "_output" + input_path[dot_index:]
    elif method == 'mediapipe_data':
        current_directory = os.getcwd()
        output_image_dir = os.path.join(current_directory, "situp_gradio")
        output_path = os.path.join(current_directory, "output_situp.mp4")
        counter.plot_image_save(output_image_dir)
        create_video_from_images(output_image_dir, output_path)
    elif method == 'rtmlib_data':
        current_directory = os.getcwd()
        output_image_dir = os.path.join(current_directory, "situp_gradio")
        output_path = os.path.join(current_directory, "output_situp.mp4")
        counter.plot_image_save_rtmlib(output_image_dir)
        create_video_from_images(output_image_dir, output_path)
    else:
        raise ValueError(f"Method '{method}' is not supported. The supported methods are 'mediapipe', 'rtmlib', 'mediapipe_data' and 'rtmlib_data'.")

    return output_path


demo_situp = gr.Interface(
                         fn=count_situp,
                         inputs=["video", gr.Slider(minimum=0, maximum=1), gr.Slider(minimum=0, maximum=100), "text"],
                         outputs=["video"])

demo_situp.launch(share=True)


def count_pullup(input_path, activate_coef=0.2, time_buffer=30, method='rtmlib'):
    counter = PullupCounter(input_path, activate_coef, time_buffer)
    if method == 'mediapipe':
        counter.count_plot(display=False)
        dot_index = input_path.rfind('.')
        output_path = input_path[:dot_index] + "_output" + input_path[dot_index:]
    elif method == 'rtmlib':
        counter.count_plot_rtmlib(display=False)
        dot_index = input_path.rfind('.')
        output_path = input_path[:dot_index] + "_output" + input_path[dot_index:]
    elif method == 'mediapipe_data':
        current_directory = os.getcwd()
        output_image_dir = os.path.join(current_directory, "pullup_gradio")
        output_path = os.path.join(current_directory, "output_pullup.mp4")
        counter.plot_image_save(output_image_dir)
        create_video_from_images(output_image_dir, output_path)
    elif method == 'rtmlib_data':
        current_directory = os.getcwd()
        output_image_dir = os.path.join(current_directory, "pullup_gradio")
        output_path = os.path.join(current_directory, "output_pullup.mp4")
        counter.plot_image_save_rtmlib(output_image_dir)
        create_video_from_images(output_image_dir, output_path)
    else:
        raise ValueError(f"Method '{method}' is not supported. The supported methods are 'mediapipe', 'rtmlib', 'mediapipe_data' and 'rtmlib_data'.")

    return output_path


demo_pullup = gr.Interface(
                         fn=count_pullup,
                         inputs=["video", gr.Slider(minimum=0, maximum=1), gr.Slider(minimum=0, maximum=100), "text"],
                         outputs=["video"])

demo_pullup.launch(share=True)
