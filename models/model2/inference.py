import os
import glob
import ffmpeg
import numpy as np
from math import floor, log
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from tensorflow.keras import Model  # type: ignore
from tensorflow.keras import Input  # type: ignore
from tensorflow.keras.layers import Conv2D, ReLU, ELU, LeakyReLU, Dropout, Dense, MaxPooling2D, Flatten, BatchNormalization  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping  # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.metrics import classification_report  # type: ignore

IMG_WIDTH = 256
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov')

def get_datagen(use_default_augmentation=True, **kwargs):
    kwargs.update({'rescale': 1./255})
    if use_default_augmentation:
        kwargs.update({
            'rotation_range': 15,
            'zoom_range': 0.2,
            'brightness_range': (0.8, 1.2),
            'channel_shift_range': 30,
            'horizontal_flip': True,
        })
    return ImageDataGenerator(**kwargs)

def predict(model, data, steps=None, threshold=0.5):
    predictions = model.predict(data, steps=steps, verbose=1)
    return predictions, np.where(predictions >= threshold, 1, 0)

def temp(test_data_dir, batch_size, shuffle=False):
    test_datagen = get_datagen(use_default_augmentation=False)
    return test_datagen.flow_from_directory(
        directory=test_data_dir,
        target_size=(IMG_WIDTH, IMG_WIDTH),
        batch_size=batch_size,
        class_mode=None,
        shuffle=shuffle
    )

def convert_video_to_frames(video_path, output_dir, target_fps=5, frame_prefix="frame_"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_pattern = os.path.join(output_dir, f"{frame_prefix}%04d.jpg")
    try:
        (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=target_fps)
            .output(output_pattern, start_number=0)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        saved_frame_count = len([f for f in os.listdir(output_dir) if f.lower().endswith('.jpg')])
        print(f"Extracted {saved_frame_count} frames from {os.path.basename(video_path)}")
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")

def run_inference_on_video_folder(model, video_folder, batch_size=64):
    data_gen = temp(video_folder, batch_size=batch_size, shuffle=False)
    predictions = model.predict(data_gen, verbose=1)
    overall_mean = predictions.mean()
    result = 'Real' if overall_mean > 0.49 else 'Fake'
    return result, overall_mean

def main():
    model_exp = load_model('run_Model2_best_model.keras')
    
    base_vid_dir = os.path.join('vid', 'v')
    if not os.path.exists(base_vid_dir):
        os.makedirs(base_vid_dir)
    
    current_dir = os.getcwd()
    video_files = [f for f in os.listdir(current_dir) if f.lower().endswith(VIDEO_EXTENSIONS)]
    
    if not video_files:
        print("No video files found in the current directory.")
        return 0

    for video_file in video_files:
        video_name, _ = os.path.splitext(video_file)
        video_output_dir = os.path.join(base_vid_dir, video_name)
        output_frames_dir = os.path.join(video_output_dir, 'frames')
        print(f"Processing video: {video_file}")
        
        convert_video_to_frames(os.path.join(current_dir, video_file),
                                output_frames_dir, target_fps=5,
                                frame_prefix="frame_")
        
        result, mean_pred = run_inference_on_video_folder(model_exp, video_output_dir, batch_size=64)
        print(f"Inference result for {video_file}: {result} (mean prediction = {mean_pred:.4f})")
    
    return 0

if __name__ == "__main__":
    main()
