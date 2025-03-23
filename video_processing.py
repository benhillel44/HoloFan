import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from pathlib import Path


NUM_OF_LEDS = 50
FAN_FREQ = 20 # num of rotations in a sec
LED_SWITCH_FREQ = 90 #num of led switches in a sec
SHOW_LED_DURATION = 1/LED_SWITCH_FREQ

original_video_path = 'Hologram skeleton dan.mp4'
time_between_snapshots = 0.1  # Time step between snapshots in seconds


def RGB_led(given_r, r_values, RGB_values) :
    led_index = np.abs(r_values - given_r).argmin()
    return RGB_values[led_index]
def rgb_snap_to_theta_rgb(snapshot, num_of_leds):
    theta_dict = {f"{i}":[] for i in range(360)}
    height = len(snapshot)
    width = len(snapshot[0])
    centerx, center_y = [width//2, height//2] # [x,y]
    radius = 1/2 * min(height, width)
    for i in range(len(snapshot)):
        for j in range(len(snapshot[0])):
            new_x, new_y = j - centerx, (height - i) - center_y
            r = np.sqrt((new_x)**2 + (new_y)**2)
            if r <= radius:
                closest_pixel_theta = int(np.degrees(np.arctan2(new_y, new_x)))
                # if np.abs(closest_pixel_theta - np.degrees(np.arctan2(new_y, new_x))) < 0.5:
                if closest_pixel_theta < 0:
                    closest_pixel_theta += 360
                theta_dict[f"{closest_pixel_theta}"].append((snapshot[i][j], r))
    for theta,theta_array in theta_dict.items():
        sorted_theta_array = sorted(theta_array, key=lambda x: x[1])
        r_values = np.array([item[1] for item in sorted_theta_array])
        RGB_values = np.array([item[0] for item in sorted_theta_array]) / 255.0
        leds_dr = radius / num_of_leds
        theta_RGB_leds_array = np.array([RGB_led(leds_dr * k, r_values, RGB_values) for k in range(num_of_leds)])
        theta_dict[theta] = theta_RGB_leds_array

        r_array = np.array([(leds_dr * k) for k in range(num_of_leds)])
        # new_x_array, new_y_array = np.cos(np.radians(int(theta)))*new_r_array, np.sin(np.radians(int(theta)))*new_r_array
        # plt.scatter(new_x_array, new_y_array, c = theta_RGB_leds_array)
        # print(theta + f"{theta_dict[theta]}")
    # plt.show()

    return theta_dict, r_array

    # print(center)
    # print(f"height: {height}, width: {width}")

def snapshots_to_theta_arrays(snapshots, snapshots_time_array):
    t=0
    data_list = []
    data_duration_array = []
    for i, snapshot in enumerate(snapshots[:-1]):
        next_snap_time = snapshots_time_array[i+1]
        snapshot_theta_dict, snapshot_r_array = rgb_snap_to_theta_rgb(snapshot, NUM_OF_LEDS)
        while t <= next_snap_time:
            num_blades = 6  # number of angles/blades
            angle_step = 360 / num_blades
            fan_theta = int((FAN_FREQ * 360 * t) % 360)
            # Lists to hold the arrays
            x_arrays = []
            y_arrays = []
            RGB_arrays = []

            for i in range(num_blades):
                angle = int((fan_theta + i * angle_step) % 360)
                angle_rad = np.radians(angle)

                x_array = np.cos(angle_rad) * snapshot_r_array
                y_array = np.sin(angle_rad) * snapshot_r_array
                RGB_array = snapshot_theta_dict[f"{int(angle)}"]  # Assuming keys are string ints like "0", "1", ..., "359"

                x_arrays.append(x_array)
                y_arrays.append(y_array)
                RGB_arrays.append(RGB_array)

            # Concatenate all arrays into full ones
            full_x_array = np.concatenate(x_arrays)
            full_y_array = np.concatenate(y_arrays)
            full_RGB_array = np.concatenate(RGB_arrays)
            data_list.append((full_x_array, full_y_array, full_RGB_array))
            data_duration_array.append(SHOW_LED_DURATION)
            t += SHOW_LED_DURATION
    return data_list, data_duration_array

def video_to_snapshots(video_path, time_between_snapshots):
    snapshots_time_array = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"fps: {fps}")
    frame_interval = int(fps * time_between_snapshots)  # How many frames to skip
    print(frame_interval)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Video info -> FPS: {fps}, Total Frames: {frame_count}, Duration: {duration:.2f}s")

    frame_idx = 0
    end_time = 5
    snapshots = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0 and frame_idx < int(fps * end_time):
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Now rgb_frame is a NumPy array of shape (height, width, 3)
            snapshots.append(rgb_frame)
            snapshots_time_array.append(frame_idx / fps)
            print(f"Captured frame at time {frame_idx / fps:.2f}s -> Matrix shape: {rgb_frame.shape}")
        frame_idx += 1
    cap.release()

    print(f"\nTotal snapshots taken: {len(snapshots)}")
    return snapshots, snapshots_time_array

def make_a_fan_video(data_list, data_duration_array):
    fps = 100  # frames per second
    frame_size = (640, 480)  # size of video frames (width, height)

    # Folder to save temporary images
    output_folder = r"C:\talpiot\semester f\hologram fan\temp_place_holder"
    image_paths = []

    for idx, (x, y, RGB_array) in enumerate(data_list):
        plt.figure(figsize=(6.4, 4.8), facecolor='black')  # figure background

        ax = plt.gca()
        ax.set_facecolor('black')  # plot area background
        plt.scatter(x, y, c = RGB_array)
        plt.title(f'Plot {idx + 1}', color='white')
        plt.xlabel('X-axis', color='white')
        plt.ylabel('Y-axis', color='white')
        plt.xlim(-180,180)
        plt.ylim(-180, 180)

        # Set the tick params to white as well
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        img_path = output_folder + "\\" + f'plot_{idx}.png'
        plt.savefig(img_path)
        plt.close()
        image_paths.append(str(img_path))

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('output_video.mp4', fourcc, fps, frame_size)

    for img_path, duration in zip(image_paths, data_duration_array):
        img = cv2.imread(img_path)
        img = cv2.resize(img, frame_size)
        num_frames = int(duration * fps)
        for _ in range(num_frames):
            video_writer.write(img)

    video_writer.release()
    print("Video successfully created!")



snapshots, snapshots_time_array = video_to_snapshots(original_video_path, time_between_snapshots)
data_list, data_duration_array = snapshots_to_theta_arrays(snapshots, snapshots_time_array)
make_a_fan_video(data_list, data_duration_array)

