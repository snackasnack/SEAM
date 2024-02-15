import numpy as np
import cv2
import argparse
import os

def apply_CAM_to_image(original_image, cam_data):
    # Create an empty list to store CAM data for each class
    cam_data_list = []

    # Iterate over the dictionary items
    for class_index, data in cam_data.items():
        # Normalize the CAM data to range [0, 1]
        normalized_data = (data - data.min()) / (data.max() - data.min())

        # Scale the normalized CAM data to range [0, 255] and convert to uint8
        data_uint8 = (normalized_data * 255).astype(np.uint8)

         # Resize the CAM data to match the size of the original image
        cam_data_resized = cv2.resize(data_uint8, (original_image.shape[1], original_image.shape[0]))

        # Add the CAM data to the list
        cam_data_list.append(cam_data_resized)

    # Merge all the CAM data into a single CAM map
    merged_cam_map = np.zeros_like(cam_data_list[0], dtype=np.uint8)
    for data in cam_data_list:
        merged_cam_map += data

    # Apply colormap to the merged CAM map
    cam_image = cv2.applyColorMap(merged_cam_map, cv2.COLORMAP_JET)
    # Overlay the CAM visualization onto the original image with transparency
    overlay = cv2.addWeighted(original_image, 0.6, cam_image, 0.4, 0)
    
    return overlay

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_image_folder", default="C:/Users/snack/Desktop/SEAM/VOC2012_data", type=str, help="Path to the original image")
    parser.add_argument("--cam_data_folder", default="C:/Users/snack/Desktop/SEAM/voc12/out_cam", type=str, help="Path to the folder containing CAM data")
    parser.add_argument("--output_folder", default="C:/Users/snack/Desktop/SEAM/overlay", type=str, help="Path to the folder to save output images")
    parser.add_argument("--overlay_list", required=True, type=str, help="Path to the file containing a list of images")
    args = parser.parse_args()

    # Load the list of images to process
    with open(args.overlay_list, 'r') as file:
        lines = file.readlines()

    total_images = len(lines)
    last_printed_progress = 0

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        original_image_path = parts[0]
        filename = os.path.basename(original_image_path)
        original_image_data_path = os.path.splitext(filename)[0]
        cam_data_path = original_image_data_path + ".npy"

        # Load the original image
        original_image = cv2.imread(os.path.join(args.original_image_folder + original_image_path))

        # Load the .npy file containing the CAM data
        cam_data = np.load(os.path.join(args.cam_data_folder, cam_data_path), allow_pickle=True).item()

        # Apply CAM visualization to the original image
        overlay = apply_CAM_to_image(original_image, cam_data)

        # Determine the type of data based on the overlay_list filename
        data_type = os.path.splitext(os.path.basename(args.overlay_list))[0]

        # Create a subfolder for the type of data (e.g., "train" or "test") in the output folder
        output_subfolder = os.path.join(args.output_folder, data_type)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        # Save the image with the CAM overlay
        output_path = os.path.join(output_subfolder, os.path.splitext(os.path.basename(original_image_path))[0] + "_CAM_overlay.png")
        cv2.imwrite(output_path, overlay)

        progress_percentage = (idx + 1) / total_images * 100
        if progress_percentage - last_printed_progress >= 20:
            print("Progress: {:.2f}%".format(progress_percentage))
            # Update the last printed progress
            last_printed_progress = progress_percentage

    print("Job completed")