import numpy as np
import cv2

# Load the .npy file containing the CAM data
data = np.load("C:/Users/snack/Desktop/SEAM/voc12/out_cam/2007_000032.npy", allow_pickle=True).item()

# Create an empty list to store CAM data for each class
cam_data_list = []

# Iterate over the dictionary items
for class_index, cam_data in data.items():
    # Normalize the CAM data to range [0, 1]
    normalized_cam_data = (cam_data - cam_data.min()) / (cam_data.max() - cam_data.min())

    # Scale the normalized CAM data to range [0, 255] and convert to uint8
    cam_data_uint8 = (normalized_cam_data * 255).astype(np.uint8)

    # Add the CAM data to the list
    cam_data_list.append(cam_data_uint8)

# Merge all the CAM data into a single CAM map
merged_cam_map = np.zeros_like(cam_data_list[0], dtype=np.uint8)
for cam_data in cam_data_list:
    merged_cam_map += cam_data

# Apply colormap to the merged CAM map
cam_image = cv2.applyColorMap(merged_cam_map, cv2.COLORMAP_JET)

# Save the visualized CAM data as an image
output_path = "output_cam_image.png"
cv2.imwrite(output_path, cam_image)

print("CAM visualization saved to:", output_path)
