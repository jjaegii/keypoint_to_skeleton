import json
import matplotlib.pyplot as plt

def draw_skeleton(keypoints, skeleton):
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Draw keypoints
    for x, y, visibility in zip(keypoints[0::3], keypoints[1::3], keypoints[2::3]):
        if visibility == 2:  # If visibility is 2, it means the keypoint is visible
            ax.plot(x, y, 'o', color='r')
    
    # Draw skeleton connections
    for connection in skeleton:
        point1_idx, point2_idx = connection
        x1, y1, v1 = keypoints[point1_idx * 3], keypoints[point1_idx * 3 + 1], keypoints[point1_idx * 3 + 2]
        x2, y2, v2 = keypoints[point2_idx * 3], keypoints[point2_idx * 3 + 1], keypoints[point2_idx * 3 + 2]
        if v1 == 2 and v2 == 2:  # If both keypoints are visible, draw the line
            ax.plot([x1, x2], [y1, y2], 'r')

    # Set axis limits
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)  # Invert the y-axis to match the image coordinate system

    # Show the plot
    plt.show()

# Load JSON data from the file
file_path = r"C:\Users\064\Downloads\032_D00_001_F\032_D00_001_F_00000696.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract keypoints and skeleton from the data
keypoints = data["annotations"][0]["keypoints"]
skeleton = data["categories"][0]["skeleton"]

# Call the function to draw the skeleton
draw_skeleton(keypoints, skeleton)
