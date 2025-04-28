import cv2
import numpy as np
import os
import glob

# Settings
image_folder = 'images'       # Folder with your images
output_folder = 'labels'      # Folder to save YOLO label files
os.makedirs(output_folder, exist_ok=True)

# Resize threshold
MAX_WIDTH = 800
MAX_HEIGHT = 800

drawing = False
dragging = False
ix, iy = -1, -1
dx, dy = 0, 0  # Difference for moving the box
box = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, img_copy, box, dragging, dx, dy

    # When user clicks to start drawing or dragging
    if event == cv2.EVENT_LBUTTONDOWN:
        if box is not None:
            # Check if click is inside the existing box to allow dragging
            x1, y1, x2, y2 = box
            if x1 < x < x2 and y1 < y < y2:
                dragging = True
                dx = x - x1
                dy = y - y1
            else:
                # Start drawing a new box
                drawing = True
                ix, iy = x, y
        else:
            # Start drawing the first box
            drawing = True
            ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = img_copy.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        elif dragging and box:
            # Move the selected bounding box
            x1, y1, x2, y2 = box
            x_offset = x - dx
            y_offset = y - dy
            box = (x_offset, y_offset, x_offset + (x2 - x1), y_offset + (y2 - y1))
            img = img_copy.copy()
            draw_box(img)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            # Finalize the box coordinates
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            box = (x1, y1, x2, y2)
            drawing = False
            img = img_copy.copy()
            draw_box(img)
        dragging = False

def draw_box(image):
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

def convert_to_yolo_format(box, img_width, img_height):
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def save_yolo_labels(filename, box, img_width, img_height):
    with open(filename, 'w') as f:
        if box:
            yolo_line = convert_to_yolo_format(box, img_width, img_height)
            f.write(yolo_line + '\n')

# Loop through images
image_paths = glob.glob(os.path.join(image_folder, '*.jpg')) + glob.glob(os.path.join(image_folder, '*.png'))
for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        continue

    # Check if image needs to be resized
    h, w = img.shape[:2]
    if w > MAX_WIDTH or h > MAX_HEIGHT:
        scale_factor = min(MAX_WIDTH / w, MAX_HEIGHT / h)
        new_size = (int(w * scale_factor), int(h * scale_factor))
        img = cv2.resize(img, new_size)
        print(f"Image resized to: {new_size}")
    
    img_copy = img.copy()

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_rectangle)

    while True:
        img_display = cv2.resize(img, (0, 0), fx=1, fy=1)  # Display original size
        cv2.imshow('Image', img_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Save
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(output_folder, base_name + '.txt')
            save_yolo_labels(label_path, box, w, h)  # Use original dimensions for labels
            print(f"Saved YOLO labels: {label_path}")
            break
        elif key == ord('r'):  # Reset
            box = None
            img = img_copy.copy()
            print("Reset box.")
        elif key == 27:  # ESC to exit
            print("Exiting.")
            exit()

cv2.destroyAllWindows()
