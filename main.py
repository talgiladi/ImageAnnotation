import cv2
import os
import glob

# Settings
image_folder = 'images'       # Folder with your images
output_folder = 'labels'      # Folder to save YOLO label files
os.makedirs(output_folder, exist_ok=True)

drawing = False
ix, iy = -1, -1
boxes = []
MAX_WIDTH = 800
MAX_HEIGHT = 800
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, img_copy, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = img_copy.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        boxes.append((x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"Box: ({x1}, {y1}) to ({x2}, {y2})")

def convert_to_yolo_format(box, img_width, img_height):
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def save_yolo_labels(filename, boxes, img_width, img_height):
    with open(filename, 'w') as f:
        for box in boxes:
            yolo_line = convert_to_yolo_format(box, img_width, img_height)
            f.write(yolo_line + '\n')

# Loop through images
image_paths = glob.glob(os.path.join(image_folder, '*.jpg')) + glob.glob(os.path.join(image_folder, '*.png'))
for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    if w > MAX_WIDTH or h > MAX_HEIGHT:
        scale_factor = min(MAX_WIDTH / w, MAX_HEIGHT / h)
        new_size = (int(w * scale_factor), int(h * scale_factor))
        img = cv2.resize(img, new_size)
        print(f"Image resized to: {new_size}")
    img_copy = img.copy()
    boxes = []

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_rectangle)

    while True:
        cv2.imshow('Image', img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Save
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(output_folder, base_name + '.txt')
            save_yolo_labels(label_path, boxes, w, h)
            print(f"Saved YOLO labels: {label_path}")
            break
        elif key == ord('r'):  # Reset
            boxes = []
            img = img_copy.copy()
            print("Reset boxes.")
        elif key == 27:  # ESC to exit
            print("Exiting.")
            exit()

cv2.destroyAllWindows()
