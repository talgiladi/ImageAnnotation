import os
import glob
import xml.etree.ElementTree as ET

# Define your classes - you'll need to adjust this based on your class names
# Each class name will be mapped to a numeric ID
class_mapping = {
    'l': 0,  # In your example, the class is 'l' with ID 0
    # Add more classes as needed:
    # 'car': 1,
    # 'person': 2,
    # etc.
}

def convert_coordinates(size, box):
    """
    Convert XML bounding box coordinates to YOLO format
    
    Args:
        size: tuple (width, height) of the image
        box: tuple (xmin, ymin, xmax, ymax) with absolute coordinates
        
    Returns:
        tuple (x_center, y_center, width, height) normalized to [0, 1]
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    # Calculate center coordinates, width, and height
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    
    # Normalize
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    return (x, y, w, h)

def convert_xml_to_yolo(xml_file):
    """
    Convert a single XML file to YOLO format
    
    Args:
        xml_file: path to the XML file
        
    Returns:
        List of YOLO format lines
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image size
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    # Prepare output
    yolo_lines = []
    
    # Process each object
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        
        # Skip classes not in our mapping
        if class_name not in class_mapping:
            print(f"Warning: Class '{class_name}' not found in mapping. Skipping.")
            continue
            
        class_id = class_mapping[class_name]
        
        # Get bounding box
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert coordinates
        x_center, y_center, width, height = convert_coordinates((img_width, img_height), (xmin, ymin, xmax, ymax))
        
        # Format the YOLO line: class_id x_center y_center width height
        yolo_line = f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)
    
    return yolo_lines

def process_all_files(input_dir, output_dir):
    """
    Process all XML files in the input directory and save YOLO files to the output directory
    
    Args:
        input_dir: directory containing XML files
        output_dir: directory to save YOLO format files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all XML files
    xml_files = glob.glob(os.path.join(input_dir, '*.xml'))
    
    for xml_file in xml_files:
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        
        # Convert to YOLO format
        yolo_lines = convert_xml_to_yolo(xml_file)
        print(f"{yolo_lines}")
        # Write to output file
        yolo_file = os.path.join(output_dir, base_name + '.txt')
        with open(yolo_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        print(f"Processed: {xml_file} -> {yolo_file}")

# Example usage
# Replace these with your actual directories
input_directory = "images"  # Directory containing XML files
output_directory = "labels"  # Directory to save YOLO format files

process_all_files(input_directory, output_directory)
print("Conversion completed!")