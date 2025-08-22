
import xml.etree.ElementTree as ET
import json

def yolo_to_pascal_voc(yolo_file, image_width, image_height, class_names, output_file):
    with open(yolo_file, 'r') as f:
        lines = f.readlines()

    annotation = ET.Element("annotation")
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_width)
    ET.SubElement(size, "height").text = str(image_height)
    ET.SubElement(size, "depth").text = "3"

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height

        xmin = int(x_center - width / 2)
        ymin = int(y_center - height / 2)
        xmax = int(x_center + width / 2)
        ymax = int(y_center + height / 2)

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_names[class_id]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree.write(output_file)

def pascal_voc_to_coco(pascal_voc_file, class_names, output_file):
    tree = ET.parse(pascal_voc_file)
    root = tree.getroot()

    image_id = 1
    annotations = []
    categories = [{"id": i+1, "name": name} for i, name in enumerate(class_names)]

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = class_names.index(class_name) + 1

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        width = xmax - xmin
        height = ymax - ymin

        annotations.append({
            "id": len(annotations) + 1,
            "image_id": image_id,
            "category_id": class_id,
            "bbox": [xmin, ymin, width, height],
            "area": width * height,
            "iscrowd": 0
        })

    coco_format = {
        "images": [{
            "id": image_id,
            "width": int(root.find('size/width').text),
            "height": int(root.find('size/height').text),
            "file_name": root.find('filename').text
        }],
        "annotations": annotations,
        "categories": categories
    }

    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

def coco_to_yolo(coco_file, image_width, image_height, output_file):
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    yolo_lines = []
    for annotation in coco_data['annotations']:
        class_id = annotation['category_id'] - 1
        xmin, ymin, width, height = annotation['bbox']

        x_center = (xmin + width / 2) / image_width
        y_center = (ymin + height / 2) / image_height
        norm_width = width / image_width
        norm_height = height / image_height

        yolo_lines.append(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}")

    with open(output_file, 'w') as f:
        f.write("\n".join(yolo_lines))
