import os, sys

root_folder = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(root_folder, "object_tracker"))
sys.path.append(os.path.join(root_folder, "externals", "YOLOv6"))
sys.path.append(os.path.join(root_folder, "externals", "deep-person-reid"))

yolo_data_yaml = os.path.join(root_folder, "externals", "YOLOv6", "data", "coco.yaml")