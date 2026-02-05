import xml.etree.ElementTree as ET
import cv2
import os

XML_PATH = "M3OT/2/rgb/val/2-08\gt/08-2.xml"
IMG_DIR = "M3OT/2/rgb/val/2-08/img1"
LABEL_DIR = "M3OT/2/rgb/val/2-08/labels"

os.makedirs(LABEL_DIR, exist_ok=True)

tree = ET.parse(XML_PATH)
root = tree.getroot()

for track in root.iter("track"):
    label = track.attrib["label"]

    # Only keep vehicles
    if label != "vehicle":
        continue

    class_id = 0

    for box in track.iter("box"):
        frame = int(box.attrib["frame"])

        xtl = float(box.attrib["xtl"])
        ytl = float(box.attrib["ytl"])
        xbr = float(box.attrib["xbr"])
        ybr = float(box.attrib["ybr"])
        img_num = frame+1
        img_name = f"{img_num:06d}.PNG"
        img_path = os.path.join(IMG_DIR, img_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        bw = xbr - xtl
        bh = ybr - ytl
        xc = xtl + bw / 2
        yc = ytl + bh / 2

        xc /= w
        yc /= h
        bw /= w
        bh /= h

        label_path = os.path.join(
            LABEL_DIR, img_name.replace(".PNG", ".txt")
        )

        with open(label_path, "a") as f:
            f.write(f"{class_id} {xc} {yc} {bw} {bh}\n")
