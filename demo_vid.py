"""Demo script to detect, track and redetect persons onto video"""

from copy import deepcopy
import _init_paths
import argparse

import cv2

from object_tracker.tracker import MultiTracker
from object_tracker.structs import RoI

from detector import Detector
from fve import FVE

def plot_box_and_label(image, lw, box, label='', ind=0, txt_color=(255, 255, 255)):
    def generate_colors(i, bgr=False):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = []
        for iter in hex:
            h = '#' + iter
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color

    color = generate_colors(ind, True)
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)

def plot_trackers(image, lw, fps, visible_objs, invisible_objs, txt_color=(255, 255, 255)):
    def putText(image, msg, h, lw, txt_color):
        mw, mh = cv2.getTextSize(msg, 0, fontScale=lw/2, thickness=1)[0]  # text width, height
        h += mh + 3
        cv2.putText(image, msg, (0, h), 0, lw / 3, txt_color, thickness=1, lineType=cv2.LINE_AA)
        return h

    h = putText(image, f"Visible {len(visible_objs)} persons", 0, lw, txt_color)
    objs = deepcopy(visible_objs)
    objs.update(invisible_objs)
    msg = f"Detected at all {len(objs)}:"
    h = putText(image, msg, h, lw, txt_color)
    for ind in sorted(objs.keys()): h = putText(image, f" #{ind} : {int(objs[ind]/fps)} sec", h, lw, txt_color)

def process_video(video_filepath, detector, fve):
    cap = cv2.VideoCapture(video_filepath)
    if not cap.isOpened():
        print(f"Cannot read video {video_filepath}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    mtracker = MultiTracker(trackerType='MOSSE', verbose=True)

    flag, img_src = cap.read()
    i = 0
    while i > 0:
        cap.read()
        i -= 1
    while flag:
        i += 1
        #  Process the frame
        dets = detector(img_src, 0.5)
        print(f"------------ {i} -----------------")
        #print(dets)
        det_rois = []
        for k, det in enumerate(dets):
            ltrb = det[0]
            chip = img_src[ltrb[1]:ltrb[3], ltrb[0]:ltrb[2]]
            fv = fve(chip)
            det_rois.append(RoI(ltrb, k, {'fv':fv}))

        rois, debub_info = mtracker(img_src, det_rois)
        visible_objs = {el[0]:el[1] for el in debub_info['active']}
        invisible_objs = {el[0]:el[1] for el in debub_info['inactive']}
        plot_trackers(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), fps, visible_objs, invisible_objs)
        #print([(r.getBox(), r.getFeature('fv')) for r in rois])

        for roi in rois:
            ind = roi.getInd()
            box = roi.getBox()
            plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), box, str(ind), int(ind))

        cv2.imshow('video', img_src)
        if cv2.waitKey(10) == 27:
            break  #  Exit on 'Escape'
        #  read next frame
        flag, img_src = cap.read()


def main(args):
    detector = Detector(weights=args.detector_model, device='cuda', yaml=args.detector_conf, img_size=640, half=False)
    fve = FVE(args.extractor_name, args.extractor_path)
    process_video(args.input_video, detector, fve)


def parse_args():
    parser = argparse.ArgumentParser("Detect-track-count persons")
    parser.add_argument('-dm', '--detector_model', type=str, required=True, help="YOLOv6 detector model path")
    parser.add_argument('-dc', '--detector_conf', type=str, default=_init_paths.yolo_data_yaml, help="YOLOv6 detector config")
    parser.add_argument('-en', '--extractor_name', type=str, required=True, help="FVE model name")
    parser.add_argument('-ep', '--extractor_path', type=str, required=True, help="FVE model path")
    parser.add_argument('-i', '--input_video', type=str, required=True, help="Input video file")
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())