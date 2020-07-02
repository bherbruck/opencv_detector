import cv2
import imutils
import numpy as np
import opencv_wrapper as cvw
from simple_tracker import Tracker


def run():
    with cvw.load_video('media/video3.mkv') as video:
        tracker = Tracker(max_distance=40, timeout=10)
        out = cv2.VideoWriter('output/video.mp4',
                              cv2.VideoWriter_fourcc(*"MP4V"), 60.0,
                              (int(video.get(3)), int(video.get(4))))
        for img in cvw.read_frames(video):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, 0)
            kernel = np.ones((3, 3), np.uint8)
            erosion = cv2.erode(binary, kernel, iterations=3)
            contours = cv2.findContours(erosion, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_NONE)
            counts = tracker.update([get_centroid(contour) for contour in
                                     imutils.grab_contours(contours)])
            for id, (cx, cy, d, f) in tracker.points.items():
                center_text(img, id, (cx, cy), scale=0.5, thickness=2)
            
            out.write(img)
            cv2.imshow('', img)
            if cv2.waitKey(1000//60) == 27:
                break


def center_text(img, text, centroid, font=cv2.FONT_HERSHEY_SIMPLEX,
                color=(123, 123, 123), scale=1, thickness=1):
    text = str(text)
    cx, cy = centroid
    width, height = cv2.getTextSize(text, font, scale, thickness)[0]
    tx = cx - (width // 2)
    ty = cy + (height // 2)
    cv2.putText(img, text, (tx, ty), font, scale, color, thickness)


def get_centroid(contour):
    try:
        m = cv2.moments(contour)
        cx = int(m['m10'] // m['m00'])
        cy = int(m['m01'] // m['m00'])
    except ZeroDivisionError:
        cx = 0
        cy = 0
    return (cx, cy)


if __name__ == '__main__':
    run()
