from DetectText.textdetector import TextDetector
from imutils import paths
import cv2
import imutils

text_detector = TextDetector()

def detect_text(img):
    raw_h, raw_w = img.shape[0: 2]
    dt_boxes = text_detector(img)    
    lst_rec = []
    lst_box_img = []
    for box in dt_boxes:
        x0, x1, x2, x3 = box
        tl_x = max(int(min(x0[0], x3[0])), 0)
        tl_y = max(int(min(x0[1], x1[1])), 0)
        tr_x = min(int(max(x2[0], x1[0])), raw_w)
        tr_y = min(int(max(x2[1], x3[1])), raw_h)
        # copy = cv2.rectangle(copy, (tl_x, tl_y), (tr_x, tr_y), (255, 0, 0), 2)
        lst_rec.append([tl_x, tl_y, tr_x, tr_y])
        lst_box_img.append(img[tl_y:tr_y, tl_x:tr_x])

    return lst_rec, lst_box_img

if __name__ == '__main__':
    for path in paths.list_images('test_data'):
        img = cv2.imread(path)

        lst_rec, lst_box_img = detect_text(img)

        for x1, y1, x2, y2 in lst_rec:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        
        cv2.imshow('', imutils.resize(img, width=500))
        cv2.waitKey(0)