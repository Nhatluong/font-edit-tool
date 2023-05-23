import onnxruntime as ort
import cv2
import numpy as np
from DetectBox.utils_detect_lp import non_max_suppression, letterbox_lp, xyxy2xywh, scale_coords

class CharDetect:
    def __init__(self, model_path, device):
        self.w = model_path
        if device == 'cpu':
            self.providers = ['CPUExecutionProvider']
        else:
            self.providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
        self.yolo_model = ort.InferenceSession(self.w, providers=self.providers)

        self.outname_yolo = [i.name for i in self.yolo_model.get_outputs()]
        self.inname_yolo = [i.name for i in self.yolo_model.get_inputs()]

    def padding(self, cv_img):
        h,w,_ = cv_img.shape
        if h > w:
            type = 0
            pad = h-w
            cv_img = cv2.copyMakeBorder(cv_img, 0, 0, pad//2, h-w-pad//2, cv2.BORDER_CONSTANT, None, [0,0,0])
        else:
            type = 1
            pad = w-h
            cv_img = cv2.copyMakeBorder(cv_img, pad//2, w-h-pad//2, 0,0, cv2.BORDER_CONSTANT, None, [0,0,0])
        return cv_img, type, pad//2

    def preprocess(self, img):
        img = np.array(img, np.float32)
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img, classes=None):
        preds = non_max_suppression(preds,
                                        0.25,
                                        0.7,
                                        agnostic=False,
                                        max_det=300,
                                        classes=None)
        pred = preds[0]
        shape = orig_img.shape
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], shape).round()
        return pred

    def get_boxes_detect(self, lst_masks):
        total_boxes = []
        for im0s in lst_masks:
            new_box = []
            
            im0s = np.stack((im0s,)*3, axis=-1)
            im0s, type, pad = self.padding(im0s)
            _h, _w, _ = im0s.shape
            im, _ ,_ = letterbox_lp(im0s, 640, auto=True, stride=32)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

            im = self.preprocess(im)
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            inp = {self.inname_yolo[0]:im}
            preds = self.yolo_model.run(self.outname_yolo, inp)[0]

            pred = self.postprocess(preds, im, im0s, None)

            gn = np.array([im0s.shape[1], im0s.shape[0], im0s.shape[1], im0s.shape[0]])

            for *xyxy, _, _ in reversed(pred):
                xywh = (xyxy2xywh(np.array(xyxy).reshape(1, 4)) / gn).reshape(-1).tolist()

                [x, y, w, h] = xywh
                x1 = int((x-w/2)*_w)
                x2 = int((x+w/2)*_w)
                y1 = int((y-h/2)*_h)
                y2 = int((y+h/2)*_h)
                if type == 0:
                    x1 = x1 - pad - 1
                    x2 = x2 - pad + 2
                else:
                    y1 = y1 - pad -1
                    y2 = y2 - pad + 2

                x1 = int(max(0, x1))
                y1 = int(max(0, y1))
                x2 = int(min(x2, _w))
                y2 = int(min(y2, _h))
                new_box.append([x1, y1, x2, y2])
            new_box.sort(key=self.sortfunc)
            total_boxes.append(new_box)
        return total_boxes

    def sortfunc(self, x):
        return x[0]