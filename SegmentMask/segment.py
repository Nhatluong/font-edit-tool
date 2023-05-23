import numpy as np
from SegmentMask.utils import *
import onnxruntime as ort
import uuid
import os
import time

class SegmentMask:
    def __init__(self, model_path, device):
        self.w = model_path
        self.sess_opts = ort.SessionOptions()
        if device == 'cpu':
            self.providers = ['CPUExecutionProvider']
        else:
            self.providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
        self.ort_sess_u2 = ort.InferenceSession(
                    self.w, providers=self.providers, sess_options=self.sess_opts
                )
        self.std = (0.229, 0.224, 0.225)
        self.size = (320, 320)
        self.mean = (0.485, 0.456, 0.406)

    def preprocess(self, img):
        # img = Image.fromarray(img)
        # im = img.convert("RGB").resize(self.size, Image.LANCZOS)

        # im_ary = np.array(im)
        # im_ary = im_ary / np.max(im_ary)

        # tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
        # tmpImg[:, :, 0] = (im_ary[:, :, 0] - self.mean[0]) / self.std[0]
        # tmpImg[:, :, 1] = (im_ary[:, :, 1] - self.mean[1]) / self.std[1]
        # tmpImg[:, :, 2] = (im_ary[:, :, 2] - self.mean[2]) / self.std[2]

        # tmpImg = tmpImg.transpose((2, 0, 1))
        
        if len(img.shape) < 3:
            np.expand_dims(img, axis=2)
        h, w = img.shape[0: 2] 
        new_w = int(96 / h * w)
        if new_w <= 1248:
            new_im = np.zeros((96, 1248, img.shape[2]), dtype=np.uint8)
            img = cv2.resize(img, (new_w, 96), interpolation=cv2.INTER_LINEAR)
            new_im[:, 0: new_w, :] = img
            img = new_im
        else:
            img = cv2.resize(img, (1248, 96), interpolation=cv2.INTER_LINEAR)
        img = (img / 255.0 - 0.5) / 1.0
        tmpImg = img.astype(np.float32).transpose(2, 0, 1)

        return tmpImg
    
    def postprocess(self, pred, h, w):
        # ma = np.max(pred)
        # mi = np.min(pred)

        # pred = (pred - mi) / (ma - mi)
        # pred = np.squeeze(pred)

        # mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")

        # open_cv_image = np.array(mask) 
        # im_bw = cv2.threshold(open_cv_image, 100, 255, cv2.THRESH_BINARY)[1]
        # im_bw = cv2.resize(im_bw, (w, h))
        
        ma = np.amax(pred)
        mi = np.amin(pred)
        pred = (pred-mi)/(ma-mi)
        new_w = int(h / 96 * 1248)
        pred = np.squeeze(pred) * 255
        im_bw = cv2.resize(pred.astype(np.uint8), (new_w, h), interpolation=cv2.INTER_LINEAR)[:, 0: w]
        # im_bw = cv2.threshold(im_bw, 90, 255, cv2.THRESH_BINARY)[1]
        return im_bw
    
    def infer_onnx(self, lst_image, lst_wh):
        lst_np = []
        lst_mask = []
        all_output = []

        split = len(lst_image)//10+1
        cut_len = len(lst_image)//split+1
        count = 0
        x_start = 0
        while count<split:
            lst_np = []
            count+=1
            for img in lst_image[x_start:min(x_start+cut_len, len(lst_image))]:
                
                tmpImg = self.preprocess(img)

                lst_np.append(tmpImg)

            outname = [i.name for i in self.ort_sess_u2.get_outputs()]
            inname = [i.name for i in self.ort_sess_u2.get_inputs()]
            inp = {inname[0]:np.array(lst_np, dtype=np.float32)}
            output = self.ort_sess_u2.run(outname, inp)[0]
            x_start = x_start+cut_len
            for pred in output:
                all_output.append(pred)

        for i, pred in enumerate(all_output):
            h, w = lst_wh[i]
            im_bw = self.postprocess(pred, h, w)
            lst_mask.append(im_bw)

        return lst_mask

    def segment_mask(self, img, list_box):
        lst_image = []
        lst_wh = []
        lst_index = []

        lst_mask_concat = []
        for c, (x1,y1, x2, y2) in enumerate(list_box):
            lineText = img[y1:y2, x1:x2]
            if lineText.shape[0] == 0 or lineText.shape[1] == 0:
                lst_mask_concat.append(None)
            
            time = 0
            while True:
                time+= 1
                cut_len =  (x2-x1) // time
                if cut_len / (y2-y1) <= 10:
                    break
            
            n_time = 0
            while True:
                n_time += 1
                if n_time == time:
                    x11 = x2
                else:
                    x11 = x1 + cut_len
                
                image = img[y1:y2, x1:x11]
                x1 = x11

                lst_image.append(image)
                lst_wh.append((image.shape[:2]))
                lst_index.append(c)
                if n_time == time:
                    break

        if len(lst_image) == 0:
            return None

        lst_mask = self.infer_onnx(lst_image, lst_wh)

        for j, index in enumerate(lst_index):
            if j > 0:
                if lst_index[j] == lst_index[j-1]:
                    part1 = mask
                    part2 = lst_mask[j]
                    mask = cv2.hconcat([part1, part2])
                else:
                    lst_mask_concat.append(mask)
                    mask = lst_mask[j]
            else:
                mask = lst_mask[j]

        lst_mask_concat.append(mask)
        return lst_mask_concat
