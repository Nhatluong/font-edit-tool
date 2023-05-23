import numpy as np
from PIL import Image 
import cv2 
import torchvision.transforms as transforms
import onnxruntime as ort
import json

class RecogChar:
    def __init__(self, model_path, device):
        self.w = model_path
        if device == 'cpu':
            self.providers = ['CPUExecutionProvider']
        else:
            self.providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
        self.model = ort.InferenceSession(self.w, providers=self.providers)

        self.label_file = json.load(open('recog_char/label_data.json'))

        self.transform_test = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
            ])

    def padding(self, cv_img):
        h,w,_ = cv_img.shape
        if h > w:
            pad = h-w
            cv_img = cv2.copyMakeBorder(cv_img, 0, 0, pad//2, h-w-pad//2, cv2.BORDER_CONSTANT, None, [0,0,0])
        else:
            pad = w-h
            cv_img = cv2.copyMakeBorder(cv_img, pad//2, w-h-pad//2, 0,0, cv2.BORDER_CONSTANT, None, [0,0,0])
        return cv_img

    def recog_character(self, cv_img):
        cv_img = self.padding(cv_img)
        img = Image.fromarray(cv_img)
        img = self.transform_test(img)
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        outname = [i.name for i in self.model.get_outputs()]
        inname = [i.name for i in self.model.get_inputs()]
        inp = {inname[0]:img}
        outputs = self.model.run(outname, inp)[0]
        preds = np.argmax(outputs[0])
        final_out = self.label_file[str(preds.item())]

        return final_out, outputs[0][preds]