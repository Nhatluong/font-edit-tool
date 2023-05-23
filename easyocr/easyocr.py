from .recognition import get_recognizer, get_text
from .utils import group_text_box, get_image_list, \
                   diff, reformat_input
from .config import *
import os

class Reader(object):

    def __init__(self, lang_list, gpu=True, model_storage_directory=None,
                 user_network_directory=None, detect_network="craft", 
                 recog_network='standard', download_enabled=True, 
                 detector=True, recognizer=True, verbose=True, 
                 quantize=True, cudnn_benchmark=False):

        self.device = 'cpu'

        self.recognition_models = recognition_models

        # check and download detection model
        self.quantize=quantize, 
        self.cudnn_benchmark=cudnn_benchmark
        
        # recognition model
        separator_list = {}

        self.setModelLanguage('korean', lang_list, ['ko','en'], '["ko","en"]')
        model = recognition_models['gen2']['korean_g2']
        recog_network = 'generation2'
        self.character = model['characters']

        model_path = 'easyocr/weights/korean_g2.pth'
        # check recognition model file

        dict_list = {}
        for lang in lang_list:
            dict_list[lang] = os.path.join(BASE_PATH, 'dict', lang + ".txt")

        network_params = {
            'input_channel': 1,
            'output_channel': 256,
            'hidden_size': 256
            }

        self.recognizer, self.converter = get_recognizer(recog_network, network_params,\
                                                        self.character, separator_list,\
                                                        dict_list, model_path, device = self.device, quantize=quantize)
    
    def setModelLanguage(self, language, lang_list, list_lang, list_lang_string):
        self.model_lang = language

    def getChar(self, fileName):
        char_file = os.path.join(BASE_PATH, 'character', fileName)
        with open(char_file, "r", encoding="utf-8-sig") as input_file:
            list = input_file.read().splitlines()
            char = ''.join(list)
        return char

    def setLanguageList(self, lang_list, model):
        self.lang_char = []
        for lang in lang_list:
            char_file = os.path.join(BASE_PATH, 'character', lang + "_char.txt")
            with open(char_file, "r", encoding = "utf-8-sig") as input_file:
                char_list =  input_file.read().splitlines()
            self.lang_char += char_list
        
        symbol = model['symbols']
        self.lang_char = set(self.lang_char).union(set(symbol))
        self.lang_char = ''.join(self.lang_char)
        

    def recognize(self, img_cv_grey, horizontal_list=None, free_list=None,\
                  decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                  workers = 0, allowlist = None, blocklist = None, detail = 1,\
                  rotation_info = None,paragraph = False,\
                  contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                  y_ths = 0.5, x_ths = 1.0, reformat=True, output_format='standard'):

        ignore_char = ''
        
        if (horizontal_list==None) and (free_list==None):
            y_max, x_max = img_cv_grey.shape
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []

        image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey, model_height = imgH)

        result = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,\
                        ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                        workers, self.device)
        if detail == 0:
            return [item[1] for item in result]
        elif output_format == 'dict':
            return [ {'boxes':item[0],'text':item[1],'confident':item[2]} for item in result]
        else:
            return result

    def readtext(self, image, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                 workers = 0, allowlist = None, blocklist = None, detail = 1,\
                 rotation_info = None, paragraph = False, min_size = 20,\
                 contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                 text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                 canvas_size = 2560, mag_ratio = 1.,\
                 slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                 width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, 
                 threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
                 output_format='standard', horizontal_list=[], free_list=[]):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        '''
        img, img_cv_grey = reformat_input(image)

        result = self.recognize(img_cv_grey, horizontal_list, free_list,\
                                decoder, beamWidth, batch_size,\
                                workers, allowlist, blocklist, detail, rotation_info,\
                                paragraph, contrast_ths, adjust_contrast,\
                                filter_ths, y_ths, x_ths, False, output_format)

        return result
