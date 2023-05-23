import time
from NamecardObjects import NameCard, FontFinder
import cv2
import os
import json
from TextDetection.textdetector import TextDetector
from SegmentMask.segment import SegmentMask
from recog_char.onnx_infer import RecogChar
from DetectBox.utils import CharDetect
from utils.read_yml import read_config
import easyocr


if __name__ == '__main__':
    config = read_config('./config/config.yml')
    font_finder = FontFinder(config['character_font_dir'], config['character_file'], config['font_dir'], float(config['font_threshold']), float(config['similar_percent']), int(config['number_thread']), config['group_font_font'])
    text_detector = TextDetector(config['text_detection_model'], config['device'])
    character_recognitor = RecogChar(config['character_recognition_model'], config['device'])
    mask_segmentor = SegmentMask(config['mask_segmentation_model'], config['device'])
    character_detector = CharDetect(config['character_detection_model'], config['device'])
    ocr_reader = easyocr.Reader(['en', 'ko'],  detector=False)
    print('LOAD COMPLETE')

    # print(config['debug_mode'])
    
    input_dir = 'json_texts_web'

    for file_name in os.listdir(input_dir):
        if file_name.endswith('a3c0ee1e-ee47-11ed-8a1b-7085c2fe7a8c.jpg'): # file_name.endswith('9f9e6710-b956-11ed-b0f0-7085c2fe7a8c.jpg')
            file_name = file_name.replace('.jpg', '')
            # file_name = '0fc8f23a-b961-11ed-b0f0-7085c2fe7a8c'
            print(file_name)
            
            image = cv2.imread(os.path.join(input_dir, file_name + '.jpg'))
            with open(os.path.join(input_dir, file_name + '.json')) as f:
                text = json.load(f)
            name_card = NameCard(text_detector, character_recognitor, mask_segmentor, character_detector, ocr_reader, font_finder, config)

            start_time = time.time()
            name_card.update_new_card(image, text)
            print(f'*******TIME = {time.time() - start_time}\n')
