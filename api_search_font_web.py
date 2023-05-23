import os.path

import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
from flask import request

from NamecardObjects import NameCard, FontFinder
import cv2
import json
from TextDetection.textdetector import TextDetector
from SegmentMask.segment import SegmentMask
from recog_char.onnx_infer import RecogChar
from DetectBox.utils import CharDetect
from utils.read_yml import read_config
import easyocr
from time import time

import uuid

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_ORIGINS'] = '*'
app.config['CORS_ALLOW_HEADERS'] = '*'
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/get_font/', methods=['POST', 'GET'])
def get_font():
    try:
        byte_image = request.files.getlist('image')[0].read()
        image_raw = cv2.imdecode(np.frombuffer(byte_image, np.uint8), -1)
        text = json.loads(request.form.get('text'))

        if config['is_write']:
            name = str(uuid.uuid1())
            cv2.imwrite(os.path.join(config['json_texts_web_dir'], '{}.jpg'.format(name)), image_raw)
            with open(os.path.join(config['json_texts_web_dir'], '{}.json'.format(name)), "w") as out_file:
                json.dump(text, out_file)
        start_time = time()
        name_card.update_new_card(image_raw, text)
        print('all time = ', time() - start_time)
        return jsonify({'text': name_card.text_replace_font}), 200
        # return jsonify({'text': text}), 200
    except Exception as e:
        print(e)
        return jsonify({'text': []}), 0


if __name__ == '__main__':
    config = read_config('./config/config.yml')
    if not os.path.isdir(config['json_texts_web_dir']):
        os.makedirs(config['json_texts_web_dir'])
    font_finder = FontFinder(config['character_font_dir'], config['character_file'], config['font_dir'], float(config['font_threshold']), float(config['similar_percent']), int(config['number_thread']), config['group_font_font'])
    text_detector = TextDetector(config['text_detection_model'], config['device'])
    character_recognitor = RecogChar(config['character_recognition_model'], config['device'])
    mask_segmentor = SegmentMask(config['mask_segmentation_model'], config['device'])
    character_detector = CharDetect(config['character_detection_model'], config['device'])
    ocr_reader = easyocr.Reader(['en', 'ko'],  detector=False)
    name_card = NameCard(text_detector, character_recognitor, mask_segmentor, character_detector, ocr_reader,
                         font_finder, config)

    app.run(host='0.0.0.0', port=4010, debug=False)
