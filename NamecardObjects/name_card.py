from __future__ import annotations

import os.path
from shutil import rmtree
import time

from utils.visualization import *
from utils.image_processing import *
from TextDetection.textdetector import TextDetector
from SegmentMask.segment import SegmentMask
from recog_char.onnx_infer import RecogChar
from DetectBox.utils import CharDetect
from easyocr import Reader
from .find_font_thread import FontFinder
from typing import Dict, List


class NameCard:
    def __init__(self, text_detector: TextDetector, character_recognitor: RecogChar, mask_segmentor: SegmentMask, character_detector: CharDetect, ocr_reader: Reader, font_finder: FontFinder, config:Dict):
        """
               :param image: namecard image
               :param text: [line_1, line_2, ..., line_n]
                      line = [tspan_1, tspan_2, ..., tspan_m]
                      tspan = [text_string, font_family, left, top]
               """
        # self.idx_sample_to_text_ij = None
        self.image, self.text, self.text_replace_font = None, None, None
        self.debug_step = None
        self.sample_lines = None
        self.split_boxes = None
        self.split_samples_lines = None
        self.fonts = None
        self.mapping_sample_box = None
        self.text_detector = text_detector
        self.character_recognitor = character_recognitor
        self.mask_segmentor = mask_segmentor
        self.character_detector = character_detector
        self.ocr_reader = ocr_reader
        self.config = config
        self.font_finder = font_finder
        self.size = 5
        
    def update_new_card(self, image: np.ndarray, text:List[List[str|None, str, float, float]]) -> None:
        self.image = image
        self.text = text
        self.text_replace_font = deepcopy(self.text)
        self.debug_step = 0
        self.fonts = []
        self.split_boxes = []
        self.split_samples_lines = []
        # self.idx_sample_to_text_ij = {}
        if self.config['is_debug']:
            if os.path.isdir(self.config['debug_dir']):
                rmtree(self.config['debug_dir'])
            os.mkdir(self.config['debug_dir'])
            display_input_top_left_text(self.image, self.text, self.config['debug_dir'], str(self.debug_step))
        self.sample_lines = self.parse_text_info()
        self.find_similar_fonts()

    def parse_text_info(self):
        set_fonts = set()
        sample_lines = []
        # idx_sample = 0

        for i, line in enumerate(self.text):
            for j, t_span in enumerate(line):
                if t_span[0] is None:
                    continue
                if isinstance(t_span[0], str):
                    set_fonts.add(t_span[1])
                    sample_lines.append([t_span[0], t_span[2], t_span[3]])
                    # self.idx_sample_to_text_ij[idx_sample] = [i, j]
                    # idx_sample += 1
        return sample_lines

    def expand_boxes(self, boxes:List[List[int, int, int, int]]) -> None:
        for i in range(len(boxes)):
            boxes[i][2] += self.size

    @classmethod
    def sort_func(cls, e):
        return e[1]

    def split_detected_boxes_by_sample_lines(self, boxes: List[List[int, int, int, int]]) -> None:
        samples_of_box = {}
        for idx in range(len(boxes)):
            samples_of_box[idx] = []

        # liet ke tat ca cac sample co the nam trong box
        for sample_text, x, y in self.sample_lines:
            # x, y la to do top, left cua box tra ve tu json
            for idx, rec in enumerate(boxes):
                tl_x, tl_y, tr_x, tr_y = rec
                if tl_x <= x <= tr_x and tl_y <= y <= tr_y:
                    samples_of_box[idx].append([sample_text, x, y])

        # voi truong hop tu 2 sample tro len thuoc cung 1 box, can phai tach box
        new_boxes = []
        for idx in range(len(boxes)):
            if len(samples_of_box[idx]) >= 2:
                x1, y1, x2, y2 = boxes[idx]
                samples_of_box[idx].sort(key=self.sort_func)
                for sample_text, x, y in samples_of_box[idx][1:]:
                    x = int(x)
                    new_boxes.append([x1, y1, x, y2])
                    x1 = x
                new_boxes.append([x1, y1, x2, y2])
            else:
                new_boxes.append(boxes[idx])

        self.mapping_sample_box = {idx_sample:-1 for idx_sample in range(len(self.sample_lines))} 
        idx_sample_line = 0
        for idx_sample, [sample_text, x, y] in enumerate(self.sample_lines):
            x = x + self.size
            for idx, rec in enumerate(new_boxes):
                tl_x, tl_y, tr_x, tr_y = rec
                if tl_x <= x <= tr_x and tl_y <= y <= tr_y:
                    self.split_boxes.append([tl_x - self.size, tl_y, tr_x, tr_y])
                    self.split_samples_lines.append(sample_text)
                    self.mapping_sample_box[idx_sample] = idx_sample_line
                    idx_sample_line += 1
                    break

    def find_similar_fonts(self):
        t0 = time.time()
        boxes = self.detect_text_boxes()
        if self.config['is_debug']:
            self.debug_step += 1
            display_detected_boxes(self.image, boxes, self.config['debug_dir'], str(self.debug_step))
        t1 = time.time()
        # print('detect boxes = ', t1 - t0)
        
        self.expand_boxes(boxes)
        if self.config['is_debug']:
            self.debug_step += 1
            display_expanded_boxes(self.image, boxes, self.config['debug_dir'], str(self.debug_step))
        t2 = time.time()
        # print('expand time = ', t2 - t1)
        
        self.split_detected_boxes_by_sample_lines(boxes)
        if self.config['is_debug']:
            self.debug_step += 1
            display_split_boxes(self.image, boxes, self.split_boxes, self.config['debug_dir'], str(self.debug_step))
        t3 = time.time()
        # print('split time = ', t3 - t2)
        
        masks = self.mask_segmentor.segment_mask(self.image, self.split_boxes)

        if self.config['is_debug']:
            self.debug_step += 1
            display_masks(self.image, self.split_boxes, masks, self.config['debug_dir'], str(self.debug_step))
        if masks is None:
            print('NO MASK')
            return None
        t4 = time.time()
        # print('mask time = ', t4 - t3)
        
        char_boxes = self.character_detector.get_boxes_detect(masks)
        if self.config['is_debug']:
            self.debug_step += 1
            display_char_boxes(masks, char_boxes, self.config['debug_dir'], str(self.debug_step))
        t5 = time.time()
        # print('char boxes = ', t5 - t4)

        characters, image_characters = self.recognize_characters(masks, char_boxes)
        if self.config['is_debug']:
            self.debug_step += 1
            display_recognize_characters(characters, image_characters, self.config['debug_dir'], str(self.debug_step))
        t6 = time.time()
        # print('recognize time = ', t6 - t5)
        
        # Nhung chu cai nao xuat hien trong line cua sample moi dua vao tim font
        filter_characters, filter_image_characters = self.filter_character_for_find_font(characters, image_characters)
        if self.config['is_debug']:
            self.debug_step += 1
            display_filter_recognize_characters(filter_characters, filter_image_characters, self.config['debug_dir'], str(self.debug_step))
        t7 = time.time()
        # print('filter char = ', t7 - t6)
        
        self.find_font(filter_characters, filter_image_characters)
        print('find font = ', time.time() - t7)
        
        self.replace_fonts()
        if self.config['is_debug']:
            self.debug_step += 1
            display_result_find_font(self.text, self.text_replace_font, self.config['debug_dir'], 'result_font')

    def filter_character_for_find_font(self, characters, image_characters):
        filter_characters = []
        filter_image_characters = []

        for i, line_character in enumerate(characters):
            filter_line_character = []
            filter_image_character = []
            for j, character in enumerate(line_character):
                if character in self.split_samples_lines[i]:
                    filter_line_character.append(character)
                    filter_image_character.append(image_characters[i][j])
            filter_characters.append(filter_line_character)
            filter_image_characters.append(filter_image_character)

        return filter_characters, filter_image_characters

    def detect_text_boxes(self) -> List[List[int, int, int, int]]:
        h, w = self.image.shape[0: 2]
        detected_boxes = self.text_detector(self.image)
        boxes = []
        for detected_box in detected_boxes:
            p0, p1, p2, p3 = detected_box
            top_left_y = int(max(0, min(p0[0], p3[0])))
            top_left_x = int(max(0, min(p0[1], p1[1])))
            bot_right_y = int(min(w, max(p2[0], p1[0])))
            bot_right_x = int(min(h, max(p2[1], p3[1])))
            boxes.append([top_left_y, top_left_x, bot_right_y, bot_right_x])
        return boxes

    def recognize_characters(self, masks, char_boxes):
        characters = []
        image_characters = []
        for i in range(len(masks)):
            mask = masks[i].copy()
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            line_image_characters = []
            line_characters = []
            for box in char_boxes[i]:
                x1, y1, x2, y2 = box
                char = mask[y1:y2, x1:x2]
                if char.shape[0] <= 0 or char.shape[1] <= 0:
                    continue

                recoged_char, conf = self.character_recognitor.recog_character(char)
                if recoged_char in ['<', ' ', '>', '=']:
                    continue
                line_characters.append(recoged_char)
                char = padding2(char)
                line_image_characters.append(cv2.cvtColor(char, cv2.COLOR_BGR2GRAY))

            image_characters.append(line_image_characters)
            characters.append(line_characters)
        return characters, image_characters

    def find_font(self, filter_characters, filter_image_characters):
        len_filter_characters = len([filter_character for filter_character in filter_characters if filter_character])
        max_character_per_line = 90 // len_filter_characters
        for index in range(len(filter_characters)):
            
            line_character = filter_characters[index]
            line_character_image = filter_image_characters[index]
            if line_character:
                # character_of_line = int(min(max_character_per_line, len(line_character)))
                # start_idx = (len(line_character) - character_of_line) // 2
                # end_idx = start_idx + character_of_line
                # line_character_image = line_character_image[start_idx: end_idx]
                # line_character = line_character[start_idx: end_idx]

                final_font = self.font_finder.find_font_multi_thread(line_character_image, line_character)
                self.fonts.append(final_font)
            else:
                self.fonts.append(None)

    def replace_fonts(self):
        idx_sample = 0
        for i, line in enumerate(self.text_replace_font):
            for j, t_span in enumerate(line):
                if t_span[0] is None:
                    continue
                if isinstance(t_span[0], str):
                    idx = self.mapping_sample_box[idx_sample]
                    if idx != -1 and self.fonts[idx] is not None:
                        self.text_replace_font[i][j][1] = self.fonts[idx]
                    idx_sample += 1

