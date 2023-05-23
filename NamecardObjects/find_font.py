import os
import pickle
import cv2
from sewar import uqi_multi
import numpy as np


class FontFinder:
    def __init__(self, character_font_dir: str, character_file: str, font_dir: str, threshold: float, similar_percent: float):
        with open(character_file, 'r') as f:
            characters = [line.strip() for line in f]
        self.idx_of_character = {characters[i]: i for i in range(len(characters))}
        self.character_font_dir = character_font_dir
        self.fonts = [file_name for file_name in os.listdir(font_dir) if
                      file_name.endswith((".ttf", ".otf", ".TTF", ".OTF"))]
        self.threshold = threshold
        self.similar_percent = similar_percent
        print('LOAD FIND FONT COMPLETE')

    def find_font(self, images, characters):
        list_found_fonts = []
        list_uqi_values = []
        for i, image in enumerate(images):
            character = characters[i]
            character_idx = self.idx_of_character[character]
            with open(os.path.join(self.character_font_dir, f'{str(character_idx)}.pkl'), 'rb') as handle:
                image_fonts = pickle.load(handle)
            image = self.standard_image(image)
            uqi_values = uqi_multi(image, image_fonts)
            idx_max = np.argmax(uqi_values)
            max_value = uqi_values[idx_max]
            append_info = [[idx, self.fonts[idx][: - 4]] for idx in range(len(uqi_values))
                           if self.threshold <= uqi_values[idx] and uqi_values[idx] >= self.similar_percent * max_value]
            append_idx = [info[0] for info in append_info]
            append_font = [info[1] for info in append_info]
            list_found_fonts += append_font
            list_uqi_values += append_idx
            # if uqi_values[idx_max] >= self.threshold:
            #     list_found_fonts.append(self.fonts[idx_max][: -4])
            #     list_uqi_values.append(uqi_values[idx_max])
        if list_found_fonts:
            font_uqi_dict = {}
            for i in range(len(list_found_fonts)):
                font_uqi_dict[list_found_fonts[i]] = list_uqi_values[i]

            set_fonts = set(list_found_fonts)
            list_count = [list_found_fonts.count(font) for font in set_fonts]
            max_count = int(max(list_count))
            curr_max_uqi = 0
            final_font = None
            for i, font in enumerate(set_fonts):
                if list_count[i] == max_count:
                    if curr_max_uqi < list_uqi_values[i]:
                        final_font = font

            # print([(font, list_found_fonts.count(font)) for font in list_found_fonts])
            # return max(set(list_found_fonts), key=list_found_fonts.count)
            return final_font
        else:
            return None

    @classmethod
    def standard_image(cls, image):
        ret, thresh = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)

        y, x, w, h = cv2.boundingRect(thresh)

        cropped_image = image[x: x + h, y: y + w]

        standarded_image = cv2.resize(cropped_image, (48, 48), cv2.INTER_LINEAR)

        return standarded_image