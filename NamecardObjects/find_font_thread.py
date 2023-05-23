import os
import pickle5 as pickle
import cv2
import time
from sewar import uqi_cython
import numpy as np
from multiprocessing import Pool


class FontFinder:
    def __init__(self, character_font_dir: str, character_file: str, font_dir: str, threshold: float, similar_percent: float, number_thread: int, group_font_font: str):
        with open(character_file, 'r') as f:
            characters = [line.strip() for line in f]
        self.idx_of_character = {characters[i]: i for i in range(len(characters))}
        self.character_font_dir = character_font_dir
        self.fonts = [file_name for file_name in os.listdir(font_dir) if
                      file_name.endswith((".ttf", ".otf", ".TTF", ".OTF"))]
        self.threshold = threshold
        self.similar_percent = similar_percent
        self.number_thread = number_thread
        with open(group_font_font, 'rb') as handle:
            self.group_font_font = pickle.load(handle)
        print('LOAD FIND FONT COMPLETE')

    def find_font_multi_thread(self, images, characters):
        # if len(images) > 1:
        #     list_found_fonts = []
        #     list_uqi_values = []

        #     input_parameter = zip(
        #         images,
        #         characters
        #     )
        #     number_thread = int(min(self.number_thread, len(images)))
        #     t1 = time.time()
        #     with Pool(number_thread) as p:
        #         for append_idx, append_font in p.imap_unordered(
        #             self.find_font_single_tuple,
        #             input_parameter,
        #         ):
        #             list_found_fonts += append_font
        #             list_uqi_values += append_idx
        #     print('aaaaaaaa = ', time.time() - t1)
        #     # danh gia
        #     # 1. font xuat hien nhieu nhat
        #     if list_found_fonts:
        #         font_uqi_dict = {}
        #         for i in range(len(list_found_fonts)):
        #             font_uqi_dict[list_found_fonts[i]] = list_uqi_values[i]

        #         set_fonts = set(list_found_fonts)
        #         list_count = [list_found_fonts.count(font) for font in set_fonts]
        #         max_count = int(max(list_count))
        #         curr_max_uqi = 0
        #         final_font = None
        #         for i, font in enumerate(set_fonts):
        #             if list_count[i] == max_count:
        #                 if curr_max_uqi < list_uqi_values[i]:
        #                     final_font = font

        #         # print([(font, list_found_fonts.count(font)) for font in list_found_fonts])
        #         # return max(set(list_found_fonts), key=list_found_fonts.count)
        #         return final_font
        #     else:
        #         return None
        # else:
            return self.find_font(images, characters)

    def find_font_single_tuple(self, t):
        return self.find_font_single(*t)

    def find_font_single(self, image, character):
        character_idx = self.idx_of_character[character]
        character_dir = os.path.join(self.character_font_dir, str(character_idx))
        with open(os.path.join(character_dir, 'center.pkl'), 'rb') as f:
            centers = pickle.load(f)
        with open(os.path.join(character_dir, 'clusters.pkl'), 'rb') as f:
            final_clusters = pickle.load(f)

        image = self.standard_image(image)
        idxs_cluster = uqi_cython.uqi_multi(image, centers)
        idx_cluster = np.argmax(idxs_cluster)
        with open(os.path.join(character_dir, f'{idx_cluster}.pkl'), 'rb') as f:
            cluster = pickle.load(f)
        
        uqi_values = uqi_multi(image, cluster)
        idx_max = np.argmax(uqi_values)
        max_value = uqi_values[idx_max]
        final_cluster = final_clusters[idx_cluster]
        append_info = [[idx, self.fonts[final_cluster[idx]][: - 4]] for idx in range(len(uqi_values))
                       if self.threshold <= uqi_values[idx] and uqi_values[idx] >= self.similar_percent * max_value]
        append_idx = [info[0] for info in append_info]
        append_font = [info[1] for info in append_info]
        return append_idx, append_font

    def find_font(self, images, characters):
        # print('\n\n*************** characters = ', characters)
        list_found_fonts = []
        list_uqi_values = []
        list_f = []
        list_uqi = []
        list_img = []
        for i, image in enumerate(images):
            character = characters[i]
            character_idx = self.idx_of_character[character]

            character_dir = os.path.join(self.character_font_dir, str(character_idx))
            with open(os.path.join(character_dir, 'center.pkl'), 'rb') as f:
                centers = pickle.load(f)
            with open(os.path.join(character_dir, 'clusters.pkl'), 'rb') as f:
                final_clusters = pickle.load(f)

            image = self.standard_image(image)
            image = image.astype(np.float64)
            centers = centers.astype(np.float64)
            idxs_cluster = uqi_cython.uqi_multi(memoryview(image), memoryview(centers))
            idx_cluster = np.argmax(idxs_cluster)
            with open(os.path.join(character_dir, f'{idx_cluster}.pkl'), 'rb') as f:
                cluster = pickle.load(f)
            
            cluster = cluster.astype(np.float64)
            uqi_values = uqi_cython.uqi_multi(memoryview(image), memoryview(cluster))
            idx_max = np.argmax(uqi_values)
            max_value = uqi_values[idx_max]
            final_cluster = final_clusters[idx_cluster]
            append_info = [[uqi_values[idx], self.fonts[final_cluster[idx]][: - 4]] for idx in range(len(uqi_values))
                       if self.threshold <= uqi_values[idx] and uqi_values[idx] >= self.similar_percent * max_value]
            
            append_uqi = [info[0] for info in append_info]
            append_font = [self.group_font_font[info[1]] for info in append_info]
            list_found_fonts += append_font
            list_uqi_values += append_uqi

        if list_found_fonts:
            # print('list_font = ', list(set(list_found_fonts)))
            font_uqi_dict = {}
            for i in range(len(list_found_fonts)):
                font_uqi_dict[list_found_fonts[i]] = list_uqi_values[i]

            set_fonts = set(list_found_fonts)
            list_count = [list_found_fonts.count(font) for font in set_fonts]
            # if 'KoPub Dotum Light' in list_found_fonts:
            #     # print('ZZZZZZZZZZZZZZZZZSS = ', list_found_fonts.count('KoPub Dotum Light'), list_found_fonts.count('ON 월인석보R'))
            #     for i, font in enumerate(set_fonts):
            #         print('****:', font, list_count[i])
            max_count = int(max(list_count))
            curr_max_uqi = 0
            final_font = None
            for i, font in enumerate(set_fonts):
                if list_count[i] == max_count:
                    if curr_max_uqi < list_uqi_values[i]:
                        final_font = font
                        curr_max_uqi = list_uqi_values[i]

            return final_font
        else:
            return None

    @classmethod
    def standard_image(cls, image):
        ret, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

        y, x, w, h = cv2.boundingRect(thresh)

        cropped_image = image[x: x + h, y: y + w]

        standarded_image = cv2.resize(cropped_image, (48, 48), cv2.INTER_LINEAR)
        # _, standarded_image = cv2.threshold(standarded_image, 100, 255, cv2.THRESH_BINARY)

        return standarded_image
