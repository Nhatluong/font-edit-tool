import os
import cv2
from copy import deepcopy
import json


def display_input_top_left_text(image, text, debug_dir, debug_step):
    debug_path = os.path.join(debug_dir, debug_step)
    print('STEP display_input_top_left_text in:', debug_path)
    os.mkdir(debug_path)
    positions = []
    position_image = deepcopy(image)
    for line in text:
        for tspan in line:
            positions.append((int(tspan[2]), int(tspan[3])))
    for position in positions:
        cv2.circle(position_image, position, 2, (0,0,255), -1)
    cv2.imwrite(os.path.join(debug_path, 'positions.png'), position_image)


def display_detected_boxes(image, boxes, debug_dir, debug_step):
    debug_path = os.path.join(debug_dir, debug_step)
    print('STEP display_detected_boxes in:', debug_path)
    os.mkdir(debug_path)
    boxes_image = deepcopy(image)
    for [top_left_y, top_left_x, bot_right_y, bot_right_x] in boxes:
        cv2.rectangle(boxes_image, (top_left_y, top_left_x), (bot_right_y, bot_right_x), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(debug_path, 'boxes.png'), boxes_image)


def display_expanded_boxes(image, boxes, debug_dir, debug_step):
    debug_path = os.path.join(debug_dir, debug_step)
    print('STEP display_expanded_boxes in:', debug_path)
    os.mkdir(debug_path)
    boxes_image = deepcopy(image)
    for [top_left_y, top_left_x, bot_right_y, bot_right_x] in boxes:
        cv2.rectangle(boxes_image, (top_left_y, top_left_x), (bot_right_y, bot_right_x), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(debug_path, 'boxes.png'), boxes_image)


def display_split_boxes(image, boxes, split_boxes, debug_dir, debug_step):
    debug_path = os.path.join(debug_dir, debug_step)
    print('STEP display_split_boxes in:', debug_path)
    os.mkdir(debug_path)
    boxes_image = deepcopy(image)
    split_boxes_image = deepcopy(image)
    for [top_left_y, top_left_x, bot_right_y, bot_right_x] in boxes:
        cv2.rectangle(boxes_image, (top_left_y, top_left_x), (bot_right_y, bot_right_x), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(debug_path, 'boxes.png'), boxes_image)

    for [top_left_y, top_left_x, bot_right_y, bot_right_x] in split_boxes:
        cv2.rectangle(split_boxes_image, (top_left_y, top_left_x), (bot_right_y, bot_right_x), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(debug_path, 'split_boxes.png'), split_boxes_image)


def display_masks(image, boxes, masks, debug_dir, debug_step):
    debug_path = os.path.join(debug_dir, debug_step)
    print('STEP display_masks in:', debug_path)
    os.mkdir(debug_path)
    for i in range(len(boxes)):
        top_left_y, top_left_x, bot_right_y, bot_right_x = boxes[i]
        cv2.imwrite(os.path.join(debug_path, f'{i}_image.png'), image[top_left_x: bot_right_x, top_left_y: bot_right_y, :])
        cv2.imwrite(os.path.join(debug_path, f'{i}_mask.png'), masks[i])


def display_char_boxes(masks, char_boxes, debug_dir, debug_step):
    debug_path = os.path.join(debug_dir, debug_step)
    print('STEP display_char_boxes in:', debug_path)
    os.mkdir(debug_path)
    for i, mask in enumerate(masks):
        mask_dir = os.path.join(debug_path, str(i))
        boxes = char_boxes[i]
        os.mkdir(mask_dir)
        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for j, box in enumerate(boxes):
            top_left_x, top_left_y, bot_right_x, bot_right_y = box
            cv2.imwrite(os.path.join(mask_dir, f'{j}.png'), mask[top_left_y: bot_right_y, top_left_x: bot_right_x])
            cv2.rectangle(color_mask, (top_left_x, top_left_y), (bot_right_x, bot_right_y), (0, 0, 255), 1)

        cv2.imwrite(os.path.join(mask_dir, 'mask.png'), color_mask)


def display_recognize_characters(characters, image_characters, debug_dir, debug_step):
    debug_path = os.path.join(debug_dir, debug_step)
    print('STEP display_recognize_characters in:', debug_path)
    os.mkdir(debug_path)
    for i in range(len(characters)):
        for j, character in enumerate(characters[i]):
            display_character = 'splash' if character == '/' else character
            cv2.imwrite(os.path.join(debug_path, f'{i}_{j}_{display_character}.png'), image_characters[i][j])


def display_filter_recognize_characters(filter_characters, filter_image_characters, debug_dir, debug_step):
    debug_path = os.path.join(debug_dir, debug_step)
    print('STEP display_recognize_characters in:', debug_path)
    os.mkdir(debug_path)
    for i in range(len(filter_characters)):
        for j, character in enumerate(filter_characters[i]):
            display_character = 'splash' if character == '/' else character
            cv2.imwrite(os.path.join(debug_path, f'{i}_{j}_{display_character}.png'), filter_image_characters[i][j])


def display_result_find_font(text, text_replace_font, debug_dir, debug_step):
    debug_path = os.path.join(debug_dir, debug_step)
    print('STEP display_result_find_font in:', debug_path)
    os.mkdir(debug_path)

    with open(os.path.join(debug_path, "text.json"), "w") as out_file:
        json.dump(text, out_file)
    with open(os.path.join(debug_path, "text_replace_font.json"), "w") as out_file:
        json.dump(text_replace_font, out_file)
