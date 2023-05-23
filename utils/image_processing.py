import cv2
import numpy as np

def padding2(cv_img):
    cv_img = cv2.copyMakeBorder(cv_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, None, [0,0,0])
    return cv_img.astype(np.uint8)


def postprocessing(text):
    index = text.find('gmailcom')
    if index != -1:
        text = text[:index - 1] + '@gmail.com' + text[index + 9:]

    index = text.find('gmail com')
    if index != -1:
        text = text[:index - 1] + '@gmail.com' + text[index + 10:]

    index = text.find('hanmailnet')
    if index != -1:
        text = text[:index - 1] + '@hanmail.net' + text[index + 11:]

    index = text.find('hanmail net')
    if index != -1:
        text = text[:index - 1] + '@hanmail.net' + text[index + 12:]

    index = text.find('www')
    if index != -1 and len(text) >= index + 4:
        if text[index + 3] == ' ':
            text = text[:index] + 'www.' + text[index + 4:]
        elif text[index + 3] == '.':
            pass
        else:
            text = text[:index] + 'www.' + text[index + 3:]

    index = text.find('WWW')
    if index != -1 and len(text) >= index + 4:
        if text[index + 3] == ' ':
            text = text[:index] + 'WWW.' + text[index + 4:]
        elif text[index + 3] == '.':
            pass
        else:
            text = text[:index] + 'WWW.' + text[index + 3:]

    index = text.find('cokr')
    if index != -1:
        if text[index - 1] == '.':
            text = text[:index] + 'co.kr' + text[index + 4:]
        elif text[index - 1] == ' ':
            text = text[:index - 1] + '.co.kr' + text[index + 4:]
        else:
            text = text[:index] + '.co.kr' + text[index + 4:]

    index = text.find('COkr')
    if index != -1:
        if text[index - 1] == '.':
            text = text[:index] + 'co.kr' + text[index + 4:]
        elif text[index - 1] == ' ':
            text = text[:index - 1] + '.co.kr' + text[index + 4:]
        else:
            text = text[:index] + '.co.kr' + text[index + 4:]

    if text.find('.co.kr') != -1 and text.find('WWW') != -1:
        index = text.find('WWW')
        text = text[:index] + 'www' + text[index + 3:]

    return text