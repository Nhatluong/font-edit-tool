import requests
import cv2
import numpy as np
import base64
from imutils import paths

url = 'http://127.0.0.1:8019/get_card_region'

img = cv2.imread('test.jpg')
_, img_encoded = cv2.imencode('.jpg', img)
_file = {'img': img_encoded}
r = requests.post(url, files=_file, timeout=100)
if r.ok:
    json_data = r.json()
    card_img = json_data['card_img']
    decoded_data = base64.b64decode(card_img)
    np_data = np.frombuffer(decoded_data,np.uint8)
    img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    cv2.imwrite('out.jpg', img)