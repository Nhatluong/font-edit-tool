from rembg.session_simple import SimpleSession
import onnxruntime as ort
import numpy as np
import imutils
import cv2
import base64
from PIL import Image
from func import *
from fastapi import Request, Form, File, UploadFile, FastAPI
import uvicorn

app = FastAPI()

# Load U2
sess_opts = ort.SessionOptions()
session_class = SimpleSession
ort_sess_u2 = session_class(
        'u2net',
        ort.InferenceSession(
            'models/u2net.onnx', providers=ort.get_available_providers(), sess_options=sess_opts
        ),
    )

def card_region_func(input_ori):
    h_ori, w_ori,_ = input_ori.shape
    input = imutils.resize(input_ori, width=700)

    img = cv2.cvtColor(
        input,
        code=cv2.COLOR_BGR2RGB,
    )
    img = Image.fromarray(img)
    output = ort_sess_u2.predict(img)[0]
    open_cv_image = np.array(output) 
    im_bw = cv2.threshold(open_cv_image, 100, 255, cv2.THRESH_BINARY)[1]
    points = extract_idcard(input, im_bw)

    points = points*w_ori/700
    points = np.array(points).astype(np.int32)
    # print(points)
    tl = points[0].tolist()
    tr = points[1].tolist()
    br = points[2].tolist()
    bl = points[3].tolist()
    
    tl = [tl[0], tl[1]]
    tr = [tr[0], tr[1]]
    br = [br[0], br[1]]
    bl = [bl[0], bl[1]]
    points = np.array([tl, tr, br, bl])
    warped = four_point_transform(input_ori, points)

    return warped

@app.post("/get_card_region")
async def get_card_region(img: UploadFile = File(...)):
    try:
        data_imgI = img.file.read()
        input_ori = cv2.imdecode(np.fromstring(data_imgI, np.uint8), cv2.IMREAD_UNCHANGED)
        
        card_img = card_region_func(input_ori)
        _, buffer = cv2.imencode('.jpg', card_img)
        jpg_as_text = base64.b64encode(buffer)

        return {'success': 1, 'card_img': jpg_as_text}
    except:
        return {'success': 0, 'card_img': ''}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8019)
    
if __name__ == '__main__':
    main()