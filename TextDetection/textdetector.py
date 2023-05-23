import copy
import torch
import onnxruntime as ort
from .db_postprocess import DBPostProcess
from .ocr_reader import DetResizeForTest
from .operators import *
from .config import *
import time

def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops

class TextDetector(object):
    def __init__(self, model_file_path, device):
        self.det_algorithm = det_algorithm
        self.use_onnx = use_onnx
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': det_limit_side_len,
                'limit_type': det_limit_type,
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {'name': 'DBPostProcess', "thresh": det_db_thresh, "box_thresh": det_db_box_thresh,
                              "max_candidates": 1000, "unclip_ratio": det_db_unclip_ratio, "use_dilation": use_dilation,
                              "score_mode": det_db_score_mode}

        config = copy.deepcopy(postprocess_params)
        module_name = config.pop('name')
        module_class = eval(module_name)(**config)

        self.postprocess_op = module_class

        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
        sess = ort.InferenceSession(model_file_path, providers=providers)

        self.predictor, self.input_tensor, self.output_tensors, self.config = sess,  sess.get_inputs()[0], None, None

        if self.use_onnx:
            pass
            # img_h, img_w = self.input_tensor.shape[2:]
            # if img_h is not None and img_w is not None and img_h > 0 and img_w > 0:
            #     pre_process_list[0] = {
            #         'DetResizeForTest': {
            #             'image_shape': [img_h, img_w]
            #         }
            #     }
        self.preprocess_op = create_operators(pre_process_list)


    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}

        data = transform(data, self.preprocess_op)
        img, shape_list = data

        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        input_dict = {}
        input_dict[self.input_tensor.name] = img
        outputs = self.predictor.run(self.output_tensors, input_dict)
        preds = {}
        preds['maps'] = outputs[0]

        #self.predictor.try_shrink_memory()
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
 
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        
        return dt_boxes
