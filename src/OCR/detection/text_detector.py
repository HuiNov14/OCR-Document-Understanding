from argparse import Namespace
from OCR.detection.text_detection import TextDetector

def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s

class PaddleTextDetector(TextDetector):
    __instance__ = None

    @staticmethod
    def getInstance():
        """ Static access method """
        if PaddleTextDetector.__instance__ == None:
            PaddleTextDetector()
        return PaddleTextDetector.__instance__
        


    def __init__(self):
        if PaddleTextDetector.__instance__ != None:
            raise Exception('Paddle Text Recognizer is a singleton!')
        else:
            PaddleTextDetector.__instance__ = self

            rec_args = Namespace(
                det_algorithm="DB",
                use_gpu=True,
                # use_npu=False,
                # use_xpu=False,
                # gpu_mem=500,
                # det_limit_side_len=960,
                # det_limit_type="max",
                # det_db_thresh=0.3,
                # det_db_box_thresh=0.5,
                # det_db_unclip_ratio=1.5,
                # max_batch_size=10,
                # use_dilation=False,
                # det_db_score_mode="fast",
                det_model_dir="model/detection_model",
                # det_box_type="quad",
                # use_onnx=False,
                # use_tensorrt=False,
                benchmark=False
                # enable_mkldnn=False
            )
        
            TextDetector.__init__(self, rec_args)
    

    def detect(self, img):
        """
        Detect the text in the image
        Input: a image
        Output: A list of bounding boxes
        """

        infer_img = img.copy()
        dt_boxes, _ = TextDetector.__call__(self, infer_img)
        if len(dt_boxes):
            return dt_boxes
        else:
            return None


