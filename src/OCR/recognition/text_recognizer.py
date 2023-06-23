from argparse import Namespace
import numpy as np
import cv2
from OCR.recognition.text_recognition import TextRecognizer
import os
import sys
from PIL import Image
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import math
import time
import traceback
import paddle

import utility as utility
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read
def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s

class TextRecognizer_TMA(TextRecognizer):
    __instance__ = None

    @staticmethod
    def getInstance():
        """ Static access method """
        if TextRecognizer_TMA.__instance__ == None:
            TextRecognizer_TMA()
        return TextRecognizer_TMA.__instance__

    def __init__(self):
        if TextRecognizer_TMA.__instance__ != None:
            raise Exception('Paddle Text Recognizer is a singleton!')
        else:
            TextRecognizer_TMA.__instance__ = self

            self.rec_args = Namespace(
                image_dir = "src/img_test_inbody.jpeg",
                rec_algorithm="SVTR_LCNet",
                rec_model_dir="model/recog_model",
                rec_image_inverse=True,
                rec_image_shape="3, 48, 320",
                rec_batch_num=6,
                max_text_length=25,
                rec_char_dict_path="src/OCR/ppocr/utils/en_dict.txt",
                use_space_char=True,
                vis_font_path="./doc/fonts/simfang.ttf",
                drop_score=0.5,
                use_onnx = False,
                benchmark = False,
                use_gpu = True,
                use_xpu = False,
                use_npu = False,
                ir_optim = True,
                use_tensorrt = False,
                min_subgraph_size = 15,
                precision = "fp32",
                gpu_mem = 500,
                gpu_id = 0
            )

            TextRecognizer.__init__(self, self.rec_args)

    def recognize(self,img_list):
        args = vars(self.rec_args)
        rec_res, time_pr = TextRecognizer.__call__(self,img_list)
        return rec_res, time_pr
    
