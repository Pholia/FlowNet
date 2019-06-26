import os, sys
import cv2
import tensorflow as tf
import numpy as np

from ..net import Mode

from .flownet2 import FlowNet2
from ..flownet_c.flownet_c import FlowNetC
from ..flownet_cs.flownet_cs import FlowNetCS
from ..flownet_css.flownet_css import FlowNetCSS
from ..flownet_s.flownet_s import FlowNetS
from ..flownet_sd.flownet_sd import FlowNetSD

from ..flowlib import flow_to_image

FLAGS = None

class FlowNet:
    
    def __init__(self, memory_fraction, model = 'fn_2'):
        self.prev = tf.placeholder(tf.float32, shape=(1, 384, 512, 3))
        self.curr = tf.placeholder(tf.float32, shape=(1, 384, 512, 3))
        
        if model == 'fn_2':
            self.net = FlowNet2(mode=Mode.TEST)
            weight_file_postfix = 'FlowNet2/flownet-2.ckpt-0'
        
        elif model == 'fn_c':
            self.net = FlowNetC(mode=Mode.TEST)
            weight_file_postfix = 'FlowNetC/flownet-C.ckpt-0'
        
        elif model == 'fn_cs':
            self.net = FlowNetCS(mode=Mode.TEST)
            weight_file_postfix = 'FlowNetCS/flownet-CS.ckpt-0'
        
        elif model == 'fn_css':
            self.net = FlowNetCSS(mode=Mode.TEST)
            weight_file_postfix = 'FlowNetCSS/flownet-CSS.ckpt-0'
        
        elif model == 'fn_s':
            self.net = FlowNetS(mode=Mode.TEST)
            weight_file_postfix = 'FlowNetS/flownet-S.ckpt-0'
        
        elif model == 'fn_sd':
            self.net = FlowNetSD(mode=Mode.TEST)
            weight_file_postfix = 'FlowNetSD/flownet-SD.ckpt-0'
        
        else:
            raise Exception

        weight_file = './TF_Flownet2/checkpoints/' + weight_file_postfix
        self.sess_info = self.net.load_session(weight_file, self.prev, self.curr, memory_fraction)
    
    def compute_feature(self, prev, curr):
        sess, pred_flow = self.sess_info
        flow = sess.run(pred_flow, feed_dict = {self.prev : np.expand_dims(cv2.resize(prev, (512, 384)) / 255., 0), self.curr : np.expand_dims(cv2.resize(curr, (512, 384)) / 255., 0)})[0, :, :, :]
        
        return flow_to_image(flow)