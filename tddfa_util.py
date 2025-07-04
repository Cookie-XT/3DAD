# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import argparse
import numpy as np
import torch
import pickle



##转化为C顺序存储
def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr



##将字符串转化为bool值
def str2bool(v):
    ##先转化为小写字母
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        ##抛出异常
        raise argparse.ArgumentTypeError('Boolean value expected')


##加载模型的权重参数
def load_model(model, checkpoint_fp):
    ##将checkpoint_fp的的['state_dict']字段名称加载出来
    ##storage 表示模型参数的存储，loc表示存储的位置，这里返回storage表示返回原始模型的设备，如果返回loc则可以指定设备
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    ##将模型的参数字典移到model_dict上
    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed

    ##将checkpoint的参数更新到模型中
    for k in checkpoint.keys():

        ##多gpu训练前缀有'module'，移去
        kc = k.replace('module.', '')
        ###如果检查点参数在模型中，替换
        if kc in model_dict.keys():
            model_dict[kc] = checkpoint[k]
        ##如果检查点事全连接层的权重，将'_param'删除，
        if kc in ['fc_param.bias', 'fc_param.weight']:
            continue

        
    ##参数加载到当前模型中
    model.load_state_dict(model_dict)
    return model


class ToTensorGjz(object):
    ##传递参数时直接调用
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    ##返回字符串的表示形式
    def __repr__(self):
        return self.__class__.__name__ + '()'

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

##归一化rensor
class NormalizeGjz(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor


##pts3d为三维坐标(3,N)：（x,y,z）
##将三维坐标缩放在roi上
def similar_transform(pts3d, roi_box, size):
    pts3d[0, :] -= 1  # for Python compatibility
    pts3d[2, :] -= 1

    ##垂直方向进行翻转
    pts3d[1, :] = size - pts3d[1, :]

    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= np.min(pts3d[2, :])
    return np.array(pts3d, dtype=np.float32)


def _parse_param(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    n = param.shape[0]
    if n == 62:
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise Exception(f'Undefined templated param parsing rule')

    R_ = param[:trans_dim].reshape(3, -1)
    ##从R里面取出前三列，组成旋转矩阵
    R = R_[:, :3]
    ##取出最后一列成为偏移量
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp
