from utils import *


@ray.remote
def build(results, pos):
    """
    根据results和对应pos去进行重建
    """

@ray.remote
def pre_build():
    """
    重建前的预处理，都是开个进程搞，之后考虑线程
    """
    pass