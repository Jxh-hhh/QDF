from utils import *


class CutterModel(ABC):
    """
    可以当成抽象类，被继承，也可以直接写找切分点的求解器(把ABC，抽象方法给删掉)
    :return:
    """
    def __init__(self):
        pass

    @abstractmethod
    def solve(self):
        """
        这个用来返回结果
        :return:
        """
        pass



def _find_cuts(qc: QuantumCircuit, cut_method="gurobi", *args, **kwargs):
    """
    找可切分点
    :return:
    """
    position = []
    return position

def _cut_parser(qc:QuantumCircuit, position, *args, **kwargs):
    """
    根据position,返回最终子电路信息以及连接信息等
    :return:
    """
    cut_solution = {}
    return cut_solution


def cut(qc: QuantumCircuit, cut_method="auto", cut_solution=None, *args, **kwargs):
    """
    用户最终调用的cut函数，输入的除了量子电路。如果输入了cut_solution，则不调用find_cuts;
    如果没有输入cut_solution，则走默认流程，此时还可以选择cut_method。
    :param cut_solution:
    :param qc:
    :param cut_method:
    :param args:
    :param kwargs:
    :return:
    """
    position = _find_cuts(qc)
    cut_solution = _cut_parser(qc, position)
    return cut_solution