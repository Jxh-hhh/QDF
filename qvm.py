from scheduler import Scheduler
from cutter import cutter
from utils import *
from qvm_help_function import *


class Qvm:
    """
    这玩意作为全局对象，还是运行电路的一次性对象，还没想明白

    暂时作为一次性对象，等能跑完整个流程再慢慢改造
    敏捷式开发
    """

    def __init__(self, method=None, system_config=None):
        """
        这里输入的method，就是之前说的，调度方案，什么CPUonly，GPUonly，mix

        system_config 按理说是自动检测的，但是没有QPU，为了方便调试，暂时加上
        """
        if method is None:
            self.method = 'auto'
        self.qc = None
        self.compute_graph = None
        self.cut_solution = None
        self.subcircuit_entries = None
        self.subcircuit_instances = None
        self.instance_init_meas_ids = None
        self.entry_init_meas_ids = None
        self.system_config = None

    def _gen_compute_graph(self) -> ComputeGraph:
        """
        根据cutsolution生成计算图，如果传入的电路还没被切分则计算图就简单了，只有一个运行节点；如果QPU width不达到要求则自动切分
        """
        return generate_compute_graph(self.cut_solution['counter'], self.cut_solution['subcircuits'],
                                      self.cut_solution['complete_path_map'])

    def cut(self, qc: QuantumCircuit,
            cut_params=None,
            cut_method="auto"):
        self.qc = qc
        self.cut_solution = cutter.cut(self.qc, cut_params, cut_method)
        return self.cut_solution

    def run(self, *args, **kwargs):
        """
        首先区分切没切，如果输入的是一个电路，则是没切，直接run的，此时就直接运行或者模拟运行
        
        如果输入的是电路列表，则切过了，此时判断是否调用了cut方法，如果调用了，则直接使用cut生成的cut_solution
        如果没调用，输入了cut_solution，也可以继续下边的流程；如果也没输入cut_solution，此时直接报错就完事了，因为一个Qvm对象只能同时跑一个大电路

        compute_graph就是在这个地方生成的，因为cut方法，不一定调用
        """
        # 先生成计算图
        self.compute_graph = self._gen_compute_graph()
        # 在这里开一个线程开全局调度器 ray的也行，反正就调用runner里的几个函数和bulider重建函数
        # 生成entry和instance
        self.subcircuit_entries, self.subcircuit_instances, self.instance_init_meas_ids, self.entry_init_meas_ids = generate_subcircuit_entries(self.compute_graph)
        scheduler = Scheduler(self.compute_graph)
        res = scheduler.run()
        return res
