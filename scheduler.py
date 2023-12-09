from utils import *
import runner
import builder 


class Scheduler:
    """
    本来要写成一个全局线程的，但是要先把一个电路跑通，所以先写成串行的，等实现了，再实现分布式也不晚
    """
    def __init__(self, compute_graph: ComputeGraph):
        self.compute_graph = compute_graph


    def run(self):
        """
        根据计算图，去调度分配资源运行，这里就需要调度算法了
        最终返回个结果就是了
        将某个任务分配给某个节点，有不同的区分
        跑子电路的，runner.run_circuit(qc, type, shots)
        跑重建或者重建预处理的，builder.prebuild   builder.build ，这里边参数你怎么方便怎么来，但是要告诉清楚哪两个子电路在重建，哪个子电路被预处理的，对应的哪个比特是被切割的。
        都是ray.remote函数
        """
        
        pass