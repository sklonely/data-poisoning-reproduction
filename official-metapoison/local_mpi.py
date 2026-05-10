class _DummyComm:
    def Get_size(self):
        return 1

    def Get_rank(self):
        return 0

    def gather(self, value, root=0):
        return [value]

    def bcast(self, value, root=0):
        return value

    def Bcast(self, value, root=0):
        return value

    def Allreduce(self, sendbuf, recvbuf, op=None):
        recvbuf[...] = sendbuf


class _DummyMPI:
    COMM_WORLD = _DummyComm()
    SUM = 'sum'


try:
    from mpi4py import MPI as _MPI
    MPI = _MPI
except Exception:
    MPI = _DummyMPI()
