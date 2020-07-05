import numpy
import pyximport
pyximport.install(setup_args={"include_dirs":numpy.get_include()},
                  reload_support=True)
from utils_c import propagate

def main():
    j = 0
    N, n, m, p, s = 501, 2, 2, 2, 2
    Ns = 2 * n * s + p
    sizes = {'N': N, 'n': n, 'm': m, 'p': p, 's': s, 'Ns': Ns}

    DynMat = numpy.ones((N,2*n,2*n,s))
    InvDynMat = numpy.ones((N,2*n,2*n,s))
    InitCondMat = numpy.eye(Ns,Ns+1)
    phip = numpy.ones((N, n, p, s))
    phipTr = numpy.ones((N, p, n, s))
    phiuTr = numpy.ones((N, m, n, s))
    phiuFu = numpy.ones((N, n, s))
    fu = numpy.ones((N, m, s))
    fx = numpy.ones((N, n, s))
    err = numpy.zeros((N, n, s))

    for k in range(100):
        propagate(j, sizes, DynMat, err, fu, fx, InitCondMat, InvDynMat,
                  phip, phipTr, phiuFu, phiuTr, grad=True, isCnull=False)
    return None

if __name__ == '__main__':
    import cProfile
    cProfile.run('main()', sort='time')
