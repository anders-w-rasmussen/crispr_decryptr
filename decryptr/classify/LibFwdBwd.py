import os
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

curpath = os.path.dirname(os.path.realpath(__file__))

if os.path.exists(curpath + "/libfwdbwdcpp.so"):
    pass
    #print("C++ forward-backward algorithm is compiled: " + str(curpath) + "/libfwdbwdcpp.so")
else:
    print("C++ forward-backward algorithm not found at: " + str(curpath) + "/libfwdbwdcpp.so")
    print("Installing and compiling the Forward-Backward Algorithm")

    os.system('wget --timestamping -nv --directory-prefix=' + curpath + ' https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip')
    os.system('unzip -q ' + curpath + '/eigen-3.3.7.zip' + ' -d ' + curpath)
    os.system('g++ ' + curpath + '/libfwdbwd.cpp -o ' + curpath + '/libfwdbwdcpp.so --shared -fPIC -DNDEBUG -O3 -I' + curpath + '/eigen-3.3.7')
    if os.path.exists(curpath + "/libfwdbwdcpp.so"):
        print("Success: C++ forward-backward algorithm is compiled: " + str(curpath) + "/libfwdbwdcpp.so")

def FwdAlg_cpp(initPi, transPi, SoftEv, order='C'):
    """
    Forward algorithm for a single HMM sequence. Implemented in C++/Eigen.
    :param initPi:
    :param transPi:
    :param SoftEv:
    :param order:
    :return:
    """
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")
    T, K = SoftEv.shape
    # Prep inputs
    initPi = np.asarray(initPi, order=order)
    transPi = np.asarray(transPi, order=order)
    SoftEv = np.asarray(SoftEv, order=order)

    # Allocate outputs
    fwdMsg = np.zeros((T, K), order=order)
    margPrObs = np.zeros(T, order=order)

    # Execute C++ code (fills in outputs in-place)
    lib.FwdAlg(initPi, transPi, SoftEv, fwdMsg, margPrObs, K, T)
    return fwdMsg, margPrObs


def BwdAlg_cpp(initPi, transPi, SoftEv, margPrObs, order='C'):
    ''' Backward algorithm for a single HMM sequence. Implemented in C++/Eigen.
    '''
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")

    # Prep inputs
    T, K = SoftEv.shape
    initPi = np.asarray(initPi, order=order)
    transPi = np.asarray(transPi, order=order)
    SoftEv = np.asarray(SoftEv, order=order)
    margPrObs = np.asarray(margPrObs, order=order)

    # Allocate outputs
    bMsg = np.zeros((T, K), order=order)

    # Execute C++ code for backward pass (fills in bMsg in-place)
    lib.BwdAlg(initPi, transPi, SoftEv, margPrObs, bMsg, K, T)
    return bMsg


""" 
This block of code loads the shared library and defines wrapper functions
that can take numpy array objects.
"""
import os
libpath = os.path.dirname(os.path.realpath(__file__))
#libpath = '/Users/arasmussen/PycharmProjects/decryptr_v3/classify'



libfilename = 'libfwdbwdcpp.so'


# libfilename = 'originallib.so'
hasEigenLibReady = True

try:
    lib = ctypes.cdll.LoadLibrary(os.path.join(libpath, libfilename))
    lib.FwdAlg.restype = None
    lib.FwdAlg.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ctypes.c_int, ctypes.c_int]

    lib.BwdAlg.restype = None
    lib.BwdAlg.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ctypes.c_int, ctypes.c_int]

    lib.SummaryAlg.restype = None
    lib.SummaryAlg.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ctypes.c_int, ctypes.c_int, ctypes.c_int]

except OSError:
    # No compiled C++ library exists
    hasEigenLibReady = False
