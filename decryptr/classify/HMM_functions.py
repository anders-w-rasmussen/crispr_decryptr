import numpy as np
from decryptr.classify.LibFwdBwd import FwdAlg_cpp, BwdAlg_cpp


def exp_log_lik(log_soft_ev, axis=1):
    lognorm_c = np.max(log_soft_ev, axis=axis)
    if axis == 0:
        log_soft_ev = log_soft_ev - lognorm_c[np.newaxis, :]
    elif axis == 1:
        log_soft_ev = log_soft_ev - lognorm_c[:, np.newaxis]
    soft_ev = np.exp(log_soft_ev)
    return soft_ev, lognorm_c

def calcRespPair_fast(PiMat, SoftEv, margPrObs, fmsg, bmsg, K, T,
                      doCopy=0):
    ''' Calculate pair-wise responsibilities for all adjacent timesteps

    Uses a fast, vectorized algorithm.

    Returns
    ---------
    respPair : 3D array, size T x K x K
        respPair[t,j,k] = prob. of the joint event that
        * step t-1 assigned to state j
        * step t assigned to state k
        Formally = p( z[t-1,j] = 1, z[t,k] = 1 | x[1], x[2], ... x[T])
        respPair[0,:,:] is undefined, but kept so indexing consistent.
    '''
    if doCopy:
        bmsgSoftEv = SoftEv * bmsg
    else:
        bmsgSoftEv = SoftEv  # alias
        bmsgSoftEv *= bmsg  # in-place multiplication

    respPair = np.zeros((T, K, K))
    respPair[1:] = fmsg[:-1][:, :, np.newaxis] * \
                   bmsgSoftEv[1:][:, np.newaxis, :]
    respPair *= PiMat[np.newaxis, :, :]
    respPair /= margPrObs[:, np.newaxis, np.newaxis]
    return respPair

def fwd_bwd_alg(lpi_init, lpi_mat, log_soft_ev):
    PiInit = np.exp(lpi_init)
    PiMat = np.exp(lpi_mat)
    K = PiMat.shape[0]
    T = log_soft_ev.shape[0]
    log_soft_ev = np.asarray(log_soft_ev, dtype=np.float64)
    SoftEv, lognormC = exp_log_lik(log_soft_ev)

    fmsg, margPrObs = FwdAlg_cpp(PiInit, PiMat, SoftEv)
    if not np.all(np.isfinite(margPrObs)):
        raise ValueError('NaN values found. Numerical badness!')
    bmsg = BwdAlg_cpp(PiInit, PiMat, SoftEv, margPrObs)

    resp = fmsg * bmsg
    respPair = calcRespPair_fast(PiMat, SoftEv, margPrObs, fmsg, bmsg, K, T)
    logMargPrSeq = np.log(margPrObs).sum() + lognormC.sum()
    return np.log(fmsg), np.log(bmsg), logMargPrSeq, resp, respPair





