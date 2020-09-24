from __future__ import absolute_import, division, print_function

import time
import sys
import functools
import itertools
import logging
from multiprocessing import Pool

import numpy
import scipy
import scipy.linalg
import scipy.sparse

log = logging.getLogger(__name__)

def cov_exp_quad(delta_x_2,alpha,rho):
    exp_rho_delta_x_2 = numpy.exp(-0.5/(rho**2)*delta_x_2)
    return scipy.linalg.toeplitz((alpha**2)*exp_rho_delta_x_2), scipy.linalg.toeplitz((2*alpha)*exp_rho_delta_x_2), scipy.linalg.toeplitz((alpha**2/(rho**3))*(delta_x_2*exp_rho_delta_x_2))

def cov_exp_quad_no_grad(delta_x_2,alpha,rho):
    return scipy.linalg.toeplitz(alpha**2*numpy.exp(-0.5/(rho**2)*delta_x_2))

def helper(alpha,rho,sn,s2,delta_x_2,A,y):
    # shard specific

    # calculate K_x_x, \frac{\partial K_x_x}{\partial \alpha}, and \frac{\partial K_x_x}{\partial \rho} 
    K = cov_exp_quad(delta_x_2,alpha,rho)

    s2_matrix = numpy.diag(s2+sn**2)
  
    # calculate A K_x_x A.T + diag(\sigma^2)
    A_K_x_x_A_T = A.dot(A.dot(K[0]).T)+s2_matrix
    # calculate A \frac{\partial K_x_x}{\partial \alpha} A.T
    A_K_x_x_alpha_A_T = A.dot(A.dot(K[1]).T)
    # calculate A \frac{\partial K_x_x}{\partial \rho} A.T 
    A_K_x_x_rho_A_T = A.dot(A.dot(K[2]).T)
    
    # calculate the Cholesky decomposition: L L.T = A K_x_x A.T  + diag(\sigma^2)
    L = numpy.linalg.cholesky(A_K_x_x_A_T)
    
    # solve (A K_x_x A.T  + diag(\sigma^2)) \beta = m => m = (A K_x_x A.T  + diag(\sigma^2))^{-1} \beta using the Cholesky decomposition
    beta = scipy.linalg.solve_triangular(L.T,scipy.linalg.solve_triangular(L,numpy.expand_dims(y,axis=1),lower=True,check_finite=False),check_finite=False)
  
    # calculate the negative log marginal likelihood: 0.5 m.T beta + \sum \log L_i_i
    neg_log_likelihood = 0.5*numpy.expand_dims(y,axis=1).T.dot(beta) + numpy.log(numpy.diagonal(L)).sum() 
  
    # calculate the gradient of the negative log marginal likelihood
    A_K_x_x_sn_A_T = 2*sn*numpy.eye(A.shape[0])
    grad_neg_log_likelihood = numpy.array([-0.5*numpy.trace(beta.dot(beta.T).dot(A_K_x_x_alpha_A_T)-scipy.linalg.solve_triangular(L.T,scipy.linalg.solve_triangular(L,A_K_x_x_alpha_A_T,lower=True,check_finite=False),check_finite=False)),
                                           -0.5*numpy.trace(beta.dot(beta.T).dot(A_K_x_x_rho_A_T)-scipy.linalg.solve_triangular(L.T,scipy.linalg.solve_triangular(L,A_K_x_x_rho_A_T,lower=True,check_finite=False),check_finite=False)),
                                           -0.5*numpy.trace(beta.dot(beta.T).dot(A_K_x_x_sn_A_T)-scipy.linalg.solve_triangular(L.T,scipy.linalg.solve_triangular(L,A_K_x_x_sn_A_T,lower=True,check_finite=False),check_finite=False))])

    #print(neg_log_likelihood,grad_neg_log_likelihood)

    return neg_log_likelihood[0][0],grad_neg_log_likelihood

class GP_Deconvolution():
    def __init__(self,maximum_distance=50):
        # maximum_distance
        self.maximum_distance = maximum_distance

    def fit(self,A,y_guides,x_bases,x_guides,s2,multiprocessing=False):
        # A convolution matrix
        # y_guides observations
        # x_bases index set of latent variables
        # x_guides index set of observed variables
        # s2 variance of observations

        if not isinstance(s2,list) or not isinstance(y_guides,list) or not isinstance(x_bases,list) or not isinstance(x_guides,list) or not isinstance(A,list):
          log.error("s2, y_guides, x_bases, and x_guides, and A should be lists!")
          sys.exit(1)

        if len(s2) != len(y_guides) != len(x_bases) != len(x_guides) != len(A):
          log.error("s2, y_guides, x_bases x_guides, and A should have the same number of elements!")
          sys.exit(1)

        if any([any(numpy.diff(foo) != 1)  for foo in x_bases]):
          log.error("elements of x_bases should be complete grids!")
          sys.exit(1)

        t = time.time()

        # divide the problem into a number of smaller problems
        delta_x_2_list,A_list,y_guides_list,s2_list,_ = self._extract_independent_problems(A,x_bases,x_guides,y_guides,s2,self.maximum_distance)

        res = scipy.optimize.minimize(
                lambda theta: self._likelihood_shards(delta_x_2_list,
                    cov_exp_quad,A_list,y_guides_list,theta[0],theta[1],theta[2],s2_list,multiprocessing),
                  [1,7.5,1],method='L-BFGS-B',jac=True,bounds=[(0.001,numpy.inf),(5,10),(0.001,numpy.inf)])

        log.info("optimization took %f seconds"%(time.time()-t))

        return res.x

    def pred(self,A,y_guides,x_bases,x_guides,alpha,rho,sn,s2,full_pred=False,full_A=None):
        # A convolution matrix
        # y_guides observations
        # x_bases index set of latent variables
        # x_guides index set of observed variables
        # alpha
        # rho
        # sn
        # s2 variance of observations

        if not isinstance(s2,list) or not isinstance(y_guides,list) or not isinstance(x_bases,list) or not isinstance(x_guides,list) or not isinstance(A,list):
          log.error("s2, y_guides, x_bases, and x_guides, and A should be lists!")
          sys.exit(1)

        if len(s2) != len(y_guides) != len(x_bases) != len(x_guides) != len(A):
          log.error("s2, y_guides, x_bases x_guides, and A should have the same number of elements!")
          sys.exit(1)

        if any([any(numpy.diff(foo) != 1)  for foo in x_bases]):
            log.error("elements of x_bases should be complete grids!")
            sys.exit(1)

        if not full_pred and full_A is not None:
            log.warning('full_A has no effect if full_pred=False!')

        # divide the problem into a number of smaller problems
        delta_x_2_list,A_list,y_guides_list,s2_list,x_truncated = self._extract_independent_problems(A,x_bases,x_guides,y_guides,s2,self.maximum_distance)

        if not full_pred:
          mean_f,var_f = self._pred_shards(delta_x_2_list,x_bases,cov_exp_quad_no_grad,A_list,y_guides_list,alpha,rho,sn,s2_list)
        else:
            if full_A is not None:
                A_list = [full_A[:,numpy.concatenate(x_truncated)]]
            else:
                print(scipy.sparse.block_diag(A,format='csr').shape)
                print(numpy.concatenate(x_truncated).shape)
                A_list = [scipy.sparse.block_diag(A,format='csr')[:,numpy.concatenate(x_truncated)]]
            mean_f,var_f = self._pred_shards_full(delta_x_2_list,x_bases,cov_exp_quad_no_grad,A_list,y_guides_list,alpha,rho,sn,s2_list)

        return mean_f,var_f,x_truncated

    def _pred_shards_full(self,delta_x_2_list,x,k,A_list,y_list,alpha,rho,sn,s2_list):
        # delta_x_2 shard-specific train index set
        # x train index set
        # k covariance function
        # A_list shard-specific convolution matrices
        # y train data
        # alpha
        # rho
        # s2

        # number of observations
        N = sum([y.shape[0] for y in y_list])

        s2I = scipy.sparse.block_diag([scipy.sparse.diags(s2+sn**2,format='csr') for s2 in s2_list],format='csr')

        A = scipy.sparse.block_diag(A_list)

        y = numpy.concatenate(y_list)

        # calculate K_x_x over shards
        t = time.time()
        K = scipy.sparse.block_diag([k(delta_x_2,alpha,rho) for delta_x_2 in delta_x_2_list],format='csr')
        log.info('calculating the covariance matrix took: %f'%(time.time()-t))
    
        # calculate A K_x_x A.T + diag(\sigma^2) over shards
        t = time.time()
        A_K_x_x_A_T = scipy.sparse.csc_matrix(A.dot(A.dot(K).T)+s2I)
        log.info('multiplying the covariance with A and A.T took: %f'%(time.time()-t))

        #log.info('%f%% of the elements of A K A.T are nonzeros'%(numpy.count_nonzero(A_K_x_x_A_T)/numpy.prod(A_K_x_x_A_T.shape)*100))

        t = time.time()
        invA = scipy.sparse.linalg.spilu(A_K_x_x_A_T)
        beta = invA.solve(numpy.expand_dims(y,axis=1))
        log.info('solving K beta = y took: %f'%(time.time()-t))
    
        # K_xs_x A.T (A K_x_X A.T + diag(s2))^(-1) A y
        t = time.time()
        A_K = A.dot(K)
        mean_f_s = A_K.T.dot(beta)
        log.info('calculating mean(f) took: %f'%(time.time()-t))
    
        # K_xs_xs - K_xs_x A.T (A K_x_x A.T + diag(s2))^(-1) A K_x_xs
        t = time.time()
        var_f_s = K.diagonal()-numpy.sum(A_K.T.multiply(invA.solve(A_K.toarray()).T),axis=1).squeeze()
        log.info('calculating var(f) took: %f'%(time.time()-t))

        return mean_f_s.squeeze(),numpy.array(var_f_s).squeeze()

    def _pred_shards(self,delta_x_2_list,x,k,A_list,y_list,alpha,rho,sn,s2_list):
        # delta_x_2 shard-specific train index set
        # x train index set
        # k covariance function
        # A_list shard-specific convolution matrices
        # y train data
        # alpha
        # rho
        # s2

        # number of observations
        N = sum([y.shape[0] for y in y_list])

        s2I_list = [numpy.diag(s2+sn**2) for s2 in s2_list]

        # calculate K_x_x over shards
        t = time.time()
        K_list = [k(delta_x_2,alpha,rho) for delta_x_2 in delta_x_2_list]
        log.info('calculating the covariance matrix took: %f'%(time.time()-t))
    
        # calculate A K_x_x A.T + diag(\sigma^2) over shards
        t = time.time()
        A_K_x_x_A_T_list = [A.dot(A.dot(K).T)+s2I for A,K,s2I in zip(A_list,K_list,s2I_list)]
        log.info('multiplying the covariance with A and A.T took: %f'%(time.time()-t))
        
        # calculate the Cholesky decompositions over shards: L L.T = A K_x_x A.T  + diag(\sigma^2)
        t = time.time()
        L_list = [numpy.linalg.cholesky(A_K_x_x_A_T) for A_K_x_x_A_T in A_K_x_x_A_T_list]
        log.info('solving Kx=m took: %f'%(time.time()-t))

        # solve (A K_x_x A.T  + diag(\sigma^2)) \beta = m => m = (A K_x_x A.T  + diag(\sigma^2))^{-1} \beta using the Cholesky decomposition over shards
        t = time.time()
        try:
            beta_list = [scipy.linalg.solve_triangular(L.T,scipy.linalg.solve_triangular(L,numpy.expand_dims(y,axis=1),lower=True,check_finite=False),check_finite=False) for L,y in zip(L_list,y_list)]
        except ValueError:
            beta_list = [scipy.linalg.solve(L.T,scipy.linalg.solve(L,numpy.expand_dims(y,axis=1),lower=True,check_finite=False),check_finite=False) for L,y in zip(L_list,y_list)]
        log.info('solving K beta = y took: %f'%(time.time()-t))

        # K_xs_x A.T (A K_x_X A.T + diag(s2))^(-1) A y
        t = time.time()
        mean_f_s = numpy.concatenate([A.dot(K_x_x).T.dot(beta) for A,K_x_x,beta in zip(A_list,K_list,beta_list)])
        log.info('calculating mean(f) took: %f'%(time.time()-t))

        # K_xs_xs - K_xs_x A.T (A K_x_x A.T + diag(s2))^(-1) A K_x_xs
        t = time.time()
        try:
            var_f_s = numpy.concatenate([numpy.diagonal(K_x_x)-numpy.diagonal(A.dot(K_x_x).T.dot(scipy.linalg.solve_triangular(L.T,scipy.linalg.solve_triangular(L,A.dot(K_x_x),lower=True,check_finite=False),check_finite=False))) for L,A,K_x_x in zip(L_list,A_list,K_list)])
        except ValueError:
            var_f_s = numpy.concatenate([numpy.diagonal(K_x_x)-numpy.diagonal(A.dot(K_x_x).T.dot(scipy.linalg.solve(L.T,scipy.linalg.solve(L,A.dot(K_x_x),lower=True,check_finite=False),check_finite=False))) for L,A,K_x_x in zip(L_list,A_list,K_list)])
        log.info('calculating var(f) took: %f'%(time.time()-t))

        #return mean_f_s,cov_f_s
        return mean_f_s.flatten(),var_f_s.flatten()

    def _likelihood_shards(self,delta_x_2_list,k,A_list,y_list,alpha,rho,sn,s2_list,multiprocessing):
        # delta_x_2_list squared euclidean distances between the train index set values over shards
        # k covariance function
        # A_list convolution matrices over shards
        # y_list observed data
        # alpha
        # rho
        # sn
        # s2

        # number of observations
        N = sum([y.shape[0] for y in y_list])

        if multiprocessing:
          with Pool(None) as p:
            tmp = p.starmap(helper,zip([alpha]*len(y_list),[rho]*len(y_list),[sn]*len(y_list),s2_list,delta_x_2_list,A_list,y_list))
        else:
            tmp = itertools.starmap(helper,zip([alpha]*len(y_list),[rho]*len(y_list),[sn]*len(y_list),s2_list,delta_x_2_list,A_list,y_list))

        #alpha_rho, beta_rho = 10000.0,0.001
        #neg_log_likelihood,grad_neg_log_likelihood = functools.reduce(lambda acc,term: [acc[0]+term[0],acc[1]+term[1]],tmp,[0.5*N*numpy.log(2*numpy.pi) - (-alpha_rho*numpy.log(beta_rho)+(alpha_rho-1.0)*numpy.log(alpha)-scipy.special.gammaln(alpha_rho)-alpha/beta_rho) - (-alpha_rho*numpy.log(beta_rho)+(alpha_rho-1.0)*numpy.log(rho)-scipy.special.gammaln(alpha_rho)-rho/beta_rho),numpy.array([-( (alpha_rho-1.0)/alpha - 1.0/beta_rho ),-( (alpha_rho-1.0)/rho - 1.0/beta_rho )])])
        neg_log_likelihood,grad_neg_log_likelihood = functools.reduce(lambda acc,term: [acc[0]+term[0],acc[1]+term[1]],tmp,[0.5*N*numpy.log(2*numpy.pi),numpy.array([0,0,0])])

        print('current: alpha=%f, rho=%f, sn=%f'%(alpha,rho,sn))
        print('current: nllh=%f, grad nllh=%s'%(neg_log_likelihood,grad_neg_log_likelihood))
        return neg_log_likelihood, grad_neg_log_likelihood

    def _get_shard_boundaries(self,A,x,x_subset,y,distance):

        if not numpy.array_equal(x,numpy.sort(x)) or not numpy.array_equal(x_subset,numpy.sort(x_subset)):
            log.error("x and x_subset should be sorted")
            sys.exit(1)
   
        # skip some of the first/last bases if the first/last guide is not close
        start = max(numpy.where(x_subset[0] == x)[0][0]-distance,0)
        end = min(numpy.where(x_subset[-1] == x)[0][0]+distance,len(x)-1)
        start_y = 0

        indices_x = []
        indices_y = []
        
        # find the gaps (>= distance) in x_subset
        gaps = numpy.where(numpy.diff(x_subset) > distance)[0]
        for gap_idx,gap in enumerate(gaps):
            # gap is < 2*distance 
            if (numpy.where(x == x_subset[gap+1])[0][0]-numpy.where(x == x_subset[gap])[0][0]) < 2*distance:
                # start/end indices of the shard 
                # break at the middle of the gap
                indices_x.append([start,(numpy.where(x == x_subset[gap])[0][0]+numpy.where(x == x_subset[gap+1])[0][0])//2])
                # start index of the next shard
                start = (numpy.where(x == x_subset[gap])[0][0]+numpy.where(x == x_subset[gap+1])[0][0])//2
            # gap is >= 2*distance
            else:
                # start/end indices of the shard 
                # gap + distance
                indices_x.append([start,numpy.where(x == (x_subset[gap]+distance))[0][0]])
                # start index of the next shard
                # gap+1 + distance
                start = numpy.where(x == (x_subset[gap+1]-distance))[0][0]

            indices_y.append([start_y,gap])
            start_y = gap

        # add the ends    
        # skip some of the last bases if the last guide is not close
        indices_x.append([start,end])

        indices_y.append([start_y,len(y)])

        return indices_x, indices_y

    def _extract_independent_problems(self,A,x,x_subset,y,s2,distance):
        # A is the convolution matrix
        # x is the vector containing the base coordinates
        # x_subset is the vector containing the guide coordinates
        # y is the vector containing the observations
        # s2 is the vector containing the variances of the observations
        # distance is the maximum distance between adjacent guides for them to dependent

        delta_x_2_list = []
        A_list = []
        y_list = []
        s2_list = []
        x_truncated_list = []

        for idx in range(0,len(A)):
            indices_x, indices_y = self._get_shard_boundaries(A[idx],x[idx],x_subset[idx],y[idx],distance)
    
            # divide x, A, y, and s2 based on the gaps
            foo = []
            for idx_x in range(len(indices_x)):
                foo += list(range(indices_x[idx_x][0],indices_x[idx_x][1]))

                delta_x_2_list.append((x[idx][indices_x[idx_x][0]:indices_x[idx_x][1]]-x[idx][indices_x[idx_x][0]])**2)
                A_list.append(scipy.sparse.csr_matrix(A[idx][indices_y[idx_x][0]:indices_y[idx_x][1],indices_x[idx_x][0]:indices_x[idx_x][1]]))
                y_list.append(y[idx][indices_y[idx_x][0]:indices_y[idx_x][1]])
                s2_list.append(s2[idx][indices_y[idx_x][0]:indices_y[idx_x][1]])

            x_truncated = numpy.zeros(len(x[idx]))
            for idx in range(len(indices_x)):
                x_truncated[indices_x[idx][0]:indices_x[idx][1]] = 1
            x_truncated = x_truncated > 0
            x_truncated_list.append(x_truncated)

        # number of shards
        log.info("we have %d shards"%(len(delta_x_2_list)))

        # number of bases per shard
        log.info("shards have %s bases"%(', '.join(map(lambda x: str(len(x)),delta_x_2_list))))

        # the maximum number of bases per shard
        log.info("the longest shard has %d bases"%(max(map(len,delta_x_2_list))))

        # number of guides per shard
        log.info("shards have %s guides"%(', '.join(map(lambda x: str(x.shape[0]),A_list))))
            
        return delta_x_2_list, A_list, y_list, s2_list, x_truncated_list

