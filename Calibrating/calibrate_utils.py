# %%
import numpy as np

import time
import scipy

import sys
sys.path.append('../')

from Pricing.rBergomi.rBergomi_utils import *


# %%
def calibrate_with_scipy(CostFunc, x_test_transform, upper_bound, lower_bound, show_res_num=3):
    """
    使用 SciPy 的优化器进行参数校准

    Args:
        CostFunc (function): 代价函数
        x_test_transform (numpy.ndarray): 转换后的测试数据
        show_res_num (int, optional): 显示结果的数量. Defaults to 3.
    Returns:
        Approx_scipy (numpy.ndarray): SciPy 优化结果
        Timing_scipy (numpy.ndarray): SciPy 优化时间
    """
    Approx_scipy, Timing_scipy = [], [] 
    solutions = np.zeros([3,4])
    times = np.zeros(3)
    init = np.zeros(4)

    for i in range(x_test_transform.shape[0]):
        disp=str(i+1)+f"/{x_test_transform.shape[0]}"
        print (disp,end="\r")
        #L-BFGS-B
        start= time.time()
        I=scipy.optimize.minimize(CostFunc,x0=init,args=i,method='L-BFGS-B',tol=1E-10,options={"maxiter":10000})
        end= time.time()
        solutions[0,:]=params_inv_scaler(I.x, upper_bound, lower_bound)
        times[0]=end-start
        #SLSQP
        start= time.time()
        I=scipy.optimize.minimize(CostFunc,x0=init,args=i,method='SLSQP',tol=1E-10,options={"maxiter":10000})
        end= time.time()
        solutions[1,:]=params_inv_scaler(I.x, upper_bound, lower_bound)
        times[1]=end-start
        #BFGS
        start= time.time()
        I=scipy.optimize.minimize(CostFunc,x0=init,args=i,method='BFGS',tol=1E-10,options={"maxiter":10000})
        end= time.time()
        solutions[2,:]=params_inv_scaler(I.x, upper_bound, lower_bound)
        times[2]=end-start
        
        Approx_scipy.append(np.copy(solutions))
        Timing_scipy.append(np.copy(times))

    Approx_scipy = np.array(Approx_scipy)

    print(f"SciPy 优化结果 (前 {show_res_num} 轮):\n {Approx_scipy[:show_res_num]}")
    print(f"SciPy 优化时间 (前 {show_res_num} 轮):\n {Timing_scipy[:show_res_num]}")

    return Approx_scipy, Timing_scipy


