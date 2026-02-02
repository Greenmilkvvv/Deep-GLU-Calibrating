# %%
import numpy as np

import time
import scipy

import torch

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



# %%
def calibrate_with_torch_lbfgs(model, y_test_transform, device='cpu'):
    """ 
    使用 LBFGS 优化器

    Args:
        model (torch.nn.Module): 模型
        y_test_transform (torch.Tensor): 隐含波动率曲面
        device (str, optional): 设备. Defaults to 'cpu'.
        
    Returns:
        Approx_lbfgs (numpy.ndarray): LBFGS 优化结果
    """

    model = model.to(device)
    model.eval()  # 评估模式
    
    Approx = []
    Timing = []
    
    # 转换为tensor
    y_test_tensor = torch.from_numpy(y_test_transform).float().to(device)
    
    for i in range(len(y_test_tensor)):
        print(f"{i+1}/{len(y_test_tensor)}", end="\r")
        
        # 初始化待优化参数（需要梯度）
        params = torch.zeros(4, requires_grad=True, dtype=torch.float64, device=device)
        
        # 获取当前样本的真实值
        target = y_test_tensor[i].unsqueeze(0)  # 形状: (1, ...)
        
        # 定义 LBFGS 优化器
        optimizer = torch.optim.LBFGS(
            [params],
            lr=1.0,
            max_iter=10000,
            tolerance_grad=1e-10,
            tolerance_change=1e-10,
            history_size=100,
            line_search_fn='strong_wolfe'
        )
        
        # 定义 closure 函数
        def closure():
            optimizer.zero_grad()
            
            # 神经网络预测（参数需要先转换格式）
            params_reshaped = params.unsqueeze(0)  # 形状: (1, 4)
            predicted = model(params_reshaped)
            
            # 计算损失（MSE）
            loss = torch.sum((predicted - target) ** 2)
            
            # 反向传播
            loss.backward()
            
            return loss
        
        # 优化
        start = time.time()
        
        # LBFGS优化循环
        max_epochs = 100
        for epoch in range(max_epochs):
            loss = optimizer.step(closure)
            
            # 提前停止条件
            if loss.item() < 1e-10:
                break
        
        end = time.time()
        
        # 记录结果
        solutions = params.detach().cpu().numpy()
        times = end - start
        
        Approx.append(solutions)
        Timing.append(times)
    
    return np.array(Approx), np.array(Timing)