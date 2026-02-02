# %%
import gzip
import numpy as np

import matplotlib.pyplot as plt
import torch.utils
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文宋体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import matplotlib.ticker as mtick


# %%
def data_read(data_path):
    # 加载数据
    f = gzip.GzipFile( 
        data_path, 
        "r"
    )
    data = np.load(f) 
    print(f"网格数据形状：{data.shape}")

    # 网格定义
    strikes=np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 ])
    maturities=np.array([0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0 ])

    # xx: 参数
    ## 前 4 列代表网格所对应的参数
    xx = data[:, :4]
    print(f"参数形状：{xx.shape}")

    # yy: 隐含波动率曲面 
    # 后 88 列表示隐含波动率曲面 8 * 11 = 88
    yy = data[:, 4:]
    print(f"隐含波动率曲面形状：{yy.shape}")

    # 参数
    print(f"参数上界: {np.max(xx, axis=0)}")
    print(f"参数下界: {np.min(xx, axis=0)}")

    return xx, yy, strikes, maturities

# xx, yy, strikes, maturities = data_read(r"../../Data/rBergomiTrainSet.txt.gz")


# %%
from mpl_toolkits.mplot3d import Axes3D

def ImpVol_surface_3d( 
        xx_data: np.ndarray, 
        yy_data: np.ndarray, 
        K: np.array = np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 ]), 
        M: list = [0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0 ],
        params_index: int = 10000, 
        save_path = None
): 
    """
    绘制隐含波动率曲面
    
    Args:
        xx_data (np.ndarray): 参数
        yy_data (np.ndarray): 隐含波动率曲面
        K (np.array, optional): 行权价. Defaults to np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 ]).
        M (list, optional): 到期时间. Defaults to [0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0 ].
        params_index (int, optional): 参数索引. Defaults to 10000.
    """

    model_params = xx_data[params_index,:]
    
    if yy_data.shape[1] != len(K) * len(M): return None

    vol_surf_with_params = yy_data.reshape(-1, len(M), len(K))[params_index]

    x_axis, y_axis = np.meshgrid(M, K, indexing='ij')

    # Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf_to_plot = ax.plot_surface( 
        x_axis, 
        y_axis, 
        vol_surf_with_params, 
        cmap='viridis'
    )
    ax.set_xlabel('到期时间')
    ax.set_ylabel('行权价')
    ax.set_zlabel('隐含波动率')
    ax.set_title(f"隐含波动率曲面 (参数: {model_params.tolist()})")
    
    # color bar
    fig.colorbar(surf_to_plot, shrink=0.5, aspect=5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        
        print(f"图像已保存至: {save_path}")
    
    plt.show()
    
    return 

# ImpVol_surface_3d(xx, yy, params_index=19999)


