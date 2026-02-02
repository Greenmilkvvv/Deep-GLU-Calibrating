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


# %%
# 工具函数——数据标准化
def x_transform(train_data, test_data, scale_x): 
    return scale_x.fit_transform(train_data), scale_x.transform(test_data)

def x_inv_transform(x, scale_x):
    return scale_x.inverse_transform(x)

def y_transform(train_data, test_data, scale_y): 
    return scale_y.fit_transform(train_data), scale_y.transform(test_data)

def y_inv_transform(y, scale_y):
    return scale_y.inverse_transform(y)


# # 训练集的 Upper and Lower Bounds
# upper_bound = np.array([0.16,4,-0.1,0.5])
# lower_bound = np.array([0.01,0.3,-0.95,0.025])

def params_scaler(x, upper_bound, lower_bound): 
    return (x - (upper_bound+lower_bound) / 2 ) * 2 / (upper_bound-lower_bound)

def params_inv_scaler(x, upper_bound, lower_bound):
    return x * (upper_bound-lower_bound) / 2 + (upper_bound+lower_bound) / 2


# 数据集划分
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_transform_train_test_data(xx, yy, upper_bound, lower_bound, test_size=0.15):
    """
    获取训练集和测试集
    Args:
        xx (np.ndarray): 参数
        yy (np.ndarray): 隐含波动率曲面
        upper_bound (np.ndarray): 参数上界
        lower_bound (np.ndarray): 参数下界
        test_size (float, optional): 测试集比例. Defaults to 0.15.
    Returns:
        x_train_transform, y_train_transform, x_test_transform, y_test_transform
    """
    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split( 
        xx, yy, 
        test_size = test_size, 
        random_state = 42
    )

    # 构建标准化的 train test data
    scale_x, scale_y = StandardScaler(), StandardScaler() 

    x_train_transform = params_scaler(x_train, upper_bound, lower_bound) 
    x_test_transform = params_scaler(x_test, upper_bound, lower_bound)

    y_train_transform, y_test_transform = y_transform(y_train, y_test, scale_y)

    return x_train_transform, y_train_transform, x_test_transform, y_test_transform


def get_torch_train_test_data(x_train_transform, y_train_transform, x_test_transform, y_test_transform, device = 'cuda'): 
    """
    将数据集转换为 torch tensor 并加载到 GPU 上
    Args:
        x_train_transform (np.ndarray): 标准化的训练集参数
        y_train_transform (np.ndarray): 标准化的训练集隐含波动率曲面
        x_test_transform (np.ndarray): 标准化的测试集参数
        y_test_transform (np.ndarray): 标准化的测试集隐含波动率曲面
        device (str, optional): 设备. Defaults to 'cuda'.
    Returns:
        data_loader, train_data, test_data
    """

    # 查找 GPU 
    if device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

    else: 
        device = torch.device('cpu')

    print(f"使用设备: {device}")


    train_dataset = torch.utils.data.TensorDataset( 
        torch.from_numpy(x_train_transform).to(device=device),
        torch.from_numpy(y_train_transform).to(device=device)
    )

    # test_dataset = torch.utils.data.TensorDataset( 
    #     torch.from_numpy(x_test_transform).to(device=device),
    #     torch.from_numpy(y_test_transform).to(device=device)
    # )

    train_data = (torch.from_numpy(x_train_transform).to(device=device),torch.from_numpy(y_train_transform).to(device=device))

    test_data = (torch.from_numpy(x_test_transform).to(device=device),torch.from_numpy(y_test_transform).to(device=device))


    data_loader = torch.utils.data.DataLoader( 
        train_dataset, batch_size=32, shuffle=True
    )

    return data_loader, train_data, test_data


def get_dataset_for_train(xx, yy, upper_bound, lower_bound, test_size=0.15, device = 'cuda'): 
    """ 
    获取训练集和测试集
    Args:
        xx (np.ndarray): 参数
        yy (np.ndarray): 隐含波动率曲面
        upper_bound (np.ndarray): 参数上界
        lower_bound (np.ndarray): 参数下界
        test_size (float, optional): 测试集比例. Defaults to 0.15.
        device (str, optional): 设备. Defaults to 'cuda'.
    Returns:
        data_loader, train_data, test_data
    """

    x_train_transform, y_train_transform, x_test_transform, y_test_transform = get_transform_train_test_data(xx, yy, upper_bound, lower_bound, test_size)

    data_loader, train_data, test_data = get_torch_train_test_data(x_train_transform, y_train_transform, x_test_transform, y_test_transform, device = device)

    return data_loader, train_data, test_data


