# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class NN_pricing(nn.Module): 
    """ 
    This basic architecture refers to "Deep Learning Volatility" by Horvath (2019).
    """
    def __init__(self, hyperparams):
        """
        hyperparams = { 
            'input_dim':5, 
            'hidden_dim':30, 
            'hidden_nums':3, 
            'output_dim':88
        }
        """

        super().__init__()
        self.input_dim = hyperparams['input_dim']
        self.hidden_dim = hyperparams['hidden_dim']
        self.hidden_nums = hyperparams['hidden_nums']
        self.output_dim = hyperparams['output_dim']

        # 架构
        self.layer_lst = [] 

        # 输入层
        ## 使用 ELU 激活, 参考 Theorem 2: Universal approximation theorem for derivatives (Hornik, Stinchcombe and White)
        
        self.layer_lst.append( 
            nn.Sequential( 
                nn.Linear(self.input_dim, self.hidden_dim), 
                nn.ELU()
            )
        )

        # 隐藏层
        for _ in range(self.hidden_nums-1): # 隐藏层数量-1
            self.layer_lst.append( 
                nn.Sequential( 
                    nn.Linear(self.hidden_dim, self.hidden_dim), 
                    nn.ELU()
                )
            )
        # 最后一个隐藏层
        self.layer_lst.append( 
            nn.Sequential( 
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        )

        self.model = nn.Sequential(*self.layer_lst)

    def forward(self, x):
        return self.model(x)
    


# %%
# 加入 Residual Block 改进
class ResNet_Block(nn.Module):
    def __init__(self, hyperparams):
        """ 
        hyperparams = {
            'hidden_dim':64,
            'block_layer_nums':3
        }
        """
        super(ResNet_Block, self ).__init__()

        self.hidden_dim = hyperparams['hidden_dim']
        self.block_layer_nums = hyperparams['block_layer_nums']


        # MLP
        self.layers = nn.ModuleList() 

        for _ in range(self.block_layer_nums):
            self.layers.append( 
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )

        # 正规化 Normalization
        self.layernorms = nn.ModuleList() 
        for _ in range(self.block_layer_nums): 
            self.layernorms.append( 
                nn.LayerNorm(self.hidden_dim)
            )


    def forward(self, x): 
        # 通过 MLP 前向通过
        out = x 
        for i in range(self.block_layer_nums): 
            out = self.layers[i](out)
            out = self.layernorms[i](out)
            out = F.relu(out)

        # 实现残差链接
        out = out + x 

        return out
        

# %%
class NN_pricing_ResNet(nn.Module):
    def __init__(self, hyperparams):
        """ 
        hyperparams = {
            'input_dim':4,
            'hidden_dim':64,
            'hidden_nums':10,
            'output_dim':88,
            'block_layer_nums':3
        }
        """
        super().__init__()
        self.input_dim = hyperparams['input_dim']
        self.hidden_dim = hyperparams['hidden_dim']
        self.hidden_nums = hyperparams['hidden_nums']
        self.output_dim = hyperparams['output_dim']
        self.block_layer_nums = hyperparams['block_layer_nums']


        self.layer_lst = [] 
        self.layer_lst.append( 
            nn.Sequential( 
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU()
            )
        )

        for _ in range(self.hidden_nums):
            self.layer_lst.append( ResNet_Block(hyperparams) )

        self.layer_lst.append( 
            nn.Linear(self.hidden_dim, self.output_dim)
        )

        self.model = nn.Sequential(*self.layer_lst)

    def forward(self, inputs): 
        return self.model(inputs)


# %%
class GatedBlock(nn.Module):
    """一个标准的GLU块"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 投影变换 (通常无偏置, 或与门共享)
        self.projection = nn.Linear(input_dim, output_dim)
        # 门控变换
        self.gate = nn.Linear(input_dim, output_dim)
        # 可选的残差连接 (如果维度匹配) 
        self.use_residual = (input_dim == output_dim)

    def forward(self, x):
        residual = x if self.use_residual else 0
        proj = self.projection(x)
        gate = torch.sigmoid(self.gate(x))  # 门控信号在0-1之间
        out = proj * gate + residual
        return out
    

class NN_pricing_GLU(nn.Module):
    """
    基于门控线性单元(GLU)的定价网络. 
    保留了MLP的全局连接性, 但通过门控机制增强了表达能力. 
    """
    def __init__(self, hyperparams):
        super().__init__()
        input_dim = hyperparams['input_dim']
        hidden_dim = hyperparams['hidden_dim']
        hidden_nums = hyperparams['hidden_nums']
        output_dim = hyperparams['output_dim']

        self.layer_lst = nn.ModuleList()

        # 输入层: 使用GLU块
        self.layer_lst.append(GatedBlock(input_dim, hidden_dim))

        # 隐藏层: 堆叠多个GLU块
        for _ in range(hidden_nums - 1):
            self.layer_lst.append(GatedBlock(hidden_dim, hidden_dim))

        # 输出层: 一个简单的线性投影
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # 可选的层归一化, 稳定训练
        self.use_norm = hyperparams.get('use_norm', True)
        if self.use_norm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(hidden_nums)])

    def forward(self, x):
        for i, layer in enumerate(self.layer_lst):
            x = layer(x)
            if self.use_norm:
                x = self.norms[i](x)
        x = self.output_layer(x)
        return x


