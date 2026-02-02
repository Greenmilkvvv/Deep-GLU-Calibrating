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


# %%
class SwiGLUBlock(nn.Module):
    """SwiGLU块: 使用Swish函数作为门控激活"""
    def __init__(self, input_dim, output_dim, beta=1.0):
        super().__init__()
        # 投影变换 (值路径)
        self.projection = nn.Linear(input_dim, output_dim)
        # 门控变换 (门路径) - 注意: 这里没有内置激活函数
        self.gate = nn.Linear(input_dim, output_dim)
        # Swish函数的beta参数, 可学习或固定
        self.beta = beta
        # 可选的层归一化, 提升训练稳定性
        self.norm = nn.LayerNorm(output_dim) if input_dim == output_dim else None

    def swish(self, x):
        """Swish激活函数: x * sigmoid(beta * x)"""
        return x * torch.sigmoid(self.beta * x)

    def forward(self, x):
        residual = x if (self.norm is not None) else 0
        # 计算值投影和门控投影
        value = self.projection(x)
        gate = self.gate(x) # 注意: Swish函数在forward中应用
        # 应用SwiGLU: 值 * Swish(门)
        gated_value = value * self.swish(gate)
        # 可选: 应用层归一化
        if self.norm is not None:
            gated_value = self.norm(gated_value)
        # 残差连接（如果维度匹配）
        return gated_value + residual


class NN_pricing_SwiGLU(nn.Module):
    """基于SwiGLU的定价网络"""
    def __init__(self, hyperparams):
        super().__init__()
        input_dim = hyperparams['input_dim']
        hidden_dim = hyperparams['hidden_dim']
        hidden_nums = hyperparams['hidden_nums']
        output_dim = hyperparams['output_dim']
        # 可选的SwiGLU beta参数
        swiglu_beta = hyperparams.get('swiglu_beta', 1.0)

        self.layer_lst = nn.ModuleList()
        # 输入层
        self.layer_lst.append(SwiGLUBlock(input_dim, hidden_dim, beta=swiglu_beta))
        # 隐藏层
        for _ in range(hidden_nums - 1):
            self.layer_lst.append(SwiGLUBlock(hidden_dim, hidden_dim, beta=swiglu_beta))
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layer_lst:
            x = layer(x)
        return self.output_layer(x)
    

# %%
# Masked Gated Linear Unit

class MGLU(nn.Linear):
    """论文原版MGLU, 一个高效的线性投影拆分器. """
    def __init__(self, in_features, out_features):
        # 注意: 这里bias=False, 因为门控和值流通常不需要独立的偏置, 或者后续统一加. 
        super(MGLU, self).__init__(in_features, out_features, False)
        self.register_parameter(
            "mask", nn.Parameter(0.01 * torch.randn(out_features, in_features), requires_grad=True)
        )

    def ste_mask(self, soft_mask):
        """直通估计器: 前向二值化, 反向传播软掩码梯度. """
        hard_mask = (soft_mask > 0).to(soft_mask.dtype)
        # 这是STE的核心: 前向用hard_mask, 反向用soft_mask的梯度
        hard_mask = (hard_mask - soft_mask).detach() + soft_mask
        return hard_mask

    def forward(self, x):
        hard_mask = self.ste_mask(self.mask)
        # 计算互补的两路投影 e1, e2
        e1 = F.linear(x, self.weight * hard_mask)
        e2 = F.linear(x, self.weight * (1.0 - hard_mask))
        return e1, e2


class MGLUBlock(nn.Module):
    """完整的MGLU门控块, 可替换SwiGLUBlock. """
    def __init__(self, input_dim, output_dim, use_swish=True):
        super().__init__()
        # 核心: MGLU投影拆分器
        self.mglu_projection = MGLU(input_dim, output_dim)
        # 可选的偏置（也可不加, 或让MGLU支持bias）
        self.bias = nn.Parameter(torch.zeros(output_dim)) if True else None
        # 门控激活函数, 默认使用Swish
        self.gate_activation = lambda g: g * torch.sigmoid(g) if use_swish else torch.sigmoid(g)
        # 层归一化, 用于稳定训练（如果做残差连接, 维度需匹配）
        self.norm = nn.LayerNorm(output_dim) if (input_dim == output_dim) else None

    def forward(self, x):
        residual = x if (self.norm is not None) else 0

        # 1. 通过MGLU得到互补的两路投影 e1, e2
        e1, e2 = self.mglu_projection(x)

        # 2. 定义哪一路做“门”（需激活）, 哪一路做“值”
        #    论文中未明确, 这是你的设计选择. 一种常见设定是让e1做门, e2做值. 
        gate = self.gate_activation(e1)  # 对门控流应用Swish激活
        value = e2                      # 值流保持线性

        # 3. 门控相乘并加上偏置
        output = gate * value
        if self.bias is not None:
            output = output + self.bias

        # 4. 可选的层归一化和残差连接
        if self.norm is not None:
            output = self.norm(output)
        return output + residual


class NN_pricing_MGLU(nn.Module):
    """基于MGLU的定价网络"""
    def __init__(self, hyperparams):
        super().__init__()
        input_dim = hyperparams['input_dim']
        hidden_dim = hyperparams['hidden_dim']
        hidden_nums = hyperparams['hidden_nums']
        output_dim = hyperparams['output_dim']
        # 可选的SwiGLU beta参数
        swiglu_beta = hyperparams.get('swiglu_beta', 1.0)

        self.layer_lst = nn.ModuleList()
        # 输入层
        self.layer_lst.append(MGLUBlock(input_dim, hidden_dim, use_swish=True))
        # 隐藏层
        for _ in range(hidden_nums - 1):
            self.layer_lst.append(MGLUBlock(hidden_dim, hidden_dim, use_swish=True))
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layer_lst:
            x = layer(x)
        return self.output_layer(x)
    
