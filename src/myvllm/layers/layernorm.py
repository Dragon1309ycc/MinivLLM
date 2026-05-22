import torch
import time 
# 归一化层 : 放在注意力层或者前馈层琴后, 保证输入token的数值稳定
class LayerNorm(torch.nn.Module): # 继承自nn.Module--PyTorch 定义自定义层的标准方式。
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        # Use nn.Parameter to make gamma learnable and loadable from checkpoints
        self.weight = torch.nn.Parameter(gamma.detach().clone()) #detach().clone() 为了拷贝一个 不带梯度历史 的新张量
        self.eps = eps

    @property
    def gamma(self):
        """Backward compatibility: gamma alias for weight"""
        return self.weight

    @torch.compile # PyTorch 2.0+ 编译器装饰器，试图把 rms_forward 编译成更快的图
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ

        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps  # pow是Tensor对象的幂运算方法, dim=-1指定最后一维,keepdim=True 保留该维度
        sqrt_variance = variance.sqrt()     # 做除法时, 自动广播(Pytorch / Numpy这类数值计算库的机制)
        x_norm = (x / sqrt_variance * self.weight)

        return x_norm

    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = x + residual
        return self.rms_forward(x), x

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:   # Python 3.10+ 的 类型联合语法：A | None 等价于 Optional[A]
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            return self.rms_forward(x)

if __name__ == "__main__":
    # Example usage
    x = torch.randn(8,4000,8000).cuda()
    gamma = torch.full((8000,), 0.5, device="cuda", dtype=x.dtype)  # 生成一个指定形状、所有元素都等于 fill_value 的张量, (8000, )表示一个一维向量, 长度为8000
    layer = LayerNorm(gamma=gamma).cuda()
    residual = torch.full_like(x,fill_value=1) # “形状/设备/类型与 x 相同，值全是 1”。

    for _ in range(10):                     # Warm-up iterations
        _ = layer(x)                        # 调用 LayerNorm.forward() (模板化)
    
    # Without residuals
    times = [] 
    for _ in range(100):                    # Timing iterations
        torch.cuda.synchronize()            # 是让 CPU 等待 GPU 上已提交的任务全部完成，常用于精确计时，避免异步执行导致的计时不准
        start_time = time.time()
        _ = layer(x)
        torch.cuda.synchronize()
        end_time = time.time()              # time是顶部导入的time模块
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)      # 总时长除以次数
    print(f"[Without residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")

    # With residuals
    times.clear()
    for _ in range(100): # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x,residual)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[With residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
    
