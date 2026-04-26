# DogDebug

## 环境配置

```bash
# 1. 创建并激活虚拟环境
conda create -n rb python=3.10
conda activate rb

# 2. 从 PyTorch 官网安装 torch（带 cuda）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. 再安装剩余普通库
pip install -r requirements.txt
```
