# README

 此文件夹存放了我CSDN blog 中的code：
> 文件命名解读, `L + 数字` 表示第几讲课程, `xx_model`表示什么模型
> `L+数字 _ Exercise` 表示第几讲课程的练习题，Fig文件夹中存放的为每次实验后的一些损失函数图
> 
> 参考学习视频: up主刘二大人 [PyTorch深度学习实战](https://www.bilibili.com/video/BV1Y7411d7Ys/?spm_id_from=333.999.0.0),学习之前需要准备一些包包含：
> - torch
> - torchvision
- `L1` 第一讲 ---- 线性模型
  - `L1_linear_model.py` 是线性模型代码文件
  - `L1_Exercise.py` 是线性模型的练习题
- `L2` 第二讲 ---- 梯度下降算法
  - `L2_gradient_descent.py` 是梯度下降算法代码文件
  - `L2_SGD.py` 是随机梯度下降(Stochastic Gradient Descent)算法
- `L3` 第三讲 ---- 反向传播
  - `L3_back_propagation.py` 是反向传播算法代码文件
  - `L3_Exercise.py` 是反向传播算法的练习题
- `L4` 第四讲 ---- 线性回归
  - `L4_linear_regression.py` 是线性回归算法代码文件
    > 视频所留下的课后练习中除了LBFGS优化器不在`L4_linear_regression.py`里面，其他优化器包含了Adagrad、Adam、Adamax、ASGD、RMSprop、Rprop、SGD
  - `L4_Exercise.py` 是线性回归算法的练习题
    > `L4_Exercise.py` 里面只有LBFGS优化器，由于LBFGS优化器写法特殊性单独使用一个文件撰写
- `L5` 第五讲 ---- 逻辑回归
  - `L5_logistic_regression.py` 是逻辑回归算法代码文件
    > *注意*：在写这个代码时候我碰到了一个bug不知是什么情况，就是`torch.tensor`定义数据训练数据的时候*会报错*，但是我前四讲都是使用`torch.tensor`定义，为了解决此问题，我使用了`torch.Tensor`就可以成功运行且不报错
- `L6` 第六讲 ---- 多维输入
  - `L6_Multiple_Dimension.py` 是多维输入代码文件
- `L7` 第七讲 ---- 加载数据集
  - `L7_Dataset_DataLoader.py` 是加载数据集代码文件






> 深度学习中可视化还是比较重要的, 打印日志 或者 可视化图表
> - 可以采用 Visdom 库实现可视化

[返回](https://github.com/cug-auto-zp/CSDN)