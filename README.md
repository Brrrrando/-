# -
一个两层神经网络分类器
该文件包含了如下部分：
1、加载MNIST数据集，这里下载了必要的数据集，并通过gzip和numpy进行读入
2、分割训练集和测试集
3、将导入的数据归一化
4、Sigmoid函数和ReLU函数
5、自定义initialize函数定义初始的权重
7、定义了神经网络需要的正向传播反向传播和损失函数，并在反向传播之中加入了L2正则化
8、训练模型、参数查找以及最后的测试
