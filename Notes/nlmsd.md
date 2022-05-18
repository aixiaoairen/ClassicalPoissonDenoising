# A new bayesian Poisson denoising algorithm based on nonlocal means and stochastic distance

## Abstract

 泊松噪声是许多成像方式退化的主要原因。然而，许多已提出的用于降低图像噪声的方法缺少正式的方法，本文基于非局部均值框架，用更适合于去噪问题的随机距离代替欧式距离，提出了一种新的、通用的、形式化的、计算高效的贝叶斯泊松去噪算法。**利用泊松分布和伽马分布的共轭性来提高计算效率**。在处理低剂量CT图像时，该算法对**正弦图**进行运算，利用**伽马分布**对**泊松噪声率**进行**建模**。基于**贝叶斯公式**和**共轭性质**，**似然**服从**泊松分布**，**先验概率分布**也服从**伽马分布**。将该算法应用于模拟和真实的低剂量CT图像。

## 1. Introduction

作者基于**非局部均值NLM**图像去噪算法，使用**随机距离**代替其中的**欧式距离**提出了新的泊松去噪算法，并从泊松分布和伽马分布的共轭关系中推导其计算效率。

该工作对**投影正弦图**的**二维域**进行去噪处理。在正弦图域中，图像采集过程通过对光子的计数来执行，并且噪声由泊松概率分布适当的建模。在被泊松噪声污染的图像中，**噪声方差**取决于图像每个区域中**光子的平均速率**。

论文组织结构如下：

- 第2节：介绍了非局部平均算法的特点、泊松噪声的特征、泊松分布和伽马分布的共轭性质、伽马分布的随机距离和所提的方法
- 第3节：演示用所提出的滤波器获得的结果

- 第4节：讨论

- 第5节：结论

## 2. Methods

### 2.1 Nonlocal mean algorithm

NLM算法旨在获得真实图像的估计$\hat X = (\hat x_i | i \in I)$，其中$\hat x_i$是通过所有图像像素的**加权平均值**获得的，$X=(x_i|i \in I)$是定义在二维网格$I$上的无噪声图像，$x_i$是坐标$i$的像素值。噪声图像$Y=(y_i|i \in I)$是噪声破坏真实图像$X$的结果。**加权平均值**的计算如下所示：
$$
\begin{aligned}
\hat x_i = \sum_{j\in I}w_{ij}y_j
\end{aligned}
$$
其中，$0 \le w_{ij}\le 1, \ \sum_{j \in I}w_{ij}=1$。过滤器的权重$w_{ij}$取决于$i$和$j$附近之间的相邻像素值的**欧几里得距离**。根据Salmon的说法，大小为$15\times 15$的**搜索窗口**确保了良好的效果。**过滤器的权重$w_{ij}$**计算如下：
$$
\begin{aligned}
w_{ij}=\frac{1}{w_i}\exp(-\frac{1}{h^2}||y_i - y_j||^2_{2, a})
\end{aligned}
$$
其中，$w_i = \sum_{j \in I}w_{ij}$是**归一化因子**，$h$是控制过滤器扩散的过滤参数，$y_i,y_j$是集中在$i, j$中相邻块的灰度向量。

- $h \rightarrow 0$：图像变得更接近**噪声图像**
- $h \rightarrow ∞$：图像类似于*均值过滤器*的结果

表达式$||y_i - y_j||^2_{2, a}$定义如下
$$
\begin{aligned}
||y_i - y_j||^2_{2, a}=\sum_{t\in T}G_a(t)|y_{i-t}-y_{j-t}|^2
\end{aligned}
$$
$g_a$是具有**标准差$a$**的高斯核，$t$表示**相似性窗口**中的所有像素。

### 2.2 Poisson noise

由几十个光子撞击探测器形成的层析投影，在每个探测器中产生泊松噪声，由*score*得到的投影的每个点$y_i$的值是**随机过程$y_i \sim P(y_i)$**实现，由方程$P(y_i = k|\lambda_i)=\frac{e^{-\lambda_i}\lambda_i^{k}}{k!}$表示，其中$\lambda_i$表示探测器$i$中光子命中的平均速率。$y_i$是观察到的*score*。$P(y_i=k|\lambda_i)$是$y_i$取值为$k$的概率， $k\in Z^+$，$\lambda_i$是光子速率。

### 2.3 Conjugate distribution :  Poisson and gamma

该去噪工作的目的是去除CT图像的正弦图中的泊松噪声，正弦图的**样本服从泊松分布**，因此，贝叶斯公式中的**似然服从泊松分布**。采用泊松分布的先验—伽马分布，从而获得与先验分布一致的**后验分布**。

### 2.4 Stochastic distance

统计散度（statistical divergence）被用来衡量概率分布之间的离散程度。这些度量与对称性相结合，称为**随机距离**。**NLM**滤波器的目的是**比较图像中的两个窗口**，随机距离被用来代替**加性高斯噪声**情况下通常使用的**欧几里得距离**。

广义的散度：**Kullback Leibler**，**Renyi**，**Hellinger**和**Bhattacharyya**。实际上，这些散度并不是距离，但它们一旦不满足对称性，就很容易通过获得散度与其对称之间的平均值而转化为距离。

作者的方法其中包括**基于不同块的后验伽马分布之间的随机距离来推导NLM算法**。NLM算法搜索窗口的计算成本很高，具有$O(N^2.|P|)$的复杂度。Buader考虑到所有捕获的图像都有多个冗余区域，提出搜索窗口可以简化为**相似性窗口中心像素的集中窗口**。这要求**搜索窗口**必须高于**相似性窗口**，此时算法复杂度为$O(N|P|, \Omega)$，其中$|\Omega|$表示**搜索窗口中的相似性窗口**的**像素个数**。

### Proposed approach

作者的主要工作就是在NLM算法的基础上，使用**随机距离**来替代**原始的欧几里得距离**，在**随机距离**的计算中，分别使用*Kullback leibler, Renyi, Hellinger和Bhattacharyya*来计算后验分布。因为，似然是泊松分布，因此施加**伽马共轭先验**，并使用**变分近似法**推导出**后验伽马分布的具体形式**。

#### Kullback-Leibler distance

![image-20220517104821911](https://qiniu.lianghao.work/markdown/image-20220517104821911.png)

#### Renyi distance

![image-20220517104857547](https://qiniu.lianghao.work/markdown/image-20220517104857547.png)

![image-20220517104911757](https://qiniu.lianghao.work/markdown/image-20220517104911757.png)

#### Hellinger distance

![image-20220517105124980](http://qiniu.lianghao.work/markdown/image-20220517105124980.png)

#### Bhattacharyya distance

![image-20220517105221284](http://qiniu.lianghao.work/markdown/image-20220517105221284.png)

**描述原始无噪图的伽马分布参数的估计是通过对有噪正弦图进行局部均值滤波来获得的**。
$$
\begin{aligned}
\alpha^{(0)} &= \frac{u^2}{\sigma^2}\\
\beta^{(0)} &= \frac{u}{\sigma^2}
\end{aligned}
$$
**后验Gamma分布的参数**被表示如下
$$
\begin{aligned}
\alpha^{(1)}&=\alpha^{(0)}+\sum_{i=1}^n x_i\\
\beta^{(1)}&=\beta^{(0)}+n
\end{aligned}
$$
其中，$n$为**相似性窗口**的像素总数。

在图像过滤中，作者分别使用大小为$3\times 3, 5\times 5,7\times 7,9\times 9, 11 \times 11$的**相似性窗口**以及大小分别为$5\times 5, 7 \times 7, 9 \times 9, 11 \times 11, 13 \times 13$的**搜索窗口**。其中，所有计算中$h=0.1$，计算参数的**平滑窗口**为$3\times 3$。

## 3. Results

[Table 1]()列出了处理*Shepp-Logan, Asymmetric and Wood phantoms*图像获得的最佳PSNR结果。

## 4. Discussion

本文用三个不同的峰值模拟了以上三种图像，*peak*分别为$20, 40, 80$。**真实的图像不允许这种方法，图像在正弦图域中进行了过滤**。在图像重建过程中，所使用的重建算法**不支持**$20$以下的峰值。



## 5. 实验

### 5.1 执行 main_scipt.m 脚本

结果如下图所示，其中的*FBP，POCS*是CT图像重建算法

![image-20220517155702228](http://qiniu.lianghao.work/markdown/image-20220517155702228.png)

**注**：

- 执行脚本之前需要更改相关路径

- 在该路径下添加以下函数

  ```matlab
  function [img] = retroprojection(sinogram)
      sinogram = log(  max(max(sinogram))./sinogram  );
      
      
      [c l] = size(sinogram);
      theta = 180/l ;
      
      ang = linspace(0, 180, l);
      
      
      [img, H] =iradon(sinogram',ang,'linear','Shepp-Logan',1,l);
      
      img = (img - min(min(img))) / (max(max(img)) - min(min(img)));
      img = fliplr(img); 
  end
  ```

### 5.2 执行 exec_all_test.m 脚本

![image-20220517162958397](http://qiniu.lianghao.work/markdown/image-20220517162958397.png)

**执行失败，缺少函数 demoandre( )**，并且该函数是整个工作的核心部分，也就是作者所提出的方法的具体化。作者并没有给出该函数的代码。**wtf**

![image-20220517163027997](http://qiniu.lianghao.work/markdown/image-20220517163027997.png)



### 5.3 改造

为了保证不破坏作者的实验代码，我使用python写了制作测试数据集的方法，并在该程序中**直接调用**作者写的去噪的函数（没有任何改动）。

```pyton
def nlmSto(noisy):
	# 启动 matlab的engine
    eng = matlab.engine.start_matlab()
    # 含噪图像
    noisy = noisy.tolist()
    # 从python数据类型转为matlab的数据类型 double
    matlab_noisy = matlab.double(noisy)
    # 调用作者的去噪算法
    denoise = eng.nlmsdPoisson(matlab_noisy)
    # 返回去噪结果，并转为python的类型
    return np.array(denoise)
```

虽然，作者的实验是针对CT图像的，但是其方法是在NLM基础上进行改进，没有针对CT图像进行改进，因此我认为这是一个通用的算法。因此，我使用**BSD68**数据集进行测试，结果如下

|           | PSNR   | SSIM   |
| --------- | ------ | ------ |
| PEAK=1.0  | 6.916  | 0.0462 |
| PEAK=4.0  | 11.487 | 0.132  |
| PEAK=8.0  | 13.776 | 0.198  |
| PEAK=20.0 | 17.319 | 0.319  |
| PEAK=40.0 | 20.172 | 0.430  |

**另外，此工作在去噪时耗费时间过长，对340张灰度图像去噪，同等配置下，I+VST+BM3D运行2h，该工作运行13h。且都是python调用matlab来实现的**。



