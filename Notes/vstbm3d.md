# Variance Stabilization for Noisy+Estimate Combination in Iterative Poisson Denoising

[Github](https://github.com/aixiaoairen/PoissonDenosing/tree/master/pythonVersion)

## Abstract

作者使用**迭代算法**对泊松图像进行去噪，该算法逐渐地提高高斯滤波器的**方差稳定变换**(variance stabilization transformations, VST)。在每一个迭代中，泊松观测值与来自前一代的去噪估计的组合被视为缩放的(scaled)泊松数据，并通过VST模型进行过滤。由于真实缩放泊松分布与这种组合之间存在轻微的失配，因此设计了一种特殊的精确无偏逆（exact unbiased inverse）,作者给出了一个基于BM3D的实现方法。在计算成本仅为非迭代方案的两倍的情况下，所提出的算法提供了显著更好的质量，特别是在**低信噪比**的情况下，其性能优于更昂贵的最新方案。

## 1. Introduction

受泊松噪声影响的图像去噪通常是通过：

- 应用方差稳定变换（VST）来标准化图像噪声
- 应用AWGN滤波器去噪
- 通过逆变换将图像恢复到其原始范围

最常见的**VST**是**Anscombe**变换，该方法成本低、操作简单、不依赖所采用的去噪算法。在**计数非常低的情况下**（每个像素不到一个计数，信噪比为0分贝），Anscombe变换可能非常不准确。

在每一个步骤中，VST方法被应用于初始观测图像及其最新估计的组合，从而提高了稳定和过滤的有效性。

## 2. Preliminaries And Motivation

假设$z$由像素$z(x),x\in\Omega\sub Z^2$组成的**观测噪声图像**，建模为参数为$y(x)\ge 0$的泊松过程的独立实现：
$$
\begin{aligned}
z(x)\sim P(y(x)),\ P(z(x)|y(x))= \left\{ \begin{array}{l}
\frac{y(x)^{z(x)}}{z(x)!}\cdot e^{-y(x)},\ \ z \in N\cup\{0\} \\\
0,\ \ elsewhere
\end{array} \right.
\end{aligned}
$$
$z(x)$的**均值**和**方差**相等且等于**y(x)**。

目标是从$z(x)$中计算出$y$的估计值$\hat y$。为此，在原型**VST**框架中，**Ansombel**前向变换$a$产生图像
$$
\begin{aligned}
a(z(x))=2\sqrt{z(x)+\frac{3}{8}}
\end{aligned}
$$
可以**将其视为被具有方差的加性高斯白噪声（AWGN）破坏**。因此，**可以使用为**AWGN**设计的任何滤波器$\Phi$对其进行去噪**。

如果去噪是理想的：

**$\Phi[a(z(x))]=E\{a(z(x))|y(x)\}$，所谓的精确无偏逆$a$：$I_{a}^P:E\{a(z(x))|y\}\mapsto E\{z(x)|y(x)\}=y$被用于将去噪图像返回到$z(x)$的原始范围，从而产生无噪图像$y(x)$的估计：$\hat y=I_a^P (\Phi[a(z(x))])$**。

然而，对于小的$y(x)$，当信噪比很低时，稳定化是**不精确**的，并且$a(z(x))$的条件分布在**尺度**和**形状**上离假设的**正态分布**很**远**，导致**AWGN滤波器$\Phi$**滤波效果不佳。这个问题通常通过在**合并后应用VST**解决，即通过稳定相邻像素的总和而不是单个像素，或者通过类似地稳定变换系数（本质上是在去噪方法中插入VST本身）。这些策略都旨在提高**VST**影响的数据的**SNR**。

**作者使用了一种替代且更直接的方法来提高VST之前的SNR，通过将噪声观察$z(x)$与之前获得的无噪声数据估计值$\hat y(x)$相结合，从而产生简单的迭代算法**。

## 3. Proposed Iterative Algorithm

通过设置$\hat y_0=z$来初始化算法，在每一次迭代$i=1,2,\cdots,K$，计算$\hat y_{i-1}$和$z$的凸组合（convex combination）
$$
\begin{aligned}
\bar z = \lambda_iz+(1-\lambda_i)\cdot\hat y_{i-1}
\end{aligned}
$$
其中，$0<\lambda_i\le 1$。假设$\hat y_{i-1}$作为$y$的代理（surrogate），$E\{\bar z_i|y\}=y=\lambda_i^{-2}\cdot var\{\bar z_i|y\}$，**因为$\bar z_i$比$z$具有更高的SNR**。然后将**VST $f_i(\cdot)$**应用于$\bar z_i$，并获得图像$\bar{\bar{\bar{z_i}}}=f_i(\bar{z_i})$，使用**AWGN的滤波器$\Phi$对图像$\bar{\bar{\bar{z_i}}}$进行去噪以获得过滤后的图像$D_i=\Phi[\bar{\bar{\bar{z_i}}}]$**。假设$D_i=E\{f_i(\bar{z})|y\}$，$f_i$的精确无偏逆为$I_{f_i}^{\lambda_i}:E\{f_i(\bar{z_i})|y\}\mapsto E\{\bar{z_i}|y\}=y $，将这个图像恢复到原始范围，产生：
$$
\begin{aligned}
\hat y_i = I_{f_i}^{\lambda_i}(D_i)
\end{aligned}
$$

### A. Farward variance-stabilizing transformation

缩放变量$\lambda_i^{-2}\bar{z}$，将$\hat y_{i-1}$建模为$y$，设置$q_i(t)=\lambda_it-\frac{1-\lambda_i}{\lambda_i}y$，条件概率如下
$$
\begin{aligned}
P(\lambda_i^{-2}\bar{z_i}|y)=\left\{ \begin{array}{l}
\frac{y(x)^{q_i(\lambda_i^{-2}\bar{z_i})}}{q_i(\lambda_i^{-2}\bar{z_i})!}\cdot e^{-y(x)},\ \ q_i(\lambda_i^{-2}\bar{z_i}) \in N\cup\{0\} \\\
0,\ \ elsewhere
\end{array} \right.
\end{aligned}
$$
除非$\lambda_i=1$，否则这不是泊松分布。然而，$\lambda_i^{-2}\bar{z_i}$的均值和方差总是一致的：
$$
\begin{aligned}
E\{\lambda_i^{-2}\bar{z_i}|y\}=Var\{\lambda_i^{-2}\bar{z_i}|y\}=\lambda_iy
\end{aligned}
$$
因此，$\lambda_i\bar{z}_i$类似于$P(\lambda^{-2}y)$并且确实可以证明它是由$Anscombe$变换$a$鉴定稳定的，因此设定$f(\cdot)=a(\lambda^{-2}_i(\cdot))$。

### B. Exact unbiased inverse transformation

精确无偏逆$I_{f_i}^{\lambda_i}$在式（11）中定义为:
$$
\begin{aligned}
E\{f_i(\bar z_i)|y\}=\sum_{\bar z_i:q_i(\lambda_i^{-2}\bar z_i)\in N\cup\{0\}}a(\lambda_i^{-2}\bar z_i)P(\lambda_i^{-2}\bar z_i|y)\mapsto E\{\bar z_i|y\}=y
\end{aligned}
$$
其中，$I_{f_i}^{\lambda_i} \approx \lambda^2_iI_a^P,\ \ I_{f_i}^{1} \approx I_a^P$

### C. Binning

将凸组合（3）与一个线性**binning**相结合是很正常的。在第一次迭代时，当$\hat y_{i-1}$对$y$的估计很差时，这可能特别有用。具体：将**binning**运算符$B_{h_i}$应用于$z_i$，从而产生较小的图像，其中来自$z_i$的$h_i \times h_i$像素的每一块（即 **bin**）被等于其和的单个像素替换。
$$
\begin{aligned}
B_{h_i}[\bar z_i]=\lambda_i B_{h_i}[z]+(1-\lambda_i)B_{h_i}[\hat y_{i-1}]
\end{aligned}
$$
**Debinning**：在精确无偏逆之后进行反binning操作
$$
\begin{aligned}
\hat y_i = B_{h_i}^{-1}[I_{f_i}^{\lambda_i}(D_i)]
\end{aligned}
$$
返回一个全尺寸的图像估计值$\hat y$
$$
\begin{aligned}
B_{h_i}[\hat y]=I_{f_i}^{\lambda_i}(D_i)
\end{aligned}
$$


整个过程可以概括如下：
$$
\begin{aligned}
\hat y_i= B^{-1}_{h_i}[I_{f_i}^{\lambda_i}(\Phi[f_i(B_{h_i}[\lambda_i z+(1-\lambda_i)\hat y_{i-1}])]]
\end{aligned}
$$
![image-20220511101500785](http://qiniu.lianghao.work/markdown/image-20220511101500785.png)

## 4. Implementation and Results

**BM3D**作为高斯滤波器$\Phi$。

计算$B_{h_i}^{-1}[I_{f_i}^{\lambda_i}(D_i)]$：

1. 首先计算$I_{f_i}^{\lambda_i}(D_i)/ h_i^2$，即将$I_{f_i}^{\lambda_i}(D_i)$除以每个**bin**内的像素总个数$h_i^2$
2. 通过**cubic（三次？立方） spline interpolation $U_{h_i}$**放缩到$z$的尺寸
3. 为了加强约束$B_{h_i}[\hat y]=I_{f_i}^{\lambda_i}(D_i)$，$U_{h_i}$的输出被$B_{h_i}$递归地*binning*，并且从目标$I_{f_i}^{\lambda_i}$中减去，从而得到上采样（unsampled）和累积（accumulated）的残差（residual）。

![image-20220511101245468](https://qiniu.lianghao.work/markdown/image-20220511101245468.png)

算法1中的参数确定含义以及确定方式如下：

- $K$(迭代次数)：事先指定
- $\lambda_K$(凸组合中的权重系数)
- $h_1$(第一个bin的尺寸)
- $h_K$(最后一个bin的尺寸)

其中，$\lambda_i=1 - \frac{i-1}{K-1}(1-\lambda_K),\ h_i = \max\{h_k, h_1-2i+2\}$。在去噪的过程中，会递减**bin**的大小$h_i$，因为**binning**操作会**导致图像细节丢失**。$B_1，B_1^{-1}$是恒等算子。

![image-20220511172431101](http://qiniu.lianghao.work/markdown/image-20220511172431101.png)

![image-20220511172513973](http://qiniu.lianghao.work/markdown/image-20220511172513973.png)
