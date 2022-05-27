# Simultaneous Nonlocal Low-Rank And Deep Priors For Poisson Denoising

## Abstract

泊松噪声一种常见的电子噪声，广泛存在于各种光限成像系统中。然而，由于泊松噪声的**信号依赖性**（signal-dependent）和**乘性特性**（multiplicative characteristics），泊松去噪仍然是一个开放的问题。本文提出了一种**同时使用非局部低阶和深度先验**（simultaneous nonlocal low-rank and deep priors， SNLDP）的新方法。在混合即插即用的框架下，**SNLDP**同时使用了**非局部自相似**和**深度图像先验**，该框架包括**多对互补的先验**，即**非局部和局部**、**浅层和深层**、**内部和外部**。为了使优化问题易于处理，在**交替最小化框架**下提出了一种有效的**交替方向乘子**（ADMM）算法。实验结果表明，**SNLDP**在**量化**（quantitative）和**视觉感知**（visual perception）

## 1. Introduction

图像去噪是图像处理中的一个基本问题，当提到图像去噪任务时，通常指的是**去除图像中独立分布的加性高斯白噪声**。在许多成像应用中，例如*天文学*和*光谱*，**退化**是由**泊松噪声**引起的，**泊松噪声**是通过计算**入射到传感器**上的**光子数量**而获得的。与传统的AWGN不同，**泊松噪声依赖于信号**。标准的高斯去噪算法不能直接用于抑制泊松噪声，导致泊松去噪仍是一个开放的问题。

泊松去噪方法通常分为两类：

- **间接法**：基于**方差稳定变换**（VST）和**高斯去噪器**来恢复受泊松噪声污染的图像，最具代表性的方法是：**VST+BM3D**。然而，当光子计数率较低，即强度较低时，其去噪性能迅速减弱。
- **非VST的传统方法**：该类方法**抛弃**了**VST**策略，**直接**研究**泊松噪声的统计量**。这类方法在去除low-rank噪声时，取得了较好的效果。然而，由于重叠块的聚集，这些方法通常产生**视觉伪影**。
- **深度学习方法**：比较出色的卷积神经网络有**TNRD**和**CANET**结构。这些基于CNN的深度方法会导致**缺少泛化能力**，它们只考虑利用**图像的局部特性**，而忽略了**图像的非局部自相似性**，这可能会限制它们的有效性。

作者所提出的**SNLDP**在混合即插即用（hybrid plug and play, H-PnP）框架下**同时**利用**图像NSS**和**深层图像先验**，该框架包含多对**互补的先验**，即**非局部和局部**、**浅层和深层**、**内部和外部**。

泊松去噪的先进方法（state of the art, sota）有：**IRCNN、FFFDnet、LRS**

## 2. Prelimiaries

### 2.1 Poisson Statistics-based PnP Model

泊松去噪的目的是从观测的噪声图像$y\in Z_+^N$中**恢复**潜在的干净图像$x\in R^N$。假定$y$中的每一个观察到的像素值$y_i$是泊松分布的**独立随机变量**，其**均值**和**方差**等于潜在的干净图像$x$的像素值$x_i$。$y$的概率分布函数描述如下
$$
\begin{aligned}
P(y|x)=\left\{ \begin{array}{l}
\prod\limits_{i=1}^N \frac{x_i^{y_i}}{y_i!}e^{-x_i}, \  \  x_i>0\\
\delta_0(y_i),\ \ \ \ \ \ \ \ \ \ x_i = 0
\end{array} \right.
\end{aligned}
$$
其中，$\delta(\cdot)$为**Kronecker delta function**。

最近的研究表示**PnP**框架是一个有效的框架，使用有效的**高斯去噪器**来解决一般的逆问题，如**泊松去噪、图像去模糊和图像修复**。在泊松去噪任务中，**PnP框架首先通过$MAP$估计器来最大化概率分布$P(y|x)$**，在此基础上，给出观测到的模型，并且高斯去噪器被集成到变分框架中以解决以下映射问题。
$$
\begin{aligned}
\arg \mathop{\min}\limits_x <x-y\log x, 1>+\lambda\Upsilon(x)
\end{aligned}
$$
其中的第一项被称为**Csiszar I-divergence**模型，被广泛应用于之前的泊松去噪算法中。$\lambda$是正则化参数，$<\cdot>$是标准内积，$\Upsilon(\cdot)$表示正则化，依赖于适当的高斯去噪先验。

### 2.2 Nonlocal Low-Rank Model

非局部低阶模型将**相似的非局部块**进行**分组**，并利用每一个**Patch Group**的**低阶**性质。具体来说，首先将图像$x\in R^N$分为$n$个重叠的块$\{x_i\}_{i=1}^n\in R^{b\times 1}$。接着，以每个$x_i$作为参考*patch*，为参考块$x_i$收集$m$个最近的块，形成一个*patch group*，即$R_ix=X_i=\{x_{i,1},\cdots,x_{i,m}\}$，其中每个$R_i:x\mapsto R_ix\in R^{b\times m}$表示一个KNN运算符，被用来从$x$收集到第$i$个块的$m$个最近（通过计算欧几里得距离对比）的块。

由于每个块组**patch group**中的所有patch都具有相似的结构，因此构建的数据矩阵具有**low-rank**属性。因此，通过解决以下**最小化**问题，可以从每个块组$R_i x$估计**low-rank**矩阵$L_i$
$$
\begin{aligned}
\hat L_i = arg\mathop{\min}\limits_{L_i}\frac{1}{2}||R_ix-L_i||_F^2+\lambda D(L_i)
\end{aligned}
$$
其中，$D(L_i)$表示描述$L_i$的低秩属性的**正则化器**，$\lambda$是一个正常数。流行的正则化器包括核范数，加权核范数和秩残差，以被证明对图像恢复有效。然而，由于重叠*patch*的聚集，这些方法不可避免地会产生**视觉伪影**。

## 3. Proposed Method

### 3.1 SNLDP Model for Poisson Denoising

基于非局部的方法和基于深度CNN的方法各有优缺点。作者融合这两种方式来实现高效泊松去噪。**非局部低秩**和**深度去噪先验**被集成到基于**正则化**的框架中，在以下**SNLDP**问题中被视为**对偶正则化器**。
$$
\begin{aligned}
(\hat x, \hat L_i)=\arg \mathop\min\limits_{x, L_i}<x-y\log x, 1>+\frac{1}{2\rho}\sum_{i=1}^n||R_ix-L_i||^2_F+\\ \lambda\sum_{i=1}^nD(L_i)+\tau \Upsilon(x)
\end{aligned}
$$
其中，$\Upsilon(x)$表示深度去噪器，用于对整个图像$x$进行正则化，同时有效地**保留图像的精细细节**。$\rho$是使上述式子更可行的**平衡因子**。与单一的方法相比，**SNLDP**在*HPnP*框架下结合了两个互补的先验，假设底层图像$x$同时满足**低秩**和**深度先验**。

### 3.2 Optimization for the SNLDP Problem

为了使优化易于处理，在**alternative minimizing**框架下提出了一种有效的**ADMM**算法。具体来说，该算法首先通过$L_i,x$来**交替**求解优化方程，对应**低秩矩阵**逼近和图像更新子问题。此后，使用ADMM算法来解决图像更新子问题。

#### 3.2.1 Low-Rank Matrix Approximation: $L_i$ Sub-Problem

对于**固定**的$x$，在式（4）中的**低秩$L_i$子问题**变为
$$
\begin{aligned}
\hat L_i = \arg\mathop\min\limits_{L_i}\frac{1}{2}||R_ix-L_i||_F^2+\rho\lambda D(L_i)
\end{aligned}
$$
**注：作者并没有指定低秩正则化器$D(L_i)$**、作者采用**加权核范数最小化 WNNM**来描述$L_i$的低秩特性。

#### 3.2.2 Image Update : $x$子问题

对于固定的$L-i$，可以通过解决以下问题来更新图像$x$

![image-20220519151554654](http://qiniu.lianghao.work/markdown/image-20220519151554654.png)

为了便于优化，首先给出一个辅助变量$z=x$，然后使用ADMM算法，将图像更新子问题转化为3个迭代步骤：

![image-20220519151952913](http://qiniu.lianghao.work/markdown/image-20220519151952913.png)

其中，$p$代表**拉格朗日乘数**，$\alpha$表示平衡因子。**图像更新问题转化为两个子问题**：$x$子问题和$z$子问题。

- $z$子问题：给定$x, p$，子问题被改写为
  $$
  \begin{aligned}
  \hat z = \arg\mathop\min\limits_z=\frac{1}{2(\sqrt{\tau/\alpha})^2}||g-z||^2_2+\Upsilon(z)
  \end{aligned}
  $$
  其中，$g = x + \frac{p}{\alpha}$。从MAP的角度来看，求解$z$通常被视为噪声**标准差**为$\sqrt(\tau/\alpha)$的**高斯去噪问题**。因此，此类去噪问题的解决方可以描述为 $\hat z = H(g, \sqrt(\tau/\alpha))$。$H(\cdot)$表示基于**特定图像先验**$\Upsilon(\cdot)$的**高斯降噪器**。一般来说，可以使用任何高斯降噪器来逼近$H(\cdot)$，作者采用$FFFDNet$方法作为求解方程的高斯去噪器。

- $x$子问题：继续使用ADMM算法求解，引入另一个辅助变量$s=x$，然后继续调用ADMM算法

![image-20220519154127370](http://qiniu.lianghao.work/markdown/image-20220519154127370.png)

![image-20220519154301825](http://qiniu.lianghao.work/markdown/image-20220519154301825.png)

## 4. Experimental Results

作者将所提出的**SNLDP**与7种流行或最先进的方法进行比较：BM3D、PURE-LET、Dn-CNN、IRCNN、FFDNET、LRPD和LRS。

- BM3D、PURE-LET、Dn-CNN、IRCNN、FFDNET是伴随**VST**预处理的间接方法
- LRPD、LRS和SNLDP是利用泊松噪声统计的直接方法
- DnCNN、IRCNN 和 FFDNet 是三种基于深度 CNN 的高斯去噪方法
-  LRPD 依赖于 WNNM 算法，它是高斯去噪的最先进方法(before)。
- **LRS 是最近的直接泊松去噪方法，具有最先进的性能**。

| BSD68 Image Sets | PSNR   | SSIM |
| ---------------- | ------ | ---- |
| peak = 1.0       | 7.308  |      |
| peak = 4.0       | 21.213 |      |
| peak = 8.0       | 23.860 |      |
| peak = 20.0      | 26.419 |      |
| peak = 40.0      | 27.648 |      |

