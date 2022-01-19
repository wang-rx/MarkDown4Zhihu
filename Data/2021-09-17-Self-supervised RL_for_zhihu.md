---
layout:     post
title:      自监督强化学习（一）—— 简介以及自监督状态表征强化学习
subtitle:   基于表征角度的自监督学习简介、state representation方法简介
date:       2021-09-18
author:     THY
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - RL
    - Self-supervised Learning
typora-root-url: ..\post_pic
typora-copy-images-to: ..\post_pic
---
---

# 自监督强化学习（一）—— 简介以及自监督状态表征强化学习

本文为自监督强化学习 (Self-Supervised Reinforcement Learning/SSRL) 系列介绍的第一部分，本系列主要介绍基于表征学习路线的自监督强化学习（下简称自监督强化学习或自监督表征强化学习）的问题和工作。本部分包括自监督强化学习的简介与部分状态表征 (State Representation) 方法简介。

本系列持续更新，欢迎大家关注交流讨论~




## 大纲（Outline）

* **简介 (Introduction)**
  * **强化学习与函数近似 (Reinforcement Learning and Function Approximation)**
  * **强化学习中的表征学习 (Representation Learning in RL)**
  * **为什么强化学习智能体 (RL agents) 需要自监督学习？**
* **自监督强化学习 (Self-Supervised Reinforcement Learning/SSRL)**
  * **自监督状态表征强化学习 (SSRL with State Representation)**
  * 自监督动作表征强化学习 (SSRL with Action Representation)
  * 自监督策略表征强化学习 (SSRL with Policy Representation)
  * 自监督任务/环境表征强化学习 (SSRL with Task/Environment Representation)
* 自监督表征强化学习的一些学习问题 (Learning Problems on Representation-based SSRL)
  * 抽象、近似与泛化理论 (Abstraction, Approximation and Generalization)
  * 基于表征进行学习时的一些问题 (Issues when Learning with Representations)
  * 在表征空间中的函数优化 (Optimization in Representation Space)
* 总结

上述大纲为本系列文章的所有内容，本文包含上述大纲中加粗标记的部分。



## 1. 简介

### 1.1 强化学习与函数近似 (Reinforcement Learning and Function Approximation)

强化学习 (RL) 是机器学习的主要分支之一，是用以学习求解通常建模为马尔可夫决策过程 (Markov Decision Process/MDP) 的序贯决策问题 (Sequential Decision-making Problem) 的有潜力的一种方法。

RL设计的主要要素有以下：

- 环境 Environment，即MDP  <img src="https://www.zhihu.com/equation?tex=M = <S,A,R,P,\gamma,\rho_0>" alt="M = <S,A,R,P,\gamma,\rho_0>" class="ee_img tr_noresize" eeimg="1"> )
- 智能体 Agent，主要为其策略 (Policy)  <img src="https://www.zhihu.com/equation?tex=\pi:S \rightarrow \Delta(A)" alt="\pi:S \rightarrow \Delta(A)" class="ee_img tr_noresize" eeimg="1"> 
- 智能体与环境交互 Agent-Environment Iteraction（如下图1-1所示）：

![./2021-09-17-Self-supervised RL/image-20210918200025744](/image-20210918200025744.png)

<center>图1-1：Agent-Environment 交互示意图</center>

- 目标 Objective
  - 最优策略 Optimal Policy  <img src="https://www.zhihu.com/equation?tex=\pi^{*}=\operatorname{argmax}_{\pi} \mathbb{E}_{\rho_{0}, \pi, M}\left[\sum_{t=0}^{\infty} \gamma^{t} R\left(s_{t}, a_{t}\right)\right]" alt="\pi^{*}=\operatorname{argmax}_{\pi} \mathbb{E}_{\rho_{0}, \pi, M}\left[\sum_{t=0}^{\infty} \gamma^{t} R\left(s_{t}, a_{t}\right)\right]" class="ee_img tr_noresize" eeimg="1"> 
  - 即agent在与环境的交互中，优化其策略以达到最优策略得以最大化长期累积收益
- 值函数 Value Functions
  - 状态值函数/V函数 State-value Function： <img src="https://www.zhihu.com/equation?tex=V^{\pi}(s)=\mathbb{E}_{\pi, M}\left[\sum_{t=0}^{\infty} \gamma^{t} R\left(s_{t}, a_{t}\right) \mid s_{0}=s\right]" alt="V^{\pi}(s)=\mathbb{E}_{\pi, M}\left[\sum_{t=0}^{\infty} \gamma^{t} R\left(s_{t}, a_{t}\right) \mid s_{0}=s\right]" class="ee_img tr_noresize" eeimg="1"> 
  - 动作值函数/Q函数 Action-value Function：  <img src="https://www.zhihu.com/equation?tex=Q^{\pi}(s, a)=\mathbb{E}_{\pi, M}\left[\sum_{t=0}^{\infty} \gamma^{t} R\left(s_{t}, a_{t}\right) \mid s_{0}=s, a_{t}=a\right]" alt="Q^{\pi}(s, a)=\mathbb{E}_{\pi, M}\left[\sum_{t=0}^{\infty} \gamma^{t} R\left(s_{t}, a_{t}\right) \mid s_{0}=s, a_{t}=a\right]" class="ee_img tr_noresize" eeimg="1"> 

RL的主要范式 (Paradigms) 包括：

- 从数据获取方式的角度区分：Online v.s. Offline
- 从学习基于的数据的来源进行区分：On-policy v.s. Off-policy
- 大部分的RL算法都遵循广义策略迭代 (Generalized Policy Iteration) 的基本范式

上面简要地介绍了RL的主要概念，其余RL的基本概念在本文中不再展开。



经典的基于表格的 (Tabular) RL通常考虑小规模有限个状态和动作 (finite states and actions)，这并不能适用于大规模（以及连续）状态和动作空间的问题 (large or continuous state-action space)。为此，RL通常采用函数近似 (Function Approximation/FA)，主要的成分 (keep components) 包括：

- 值函数近似器 Value Function Approximator (VFA)  <img src="https://www.zhihu.com/equation?tex=Q_{\theta},V_{\theta}" alt="Q_{\theta},V_{\theta}" class="ee_img tr_noresize" eeimg="1">  ， <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 为VFA的（可学习）参数
- 参数化的策略函数 Parameterized Policy Function  <img src="https://www.zhihu.com/equation?tex=\pi_{\phi}" alt="\pi_{\phi}" class="ee_img tr_noresize" eeimg="1">  , <img src="https://www.zhihu.com/equation?tex=\phi" alt="\phi" class="ee_img tr_noresize" eeimg="1"> 为policy的（可学习）参数
- 其他：环境模型 World Model 等等

常见的函数近似选择有：

- 线性函数近似 Linear FA
  - 状态-动作特征 (state-action features)  <img src="https://www.zhihu.com/equation?tex=x(s,a)" alt="x(s,a)" class="ee_img tr_noresize" eeimg="1">  （或state features  <img src="https://www.zhihu.com/equation?tex=x(s)" alt="x(s)" class="ee_img tr_noresize" eeimg="1"> ）；以及线性权重向量 (linear weight vector)  <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1"> 
  - E.g.,  <img src="https://www.zhihu.com/equation?tex=\hat{Q}(s,a)=w^{\top}x(s,a)" alt="\hat{Q}(s,a)=w^{\top}x(s,a)" class="ee_img tr_noresize" eeimg="1"> 
- 深度神经网络近似 Deep NN FA (Non-linear FA, thus DRL is derived)
  - 典型地，RL中的主要成分函数用DNN端对端地 (End-to-End) 进行拟合
  - E.g.,  <img src="https://www.zhihu.com/equation?tex=\theta, \phi" alt="\theta, \phi" class="ee_img tr_noresize" eeimg="1">  为DNN 参数
  - 进一步地，DNN网络的前若干层可以视作非线性表征网络



### 1.2 强化学习中的表征学习 (Representation Learning in RL)

深度神经网络函数近似下（Deep Learning中），表征学习不可避免地显式或者隐式地进行。下面简要地介绍表征学习的通用形式化描述。

考虑原始的数据空间 <img src="https://www.zhihu.com/equation?tex=x \in \mathbb{X} \in \mathbb{R}^{n}" alt="x \in \mathbb{X} \in \mathbb{R}^{n}" class="ee_img tr_noresize" eeimg="1"> ，这样的原始空间通常具有较高的维度 (high dimensionality) 以及与特定学习任务无关的冗余或者干扰的信息 (redundant and distractive information)。对于某一个学习任务，我们通常想要学习一个函数 <img src="https://www.zhihu.com/equation?tex=f_{\theta}(x)" alt="f_{\theta}(x)" class="ee_img tr_noresize" eeimg="1"> ，以 

- 近似某真实的函数 <img src="https://www.zhihu.com/equation?tex=F" alt="F" class="ee_img tr_noresize" eeimg="1">  
- 或是优化 <img src="https://www.zhihu.com/equation?tex=f_{\theta}(x)" alt="f_{\theta}(x)" class="ee_img tr_noresize" eeimg="1"> 使得其最大（或最小化）相应的目标函数 <img src="https://www.zhihu.com/equation?tex=J(f_{\theta})" alt="J(f_{\theta})" class="ee_img tr_noresize" eeimg="1"> 

一个表征（函数） <img src="https://www.zhihu.com/equation?tex=g:\mathbb{X} \rightarrow \mathbb{Z} \in \mathbb{R}^{m}" alt="g:\mathbb{X} \rightarrow \mathbb{Z} \in \mathbb{R}^{m}" class="ee_img tr_noresize" eeimg="1">  ，通常 <img src="https://www.zhihu.com/equation?tex=m << n" alt="m << n" class="ee_img tr_noresize" eeimg="1"> （因而表征空间通常是紧凑的，即**compact**）；一般来说，这样的表征函数可以参数化的 (parameterized) 也可以是非参数化的 (non-parameterized)。宏观来说，我们希望这样的表征能够帮助我们的学习过程，换言之， <img src="https://www.zhihu.com/equation?tex=\tilde{f}_{\theta}(z)" alt="\tilde{f}_{\theta}(z)" class="ee_img tr_noresize" eeimg="1"> 更容易学习或者优化。

一般来说，表征在各类学习问题中关键讨论的属性是其拟合和泛化的能力 (the ability of apprproximation and generalization) 。好的泛化能力通常指 <img src="https://www.zhihu.com/equation?tex=\tilde{f}_{\theta}(z)" alt="\tilde{f}_{\theta}(z)" class="ee_img tr_noresize" eeimg="1"> 在对一些已知数据进行学习拟合之后，能够对未知的数据 <img src="https://www.zhihu.com/equation?tex=x^{\prime}" alt="x^{\prime}" class="ee_img tr_noresize" eeimg="1">  (i.e., corresponding representation  <img src="https://www.zhihu.com/equation?tex=z^{\prime}" alt="z^{\prime}" class="ee_img tr_noresize" eeimg="1"> ) 得到高质量的估计（预测） <img src="https://www.zhihu.com/equation?tex=\tilde{f}_{\theta}(z^{\prime})" alt="\tilde{f}_{\theta}(z^{\prime})" class="ee_img tr_noresize" eeimg="1"> 。这通常也被叫做**“函数 <img src="https://www.zhihu.com/equation?tex=\tilde{f}" alt="\tilde{f}" class="ee_img tr_noresize" eeimg="1"> 在表征 <img src="https://www.zhihu.com/equation?tex=z" alt="z" class="ee_img tr_noresize" eeimg="1"> 上（或表征空间 <img src="https://www.zhihu.com/equation?tex=\mathbb{Z}" alt="\mathbb{Z}" class="ee_img tr_noresize" eeimg="1"> ）的泛化”**。泛化的好坏对于单任务学习（例如，泛化到学习过程中的新的阶段）和多任务学习（例如，泛化的新的任务），以及对于预测 (Supervised Learning Prediction) 和控制任务 (RL Control) 都具有重要意义。

泛化能力的强弱主要可以考虑由以下两方面决定：

- 表征函数 <img src="https://www.zhihu.com/equation?tex=g:\mathbb{X} \rightarrow \mathbb{Z} \in \mathbb{R}^{m}" alt="g:\mathbb{X} \rightarrow \mathbb{Z} \in \mathbb{R}^{m}" class="ee_img tr_noresize" eeimg="1">  ，例如SSRL过程中学习的state representation
- 学习的目标函数  <img src="https://www.zhihu.com/equation?tex=\tilde{f}" alt="\tilde{f}" class="ee_img tr_noresize" eeimg="1"> ，例如RL中VFA和policy network的归纳偏置 (Inductive Biases)

在当前的RL community中，前者在近两年的研究较多，后者在RL中研究地相对较少（其本身也是相对更一般化的DL问题）。在本系列中，主要讨论不同的表征学习方法对于不同RL学习场景下的理论和实验效用。



上面对一般化的表征学习和泛化进行了介绍，进一步地，考虑RL中的表征学习。

RL众所周知的两个主要缺陷是**样本效率低 (sample inefficient)、鲁棒性和泛化能力差 (not robust and poor in generalization)**，尤其体现在利用DRL解决实际的复杂任务的场景中。从表征的角度而言，上述缺陷的原因在于：

- **DRL中通过端对端的方式隐式学习的表征 (implicit representation learned in an end-to-end fashion) 不足以确保一个好的学习过程**
- **现有的方法缺乏对目标泛化对象的表征和利用，因而局限了DRL的近似和泛化能力**

遵循上述通用的表征学习描述，考虑表征在RL中的作用。首先，RL中学习的**目标函数 <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1">  (Functions to Approximate and Generalize)** 主要包括：

- Value Function  <img src="https://www.zhihu.com/equation?tex=Q^{\pi}(s, a),V^{\pi}(s)" alt="Q^{\pi}(s, a),V^{\pi}(s)" class="ee_img tr_noresize" eeimg="1"> 
- Policy  <img src="https://www.zhihu.com/equation?tex=\pi" alt="\pi" class="ee_img tr_noresize" eeimg="1"> 
- World model, e.g., dynamics function  <img src="https://www.zhihu.com/equation?tex=P(s^{\prime} |s,a)" alt="P(s^{\prime} |s,a)" class="ee_img tr_noresize" eeimg="1">  and reward function  <img src="https://www.zhihu.com/equation?tex=R(s,a)" alt="R(s,a)" class="ee_img tr_noresize" eeimg="1"> 

接着，考虑RL中需要**表征对象  <img src="https://www.zhihu.com/equation?tex=\mathbb{X}" alt="\mathbb{X}" class="ee_img tr_noresize" eeimg="1">  (Representations in RL)**，主要包括：

- **状态表征 (State Representation)**
  - 对环境中的状态进行表征，提取状态中与学习任务相关的信息以提升例如值函数、策略函数在状态空间中学习效率，或提取与单学习任务无关而对多任务通用的信息使得在多任务的学习中迁移状态表征来泛化和加速
- **动作表征 (Action Representation)**
  - 对动作进行表征，压缩大的、高维的动作空间，约减探索和学习的采样开销。提取动作语义，表征动作空间的内在结构，提升函数在动作空间中的近似与泛化表现，提升学习的效果
- **策略表征 (Policy Representation)**
  - 对策略进行表征，使得值函数、策略影响的环境动态等在策略空间能够泛化，拓展RL算法的研究领域。压缩策略空间，提供策略表征空间策略优化的可能。提供分析策略学习、演进过程的方式
- **任务/环境表征 (Task/Environment Representation)**
  - 对RL学习任务和环境进行表征，在表征空间建立多任务、多环境的相似与区别，使得多任务、多环境的学习互相泛化和促进

上述各表征角度的公共核心思想，是提取原始数据中关于RL学习任务的有效信息，压缩例如值函数学习拟合和策略优化的空间，同时学习到的RL函数在相似未知的数据上提供一定质量的泛化，提升RL整体的学习效率和泛化效果。



我们对表征在RL中的作用进行了直观和抽象的描述，具体而言，自然而然接下来的核心问题是：

- **好的表征应该是什么样的？**
  - 形式化框架，各类表征方法的属性和关系，表征的最优性定义和理论等
- **我们如何获得这样的好的表征？**
  - 表征学习方法，以及表征与RL结合的方式等

这两点分别对应了RL中表征的理论框架和方法，是本系列文章以及近年来工作关注的重点。



### 1.3 为什么强化学习智能体需要自监督学习？

前面提到仅基于端对端的学习，其表征并不能够确保一个好的RL学习过程，这是由于RL本身学习的**不稳定性**（数据分布以及拟合目标函数的变化）以及过程中序贯数据生成（online交互经验采集），使得基于**有限样本**、关于**局部学习任务**优化的表征所刻画的信息**易于特化**，不具备面向RL优化过程提供充分和有效的表征能力。

自监督学习提供了利用丰富的自监督数据设计自监督学习任务以提升RL表征的丰富性和鲁棒性的巨大空间，具有回答上述表征和RL学习问题的巨大潜力。在本系列中，我们将基于自监督表征学习的RL称为自监督强化学习 (Self-Supervised RL/SSRL)。更一般化地来说，RL领域也有其他实现“自监督强化学习”概念的研究方向，例如Skill Discovery, Pure Exploration研究等，而本系列中主要讨论从表征的角度出发的研究工作。

在本系列文章中，我们提出以下框架，从**三个关键方面 (Three Key Aspects)** 对各SSRL方法进行讨论和理解：

- **自监督信号 (Self-supervisions)**
  - 定义（非正式）：自监督信号是任何agent**创造 (*created* by agent, i.e., outer)** 或**拥有 (*possessed* by agent, i.e., inner)** 的信息
  - 例如，状态、动作、转移等MDP元素，agent的历史知识 (e.g., learned values, historical policies)
- **自监督表征学习 (Self-supervised Representation Learning)**
  - 辅助学习任务 (Pretext Tasks or Auxiliary Tasks)
  - 对比学习和数据增广 (Contrastive Learning and Data Augmentation)
  - 深度度量学习 (Deep Metric Learning)

- **基于自监督表征的RL (RL with Self-supervised Representation)**
  - 耦合与解耦 (Coupling v.s. Decoupling)
  - 泛化与优化 (Generalization v.s. Optimization)

对比学习和数据增广以及深度度量学习一般意义上来说也属于辅助学习任务，由于这两大类在近年来工作的主流趋势，这里我们将其单独列举。此外，端对端的表征学习中的表征实际上基于值函数拟合和策略优化进行。

基于自监督表征的RL的方式中的耦合与解耦是指表征学习的过程**与基于表征的（下游）学习任务是否耦合**，例如，常见的基于state representation进行online RL的场景中，state representation learning的过程是否依赖或涉及reward；若否，该场景中的RL with State Representation是解耦的，也意味着表征学习的过程不依赖由reward刻画的学习任务，因而学习得到的表征能够在相同环境动态中定义的不同学习任务上（考虑同一个仓库中agent的两项物流任务）进行运用。泛化与优化指代表征在RL中扮演的角色，前者表示利用近似函数关于表征输入的泛化性提升学习的效率和效果，后者则对应表征空间也是学习任务的解空间，从中进行表征优化以完成学习任务。

上述**三个关键方面**对应表征自监督的信息来源、表征的学习和构建方式以及表征在RL中的运用方式，本系列后续的讨论遵循该框架进行介绍、分析和总结。





## 2. 自监督强化学习 (Self-Supervised Reinforcement Learning/SSRL)

本章节介绍状态、动作、策略、环境/任务表征下的SSRL，主要围绕上一章节所述的Three Key Aspects，介绍当前研究的代表性方法以及系列工作发展脉络、分析不同思想和方法的优劣异同。



### 2.1 自监督状态表征强化学习 (SSRL with State Representation)

状态 (State) 是MDP中最主要的元素之一，state中包含了环境当前的完全信息，基于state定义的环境状态转移函数具有马尔可夫性 (Markov Property)。几乎所有RL中的主要函数都以state为输入，agent基于state决策与环境交互的动作、预测policy的value、推断环境的动态等等。

**为什么需要状态表征 (State Representation)？状态表征是对state中有效信息的提取，提升RL学习过程中的函数关于state的近似和泛化，是决定函数近似下RL学习效率和效果的一个根本性问题。**复杂的问题尤其是现实世界中的问题，环境、系统的状态中的信息维度较高，信息丰富但冗杂，如上面讨论的，基于有限样本、局部阶段的RL学习任务不足以对状态中提取出紧凑而充分的信息，因而自监督表征学习成为提升状态表征能力进而提升RL的方法之一。



#### 双时间尺度网络的框架 (The Framework of Two-timescale Network) [2]

我们采用Two-timescale Network作为常见state representation工作方法的通用描述框架，如下图2-1-1所示。环境state为某网络近似的函数（V-function为例）的输入，**整个网络可分视为前若干层组成的state representation network  <img src="https://www.zhihu.com/equation?tex=x_{\theta}(s)" alt="x_{\theta}(s)" class="ee_img tr_noresize" eeimg="1"> ，以及后若干层组成的下游 (downstream) 近似网络**  <img src="https://www.zhihu.com/equation?tex=\hat{V}(s)" alt="\hat{V}(s)" class="ee_img tr_noresize" eeimg="1"> ；另外，**存在一个以state representation  <img src="https://www.zhihu.com/equation?tex=x_{\theta}(s)" alt="x_{\theta}(s)" class="ee_img tr_noresize" eeimg="1"> 为输入的旁支网络输出 <img src="https://www.zhihu.com/equation?tex=\hat{Y}(s)" alt="\hat{Y}(s)" class="ee_img tr_noresize" eeimg="1"> ，用以优化某代理损失 (surrogate loss)**。一般化地，该surrogate loss可以理解成某种自监督表征学习损失函数，用以学习state representation。

<img src="https://raw.githubusercontent.com/mamengyiyi/Markdown4Zhihu/master/Data/2021-09-17-Self-supervised RL/image-20210918195959545.png" alt="image-20210918195959545" style="zoom:50%;" />

<center>图2-1-1：Two-timescale Nework （修改自[2]）</center>

在[2]的原文中，作者提出的Two-timescale Network具体包括多层组成的非线性的state representation network  <img src="https://www.zhihu.com/equation?tex=x_{\theta}(s)" alt="x_{\theta}(s)" class="ee_img tr_noresize" eeimg="1"> （也叫做*slow part*）以及最后以线性权重向量  <img src="https://www.zhihu.com/equation?tex=w,\bar{w}" alt="w,\bar{w}" class="ee_img tr_noresize" eeimg="1">  表示的线性函数（也叫做*fast part*）；在本部分中，我们借用并拓展其框架，考虑两部分一般化的函数表示，用做常见state representation工作的统一描述。

另外，此处我们用Two-timescale表示state representation以及downstream function学习过程的直观上的差异。通常而言，state representation的学习需要相对慢于downstream function的学习速率，以在温和条件下从理论上保证学习的收敛，以及实际实验中取得较为有效的结果。



#### 状态表征前言研究的分类  (A Taxonomy of Modern Arts of State Representation)

近年来的SSRL with state representation的工作从不同的角度提出了各自的贡献，在这里我们列举出主要的一些分支：

- **基于一般化无监督/自监督学习原则发展的状态表征 (State Representation Developed from General Un-/Self-supervised Learning Principles)**
  - E.g., DrQ [3], RAD [4], DrQ-v2 [5], CURL [6], Proto-RL [7], MPR [8], PlayVirtual [9]
- **基于RL元素构造的状态表征 (State Representation Built on RL Elements)**
  - E.g., Deep Bisimulation [10], Return-based Contrastive Learning [11], Actionable Representation [12]
- **基于模型的RL中的状态表征 (State Representation for Model-based RL)**
  - E.g., PlaNet [13], Dreamer [14], Cycle-Consistency World Model (CCWM) [15], Contrastively-trained Structured World Models (C-SWMs) [16]
- **奖励无关/解耦的状态表征 (Reward-agnositic/Decoupled State Representation)**
  - E.g., Proto-RL [7], Augmented Temporal Contrast (ATC) [17], Active Pre-Training (APT) [18], SPR, Goal-conditioned RL and Inverse Modeling (SGI) [19], APS [20]
- **其他角度下的状态表征 (Others)**
  - 网络架构 (Network Architecture) [21-23]
  - Actor-Critic状态表征干涉 (Actor-Critic State Representation Interference) [24-25]
  - 稀疏状态表征 (Sparse State Representation) [26-28]
  - ......

接下来，我们从上面列举的这些分支，详略地对各分支内的方法进行介绍和讨论。





后续内容见**《自监督强化学习（二）—— 自监督状态表征强化学习》**。







## 参考文献 (Reference)

1. Richard S. Sutton, Andrew G. Barto. Reinforcement Learning: An Introduction. IEEE Trans. Neural Networks 9(5): 1054-1054 (1998).
2. Wesley Chung, Somjit Nath, Ajin Joseph, Martha White. Two-Timescale Networks for Nonlinear Value Function Approximation. ICLR (Poster) 2019.
3. Michael Laskin, Aravind Srinivas, Pieter Abbeel. CURL: Contrastive Unsupervised Representations for Reinforcement Learning. ICML 2020.
4. Ilya Kostrikov, Denis Yarats, Rob Fergus. Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels.  arXiv:2004.13649.
5. Michael Laskin, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, Aravind Srinivas. Reinforcement Learning with Augmented Data. NeurIPS 2020.
6. Denis Yarats, Rob Fergus, Alessandro Lazaric, Lerrel Pinto. Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning. arXiv:2107.09645.
7. Denis Yarats, Rob Fergus, Alessandro Lazaric, Lerrel Pinto. Reinforcement Learning with Prototypical Representations.  ICML 2021.
8. Max Schwarzer, Ankesh Anand, Rishab Goel, R. Devon Hjelm, Aaron C. Courville, Philip Bachman. Data-Efficient Reinforcement Learning with Momentum/Self- Predictive Representations. arXiv:2007.05929.
9. Tao Yu, Cuiling Lan, Wenjun Zeng, Mingxiao Feng, Zhibo Chen. PlayVirtual: Augmenting Cycle-Consistent Virtual Trajectories for Reinforcement Learning. arXiv:2106.04152.
10. Amy Zhang, Rowan McAllister, Roberto Calandra, Yarin Gal, Sergey Levine. Learning Invariant Representations for Reinforcement Learning without Reconstruction. ICLR 2021.
11. Guoqing Liu, Chuheng Zhang, Li Zhao, Tao Qin, Jinhua Zhu, Jian Li, Nenghai Yu, Tie-Yan Liu. Return-Based Contrastive Representation Learning for Reinforcement Learning. ICLR 2021.
12. Dibya Ghosh, Abhishek Gupta, Sergey Levine. Learning Actionable Representations with Goal Conditioned Policies. ICLR (Poster) 2019.
13. Danijar Hafner, Timothy P. Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson. Learning Latent Dynamics for Planning from Pixels. ICML 2019.
14. Danijar Hafner, Timothy P. Lillicrap, Jimmy Ba, Mohammad Norouzi. Dream to Control: Learning Behaviors by Latent Imagination. ICLR 2020.
15. Changmin Yu, Dong Li, Hangyu Mao, Jianye Hao, Neil Burgess. Learning State Representations via Temporal Cycle-Consistency Constraint in Model-Based Reinforcement Learning. ICLR 2021 Workshop on SSL-RL.
16. Thomas N. Kipf, Elise van der Pol, Max Welling. Contrastive Learning of Structured World Models. ICLR 2020.
17. Adam Stooke, Kimin Lee, Pieter Abbeel, Michael Laskin. Decoupling Representation Learning from Reinforcement Learning. ICML 2021.
18. Hao Liu, Pieter Abbeel. Unsupervised Active Pre-Training for Reinforcement Learning. ICLR 2021 (rejected).
19. Max Schwarzer, Nitarshan Rajkumar, Michael Noukhovitch, Ankesh Anand, Laurent Charlin, R. Devon Hjelm, Philip Bachman, Aaron C. Courville. Pretraining Representations for Data-Efficient Reinforcement Learning. arXiv.2106.04799.
20. Hao Liu, Pieter Abbeel. APS: Active Pretraining with Successor Features. ICML 2021.
21. Kei Ota, Tomoaki Oiki, Devesh K. Jha, Toshisada Mariyama, Daniel Nikovski. Can Increasing Input Dimensionality Improve Deep Reinforcement Learning? ICML 2020.
22. Samarth Sinha, Homanga Bharadhwaj, Aravind Srinivas, Animesh Garg. D2RL: Deep Dense Architectures in Reinforcement Learning. arXiv:2010.09163.
23. Kei Ota, Devesh K. Jha, Asako Kanezaki. Training Larger Networks for Deep Reinforcement Learning.  arXiv:2102.07920.
24. Karl Cobbe, Jacob Hilton, Oleg Klimov, John Schulman. Phasic Policy Gradient. ICML 2021.
25. Roberta Raileanu, Rob Fergus. Decoupling Value and Policy for Generalization in Reinforcement Learning. ICML 2021.
26. Vincent Liu, Raksha Kumaraswamy, Lei Le, Martha White. The Utility of Sparse Representations for Control in Reinforcement Learning. AAAI 2019.
27. Yangchen Pan, Kirby Banman, Martha White. Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online. ICLR 2021.
28. Sina Ghiassian, Banafsheh Rafiee, Yat Long Lo, Adam White. Improving Performance in Reinforcement Learning by Breaking Generalization in Neural Networks. AAMAS 2020.

