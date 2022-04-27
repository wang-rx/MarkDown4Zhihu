# 论文分享：Influence Towards Stable Multi-Agent Interactions

## Abstract

该文章发表于CoRL2021，属于对手建模工作，主要针对MA环境中由于对手策略改变所导致的agent学习的Non-stationary问题。通过对对手的strategy dynamics进行建模，agent可以预测自身动作对对手策略的影响。基于对手的strategy dynamics model，文章提出让agent学习如何使对手策略趋于稳定（单一的策略）从而促进自身的学习。

## Motivation

如上文所说，MA环境中Agent学习的non-stationary problem来源于对手变化的策略。对手建模是解决该问题的一类典型方法，现有的对手建模工作首先需要学习一个对手策略的动态模型（strategy dynamics model）。基于该模型，agent能够将自身行为对对手策略的影响考虑到策略优化中，通过影响对手的策略实现奖励最大化，从而应对non-stationary，典型的工作是这篇文章的前一篇”Learning latent representations to influence multi-agent interaction。然而这类对手建模方法存在以下问题：

（1）实际学习过程中的数据量有限，难以支撑对手模型的学习；

（2）基于对手策略动态模型的策略学习需要大量的对手策略数据，学习过程样本利用率低。

针对上述问题，该文章提出SILI方法：Stable Influencing of Latent Intent。SILI同样需要建模对手的strategy dynamics，但在基于strategy dynamics model的策略学习上，SILI不采用直接影响对手策略从而使自身Long-term reward最大化的方式，而是学习如何使对手策略趋于稳定从而减少自身学习的负担，从而提升学习效率。

以下图中的沙滩排球游戏为例，其中包含两个合作agent，当排球即将下落到两个agent中间位置时，此时agent需要决定由谁接球。此时ego agent如果采用以往的对手建模方法对Partner的strategy dynamics进行建模，则需要大量的交互数据，并且不断地在抢球和不抢球之间切换角色会让学习变得困难。此时，如果ego agent选择主动退后表明自己的意图，Partner就能很容易地选择接球（策略稳定）。这反过来简化了ego agent学习过程，减少了对学习Partner复杂策略动态的需要，使任务更加稳定。![](.\Pics\p1.png)

## Problem Statement

采用Two-player multi-agent setting，其中包含一个ego agent以及一个opponnet，考虑ego agent的学习。考虑Hidden Strategy：

### Hidden Strategy

* agent之间通过多次interaction进行交互，每次interaction持续多个时间步；
* 对手策略在每个Interaction内是固定的，但在interaction之间可以改变；
* ego agent无法直接观测到opponent的策略，因此决策过程可用Hidden Parameter MDP建模。

### Hidden Parameter MDP

可用元组表示：$\mathcal{M}=<\mathcal{S}, \mathcal{A}, \mathcal{Z}, \mathcal{T}, \mathcal{T}^z, \mathcal{R}, \mathcal{H}, \gamma>$

![](.\Pics\p2.png)

### Stable Opponent Strategy

* 对手策略稳定的定义：

![](.\Pics\p3.png)

* 稳定的对手策略可以表示为：

![](.\Pics\p4.png)

* 对比Influence opponent to maximize the reward和Influence opponent to be stable两种方式：

Influence opponent to maximize the reward: 需要在整个HiP-MDP上进行探索和学习。由于无法直接观测到对手的策略，ego agent难以准确建立对手模型并得到针对性的最优策略。

•Influence opponent to be stable: 只需要在fully observable MDP上进行探索学习可认为是HiP-MDP的子集。由于对手策略稳定，ego agent只需要在稳定的对手策略下进行学习，学习难度大大降低。

## Method

文章提出的SILI方法主要包含两部分：对手策略动态模型学习（Learning Latent Strategies）以及ego agent策略学习（Stable Influencing），如下图所示。

![](.\Pics\p5.png)

### Learning Latent Strategy

在对手策略动态模型学习中，采用encoder-decoder方式学习$j^{th}$interaction中对手策略的表示$z^j$。其中Encoder输入ego agent的观测轨迹${\tau}^{j-1}$，输出对手策略表示$z^j$，Decoder输入$t$时刻环境状态$s^j_t$,ego agent动作$a^j_t$以及$z^j$，预测$t+1$时刻环境状态$s^j_{t+1}$以及奖励$r^j_t$，encoder-decoder采用极大似然原则进行优化。

![](.\Pics\p6.png)

### Reinforcement Learning with Stable Latent Strategies

得到opponent strategy dynamics model后，很自然地将其考虑到ego agent的策略学习中。由于ego agent学习的目标是使得opponent的策略稳定，因此ego agent policy的输入应包含至少最近的两个对手策略预测，表示为$\pi_\theta$。Reward方面，除了任务奖励$\mathcal{R}_{task}$，文章定义了与对手策略稳定性相关的奖励$\mathcal{R}_{stable}$。

![](.\Pics\p7.png)

## 实验

实验环境包括：Circle Mass Point，Driving，Sawyer-Reach，Detour Speaker-Listener，这里以Circle Mass Point和Driving为例说明。在Circle Mass Point中，opponent只能在圆圈上的若干点位之间移动，ego agent的目标是接近opponent，但ego agent并不能直接观测到opponent的位置（策略）。Opponent遵循以下移动策略：当结束一次interaction时如果ego agent仍处于圆内，则逆时针移动一个点位，否则不动。在此场景中，agent的期望行为应该是移动到圆外从而使得opponent的策略稳定（不动），进而简化自身的策略训练。在Driving环境中，ego agent（蓝车）需要超越前方车辆（Opponent，灰车），但由于前方有路障，两车都需要进行变道。Opponent的策略是：当蓝车在红线之前超越自身（从左侧超车）时，灰车会自动往右侧慢车道变道，从而成功完成超车+避障过程。否则灰车同样将向左侧车道变道，引起碰撞。此场景下，蓝车应该学会在红线前超车从而避免冲突。

![](.\Pics\p8.png)

实验结果表明，文章提出的SILI方法在大部分场景上都取得了和Oracle方法相当的性能，其中在Driving环境中性能比Oracle更好。Oracle方法中ego agent能够直接观测到opponent的策略（fully observable MDP)，因此说明即使在SILI通过稳定对手策略使得ego agent在HiP-MDP的训练上得到简化。同时对比SILI和LILI（Influence opponent to maximize the reward）说明，相比于传统的opponent modeling方法，influence opponent to be stable在性能上具有优势。同时可视化ego agent的策略发现，在circle mass point和driving中ego agent都学到了前文所述的期望策略。

![](.\Pics\p9.png)

## 结论

文章针对MA环境中由于对手策略变化导致的agent学习不稳定问题，调整了传统对手建模方法的动机，通过稳定对手策略简化ego agent的策略学习，在实验场景中取得来了较好的效果。

论文地址：

前作（LILI）：https://arxiv.org/pdf/2011.06619.pdf

SILI：https://proceedings.mlr.press/v164/wang22f/wang22f.pdf

