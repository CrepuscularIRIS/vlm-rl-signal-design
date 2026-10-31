# PO两种方法：SP与CA的设计及实跑公式

2026-10-31。对应已完成的两组32步实验，非新方案。研究动机和结果见[导师汇报](../context/MENTOR_UPDATE_20261031.md)。本页按冻结实现核对，不把原设计意图写成已达到的效果。

## 1. 共同主任务与状态

底座为相同Qwen3-VL-4B系列A-SFT。主任务沿用caption v5.1协议：合法终答正确为1，否则为0；没有新增裁图成本、幻觉奖励或KL奖励。

一题生成$G=8$条主轨迹，组均值$\bar R$落在$(0.01,0.99)$内才保留进入DAPO更新，并补采至128组。二元回报下就是保留混合组。主优势仍为：

$$
A_i^{\mathrm{main}}=\frac{R_i-\bar R}{\operatorname{std}(R_1,\ldots,R_G)+10^{-6}}.
$$

标准差沿用实现的样本标准差。主response mask只让模型生成的训练token参与，不把工具回执当作模型动作训练。

每个主step从最终保留的128组中按固定hash选$H=8$组；每组固定随机选一条rollout，再选它实际到达的一处pre-caption状态$s$。状态包括历史token、已见图像、裁图计数和剩余上下文／轮次预算。选择过程不按局部收益排序，不额外选择全错组。

## 2. SP：整轮候选的局部leave-one-out

从同一个$s$自然生成$M=2$个候选整轮$u_j$，并执行各自动作。每个非终止候选从执行后的状态做$K=4$次完整续写，用终答均值估计路线收益：

$$
\widehat Q(s,u_j)=\frac1K\sum_{k=1}^{K}R_{j,k}.
$$

已经终止的候选直接使用实际回报。候选的局部优势为：

$$
A_j^{\mathrm{SP}}=\widehat Q(s,u_j)-\frac1{M-1}\sum_{\ell\ne j}\widehat Q(s,u_\ell).
$$

本次$M=2$，即$A_1=\widehat Q_1-\widehat Q_2$、$A_2=-A_1$。不做标准差归一化。

整轮候选token都接收同一$A_j^{\mathrm{SP}}$，包括描述／思考和动作。历史prefix、工具回执以及估值续写不进入局部loss。重复候选、非法候选、零优势候选按原规则保留。

**它比较的是：同一状态下不同整轮选择的后续结果。** 不再把一个整轮内部的描述与动作分别归因。

## 3. CA：描述与动作的分段价值差

从$s$新生成一个整轮$u=(c,a)$。$c$是动作标签前的文本，包含`<think>…</think>`；$a$从`<grounding…>`或`<answer…>`开始，包含该动作输出的全部后续token。

定义：

- $h=(s,c)$：固定候选的描述，但后续动作重新生成。
- $s'$：固定并执行候选动作后的状态；裁图时包含实际新观察。
- $\widehat V(x)$：从$x$继续到终止的$K=4$个回报的均值，不是学习得到的critic。

估计$s$、$h$、非终止$s'$时使用独立随机种子。每支继承真实剩余预算，不从头给一份新的完整预算。

$$
A^{c}=\widehat V(h)-\widehat V(s),\qquad
A^{a}=r+\widehat V(s')-\widehat V(h).
$$

这里隐含$\gamma=1$。非终止动作$r=0$；本轮结束时$\widehat V(s')=0$，合法正确终答$r=1$，错误／非法／截断等$r=0$。不把终答回报同时计入$r$和$\widehat V(s')$。

$$
A_t^{\mathrm{CA}}=\begin{cases}
A^c,&t\in c,\\
A^a,&t\in a.
\end{cases}
$$

两段信用相加等于整轮的价值差：

$$
A^c+A^a=r+\widehat V(s')-\widehat V(s).
$$

这是信用数值的等式；训练时两段作用于不同token，并分别经过裁剪。候选是同一条实际生成文本，不重编码来拼接假想token。若描述与动作交界无法对应真实token边界，整轮使用$A^{\mathrm{fallback}}=r+\widehat V(s')-\widehat V(s)$。

**它比较的是：固定描述前后，以及固定动作前后，后续终答收益估计的变化。** 没有加入独立的视觉事实正确性评分器。

## 4. 实际优化目标

对当前待训练token，使用冻结旧策略的log-prob与当前log-prob之差$d_t$：

$$
\rho_t=\exp(\operatorname{clip}(d_t,-20,20)),\qquad
\bar\rho_t=\exp(\operatorname{clip}(d_t,\log0.8,\log1.28)).
$$

本次沿用主目标的非对称clip和dual clip=3。以最小化loss表示：

$$
b_t=\max(-\rho_t A_t,-\bar\rho_t A_t),\qquad
\ell_t=\begin{cases}
\min(b_t,-3A_t),&A_t<0,\\
b_t,&A_t\ge0.
\end{cases}
$$

主loss使用$A^{\mathrm{main}}$，局部loss使用SP整轮优势或CA分段优势。对每次optimizer更新，各自按有效token归一化，再相加：

$$
\mathcal L=\mathcal L_{\mathrm{DAPO}}+\beta\mathcal L_{\mathrm{local}},\qquad
\mathcal L_{\mathrm{local}}=\frac{\sum_{t\in\mathcal T_{\mathrm{local}}}\ell_t}{|\mathcal T_{\mathrm{local}}|}.
$$

分母跨两rank统计，包含零优势候选的token。空局部批次的局部loss为0，不转移到其他更新。不是先相加main/local advantage再做一次clip，也不是把SP和CA同时加在一个实验中。

所有局部估值都停止梯度；所有局部old log-prob都在本step四次optimizer更新开始前计算并冻结。局部样本归属于其主样本所在的optimizer minibatch，续写只提供回报估值。

## 5. 本轮参数与运行记录

| 项目 | SP | CA |
|---|---|---|
| 主训练 | DAPO，128题×8主rollout，32步 | 相同 |
| 局部状态数$H$ | 8/step | 8/step |
| 候选数 | $M=2$ | 1 |
| 每个非终止估值点续写数$K$ | 4 | 4 |
| 局部权重$\beta$ | 0.125 | 0.125 |
| 学习率／warmup | $10^{-6}$常数／1步 | 相同 |
| actor更新 | 32题一批，4次更新/step，ppo_epochs=1 | 相同 |
| clip／dual／max grad norm | 0.2/0.28，3，1 | 相同 |
| KL／主奖励 | 无／原0–1终答奖励 | 相同 |
| seed | 42 | 42 |

$\beta$按事先记录的探索规则，从$\{1,0.5,0.25,0.125\}$中选择：要求单步接线检查中两组最大局部／主梯度范数比乘$\beta\le0.5$。实测最大比值2.8869，因此选0.125，缩放比约0.36086；不是按E1分数调参。

两组原运行中断后都从完整step20恢复至累计32步。最终E1固定step32；CA原中断分支未保存的21–23步不算作恢复后的有效更新，其成本另列。完整结果见[最终报告](../results/po_final32_20261031/REPORT.md)。

## 6. 公式核对来源

以上SP、CA、mask与采样说明核对了`po_core.py`、`po_rollout.py`、`po_integration.py`、`run.sh`；四文件与正式运行冻结SHA一致。损失公式核对了运行环境的`compute_policy_loss`。文件摘要与函数对应见[公式来源](../provenance/PO_METHODS_FORMULAS.json)。仅作只读代码核对，未启动新训练或评测。
