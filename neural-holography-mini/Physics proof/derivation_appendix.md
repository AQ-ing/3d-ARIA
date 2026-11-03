## 附录 A：从波动方程到传播组合关系的逐步推导

以下证明对象是均匀介质内的单色标量传播，不是“任意图像经过少量训练必然泛化”的证明。采用时间因子 exp(-iωt)，正向波沿 +z 传播。λ 表示介质内波长；当前仿真按空气参数处理。

设标量场满足 Helmholtz 方程：

$$
\nabla^2 U+k^2U=0,\qquad k=2\pi/\lambda.
$$

在横向 x、y 上进行傅里叶变换，令频率向量 f=(fx,fy)，得到：

$$
\frac{\partial^2\widehat U}{\partial z^2}
+\left[k^2-4\pi^2(f_x^2+f_y^2)\right]\widehat U=0.
$$

选取向前传播解，在传播频带 fx²+fy² < λ⁻² 内：

$$
\widehat U(\mathbf f,z)=\widehat U(\mathbf f,0)e^{ik_z z},
\qquad
k_z=2\pi\sqrt{\lambda^{-2}-f_x^2-f_y^2}.
$$

对任意保留频率，H_a H_b=exp(ik_z a)exp(ik_z b)=H_(a+b)，逆变换得到组合性质。这里要求中间状态完整保留；若每步都增加截断、滤波或吸收，组合算子随之改变。

传播波频带上 |H_z|=1，由 Parseval 定理：

$$
\|\mathcal P_z U\|_2=\|U\|_2,\qquad
\|\mathcal P_z(U-V)\|_2=\|U-V\|_2.
$$

所以不能指望传播本身消除取相位造成的复场误差；它可以重新分布误差，使部分误差落到不评价区域。这也解释了 ROI 优化与完整复场保真的区别。

若共同带限投影为 B，则组合关系在 B 的值域成立，而不是对任意输入都等于恒等映射。若频带外含消逝波，反向恢复还涉及指数放大与噪声稳定性，不在当前验证范围内。

## 附录 B：载波对齐公式的逐步推导

令 f²=fx²+fy²，频率传递相位可写成：

$$
k_z=k+(k_z-k).
$$

为避免直接相减导致有效位损失，将后一项有理化：

$$
k_z-k
=2\pi\left(\sqrt{\lambda^{-2}-f^2}-\lambda^{-1}\right)
=-\frac{2\pi f^2}{\sqrt{\lambda^{-2}-f^2}+\lambda^{-1}}.
$$

因此 H_z=exp(ikz) exp[i(k_z-k)z]。把第二部分记作去载波核，固定目标复场 Q：

$$
V_0=e^{-ikz_0}\widetilde{\mathcal P}_{-z_0}Q,\qquad
V_z^{raw}=e^{-ikz}\widetilde{\mathcal P}_{-z}Q.
$$

乘以解析标量 exp[ik(z-z0)]：

$$
V_z^{aligned}
=e^{ik(z-z_0)}V_z^{raw}
=e^{-ikz_0}\widetilde{\mathcal P}_{-z}Q.
$$

在理想共同网格上：

$$
\widetilde{\mathcal P}_{z_0-z}V_0
=e^{-ikz_0}
\widetilde{\mathcal P}_{z_0-z}\widetilde{\mathcal P}_{-z_0}Q
=V_z^{aligned}.
$$

此等式证明参考相位可统一、物理迁移算子可共享；它不证明冻结编码器在所有深度必然高质量。29 dB 左右的结果属于本次实验支持。对复场施加全局旋转前后，其单独传播振幅相同，但非等变神经网络的输出可能不同。

在原代码中保留一次直接反传，再用 Python 双精度三角函数构造标量旋转，避免多次中间裁剪。相位补偿的正负号必须随原模型传播约定调整。

## 附录 C：多层复用、退化一致性与物理边界

定义 β_j=w_j exp[ik(zj-z0)]。由传播线性性和组合性质：

$$
\mathcal P_{z_k}\sum_j\beta_j\mathcal P_{-z_j}Q_j
=\sum_j\beta_j\mathcal P_{z_k-z_j}Q_j
=\beta_kQ_k+C_k.
$$

其中

$$
C_k=\sum_{j\ne k}\beta_j\mathcal P_{z_k-z_j}Q_j.
$$

不同层的空间分区互斥，并不让跨层传播项归零，因为一般有：

$$
\mathcal P_z(MU)\ne M\mathcal P_z(U).
$$

若 zj=z0、wj=1，所有载波补偿均为 1，且 ΣMj=1，则：

$$
V=\sum_j\mathcal P_{-z_0}(M_jQ)
=\mathcal P_{-z_0}\left[\left(\sum_jM_j\right)Q\right]
=V_0.
$$

因此共享编码器的输入与原始单平面一致。这是一个解析的一致性条件，不依赖适配器是否训练。

多平面强度并非可独立任意指定。所有平面受同一个 SLM 场、功率和空间带宽约束。当前以分区后的振幅拼接为目标，非目标区域不要求为黑色；远层在近层离焦、近层在远层离焦是预期现象。要验证真正的轴向选择性，还需要轴向扫描、离焦曲线和非目标区域评价。

## 附录 D：二次透镜相位迁移的完整代数步骤

设输出坐标 r、输入坐标 ρ，平方范数均为二维横向物理距离。将 L_z 乘到 U0 后，Fresnel 积分指数中与输入坐标有关的项为：

$$
\frac{|\mathbf r-\boldsymbol\rho|^2}{z}
+\left(\frac1{z_0}-\frac1z\right)|\boldsymbol\rho|^2
=\frac{|\mathbf r|^2}{z}
-\frac{2\mathbf r\cdot\boldsymbol\rho}{z}
+\frac{|\boldsymbol\rho|^2}{z_0}.
$$

令 m=z/z0、r'=r/m，注意 r/z=r'/z0。与 z0 传播积分比较：

$$
\frac{|\mathbf r'-\boldsymbol\rho|^2}{z_0}
=\frac{|\mathbf r'|^2}{z_0}
-\frac{2\mathbf r\cdot\boldsymbol\rho}{z}
+\frac{|\boldsymbol\rho|^2}{z_0}.
$$

两者只差与积分变量无关的输出二次相位：

$$
\frac{|\mathbf r|^2}{z}
-\frac{|\mathbf r'|^2}{z_0}
=|\mathbf r|^2\left(\frac1z-\frac{z_0}{z^2}\right).
$$

积分前因子比值为：

$$
\frac{e^{ikz}/(i\lambda z)}
{e^{ikz_0}/(i\lambda z_0)}
=\frac1m e^{ik(z-z_0)}.
$$

组合后：

$$
\mathcal P_z^F(U_0L_z)(\mathbf r)
=\frac1m e^{ik(z-z_0)}
e^{\,ik|\mathbf r|^2(1/z-z_0/z^2)/2}
\mathcal P_{z_0}^F U_0(\mathbf r/m).
$$

模长得到 A_z(r)=A_0(r/m)/m，平方得到强度倍率 1/m²。二维面积元随放大变化 m²，因此理想全平面积分功率守恒。

与薄透镜相位 exp[-ik|r|²/(2f)] 对比：

$$
\frac1f=\frac1z-\frac1{z_0}.
$$

z0=20 cm 时，z=18 cm 对应 f=+1.8 m，z=22 cm 对应 f=-2.2 m。有限支撑的输入 U0 同样满足这个 Fresnel 恒等式；孔径本身不能被误写为该恒等式失效的理由。

## 附录 E：哪些是证明，哪些是实验，哪些仍是研究问题

| 内容 | 证据性质 | 当前结论 |
| --- | --- | --- |
| 理想角谱传播组合关系 | 解析推导；共同网格数值检查 | 在规定频率子空间和无中间信息丢失条件下成立 |
| 去载波后的深度转换 | 解析代数 | 可以统一训练时的全局相位参考 |
| 冻结编码器保持高质量 | 单图、几个深度的仿真 | 本次为 29.13–29.79 dB，未证明跨图成立 |
| 一张纯相位同时显示两层 | 单图仿真 | 对齐后拼接 25.47 dB，仍有串扰和边界伪影 |
| 二次相位移动单层 | Fresnel 条件下的解析推导与 ASM 检查 | 伴随倍率变化，预变形后本次约 27 dB |
| 少量训练后的通用多深度模型 | 尚未完成验证 | 需未见图片、未见深度组合与成本实验 |

图表展示参考用户提供的 PHAROS 论文 Fig. 6–8 的并排重建与局部放大形式。该论文的任务、训练数据、分辨率和评价口径不同，本文没有进行与论文模型的同条件复现，也不把本文数值作为优于论文的证据。
