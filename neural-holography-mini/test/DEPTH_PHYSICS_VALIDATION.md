# 单深度模型向多深度扩展：物理验证

日期：2026-11-03。全部验证与新增文件均在 test 下；未修改主目录，未更新网络权重。

## 结论与适用范围

深度之间存在明确、可复用的解析传播关系。本次发现：**反向传播的全局载波相位随深度改变，而冻结的最终相位编码器不具备全局相位等变性。**

仅修改物理传播深度，并把输入编码器的复场载波相位对齐到训练深度，无须新增网络或训练，即可把这张图的单平面重建从 20 cm 的 31.13 dB 迁移到 18/19/21/22 cm 的 29.13/29.79/29.72/29.38 dB。直接改距离而不对齐时只有 13.84/14.61/18.11/13.78 dB。

双平面同时显示时，分层反传、对齐、复场叠加、冻结编码器得到 25.47 dB，未对齐为 13.77 dB。多平面仍有相干串扰、有限孔径及编码误差，不能说已经达到单深度质量，也不能由单张图证明“少量训练即可泛化”。

本次是标量衍射仿真验证，不是光学硬件实验。基础 checkpoint 是 0031 的单图训练结果；当前证据只覆盖该图及附近几个深度。

## 设置与指标

- 图像/深度：data/image/0031.png、data/depth/0031.png。
- checkpoint：test/experiments/20261102_102239_off_reference/checkpoint/current.pth。
- 输入宽×高 1920×1072，中央评价区域 1600×880，绿色通道振幅。
- 波长 520 nm，像素间距 6.4 μm，训练深度 z0 = 0.20 m。
- 双深度 z1 = 0.18 m、z2 = 0.22 m；depth > 0.1 为近景。
- 训练步数 0，G 与 E 均冻结；沿用当前主目录 ASM。
- 指标：原始浮点振幅、峰值 1、ROI 内最优统一亮度尺度后的 PSNR，不是 RGB/sRGB PSNR。

设 ROI 为 Ω，a 为重建振幅，A 为目标振幅：

$$
s^*=\frac{\sum_{\Omega}aA}{\sum_{\Omega}a^2},\qquad
\mathrm{MSE}=\frac1{|\Omega|}\sum_{\Omega}(s^*a-A)^2,\qquad
\mathrm{PSNR}=-10\log_{10}\mathrm{MSE}.
$$

双平面使用 a = M1|P_z1 U| + M2|P_z2 U|，两层共用一个 s*。只评价各层应显示的像素；不能替代非目标区域、离焦表现和轴向串扰的评价。

## 1. 不同深度的基本关系

令 U 为复光场，F 为二维傅里叶变换，采用 exp(+ikz) 正向传播约定，k=2π/λ。在同一均匀介质内、保留传播波频带：

$$
\mathcal P_z[U]=\mathcal F^{-1}\{\mathcal F[U]H_z\},\quad
H_z(f_x,f_y)=
\exp[i2\pi z\sqrt{\lambda^{-2}-f_x^2-f_y^2}].
$$

同一频率网格、没有中间裁剪且没有额外丢弃频率时：

$$
\mathcal P_a\mathcal P_b=\mathcal P_{a+b},\qquad
\mathcal P_{-z}\mathcal P_z=I.
$$

若 U0 在 z0 的重建为 Q0=P_z0 U0，则

$$
U_z=\mathcal P_{z_0-z}U_0
\quad\Longrightarrow\quad
\mathcal P_zU_z=\mathcal P_{z_0}U_0=Q_0.
$$

这是保持横向坐标不变的复场深度迁移，与生成 U0 的网络结构无关。严格可逆性限于无损、未丢失的频率子空间，不含恢复消逝波或被裁掉的光场。

角谱传播与离散采样限制参考 Matsushima 与 Shimobaba 的 [Band-limited angular spectrum method (2009)](https://pubmed.ncbi.nlm.nih.gov/19997186/)；组合关系由频率传递函数直接相乘推得。

| 深度 | 公共加倍 FFT 网格、公共带宽、无中间裁剪的复场相对 L2 误差 | 加入中间 SLM 孔径后的相对误差 | 中间孔径外能量比例 |
| --- | ---: | ---: | ---: |
| 18 cm | 4.25e-7 | 8.61% | 0.75% |
| 19 cm | 4.34e-7 | 7.94% | 0.64% |
| 21 cm | 4.37e-7 | 4.74% | 0.23% |
| 22 cm | 4.34e-7 | 5.85% | 0.35% |

第一列验证共同周期网格上的传播代数，不是无穷空间或硬件精度保证。复场 L2 相对误差与能量百分比不是同一种量。

## 2. 为什么传播正确，纯相位迁移仍会失败

现有 SLM 可实现 U_SLM=a_in exp(iφ)，本仿真 a_in=1。迁移后 U_z=B_z exp(iθ_z) 一般不满足 B_z=1。直接取角度等于：

$$
\widehat U_z=e^{i\theta_z},\qquad
e_z=\widehat U_z-U_z=(1-B_z)e^{i\theta_z}.
$$

重建误差为 P_z e_z，迁移所需的振幅信息不能无损丢弃。

| 目标深度 | 原相位不变、在新深度看 | 迁移后保留复振幅 | 迁移后只取相位 |
| --- | ---: | ---: | ---: |
| 18 cm | 17.65 | 30.25 | 16.58 |
| 19 cm | 18.36 | 30.70 | 13.20 |
| 21 cm | 18.64 | 29.88 | 13.83 |
| 22 cm | 17.75 | 29.53 | 16.24 |

单位 dB。复振幅列是诊断参考，单片纯相位 SLM 不能直接显示，也不是多平面性能上界。

当前实现每次传播包含零填充 E_pad、带限 B_z、裁剪 C：

$$
\mathcal P_z^{code}
=C\mathcal F^{-1}B_zH_z\mathcal FE_{pad}.
$$

中间 E_pad C、随距离变化的 B_z 以及数值精度，阻止连续两次代码调用严格约掉。当前核还使用 float32 计算约 2.4e6 rad 的载波相位，存在精度风险。

公共网格验证采用 float64 构造去载波核，再转 complex64：

$$
\widetilde H_z
=\exp\left[
-i2\pi z\frac{f_x^2+f_y^2}
{\sqrt{\lambda^{-2}-f_x^2-f_y^2}+\lambda^{-1}}
\right].
$$

主目录传播实现未修改。替换传播核可能影响旧 checkpoint，不能直接视为无影响优化。

## 3. 当前模型可复用的关键：统一载波相位参考

提出空间不变载波：

$$
H_z=e^{ikz}\widetilde H_z,\qquad
\mathcal P_z=e^{ikz}\widetilde{\mathcal P}_z.
$$

当前 HoloNet 的两个冻结网络与反传：

$$
\psi=G_\theta(A),\quad Q=Ae^{i\psi},\quad
V_0=\mathcal P_{-z_0}Q,\quad
\phi_0=E_\theta(|V_0|,\arg V_0).
$$

G 是 initial_phase，E 是 final_phase_only。直接改变距离：

$$
V_z^{raw}=\mathcal P_{-z}Q=e^{-ikz}\widetilde{\mathcal P}_{-z}Q.
$$

空间包络变化小，也可能伴随全局相位跨越多个周期。整体相位不影响单个光场的强度；但 E 输入原始 angle，没有强制周期连续性或下述全局相位等变性：

$$
E_\theta(B,\mathrm{wrap}(\vartheta+\alpha))
\equiv E_\theta(B,\vartheta)+\alpha\pmod{2\pi}.
$$

对齐到训练时的载波参考：

$$
\boxed{
V_z^{aligned}
=e^{ik(z-z_0)}\mathcal P_{-z}Q
=e^{-ikz_0}\widetilde{\mathcal P}_{-z}Q.
}
$$

不改变复场振幅，不需要监督或参数拟合。继续复用 E：

$$
\phi_z=E_\theta(|V_z^{aligned}|,\arg V_z^{aligned}),\qquad
\widehat A_z=|\mathcal P_z e^{i\phi_z}|.
$$

理想共同网格下还可写成：

$$
\boxed{V_z^{aligned}=\widetilde{\mathcal P}_{z_0-z}V_0.}
$$

实际优先一次直接反传再做全局对齐，避免多次裁剪。

| 深度 | 直接改反传距离，冻结 G/E | 加载波对齐，冻结 G/E |
| --- | ---: | ---: |
| 18 cm | 13.84 | **29.13** |
| 19 cm | 14.61 | **29.79** |
| 20 cm | 31.13 | **31.13** |
| 21 cm | 18.11 | **29.72** |
| 22 cm | 13.78 | **29.38** |

先前简单改距离的失败，很大一部分来自相位参考变化，不能据此认定编码能力不可复用。剩余差距可能含输入分布、孔径、带限和浮点误差，尚未逐项消融。

其他模型若已去载波或传播符号不同，应匹配其训练约定，不能机械乘这个因子。本方法要求访问中间复场与编码器，尚非任意黑盒模型的通用插件。

## 4. 多平面同时显示的公式与剩余问题

二值分区满足 Σ_j M_j=1，在目标平面分层：

$$
Q_j=M_j A e^{iG_\theta(A)}.
$$

各层反传并统一载波参考：

$$
\alpha_j=e^{ik(z_j-z_0)},\qquad
V=\sum_j w_j\alpha_j\mathcal P_{-z_j}Q_j.
$$

本次 w_j=1，不拟合权重。只调用一次共享 E：

$$
\phi=E_\theta(|V|,\arg V),\qquad
\widehat A_j=|\mathcal P_{z_j}e^{i\phi}|.
$$

增加层数只增加解析传播与求和，不要求每层复制 U-Net。若所有 z_j=z0 且 w_j=1，则 V=P_-z0 Q，退化为原单深度模型。分区在目标平面完成，不能把目标 mask 直接乘在 SLM 相位上当作等价操作。

即使保留完整复场，在第 k 层也有：

$$
\mathcal P_{z_k}V
=w_k\alpha_kQ_k+
\underbrace{\sum_{j\ne k}w_j\alpha_j
\mathcal P_{z_k-z_j}Q_j}_{C_k}.
$$

强度为：

$$
|\mathcal P_{z_k}V|^2=
|w_kQ_k|^2+|C_k|^2+
2\operatorname{Re}(w_k\alpha_kQ_kC_k^*).
$$

目标 mask 不重叠不意味着传播后的 C_k=0。全局相位不影响单层强度，但层间相对相位影响交叉项；对齐是一种确定的相位选择，不保证串扰最小。

纯相位编码还引入 e_enc=e^{iφ}-V：

$$
\mathcal P_{z_k}e^{i\phi}
=w_k\alpha_kQ_k+C_k+\mathcal P_{z_k}e_{enc}.
$$

可学习模块应针对串扰和编码残差，把已知传播留给物理算子。这是误差分解，不表示各项独立或必须单独最小化。

| 方法 | 合成 PSNR | 近景 PSNR | 远景 PSNR |
| --- | ---: | ---: | ---: |
| 两层各自反传、重建后拼接（非同时显示，复场诊断） | 27.14 | 28.17 | 25.39 |
| 未对齐，叠加复场直接传播 | 23.90 | 24.63 | 22.56 |
| 对齐后，叠加复场直接传播 | 23.83 | 24.55 | 22.50 |
| 未对齐，叠加后经过冻结编码器 | 13.77 | 13.61 | 14.17 |
| **对齐后，叠加后经过冻结编码器** | **25.47** | **25.92** | **24.58** |

E 优化重建振幅，不要求忠实再现这个指定复场，结果高于简单复场求和不矛盾。未对齐复场求和时，其他层在本层目标区域内的能量约为目标层的 1.79%/1.51%，相干交叉项仍可造成明显误差。预览可见分区边界振铃和纹理残留。

此前逐像素相位优化的独立复算是 48.7961 dB，支持当前仿真和拼接目标存在高质量相位解。这不是本次零训练插件的成绩，也不证明非目标区域离焦表现或跨图泛化。

## 5. 通用输出接口的另一条路线：二次透镜相位

若只有基础模型输出 φ0，在 Fresnel 近似下可加：

$$
L_z(\mathbf r)=\exp\left[
\frac{ik|\mathbf r|^2}{2}
\left(\frac1{z_0}-\frac1z\right)
\right],\qquad m=\frac z{z_0}>0.
$$

在连续 Fresnel 积分

$$
\mathcal P_z^F U(\mathbf r)
=\frac{e^{ikz}}{i\lambda z}
\int U(\boldsymbol\rho)
\exp\left[\frac{ik}{2z}|\mathbf r-\boldsymbol\rho|^2\right]d^2\boldsymbol\rho
$$

中代入 U0Lz，合并 ρ² 系数、令 r'=r/m：

$$
\mathcal P_z^F(U_0L_z)(\mathbf r)
=\frac1m e^{ik(z-z_0)}
\exp\left[\frac{ik|\mathbf r|^2}{2}
\left(\frac1z-\frac{z_0}{z^2}\right)\right]
\mathcal P_{z_0}^F U_0(\mathbf r/m).
$$

所以

$$
A_z(\mathbf r)=\frac1m A_0(\mathbf r/m),
\qquad I_z(\mathbf r)=\frac1{m^2}I_0(\mathbf r/m).
$$

保持纯相位，但放缩图像。同尺寸输出需输入预变形 A_pre(r')=m A(mr')。本次省略整体 m 倍振幅以保持输入范围，由评价尺度 s* 处理全局亮度，保留空间预变形。

Fresnel 核和数字透镜方法参见 Verrier 与 Atlan 的 [Off-axis digital hologram reconstruction: some practical considerations (2011)](https://www.institut-langevin.espci.fr/biblio/2020/3/5/744/files/2011_off-axis_digital_hologram_reconstruction_some_practical_considerations.pdf)；以上 z0 到 z 的等式由该积分自行推导。

当前实际 ASM 上，“预变形 → 原模型 → 加二次相位”的零训练结果：

| 深度 | m | 与原始同尺寸目标比较的 PSNR |
| --- | ---: | ---: |
| 18 cm | 0.90 | 27.28 |
| 19 cm | 0.95 | 27.16 |
| 21 cm | 1.05 | 26.83 |
| 22 cm | 1.10 | 26.82 |

不预变形时，实际重建与公式放缩后的原重建之间约 32.3–32.7 dB。这是两张重建的一致性，不是目标重建 PSNR。缩放会把原 ROI 外未约束区域带入评价窗口，不能直接横比不同倍率的目标 PSNR。

有限输入孔径本身不破坏连续 Fresnel 缩放等式。实际限制是 Fresnel/ASM 差异、离散采样和带限、输出截取、预变形越界、插值及模型对变形输入的能力。公式移动整个单层，不能将不同深度的相位直接相加来实现多层光场叠加。

## 6. 后续低成本适配的合理起点

~~~text
RGB 振幅 A ── 冻结 G ── 目标复场 Q
                         │
depth ── 分层 Mj ─────────┤
                         ▼
                  Qj = Mj Q
                         │
        已知深度反传 + 训练载波参考对齐
                         │
                    复场求和 V
                         │
                 少量可学习修正（待验证）
                         │
                    冻结编码器 E
                         │
                    一张 SLM 相位
                         │
                  各深度 ASM 重建
~~~

目前跑通的是没有“可学习修正”的版本。修正应考虑复场实部/虚部、周期相位或传播残差；结构与训练成本还需实验决定。冻结主干并不保证任意修正有效，本次不宣称找到最终最小模块。

先用多张未见图验证规范对齐，再少量适配，并测试未见深度和层数组合。分别报告单深度质量、各层对应区域、合成图、完整平面离焦表现与推理成本，才能判断是否满足可复用、低成本目标。

## 文件与复现

从 neural-holography-mini 运行，均无优化器、无训练：

~~~powershell
conda run --no-capture-output -n e2e python -B test/validate_depth_physics.py
conda run --no-capture-output -n e2e python -B test/validate_quadratic_depth.py
conda run --no-capture-output -n e2e python -B test/validate_encoder_gauge.py
~~~

结果：

- test/experiments/20261103_110638_depth_physics/results.json：传播组合、相位投影与多层诊断。
- test/experiments/20261103_150105_quadratic_depth/results.json：透镜相位与预变形。
- test/experiments/20261103_150411_encoder_gauge/results.json：完整单/双深度对齐，output 保存各平面、合成图和目标。
- test/experiments/20261103_150309_encoder_gauge/results.json：较早的单深度对齐验证，以完整结果为准。
- test/experiments/20261103_101235_261660_phase_adapter_diagnosis/pixel_lr01/verified_metrics.json：历史逐像素优化独立复算，仅作可实现性诊断。
