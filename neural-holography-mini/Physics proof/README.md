# Physics proof

当前物理验证的完整推导与论文式展示。所有新增文件均在本文件夹，不修改主目录网络或训练代码，也不启动训练。

优先打开：

- [结果展示及中文说明（PDF）](results.pdf) / [离线 HTML](results.html)
- [五张高分辨率结果图（PDF）](results_figures.pdf)
- [完整物理证明与验证（PDF）](physics_proof.pdf) / [离线 HTML](physics_proof.html) / [Markdown](physics_proof.md)

图片单独保存于 figures，包含单平面迁移、双平面各深度与拼接、输入分区、局部放大、指标曲线。图中英文标签方便用于论文式排版，中文解释见 results。

data 保存本次引用的原始指标快照、CSV 和两个分区掩码。manifest.json 保存图像来源、SHA-256 与局部裁剪坐标。previews 为检查排版用的缩略图。

展示参考 D:/wy/文献/main.pdf 中 PHAROS 的 Fig. 6–8；这里只类比排版，未复现论文方法或进行同条件性能比较。当前数据为单图仿真，新增训练步数为零。

从 neural-holography-mini 目录重新生成：

~~~powershell
conda run --no-capture-output -n e2e python -B "Physics proof/build_presentation.py"
conda run --no-capture-output -n e2e python -B "Physics proof/render_documents.py"
~~~

第一条只读取已保存图像与指标，将 test 中物理报告与 derivation_appendix.md 合成完整证明。第二条将 Markdown 渲染为 PDF/HTML；当前使用 e2e 的 Matplotlib 和本机已配置的 ReportLab 依赖路径。
