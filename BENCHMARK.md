# OMNI-DC v1.0：NYU 随机权重纯推理测速

从仓库根目录执行，使用独立环境，不运行 Apex 训练入口，不需要 mmcv/mmseg/OpenCV，也不下载任何 checkpoint：

```bash
conda env create -f environment.yaml
conda activate OMNI-DC
mkdir -p data
ln -s ../../data/nyudepthv2_h5 data/nyudepthv2_h5
python -m pytest tests/test_inference.py -q
python scripts/benchmark_nyu.py --output outputs/benchmark_nyu_new_run
```

已有环境/链接时不要重复创建；输出目录必须尚不存在。移动仓库后只需调整数据软链接。`figures/`、`data_json/` 和已有实验保持不动。

## 发布配置和输入

- 采用 `src/testing_scripts/test_void_nyu.sh` 的 v1.0 NYU 配置：PVT `[3,4,6,3]` + ResNet34、1 次 GRU、3 个积分分辨率、uniform 梯度权重、输入置信度、log-depth、稀疏深度中位数 whitening、GRU median whitening、6 次/5 邻点 DySPN。
- `load_dav2=False`：不包含 v1.1 的 Depth Anything V2，也不把未运行的基础模型计入参数或速度。新增 `from_scratch` 仅供随机入口关闭 PVT/ResNet 预训练加载，原初始化/模型结构保持不变。
- 外部统一 NYU 228×304、500 点、seed=2023、ImageNet RGB 归一化，逐样本确定性稀疏采样。为满足三分辨率积分的 16 倍数约束，wrapper 内底部补 12 行零至 **240×304**，预测裁回原尺寸；补边计入前向时间/显存，MACs 使用实际内部尺寸。
- NYU 相机 K 经 0.5 缩放、left=8/top=6 中心裁剪调整，随机点 pattern=0；K/pattern 提前放 GPU。

## 保留完整求解和测量口径

- 原 `DepthGradOptimLayer`、完整 CG 迭代/早停、rtol=1e-5、maxiter=5000 均保留；不缓存解、不减少最大迭代、不移除收敛判断中的 CPU/GPU 同步。迭代数依赖输入及随机权重，不代表训练后 checkpoint 的延迟。
- RTX 4060 Ti、FP32、batch=1；TF32/autocast/compile/CUDA graphs 关闭，cuDNN benchmark=False、deterministic=False，CPU threads=1；`eval()` + `inference_mode()`。
- 索引 0、326、653，各预热 100 次、CUDA events 逐帧同步计时 1000 次。合并全部 3000 次计算平均/P95，FPS=1000/平均毫秒；同时记录同步主机墙钟时间。CG 内部的主机调度和收敛同步开销属于模型前向，保留在时间内。
- 输入预放 GPU，排除加载/H2D、真值、损失、指标、日志、可视化；峰值 allocated 显存另测单次前向，含模型/输入/临时张量。参数计数含冻结和未执行兼容模块，共享参数只计一次；纯 FP32 state_dict 文件大小含 buffers/序列化开销。

## MACs 覆盖范围（部分统计）

自定义 autograd CG 层不能当作完整可追踪神经网络 MACs。单独用 hooks 统计卷积（含转置）、Linear、PVT attention matmul；**不包含 CG/有限差分/归约、DySPN grid_sample、归一化和逐元素操作**。结果是神经网络部分 MACs，不是完整模型计算量。记录 profile 样本的 CG 迭代数，profile hooks 在正式计时前移除。

每次保存 JSON、逐次延迟/GPU 状态、配置、随机权重、源码哈希/完整快照、Git 差异、独立环境清单。三个真实样本与完整原前向须 allclose(rtol=1e-5, atol=1e-6)，随机权重不报告 RMSE。短验证可加 `--warmup 2 --iterations 5 --groups 1`。
