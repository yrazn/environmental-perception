# 环境感知 Python 实现代码说明文档

## 1. 文档目的

本文档说明 `rescue_perception_py` 的当前实现状态、模块职责、算法映射、运行方式和验证结果，并给出后续开发计划。该实现以 `floofy-dazzling-cosmos.md` 中的《环境感知层算法框架完整实现方案》为设计依据。

当前代码定位为：

- 可独立运行的纯 Python + NumPy 实现
- 不依赖 ROS 2，可先完成算法验证和仿真测试
- 通过后端接口接入真实模型推理
- 通过惰性导入方式支持后续 ROS 2 节点包装

## 2. 与方案文档的对应关系

| 方案章节 | Python 实现 |
| --- | --- |
| 二、自定义消息定义 | `rescue_perception/types.py` |
| 3.1 传感器时间同步 | `rescue_perception/io/sensor_sync.py` |
| 3.2 RGB 检测 | `rescue_perception/detectors/rgb_detector.py` |
| 3.3 热成像检测 | `rescue_perception/detectors/thermal_detector.py` |
| 3.4 LiDAR 聚类 | `rescue_perception/detectors/lidar_cluster.py` |
| 3.5 4D 毫米波跟踪 | `rescue_perception/detectors/radar_tracker.py` |
| 3.6 烟雾估计 | `rescue_perception/detectors/smoke_estimator.py` |
| 3.7 气体风险 | `rescue_perception/detectors/gas_risk.py` |
| 3.8 裂缝检测 | `rescue_perception/detectors/crack_segmenter.py` |
| 3.9 积水检测 | `rescue_perception/detectors/water_detector.py` |
| 3.10 设施异常 | `rescue_perception/detectors/facility_inspector.py` |
| 4.2 人员融合 | `rescue_perception/fusion/person_fusion.py` |
| 4.3 火源融合 | `rescue_perception/fusion/fire_fusion.py` |
| 4.4 车辆融合 | `rescue_perception/fusion/vehicle_fusion.py` |
| 4.5 统一跟踪 | `rescue_perception/fusion/semantic_tracker.py` |
| 4.6 语义定位 | `rescue_perception/fusion/semantic_localizer.py` |
| 5.1 质量监控 | `rescue_perception/management/quality_monitor.py` |
| 5.2 退化管理 | `rescue_perception/management/degradation_manager.py` |
| 5.3 风险评估 | `rescue_perception/management/risk_assessor.py` |
| 11 核心算法 | 各 detectors/fusion/management 模块 |
| 12 端到端流程 | `rescue_perception/perception_pipeline.py` |
| 15 C++ 系统实现指南 | Python 版类结构和管线阶段 |

## 3. 总体架构

实现遵循方案中的“后融合”和“分层冗余感知”原则：

```text
传感器层
  RGB / Thermal / LiDAR / Radar / Gas / PM
        |
        v
感知管线 PerceptionPipeline
  阶段0 传感器健康检查
  阶段1 各传感器独立检测
  阶段2 目标级融合（人员/火源/车辆）
  阶段3 统一多目标跟踪
  阶段4 地图坐标转换
  阶段5 场景级风险评估
  阶段6 安全策略映射
  阶段7 输出
```

核心设计：

- 后融合：各传感器先独立检测，再统一关联融合
- 动态权重：按退化模式和传感器可信度实时调整
- 四级退化：`CLEAR -> LOW_VISIBILITY -> HEAVY_SMOKE -> PERCEPTION_DEGRADED`
- 安全优先：极端情况下输出候选目标、大协方差和保守安全建议

## 4. 目录结构

```text
rescue_perception_py/
├── CODE_DOCUMENTATION.md
├── README.md
├── pyproject.toml
├── config/
│   ├── perception_master.yaml
│   └── degradation.yaml
├── rescue_perception/
│   ├── __init__.py
│   ├── config.py
│   ├── types.py
│   ├── perception_pipeline.py
│   ├── detectors/
│   │   ├── rgb_detector.py
│   │   ├── thermal_detector.py
│   │   ├── lidar_cluster.py
│   │   ├── radar_tracker.py
│   │   ├── smoke_estimator.py
│   │   ├── gas_risk.py
│   │   ├── crack_segmenter.py
│   │   ├── water_detector.py
│   │   └── facility_inspector.py
│   ├── fusion/
│   │   ├── association.py
│   │   ├── person_fusion.py
│   │   ├── fire_fusion.py
│   │   ├── vehicle_fusion.py
│   │   ├── semantic_tracker.py
│   │   └── semantic_localizer.py
│   ├── management/
│   │   ├── quality_monitor.py
│   │   ├── degradation_manager.py
│   │   └── risk_assessor.py
│   ├── io/
│   │   ├── sensor_sync.py
│   │   └── ros2_bridge.py
│   └── sim/
│       ├── scenario.py
│       └── backends.py
├── web/
│   ├── app.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
├── scripts/
│   ├── run_demo.py
│   └── run_visualizer.py
└── tests/
    ├── test_algorithms.py
    ├── test_pipeline.py
    └── test_web_api.py
```

## 5. 核心数据模型

`types.py` 定义了以下核心类型：

- `SyncedSensorData`：一帧同步后的多传感器数据
- `Detection3D`：传感器级三维检测候选
- `TrackedObject`：统一跟踪目标
- `EnvironmentQuality`：环境质量与传感器可信度
- `SensorHealth`：单个传感器健康状态
- `SensorWeights`：六类传感器权重
- `SmokeEstimateResult`：综合烟雾指数
- `GasReadings`、`GasRiskResult`：气体读数与风险
- `ThermalHotspot`：热成像热点
- `RiskAssessment`、`SafetyDecision`：场景风险和安全建议
- `PipelineOutput`：主管线最终输出

目标类别枚举与方案一致：

```text
1 火焰
2 烟雾
3-7 人员（站立/行走/蹲伏/倒地/遮挡）
8-12 车辆
13 裂缝
14-15 积水/渗水
16 设施异常
17 高温禁区
18 通用障碍物
19 疑似人员
20 疑似火源
21 热异常点
22 未知高温物体
```

## 6. 主管线流程

`PerceptionPipeline.spin_once()` 实现方案中的主感知管线：

```text
阶段0 传感器健康检查
  质量监控评估 RGB/Thermal/LiDAR/Radar/Gas 可信度
  退化状态机更新当前模式
  生成动态传感器权重

阶段1 各传感器独立检测
  RGB 检测
  热成像热点检测与热风险
  LiDAR 地面分割与聚类
  4D 毫米波多目标跟踪
  烟雾指数估计
  气体风险估计

阶段2 目标级融合
  人员融合
  火源融合
  车辆融合

阶段3 统一跟踪
  类 DeepSORT 语义跟踪
  卡尔曼预测/更新
  试探/确认/丢失生命周期

阶段4 语义定位
  传感器坐标 -> map 坐标
  协方差传播

阶段5 场景风险
  热风险、气体、障碍、占道、深度、RGB、运动加权融合

阶段6 安全映射
  限速、停车、远程确认、急停

阶段7 输出
  PipelineOutput 汇总所有结果
```

## 7. 模块说明

### 7.1 detectors 检测层

#### rgb_detector.py

- 提供 `DetectionBackend` 接口，隔离 YOLO/ONNX 推理
- 实现火焰时序验证、类别阈值过滤
- 实现烟雾掩膜统计：覆盖率、烟雾概率、质心

#### yolo_backend.py

- 使用 Ultralytics 官方 `yolo11n.pt` COCO 权重完成基准推理
- 将 `xyxy`、置信度和模型类别映射为统一 `Detection3D`
- 当前映射 person、car、truck、bus、motorcycle，忽略无关 COCO 类别
- 在无 RGB-D/LiDAR 投影时使用目标框高度粗估距离，并显式设置
  `depth_valid=False`
- Ultralytics 为可选依赖，合成演示和测试不会强制加载模型

#### thermal_detector.py

- 坏点修复：3x3 中值滤波，仅替换异常跳变像素
- 背景基线：最近 30 帧逐像素最小值，取中心 50% 区域统计
- 自适应阈值分割：`T_bg + max(30, 3 * std)`
- 8 连通域分析、周长、圆形度、边缘梯度
- 时序匹配：温升速率 `dT/dt`、面积增长率 `dA/dt`、持续时间
- 热点分类：明火、疑似火源、热设备、温暖表面、一般热异常、人体热源
- 热风险等级：0-4 级

#### lidar_cluster.py

- 距离直通滤波
- 按退化模式调整体素下采样
- RANSAC 地面平面拟合
- 空间哈希欧氏聚类
- PCA 三维包围盒拟合
- 尺寸规则分类
- 点云质量指标：有效点比例、近场散射、最大回波距离、强度衰减

#### radar_tracker.py

- 静态杂波过滤
- 静止大目标持续存在判定
- 6 状态卡尔曼滤波
- 贪心数据关联
- 基于 RCS、速度、微多普勒的目标分类

#### smoke_estimator.py

- 四因子烟雾指数：图像、PM、LiDAR 衰减、CO
- 基准权重：`0.30 / 0.25 / 0.30 / 0.15`
- 传感器退化动态降权
- 重烟模式额外调整
- EMA 平滑
- 可见度等级离散化

#### gas_risk.py

- CO、CO2、O2、CH4、H2S 独立分级
- 温湿度修正
- 综合风险取最大值
- 火灾签名、爆炸风险、窒息风险检测
- 告警输出决策

### 7.2 fusion 融合层

#### association.py

- 匈牙利最小代价匹配
- 代价门限过滤
- 2D IoU 工具

#### person_fusion.py

- RGB、热成像、LiDAR、毫米波人员候选收集
- 动态权重置信度融合
- 温度合理性、人体尺寸、雷达微动验证
- 多源确认判定
- 重烟模式候选升格

#### fire_fusion.py

- 火源确认决策矩阵，输出 0-4 级
- RGB、热成像、气体、爆炸风险证据组合
- 高温禁区半径计算

#### vehicle_fusion.py

- RGB、LiDAR、毫米波车辆关联
- 事故状态保留
- 占道比例与阻断评估

#### semantic_tracker.py

- 统一目标跟踪池
- 类别分组匹配
- 8 状态卡尔曼
- 试探/确认/丢失生命周期
- 近距离同类别目标合并

#### semantic_localizer.py

- 4x4 变换矩阵注册
- 传感器坐标到地图坐标转换
- 协方差旋转传播
- Marker 输出

### 7.3 management 管理层

#### quality_monitor.py

- RGB 清晰度、亮度、对比度评估
- 热成像有效像素率、均匀性、窗口污染
- LiDAR 点云质量
- 毫米波目标稳定性与 RCS 一致性
- 气体湿度交叉干扰修正
- 综合可见度等级

#### degradation_manager.py

- 四级退化状态机
- 带滞后退出条件
- 3 秒振荡保护
- 动态权重分配
- 行为建议生成

#### risk_assessor.py

- 环境风险多因子加权
- 最大风险原则
- 高温禁区生成
- 事件生成
- 安全策略映射

### 7.4 io 输入输出层

#### sensor_sync.py

- 多 Topic 时间戳缓冲
- 近似时间同步
- 同步误差统计

#### ros2_bridge.py

- 惰性导入 `rclpy`
- 最小 ROS 2 节点包装

### 7.5 sim 仿真层

#### scenario.py

- 合成隧道火灾场景
- RGB、热成像、LiDAR、雷达、气体、PM、温湿度同步生成
- 可配置烟雾曲线

#### backends.py

- `SyntheticRgbBackend`
- `SyntheticCrackBackend`
- `SyntheticWaterBackend`
- `SyntheticFacilityBackend`

### 7.6 web 可视化层

#### app.py

- 基于 Python 标准库实现 HTTP + SSE 服务
- `/` 返回单页仪表盘
- `/api/state` 返回当前帧 JSON
- `/events` 推送实时状态
- `/api/control` 支持暂停、继续、单步、重置、倍速

#### static/

- `index.html`：单页仪表盘布局、任务/感知模式、连接状态和内容安全策略
- `app.js`：SSE 数据接收、风险着色、传感器健康、Canvas 隧道俯视图、趋势曲线、巡检详情和事件去重
- `style.css`：高对比风险操作台、响应式布局和状态动画

可视化控制会正确暂停仿真、推进单帧，并在重置时重建所有有状态算法，
避免保留旧轨迹、烟雾平滑值和退化状态。

## 8. 配置系统

当前实现内置默认配置，并支持从 YAML 加载：

- `config/perception_master.yaml`：系统、传感器、检测器、模式配置
- `config/degradation.yaml`：退化转移、权重、行为、深度优先级

`PerceptionConfig.from_yaml()` 支持主配置、退化配置和运行模式覆盖。
`rescue`、`inspection`、`minimal` 模式会实际控制人员与巡检任务执行，
传感器开关会同步控制检测器和动态权重。

## 9. 运行方式

环境要求：

- Python >= 3.8
- NumPy >= 1.20

安装：

```bash
cd /home/ubuntu/桌面/环境感知/rescue_perception_py
pip install -e .
```

运行演示：

```bash
python3 scripts/run_demo.py --mode rescue
python3 scripts/run_demo.py --mode inspection
```

运行可视化仪表盘：

```bash
python3 scripts/run_visualizer.py --port 9100 --mode rescue
```

浏览器打开：

```text
http://127.0.0.1:9100
```

运行测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

语法检查：

```bash
python3 -m compileall -q rescue_perception scripts tests
```

## 10. 当前验证结果

已完成验证：

- 21 项单元测试全部通过
- 端到端合成场景演示可运行
- 所有 Python 文件通过编译检查
- Web 服务、静态页面、SSE 实时流可访问
- YAML 配置和运行模式可生效
- `SensorSync` 可直接驱动主管线
- inspection 模式可输出裂缝、积水和设施异常
- 独立 PM 输入、非单位坐标变换和类别边界已有回归测试

演示场景覆盖：

- 烟雾从正常逐步上升到重烟
- 退化模式从 `LOW_VISIBILITY` 进入 `HEAVY_SMOKE`
- RGB 权重下降，热成像和毫米波保持较高权重
- 输出人员候选、火源、车辆目标
- 输出气体危险告警和高温禁区
- 输出安全策略和限速建议

## 11. 已完成与未完成

### 已完成

- 完整 Python 包结构
- 核心消息和数据类型
- 全部检测、融合、管理算法模块
- 主管线调度
- YAML 配置加载和运行模式开关
- `SensorSync` 主管线入口与同步误差记录
- 巡检模块主管线、公共输出和 Web API 接入
- 修复独立 PM 输入、坐标变换和车辆误判为人员问题
- 合成场景和演示脚本
- 单元测试
- 零依赖 Web 可视化仪表盘
- README、配置示例、文档

### 未完成

- 真实模型推理后端未接入
- 真实传感器数据和 rosbag 未实测
- ROS 2 话题通信未完整验证
- 相机内参、外参、TF 坐标链未接入
- 巡检真实模型后端未实现（当前仅有接口和合成后端）
- Jetson 部署和延迟优化未做
- 可视化仪表盘尚未接入真实模型和真实传感器数据

## 12. 后续开发计划

### 阶段 1：真实数据接口

优先级：P0

任务：

- 接入真实传感器数据或 rosbag 回放
- 完善 ROS 2 订阅和发布

验收：

- 真实数据连续运行 10 分钟不中断
- 同步误差可统计
- 模式、权重、目标输出正常

### 阶段 2：模型推理后端

优先级：P0

任务：

- 将已实现的 Ultralytics YOLO 基准后端导出并迁移到 ONNX Runtime
- 实现热成像人员检测后端
- 实现预处理、NMS、类别映射
- 与现有 `Detection3D` 对齐

验收：

- 火焰、人员、车辆模型可运行
- 单帧推理延迟满足目标
- 正常场景人员召回率 >= 90%

### 阶段 3：三维定位与深度有效性

优先级：P1

任务：

- 接入相机内参
- 接入 LiDAR-相机外参
- 接入 TF 坐标链
- 实现深度源优先级选择
- 完善协方差传播

验收：

- 目标在 map 坐标系下位置可量化
- 浓烟下输出方向候选和 `depth_valid=False`

### 阶段 4：融合与退化调优

优先级：P1

任务：

- 用真实数据标定人员关联阈值
- 验证火源决策矩阵
- 验证退化状态机切换
- 增加传感器失效注入测试

验收：

- 退化切换延迟 <= 1s
- 浓烟下安全策略正确降速或停车

### 阶段 5：巡检模式与任务联动

优先级：P2

任务：

- 接入裂缝、积水、设施模型
- 接入巡检模式输出
- 与任务状态机、Nav2、Safety Supervisor 对接

验收：

- 完成巡检到火情、救援、撤退端到端 rosbag 测试

### 阶段 6：性能与部署

优先级：P2

任务：

- Jetson AGX Orin TensorRT FP16 固化
- 管线延迟测量
- GPU 利用率分析
- 零拷贝通信优化

验收：

- 端到端延迟 <= 100ms @10Hz
- 可在 Jetson 上运行

## 13. 已知限制

- RGB 已接入 Ultralytics YOLO 官方 COCO 基准模型；目前只覆盖人员和常见
  车辆，单目距离仍是近似值，火焰/烟雾需要专用模型
- 热成像人员检测使用温度/尺寸启发式，未接入分类模型
- LiDAR 聚类使用纯 NumPy，点云规模增大后需要优化
- 雷达跟踪使用简化 2D 坐标，未接入真实雷达点云
- 裂缝、积水、设施检测只有接口骨架
- ROS 2 桥接未经过真实 ROS 2 环境测试
- 合成场景数据不能替代真实标定数据
