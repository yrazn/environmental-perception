# 环境感知层算法框架完整实现方案

为隧道火灾救援机器人设计完整的环境感知层算法框架。该系统必须在火灾、车祸、浓烟、低照度等极端条件下可靠运行，不能按普通巡检机器人"相机+YOLO"来设计，而应按分层冗余感知系统来设计。

核心原则：
- 后融合策略：各传感器独立检测 → 时空关联 → 统一融合
- 动态权重：传感器权重根据烟雾等级和传感器健康度实时切换
- 退化模式：CLEAR → LOW_VISIBILITY → HEAVY_SMOKE → PERCEPTION_DEGRADED 四级递进
- 五类输出：目标类别、目标位置、目标属性、环境风险等级、传感器可信度

---

## 一、项目结构总览

```
rescue_perception/
├── rescue_perception_msgs/           # 自定义消息包
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── msg/
│   │   ├── SemanticObject.msg        # 统一目标消息
│   │   ├── SemanticObjectArray.msg   # 目标数组
│   │   ├── SensorHealth.msg          # 传感器健康状态
│   │   ├── EnvironmentQuality.msg    # 环境质量评估
│   │   ├── SmokeEstimate.msg         # 烟雾估计
│   │   ├── GasRisk.msg               # 气体风险
│   │   ├── HotZone.msg               # 高温禁区
│   │   ├── CrackDefect.msg           # 裂缝缺陷
│   │   ├── WaterRegion.msg           # 积水区域
│   │   └── FacilityAnomaly.msg       # 设施异常
│   └── srv/
│       ├── SetPerceptionMode.srv     # 设置感知模式
│       └── GetPerceptionStatus.srv   # 获取感知状态
│
├── rescue_perception_core/           # 核心感知节点包
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── include/rescue_perception_core/
│   │   ├── sensor_sync.hpp
│   │   ├── rgb_detector.hpp
│   │   ├── thermal_detector.hpp
│   │   ├── lidar_cluster.hpp
│   │   ├── radar_tracker.hpp
│   │   ├── smoke_estimator.hpp
│   │   ├── gas_risk.hpp
│   │   ├── crack_segmenter.hpp
│   │   ├── water_detector.hpp
│   │   ├── facility_inspector.hpp
│   │   ├── person_fusion.hpp
│   │   ├── fire_fusion.hpp
│   │   ├── vehicle_fusion.hpp
│   │   ├── semantic_tracker.hpp
│   │   ├── semantic_localizer.hpp
│   │   ├── risk_assessor.hpp
│   │   ├── quality_monitor.hpp
│   │   └── degradation_manager.hpp
│   └── src/
│       ├── sensor_sync.cpp
│       ├── rgb_detector.cpp
│       ├── thermal_detector.cpp
│       ├── lidar_cluster.cpp
│       ├── radar_tracker.cpp
│       ├── smoke_estimator.cpp
│       ├── gas_risk.cpp
│       ├── crack_segmenter.cpp
│       ├── water_detector.cpp
│       ├── facility_inspector.cpp
│       ├── person_fusion.cpp
│       ├── fire_fusion.cpp
│       ├── vehicle_fusion.cpp
│       ├── semantic_tracker.cpp
│       ├── semantic_localizer.cpp
│       ├── risk_assessor.cpp
│       ├── quality_monitor.cpp
│       └── degradation_manager.cpp
│
├── rescue_perception_bringup/        # 启动与配置
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── launch/
│   │   ├── perception_bringup.launch.py    # 完整启动
│   │   ├── perception_inspection.launch.py  # 巡检模式
│   │   ├── perception_rescue.launch.py      # 救援模式
│   │   ├── perception_minimal.launch.py     # 退化最小模式
│   │   └── perception_rviz.launch.py        # 可视化
│   ├── config/
│   │   ├── perception_master.yaml           # 全局参数
│   │   ├── sensors.yaml                     # 传感器参数
│   │   ├── detectors.yaml                   # 检测器参数
│   │   ├── fusion.yaml                      # 融合参数
│   │   ├── degradation.yaml                 # 退化策略
│   │   └── risk.yaml                        # 风险评估参数
│   └── rviz/
│       └── perception.rviz
│
├── rescue_perception_models/         # 模型与工具
│   ├── models/
│   │   ├── rgb_flame.onnx
│   │   ├── rgb_vehicle.onnx
│   │   ├── rgb_person.onnx
│   │   ├── rgb_smoke_seg.onnx
│   │   ├── thermal_person.onnx
│   │   ├── thermal_fire.onnx
│   │   ├── crack_seg.onnx
│   │   ├── water_seg.onnx
│   │   ├── vehicle_state_cls.onnx
│   │   └── person_pose_cls.onnx
│   └── scripts/
│       ├── calibrate_sensors.py
│       ├── validate_pipeline.py
│       └── benchmark_pipeline.py
│
└── README.md
```

---

## 二、自定义消息定义

### 2.1 SemanticObject.msg（核心统一目标消息）

```
std_msgs/Header header

# 目标标识
uint32 track_id                  # 全局唯一跟踪ID
uint8 class_id                   # 语义类别ID（见枚举）
string class_name                # 语义类别名称

# 空间位置
geometry_msgs/PoseWithCovariance pose    # 地图坐标系下的位姿+协方差
geometry_msgs/Vector3 dimensions         # 目标长宽高(m)

# 运动状态
geometry_msgs/Twist velocity             # 速度

# 目标属性
float32 temperature_max          # 最高温度(°C)
float32 temperature_min          # 最低温度(°C)
float32 confidence               # 综合置信度 [0,1]

# 风险
uint8 risk_level                 # 风险等级 0=正常 1=关注 2=警告 3=危险 4=致命

# 来源信息
uint8 source_mask                # 位掩码: RGB=1 THERMAL=2 LIDAR=4 RADAR=8 GAS=16
uint8 visibility_level           # 检测时的环境可见度等级

# 确认状态
bool confirmed                   # 是否经多传感器确认

# 类别ID枚举
# 1=flame, 2=smoke, 3=person_standing, 4=person_walking, 5=person_crouching,
# 6=person_lying, 7=person_occluded, 8=vehicle_car, 9=vehicle_truck,
# 10=vehicle_bus, 11=vehicle_motorcycle, 12=vehicle_unknown,
# 13=crack, 14=water_puddle, 15=water_seepage,
# 16=equipment_damaged, 17=hot_zone, 18=obstacle_generic
```

### 2.2 EnvironmentQuality.msg

```
std_msgs/Header header

# 可见度 (0-4)
uint8 visibility_level           # 0=CLEAR 1=LOW_LIGHT 2=LIGHT_SMOKE 3=HEAVY_SMOKE 4=BLIND
float32 visibility_score         # 连续可见度分数 [0,1]

# 烟雾
float32 smoke_density            # 综合烟雾密度 [0,1]
float32 smoke_coverage           # 烟雾覆盖比例 [0,1]

# 各传感器可信度 [0,1]
float32 rgb_credibility
float32 thermal_credibility
float32 lidar_credibility
float32 radar_credibility
float32 gas_credibility

# 环境条件
float32 ambient_temperature      # 环境温度(°C)
float32 ambient_humidity         # 环境湿度(%)
float32 illuminance_lux          # 环境照度(lux)
```

### 2.3 SensorHealth.msg

```
std_msgs/Header header
string sensor_name               # 传感器名称
float32 data_quality             # 数据质量 [0,1]
float32 effective_ratio          # 有效数据比例
float32 noise_level              # 噪声水平
bool window_contaminated         # 窗口是否污染
float32 internal_temperature     # 内部温度(°C)
float32 sync_error_ms            # 时间同步误差(ms)
float32 last_update_age          # 最后更新距今(ms)
uint8 status                     # 0=OK 1=DEGRADED 2=FAILED 3=TIMEOUT
```

### 2.4 其他辅助消息

**SmokeEstimate.msg**
```
std_msgs/Header header
float32 smoke_score              # 综合烟雾指数 [0,1]
float32 image_smoke_prob         # 图像烟雾概率
float32 pm_rise_rate             # PM变化率
float32 lidar_attenuation        # LiDAR衰减指数
float32 co_rise_rate             # CO变化率
sensor_msgs/RegionOfInterest smoke_mask_roi
float32 coverage_ratio
float32[] motion_direction       # 烟雾运动方向向量
```

**GasRisk.msg**
```
std_msgs/Header header
float32 co_ppm
float32 co2_percent
float32 o2_percent
float32 ch4_percent
float32 h2s_ppm
float32 voc_ppm
uint8 risk_level                 # 0=SAFE 1=CAUTION 2=DANGER 3=DEADLY
bool fire_risk
bool asphyxiation_risk
bool explosion_risk
```

**HotZone.msg**
```
std_msgs/Header header
geometry_msgs/Polygon polygon    # 高温区域多边形
float32 max_temperature
float32 avg_temperature
float32 area_m2
uint8 severity                   # 0=WARM 1=HOT 2=CRITICAL
```

---

## 三、节点详细设计

### 3.1 sensor_sync_node —— 传感器时间同步节点

**职责**：接收所有原始传感器数据，进行时间对齐、去畸变、运动补偿，发布同步后的数据。

**输入Topics**：
```
/camera/front/image_raw          (sensor_msgs/Image)
/camera/ptz/image_raw            (sensor_msgs/Image)
/thermal/image_raw               (sensor_msgs/Image)
/thermal/temperature_array       (自定义Float32MultiArray)
/lidar/points_raw                (sensor_msgs/PointCloud2)
/radar/tracks                    (自定义RadarTrackArray)
/scan/front                      (sensor_msgs/LaserScan)
/scan/rear                       (sensor_msgs/LaserScan)
/range/front_left                (sensor_msgs/Range)
/range/front_right               (sensor_msgs/Range)
/range/rear_left                 (sensor_msgs/Range)
/range/rear_right                (sensor_msgs/Range)
/gas/status                      (GasSensorArray)
/pm/status                       (PMSensorArray)
/temperature/humidity            (sensor_msgs/Temperature+sensor_msgs/RelativeHumidity)
/tf                              (tf2_msgs/TFMessage)
/tf_static                       (tf2_msgs/TFMessage)
/odometry/filtered               (nav_msgs/Odometry)
```

**输出Topics**：
```
/perception/synced/rgb_image
/perception/synced/thermal_image
/perception/synced/thermal_temp
/perception/synced/lidar_points
/perception/synced/radar_tracks
/perception/synced/scans
/perception/synced/environment    (整合的气体/PM/温湿度)
/perception/synced/odometry
/perception/sync_status           (同步质量报告)
```

**内部算法**：
```
1. 时间戳对齐：
   - 使用 message_filters::Synchronizer 进行近似时间同步
   - 主同步源：LiDAR 或相机（取较高频率者）
   - 策略：ApproximateTime，容忍窗口 10ms（关键数据）到 50ms（环境数据）

2. 点云运动补偿：
   - 使用 odometry + IMU 对 LiDAR 点云进行去畸变
   - 基于分段线性插值的位姿补偿

3. 时间同步质量监控：
   - 计算各传感器间时间戳差异
   - 发布同步误差到 /perception/sync_status
   - 超过阈值（100ms）标记为 DEGRADED

4. 传感器窗口污染检测：
   - 图像梯度分析（低梯度可能表示窗口模糊）
   - LiDAR 近场回波异常检测
   - 热成像均匀性检查
```

**参数**：
```yaml
sensor_sync:
  sync_tolerance_ms: 10           # 主同步容忍窗口
  env_sync_tolerance_ms: 50       # 环境传感器容忍窗口
  motion_compensation: true       # 是否进行运动补偿
  publish_diagnostics: true       # 是否发布诊断
  contamination_check_interval: 5.0  # 污染检查间隔(秒)
```

---

### 3.2 rgb_detector_node —— 可见光检测节点

**职责**：对可见光图像执行火焰、人员、车辆、烟雾的检测与分割。

**输入Topics**：
```
/perception/synced/rgb_image      (sensor_msgs/Image)
/perception/synced/odometry       (nav_msgs/Odometry)
```

**输出Topics**：
```
/perception/rgb/detections        (SemanticObjectArray)
/perception/rgb/flame_mask        (sensor_msgs/Image)
/perception/rgb/smoke_mask        (sensor_msgs/Image)
```

**内部算法管道**：

```
// ====== 火焰检测管道 ======
FlamePipeline:
  Input:  RGB image (1080P)
  1. 预处理: resize→640x640, normalize([0,1], mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  2. 检测: YOLO11-nano-flame (ONNX) → 火焰候选框 + 置信度
  3. 分割: YOLO-Seg-Flame → 火焰区域掩膜
  4. 时序验证: 维护5帧火焰缓冲
     - 计算连续帧间火焰区域IoU
     - 计算边缘运动不规则性 (光流方差)
     - 计算亮度时序波动 (帧间强度变化)
     - ConvLSTM 二分类: 真实火焰 vs 误检
  5. 输出: flame_bbox + flame_mask + flame_confidence

// ====== 车辆检测管道 ======
VehiclePipeline:
  Input:  RGB image
  1. 检测: YOLO11-m-vehicle (ONNX) → 车辆候选框
     Classes: car, truck, bus, motorcycle, unknown_vehicle
  2. 事故状态分类: 对每个检测框:
     - 裁剪 ROI → ResNet34-VehicleState (ONNX)
     - 分类: normal/collision_damaged/side_overturned/
             fully_overturned/cross_blocking/burning/severe_deformation
  3. 输出: vehicle_bbox + vehicle_class + accident_state + confidence

// ====== 人员检测管道 ======
PersonPipeline:
  Input:  RGB image
  1. 检测: YOLO11-s-person (ONNX) → 人员候选框
     Classes: person_generic
  2. 姿态分类: ResNet18-Pose (ONNX)
     Classes: standing, walking, crouching, lying, sitting
  3. 仅在可见光置信度≥0.5时输出（低照度自动降权）
  4. 输出: person_bbox + person_pose + confidence

// ====== 烟雾分割管道 ======
SmokeSegPipeline:
  Input:  RGB image
  1. 分割: SegFormer-B0-smoke (ONNX) → 烟雾掩膜
  2. 时序分析: 3帧差分 → 烟雾扩散方向
  3. 统计: 覆盖率 = 烟雾像素/总像素
  4. 输出: smoke_mask + smoke_coverage + smoke_direction
```

**参数**：
```yaml
rgb_detector:
  flame:
    model: "rgb_flame.onnx"
    conf_threshold: 0.35
    nms_threshold: 0.45
    temporal_window: 5           # 时序验证帧数
    flicker_threshold: 0.15      # 闪烁阈值
  vehicle:
    model: "rgb_vehicle.onnx"
    conf_threshold: 0.45
    state_classifier: "vehicle_state_cls.onnx"
  person:
    model: "rgb_person.onnx"
    conf_threshold: 0.4
    pose_classifier: "person_pose_cls.onnx"
    min_confidence_output: 0.5   # 低于此值不在融合中使用
  smoke:
    model: "rgb_smoke_seg.onnx"
    coverage_update_rate: 5.0    # 覆盖率更新频率(Hz)
  inference:
    device: "cuda"
    precision: "fp16"
    batch_size: 1
```

---

### 3.3 thermal_detector_node —— 热成像检测节点

**职责**：检测热成像中的人员、高温区域、火源，输出辐射测温结果。

**输入Topics**：
```
/perception/synced/thermal_image     (sensor_msgs/Image)
/perception/synced/thermal_temp      (Float32MultiArray, 640×512)
```

**输出Topics**：
```
/perception/thermal/detections       (SemanticObjectArray)
/perception/thermal/hot_zones        (HotZoneArray)
/perception/thermal/temperature_map  (sensor_msgs/Image, 伪彩色)
```

**内部算法管道**：

```
// ====== 热成像预处理 ======
ThermalPreprocess:
  Input:  temperature_matrix[640][512] (14-bit radiometric)
  1. 坏点修复: 3×3 中值滤波 (仅对异常跳变像素)
  2. 非均匀性校正: 两点校正法 (gain + offset)
  3. 环境温度自适应基线: rolling_median(frame[-30:]) 作为背景基线
  4. 温度归一化: (T - T_bg) / (T_bg * emissivity_factor)
  5. 生成三通道伪彩色图: 用于分类网络的输入
     R通道: T > T_bg+50°C → 高温区域增强
     G通道: |T - T_bg| < 10°C → 背景区域
     B通道: T < T_bg-5°C → 低温区域

// ====== 人员检测管道 ======
ThermalPersonPipeline:
  Input:  thermal pseudo-color image + temperature_matrix
  1. 检测: YOLO11-m-thermal-person (ONNX) → 人体候选框
     Classes: person_standing, person_walking, person_crouching,
              person_lying, person_occluded, person_unknown
  2. 实例分割: 对每个检测框 → 热斑精确轮廓
  3. 姿态确认:
     - 计算目标长宽比: h/w > 2.0 → 站立, h/w < 1.2 → 倒地
     - 计算目标高度(基于像素+估计距离): <0.8m → 倒地候选
     - 温度分布: 人体温度范围检查 (28°C~42°C 皮肤温度)
     - 尺寸合理性: 0.3m²~2.0m² 投影面积
  4. 人体真实性判定:
     C_person = 0.0
     if 28 < T_mean < 42:               C_person += 0.3
     if 形态匹配(h/w合理):               C_person += 0.2
     if 连续存在>3帧:                    C_person += 0.2
     if 尺寸合理(投影面0.3~2.0m²):        C_person += 0.15
     if 毫米波/点云位置匹配:             C_person += 0.15
  5. 输出: person bbox + class + confidence + temperature_range

// ====== 高温区域/火源检测 ======
FireHotspotPipeline:
  Input:  temperature_matrix
  1. 高温区域分割:
     - 自适应阈值: T > T_bg + max(30, 3*std_bg)
     - 区域生长: 8连通域，最小面积 9 pixels
  2. 特征提取(每个连通域):
     - T_max, T_mean, T_min
     - 面积 A (投影面积)
     - 周长 P
     - 圆形度: 4πA/P² (火焰: 低圆形度; 热设备: 高圆形度)
     - 温度梯度: max_gradient 在边缘处
  3. 时序分析(5帧滑窗):
     - dT/dt: 温升速度 (火灾: dT/dt > 5°C/s)
     - dA/dt: 面积增长率 (火灾: dA/dt > 10%/s)
     - 持续时间: 稳定高温 > 3s → 抑制瞬时热源误报
  4. 火源分类:
     if T_max > 200°C AND dT/dt > 5°C/s AND dA/dt > 10% AND 圆形度 < 0.7:
       → fire_source (置信度 0.85+)
     elif T_max > 150°C AND dT/dt > 3°C/s AND 持续时间 > 3s:
       → fire_suspected (置信度 0.6+)
     elif T_max > 80°C AND 圆形度 > 0.8 AND dT/dt < 1°C/s:
       → hot_equipment (热设备/发动机)
     else:
       → hot_anomaly (一般热异常)
  5. 辐射测温补偿:
     - 距离衰减补偿: T_corrected = T_measured / (ε * τ_atm(d))
     - 发射率: ε_human=0.98, ε_metal=0.3~0.6, ε_fire=0.95
     - 大气透过率: τ_atm(d) = exp(-α * d), α from humidity + smoke
  6. 输出: hotspot_type + confidence + temperature_corrected + area
```

**参数**：
```yaml
thermal_detector:
  person:
    model: "thermal_person.onnx"
    conf_threshold: 0.3
    min_temperature: 24.0         # 人体最低温度(含低温环境)
    max_temperature: 44.0         # 人体最高温度(含发热)
    min_height_m: 0.3             # 倒地人员最低高度
    max_aspect_ratio_lying: 1.5   # 卧姿最大长宽比
    temporal_consistency: 3       # 需要连续出现的帧数
  fire:
    model: "thermal_fire.onnx"
    hotspot:
      min_area_pixels: 9
      bg_window_size: 30          # 背景估计窗口帧数
      temp_rise_threshold: 5.0    # 温升阈值(°C/s)
      area_growth_threshold: 0.1  # 面积增长率(/s)
      fire_temp_min: 200.0        # 明火最低温度
      suspected_temp_min: 150.0
      circularity_fire_max: 0.7
      circularity_equipment_min: 0.8
    emissivity:
      human: 0.98
      metal: 0.45
      fire: 0.95
      default: 0.90
```

---

### 3.4 lidar_cluster_node —— LiDAR 障碍物聚类节点

**职责**：对3D LiDAR点云进行地面分割、聚类、3D包围盒拟合；对2D LiDAR进行近地障碍检测。

**输入Topics**：
```
/perception/synced/lidar_points  (sensor_msgs/PointCloud2)
/perception/synced/scans         (sensor_msgs/LaserScan[])
```

**输出Topics**：
```
/perception/lidar/clusters       (SemanticObjectArray)
/perception/lidar/ground_cloud   (sensor_msgs/PointCloud2)  # 地面点云(用于裂缝/积水)
/perception/lidar/quality        (自定义/点云质量指标)
```

**内部算法管道**：

```
// ====== 3D LiDAR处理 ======
Lidar3DPipeline:
  Input:  PointCloud (已运动补偿)

  1. 直通滤波: 保留 [0.5m, 80m] 范围点

  2. 体素下采样: leaf_size=0.1m (正常), 0.2m (轻烟), 0.3m (重烟)
     - 根据环境质量自适应调整

  3. 地面分割: RANSAC 平面拟合
     - 最大迭代: 100
     - 距离阈值: 0.05m (平坦隧道), 0.1m (坡道)
     - 输出: ground_cloud, non_ground_cloud

  4. 非地面点聚类: Euclidean Cluster Extraction 或 DBSCAN
     - ClusterTolerance: 0.3m (正常), 0.5m (退化)
     - MinClusterSize: 20 points
     - MaxClusterSize: 50000 points

  5. 包围盒拟合: 对每个聚类
     - PCA 主成分分析 → 主轴方向
     - L-Shape fitting → 车辆轮廓
     - 输出: center(x,y,z), dimensions(l,w,h), yaw

  6. 尺寸规则分类:
     - l>2m, w>1.2m, h>1m → vehicle_candidate
     - h<0.3m, area<1m² → small_obstacle
     - h>0.5m, area<0.5m² → pole/person_candidate
     - 其他 → obstacle_generic

  7. 点云质量评估:
     有效点比例 = count(range>5m) / 标称点数
     近场散射点比例 = count(range<1m, intensity<mean-2*std)
     远距离回波消失距离 = 最大有效回波距离
     强度衰减率 = 线性拟合: log(intensity) ~ range
     输出: /perception/lidar/quality

// ====== 2D LiDAR处理 ======
Lidar2DPipeline:
  Input:  LaserScan (front + rear)
  1. 距离分割: 相邻点距离跳跃 > 0.15m → 新物体
  2. 聚类: 最小3点/段
  3. 输出: 近地障碍列表 (class=obstacle_generic)
```

**参数**：
```yaml
lidar_cluster:
  three_d:
    passthrough_min: 0.5           # 最小距离(m)
    passthrough_max: 80.0          # 最大距离(m)
    voxel_leaf_normal: 0.1         # 正常体素大小
    voxel_leaf_degraded: 0.3       # 退化体素大小
    ransac_dist_thresh: 0.05       # RANSAC平面距离阈值
    ransac_max_iter: 100
    cluster_tolerance_normal: 0.3
    cluster_tolerance_degraded: 0.5
    min_cluster_size: 20
    max_cluster_size: 50000
  two_d:
    distance_jump_threshold: 0.15  # 距离跳跃阈值(m)
    min_segment_points: 3
  quality:
    near_field_range: 1.0          # 近场判定距离
    intensity_scatter_threshold: -2.0  # 散射点标准差倍数
```

---

### 3.5 radar_tracker_node —— 4D毫米波雷达跟踪节点

**职责**：处理4D毫米波雷达输出（点云或目标级），进行多目标跟踪。

**输入Topics**：
```
/perception/synced/radar_tracks   (自定义RadarTrackArray)
```

**输出Topics**：
```
/perception/radar/tracks          (SemanticObjectArray)
/perception/radar/pointcloud      (sensor_msgs/PointCloud2, 毫米波点云)
```

**内部算法管道**：

```
RadarPipeline:
  Input:  radar_targets[] (每个目标: range, azimuth, elevation, doppler, rcs)

  1. 目标筛选:
     - 去除静态杂波: |doppler| > 0.1 m/s
     - 保留静止大目标: RCS > 0 dBsm 且连续存在 > 5帧
     - 去除多径虚影: range连续性 + doppler一致性检查

  2. 毫米波点云聚类 (如果雷达提供点云级输出):
     - DBSCAN: eps=0.5m, min_samples=5
     - 聚类中心 + 散布作为目标估计

  3. 多目标跟踪: 卡尔曼滤波器
     状态向量: [x, y, vx, vy, ax, ay] (2D平面)
     观测向量: [range, azimuth, doppler]
     过程噪声: Q = diag(0.1, 0.1, 0.5, 0.5, 1.0, 1.0)
     观测噪声: R = diag(0.15, 0.02, 0.1)  # range(m), azimuth(rad), doppler(m/s)

  4. 目标分类 (基于雷达特征):
     - RCS > 10 dBsm, 尺寸>2m → large_object (车辆级)
     - RCS 0~10 dBsm, 速度<2m/s → medium_object (人员级)
     - RCS < 0 dBsm, 速度<0.5m/s → small_object
     - 微动检测: doppler微变 ±0.5m/s, 周期 0.3~2Hz → 人体呼吸/微动

  5. 占道检测:
     - 目标宽度 > 隧道宽 * 0.5 → 大范围占道
     - 目标横跨车道中心线 > 3m → 横向占道

  6. 输出: tracked objects + confidence (重烟下置信度0.7, 正常0.5)
     → 标注 source_mask = RADAR (8)
     → class_name = "vehicle_like_obstacle" (诚实标记)
```

**参数**：
```yaml
radar_tracker:
  static_clutter_doppler_thresh: 0.1   # 静态杂波多普勒阈值
  static_object_min_frames: 5          # 静止目标最小帧数
  rcs_min: -10.0                       # 最小RCS(dBsm)
  multipath_check: true
  kalman:
    process_noise: [0.1, 0.1, 0.5, 0.5, 1.0, 1.0]
    observation_noise: [0.15, 0.02, 0.1]
    max_coast_frames: 10               # 最大丢失帧数
  classification:
    vehicle_rcs_min: 10.0
    person_rcs_max: 10.0
    person_rcs_min: -5.0
    micro_doppler_amplitude: 0.5
    micro_doppler_period_min: 0.3
    micro_doppler_period_max: 2.0
```

---

### 3.6 smoke_estimator_node —— 烟雾估计节点

**职责**：融合图像烟雾分割、PM传感器、LiDAR衰减、CO变化率，计算综合烟雾指数。

**输入Topics**：
```
/perception/rgb/smoke_mask        (sensor_msgs/Image)
/perception/lidar/quality         (自定义)
/pm/status                        (PMSensorArray)
/gas/status                       (GasSensorArray)
```

**输出Topics**：
```
/perception/smoke/estimate        (SmokeEstimate)
```

**内部算法**：

```
SmokeEstimator:

  1. 图像烟雾概率提取:
     image_smoke_prob = mean(smoke_mask_confidence) * coverage_ratio

  2. PM变化率:
     pm_now = (PM2.5 + PM10) / 2
     pm_rise_rate = (pm_now - pm_baseline_30s) / pm_baseline_30s
     pm_baseline更新: 取最近30秒PM最小值的滑动平均

  3. LiDAR衰减指数:
     lidar_attenuation = 1.0 - effective_ratio
         + 0.5 * near_field_scatter_ratio
         + 0.3 * (1.0 - max_range / nominal_max_range)

  4. CO变化率:
     co_rise_rate = (CO_now - CO_baseline_60s) / CO_baseline_60s

  5. 综合烟雾指数:
     SmokeScore = w1*image_smoke_prob + w2*pm_rise_rate
                + w3*lidar_attenuation + w4*co_rise_rate

     固定基准权重: w1=0.30, w2=0.25, w3=0.30, w4=0.15

     动态调节:
     - 可见光退化 → w1*=0.5 (图像不可靠)
     - PM传感器污染 → w2*=0.3
     - LiDAR退化 → w3*=0.3 (LiDAR本身已被影响)
     - 气体传感器故障 → w4*=0.2

  6. 输出:
     smoke_score: 限制在 [0,1]
     visibility_level:
       score < 0.15 → CLEAR
       score < 0.35 → LIGHT_SMOKE
       score < 0.65 → HEAVY_SMOKE
       score ≥ 0.65 → BLIND
     smoke_direction: 从3帧烟雾掩膜中心位移计算
```

**参数**：
```yaml
smoke_estimator:
  weights:
    image: 0.30
    pm: 0.25
    lidar: 0.30
    co: 0.15
  dynamic_degradation:
    image_degraded_factor: 0.5
    pm_degraded_factor: 0.3
    lidar_degraded_factor: 0.3
    gas_degraded_factor: 0.2
  pm_baseline_window_s: 30.0
  co_baseline_window_s: 60.0
  thresholds:
    clear: 0.15
    light_smoke: 0.35
    heavy_smoke: 0.65
```

---

### 3.7 gas_risk_node —— 气体风险评估节点

**职责**：评估有毒/可燃气体风险等级。

**输入Topics**：
```
/gas/status                       (GasSensorArray)
/temperature/humidity             (sensor_msgs/Temperature, RelativeHumidity)
```

**输出Topics**：
```
/perception/gas/risk              (GasRisk)
```

**内部算法**：

```
GasRiskAssessor:

  1. 各气体独立评估:

     CO:
       < 50 ppm  → SAFE
       50~200    → CAUTION
       200~1200  → DANGER
       > 1200    → DEADLY

     CO₂:
       < 1000 ppm → SAFE
       1000~5000  → CAUTION
       > 5000     → DANGER

     O₂:
       > 19.5%    → SAFE
       16~19.5%   → CAUTION (缺氧)
       10~16%     → DANGER
       < 10%      → DEADLY

     CH₄ (甲烷):
       < 1% LEL   → SAFE
       1~10% LEL  → CAUTION
       10~25% LEL → DANGER (爆炸风险)
       > 25% LEL  → DEADLY

     H₂S:
       < 10 ppm   → SAFE
       10~100     → DANGER
       > 100      → DEADLY

  2. 综合风险:
     risk_level = max(各气体独立风险)

  3. 组合风险检测:
     fire_risk:        CO快速上升 + CO₂快速上升 + O₂下降 + 温度上升
     asphyxiation_risk: O₂ < 19.5% OR CO > 200 ppm
     explosion_risk:  CH₄ > 10% LEL
```

**参数**：
```yaml
gas_risk:
  thresholds:
    co: [50, 200, 1200]             # CAUTION, DANGER, DEADLY (ppm)
    co2: [1000, 5000, 10000]        # ppm
    o2: [19.5, 16.0, 10.0]          # % (下限)
    ch4_lel: [1.0, 10.0, 25.0]      # % LEL
    h2s: [10, 50, 100]              # ppm
  fire_combo:
    co_rise_rate_threshold: 5.0     # ppm/s
    co2_rise_rate_threshold: 50.0   # ppm/s
    o2_drop_rate_threshold: 0.1     # %/s
    temp_rise_threshold: 2.0        # °C/s
```

---

### 3.8 crack_segmenter_node —— 裂缝检测节点

**职责**：检测隧道衬砌裂缝（仅在巡检模式低速时运行）。

**输入Topics**：
```
/perception/synced/rgb_image      (sensor_msgs/Image)
/perception/lidar/ground_cloud    (sensor_msgs/PointCloud2) # 壁面点云
/perception/synced/odometry       (nav_msgs/Odometry)
```

**输出Topics**：
```
/perception/cracks                (CrackDefectArray)
```

**内部算法管道**：

```
CrackPipeline:
  Input:  RGB image (高分辨率, 固定焦距, 稳定云台)

  1. 预处理:
     - 图像去畸变 (camera_info)
     - CLAHE 局部直方图均衡 (增强低对比度裂缝)
     - 光照均衡: divide_by_gaussian_blur(image, sigma=50)

  2. 衬砌表面区域分割:
     - SegFormer-B0-concrete → 分割出混凝土表面区域
     - 仅对混凝土区域进行裂缝检测 (排除管线、设备)

  3. 裂缝语义分割:
     - CrackFormer-L (ONNX) → 裂缝像素级分割
     - 后处理: 形态学闭运算连接断裂裂缝 (kernel=3)

  4. 裂缝骨架提取:
     - Zhang-Suen 细化算法 → 单像素宽度骨架
     - 分支点检测: 8邻域 ≥ 3个骨架像素

  5. 裂缝测量:
     - 长度: 骨架像素数 × pixel_to_mm_scale
       pixel_to_mm_scale 由激光点云表面距离计算
     - 宽度: 每个骨架点处，沿法线方向找裂缝边缘
       → 亚像素宽度估计 (灰度梯度边缘定位)
     - 输出: max_width, mean_width, length, num_branches

  6. 定位:
     - 像素坐标 → LiDAR表面3D点投影 → 地图坐标
     - 记录: 隧道里程 + 壁面位置(左/右/拱顶) + 地图坐标

  7. 历史对比:
     - 对比同位置历史裂缝记录
     - 输出: length_change, width_change, is_new

  8. 输出:
     - crack_id, position_map, wall_face, length_m, width_mm, width_max_mm
     - orientation, num_branches, confidence, is_growing
```

**参数**：
```yaml
crack_segmenter:
  model: "crack_seg.onnx"
  min_length_mm: 50.0            # 最小裂缝长度(过滤噪点)
  min_width_mm: 0.5              # 最小可检测宽度
  clahe_clip_limit: 2.0
  clahe_tile_size: [8, 8]
  morph_close_kernel: 3
  skeleton_method: "zhang_suen"
  pixel_scale_from_lidar: true   # 从LiDAR点云计算像素-毫米比
  historical_compare: true       # 进行历史对比
  max_position_error_m: 0.5      # 历史对比位置容忍
```

---

### 3.9 water_detector_node —— 积水/渗水检测节点

**职责**：检测路面积水和壁面渗水。

**输入Topics**：
```
/perception/synced/rgb_image      (sensor_msgs/Image)
/perception/lidar/ground_cloud    (sensor_msgs/PointCloud2)
/perception/synced/thermal_image  (sensor_msgs/Image)
/scan/front                       (sensor_msgs/LaserScan) # 低位2D LiDAR
```

**输出Topics**：
```
/perception/water                 (WaterRegionArray)
```

**内部算法管道**：

```
WaterPipeline:

  // ====== 积水检测 ======
  WaterPuddleDetect:
    Input:  RGB + ground_cloud + scan_front

    1. 图像积水检测:
       - SegFormer-B0-water (ONNX) → 积水区域分割
       - 辅助特征:
         * 偏振差异分析 (如果有偏振片)
         * 镜面反射检测 (亮度峰值 + 低纹理)
         * 时序一致性 (积水区域在连续帧中稳定)

    2. 几何确认:
       - LiDAR点云缺失检测: 镜面反射导致地面点云缺失
       - 低位2D LiDAR: 水面回波距离 vs 预期地面距离
         * |measured_range - expected_ground_range| > 0.05m → 高度变化
       - 水面范围: 点云缺失区域 + 周围地面点云边界

    3. 深度粗略估计:
       - 不能仅靠图像估计
       - 方法A: 路面基准高程 - 水面高程 (需要水面点云回波)
       - 方法B: 底盘姿态变化 (进入水面时)
       - 输出: depth_level = shallow (<2cm) / moderate (2~5cm) / deep (>5cm)

    4. 输出: water_polygon + depth_level + confidence

  // ====== 渗水检测 ======
  WaterSeepageDetect:
    Input:  RGB + thermal_image

    1. 可见光湿斑分割:
       - SegFormer微调模型 → 湿斑区域
       - 纹理特征: 湿斑区域通常反射率低、纹理暗化
       - 环境温湿度作为先验: 高湿度环境降低检出阈值

    2. 热成像确认:
       - 湿斑区域温度通常比干燥混凝土低1~5°C (蒸发冷却)
       - 温差 < 2°C 时可能是冷凝 (结合环境温湿度判断)

    3. 输出: seepage_polygon + wall_face + confidence + if_condensation
```

**参数**：
```yaml
water_detector:
  puddle:
    model: "water_seg.onnx"
    conf_threshold: 0.4
    specular_threshold: 200       # 镜面反射亮度阈值
    ground_height_tolerance: 0.05 # 地面高度偏差阈值(m)
    shallow_depth_max: 0.02
    moderate_depth_max: 0.05
  seepage:
    model: "water_seg.onnx"       # 可共用分割模型
    temp_difference_min: 1.0      # 最小温差(°C)
    temp_difference_max: 5.0      # 典型温差范围
    condensation_humidity_min: 85.0 # 冷凝湿度阈值(%)
```

---

### 3.10 facility_inspector_node —— 设施异常检测节点

**职责**：检测隧道固定设施的状态异常（巡检模式）。

**检测对象**：消防栓箱、灭火器、应急电话、照明灯、风机、指示牌、配电箱、线缆、门体、摄像机

**输入Topics**：
```
/perception/synced/rgb_image      (sensor_msgs/Image)
/perception/synced/thermal_image  (sensor_msgs/Image)
/perception/synced/odometry       (nav_msgs/Odometry)
```

**输出Topics**：
```
/perception/facility_anomalies    (FacilityAnomalyArray)
```

**内部算法管道**：

```
FacilityInspection:

  1. 第一阶段：设施检测
     - YOLO11-m-facility (ONNX) → 多类别设施检测
     - Classes: fire_hydrant, extinguisher, emergency_phone,
                light, fan, sign, panel, cable_tray, door, camera
     - 输出: facility_class + bbox + confidence

  2. 第二阶段：异常判定
     对每个检测到的设施:

     a) 完整性检查:
        - 正常基准模板匹配 (从数据库加载对应设施的参考图)
        - 图像配准: ECC算法 或 特征匹配
        - 差异检测: 结构相似度 SSIM < 0.85 → 异常
        - 异常区域分割

     b) 外观异常分类 (ResNet34-Anomaly):
        - 完整/缺失/箱门打开/表面破损/倾斜/脱落/遮挡/
          指示灯熄灭/锈蚀/过热

     c) 热成像辅助:
        - 配电柜: 温度 > 60°C → 过热
        - 风机: 温度异常 → 电机故障
        - 照明: 未点亮但温度高于环境 → 疑似故障

     d) 历史对比:
        - 与同位置历史检测结果对比
        - 标记新增异常 vs 已知异常

  3. 输出: facility_class + anomaly_type + position_map + confidence + is_new

  4. 性能优化:
     - 仅在设施预期位置附近运行第二阶段
     - 基于里程计数 trigger (每N米触发一次检测)
     - 基准图缓存: 预加载当前段隧道基准图
```

**参数**：
```yaml
facility_inspector:
  detector_model: "facility_detector.onnx"
  anomaly_classifier: "facility_anomaly_cls.onnx"
  ssim_threshold: 0.85            # 结构相似度异常阈值
  thermal_overheat:
    panel_temp_max: 60.0          # 配电箱过热温度(°C)
    fan_temp_max: 70.0            # 风机过热温度(°C)
    motor_temp_max: 80.0          # 电机过热温度(°C)
  inspection_interval_m: 5.0      # 检测触发间隔(米)
  baseline_cache_size: 20         # 基准图缓存数量
```

---

## 四、融合节点设计

### 4.1 融合架构总述

6个融合节点形成二级融合体系：

```
一级融合（目标级融合）:             二级融合（场景级融合）:
┌─────────────────────┐        ┌──────────────────────┐
│ person_fusion_node  │──┐     │                      │
├─────────────────────┤  │     │  risk_assessor_node  │
│ fire_fusion_node    │──┼────▶│                      │──▶ Nav2/Safety
├─────────────────────┤  │     │  + 风险评估            │
│ vehicle_fusion_node │  │     │  + 场景理解            │
├─────────────────────┤  │     │  + 高温禁入区          │
│ semantic_tracker    │──┘     │                      │
├─────────────────────┤        └──────────────────────┘
│ semantic_localizer  │
├─────────────────────┤
│ quality_monitor     │──▶ degradation_manager
└─────────────────────┘
```

### 4.2 person_fusion_node —— 人员融合节点

**输入Topics**：
```
/perception/rgb/detections            (SemanticObjectArray)
/perception/thermal/detections        (SemanticObjectArray)
/perception/lidar/clusters            (SemanticObjectArray)
/perception/radar/tracks              (SemanticObjectArray)
/perception/synced/odometry           (nav_msgs/Odometry)
/perception/environment_quality       (EnvironmentQuality)
```

**输出Topics**：
```
/perception/person_fused              (SemanticObjectArray)
```

**内部算法**：

```
PersonFusion:

  // 步骤1: 收集所有人员候选
  candidates = []
  for each detector source in [RGB, THERMAL, LIDAR, RADAR]:
    if source.credibility > 0.3:  // 跳过不可信传感器
      for obj in source.detections:
        if obj.class_id in PERSON_CLASSES:
          candidates.append({
            source: source,
            obj: obj,
            position_3d: compute_3d_position(obj, source)  // 调用定位方法A/B/C/D
          })

  // 步骤2: 数据关联 (匈牙利匹配)
  cost_matrix = zeros(N_candidates, N_tracks)
  for i, cand in candidates:
    for j, track in active_tracks:
      cost_matrix[i][j] = association_cost(cand, track)

  关联代价函数:
    cost = w_pos * ||cand.pos - track.pos||
         + w_class * (cand.class != track.class ? 1 : 0)
         + w_temp * |cand.temp - track.temp| / 10.0
         + w_time * |cand.t - track.last_update|
    w_pos=0.4, w_class=0.3, w_temp=0.2, w_time=0.1

  门限过滤: 空间距离 > 3.0m AND 类别不兼容 → cost=INF

  // 步骤3: 匈牙利匹配 + 卡尔曼更新
  matched_pairs, unmatched_cands, unmatched_tracks = hungarian(cost_matrix, max_cost=2.0)

  for (cand, track) in matched_pairs:
    // 卡尔曼更新
    track.kalman.update(cand.position_3d)
    // 融合置信度 (动态权重)
    w_rgb = env_quality.rgb_credibility
    w_thermal = env_quality.thermal_credibility
    w_lidar = env_quality.lidar_credibility
    w_radar = env_quality.radar_credibility
    // 归一化
    sum_w = w_rgb + w_thermal + w_lidar + w_radar
    track.confidence += (1 - track.confidence) *
      (w_rgb*cand.rgb_conf + w_thermal*cand.thermal_conf +
       w_lidar*cand.lidar_conf + w_radar*cand.radar_conf) / sum_w
    track.source_mask |= cand.source_mask
    track.last_seen = now()

  for cand in unmatched_cands:
    if cand.confidence > 0.5:  // 高置信新目标
      tracks.append(create_track(cand))

  for track in unmatched_tracks:
    track.coast_frames++
    if track.coast_frames > 15:
      tracks.remove(track)      // 移除丢失目标

  // 步骤4: 人员3D定位 (从最高质量源获取深度)
  for track in tracks:
    depth_sources = sort_by_credibility(
      [lidar_depth, radar_range, laser_range, stereo_depth, ground_projection_estimate])
    track.pose = pick_best_valid_depth(depth_sources)

  // 步骤5: 统一输出
  publish PersonFusedArray
```

### 4.3 fire_fusion_node —— 火源融合节点

**内部算法**：

```
FireFusion:

  // 四路确认逻辑
  for each fire_candidate:
    evidence = {
      rgb_flame:    get_rgb_flame_confidence(candidate),
      thermal_hot:  get_thermal_hotspot_confidence(candidate),
      uv_ir_sensor: get_uv_ir_alarm(candidate),  // 如果安装
      gas_combo:    check_gas_fire_signature()     // CO↑ + CO₂↑ + O₂↓ + Temp↑
    }

    // 火灾确认决策矩阵
    confirmed = false
    if evidence.rgb_flame > 0.7 AND evidence.thermal_hot:
      → fire_confirmed (置信度 0.9+, 等级2)
    elif evidence.thermal_hot AND evidence.gas_combo:
      → fire_confirmed (置信度 0.85+, 等级2)
    elif evidence.thermal_hot AND evidence.uv_ir_sensor:
      → fire_confirmed (置信度 0.95+, 等级2)
    elif evidence.rgb_flame > 0.5 AND evidence.gas_combo:
      → fire_suspected (置信度 0.7, 等级1)
    elif evidence.thermal_hot:
      → fire_suspected (置信度 0.6, 等级1)
    else:
      → noise (抑制)

    // 大面积 + 高温 + 气体恶化 = 高危
    if confirmed AND evidence.covers_large_area AND evidence.temp > 300°C:
      fire_level = 3  // HIGH_RISK
    if confirmed AND evidence.gas_o2_dropping AND evidence.temp > 500°C:
      fire_level = 4  // CRITICAL (禁入)

  // 火源定位: 同人员定位逻辑
  // 发布 HotZone 多边形给 Nav2 Costmap
```

### 4.4 vehicle_fusion_node —— 车辆融合节点

**内部算法**：

```
VehicleFusion:

  1. 收集: 可见光车辆检测 + LiDAR聚类(vehicle_candidate) + 毫米波(large_object)

  2. 关联:
     - 可见光车辆框 ↔ LiDAR 3D框: 通过2D投影IoU匹配
     - LiDAR 3D框 ↔ 毫米波目标: 通过空间距离+速度匹配

  3. 事故状态融合:
     if rgb_accident_state AND lidar_anomaly_pose:
       → 事故状态置信度提升
     if only_lidar (无可见光):
       → 事故状态标记为 "unknown_accident_state"
     if only_mmwave (重烟):
       → class_name = "vehicle_like_obstacle" (诚实标记)

  4. 占道评估:
     footprint_polygon = project_3d_box_to_ground(track)
     lane_overlap = compute_lane_overlap(footprint_polygon, lane_boundary)
     if lane_overlap > 0.7: → BLOCKING
     if lane_overlap > 0.3: → PARTIAL_BLOCKING

  5. 输出: /perception/vehicle_fused → Nav2 Costmap 更新
```

### 4.5 semantic_tracker_node —— 统一多目标跟踪节点

**职责**：对所有融合后的目标进行统一多目标跟踪。

**输入Topics**：
```
/perception/person_fused
/perception/fire_fused
/perception/vehicle_fused
/perception/lidar/clusters
```

**输出Topics**：
```
/perception/tracks/all             (SemanticObjectArray)
/perception/tracks/events           (自定义, 跟踪事件: NEW/LOST/MERGED/SPLIT)
```

**内部算法**：

```
SemanticTracker:

  全局跟踪池: track_pool[track_id]
  最大跟踪数量: 256

  1. 关联算法 (类DeepSORT):
     - 运动模型: 8维卡尔曼 [x, y, z, w, l, h, vx, vy]
     - 外观模型: 不需要 (已有类别标签)
     - 级联匹配: 优先匹配最近活跃的跟踪

  2. 关联特征:
     spatial_distance: 马氏距离 (考虑协方差)
     temporal_gap: 匹配优先级递减
     class_consistency: 类别不突变

  3. 跟踪管理:
     - 确认跟踪: 连续3帧匹配 → confirmed
     - 试探跟踪: 新目标 → tentative (3帧内未确认 → 删除)
     - 丢失跟踪: 超15帧未匹配 → lost → 保持10s后删除
     - 合并: 两个跟踪距离 < 0.5m 且类别相同 → merge

  4. 平滑输出:
     - 卡尔曼预测位置 (减少跳动)
     - 速度估计: 从位置差分 + 雷达/毫米波速度融合
```

---

### 4.6 semantic_localizer_node —— 语义定位节点

**职责**：将所有目标从传感器坐标系转换到全局地图坐标系。

**输入Topics**：
```
/perception/tracks/all              (SemanticObjectArray)
/tf                                (tf2_msgs/TFMessage)
/tf_static                         (tf2_msgs/TFMessage)
```

**输出Topics**：
```
/perception/semantic_markers        (visualization_msgs/MarkerArray)
/perception/semantic_map_objects    (SemanticObjectArray, 地图坐标)
```

**内部算法**：

```
SemanticLocalizer:

  坐标系: map → odom → base_link → sensor_link
  使用 TF2 进行坐标变换

  for each tracked_object:
    // 步骤1: 获取传感器坐标系下的位置
    P_sensor = tracked_object.pose_sensor

    // 步骤2: 查找TF变换
    try:
      T_base_sensor = tf_buffer.lookup("base_link", tracked_object.sensor_frame)
      T_map_base = tf_buffer.lookup("map", "base_link")

      // 步骤3: 坐标链
      P_map = T_map_base * T_base_sensor * P_sensor

      // 步骤4: 协方差传播
      cov_map = J * cov_sensor * J^T
      其中 J = ∂P_map/∂P_sensor  (雅可比)

    catch TransformException:
      // TF不可用 → 标记为相对坐标, 不发布地图坐标

    // 步骤5: 输出
    tracked_object.pose.map = P_map
    tracked_object.pose_covariance = cov_map

  // 同时发布 RViz Marker 用于可视化
```

---

## 五、系统管理节点

### 5.1 perception_quality_monitor —— 环境质量评估节点

**职责**：实时评估各传感器数据质量和环境条件，输出统一的环境质量报告。

**输入Topics**：
```
/perception/synced/rgb_image
/perception/synced/thermal_image
/perception/synced/lidar_points
/perception/lidar/quality
/perception/radar/tracks
/gas/status, /pm/status, /temperature/humidity
/perception/sync_status
```

**输出Topics**：
```
/perception/environment_quality    (EnvironmentQuality)
/perception/sensor_health          (SensorHealth[])
```

**内部算法**：

```
QualityMonitor:

  // ===== 可见光质量 ======
  rgb_quality:
    - 图像清晰度: Laplacian方差 (Var < 50 → 模糊)
    - 图像亮度: 平均像素值 (mean < 30 → 过暗, mean > 240 → 过曝)
    - 图像对比度: RMS contrast
    - 综合: rgb_credibility = f(sharpness, brightness, contrast)
    - 窗口污染: 图像梯度持续低 + 角点检测减少 → contamination

  // ===== 热成像质量 ======
  thermal_quality:
    - 有效像素率: 非饱和像素 / 总像素 (> 95% → OK)
    - 温度均匀性: 平场区域标准差
    - NUC状态: 是否有残留非均匀性
    - 窗口污染: 图像平滑度异常高 → 可能污染
    - 综合: thermal_credibility = f(valid_pixel_rate, uniformity, window_state)

  // ===== LiDAR质量 ======
  lidar_quality: (来自 lidar_cluster_node)
    - 有效点比例、近场散射点、远距回波

  // ===== 毫米波质量 ======
  radar_quality:
    - 有效目标数量: > 0 → OK
    - RCS一致性: RCS在连续帧间合理变化
    - 多径检测: 有异常镜像目标 → 降级

  // ===== 气体传感器 ======
  gas_quality:
    - 预热状态: 上电后 > 30s → ready
    - 传感器响应: 基线漂移检查
    - 交叉干扰: 高湿度影响电化学传感器

  // ===== 综合可见度等级 ======
  visibility_level = compute_visibility(
    rgb_credibility,
    thermal_credibility,
    smoke_score (来自smoke_estimator),
    rgb_brightness
  )

  决策表:
    smoke < 0.15 AND rgb_brightness > 50 AND rgb_sharpness > 80 → CLEAR
    smoke < 0.15 AND rgb_brightness < 50 → LOW_LIGHT
    smoke 0.15~0.35 → LIGHT_SMOKE
    smoke 0.35~0.65 → HEAVY_SMOKE
    smoke >= 0.65 → BLIND
```

---

### 5.2 degradation_manager —— 退化模式管理节点

**职责**：管理感知系统的四级退化模式，控制传感器权重切换和机器人行为建议。

**输入Topics**：
```
/perception/environment_quality    (EnvironmentQuality)
/perception/sensor_health          (SensorHealth[])
/perception/smoke/estimate         (SmokeEstimate)
```

**输出Topics**：
```
/perception/degradation_mode       (std_msgs/String)        # 当前模式名
/perception/sensor_weights         (自定义/Float32MultiArray) # 传感器权重
/perception/behavior_recommendation (自定义)                 # 行为建议
```

**内部状态机**：

```
State Machine:

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    ▼                                                     │
  ┌─────────┐   烟雾>0.15     ┌──────────────┐           │
  │  CLEAR  │──────────────▶│ LOW_VISIBILITY│           │
  └─────────┘◀──────────────└──────────────┘           │
       ▲     烟雾<0.1              │                      │
       │                           │ 烟雾>0.35            │
       │         烟雾<0.25         ▼                      │
       │              ┌───────────────┐                   │
       │              │ HEAVY_SMOKE   │                   │
       │              └───────────────┘                   │
       │                    │                             │
       │    多传感器失效     │ 多传感器失效               │
       │                    ▼                             │
       │           ┌─────────────────────┐               │
       └───────────│PERCEPTION_DEGRADED  │──────────────┘
    恢复条件满足   └─────────────────────┘  维持退化
                    (机器人低速撤退/远程接管)

  状态转移条件:
    CLEAR → LOW_VISIBILITY:
      smoke_score > 0.15 OR rgb_credibility < 0.5 OR brightness < 30

    LOW_VISIBILITY → HEAVY_SMOKE:
      smoke_score > 0.35 OR (rgb_credibility < 0.2 AND lidar_credibility < 0.3)

    HEAVY_SMOKE → PERCEPTION_DEGRADED:
      (rgb_credibility < 0.1 AND lidar_credibility < 0.2 AND thermal_credibility < 0.3)
      OR gas_sensor_status = FAILED

    PERCEPTION_DEGRADED → HEAVY_SMOKE:
      thermal_credibility > 0.5 AND radar_credibility > 0.5

    HEAVY_SMOKE → LOW_VISIBILITY:
      smoke_score < 0.25 AND rgb_credibility > 0.3

    LOW_VISIBILITY → CLEAR:
      smoke_score < 0.1 AND rgb_credibility > 0.7 AND brightness > 50

  滞后: 进入条件比退出条件宽松20% (防止振荡)

  传感器权重表:

  ┌─────────────────┬────────┬──────────────┬────────────┬───────────────────┐
  │ 传感器           │ CLEAR  │ LOW_VIS      │ HEAVY_SMOKE│ PERCEPTION_DEGR   │
  ├─────────────────┼────────┼──────────────┼────────────┼───────────────────┤
  │ RGB 可见光       │ 0.90   │ 0.50         │ 0.10 关闭  │ 0.00 关闭         │
  │ Thermal 热成像   │ 0.40   │ 0.80         │ 0.85       │ 0.30*             │
  │ LiDAR 3D        │ 0.85   │ 0.60         │ 0.25       │ 0.00 关闭         │
  │ 4D mmWave       │ 0.30   │ 0.50         │ 0.85       │ 0.80              │
  │ 2D LiDAR/TOF    │ 0.20   │ 0.30         │ 0.50       │ 0.60              │
  │ Gas/PM          │ 0.20   │ 0.40         │ 0.60       │ 0.60              │
  └─────────────────┴────────┴──────────────┴────────────┴───────────────────┘

  * 热成像权重在PERCEPTION_DEGRADED降低: 若窗口被严重污染

  行为建议:
    CLEAR:             全速巡检测模式
    LOW_VISIBILITY:    限速80%，增加感知确认时间
    HEAVY_SMOKE:       限速30%，仅关键任务(人员搜救)，准备撤退
    PERCEPTION_DEGRADED: 限速10%，低速撤退，建议远程接管
```

---

### 5.3 risk_assessor_node —— 风险评估节点

**职责**：综合所有感知结果，生成场景级风险评估。

**输入Topics**：
```
/perception/tracks/all
/perception/environment_quality
/perception/gas/risk
/perception/smoke/estimate
```

**输出Topics**：
```
/perception/risk/assessment       (自定义RiskAssessment)
/perception/risk/hot_zones        (HotZoneArray)
/perception/risk/costmap_overlay  (nav_msgs/OccupancyGrid, Nav2兼容)
```

**内部算法**：

```
RiskAssessor:

  // === 全局风险等级 ====
  global_risk = max(
    max_fire_risk,       // 火源综合风险
    max_gas_risk,        // 气体综合风险
    max_person_risk,     // 人员安全风险 (人员位于高危区)
    nav_blockage_risk,   // 通行阻断风险
    perception_lost_risk // 感知失效风险
  )

  // === 高温禁区生成 ====
  hot_zones = []
  for fire_track in fire_tracks:
    hot_zone = HotZone()
    hot_zone.polygon = expand(fire_track.position, fire_track.temperature_based_radius)
    hot_zone.max_temp = fire_track.temperature_max
    hot_zone.severity = classify(fire_track.temperature_max, fire_track.area)
    hot_zones.append(hot_zone)

  // 写入 Nav2 local_costmap
  for zone in hot_zones:
    costmap.set_cost(zone.polygon, LETHAL_OBSTACLE)  // 致命禁区

  // === 通行风险评估 ====
  for vehicle_track in vehicle_tracks:
    if vehicle_track.blockage_ratio > 0.7:
      nav_blockage = BLOCKED
      recommend_stop()
    elif vehicle_track.blockage_ratio > 0.3:
      nav_blockage = PARTIAL
      recommend_reroute()

  // === 事件生成 ====
  if new_fire_confirmed:  emit_event("fire_confirmed")
  if new_person_found:    emit_event("person_found")
  if vehicle_blocking:    emit_event("accident_vehicle_blocking")
  if perception_lost:     emit_event("perception_lost")
  if gas_danger:          emit_event("gas_danger_level")
```

---

## 六、配置系统

### 6.1 perception_master.yaml（全局主配置）

```yaml
# perception_master.yaml
perception:
  # 系统设置
  system:
    namespace: ""                 # ROS命名空间
    pipeline_rate: 100.0          # 主管线频率(Hz)
    diagnostic_rate: 10.0         # 诊断频率(Hz)
    frame_id_map: "map"
    frame_id_odom: "odom"
    frame_id_base: "base_link"
    target_hardware: "jetson_agx_orin"
    num_cuda_devices: 1
    intra_process_comms: true     # 使用节点内零拷贝

  # 传感器使能
  sensors:
    rgb_front: true
    rgb_ptz: true
    thermal: true
    lidar_3d: true
    radar_4d: true
    lidar_2d_front: true
    lidar_2d_rear: true
    ultrasonic_front: true
    ultrasonic_rear: true
    gas_sensors: true
    pm_sensors: true
    temperature_humidity: true
    microphone_array: false       # 可选

  # 检测器使能
  detectors:
    flame_detection: true
    smoke_detection: true
    person_detection: true
    vehicle_detection: true
    crack_detection: true
    water_detection: true
    facility_inspection: true

  # 模式相关
  modes:
    inspection:                   # 巡检模式
      crack_detection: true
      water_detection: true
      facility_inspection: true
      flame_detection: true
      person_detection: false     # 巡检不主动搜救
    rescue:                       # 救援模式
      crack_detection: false
      water_detection: false
      facility_inspection: false
      flame_detection: true
      person_detection: true
    minimal:                      # 退化最小模式
      crack_detection: false
      water_detection: false
      facility_inspection: false
      flame_detection: false
      person_detection: true      # 仅保留人员搜索
```

### 6.2 degradation.yaml（退化策略配置）

```yaml
degradation:
  # 状态转移
  transitions:
    clear_to_lowvis:
      smoke_score_above: 0.15
      rgb_credibility_below: 0.5
      brightness_below: 30
    lowvis_to_heavy:
      smoke_score_above: 0.35
      rgb_lidar_both_below: [0.2, 0.3]
    heavy_to_degraded:
      all_sensors_below: [0.1, 0.2, 0.3]  # rgb, lidar, thermal
    hysteresis_factor: 0.2       # 滞后因子

  # 传感器权重 [rgb, thermal, lidar, radar, 2dlidar, gas]
  weights:
    clear:        [0.90, 0.40, 0.85, 0.30, 0.20, 0.20]
    low_vis:      [0.50, 0.80, 0.60, 0.50, 0.30, 0.40]
    heavy_smoke:  [0.10, 0.85, 0.25, 0.85, 0.50, 0.60]
    degraded:     [0.00, 0.30, 0.00, 0.80, 0.60, 0.60]

  # 行为限制
  behavior:
    clear:        {max_speed_pct: 100, require_confirmation: false, allow_autonomous: true}
    low_vis:      {max_speed_pct: 80,  require_confirmation: false, allow_autonomous: true}
    heavy_smoke:  {max_speed_pct: 30,  require_confirmation: true,  allow_autonomous: true}
    degraded:     {max_speed_pct: 10,  require_confirmation: true,  allow_autonomous: false, suggest_teleop: true}

  # 深度源优先级 (CLEAR / LOW_VIS / HEAVY_SMOKE / DEGRADED)
  depth_priority:
    - [lidar, stereo, laser_range, mmwave, ground_project]
    - [lidar, mmwave, laser_range, stereo, ground_project]
    - [mmwave, thermal_lidar_proj, laser_range, ground_project]
    - [mmwave, tof, uwb_position, ground_project]
```

---

## 七、启动文件结构

### perception_bringup.launch.py（完整启动）

```python
def generate_launch_description():
    # 1. 配置加载
    config_dir = get_package_share_directory('rescue_perception_bringup') + '/config/'

    # 2. 传感器同步 (必须最先启动)
    sensor_sync = Node(
        package='rescue_perception_core',
        executable='sensor_sync_node',
        parameters=[config_dir + 'perception_master.yaml',
                    config_dir + 'sensors.yaml'],
        output='screen'
    )

    # 3. 质量监控 (第二个启动，为其他节点提供环境质量)
    quality_monitor = Node(
        package='rescue_perception_core',
        executable='perception_quality_monitor',
        parameters=[config_dir + 'perception_master.yaml'],
        output='screen'
    )

    # 4. 检测节点组 (并行启动)
    detection_nodes = [
        Node(package='rescue_perception_core', executable='rgb_detector_node',
             parameters=[config_dir + 'perception_master.yaml', config_dir + 'detectors.yaml']),
        Node(package='rescue_perception_core', executable='thermal_detector_node',
             parameters=[config_dir + 'perception_master.yaml', config_dir + 'detectors.yaml']),
        Node(package='rescue_perception_core', executable='lidar_cluster_node',
             parameters=[config_dir + 'perception_master.yaml', config_dir + 'detectors.yaml']),
        Node(package='rescue_perception_core', executable='radar_tracker_node',
             parameters=[config_dir + 'perception_master.yaml', config_dir + 'detectors.yaml']),
        Node(package='rescue_perception_core', executable='smoke_estimator_node',
             parameters=[config_dir + 'perception_master.yaml']),
        Node(package='rescue_perception_core', executable='gas_risk_node',
             parameters=[config_dir + 'perception_master.yaml']),
    ]

    # 5. 融合节点组
    fusion_nodes = [
        Node(package='rescue_perception_core', executable='person_fusion_node',
             parameters=[config_dir + 'perception_master.yaml', config_dir + 'fusion.yaml']),
        Node(package='rescue_perception_core', executable='fire_fusion_node',
             parameters=[config_dir + 'perception_master.yaml', config_dir + 'fusion.yaml']),
        Node(package='rescue_perception_core', executable='vehicle_fusion_node',
             parameters=[config_dir + 'perception_master.yaml', config_dir + 'fusion.yaml']),
        Node(package='rescue_perception_core', executable='semantic_tracker_node',
             parameters=[config_dir + 'perception_master.yaml', config_dir + 'fusion.yaml']),
        Node(package='rescue_perception_core', executable='semantic_localizer_node',
             parameters=[config_dir + 'perception_master.yaml']),
    ]

    # 6. 管理节点
    mgmt_nodes = [
        Node(package='rescue_perception_core', executable='degradation_manager',
             parameters=[config_dir + 'perception_master.yaml', config_dir + 'degradation.yaml']),
        Node(package='rescue_perception_core', executable='risk_assessor_node',
             parameters=[config_dir + 'perception_master.yaml', config_dir + 'risk.yaml']),
    ]

    # 7. 巡检专用节点 (条件启动)
    inspection_nodes = [
        Node(package='rescue_perception_core', executable='crack_segmenter_node',
             parameters=[config_dir + 'perception_master.yaml', config_dir + 'detectors.yaml']),
        Node(package='rescue_perception_core', executable='water_detector_node',
             parameters=[config_dir + 'perception_master.yaml', config_dir + 'detectors.yaml']),
        Node(package='rescue_perception_core', executable='facility_inspector_node',
             parameters=[config_dir + 'perception_master.yaml', config_dir + 'detectors.yaml']),
    ]

    return LaunchDescription([
        sensor_sync,
        quality_monitor,
        *detection_nodes,
        *fusion_nodes,
        *mgmt_nodes,
        *inspection_nodes,  # 或通过参数条件启动
    ])
```

---

## 八、分阶段实施路线

### Phase 1: 基础感知 MVP（4~6周）

**目标**：实现基本的目标检测和ROS 2发布

- [ ] `rescue_perception_msgs` 包：所有自定义消息定义
- [ ] `sensor_sync_node`：时间同步，适配所有传感器(模拟数据可暂用rosbag)
- [ ] `rgb_detector_node`：实现火焰、车辆、人员三个管道的ONNX推理
- [ ] `thermal_detector_node`：实现人员检测 + 高温区分割
- [ ] `lidar_cluster_node`：地面分割 + 欧氏聚类
- [ ] `gas_risk_node`：气体阈值判定
- [ ] 基础启动文件 + RViz 可视化

**验收**：使用rosbag回放数据，各检测器独立输出检测框和目标列表

### Phase 2: 目标三维定位（4~6周）

- [ ] `semantic_localizer_node`：TF2坐标变换链
- [ ] LiDAR-相机外参标定工具
- [ ] 热成像-LiDAR投影定位 (方法A)
- [ ] 毫米波-热成像关联定位 (方法C)
- [ ] `multi_object_tracker_node`（后改为 semantic_tracker_node）：卡尔曼+匈牙利
- [ ] 目标地图坐标发布 + RViz Marker

**验收**：目标在地图中正确显示三维位置，跟踪ID稳定

### Phase 3: 多传感器融合（4~6周）

- [ ] `person_fusion_node`：人员多源关联 + 动态置信度
- [ ] `fire_fusion_node`：四路确认逻辑 + 火灾分级
- [ ] `vehicle_fusion_node`：可见光+LiDAR+毫米波车辆融合
- [ ] `smoke_estimator_node`：SmokeScore 四因子加权
- [ ] 关联算法优化：匈牙利匹配 + 门限调优

**验收**：同一目标只产生一条融合结果，不同传感器源正确互补

### Phase 4: 环境感知与退化管理（4~6周）

- [ ] `perception_quality_monitor`：全部传感器质量评估
- [ ] `degradation_manager`：四级状态机 + 动态权重切换
- [ ] `risk_assessor_node`：场景风险 + 高温禁区 + Nav2 Costmap
- [ ] 重烟/低照度 rosbag 测试 + 权重切换验证
- [ ] 行为建议输出集成

**验收**：环境变化时感知模式自动切换，传感器权重正确调整

### Phase 5: 巡检功能 + 场景级集成（4~6周）

- [ ] `crack_segmenter_node`：CrackFormer ONNX + 骨架提取 + 亚像素测量
- [ ] `water_detector_node`：积水/渗水分割 + 几何确认
- [ ] `facility_inspector_node`：设备检测 + 异常分类 + 基准图对比
- [ ] 完整 Nav2 Costmap 集成
- [ ] 任务状态机联动 (PATROL/FIRE_RESCUE/SEARCH_PERSON/DEGRADED)
- [ ] Safety Supervisor 联动 (限速/停车/撤退)
- [ ] 端到端 rosbag 测试 + 性能基准

**验收**：完整巡检→火灾→救援→撤退全流程 rosbag 测试通过

---

## 九、关键技术决策汇总

| 决策点 | 选择 | 理由 |
|---|---|---|
| 融合策略 | 后融合(Post-Fusion) | 易调试、单传感器失效不拖垮全局、适合安全系统 |
| 目标检测框架 | YOLO11 + ONNX Runtime (TensorRT后端) | Jetson Orin最佳推理性能 |
| 分割框架 | SegFormer-B0 | 轻量且精度足够 |
| 跟踪算法 | 卡尔曼 + 匈牙利匹配 (类DeepSORT) | 工程成熟、确定性高 |
| 三维检测 | 点云聚类 + 尺寸规则 (Phase 1) | 隧道场景大型目标为主，无需复杂3D网络 |
| 时间同步 | message_filters ApproximateTime | ROS 2 标准方案 |
| 推理后端 | ONNX Runtime + TensorRT FP16 | Jetson Orin GPU最优 |
| 配置文件格式 | YAML + ROS 2参数服务器 | 运行时动态调整 |
| 消息序列化 | ROS 2 CDR (默认) | 满足10Hz延迟要求 |
| 节点间通信 | 组件化 (ComposableNode) 零拷贝 | 降低延迟和CPU开销 |

---

## 十、验证方案

1. **单元测试**：每个检测器节点独立测试，使用预标注数据集验证mAP
2. **集成测试**：使用多传感器rosbag回放，验证端到端管线
3. **退化测试**：模拟传感器失效(注入噪声/丢帧/延迟)，验证退化切换
4. **性能测试**：`ros2 run rescue_perception_core benchmark_pipeline` 测量端到端延迟
5. **验收标准**：
   - 火焰检测召回率 ≥ 90% (正常环境)
   - 人员识别召回率 ≥ 90% (正常/低照度)
   - 端到端延迟 ≤ 100ms @10Hz
   - 退化切换延迟 ≤ 1s
   - 传感器时间同步误差 ≤ 10ms (关键数据)

---

## 十一、核心算法实现框架

本章给出所有关键算法的伪代码实现框架，可直接映射为 C++ 实现。每个算法均标注输入输出、时间复杂度和适用的退化等级。

---

### 11.1 主感知管线调度算法

```
┌──────────────────────────────────────────────────────────────────┐
│                  PerceptionPipeline::spinOnce()                  │
├──────────────────────────────────────────────────────────────────┤
│ 输入: 无 (从订阅的 Topics 回调中获取最新同步数据)                   │
│ 输出: /perception/target_observations                           │
│       /perception/status                                        │
│       /perception/environment_risk                              │
│       /thermal/risk_level                                       │
│       /gas_alert                                                │
│ 适用: CLEAR / LOW_VISIBILITY / HEAVY_SMOKE / PERCEPTION_DEGRADED│
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  // ===== 阶段0: 传感器健康检查 =====                             │
│  env_quality = quality_monitor->assess(synced_data)              │
│  degradation_mode = degradation_mgr->update(env_quality)         │
│  sensor_weights = degradation_mgr->getWeights(degradation_mode)  │
│                                                                  │
│  // ===== 阶段1: 各传感器独立检测 (并行) =====                     │
│  ParallelLaunch:                                                 │
│    rgb_detections   = rgb_detector->detect(synced_data.rgb)      │
│    thermal_detections = thermal_detector->detect(                │
│                          synced_data.thermal_img,                │
│                          synced_data.temp_matrix)                │
│    lidar_clusters   = lidar_cluster->cluster(                    │
│                          synced_data.lidar_points,               │
│                          env_quality.lidar_credibility)          │
│    radar_tracks     = radar_tracker->track(                      │
│                          synced_data.radar_data)                 │
│    smoke_estimate   = smoke_estimator->estimate(                 │
│                          rgb_detections.smoke_mask,              │
│                          env_quality)                            │
│    gas_risk         = gas_risk->assess(synced_data.gas)          │
│                                                                  │
│  // ===== 阶段2: 目标级融合 (按类别) =====                         │
│  ParallelLaunch:                                                 │
│    person_fused  = person_fusion->fuse(                          │
│                      rgb_detections.persons,                     │
│                      thermal_detections.persons,                 │
│                      lidar_clusters.person_candidates,           │
│                      radar_tracks.medium_objects,                │
│                      sensor_weights, degradation_mode)           │
│    fire_fused    = fire_fusion->fuse(                            │
│                      rgb_detections.flames,                      │
│                      thermal_detections.hotspots,                │
│                      gas_risk, sensor_weights)                   │
│    vehicle_fused = vehicle_fusion->fuse(                         │
│                      rgb_detections.vehicles,                    │
│                      lidar_clusters.vehicle_candidates,          │
│                      radar_tracks.large_objects,                 │
│                      sensor_weights, degradation_mode)           │
│                                                                  │
│  // ===== 阶段3: 统一跟踪 =====                                    │
│  all_tracks = semantic_tracker->update(                          │
│                 person_fused + fire_fused + vehicle_fused)       │
│                                                                  │
│  // ===== 阶段4: 坐标转换 =====                                    │
│  map_objects = semantic_localizer->transform(all_tracks, tf)     │
│                                                                  │
│  // ===== 阶段5: 场景级风险评估 =====                               │
│  scene_risk = risk_assessor->assess(                             │
│                 map_objects, env_quality, gas_risk,              │
│                 smoke_estimate, degradation_mode)                │
│                                                                  │
│  // ===== 阶段6: 安全联动 =====                                    │
│  safety_action = safety_mapper->map(                             │
│                    scene_risk, thermal_detections.risk_level,    │
│                    degradation_mode)                             │
│                                                                  │
│  // ===== 阶段7: 输出发布 =====                                    │
│  publish("/perception/target_observations", map_objects)         │
│  publish("/perception/status", degradation_mode)                 │
│  publish("/perception/environment_risk", scene_risk)             │
│  publish("/thermal/risk_level", thermal_detections.risk_level)   │
│  publish("/gas_alert", gas_risk)                                 │
│  publishSafetyRecommendation(safety_action)                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 11.2 传感器可信度计算算法

```
┌──────────────────────────────────────────────────────────────────┐
│  SensorCredibility::compute(sensor_type, raw_data, history)      │
├──────────────────────────────────────────────────────────────────┤
│ 输入: sensor_type ∈ {RGB, THERMAL, LIDAR, RADAR, GAS, PM}       │
│       raw_data: 当前帧原始数据                                    │
│       history: 最近 N 帧历史 (N = 30 for RGB/LiDAR, 60 for GAS)  │
│ 输出: credibility ∈ [0.0, 1.0]                                  │
│       effective_ratio ∈ [0.0, 1.0]                              │
│       degradation_cause: string                                  │
│ 复杂度: O(W*H) for image sensors, O(N) for point sensors         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  switch (sensor_type):                                           │
│                                                                  │
│    case RGB:                                                     │
│      // 1. 清晰度评估                                             │
│      gray = rgb_to_gray(raw_data.image)                          │
│      laplacian_var = variance(laplacian(gray, CV_64F))           │
│      sharpness = clamp(laplacian_var / 500.0, 0.0, 1.0)         │
│      // 2. 亮度评估                                               │
│      mean_brightness = mean(gray)                                │
│      brightness_score = 1.0 - abs(mean_brightness - 128) / 128   │
│      // 3. 对比度评估 (RMS contrast)                              │
│      rms_contrast = sqrt(mean((gray - mean_brightness)^2))       │
│      contrast_score = clamp(rms_contrast / 80.0, 0.0, 1.0)      │
│      // 4. 污染检测                                               │
│      corners = goodFeaturesToTrack(gray, maxCorners=100)         │
│      corner_ratio = len(corners) / 100.0                         │
│      contamination = corner_ratio < 0.1 ? 0.2 : 1.0             │
│      // 5. 综合                                                  │
│      credibility = sharpness * 0.35                              │
│                   + brightness_score * 0.25                      │
│                   + contrast_score * 0.25                        │
│                   + contamination * 0.15                         │
│      effective_ratio = credibility  // 图像级直接使用             │
│      degradation_cause = credibility < 0.3 ? "low_visibility"    │
│                         : credibility < 0.5 ? "moderate"         │
│                         : "normal"                               │
│                                                                  │
│    case THERMAL:                                                 │
│      // 1. 有效像素率 (非饱和)                                    │
│      saturated = count(temp_matrix > 500°C OR temp_matrix < -40) │
│      valid_ratio = 1.0 - saturated / total_pixels                │
│      // 2. NUC均匀性                                              │
│      flat_region = temp_matrix[center_roi]                       │
│      uniformity = 1.0 - std(flat_region) / 50.0                  │
│      // 3. 窗口污染                                              │
│      gradient_magnitude = mean(sobel(thermal_img))               │
│      window_score = gradient_magnitude > 10 ? 1.0                │
│                     : gradient_magnitude / 10.0                  │
│      // 4. 综合                                                  │
│      credibility = valid_ratio * 0.35                            │
│                   + clamp(uniformity,0,1) * 0.35                 │
│                   + window_score * 0.30                          │
│      effective_ratio = valid_ratio                               │
│                                                                  │
│    case LIDAR:                                                   │
│      // 1. 有效点比例                                             │
│      points_in_range = count(points where range in [0.5, 80.0])  │
│      effective_ratio = points_in_range / nominal_point_count     │
│      // 2. 近场散射检测                                          │
│      near_field = points where range < 1.0                       │
│      intensity_mean = mean(near_field.intensity)                  │
│      intensity_std = std(all_points.intensity)                    │
│      scatter_ratio = count(near_field where                      │
│                        intensity < intensity_mean - 2*intensity_std│
│                       ) / len(near_field)                         │
│      // 3. 远距回波评估                                           │
│      max_effective_range = percentile(points.range, 0.95)         │
│      range_score = clamp(max_effective_range / 80.0, 0.0, 1.0)   │
│      // 4. 衰减指数                                               │
│      slope, intercept = linear_fit(log(intensity) ~ range)       │
│      attenuation = clamp(-slope / 0.05, 0.0, 1.0)               │
│      // 5. 综合                                                  │
│      credibility = effective_ratio * 0.30                        │
│                   + (1.0 - scatter_ratio) * 0.25                 │
│                   + range_score * 0.25                           │
│                   + (1.0 - attenuation) * 0.20                   │
│                                                                  │
│    case RADAR:                                                   │
│      // 1. 目标稳定性                                             │
│      target_count_now = len(raw_data.targets)                     │
│      target_count_mean = mean(history.target_counts[-10:])       │
│      stability = 1.0 - abs(target_count_now - target_count_mean) │
│                        / max(target_count_mean, 1)               │
│      // 2. RCS一致性                                             │
│      rcs_variance = variance([t.rcs for t in raw_data.targets])  │
│      rcs_score = 1.0 - clamp(rcs_variance / 100.0, 0.0, 1.0)    │
│      // 3. 多径检测                                              │
│      mirror_count = detect_mirror_targets(raw_data.targets)      │
│      multipath_score = 1.0 - mirror_count / max(target_count_now,1)│
│      // 4. 综合                                                  │
│      credibility = stability * 0.35                              │
│                   + rcs_score * 0.35                             │
│                   + multipath_score * 0.30                       │
│      effective_ratio = target_count_now > 0 ? 1.0 : 0.0         │
│                                                                  │
│    case GAS:                                                     │
│      // 1. 预热检查                                              │
│      uptime_s = now() - raw_data.boot_time                       │
│      warmup_score = uptime_s > 30.0 ? 1.0 : uptime_s / 30.0     │
│      // 2. 基线漂移                                              │
│      baseline = mean(history.values[0:60])                        │
│      drift = abs(raw_data.value - baseline) / baseline           │
│      drift_score = 1.0 - clamp(drift / 0.2, 0.0, 1.0)           │
│      // 3. 交叉干扰 (湿度影响电化学传感器)                         │
│      humidity_factor = raw_data.humidity > 90 ? 0.7              │
│                        : raw_data.humidity > 70 ? 0.85           │
│                        : 1.0                                     │
│      // 4. 综合                                                  │
│      credibility = warmup_score * 0.25                           │
│                   + drift_score * 0.35                           │
│                   + humidity_factor * 0.40                       │
│      effective_ratio = credibility                                │
│                                                                  │
│  return {credibility, effective_ratio, degradation_cause}        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 11.3 动态传感器权重分配算法

```
┌──────────────────────────────────────────────────────────────────┐
│  SensorWeightAllocator::allocate(env_quality, degradation_mode)  │
├──────────────────────────────────────────────────────────────────┤
│ 输入: env_quality: EnvironmentQuality                            │
│       degradation_mode ∈ {CLEAR, LOW_VIS, HEAVY_SMOKE, DEGRADED}│
│ 输出: weights = {rgb, thermal, lidar, radar, lidar2d, gas}      │
│       depth_priority: 排序后的深度源列表                           │
│ 复杂度: O(1)                                                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  // ===== 步骤1: 加载基准权重(查表) =====                           │
│  static const BASE_WEIGHTS = {                                   │
│    CLEAR:     {0.90, 0.40, 0.85, 0.30, 0.20, 0.20},            │
│    LOW_VIS:   {0.50, 0.80, 0.60, 0.50, 0.30, 0.40},            │
│    HEAVY_SMOKE:{0.10, 0.85, 0.25, 0.85, 0.50, 0.60},           │
│    DEGRADED:  {0.00, 0.30, 0.00, 0.80, 0.60, 0.60}             │
│  }                                                               │
│  base = BASE_WEIGHTS[degradation_mode]                           │
│                                                                  │
│  // ===== 步骤2: 根据实时传感器可信度动态调节 =====                   │
│  cred = {                                                        │
│    rgb:    env_quality.rgb_credibility,                          │
│    thermal:env_quality.thermal_credibility,                      │
│    lidar:  env_quality.lidar_credibility,                        │
│    radar:  env_quality.radar_credibility,                        │
│    lidar2d:env_quality.lidar_credibility * 0.8,  // 2D比3D鲁棒   │
│    gas:    env_quality.gas_credibility                           │
│  }                                                               │
│                                                                  │
│  // 动态修正因子: 可信度低于0.3时大幅降权                           │
│  for each sensor s:                                              │
│    if cred[s] < 0.1:                                             │
│      factor[s] = 0.0          // 传感器基本失效，权重清零          │
│    elif cred[s] < 0.3:                                            │
│      factor[s] = cred[s] / 0.3  // 线性缩放 [0, 1]               │
│    else:                                                          │
│      factor[s] = 1.0          // 正常                             │
│                                                                  │
│  // ===== 步骤3: 计算最终权重 =====                                 │
│  for each sensor s:                                              │
│    weights[s] = base[s] * factor[s]                              │
│                                                                  │
│  // ===== 步骤4: 退化模式下特殊处理 =====                            │
│  if degradation_mode >= HEAVY_SMOKE:                             │
│    // 重烟下 RGB 几乎不可用，强制降权                               │
│    weights.rgb *= 0.3                                            │
│    // 提升毫米波权重 (可穿透烟雾)                                   │
│    weights.radar = max(weights.radar, 0.60)                      │
│    // 提升气体传感器权重 (环境风险感知关键)                          │
│    weights.gas = max(weights.gas, 0.50)                          │
│                                                                  │
│  if degradation_mode == DEGRADED:                                │
│    // 仅保留可穿透烟雾的传感器                                      │
│    weights.lidar = 0.0                                           │
│    weights.rgb = 0.0                                             │
│    weights.thermal = weights.thermal * factor[thermal]           │
│    weights.radar = max(weights.radar, 0.70)                      │
│                                                                  │
│  // ===== 步骤5: 归一化 (可选，保留绝对大小也可) =====               │
│  sum_w = sum(weights)                                            │
│  if sum_w > 0:                                                   │
│    for each sensor s: weights[s] /= sum_w                         │
│                                                                  │
│  // ===== 步骤6: 深度源优先级排序 =====                              │
│  depth_sources = [                                               │
│    {name:"lidar",         score: cred.lidar * 1.0},              │
│    {name:"stereo",        score: cred.rgb * 0.5},                │
│    {name:"radar_range",   score: cred.radar * 0.9},              │
│    {name:"laser_range",   score: cred.lidar2d * 0.8},            │
│    {name:"tof",           score: cred.lidar2d * 0.6},            │
│    {name:"ground_project",score: 0.3}                            │
│  ]                                                               │
│  sort depth_sources by score descending                          │
│                                                                  │
│  return {weights, depth_sources}                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 11.4 退化模式状态机算法（带滞后）

```
┌──────────────────────────────────────────────────────────────────┐
│  DegradationStateMachine::update(env_quality, smoke_estimate)    │
├──────────────────────────────────────────────────────────────────┤
│ 输入: env_quality: 当前环境质量                                   │
│       smoke_estimate: 综合烟雾估计                                │
│ 输出: new_mode ∈ {CLEAR, LOW_VISIBILITY, HEAVY_SMOKE, DEGRADED} │
│       mode_changed: bool                                         │
│       behavior_recommendation: BehaviorRecommendation            │
│ 复杂度: O(1)                                                     │
│ 关键设计: 滞后(Hysteresis), 进入条件 ≠ 退出条件, 防止模式振荡       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  // ===== 滞后因子 =====                                           │
│  const HYSTERESIS = 0.20   // 退出条件比进入条件宽松20%            │
│                                                                  │
│  current = this->current_mode                                    │
│  smoke = smoke_estimate.smoke_score                              │
│  rgb_c = env_quality.rgb_credibility                             │
│  lidar_c = env_quality.lidar_credibility                         │
│  thermal_c = env_quality.thermal_credibility                     │
│  radar_c = env_quality.radar_credibility                         │
│  brightness = env_quality.illuminance_lux                        │
│                                                                  │
│  // ===== 状态转移判断 =====                                        │
│  switch (current):                                               │
│                                                                  │
│    case CLEAR:                                                   │
│      // 进入 LOW_VIS 的条件                                       │
│      if (smoke > 0.15)                                          │
│         OR (rgb_c < 0.50)                                       │
│         OR (brightness < 30):                                    │
│        new_mode = LOW_VISIBILITY                                 │
│      else:                                                       │
│        new_mode = CLEAR                                          │
│                                                                  │
│    case LOW_VISIBILITY:                                          │
│      // 进入 HEAVY_SMOKE 的条件                                   │
│      if (smoke > 0.35)                                          │
│         OR (rgb_c < 0.20 AND lidar_c < 0.30):                   │
│        new_mode = HEAVY_SMOKE                                    │
│      // 回退到 CLEAR 的条件 (滞后: 需要 smoke < 0.10)             │
│      elif (smoke < 0.10)                                        │
│           AND (rgb_c > 0.70)                                    │
│           AND (brightness > 50):                                 │
│        new_mode = CLEAR                                          │
│      else:                                                       │
│        new_mode = LOW_VISIBILITY                                 │
│                                                                  │
│    case HEAVY_SMOKE:                                             │
│      // 进入 DEGRADED 的条件（多传感器失效）                        │
│      if (rgb_c < 0.10 AND lidar_c < 0.20 AND thermal_c < 0.30) │
│         OR (sensor_health.any(s => s.status == FAILED)):         │
│        new_mode = PERCEPTION_DEGRADED                            │
│      // 回退到 LOW_VIS 的条件 (滞后: 需要 smoke < 0.25)            │
│      elif (smoke < 0.25)                                        │
│           AND (rgb_c > 0.30)                                    │
│           AND (lidar_c > 0.30):                                  │
│        new_mode = LOW_VISIBILITY                                 │
│      else:                                                       │
│        new_mode = HEAVY_SMOKE                                    │
│                                                                  │
│    case PERCEPTION_DEGRADED:                                     │
│      // 严重退化下的恢复条件（需要至少两种传感器可用）               │
│      reliable_sensors = count([thermal_c, radar_c, lidar_c]      │
│                               where _ > 0.50)                    │
│      if reliable_sensors >= 2                                    │
│         AND smoke < 0.50:                                        │
│        new_mode = HEAVY_SMOKE                                    │
│      elif reliable_sensors >= 3                                  │
│           AND smoke < 0.30:                                      │
│        new_mode = LOW_VISIBILITY                                 │
│      else:                                                       │
│        new_mode = PERCEPTION_DEGRADED                            │
│                                                                  │
│  // ===== 模式变化时生成行为建议 =====                               │
│  if new_mode != current:                                         │
│    mode_changed = true                                           │
│    // 振荡保护: 如果3秒内切换超过2次，锁定当前模式3秒                 │
│    if oscillation_detected():                                    │
│      new_mode = current                                          │
│      mode_changed = false                                        │
│                                                                  │
│  behavior = {                                                    │
│    CLEAR:      {max_speed: 1.00, confirm: false, auto: true},   │
│    LOW_VIS:    {max_speed: 0.80, confirm: false, auto: true},   │
│    HEAVY_SMOKE:{max_speed: 0.30, confirm: true,  auto: true},   │
│    DEGRADED:   {max_speed: 0.10, confirm: true,  auto: false,   │
│                  suggest_teleop: true}                           │
│  }[new_mode]                                                     │
│                                                                  │
│  return {new_mode, mode_changed, behavior}                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 11.5 热风险等级计算算法

```
┌──────────────────────────────────────────────────────────────────┐
│  ThermalRiskLevel::compute(temperature_matrix, history_30frames) │
├──────────────────────────────────────────────────────────────────┤
│ 输入: temp_matrix[640][512]: 14-bit辐射测温矩阵(°C)               │
│       history: 最近30帧的温度矩阵和连通域历史                       │
│ 输出: risk_level ∈ {0,1,2,3,4}                                  │
│       hotspots[]: 高温区域列表                                    │
│       max_temperature: 全局最高温                                  │
│       confidence: 风险等级置信度                                   │
│ 复杂度: O(W*H) for thresholding + O(K) for connected components  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  // ===== 步骤1: 预处理 =====                                      │
│  // 坏点修复                                                      │
│  temp_fixed = median_filter_3x3(temp_matrix,                      │
│      condition = |pixel - median(neighbors)| > 20°C)             │
│  // 计算自适应背景基线                                             │
│  bg_baseline = rolling_min(temp_fixed, window=30frames)           │
│  T_bg = mean(bg_baseline[central_50_percent])                    │
│                                                                  │
│  // ===== 步骤2: 高温区域分割 =====                                 │
│  // 自适应阈值: 背景 + max(30°C, 3倍背景标准差)                     │
│  bg_std = std(bg_baseline)                                       │
│  threshold = T_bg + max(30.0, 3.0 * bg_std)                      │
│  hot_mask = temp_fixed > threshold                                │
│                                                                  │
│  // 连通域分析 (8邻域区域生长)                                      │
│  components = connected_components_8(hot_mask)                     │
│  hotspots = []                                                    │
│  for each comp in components:                                     │
│    if comp.area < 9:  // 最小面积过滤                              │
│      continue                                                     │
│                                                                   │
│    region = temp_fixed[comp.mask]                                 │
│    hotspot = {                                                    │
│      T_max:     max(region),                                     │
│      T_mean:    mean(region),                                    │
│      T_min:     min(region),                                     │
│      area_px:   comp.area,                                        │
│      perimeter: comp.perimeter,                                   │
│      centroid:  comp.centroid,                                    │
│      // 圆形度: 4π * area / perimeter² (火焰低圆形度)              │
│      circularity: 4*PI*comp.area / (comp.perimeter^2),            │
│      // 温度边缘梯度                                              │
│      edge_gradient: max_gradient_at_boundary(temp_fixed, comp)    │
│    }                                                              │
│                                                                   │
│    // ===== 步骤3: 时序分析 =====                                   │
│    // 在历史中查找匹配的连通域 (空间IoU > 0.3)                       │
│    matched_history = find_matching_hotspots(                      │
│                        hotspot.centroid, hotspot.area_px, history) │
│    if matched_history.found:                                      │
│      // 温升速率                                                  │
│      hotspot.dT_dt = (hotspot.T_max - matched_history.T_max)     │
│                      / matched_history.delta_t                    │
│      // 面积增长率                                                │
│      hotspot.dA_dt = (hotspot.area_px - matched_history.area_px) │
│                      / (matched_history.area_px * matched_history.delta_t)│
│      // 持续时间                                                  │
│      hotspot.duration = matched_history.duration + frame_interval │
│    else:                                                          │
│      hotspot.dT_dt = 0.0                                          │
│      hotspot.dA_dt = 0.0                                          │
│      hotspot.duration = 0.0                                       │
│                                                                   │
│    // ===== 步骤4: 热点分类 =====                                    │
│    hotspot.type = classify_hotspot(hotspot):                      │
│      if T_max > 200 AND dT_dt > 5.0 AND dA_dt > 0.10             │
│         AND circularity < 0.70:                                   │
│        → "fire_source"        // 明火 (置信度 0.85+)               │
│      elif T_max > 150 AND dT_dt > 3.0 AND duration > 3.0:        │
│        → "fire_suspected"     // 疑似火源 (置信度 0.60+)           │
│      elif T_max > 80 AND circularity > 0.80 AND dT_dt < 1.0:     │
│        → "hot_equipment"      // 热设备 (置信度 0.75+)             │
│      elif T_max > 50 AND dT_dt < 0.5 AND circularity > 0.85:     │
│        → "warm_surface"       // 温暖表面                          │
│      else:                                                        │
│        → "hot_anomaly"        // 一般热异常 (置信度 0.50+)         │
│                                                                   │
│    hotspots.append(hotspot)                                       │
│                                                                   │
│  // ===== 步骤5: 计算综合热风险等级 =====                            │
│  max_temp = max([h.T_max for h in hotspots] + [T_bg])            │
│  fire_count = count([h for h in hotspots                          │
│                      if h.type in ("fire_source","fire_suspected")])│
│                                                                   │
│  if max_temp > 500 OR fire_count >= 3:                            │
│    risk_level = 4   // 致命: 极高温度或大面积火情                   │
│  elif max_temp > 300 OR fire_count >= 2:                          │
│    risk_level = 3   // 危险: 高温或多处火情                        │
│  elif max_temp > 150 OR fire_count >= 1:                          │
│    risk_level = 2   // 警告: 存在火情风险                          │
│  elif max_temp > T_bg + 30:                                       │
│    risk_level = 1   // 关注: 温度异常                              │
│  else:                                                             │
│    risk_level = 0   // 正常                                        │
│                                                                   │
│  // 置信度: 有效热点越多，置信度越高                                  │
│  confidence = clamp(fire_count * 0.3 + len(hotspots) * 0.1,       │
│                     0.0, 1.0)                                     │
│                                                                   │
│  return {risk_level, hotspots, max_temp, confidence}              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

### 11.6 综合烟雾指数计算算法

```
┌──────────────────────────────────────────────────────────────────┐
│  SmokeScore::compute(inputs, sensor_health, degradation_mode)    │
├──────────────────────────────────────────────────────────────────┤
│ 输入: inputs = {image_smoke_prob, coverage_ratio, pm_now,        │
│                  pm_baseline_30s, co_now, co_baseline_60s,       │
│                  lidar_effective_ratio, lidar_max_range,         │
│                  near_field_scatter_ratio}                       │
│       sensor_health: 各传感器健康状态                              │
│       degradation_mode: 当前退化模式                              │
│ 输出: smoke_score ∈ [0.0, 1.0]                                  │
│       visibility_level ∈ {CLEAR, LIGHT_SMOKE, HEAVY_SMOKE, BLIND}│
│       individual_scores: 各因子分数                                │
│       motion_direction: 烟雾运动方向向量                           │
│ 复杂度: O(1)                                                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  // ===== 步骤1: 计算四个子因子 =====                                │
│                                                                  │
│  // 因子1: 图像烟雾概率                                           │
│  f_image = inputs.image_smoke_prob * inputs.coverage_ratio        │
│  // clamp to [0, 1]                                              │
│  f_image = clamp(f_image, 0.0, 1.0)                              │
│                                                                  │
│  // 因子2: PM变化率                                               │
│  if inputs.pm_baseline_30s > 0:                                  │
│    f_pm = (inputs.pm_now - inputs.pm_baseline_30s)                │
│          / inputs.pm_baseline_30s                                 │
│    f_pm = clamp(f_pm, 0.0, 1.0)                                  │
│  else:                                                            │
│    f_pm = inputs.pm_now > 50 ? min(inputs.pm_now / 500.0, 1.0) : 0│
│                                                                  │
│  // 因子3: LiDAR衰减指数                                          │
│  f_lidar = (1.0 - inputs.lidar_effective_ratio)                  │
│          + 0.5 * inputs.near_field_scatter_ratio                  │
│          + 0.3 * (1.0 - inputs.lidar_max_range / 80.0)           │
│  f_lidar = clamp(f_lidar / 1.8, 0.0, 1.0)  // 归一化             │
│                                                                  │
│  // 因子4: CO变化率                                               │
│  if inputs.co_baseline_60s > 0:                                  │
│    f_co = (inputs.co_now - inputs.co_baseline_60s)                │
│          / inputs.co_baseline_60s                                 │
│    f_co = clamp(f_co, 0.0, 1.0)                                  │
│  else:                                                            │
│    f_co = inputs.co_now > 10 ? min(inputs.co_now / 500.0, 1.0) : 0│
│                                                                  │
│  // ===== 步骤2: 基准权重 =====                                     │
│  w_image = 0.30, w_pm = 0.25, w_lidar = 0.30, w_co = 0.15       │
│                                                                  │
│  // ===== 步骤3: 动态权重调节 =====                                  │
│  // 根据传感器健康状态动态降权                                      │
│  if sensor_health.rgb.status >= DEGRADED:                        │
│    w_image *= 0.5    // 可见光退化 → 图像证据不可靠                 │
│                                                                  │
│  if sensor_health.pm.status >= DEGRADED:                         │
│    w_pm *= 0.3        // PM传感器污染/故障                         │
│                                                                  │
│  if sensor_health.lidar.status >= DEGRADED:                      │
│    w_lidar *= 0.3     // LiDAR已被烟雾严重影响                     │
│                                                                  │
│  if sensor_health.gas.status >= DEGRADED:                        │
│    w_co *= 0.2        // CO传感器故障                              │
│                                                                  │
│  // 退化模式下额外调整                                              │
│  if degradation_mode >= HEAVY_SMOKE:                             │
│    w_image *= 0.3     // 重烟下图像几乎不可用                       │
│    w_lidar *= 0.5     // LiDAR也受影响                             │
│    w_pm *= 1.5        // PM权重提升(物理传感器更可靠)               │
│    w_co *= 1.3         // CO权重提升                               │
│                                                                  │
│  // ===== 步骤4: 归一化权重 =====                                    │
│  sum_w = w_image + w_pm + w_lidar + w_co                         │
│  w_image /= sum_w; w_pm /= sum_w                                 │
│  w_lidar /= sum_w; w_co /= sum_w                                 │
│                                                                  │
│  // ===== 步骤5: 计算综合分数 =====                                  │
│  smoke_score = w_image * f_image + w_pm * f_pm                   │
│              + w_lidar * f_lidar + w_co * f_co                   │
│  smoke_score = clamp(smoke_score, 0.0, 1.0)                      │
│                                                                  │
│  // EMA平滑 (指数移动平均，α=0.3)                                   │
│  smoke_score = 0.3 * smoke_score + 0.7 * previous_smoke_score    │
│                                                                  │
│  // ===== 步骤6: 离散化可见度等级 =====                              │
│  if smoke_score < 0.15:                                          │
│    visibility = CLEAR                                             │
│  elif smoke_score < 0.35:                                        │
│    visibility = LIGHT_SMOKE                                      │
│  elif smoke_score < 0.65:                                        │
│    visibility = HEAVY_SMOKE                                      │
│  else:                                                            │
│    visibility = BLIND                                             │
│                                                                  │
│  // ===== 步骤7: 烟雾运动方向(从3帧掩膜中心位移) =====               │
│  if has_smoke_mask_history(3):                                   │
│    c0 = centroid(smoke_mask[-3])                                 │
│    c2 = centroid(smoke_mask[-1])                                 │
│    motion_direction = normalize(c2 - c0)                         │
│    motion_speed_px = norm(c2 - c0) / (2 * frame_interval)        │
│  else:                                                            │
│    motion_direction = [0, 0]                                     │
│                                                                  │
│  return {                                                         │
│    smoke_score, visibility, motion_direction,                     │
│    individual: {f_image, f_pm, f_lidar, f_co}                    │
│  }                                                                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 11.7 人员多源融合置信度算法

```
┌──────────────────────────────────────────────────────────────────┐
│  PersonConfidenceFusion::compute(candidates, weights, mode)      │
├──────────────────────────────────────────────────────────────────┤
│ 输入: candidates[]: 各传感器的人员候选 (已通过空间关联匹配)         │
│       weights: 当前传感器权重                                      │
│       mode: 退化模式                                              │
│ 输出: fused_confidence ∈ [0.0, 1.0]                              │
│       person_class ∈ {person, person_candidate, unknown}         │
│       confirmed: bool (是否经多传感器确认)                         │
│       source_mask: 确认来源位掩码                                  │
│ 复杂度: O(N) where N = number of candidates                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  // ===== 步骤1: 收集各源证据 =====                                 │
│  evidence = {                                                    │
│    rgb:     {conf: 0, found: false, class: ""},                  │
│    thermal: {conf: 0, found: false, temp_range: [0,0]},         │
│    lidar:   {conf: 0, found: false, dimensions: [0,0,0]},       │
│    radar:   {conf: 0, found: false, velocity: 0, rcs: 0}        │
│  }                                                               │
│                                                                  │
│  for cand in candidates:                                         │
│    switch cand.source:                                            │
│      case RGB:                                                    │
│        evidence.rgb.found = true                                  │
│        evidence.rgb.conf = cand.confidence                        │
│        evidence.rgb.class = cand.pose_class  // standing, etc.    │
│      case THERMAL:                                                │
│        evidence.thermal.found = true                              │
│        evidence.thermal.conf = cand.confidence                    │
│        evidence.thermal.temp_range = cand.temperature_range       │
│      case LIDAR:                                                  │
│        evidence.lidar.found = true                                │
│        evidence.lidar.conf = 0.4  // LiDAR无分类, 基础置信度      │
│        evidence.lidar.dimensions = cand.dimensions                │
│      case RADAR:                                                  │
│        evidence.radar.found = true                                │
│        evidence.radar.conf = 0.5  // 毫米波基础置信度              │
│        evidence.radar.velocity = cand.velocity                    │
│        evidence.radar.rcs = cand.rcs                              │
│                                                                  │
│  // ===== 步骤2: 计算融合置信度 =====                                │
│  // 初始化                                                        │
│  fused_conf = 0.0                                                │
│  conf_weight_sum = 0.0                                           │
│                                                                  │
│  // RGB证据 (受退化模式影响)                                        │
│  if evidence.rgb.found:                                           │
│    rgb_weight = weights.rgb                                       │
│    if mode >= HEAVY_SMOKE:                                        │
│      // 重烟下RGB大幅降权                                         │
│      rgb_weight *= 0.1                                            │
│    fused_conf += rgb_weight * evidence.rgb.conf                   │
│    conf_weight_sum += rgb_weight                                  │
│                                                                  │
│  // 热成像证据 (极端情况下最可靠的来源)                               │
│  if evidence.thermal.found:                                       │
│    thermal_weight = weights.thermal                               │
│    // 热源温度合理性加权                                           │
│    T_mean = mean(evidence.thermal.temp_range)                     │
│    temp_validity = 1.0                                            │
│    if T_mean < 24.0 OR T_mean > 44.0:   // 不在人体温度范围        │
│      temp_validity = 0.3                                          │
│    elif T_mean < 28.0 OR T_mean > 42.0:  // 边界温度               │
│      temp_validity = 0.7                                          │
│    thermal_conf = evidence.thermal.conf * temp_validity           │
│    fused_conf += thermal_weight * thermal_conf                    │
│    conf_weight_sum += thermal_weight                              │
│                                                                  │
│  // LiDAR证据 (形状尺寸验证)                                        │
│  if evidence.lidar.found:                                         │
│    dims = evidence.lidar.dimensions                               │
│    // 人体尺寸合理性: 宽0.3~1.0m, 深0.2~0.8m, 高0.3~2.0m         │
│    size_valid = (0.3 <= dims.w <= 1.0) AND                        │
│                 (0.2 <= dims.d <= 0.8) AND                        │
│                 (0.3 <= dims.h <= 2.0)                            │
│    aspect_valid = dims.h / max(dims.w, 0.1) > 1.2  // 高>宽      │
│    shape_score = size_valid ? (aspect_valid ? 1.0 : 0.7) : 0.3   │
│    lidar_conf = evidence.lidar.conf * shape_score                 │
│    fused_conf += weights.lidar * lidar_conf                       │
│    conf_weight_sum += weights.lidar                               │
│                                                                  │
│  // 毫米波证据 (运动特征)                                           │
│  if evidence.radar.found:                                         │
│    // 人体级RCS: -5 ~ 10 dBsm, 速度 < 2m/s                        │
│    rcs_valid = -5.0 <= evidence.radar.rcs <= 10.0                │
│    vel_valid = abs(evidence.radar.velocity) < 2.0                │
│    micro_doppler = has_micro_doppler(evidence.radar)  // 微动检测  │
│    radar_score = (rcs_valid ? 0.4 : 0.1)                          │
│                + (vel_valid ? 0.3 : 0.1)                          │
│                + (micro_doppler ? 0.3 : 0.0)                      │
│    radar_conf = evidence.radar.conf * radar_score                 │
│    fused_conf += weights.radar * radar_conf                       │
│    conf_weight_sum += weights.radar                               │
│                                                                  │
│  // ===== 步骤3: 归一化 =====                                       │
│  if conf_weight_sum > 0:                                          │
│    fused_conf = fused_conf / conf_weight_sum                      │
│  else:                                                            │
│    fused_conf = 0.0                                               │
│                                                                  │
│  // ===== 步骤4: 多传感器确认判定 =====                              │
│  // 统计多少传感器给出了正面证据 (置信度 > 各自阈值)                   │
│  positive_sources = count([                                      │
│    evidence.rgb.conf > 0.5,                                       │
│    evidence.thermal.conf > 0.3,                                   │
│    evidence.lidar.conf > 0.3 AND shape_score > 0.5,              │
│    evidence.radar.conf > 0.3 AND radar_score > 0.5               │
│  ])                                                               │
│                                                                  │
│  confirmed = positive_sources >= 2                                │
│                                                                  │
│  // ===== 步骤5: 确定人员类别 =====                                  │
│  if confirmed AND fused_conf >= 0.70:                             │
│    person_class = "person"             // 高置信人员               │
│  elif fused_conf >= 0.40:                                          │
│    person_class = "person_candidate"    // 疑似人员,需要确认        │
│  else:                                                            │
│    person_class = "unknown"             // 不确定,记录但不行动       │
│                                                                  │
│  // ===== 步骤6: 极端情况特殊处理 =====                              │
│  if mode >= HEAVY_SMOKE:                                          │
│    // 重烟下降级: 任何有热源+毫米波运动目标的候选                    │
│    if evidence.thermal.found AND evidence.radar.found:             │
│      if evidence.radar.velocity > 0.3:   // 有运动                 │
│        person_class = max(person_class, "person_candidate")       │
│        fused_conf = max(fused_conf, 0.45)                         │
│    // RGB不可信但热源明确 → 至少是候选                              │
│    if evidence.thermal.found                                       │
│       AND evidence.thermal.conf > 0.4                             │
│       AND temp_validity > 0.6:                                    │
│      person_class = max(person_class, "person_candidate")         │
│      fused_conf = max(fused_conf, 0.40)                           │
│                                                                  │
│  // ===== 步骤7: 构建 source_mask =====                             │
│  source_mask = 0                                                  │
│  if evidence.rgb.found:     source_mask |= 0x01  // RGB           │
│  if evidence.thermal.found: source_mask |= 0x02  // THERMAL       │
│  if evidence.lidar.found:   source_mask |= 0x04  // LIDAR         │
│  if evidence.radar.found:   source_mask |= 0x08  // RADAR         │
│                                                                  │
│  return {fused_conf, person_class, confirmed, source_mask}       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 11.8 火源确认决策矩阵算法

```
┌──────────────────────────────────────────────────────────────────┐
│  FireConfirmationMatrix::confirm(evidence, gas_risk, weights)    │
├──────────────────────────────────────────────────────────────────┤
│ 输入: evidence = {rgb_flame_conf, thermal_hotspot, gas_combo,    │
│                    uv_ir_alarm, hotspot_count}                   │
│       gas_risk: 气体风险评估结果                                    │
│       weights: 当前传感器权重                                      │
│ 输出: fire_level ∈ {0=NONE, 1=SUSPECTED, 2=CONFIRMED,            │
│                      3=HIGH_RISK, 4=CRITICAL}                    │
│       confidence ∈ [0.0, 1.0]                                   │
│       should_stop: bool                                           │
│       hot_zone_radius_m: float                                   │
│ 复杂度: O(1) - 纯决策树                                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  // ===== 步骤1: 量化各证据强度 =====                                │
│  quant = {                                                       │
│    rgb:     evidence.rgb_flame_conf,                             │
│    thermal: evidence.thermal_hotspot ?                           │
│               evidence.thermal_hotspot.confidence : 0.0,         │
│    gas:     gas_risk.fire_risk ? 0.7 :                           │
│              (gas_risk.combined_risk >= DANGER ? 0.5 : 0.0),    │
│    uv_ir:   evidence.uv_ir_alarm ? 1.0 : 0.0,                   │
│  }                                                               │
│                                                                  │
│  // ===== 步骤2: 决策树 =====                                       │
│  fire_conf = 0.0                                                  │
│  fire_level = 0                                                   │
│                                                                  │
│  // 等级4: 致命火灾 (最高优先级)                                     │
│  if (quant.thermal > 0.8 AND evidence.hotspot_count >= 3         │
│      AND evidence.max_temp > 500)                                │
│     OR (quant.thermal > 0.7 AND quant.gas > 0.7                   │
│         AND gas_risk.o2_percent < 16.0):                         │
│    fire_level = 4                                                 │
│    fire_conf = 0.95                                               │
│                                                                  │
│  // 等级3: 高危火灾                                                │
│  elif (quant.thermal > 0.7 AND evidence.max_temp > 300           │
│         AND evidence.hotspot_count >= 2)                         │
│       OR (quant.thermal > 0.6 AND quant.gas > 0.6                 │
│           AND evidence.area_m2 > 10.0):                          │
│    fire_level = 3                                                 │
│    fire_conf = 0.85                                               │
│                                                                  │
│  // 等级2: 确认火灾                                                │
│  elif (quant.rgb > 0.7 AND quant.thermal > 0.6)                  │
│       OR (quant.thermal > 0.6 AND quant.gas > 0.5)               │
│       OR (quant.thermal > 0.6 AND quant.uv_ir > 0.9)             │
│       OR (quant.rgb > 0.5 AND quant.gas > 0.5                    │
│            AND quant.thermal > 0.4):                             │
│    fire_level = 2                                                 │
│    fire_conf = 0.75                                               │
│                                                                  │
│  // 等级1: 疑似火灾                                                │
│  elif (quant.rgb > 0.5 AND quant.gas > 0.4)                      │
│       OR (quant.thermal > 0.5)                                   │
│       OR (quant.rgb > 0.6 AND quant.thermal > 0.3):             │
│    fire_level = 1                                                 │
│    fire_conf = 0.50                                               │
│                                                                  │
│  // 等级0: 无火灾                                                  │
│  else:                                                            │
│    fire_level = 0                                                 │
│    fire_conf = 0.0                                                │
│                                                                  │
│  // ===== 步骤3: 传感器权重修正 =====                                │
│  // 高温环境下热成像权重提升                                        │
│  if evidence.max_temp > 150:                                      │
│    fire_conf = fire_conf * 0.7 + quant.thermal * weights.thermal * 0.3 │
│                                                                  │
│  // 气体传感器权重复合                                              │
│  if gas_risk.fire_risk:                                           │
│    fire_conf = min(fire_conf + 0.10, 1.0)                       │
│  if gas_risk.explosion_risk:                                      │
│    fire_conf = min(fire_conf + 0.15, 1.0)                       │
│    fire_level = max(fire_level, 3)                                │
│                                                                  │
│  // ===== 步骤4: 安全建议 =====                                      │
│  should_stop = (fire_level >= 3)                                  │
│             OR (fire_level == 2 AND evidence.max_temp > 300)     │
│                                                                  │
│  // 高温禁区半径 (基于最高温估算)                                     │
│  if evidence.max_temp > 500:                                      │
│    hot_zone_radius_m = 8.0    // 致命半径                         │
│  elif evidence.max_temp > 300:                                    │
│    hot_zone_radius_m = 5.0                                        │
│  elif evidence.max_temp > 150:                                    │
│    hot_zone_radius_m = 3.0                                        │
│  else:                                                            │
│    hot_zone_radius_m = 1.5                                        │
│                                                                  │
│  return {fire_level, fire_conf, should_stop, hot_zone_radius_m}  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 11.9 环境风险融合算法

```
┌──────────────────────────────────────────────────────────────────┐
│  EnvironmentRiskFusion::assess(thermal_risk, gas_risk,           │
│    radar_obstacles, navigation_blocked, depth_quality,           │
│    rgb_quality, degradation_mode)                                │
├──────────────────────────────────────────────────────────────────┤
│ 输入: 各风险评估子模块的输出                                       │
│ 输出: risk_level ∈ {LOW, MEDIUM, HIGH, CRITICAL}                 │
│       risk_factors: 各风险因子贡献                                │
│       recommended_action: string                                 │
│ 复杂度: O(1)                                                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  // ===== 步骤1: 各因子独立量化 =====                                │
│  factors = {                                                     │
│    thermal:  thermal_risk.level / 4.0,    // 归一化到 [0,1]      │
│    gas:      gas_risk.risk_level / 3.0,   // SAFE=0, DEADLY=1   │
│    obstacle: radar_obstacles.nearest_range < 2.0 ? 1.0           │
│              : radar_obstacles.nearest_range < 5.0 ? 0.6         │
│              : radar_obstacles.nearest_range < 10.0 ? 0.3        │
│              : 0.0,                                              │
│    blockage: navigation_blocked ? 0.8 : 0.0,                    │
│    depth:    1.0 - depth_quality,          // 深度失效程度        │
│    rgb:      1.0 - rgb_quality,            // RGB退化程度         │
│    motion:   radar_obstacles.has_moving_target ? 0.5 : 0.0,     │
│  }                                                               │
│                                                                  │
│  // ===== 步骤2: 退化模式调整因子重要性 =====                         │
│  // 极端情况下热风险、障碍和气体权重提升                              │
│  if degradation_mode >= HEAVY_SMOKE:                             │
│    w = {thermal:0.35, gas:0.20, obstacle:0.20, blockage:0.10,   │
│          depth:0.05, rgb:0.05, motion:0.05}                      │
│  elif degradation_mode == LOW_VISIBILITY:                        │
│    w = {thermal:0.25, gas:0.15, obstacle:0.15, blockage:0.10,   │
│          depth:0.15, rgb:0.15, motion:0.05}                      │
│  else:  // CLEAR                                                  │
│    w = {thermal:0.20, gas:0.15, obstacle:0.10, blockage:0.10,   │
│          depth:0.15, rgb:0.20, motion:0.10}                      │
│                                                                  │
│  // ===== 步骤3: 加权聚合 =====                                     │
│  weighted_sum = sum(factors[k] * w[k] for k in factors)          │
│  norm = sum(w[k] for k in w)                                     │
│  combined_score = weighted_sum / norm                            │
│                                                                  │
│  // ===== 步骤4: 最大风险原则 =====                                  │
│  // 某些因子足够危险时，直接提升整体等级                              │
│  max_factor = max(factors.thermal, factors.gas, factors.obstacle)│
│                                                                  │
│  if factors.thermal >= 1.0:     // 热风险4级                      │
│    combined_score = max(combined_score, 0.95)                    │
│  if factors.gas >= 1.0:         // 气体致命                       │
│    combined_score = max(combined_score, 0.90)                    │
│  if factors.obstacle >= 1.0:    // 极近距离障碍                   │
│    combined_score = max(combined_score, 0.85)                    │
│  if factors.blockage > 0:       // 通道阻断                       │
│    combined_score = max(combined_score, 0.70)                    │
│                                                                  │
│  // ===== 步骤5: 离散化 =====                                       │
│  if combined_score >= 0.85:                                       │
│    risk_level = CRITICAL                                          │
│    action = "STOP_AND_WAIT_REMOTE_CONFIRM"                        │
│  elif combined_score >= 0.60:                                     │
│    risk_level = HIGH                                              │
│    action = "LOW_SPEED_WAIT_CONFIRMATION"                         │
│  elif combined_score >= 0.30:                                     │
│    risk_level = MEDIUM                                            │
│    action = "LIMIT_SPEED_AND_LOG"                                 │
│  else:                                                            │
│    risk_level = LOW                                               │
│    action = "NORMAL_OPERATION"                                    │
│                                                                  │
│  // ===== 步骤6: 紧急覆盖规则 =====                                  │
│  // 防撞条/物理急停 --> 无条件最高风险                               │
│  if bumper_triggered OR physical_estop:                           │
│    risk_level = CRITICAL                                          │
│    action = "ESTOP_IMMEDIATE"                                     │
│                                                                  │
│  return {risk_level, combined_score, factors, action}             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 11.10 安全策略映射算法

```
┌──────────────────────────────────────────────────────────────────┐
│  SafetyMapper::map(env_risk, thermal_risk_level,                 │
│    degradation_mode, radar_nearest_range, is_estop)              │
├──────────────────────────────────────────────────────────────────┤
│ 输入: env_risk: 环境风险等级                                      │
│       thermal_risk_level: 热风险 0-4                             │
│       degradation_mode: 退化模式                                  │
│       radar_nearest_range: 毫米波最近障碍距离(m)                   │
│       is_estop: 物理急停信号                                      │
│ 输出: safety_level ∈ {SAFETY_OK, SAFETY_WARNING,                 │
│         SAFETY_DEGRADED, SAFETY_STOP_REQUIRED, SAFETY_ESTOP}     │
│       speed_limit: 速度限制比例 [0.0, 1.0]                        │
│       required_action: string                                    │
│       require_remote_confirm: bool                               │
│ 复杂度: O(1)                                                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  // ===== 优先级从高到低判断 =====                                   │
│                                                                  │
│  // 1. 物理急停（最高优先级，无条件）                                │
│  if is_estop:                                                     │
│    return {SAFETY_ESTOP, speed: 0.0,                              │
│            action: "IMMEDIATE_ESTOP", confirm: false}             │
│                                                                  │
│  // 2. 热风险4级 或 感知完全丢失 或 极近距离障碍                     │
│  if thermal_risk_level >= 4                                       │
│     OR degradation_mode == PERCEPTION_DEGRADED                    │
│     OR radar_nearest_range < 1.0:                                │
│    return {SAFETY_STOP_REQUIRED, speed: 0.0,                      │
│            action: "STOP_AND_WAIT", confirm: true}                │
│                                                                  │
│  // 3. 热风险3级 或 深度大面积失效 或 毫米波近距离障碍              │
│  if thermal_risk_level >= 3                                       │
│     OR env_risk >= HIGH                                           │
│     OR radar_nearest_range < 3.0:                                │
│    return {SAFETY_DEGRADED, speed: 0.20,                          │
│            action: "LOW_SPEED_CAUTIOUS", confirm: true}           │
│                                                                  │
│  // 4. 热风险2级 或 感知退化 或 毫米波中距离障碍                    │
│  if thermal_risk_level >= 2                                       │
│     OR degradation_mode >= LOW_VISIBILITY                         │
│     OR radar_nearest_range < 5.0:                                │
│    return {SAFETY_WARNING, speed: 0.50,                           │
│            action: "REDUCED_SPEED", confirm: false}               │
│                                                                  │
│  // 5. 正常                                                        │
│  return {SAFETY_OK, speed: 1.0,                                   │
│          action: "NORMAL_OPERATION", confirm: false}              │
│                                                                  │
│  // ===== 补充规则 =====                                            │
│  // 气体高风险默认不直接限速，但触发远程告警和任务确认                  │
│  // (在 task_manager 层处理，不在此处)                              │
│  //                                                               │
│  // 通道阻断由 Nav2 自行处理绕行, 此处仅反映在导航状态中               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 11.11 目标坐标融合与深度有效性判断算法

```
┌──────────────────────────────────────────────────────────────────┐
│  DepthValidator::validate(bbox, depth_map, point_cloud,          │
│    radar_range, degradation_mode)                                │
├──────────────────────────────────────────────────────────────────┤
│ 输入: bbox: 检测框 (图像坐标)                                     │
│       depth_map: 深度图 (如可用)                                   │
│       point_cloud: LiDAR点云 (如可用)                              │
│       radar_range: 毫米波距离 (如可用)                              │
│       degradation_mode: 当前退化模式                              │
│ 输出: position_3d: (x, y, z) 估计三维位置                         │
│       position_covariance: 协方差矩阵 (6x6)                        │
│       depth_valid: bool                                           │
│       bearing_angle: 方位角 (即使深度无效也能提供)                   │
│       elevation_angle: 仰角                                       │
│ 复杂度: O(K) where K = points in bbox region                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  // ===== 步骤1: 按优先级尝试深度源 =====                            │
│                                                                  │
│  // 源1: LiDAR点云投影 (最精确)                                     │
│  if point_cloud is valid AND degradation_mode < HEAVY_SMOKE:     │
│    // 取检测框中心区域对应的点云                                    │
│    roi_center = bbox_center_region(bbox, ratio=0.3)              │
│    points_3d = project_to_pointcloud(roi_center, point_cloud)     │
│    // 过滤无效点                                                  │
│    valid_points = filter(points_3d,                              │
│      p => not isnan(p.z) AND p.z > 0.5 AND p.z < 80.0)          │
│    valid_ratio = len(valid_points) / len(points_3d)              │
│                                                                  │
│    if valid_ratio > 0.3 AND len(valid_points) >= 5:              │
│      // 使用30%分位数深度 (比中值更抗噪)                             │
│      depth_30pct = percentile(valid_points.z, 0.30)              │
│      position_3d = back_project(bbox_center, depth_30pct,         │
│                                  camera_intrinsics)               │
│      position_covariance = compute_covariance(valid_points)       │
│      depth_valid = true                                           │
│      goto output                                                  │
│                                                                  │
│  // 源2: 深度图 (双目相机)                                          │
│  if depth_map is valid AND degradation_mode < HEAVY_SMOKE:       │
│    roi_depth = depth_map[bbox_center_region]                      │
│    valid_depths = filter(roi_depth,                               │
│      d => d > 0 AND d < 80.0 AND not isnan(d))                   │
│    valid_ratio = len(valid_depths) / len(roi_depth)              │
│                                                                  │
│    if valid_ratio > 0.3:                                          │
│      depth_median = median(valid_depths)                          │
│      position_3d = back_project(bbox_center, depth_median,        │
│                                  camera_intrinsics)               │
│      position_covariance = diag(0.1, 0.1, 0.05)  // 粗略估计      │
│      depth_valid = true                                           │
│      goto output                                                  │
│                                                                  │
│  // 源3: 毫米波雷达距离 (浓烟环境首选)                               │
│  if radar_range is valid:                                         │
│    // 毫米波给出的是径向距离, 配合检测框方位角                       │
│    bearing = pixel_to_angle(bbox_center.x, image_width, hfov)    │
│    elevation = pixel_to_angle(bbox_center.y, image_height, vfov)  │
│    position_3d = spherical_to_cartesian(                          │
│                    radar_range, bearing, elevation)               │
│    // 毫米波角度分辨率有限, 协方差较大                               │
│    position_covariance = diag(                                    │
│      0.3, 0.3, 0.15)  // 较大不确定性                             │
│    depth_valid = true                                             │
│    goto output                                                    │
│                                                                  │
│  // 源4: 地面投影估计 (最后手段)                                     │
│  // 假设目标站立在地面上, 利用相机高度和检测框底部推算距离              │
│  bearing = pixel_to_angle(bbox_center.x, image_width, hfov)      │
│  elevation = pixel_to_angle(bbox.bottom, image_height, vfov)     │
│  // 地面投影: d = camera_height / tan(elevation)                   │
│  ground_range = camera_height_m / tan(abs(elevation) + 1e-6)     │
│  position_3d = spherical_to_cartesian(                            │
│                  ground_range, bearing, elevation)                │
│  position_covariance = diag(                                      │
│    2.0, 2.0, 0.5)  // 很大不确定性                                │
│  depth_valid = false  // 标记为低可信坐标                           │
│  // 不输出精确坐标, 仅输出方位和粗略距离估计                          │
│                                                                  │
│  // ===== 步骤2: 协方差传播 (TF变换后) =====                         │
│  label output:                                                    │
│  if depth_valid:                                                  │
│    // 增加传感器噪声贡献                                           │
│    covariance *= (1.0 + 0.5 * (1.0 - valid_ratio))               │
│    // 退化模式下额外扩大协方差                                      │
│    if degradation_mode >= HEAVY_SMOKE:                            │
│      covariance *= 3.0                                            │
│    elif degradation_mode >= LOW_VISIBILITY:                       │
│      covariance *= 1.5                                            │
│                                                                  │
│  return {                                                         │
│    position_3d, position_covariance, depth_valid,                │
│    bearing_angle: bearing, elevation_angle: elevation            │
│  }                                                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 11.12 气体风险综合评估算法

```
┌──────────────────────────────────────────────────────────────────┐
│  GasRiskAssessor::assess(gas_readings, history_60s,              │
│    ambient_temp, ambient_humidity)                               │
├──────────────────────────────────────────────────────────────────┤
│ 输入: gas_readings: {co_ppm, co2_ppm, o2_pct, ch4_lel, h2s_ppm, │
│                       voc_ppm, pm25, pm10}                       │
│       history_60s: 最近60秒各气体读数序列                          │
│       ambient_temp: 环境温度                                      │
│       ambient_humidity: 环境湿度                                  │
│ 输出: risk_level: GAS_SAFE / GAS_CAUTION / GAS_DANGER / GAS_DEADLY│
│       individual_levels: 各气体的独立风险等级                       │
│       fire_signature: bool (CO↑ + CO₂↑ + O₂↓ + Temp↑)           │
│       explosion_risk: bool                                        │
│       asphyxiation_risk: bool                                     │
│       timestamp: 时间戳                                           │
│ 复杂度: O(1)                                                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  // ===== 步骤1: 湿度和温度修正 =====                                │
│  // 高湿度影响电化学传感器精度                                       │
│  humidity_correction = 1.0                                        │
│  if ambient_humidity > 90.0:                                      │
│    humidity_correction = 0.70  // 严重交叉干扰                     │
│  elif ambient_humidity > 70.0:                                    │
│    humidity_correction = 0.85                                     │
│                                                                  │
│  // 温度对气体传感器基线的影响                                       │
│  temp_correction = 1.0 + max(0, (ambient_temp - 40.0) * 0.01)   │
│                                                                  │
│  corrected_co  = gas_readings.co_ppm * humidity_correction        │
│                                                   * temp_correction│
│  corrected_h2s = gas_readings.h2s_ppm * humidity_correction       │
│                                                                  │
│  // ===== 步骤2: 各气体独立评估 =====                                │
│  individual = {}                                                  │
│                                                                  │
│  // CO评估                                                        │
│  co = corrected_co                                                │
│  if co < 50:       individual.co = SAFE                           │
│  elif co < 200:    individual.co = CAUTION                        │
│  elif co < 1200:   individual.co = DANGER                         │
│  else:             individual.co = DEADLY                         │
│                                                                  │
│  // CO₂评估                                                       │
│  co2 = gas_readings.co2_ppm                                       │
│  if co2 < 1000:    individual.co2 = SAFE                          │
│  elif co2 < 5000:  individual.co2 = CAUTION                       │
│  else:             individual.co2 = DANGER                        │
│                                                                  │
│  // O₂评估 (缺氧)                                                  │
│  o2 = gas_readings.o2_pct                                         │
│  if o2 > 19.5:     individual.o2 = SAFE                           │
│  elif o2 > 16.0:   individual.o2 = CAUTION                        │
│  elif o2 > 10.0:   individual.o2 = DANGER                         │
│  else:             individual.o2 = DEADLY                         │
│                                                                  │
│  // CH₄评估 (爆炸风险)                                              │
│  ch4 = gas_readings.ch4_lel                                       │
│  if ch4 < 1.0:     individual.ch4 = SAFE                          │
│  elif ch4 < 10.0:  individual.ch4 = CAUTION                       │
│  elif ch4 < 25.0:  individual.ch4 = DANGER                        │
│  else:             individual.ch4 = DEADLY                        │
│                                                                  │
│  // H₂S评估                                                        │
│  h2s = corrected_h2s                                              │
│  if h2s < 10:      individual.h2s = SAFE                          │
│  elif h2s < 50:    individual.h2s = CAUTION                       │
│  elif h2s < 100:   individual.h2s = DANGER                        │
│  else:             individual.h2s = DEADLY                        │
│                                                                  │
│  // ===== 步骤3: 综合风险 = max(各独立风险) =====                    │
│  risk_level = max(individual.co, individual.co2,                  │
│                   individual.o2, individual.ch4, individual.h2s)  │
│                                                                  │
│  // ===== 步骤4: 组合风险检测 =====                                  │
│                                                                  │
│  // 火灾签名: CO快速上升 + CO₂快速上升 + O₂下降 + 温度上升            │
│  co_trend = linear_slope(history_60s.co[-10:])                   │
│  co2_trend = linear_slope(history_60s.co2[-10:])                  │
│  o2_trend = linear_slope(history_60s.o2[-10:])                    │
│  temp_trend = linear_slope(history_60s.temp[-10:])                │
│                                                                  │
│  fire_signature = (co_trend > 5.0) AND (co2_trend > 50.0)       │
│                   AND (o2_trend < -0.1) AND (temp_trend > 2.0)   │
│                                                                  │
│  // 爆炸风险: CH₄ > 10% LEL                                       │
│  explosion_risk = gas_readings.ch4_lel >= 10.0                   │
│                                                                  │
│  // 窒息风险: O₂ < 19.5% OR CO > 200ppm                           │
│  asphyxiation_risk = (gas_readings.o2_pct < 19.5)                │
│                      OR (corrected_co > 200.0)                    │
│                                                                  │
│  // ===== 步骤5: 告警输出决策 =====                                  │
│  if risk_level >= DANGER OR fire_signature:                       │
│    alert = {                                                     │
│      type: "GAS_DANGER_ALERT",                                    │
│      remote_notify: true,        // 远程告警                      │
│      mission_confirm: true,      // 需要任务层确认                 │
│      auto_retreat: false         // 默认不自动撤退                 │
│    }                                                             │
│  elif risk_level >= CAUTION:                                      │
│    alert = {                                                     │
│      type: "GAS_CAUTION_ALERT",                                   │
│      remote_notify: true,                                         │
│      mission_confirm: false,                                      │
│      auto_retreat: false                                          │
│    }                                                             │
│  else:                                                            │
│    alert = {type: "GAS_NORMAL", remote_notify: false,            │
│             mission_confirm: false, auto_retreat: false}          │
│                                                                  │
│  return {risk_level, individual, fire_signature,                  │
│          explosion_risk, asphyxiation_risk, alert}               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 十二、端到端算法实现流程

本章以时序流程图方式描述完整的数据处理链路，从传感器原始数据输入到最终的安全决策输出。

---

### 12.1 总体数据流拓扑图

```
                        ┌─────────────────────────┐
                        │   Robot Sensors Layer    │
                        └───────────┬─────────────┘
                                    │
        ┌───────────┬───────────────┼───────────────┬───────────┐
        ▼           ▼               ▼               ▼           ▼
   ┌─────────┐ ┌─────────┐   ┌───────────┐   ┌─────────┐ ┌─────────┐
   │RGB Cam  │ │Thermal  │   │3D LiDAR   │   │4D Radar │ │Gas/PM   │
   │1080P@30 │ │640x512  │   │RS-Helios  │   │ARS548   │ │Sensors  │
   └────┬────┘ └────┬────┘   └─────┬─────┘   └────┬────┘ └────┬────┘
        │           │              │              │           │
        ▼           ▼              ▼              ▼           ▼
   ╔═══════════════════════════════════════════════════════════════╗
   ║              sensor_sync_node (时间同步 + 运动补偿)             ║
   ╚═══════════════════════════════════════════════════════════════╝
        │           │              │              │           │
        ▼           ▼              ▼              ▼           ▼
   ┌─────────┐ ┌─────────┐   ┌───────────┐   ┌─────────┐ ┌─────────┐
   │rgb_     │ │thermal_ │   │lidar_     │   │radar_   │ │gas_     │
   │detector │ │detector │   │cluster    │   │tracker  │ │risk     │
   └────┬────┘ └────┬────┘   └─────┬─────┘   └────┬────┘ └────┬────┘
        │           │              │              │           │
        │   ┌───────┴──────┐       │              │           │
        │   │smoke_        │       │              │           │
        │   │estimator     │◄──────┘              │           │
        │   └───────┬──────┘                      │           │
        │           │                             │           │
        ▼           ▼              ▼              ▼           ▼
   ╔═══════════════════════════════════════════════════════════════╗
   ║                   一级融合层 (Target-Level Fusion)             ║
   ║  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   ║
   ║  │person_fusion │  │fire_fusion   │  │vehicle_fusion    │   ║
   ║  └──────┬───────┘  └──────┬───────┘  └───────┬──────────┘   ║
   ╚═══════════════════════════╪══════════════════╪═══════════════╝
                               │                  │
                               ▼                  ▼
   ╔═══════════════════════════════════════════════════════════════╗
   ║         semantic_tracker + semantic_localizer                 ║
   ║         (统一跟踪 + 坐标转换到 map 坐标系)                       ║
   ╚═══════════════════════════════════════════════════════════════╝
                               │
                               ▼
   ╔═══════════════════════════════════════════════════════════════╗
   ║                  二级融合层 (Scene-Level Fusion)               ║
   ║  ┌──────────────────┐  ┌──────────────────┐                  ║
   ║  │ risk_assessor    │  │degradation_mgr   │                  ║
   ║  └────────┬─────────┘  └────────┬─────────┘                  ║
   ╚══════════════════╪══════════════╪════════════════════════════╝
                      │              │
                      ▼              ▼
   ╔═══════════════════════════════════════════════════════════════╗
   ║                      输出层 (Output Layer)                     ║
   ║  /perception/target_observations                              ║
   ║  /perception/status         /perception/environment_risk      ║
   ║  /thermal/risk_level        /gas_alert                       ║
   ║  /perception/risk/costmap_overlay                            ║
   ╚═══════════════════════════════════════════════════════════════╝
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │Nav2      │ │Safety    │ │Remote    │
   │Planner   │ │Supervisor│ │Comm Node │
   └──────────┘ └──────────┘ └──────────┘
```

---

### 12.2 正常情况（CLEAR / LOW_VISIBILITY）处理流程

```
┌─ 正常情况主流程 ──────────────────────────────────────────────────┐
│                                                                  │
│  T=0ms: 传感器数据到达                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ sensor_sync_node:                                       │    │
│  │   - ApproximateTimePolicy 对齐 RGB + Thermal + LiDAR    │    │
│  │   - 容忍窗口 10ms                                        │    │
│  │   - 运动补偿 LiDAR 点云 (使用 odometry)                   │    │
│  │   - 发布 /perception/synced/*                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  T=5ms: 独立检测 (并行4路)                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │rgb_      │ │thermal_  │ │lidar_    │ │radar_    │          │
│  │detector  │ │detector  │ │cluster   │ │tracker   │          │
│  │          │ │          │ │          │ │          │          │
│  │YOLO11    │ │temp_thr +│ │RANSAC    │ │Kalman    │          │
│  │ONNX推理  │ │连通域    │ │地面分割  │ │多目标跟踪│          │
│  │          │ │          │ │          │ │          │          │
│  │→person   │ │→hotspots │ │→clusters │ │→tracks   │          │
│  │→vehicle  │ │→person   │ │→quality  │ │→velocity │          │
│  │→flame    │ │  thermal │ │          │ │          │          │
│  │→smoke    │ │          │ │          │ │          │          │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
│       │            │            │            │                  │
│  T=25ms: 目标级融合 (并行3路)                                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ person_fusion_node:                                    │    │
│  │   1. 收集: RGB persons + Thermal persons               │    │
│  │           + LiDAR person_candidates + Radar medium_obj │    │
│  │   2. 关联: 匈牙利匹配 (位置+类别+温度+时间)              │    │
│  │   3. 融合: 动态权重置信度计算                            │    │
│  │   4. 分类: person / person_candidate / unknown         │    │
│  │   5. 定位: LiDAR深度 > 双目深度 > 地面投影              │    │
│  │                                                        │    │
│  │ 权重示例 (CLEAR):                                       │    │
│  │   RGB=0.90, Thermal=0.40, LiDAR=0.85, Radar=0.30      │    │
│  └────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ fire_fusion_node:                                      │    │
│  │   1. 四路确认: RGB火焰 + 热成像hotspot + 气体 + UV/IR   │    │
│  │   2. 决策矩阵 → fire_level 0~4                         │    │
│  │   3. 高温禁区半径计算: f(max_temp)                       │    │
│  └────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ vehicle_fusion_node:                                   │    │
│  │   1. RGB车辆 + LiDAR大聚类 + 毫米波大目标                │    │
│  │   2. 事故状态分类 (ResNet34)                             │    │
│  │   3. 占道评估: footprint ∩ lane → blockage_ratio       │    │
│  └────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  T=35ms: 跟踪 + 坐标转换                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ semantic_tracker:                                       │    │
│  │   - 级联匹配 (优先匹配最近活跃的跟踪)                     │    │
│  │   - 卡尔曼更新: [x,y,z,w,l,h,vx,vy]                   │    │
│  │   - 跟踪管理: tentative(3帧) → confirmed → lost(15帧)   │    │
│  │                                                         │    │
│  │ semantic_localizer:                                     │    │
│  │   - TF2查询: map→odom→base_link→sensor_link            │    │
│  │   - 协方差传播: cov_map = J * cov_sensor * J^T         │    │
│  │   - 发布 RViz MarkerArray                               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  T=45ms: 场景级评估                                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ risk_assessor:                                          │    │
│  │   1. 全局风险 = max(火源风险, 气体风险, 人员风险,         │    │
│  │                      通行阻断, 感知失效)                  │    │
│  │   2. 高温禁区 → Nav2 costmap (LETHAL_OBSTACLE)          │    │
│  │   3. 事件生成: fire_confirmed / person_found / ...      │    │
│  │                                                         │    │
│  │ degradation_manager:                                    │    │
│  │   1. 可见度 = f(smoke_score, rgb_quality, brightness)    │    │
│  │   2. 状态机: CLEAR ⇄ LOW_VIS ⇄ HEAVY_SMOKE ⇄ DEGRADED  │    │
│  │   3. 滞后保护: 退出条件比进入宽松20%                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  T=50ms: 输出发布 (端到端总延迟 ≈ 50ms)                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ publish("/perception/target_observations", ...)          │    │
│  │ publish("/perception/status", "OK")                      │    │
│  │ publish("/perception/environment_risk", "LOW")           │    │
│  │ publish("/thermal/risk_level", 0~2)                      │    │
│  │ publish("/gas_alert", ...)                                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 12.3 极端情况（HEAVY_SMOKE / PERCEPTION_DEGRADED）处理流程

```
┌─ 极端情况主流程 ──────────────────────────────────────────────────┐
│                                                                  │
│  前置条件: degradation_mode ∈ {HEAVY_SMOKE, PERCEPTION_DEGRADED} │
│  核心策略: "红外主风险 + 毫米波辅助 + 气体告警 + RGB降权"           │
│                                                                  │
│  T=0ms: 传感器数据到达                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ sensor_sync_node:                                       │    │
│  │   - 同步容忍窗口放宽到 20ms (传感器数据质量下降)          │    │
│  │   - RGB/深度图跳过运动补偿 (数据不可靠)                   │    │
│  │   - 重点保证 Thermal + Radar + Gas 的同步                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  T=5ms: 独立检测 (RGB降权运行，Thermal + Radar 提升优先级)         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │rgb_detector  │ │thermal_      │ │radar_tracker │            │
│  │(降权模式)    │ │detector      │ │(权重提升)    │            │
│  │              │ │(主风险传感器)│ │              │            │
│  │YOLO仍运行但  │ │              │ │              │            │
│  │仅作低置信候选│ │thermal_risk  │ │radar作为     │            │
│  │              │ │_level 3 or 4 │ │浓烟下主要    │            │
│  │输出带        │ │→ fire/hotspot│ │距离/运动     │            │
│  │"candidate"   │ │  candidate   │ │传感器        │            │
│  │标记          │ │              │ │              │            │
│  │              │ │人体热源候选  │ │运动目标+     │            │
│  │              │ │→ person_     │ │RCS分类+      │            │
│  │              │ │  candidate   │ │微动检测      │            │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘            │
│         │               │               │                       │
│  ┌──────┴───────┐ ┌─────┴──────┐ ┌─────┴──────┐                │
│  │lidar_cluster │ │gas_risk    │ │smoke_      │                │
│  │(退化模式)    │ │(权重提升)  │ │estimator   │                │
│  │              │ │            │ │            │                │
│  │体素下采样    │ │CO/H₂S/CH₄ │ │image权重   │                │
│  │0.2m→0.3m    │ │O₂/CO₂评估 │ │降至0.09    │                │
│  │              │ │            │ │            │                │
│  │聚类容忍度    │ │火灾签名:   │ │LiDAR权重   │                │
│  │0.3m→0.5m    │ │CO↑+CO₂↑   │ │降至0.15    │                │
│  │              │ │+O₂↓+Temp↑ │ │            │                │
│  │仅输出近距离  │ │            │ │PM权重      │                │
│  │大型障碍物    │ │爆炸/窒息   │ │提升至0.375 │                │
│  │              │ │风险评估    │ │CO权重      │                │
│  │              │ │            │ │提升至0.195 │                │
│  └──────┬───────┘ └─────┬──────┘ └─────┬──────┘                │
│         │               │               │                       │
│  T=20ms: 保守融合                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ person_fusion (极端模式):                                │    │
│  │                                                         │    │
│  │ 权重: RGB=0.05, Thermal=0.85, LiDAR=0.10, Radar=0.80   │    │
│  │                                                         │    │
│  │ 决策规则:                                                │    │
│  │   Thermal人体热源 + Radar同方向运动目标                  │    │
│  │     → person_candidate (置信度提升至0.45+)               │    │
│  │                                                         │    │
│  │   Thermal人体热源单独存在                                │    │
│  │     → person_candidate (置信度0.35+)                     │    │
│  │                                                         │    │
│  │   RGB检测到人 但 visibility_level差                      │    │
│  │     → 仅作为低置信度候选，需红外确认                      │    │
│  │                                                         │    │
│  │   Radar微动检测 + Thermal热源                           │    │
│  │     → person_candidate (置信度0.50+)  // 呼吸微动特征    │    │
│  │                                                         │    │
│  │ 输出: 优先输出 person_candidate，而非 person              │    │
│  │       position_covariance 扩大 3x                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ fire_fusion (极端模式):                                  │    │
│  │                                                         │    │
│  │ 决策规则:                                                │    │
│  │   thermal_risk_level >= 3 + hotspot持续存在              │    │
│  │     → fire candidate (主要依据)                          │    │
│  │                                                         │    │
│  │   thermal_risk_level == 4                                │    │
│  │     → safety_supervisor 停车                             │    │
│  │                                                         │    │
│  │   RGB火焰 + 红外高温                                     │    │
│  │     → 提高 fire 置信度                                   │    │
│  │                                                         │    │
│  │   仅RGB火焰 (红外无高温)                                  │    │
│  │     → 低置信度 fire_candidate                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ vehicle_fusion (极端模式):                               │    │
│  │                                                         │    │
│  │   RGB基本不可信 → 依赖 LiDAR + Radar                     │    │
│  │                                                         │    │
│  │   LiDAR大聚类 (>2m 宽度) + Radar大目标 (RCS>10dBsm)     │    │
│  │     → "vehicle_like_obstacle" (诚实标记)                 │    │
│  │                                                         │    │
│  │   事故状态: 标记为 "unknown_accident_state"               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  T=30ms: 深度无效处理                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ depth_valid = false (有效深度点 < 30%)                    │    │
│  │                                                         │    │
│  │ 处理:                                                    │    │
│  │   - 保留 bearing_angle + elevation_angle                 │    │
│  │   - 不生成 position_map (高可信坐标)                       │    │
│  │   - position_covariance 扩大到 3~5x                       │    │
│  │   - 优先使用 毫米波距离 或 LiDAR 近距离回波               │    │
│  │                                                         │    │
│  │ 如果导航模块有毫米波障碍距离                              │    │
│  │   → 发布低可信目标方向候选                                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  T=40ms: 环境风险融合 (保守加权)                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 融合因子 (HEAVY_SMOKE 权重):                             │    │
│  │   thermal_risk   * 0.35  // 主导                          │    │
│  │   gas_risk       * 0.20  // 环境告警                      │    │
│  │   radar_obstacle * 0.20  // 浓烟下关键                    │    │
│  │   nav_blockage   * 0.10                                  │    │
│  │   depth_quality  * 0.05  // 大幅降权                      │    │
│  │   rgb_quality    * 0.05  // 大幅降权                      │    │
│  │   motion_target  * 0.05                                  │    │
│  │                                                         │    │
│  │ combined_score → risk_level:                             │    │
│  │   ≥0.85 → CRITICAL → 停车+远程确认                        │    │
│  │   ≥0.60 → HIGH     → 低速+等待确认                        │    │
│  │   ≥0.30 → MEDIUM   → 限速+记录                            │    │
│  │   <0.30 → LOW      → 低速运行                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  T=45ms: 安全联动                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ safety_mapper:                                           │    │
│  │                                                         │    │
│  │ 热风险4级 → SAFETY_STOP_REQUIRED (零速)                  │    │
│  │ 热风险3级 → SAFETY_DEGRADED (限速20%)                    │    │
│  │ 感知LOST  → SAFETY_STOP_REQUIRED                         │    │
│  │ 感知DEGRADED → SAFETY_WARNING (限速50%)                  │    │
│  │ 毫米波<1m 障碍 → SAFETY_STOP_REQUIRED                    │    │
│  │ 毫米波<3m 障碍 → SAFETY_DEGRADED                          │    │
│  │ 毫米波<5m 障碍 → SAFETY_WARNING                           │    │
│  │                                                         │    │
│  │ 气体异常 → 不直接限速，触发远程告警 + 任务层确认           │    │
│  │ 通道阻断 → Nav2尝试绕行，失败后任务层决策                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  T=50ms: 输出发布 (端到端总延迟 ≈ 50ms)                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ publish("/perception/target_observations",               │    │
│  │   candidates + bearing_only_targets)                     │    │
│  │ publish("/perception/status", "DEGRADED" or "LOST")      │    │
│  │ publish("/perception/environment_risk", "HIGH")           │    │
│  │ publish("/thermal/risk_level", 3 or 4)                    │    │
│  │ publish("/gas_alert", {risk_level, fire_signature, ...}) │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─ 关键差异总结 (正常 vs 极端) ─────────────────────────────┐    │
│  │                                                         │    │
│  │  正常情况:                                               │    │
│  │    RGB 主识别 → person / vehicle / fire                  │    │
│  │    LiDAR/深度 精确定位 → position_map                    │    │
│  │    Thermal 辅助确认                                      │    │
│  │    输出高置信度目标                                       │    │
│  │                                                         │    │
│  │  极端情况:                                               │    │
│  │    Thermal 主风险 → hotspot / fire_candidate             │    │
│  │    Radar 辅助障碍/运动 → vehicle_like_obstacle           │    │
│  │    RGB 降权候选 → person_candidate                       │    │
│  │    深度无效 → bearing_only + 大协方差                     │    │
│  │    输出候选 + 方向 + 保守风险评估                          │    │
│  │    安全优先于任务连续性                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 12.4 状态机联动决策流程

```
┌─ 状态机联动流程 ──────────────────────────────────────────────────┐
│                                                                  │
│  本流程描述感知模块与 Safety Supervisor、Mission Manager、        │
│  Remote Comm Node 之间的联动关系。                                │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 输入: /perception/status, /thermal/risk_level,           │    │
│  │       /perception/environment_risk, /gas_alert,           │    │
│  │       /navigation_blocked, /radar/tracks                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│           ┌───────────────┼───────────────┐                      │
│           ▼               ▼               ▼                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ 巡检模式     │  │ 火情侦察     │  │ 人员搜救     │              │
│  │ PATROL/     │  │ FIRE_RESCUE │  │SEARCH_PERSON│              │
│  │INSPECT_POINT│  │             │  │             │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │               │               │                        │
│         ▼               ▼               ▼                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 任务无关的安全规则 (Always Active)         │    │
│  │                                                         │    │
│  │  /perception/status = LOST                               │    │
│  │    → 停车或暂停任务, 等待远程确认                         │    │
│  │                                                         │    │
│  │  /thermal/risk_level = 4                                 │    │
│  │    → safety_supervisor 停车 (SAFETY_STOP_REQUIRED)       │    │
│  │                                                         │    │
│  │  /gas_alert with risk_level >= DANGER                    │    │
│  │    → 远程告警, mission_manager 等待确认                   │    │
│  │                                                         │    │
│  │  /navigation_blocked = true                              │    │
│  │    → Nav2 尝试绕行, 失败后 mission_manager 决策           │    │
│  │                                                         │    │
│  │  毫米波近距离运动目标 (< 3m)                              │    │
│  │    → 减速 + 红外热源确认人/车候选                         │    │
│  │    → 重烟下: 停车观察, 等待红外确认                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│         ┌─────────────────┼─────────────────┐                    │
│         ▼                 ▼                 ▼                    │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │Safety      │  │Mission       │  │Remote        │            │
│  │Supervisor  │  │Manager       │  │Comm Node     │            │
│  │            │  │              │  │              │            │
│  │决策:       │  │决策:         │  │决策:         │            │
│  │SAFETY_OK   │  │继续当前任务  │  │确认/拒绝     │            │
│  │→原速       │  │切换安全任务  │  │人工接管      │            │
│  │            │  │暂停等待确认  │  │              │            │
│  │SAFETY_WARN │  │标记候选点    │  │远程确认:     │            │
│  │→限速80%    │  │生成救援报告  │  │person_       │            │
│  │            │  │进入FIRE_RESC│  │candidate→    │            │
│  │SAFETY_DEGR │  │进入SEARCH_  │  │person        │            │
│  │→限速30%    │  │PERSON       │  │              │            │
│  │            │  │撤退         │  │fire_candidate│            │
│  │SAFETY_STOP │  │              │  │→ fire        │            │
│  │→零速       │  │              │  │              │            │
│  │            │  │              │  │确认撤退指令  │            │
│  │SAFETY_ESTOP│  │              │  │              │            │
│  │→急停       │  │              │  │              │            │
│  └────────────┘  └──────────────┘  └──────────────┘            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 12.5 退化模式切换与传感器权重跟随流程

```
┌─ 退化模式切换与权重跟随流程 ──────────────────────────────────────┐
│                                                                  │
│  State: CLEAR (全传感器正常)                                     │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ RGB=0.90 Thermal=0.40 LiDAR=0.85 Radar=0.30          │       │
│  │ 深度源: LiDAR > Stereo > Laser > Radar > GroundProj  │       │
│  │ 速度: 100%  输出: person / vehicle / fire            │       │
│  └──────────────────────────────────────────────────────┘       │
│         │                                                        │
│         │ 触发: smoke_score > 0.15 OR rgb_credibility < 0.5       │
│         │       OR brightness < 30                                │
│         ▼                                                        │
│  State: LOW_VISIBILITY (轻度退化)                                 │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ RGB=0.50 Thermal=0.80 LiDAR=0.60 Radar=0.50          │       │
│  │ 深度源: LiDAR > Radar > Laser > Stereo > GroundProj  │       │
│  │ 速度: 80%  输出: person / person_candidate / vehicle │       │
│  └──────────────────────────────────────────────────────┘       │
│         │                                                        │
│         │ 触发: smoke_score > 0.35                                │
│         │       OR (rgb_cred < 0.20 AND lidar_cred < 0.30)       │
│         ▼                                                        │
│  State: HEAVY_SMOKE (重烟退化)                                    │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ RGB=0.03 Thermal=0.85 LiDAR=0.25 Radar=0.85          │       │
│  │ (RGB 基准0.10 * 0.3强制降权 = 0.03)                   │       │
│  │ 深度源: Radar > Laser > ThermalProj > GroundProj      │       │
│  │ 速度: 30%  输出: person_candidate / fire_candidate    │       │
│  │                  hotspot / vehicle_like_obstacle      │       │
│  └──────────────────────────────────────────────────────┘       │
│         │                                                        │
│         │ 触发: rgb_cred < 0.10 AND lidar_cred < 0.20            │
│         │       AND thermal_cred < 0.30                          │
│         │       OR gas_sensor_status = FAILED                     │
│         ▼                                                        │
│  State: PERCEPTION_DEGRADED (感知严重退化)                        │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ RGB=0.00 Thermal=0.30 LiDAR=0.00 Radar=0.80          │       │
│  │ 深度源: Radar > TOF > UWB > GroundProj                │       │
│  │ 速度: 10%  输出: bearing_only targets               │       │
│  │ 行为: 低速撤退, 建议远程接管                          │       │
│  └──────────────────────────────────────────────────────┘       │
│         │                                                        │
│         │ 恢复: thermal_cred > 0.5 AND radar_cred > 0.5          │
│         │       → HEAVY_SMOKE                                     │
│         │ 恢复: 更多传感器恢复                                     │
│         │       → LOW_VISIBILITY → CLEAR                          │
│         ▼                                                        │
│                                                                  │
│  滞后保护机制:                                                    │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ 进入条件                          退出条件             │       │
│  │ CLEAR→LOW_VIS: smoke>0.15        LOW_VIS→CLEAR:      │       │
│  │                                    smoke<0.10 (更严格)│       │
│  │                                                       │       │
│  │ LOW_VIS→HEAVY: smoke>0.35        HEAVY→LOW_VIS:       │       │
│  │                                    smoke<0.25 (更严格)│       │
│  │                                                       │       │
│  │ 振荡保护: 3秒内切换超2次         → 锁定模式3秒        │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 12.6 目标输出决策树（正常 + 极端统一）

```
┌─ 目标输出决策树 ──────────────────────────────────────────────────┐
│                                                                  │
│  输入: 各传感器检测结果 + 环境质量 + 退化模式 + 导航辅助信息        │
│                                                                  │
│  ═══════════════════════════════════════════════════════════════ │
│  PERSON 人员决策树                                                │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                  │
│  RGB_person_conf > 0.6 AND Thermal人体热源?                      │
│    ├─ YES AND degradation_mode < HEAVY_SMOKE                     │
│    │   └─ → "person" (置信度 0.85+)                              │
│    │        source_mask = RGB | THERMAL                           │
│    │                                                             │
│    ├─ YES AND degradation_mode >= HEAVY_SMOKE                     │
│    │   └─ → "person_candidate" (置信度 0.55+)                    │
│    │        (重烟下RGB不可信, 降级输出)                           │
│    │                                                             │
│    └─ NO → 检查其他路径:                                          │
│                                                                  │
│  Thermal人体热源 + Radar同方向运动目标?                           │
│    ├─ YES → "person_candidate" (置信度 0.45+)                     │
│    │        source_mask = THERMAL | RADAR                        │
│    │        需要远程确认                                          │
│    │                                                             │
│    └─ NO → 检查: Thermal单独热源?                                 │
│             ├─ YES AND temp 28~42°C AND 形状合理                   │
│             │   └─ → "person_candidate" (置信度 0.35+)            │
│             │        source_mask = THERMAL                        │
│             │        建议停车观察                                  │
│             │                                                     │
│             └─ NO → 检查: LiDAR person_candidate?                 │
│                      ├─ YES AND 尺寸合理 (h>0.8m, w<1m)            │
│                      │   └─ → "obstacle" (不标记为人员)           │
│                      │                                             │
│                      └─ NO → "unknown" (不输出)                    │
│                                                                  │
│  ═══════════════════════════════════════════════════════════════ │
│  FIRE 火源决策树                                                  │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                  │
│  Thermal hotspot T_max > 200°C AND dT/dt > 5°C/s?               │
│    ├─ YES → "fire_source" (fire_level=2, conf 0.85+)             │
│    │        + 发布 HotZone 到 Nav2 costmap                        │
│    │                                                             │
│    └─ NO → 检查: Thermal hotspot T_max > 150°C?                  │
│             ├─ YES AND dT/dt > 3°C/s AND 持续 > 3s               │
│             │   └─ → "fire_suspected" (fire_level=1, conf 0.60+) │
│             │                                                   │
│             └─ NO → 检查: RGB flame + Thermal?                    │
│                      ├─ YES → "fire" (conf 提升)                  │
│                      ├─ ONLY RGB, NO thermal                     │
│                      │   └─ → "fire_candidate" (低置信度 0.40+)   │
│                      └─ ONLY gas fire_signature                   │
│                          └─ → "fire_suspected" + gas告警          │
│                                                                  │
│  Thermal T_max > 500°C OR 多hotspot                               │
│    └─ → fire_level=4 (CRITICAL) → safety_stop                   │
│                                                                  │
│  ═══════════════════════════════════════════════════════════════ │
│  VEHICLE 车辆决策树                                               │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                  │
│  degradation_mode < HEAVY_SMOKE?                                  │
│    ├─ YES: RGB_vehicle_conf > 0.6                                │
│    │   ├─ + LiDAR 3D框匹配 → "vehicle" (conf 0.80+)              │
│    │   │   + 事故状态分类                                         │
│    │   ├─ + 仅RGB (无LiDAR) → "vehicle" (conf 0.65+)             │
│    │   └─ + 占道评估 → BLOCKING / PARTIAL / CLEAR                │
│    │                                                             │
│    └─ NO (HEAVY_SMOKE+):                                          │
│        ├─ LiDAR大聚类 (w>2m) + Radar大目标 (RCS>10)               │
│        │   └─ → "vehicle_like_obstacle" (conf 0.55+)             │
│        │       事故状态 = "unknown"                                │
│        │       占道评估: 从LiDAR footprint计算                    │
│        │                                                         │
│        ├─ 仅Radar大目标 (无LiDAR)                                 │
│        │   └─ → "large_obstacle" (conf 0.40+)                     │
│        │       仅提供距离和方位                                    │
│        │                                                         │
│        └─ 仅LiDAR中等聚类                                         │
│            └─ → "obstacle_generic" (conf 0.35+)                   │
│                                                                  │
│  共性规则:                                                        │
│    - 占道 > 70% → /navigation_blocked = true                     │
│    - 占道 > 30% → Nav2 costmap 标记                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 十三、核心算法复杂度与性能分析

| 算法模块 | 时间复杂度 | 典型延迟 | GPU加速 | 备注 |
|---|---|---|---|---|
| sensor_sync (时间同步) | O(N_sensors) | <1ms | 否 | N_sensors ≤ 15 |
| rgb_detector (YOLO11×4) | O(HW) | 12~18ms | 是 | TensorRT FP16 批量推理 |
| thermal_detector (连通域) | O(W*H + K) | 3~5ms | 否 | K=连通域数量, 典型 <50 |
| lidar_cluster (欧氏聚类) | O(N_logN) | 5~10ms | 否 | 体素下采样后 N≈30k |
| radar_tracker (卡尔曼) | O(M²) | 2~3ms | 否 | M=目标数, 典型 <20 |
| smoke_estimator | O(1) | <1ms | 否 | 纯数值计算 |
| gas_risk | O(1) | <1ms | 否 | 纯查表 |
| person_fusion (匈牙利) | O(N³) | 3~5ms | 否 | N=候选数, 典型 <30 |
| fire_fusion (决策树) | O(F) | <1ms | 否 | F=候选数, 典型 <10 |
| vehicle_fusion | O(V) | 2~3ms | 否 | V=候选数, 典型 <10 |
| semantic_tracker (级联匹配) | O(T*logT) | 2~4ms | 否 | T=跟踪数, 典型 <50 |
| semantic_localizer (TF) | O(T) | 1~3ms | 否 | TF2查找 |
| risk_assessor | O(T) | 1~2ms | 否 | T=跟踪目标数 |
| degradation_manager | O(1) | <1ms | 否 | 状态机 |
| **端到端总计 (并行)** | - | **≤50ms** | - | 满足100ms@10Hz目标 |

---

## 十四、第一版实施建议（与参考方案的对齐）

基于"正常情况方案"和"极端情况方案"的拆分建议，第一版最小可行实现应包含：

```
Phase 1 MVP 核心链路:

正常链路:
  RGB YOLO 检测 → 深度区域统计 → 红外热证据 → 统一目标观测
  → 导航坐标转换 → 任务模块

极端链路:
  红外热成像(主风险) → 毫米波(辅助) → 气体(告警)
  → RGB(低权重候选) → 保守融合 → 安全联动
```

**关键实现优先级:**

| 优先级 | 模块 | 理由 |
|---|---|---|
| P0 | thermal_detector (hotspot + thermal_risk_level) | 两种模式都必需 |
| P0 | rgb_detector (person/vehicle/flame) | 正常模式的核心 |
| P0 | sensor_sync | 所有下游依赖 |
| P1 | person_fusion (含 person_candidate 逻辑) | 安全关键 |
| P1 | fire_fusion (决策矩阵) | 安全关键 |
| P1 | degradation_manager (四级状态机) | 模式切换必需 |
| P1 | gas_risk | 环境告警 |
| P2 | radar_tracker | 极端模式核心 |
| P2 | smoke_estimator | 退化判断 |
| P2 | risk_assessor + safety_mapper | 安全联动 |
| P3 | vehicle_fusion | 通行性判断 |
| P3 | lidar_cluster | 定位辅助 |
| P3 | semantic_tracker + localizer | 跟踪和地图坐标 |

**与参考方案的对应关系:**

| 本文档章节 | 正常方案对应 | 极端方案对应 |
|---|---|---|
| 11.2 传感器可信度 | 三、传感器策略 | 三、传感器退化策略 |
| 11.3 动态权重 | — | 五、融合策略 |
| 11.4 退化状态机 | — | — (本文档设计) |
| 11.5 热风险等级 | 4.3 红外热成像分析 | 4.2 红外热成像主风险判断 |
| 11.6 综合烟雾指数 | — | 三、退化策略 |
| 11.7 人员融合 | 5.1 人员融合 | 5.2 人员融合 |
| 11.8 火源确认 | 5.2 火情融合 | 5.1 火情融合 |
| 11.9 环境风险 | — | 5.5 环境风险融合 |
| 11.10 安全策略 | — | 八、安全策略 |
| 12.2 正常流程 | 二、总体链路 | — |
| 12.3 极端流程 | — | 二、总体链路 |
| 12.4 状态机联动 | 七、状态机联动 | 七、状态机联动 |
| 12.5 退化切换 | — | 三、传感器退化策略 |

---

## 十五、C++ 系统代码实现指南

本章给出从伪代码到生产级 C++/ROS 2 代码的完整映射，包含核心类的接口设计、实现模式和关键代码片段。目标平台 NVIDIA Jetson AGX Orin，ROS 2 Humble，C++17。

---

### 15.1 核心基础类型定义

所有算法实现依赖以下基础数据结构，它们桥接 ROS 2 消息和内部算法：

```cpp
// ============================================================================
// rescue_perception_core/include/rescue_perception_core/core/types.hpp
// ============================================================================

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <array>

namespace rescue_perception {

// ---- 时间 ----
using Timestamp = std::chrono::steady_clock::time_point;
using Duration  = std::chrono::steady_clock::duration;

inline double to_seconds(Duration d) {
  return std::chrono::duration<double>(d).count();
}

// ---- 基础几何 ----
using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Isometry3d = Eigen::Isometry3d;
using Quaterniond = Eigen::Quaterniond;

// ---- 传感器标识 ----
enum class SensorType : uint8_t {
  RGB           = 0,
  THERMAL       = 1,
  LIDAR_3D      = 2,
  RADAR_4D      = 3,
  LIDAR_2D      = 4,
  GAS           = 5,
  PM            = 6,
  TEMP_HUMIDITY = 7,
  COUNT         = 8
};

// 来源位掩码 (与消息定义对齐)
enum SourceMask : uint8_t {
  SRC_RGB     = 0x01,
  SRC_THERMAL = 0x02,
  SRC_LIDAR   = 0x04,
  SRC_RADAR   = 0x08,
  SRC_GAS     = 0x10,
};

// ---- 退化模式 ----
enum class DegradationMode : uint8_t {
  CLEAR                = 0,
  LOW_VISIBILITY       = 1,
  HEAVY_SMOKE          = 2,
  PERCEPTION_DEGRADED  = 3
};

// ---- 感知状态 (输出) ----
enum class PerceptionStatus : uint8_t {
  OK        = 0,
  DEGRADED  = 1,
  LOST      = 2
};

// ---- 安全等级 ----
enum class SafetyLevel : uint8_t {
  SAFETY_OK             = 0,
  SAFETY_WARNING        = 1,
  SAFETY_DEGRADED       = 2,
  SAFETY_STOP_REQUIRED  = 3,
  SAFETY_ESTOP          = 4
};

// ---- 环境风险 ----
enum class EnvironmentRisk : uint8_t {
  LOW      = 0,
  MEDIUM   = 1,
  HIGH     = 2,
  CRITICAL = 3
};

// ---- 可见度等级 ----
enum class VisibilityLevel : uint8_t {
  CLEAR        = 0,
  LOW_LIGHT    = 1,
  LIGHT_SMOKE  = 2,
  HEAVY_SMOKE  = 3,
  BLIND        = 4
};

// ---- 目标类别 ----
enum class ObjectClass : uint8_t {
  FLAME              = 1,
  SMOKE              = 2,
  PERSON_STANDING    = 3,
  PERSON_WALKING     = 4,
  PERSON_CROUCHING   = 5,
  PERSON_LYING       = 6,
  PERSON_OCCLUDED    = 7,
  VEHICLE_CAR        = 8,
  VEHICLE_TRUCK      = 9,
  VEHICLE_BUS        = 10,
  VEHICLE_MOTORCYCLE = 11,
  VEHICLE_UNKNOWN    = 12,
  CRACK              = 13,
  WATER_PUDDLE       = 14,
  WATER_SEEPAGE      = 15,
  EQUIPMENT_DAMAGED  = 16,
  HOT_ZONE           = 17,
  OBSTACLE_GENERIC   = 18,
  PERSON_CANDIDATE   = 19,   // 疑似人员(极端情况)
  FIRE_CANDIDATE     = 20,   // 疑似火源(极端情况)
  HOTSPOT            = 21,   // 热异常点
  UNKNOWN_HOT_OBJECT = 22,
};

// ---- 传感器可信度 ----
struct SensorCredibility {
  float credibility       = 1.0f;   // [0, 1] 综合可信度
  float effective_ratio   = 1.0f;   // [0, 1] 有效数据比例
  std::string cause;                // 退化原因描述
  bool is_failed          = false;  // 是否完全失效
};

// ---- 环境质量 (内部结构, 对应 EnvironmentQuality.msg) ----
struct EnvironmentQuality {
  Timestamp stamp;
  VisibilityLevel visibility = VisibilityLevel::CLEAR;
  float visibility_score     = 1.0f;  // [0,1]

  // 各传感器可信度
  std::array<SensorCredibility, static_cast<size_t>(SensorType::COUNT)> sensors;

  // 环境参量
  float smoke_density      = 0.0f;
  float smoke_coverage     = 0.0f;
  float ambient_temperature = 25.0f;
  float ambient_humidity    = 50.0f;
  float illuminance_lux     = 500.0f;

  // 便捷访问
  float rgb_credibility()     const { return sensors[0].credibility; }
  float thermal_credibility() const { return sensors[1].credibility; }
  float lidar_credibility()   const { return sensors[2].credibility; }
  float radar_credibility()   const { return sensors[3].credibility; }
  float lidar2d_credibility() const { return sensors[4].credibility; }
  float gas_credibility()     const { return sensors[5].credibility; }
};

// ---- 传感器权重 (6维) ----
struct SensorWeights {
  float rgb     = 0.0f;
  float thermal = 0.0f;
  float lidar   = 0.0f;
  float radar   = 0.0f;
  float lidar2d = 0.0f;
  float gas     = 0.0f;

  float operator[](size_t i) const {
    const float* p = &rgb;
    return p[i];
  }
  float& operator[](size_t i) {
    float* p = &rgb;
    return p[i];
  }
  void normalize() {
    float sum = rgb + thermal + lidar + radar + lidar2d + gas;
    if (sum > 1e-6f) {
      float inv = 1.0f / sum;
      rgb *= inv; thermal *= inv; lidar *= inv;
      radar *= inv; lidar2d *= inv; gas *= inv;
    }
  }
};

// ---- 行为建议 ----
struct BehaviorRecommendation {
  float max_speed_pct       = 1.0f;   // 最大速度比例
  bool require_confirmation  = false;  // 是否需要远程确认
  bool allow_autonomous      = true;   // 是否允许自主行动
  bool suggest_teleop        = false;  // 是否建议远程接管
};

// ---- 3D 检测候选 ----
struct Detection3D {
  Timestamp stamp;
  uint32_t track_id      = 0;
  ObjectClass class_id   = ObjectClass::OBSTACLE_GENERIC;
  float confidence       = 0.0f;
  SensorType source      = SensorType::LIDAR_3D;

  // 空间信息
  Vector3d position      = Vector3d::Zero();   // 传感器坐标系下
  Vector3d dimensions    = Vector3d::Zero();   // l, w, h
  double yaw             = 0.0;

  // 协方差
  Matrix6d covariance    = Matrix6d::Identity();

  // 运动
  Vector3d velocity      = Vector3d::Zero();

  // 热学属性 (仅 THERMAL 源有效)
  float temperature_max  = 0.0f;
  float temperature_min  = 0.0f;
};

// ---- 跟踪目标 ----
struct TrackedObject {
  uint32_t track_id      = 0;
  ObjectClass class_id   = ObjectClass::OBSTACLE_GENERIC;
  std::string class_name;
  float confidence       = 0.0f;
  uint8_t source_mask    = 0;
  bool confirmed         = false;       // 经多传感器确认

  // 状态向量 [x, y, z, l, w, h, vx, vy]
  Eigen::Matrix<double, 8, 1> state = Eigen::Matrix<double, 8, 1>::Zero();
  Eigen::Matrix<double, 8, 8> cov   = Eigen::Matrix<double, 8, 8>::Identity();

  // 地图坐标
  Isometry3d pose_map    = Isometry3d::Identity();
  Matrix6d pose_cov_map  = Matrix6d::Identity();

  // 生命周期
  int coast_frames       = 0;   // 连续未匹配帧数
  int confirmed_frames   = 0;   // 已确认帧数
  Timestamp last_seen;
  Timestamp first_seen;

  // 风险
  EnvironmentRisk risk   = EnvironmentRisk::LOW;
  float temperature_max  = 0.0f;
};

// ---- 热成像热点 ----
struct ThermalHotspot {
  Vector2d centroid_px;          // 图像坐标
  float T_max    = 0.0f;         // 最高温度 °C
  float T_mean   = 0.0f;         // 平均温度
  float T_min    = 0.0f;
  int area_px    = 0;            // 像素面积
  int perimeter  = 0;
  float circularity = 0.0f;      // 0~1, 火焰低圆形度
  float dT_dt    = 0.0f;         // 温升速率 °C/s
  float dA_dt    = 0.0f;         // 面积增长率 /s
  float duration_s = 0.0f;       // 持续时间
  float edge_gradient = 0.0f;    // 边缘温度梯度

  enum Type {
    FIRE_SOURCE,      // 明火
    FIRE_SUSPECTED,   // 疑似火源
    HOT_EQUIPMENT,    // 热设备
    WARM_SURFACE,     // 温暖表面
    HOT_ANOMALY,      // 一般热异常
    HUMAN_THERMAL,    // 人体热源
  };
  Type type = HOT_ANOMALY;
  float type_confidence = 0.0f;
};

// ---- 气体读数 ----
struct GasReadings {
  float co_ppm    = 0.0f;
  float co2_ppm   = 400.0f;
  float o2_pct    = 20.9f;
  float ch4_lel   = 0.0f;
  float h2s_ppm   = 0.0f;
  float voc_ppm   = 0.0f;
  float pm25      = 0.0f;
  float pm10      = 0.0f;
};

// ---- 气体风险 ----
struct GasRiskResult {
  enum Level { SAFE = 0, CAUTION = 1, DANGER = 2, DEADLY = 3 };
  Level risk_level = SAFE;
  bool fire_signature      = false;
  bool explosion_risk      = false;
  bool asphyxiation_risk   = false;
  struct Alert {
    std::string type;
    bool remote_notify    = false;
    bool mission_confirm  = false;
    bool auto_retreat     = false;
  } alert;
};

// ---- 退化状态转移事件 ----
enum class DegradationEvent {
  NONE,
  ENTER_LOW_VIS,
  ENTER_HEAVY_SMOKE,
  ENTER_DEGRADED,
  RECOVER_TO_HEAVY,
  RECOVER_TO_LOW_VIS,
  RECOVER_TO_CLEAR,
};

} // namespace rescue_perception
```

---

### 15.2 感知管线主控制器

这是整个感知系统的大脑，负责调度所有检测器、融合器和评估器：

```cpp
// ============================================================================
// rescue_perception_core/include/rescue_perception_core/perception_pipeline.hpp
// ============================================================================

#pragma once

#include "rescue_perception_core/core/types.hpp"
#include "rescue_perception_core/sensor_sync.hpp"
#include "rescue_perception_core/rgb_detector.hpp"
#include "rescue_perception_core/thermal_detector.hpp"
#include "rescue_perception_core/lidar_cluster.hpp"
#include "rescue_perception_core/radar_tracker.hpp"
#include "rescue_perception_core/smoke_estimator.hpp"
#include "rescue_perception_core/gas_risk.hpp"
#include "rescue_perception_core/person_fusion.hpp"
#include "rescue_perception_core/fire_fusion.hpp"
#include "rescue_perception_core/vehicle_fusion.hpp"
#include "rescue_perception_core/semantic_tracker.hpp"
#include "rescue_perception_core/semantic_localizer.hpp"
#include "rescue_perception_core/risk_assessor.hpp"
#include "rescue_perception_core/quality_monitor.hpp"
#include "rescue_perception_core/degradation_manager.hpp"

#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <vector>

namespace rescue_perception {

class PerceptionPipeline {
public:
  PerceptionPipeline(rclcpp::Node* node);
  ~PerceptionPipeline();

  // ---- 生命周期 ----
  void initialize();       // 加载模型、初始化推理引擎
  void start();            // 开始管线循环
  void stop();             // 停止管线
  void setMode(DegradationMode mode);  // 手动切换模式

  // ---- 每帧调用 (由 sensor_sync 的回调驱动) ----
  void spinOnce(const SyncedSensorData& synced);

  // ---- 状态查询 ----
  DegradationMode   currentMode()     const;
  EnvironmentRisk    currentRisk()     const;
  PerceptionStatus   currentStatus()   const;
  SensorWeights      currentWeights()  const;

private:
  // ---- 管线阶段 ----
  void phase0_healthCheck(const SyncedSensorData& synced);
  void phase1_detection(const SyncedSensorData& synced);
  void phase2_fusion();
  void phase3_tracking();
  void phase4_localization();
  void phase5_riskAssessment();
  void phase6_safetyMapping();
  void phase7_publish();

  // ---- ROS 2 发布器 (在 node_ 上创建) ----
  rclcpp::Node* node_;
  rclcpp::Publisher<SemanticObjectArray>::SharedPtr pub_targets_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_env_risk_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr pub_thermal_risk_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_gas_alert_;
  // ... 其他发布器

  // ---- 子模块 (unique_ptr 管理生命周期) ----
  std::unique_ptr<SensorSync>          sensor_sync_;
  std::unique_ptr<QualityMonitor>      quality_monitor_;
  std::unique_ptr<DegradationManager>  degradation_mgr_;
  std::unique_ptr<RgbDetector>         rgb_detector_;
  std::unique_ptr<ThermalDetector>     thermal_detector_;
  std::unique_ptr<LidarCluster>        lidar_cluster_;
  std::unique_ptr<RadarTracker>        radar_tracker_;
  std::unique_ptr<SmokeEstimator>      smoke_estimator_;
  std::unique_ptr<GasRisk>             gas_risk_;
  std::unique_ptr<PersonFusion>        person_fusion_;
  std::unique_ptr<FireFusion>          fire_fusion_;
  std::unique_ptr<VehicleFusion>       vehicle_fusion_;
  std::unique_ptr<SemanticTracker>     semantic_tracker_;
  std::unique_ptr<SemanticLocalizer>   semantic_localizer_;
  std::unique_ptr<RiskAssessor>        risk_assessor_;

  // ---- 管线中间状态 ----
  EnvironmentQuality    env_quality_;
  DegradationMode       degradation_mode_ = DegradationMode::CLEAR;
  SensorWeights         sensor_weights_;

  // 各检测器输出 (帧级缓存)
  std::vector<Detection3D> rgb_detections_;
  std::vector<Detection3D> thermal_detections_;
  std::vector<Detection3D> lidar_clusters_;
  std::vector<Detection3D> radar_tracks_;
  SmokeEstimateResult     smoke_estimate_;
  GasRiskResult           gas_risk_result_;
  int                     thermal_risk_level_ = 0;

  // 融合输出
  std::vector<TrackedObject> person_fused_;
  std::vector<TrackedObject> fire_fused_;
  std::vector<TrackedObject> vehicle_fused_;

  // 跟踪输出
  std::vector<TrackedObject> all_tracks_;

  // 风险评估
  EnvironmentRisk  env_risk_ = EnvironmentRisk::LOW;
  SafetyLevel      safety_level_ = SafetyLevel::SAFETY_OK;
  float            speed_limit_  = 1.0f;

  // 性能计时
  Timestamp frame_start_;
};

} // namespace rescue_perception
```

```cpp
// ============================================================================
// rescue_perception_core/src/perception_pipeline.cpp
// ============================================================================

#include "rescue_perception_core/perception_pipeline.hpp"

namespace rescue_perception {

PerceptionPipeline::PerceptionPipeline(rclcpp::Node* node)
  : node_(node)
{
  // 模块将在 initialize() 中构造
}

void PerceptionPipeline::initialize() {
  RCLCPP_INFO(node_->get_logger(), "[Pipeline] Initializing perception pipeline...");

  // 1. 构造子模块 (构造顺序即依赖顺序)
  sensor_sync_      = std::make_unique<SensorSync>(node_);
  quality_monitor_  = std::make_unique<QualityMonitor>(node_);
  degradation_mgr_  = std::make_unique<DegradationManager>(node_);
  rgb_detector_     = std::make_unique<RgbDetector>(node_);
  thermal_detector_ = std::make_unique<ThermalDetector>(node_);
  lidar_cluster_    = std::make_unique<LidarCluster>(node_);
  radar_tracker_    = std::make_unique<RadarTracker>(node_);
  smoke_estimator_  = std::make_unique<SmokeEstimator>(node_);
  gas_risk_         = std::make_unique<GasRisk>(node_);
  person_fusion_    = std::make_unique<PersonFusion>(node_);
  fire_fusion_      = std::make_unique<FireFusion>(node_);
  vehicle_fusion_   = std::make_unique<VehicleFusion>(node_);
  semantic_tracker_ = std::make_unique<SemanticTracker>(node_);
  semantic_localizer_ = std::make_unique<SemanticLocalizer>(node_);
  risk_assessor_    = std::make_unique<RiskAssessor>(node_);

  // 2. 初始化各模块 (加载模型、分配内存)
  rgb_detector_->initialize();
  thermal_detector_->initialize();
  // ... 其他模块

  // 3. 创建发布器
  pub_targets_ = node_->create_publisher<SemanticObjectArray>(
    "/perception/target_observations", 10);
  pub_status_ = node_->create_publisher<std_msgs::msg::String>(
    "/perception/status", 10);
  pub_env_risk_ = node_->create_publisher<std_msgs::msg::String>(
    "/perception/environment_risk", 10);
  pub_thermal_risk_ = node_->create_publisher<std_msgs::msg::Int32>(
    "/thermal/risk_level", 10);
  pub_gas_alert_ = node_->create_publisher<std_msgs::msg::String>(
    "/gas_alert", 10);

  RCLCPP_INFO(node_->get_logger(), "[Pipeline] Initialization complete.");
}

// ═════════════════════════════════════════════════════════════════
// 核心: 每帧调度
// ═════════════════════════════════════════════════════════════════
void PerceptionPipeline::spinOnce(const SyncedSensorData& synced) {
  frame_start_ = std::chrono::steady_clock::now();

  // 阶段0: 传感器健康检查 (必须先运行, 为后续提供权重)
  phase0_healthCheck(synced);

  // 阶段1: 各传感器独立检测 (可并行, 实际串行调用也能满足延迟要求)
  phase1_detection(synced);

  // 阶段2: 目标级融合
  phase2_fusion();

  // 阶段3: 统一跟踪
  phase3_tracking();

  // 阶段4: 坐标转换
  phase4_localization();

  // 阶段5: 场景级风险评估
  phase5_riskAssessment();

  // 阶段6: 安全策略映射
  phase6_safetyMapping();

  // 阶段7: 发布所有输出
  phase7_publish();

  // 性能日志
  auto elapsed = std::chrono::steady_clock::now() - frame_start_;
  if (to_seconds(elapsed) > 0.080) {  // > 80ms 时告警
    RCLCPP_WARN(node_->get_logger(),
      "[Pipeline] Frame took %.1f ms (target < 50ms)",
      to_seconds(elapsed) * 1000.0);
  }
}

// ═════════════════════════════════════════════════════════════════
// 阶段0: 传感器健康检查 → 更新退化模式和权重
// ═════════════════════════════════════════════════════════════════
void PerceptionPipeline::phase0_healthCheck(const SyncedSensorData& synced) {
  // 评估所有传感器质量
  env_quality_ = quality_monitor_->assess(synced);

  // 更新退化模式 (带滞后的状态机)
  degradation_mode_ = degradation_mgr_->update(env_quality_, smoke_estimate_);

  // 获取当前退化模式下的传感器权重
  sensor_weights_ = degradation_mgr_->getWeights(degradation_mode_, env_quality_);
}

// ═════════════════════════════════════════════════════════════════
// 阶段1: 独立检测 (各传感器独立推理, 无相互依赖)
// ═════════════════════════════════════════════════════════════════
void PerceptionPipeline::phase1_detection(const SyncedSensorData& synced) {
  // RGB 检测 — 退化模式下仍运行但结果会被降权
  if (synced.rgb_image.has_value() && sensor_weights_.rgb > 0.01f) {
    rgb_detections_ = rgb_detector_->detect(
      synced.rgb_image.value(), degradation_mode_);
  } else {
    rgb_detections_.clear();
  }

  // 热成像检测 — 所有模式下都运行 (极端情况下的主传感器)
  if (synced.thermal_image.has_value() && synced.temp_matrix.has_value()) {
    auto result = thermal_detector_->detect(
      synced.thermal_image.value(),
      synced.temp_matrix.value());
    thermal_detections_ = result.detections;
    thermal_risk_level_ = result.risk_level;
  } else {
    thermal_detections_.clear();
    thermal_risk_level_ = 0;
  }

  // LiDAR 聚类
  if (synced.lidar_points.has_value() && sensor_weights_.lidar > 0.01f) {
    lidar_clusters_ = lidar_cluster_->cluster(
      synced.lidar_points.value(), degradation_mode_);
  } else {
    lidar_clusters_.clear();
  }

  // 毫米波跟踪
  if (synced.radar_data.has_value() && sensor_weights_.radar > 0.01f) {
    radar_tracks_ = radar_tracker_->track(synced.radar_data.value());
  } else {
    radar_tracks_.clear();
  }

  // 烟雾估计
  smoke_estimate_ = smoke_estimator_->estimate(
    rgb_detections_, env_quality_, degradation_mode_);

  // 气体风险
  if (synced.gas_data.has_value()) {
    gas_risk_result_ = gas_risk_->assess(
      synced.gas_data.value(), env_quality_);
  }
}

// ═════════════════════════════════════════════════════════════════
// 阶段2: 目标级融合
// ═════════════════════════════════════════════════════════════════
void PerceptionPipeline::phase2_fusion() {
  // 人员融合
  person_fused_ = person_fusion_->fuse(
    rgb_detections_, thermal_detections_,
    lidar_clusters_, radar_tracks_,
    sensor_weights_, degradation_mode_, env_quality_);

  // 火源融合
  fire_fused_ = fire_fusion_->fuse(
    rgb_detections_, thermal_detections_,
    gas_risk_result_, sensor_weights_);

  // 车辆融合
  vehicle_fused_ = vehicle_fusion_->fuse(
    rgb_detections_, lidar_clusters_, radar_tracks_,
    sensor_weights_, degradation_mode_);
}

void PerceptionPipeline::phase3_tracking() {
  // 合并所有融合结果, 送入统一跟踪器
  std::vector<TrackedObject> all_fused;
  all_fused.insert(all_fused.end(), person_fused_.begin(), person_fused_.end());
  all_fused.insert(all_fused.end(), fire_fused_.begin(), fire_fused_.end());
  all_fused.insert(all_fused.end(), vehicle_fused_.begin(), vehicle_fused_.end());

  all_tracks_ = semantic_tracker_->update(all_fused);
}

void PerceptionPipeline::phase4_localization() {
  semantic_localizer_->transform(all_tracks_);
}

void PerceptionPipeline::phase5_riskAssessment() {
  env_risk_ = risk_assessor_->assess(
    all_tracks_, env_quality_, gas_risk_result_,
    smoke_estimate_, thermal_risk_level_, degradation_mode_);
}

void PerceptionPipeline::phase6_safetyMapping() {
  auto safety = risk_assessor_->mapToSafety(
    env_risk_, thermal_risk_level_, degradation_mode_,
    radar_tracker_->nearestRange());
  safety_level_ = safety.level;
  speed_limit_  = safety.speed_limit;
}

void PerceptionPipeline::phase7_publish() {
  // 1. 目标观测
  auto msg_targets = toRosMsg(all_tracks_);
  msg_targets.header.stamp = node_->now();
  msg_targets.header.frame_id = "map";
  pub_targets_->publish(msg_targets);

  // 2. 感知状态
  std_msgs::msg::String status_msg;
  switch (degradation_mode_) {
    case DegradationMode::CLEAR:   status_msg.data = "OK"; break;
    case DegradationMode::LOW_VISIBILITY:
    case DegradationMode::HEAVY_SMOKE: status_msg.data = "DEGRADED"; break;
    case DegradationMode::PERCEPTION_DEGRADED: status_msg.data = "LOST"; break;
  }
  pub_status_->publish(status_msg);

  // 3. 综合环境风险
  std_msgs::msg::String risk_msg;
  switch (env_risk_) {
    case EnvironmentRisk::LOW:      risk_msg.data = "LOW";      break;
    case EnvironmentRisk::MEDIUM:   risk_msg.data = "MEDIUM";   break;
    case EnvironmentRisk::HIGH:     risk_msg.data = "HIGH";     break;
    case EnvironmentRisk::CRITICAL: risk_msg.data = "CRITICAL"; break;
  }
  pub_env_risk_->publish(risk_msg);

  // 4. 热风险等级
  std_msgs::msg::Int32 thermal_msg;
  thermal_msg.data = thermal_risk_level_;
  pub_thermal_risk_->publish(thermal_msg);

  // 5. 气体告警
  if (gas_risk_result_.risk_level >= GasRiskResult::CAUTION) {
    auto gas_msg = std_msgs::msg::String();
    gas_msg.data = formatGasAlert(gas_risk_result_);
    pub_gas_alert_->publish(gas_msg);
  }
}

} // namespace rescue_perception
```

---

### 15.3 关键算法模块实现

#### 15.3.1 退化管理器 —— 状态机 + 权重分配

```cpp
// ============================================================================
// rescue_perception_core/include/rescue_perception_core/degradation_manager.hpp
// ============================================================================

#pragma once

#include "rescue_perception_core/core/types.hpp"
#include <rclcpp/rclcpp.hpp>
#include <array>
#include <deque>

namespace rescue_perception {

class DegradationManager {
public:
  explicit DegradationManager(rclcpp::Node* node);

  // 更新退化状态 (每帧调用)
  DegradationMode update(const EnvironmentQuality& env,
                         const SmokeEstimateResult& smoke);

  // 获取当前传感器权重
  SensorWeights getWeights(DegradationMode mode,
                           const EnvironmentQuality& env) const;

  // 获取行为建议
  BehaviorRecommendation getBehavior(DegradationMode mode) const;

  // 当前状态
  DegradationMode currentMode() const { return current_mode_; }

private:
  // ---- 滞后因子 ----
  static constexpr float HYSTERESIS = 0.20f;

  // ---- 基准权重表 [CLEAR][LOW_VIS][HEAVY_SMOKE][DEGRADED] ----
  static constexpr std::array<std::array<float, 6>, 4> BASE_WEIGHTS = {{
    // rgb, thermal, lidar, radar, lidar2d, gas
    {{0.90f, 0.40f, 0.85f, 0.30f, 0.20f, 0.20f}},  // CLEAR
    {{0.50f, 0.80f, 0.60f, 0.50f, 0.30f, 0.40f}},  // LOW_VIS
    {{0.10f, 0.85f, 0.25f, 0.85f, 0.50f, 0.60f}},  // HEAVY_SMOKE
    {{0.00f, 0.30f, 0.00f, 0.80f, 0.60f, 0.60f}},  // DEGRADED
  }};

  // ---- 行为建议表 ----
  static constexpr std::array<BehaviorRecommendation, 4> BEHAVIORS = {{
    {1.00f, false, true,  false},   // CLEAR
    {0.80f, false, true,  false},   // LOW_VIS
    {0.30f, true,  true,  false},   // HEAVY_SMOKE
    {0.10f, true,  false, true },   // DEGRADED
  }};

  // ---- 振荡保护 ----
  struct ModeChangeRecord {
    DegradationMode mode;
    Timestamp time;
  };
  std::deque<ModeChangeRecord> change_history_;
  Timestamp lock_until_;
  static constexpr double OSCILLATION_WINDOW_S = 3.0;
  static constexpr int    OSCILLATION_MAX_SWITCHES = 2;

  DegradationMode current_mode_ = DegradationMode::CLEAR;
  rclcpp::Node* node_;

  // ---- 内部方法 ----
  bool transitionTo(DegradationMode target);
  bool shouldExit(DegradationMode from, DegradationMode to,
                  const EnvironmentQuality& env,
                  const SmokeEstimateResult& smoke);
  bool checkOscillation(DegradationMode target);
};

} // namespace rescue_perception
```

```cpp
// ============================================================================
// rescue_perception_core/src/degradation_manager.cpp
// ============================================================================

#include "rescue_perception_core/degradation_manager.hpp"

namespace rescue_perception {

DegradationManager::DegradationManager(rclcpp::Node* node)
  : node_(node), lock_until_(Timestamp::min())
{
  RCLCPP_INFO(node_->get_logger(), "[DegradationMgr] Initialized in CLEAR mode");
}

DegradationMode DegradationManager::update(
    const EnvironmentQuality& env,
    const SmokeEstimateResult& smoke)
{
  auto now = std::chrono::steady_clock::now();

  // 如果处于锁定状态, 不处理状态转移
  if (now < lock_until_) {
    return current_mode_;
  }

  DegradationMode target = current_mode_;
  float s = smoke.smoke_score;
  float rc = env.rgb_credibility();
  float lc = env.lidar_credibility();
  float tc = env.thermal_credibility();
  float bc = env.illuminance_lux;

  switch (current_mode_) {

    case DegradationMode::CLEAR:
      if (s > 0.15f || rc < 0.50f || bc < 30.0f) {
        target = DegradationMode::LOW_VISIBILITY;
      }
      break;

    case DegradationMode::LOW_VISIBILITY:
      if (s > 0.35f || (rc < 0.20f && lc < 0.30f)) {
        target = DegradationMode::HEAVY_SMOKE;
      }
      // 恢复: 需要 smoke < 0.10 (滞后)
      else if (shouldExit(DegradationMode::LOW_VISIBILITY,
                          DegradationMode::CLEAR, env, smoke)) {
        target = DegradationMode::CLEAR;
      }
      break;

    case DegradationMode::HEAVY_SMOKE:
      if ((rc < 0.10f && lc < 0.20f && tc < 0.30f)
          || env.sensors[static_cast<int>(SensorType::GAS)].is_failed) {
        target = DegradationMode::PERCEPTION_DEGRADED;
      }
      // 恢复: 需要 smoke < 0.25 (滞后)
      else if (shouldExit(DegradationMode::HEAVY_SMOKE,
                          DegradationMode::LOW_VISIBILITY, env, smoke)) {
        target = DegradationMode::LOW_VISIBILITY;
      }
      break;

    case DegradationMode::PERCEPTION_DEGRADED:
      // 统计可用传感器数量
      int reliable = 0;
      if (tc > 0.50f) reliable++;
      if (env.radar_credibility() > 0.50f) reliable++;
      if (lc > 0.50f) reliable++;

      if (reliable >= 3 && s < 0.30f) {
        target = DegradationMode::LOW_VISIBILITY;
      } else if (reliable >= 2 && s < 0.50f) {
        target = DegradationMode::HEAVY_SMOKE;
      }
      break;
  }

  // 执行转移 (含振荡检测)
  if (target != current_mode_) {
    transitionTo(target);
  }

  return current_mode_;
}

bool DegradationManager::shouldExit(
    DegradationMode from, DegradationMode to,
    const EnvironmentQuality& env,
    const SmokeEstimateResult& smoke)
{
  // 滞后: 退出条件比进入条件宽松 20%
  float h = 1.0f - HYSTERESIS;
  float s = smoke.smoke_score;

  if (from == DegradationMode::LOW_VISIBILITY && to == DegradationMode::CLEAR) {
    return s < 0.10f && env.rgb_credibility() > 0.70f && env.illuminance_lux > 50.0f;
  }
  if (from == DegradationMode::HEAVY_SMOKE && to == DegradationMode::LOW_VISIBILITY) {
    return s < 0.25f && env.rgb_credibility() > 0.30f && env.lidar_credibility() > 0.30f;
  }
  return false;
}

bool DegradationManager::transitionTo(DegradationMode target) {
  auto now = std::chrono::steady_clock::now();

  // 记录切换
  change_history_.push_back({target, now});

  // 清理 3 秒前的记录
  while (!change_history_.empty() &&
         to_seconds(now - change_history_.front().time) > OSCILLATION_WINDOW_S) {
    change_history_.pop_front();
  }

  // 振荡检测: 3 秒内切换超过 2 次 → 锁定
  if (change_history_.size() > OSCILLATION_MAX_SWITCHES) {
    RCLCPP_WARN(node_->get_logger(),
      "[DegradationMgr] Oscillation detected! Locking mode to %d for %.1fs",
      static_cast<int>(current_mode_), OSCILLATION_WINDOW_S);
    lock_until_ = now + std::chrono::duration_cast<Duration>(
      std::chrono::duration<double>(OSCILLATION_WINDOW_S));
    return false;
  }

  auto old = current_mode_;
  current_mode_ = target;

  RCLCPP_INFO(node_->get_logger(),
    "[DegradationMgr] Mode transition: %d → %d (smoke=%.2f)",
    static_cast<int>(old), static_cast<int>(target),
    0.0f);  // TODO: pass smoke score

  return true;
}

SensorWeights DegradationManager::getWeights(
    DegradationMode mode, const EnvironmentQuality& env) const
{
  const auto& base = BASE_WEIGHTS[static_cast<int>(mode)];

  // 加载基准权重
  SensorWeights w;
  w.rgb     = base[0];
  w.thermal = base[1];
  w.lidar   = base[2];
  w.radar   = base[3];
  w.lidar2d = base[4];
  w.gas     = base[5];

  // ---- 根据实时可信度动态修正 ----
  auto applyFactor = [](float& weight, float credibility) {
    if (credibility < 0.10f) {
      weight = 0.0f;              // 完全失效 → 清零
    } else if (credibility < 0.30f) {
      weight *= credibility / 0.30f;  // 线性缩放
    }
    // credibility ≥ 0.3 → 保持基准权重
  };

  applyFactor(w.rgb,     env.rgb_credibility());
  applyFactor(w.thermal, env.thermal_credibility());
  applyFactor(w.lidar,   env.lidar_credibility());
  applyFactor(w.radar,   env.radar_credibility());
  applyFactor(w.gas,     env.gas_credibility());

  // ---- 退化模式特殊处理 ----
  if (mode >= DegradationMode::HEAVY_SMOKE) {
    w.rgb   *= 0.30f;                        // RGB 强制降权
    w.radar  = std::max(w.radar, 0.60f);     // 毫米波保底提升
    w.gas    = std::max(w.gas,   0.50f);     // 气体保底提升
  }

  if (mode == DegradationMode::PERCEPTION_DEGRADED) {
    w.lidar = 0.0f;
    w.rgb   = 0.0f;
    w.radar = std::max(w.radar, 0.70f);
  }

  // 归一化
  w.normalize();

  return w;
}

BehaviorRecommendation DegradationManager::getBehavior(DegradationMode mode) const {
  return BEHAVIORS[static_cast<int>(mode)];
}

} // namespace rescue_perception
```

---

#### 15.3.2 人员融合节点 —— 多源证据合成

```cpp
// ============================================================================
// rescue_perception_core/include/rescue_perception_core/person_fusion.hpp
// ============================================================================

#pragma once

#include "rescue_perception_core/core/types.hpp"
#include <vector>
#include <unordered_map>

namespace rescue_perception {

class PersonFusion {
public:
  explicit PersonFusion(rclcpp::Node* node);

  // 核心融合接口
  std::vector<TrackedObject> fuse(
    const std::vector<Detection3D>& rgb_detections,
    const std::vector<Detection3D>& thermal_detections,
    const std::vector<Detection3D>& lidar_clusters,
    const std::vector<Detection3D>& radar_tracks,
    const SensorWeights& weights,
    DegradationMode mode,
    const EnvironmentQuality& env_quality);

private:
  // ---- 数据关联 ----
  struct AssociationResult {
    std::vector<std::pair<int, int>> matches;       // (cand_idx, track_idx)
    std::vector<int> unmatched_candidates;
    std::vector<int> unmatched_tracks;
  };

  AssociationResult associate(
    const std::vector<Detection3D>& candidates,
    const std::vector<TrackedObject>& active_tracks);

  float associationCost(const Detection3D& cand, const TrackedObject& track);

  // ---- 置信度融合 (见 11.7 算法) ----
  float fuseConfidence(
    const Detection3D* rgb,
    const Detection3D* thermal,
    const Detection3D* lidar,
    const Detection3D* radar,
    const SensorWeights& weights,
    DegradationMode mode,
    ObjectClass& out_class,
    bool& out_confirmed,
    uint8_t& out_source_mask);

  // ---- 3D 定位 ----
  Vector3d localize3D(const Detection3D& primary, const Detection3D* lidar,
                      const Detection3D* radar);

  // ---- 卡尔曼滤波 ----
  void kalmanPredict(TrackedObject& track, double dt);
  void kalmanUpdate(TrackedObject& track, const Vector3d& measurement);

  // ---- 跟踪池 ----
  std::vector<TrackedObject> active_tracks_;
  uint32_t next_track_id_ = 1;
  rclcpp::Node* node_;

  // ---- 参数 ----
  float max_association_dist_ = 3.0f;   // 最大关联距离 (m)
  float max_association_cost_ = 2.0f;   // 最大关联代价
  int   max_coast_frames_     = 15;     // 最大丢失帧数
  int   min_confirm_frames_   = 3;      // 最小确认帧数

  // 关联权重
  float w_pos_   = 0.4f;
  float w_class_ = 0.3f;
  float w_temp_  = 0.2f;
  float w_time_  = 0.1f;
};

} // namespace rescue_perception
```

```cpp
// ============================================================================
// rescue_perception_core/src/person_fusion.cpp (核心融合逻辑)
// ============================================================================

#include "rescue_perception_core/person_fusion.hpp"
#include <hungarian_algorithm.hpp>   // 第三方匈牙利匹配库
#include <cmath>

namespace rescue_perception {

std::vector<TrackedObject> PersonFusion::fuse(
    const std::vector<Detection3D>& rgb_detections,
    const std::vector<Detection3D>& thermal_detections,
    const std::vector<Detection3D>& lidar_clusters,
    const std::vector<Detection3D>& radar_tracks,
    const SensorWeights& weights,
    DegradationMode mode,
    const EnvironmentQuality& env_quality)
{
  // 如果所有传感器权重都接近 0, 直接返回空
  if (weights.rgb < 0.01f && weights.thermal < 0.01f
      && weights.lidar < 0.01f && weights.radar < 0.01f) {
    return {};
  }

  // ===== 步骤1: 收集所有人员候选 =====
  std::vector<Detection3D> candidates;

  auto isPersonClass = [](ObjectClass c) {
    return c >= ObjectClass::PERSON_STANDING && c <= ObjectClass::PERSON_OCCLUDED;
  };

  for (const auto& d : rgb_detections) {
    if (isPersonClass(d.class_id) && d.confidence > 0.3f
        && env_quality.rgb_credibility() > 0.1f) {
      candidates.push_back(d);
    }
  }
  for (const auto& d : thermal_detections) {
    if (isPersonClass(d.class_id) && d.confidence > 0.2f
        && env_quality.thermal_credibility() > 0.1f) {
      candidates.push_back(d);
    }
  }
  for (const auto& d : lidar_clusters) {
    if (d.class_id == ObjectClass::OBSTACLE_GENERIC
        && d.dimensions.z() >= 0.3f && d.dimensions.z() <= 2.0f   // 人体高度范围
        && d.dimensions.x() <= 1.0f) {                             // 人体宽度范围
      candidates.push_back(d);
    }
  }
  for (const auto& d : radar_tracks) {
    if (d.class_id == ObjectClass::OBSTACLE_GENERIC
        && d.confidence > 0.3f) {
      candidates.push_back(d);
    }
  }

  // ===== 步骤2: 卡尔曼预测 =====
  auto now = std::chrono::steady_clock::now();
  for (auto& track : active_tracks_) {
    double dt = to_seconds(now - track.last_seen);
    if (dt > 0 && dt < 1.0) {  // 合理的时间间隔
      kalmanPredict(track, dt);
    }
  }

  // ===== 步骤3: 匈牙利匹配 =====
  auto assoc = associate(candidates, active_tracks_);

  std::vector<TrackedObject> results;

  // ===== 步骤4: 更新已匹配的跟踪 =====
  for (auto [cand_idx, track_idx] : assoc.matches) {
    auto& track = active_tracks_[track_idx];
    const auto& cand = candidates[cand_idx];

    // 卡尔曼更新
    kalmanUpdate(track, cand.position);

    // 重新计算融合置信度
    // 找到同方向的各源候选
    const Detection3D* rgb_ptr = nullptr;
    const Detection3D* thermal_ptr = nullptr;
    const Detection3D* lidar_ptr = nullptr;
    const Detection3D* radar_ptr = nullptr;
    // (简化: 实际应通过空间邻近找到同目标的多源候选)
    if (cand.source == SensorType::RGB) rgb_ptr = &cand;
    if (cand.source == SensorType::THERMAL) thermal_ptr = &cand;
    if (cand.source == SensorType::LIDAR_3D) lidar_ptr = &cand;
    if (cand.source == SensorType::RADAR_4D) radar_ptr = &cand;

    ObjectClass out_class;
    bool confirmed;
    uint8_t source_mask;
    track.confidence = fuseConfidence(
      rgb_ptr, thermal_ptr, lidar_ptr, radar_ptr,
      weights, mode, out_class, confirmed, source_mask);

    track.class_id    = out_class;
    track.confirmed   = confirmed;
    track.source_mask = source_mask;
    track.coast_frames = 0;
    track.confirmed_frames++;
    track.last_seen = now;

    results.push_back(track);
  }

  // ===== 步骤5: 创建新跟踪 =====
  for (int idx : assoc.unmatched_candidates) {
    const auto& cand = candidates[idx];

    // 高置信度候选才创建新跟踪
    float creation_threshold = (mode >= DegradationMode::HEAVY_SMOKE) ? 0.35f : 0.50f;
    if (cand.confidence > creation_threshold) {
      TrackedObject new_track;
      new_track.track_id   = next_track_id_++;
      new_track.class_id   = cand.class_id;
      new_track.confidence = cand.confidence;
      new_track.source_mask = (cand.source == SensorType::RGB)     ? SRC_RGB
                            : (cand.source == SensorType::THERMAL) ? SRC_THERMAL
                            : (cand.source == SensorType::LIDAR_3D)? SRC_LIDAR
                            : SRC_RADAR;
      new_track.state.head<3>() = cand.position;
      new_track.state.segment<3>(3) = cand.dimensions;
      new_track.confirmed_frames = 0;
      new_track.coast_frames = 0;
      new_track.first_seen = now;
      new_track.last_seen  = now;

      active_tracks_.push_back(new_track);
      results.push_back(new_track);
    }
  }

  // ===== 步骤6: 清理丢失的跟踪 =====
  for (int idx : assoc.unmatched_tracks) {
    auto& track = active_tracks_[idx];
    track.coast_frames++;
    if (track.coast_frames <= max_coast_frames_) {
      results.push_back(track);  // 仍输出, 但标记为丢失
    }
    // 超过 max_coast_frames_ 的不加入结果 (后续在清理阶段删除)
  }

  // 清理长期丢失的跟踪
  active_tracks_.erase(
    std::remove_if(active_tracks_.begin(), active_tracks_.end(),
      [this](const TrackedObject& t) {
        return t.coast_frames > max_coast_frames_;
      }),
    active_tracks_.end());

  return results;
}

// ═════════════════════════════════════════════════════════════════
// 核心: 多源置信度融合 (对应 11.7 算法)
// ═════════════════════════════════════════════════════════════════
float PersonFusion::fuseConfidence(
    const Detection3D* rgb,
    const Detection3D* thermal,
    const Detection3D* lidar,
    const Detection3D* radar,
    const SensorWeights& weights,
    DegradationMode mode,
    ObjectClass& out_class,
    bool& out_confirmed,
    uint8_t& out_source_mask)
{
  float fused = 0.0f;
  float weight_sum = 0.0f;
  int positive_sources = 0;
  out_source_mask = 0;

  // ---- RGB 证据 ----
  if (rgb && rgb->confidence > 0.3f) {
    float w = weights.rgb;
    if (mode >= DegradationMode::HEAVY_SMOKE) {
      w *= 0.10f;  // 重烟下 RGB 大幅降权
    }
    fused += w * rgb->confidence;
    weight_sum += w;
    out_source_mask |= SRC_RGB;
    if (rgb->confidence > 0.50f) positive_sources++;
  }

  // ---- 热成像证据 ----
  if (thermal && thermal->confidence > 0.2f) {
    float w = weights.thermal;

    // 温度合理性加权
    float temp_validity = 1.0f;
    if (thermal->temperature_max > 0) {
      float T = thermal->temperature_max;
      if (T < 24.0f || T > 44.0f)       temp_validity = 0.3f;
      else if (T < 28.0f || T > 42.0f)  temp_validity = 0.7f;
    }
    float conf = thermal->confidence * temp_validity;

    fused += w * conf;
    weight_sum += w;
    out_source_mask |= SRC_THERMAL;
    if (conf > 0.30f) positive_sources++;
  }

  // ---- LiDAR 证据 ----
  if (lidar && lidar->confidence > 0.1f) {
    // 形状验证
    float w = lidar->dimensions.z() / std::max(lidar->dimensions.x(), 0.1f);
    bool aspect_ok = w > 1.2f;  // 高 > 宽
    bool size_ok = (lidar->dimensions.x() >= 0.3f && lidar->dimensions.x() <= 1.0f)
                && (lidar->dimensions.y() >= 0.2f && lidar->dimensions.y() <= 0.8f)
                && (lidar->dimensions.z() >= 0.3f && lidar->dimensions.z() <= 2.0f);

    float shape_score = size_ok ? (aspect_ok ? 1.0f : 0.7f) : 0.3f;
    float conf = lidar->confidence * shape_score;

    fused += weights.lidar * conf;
    weight_sum += weights.lidar;
    out_source_mask |= SRC_LIDAR;
    if (conf > 0.30f && shape_score > 0.5f) positive_sources++;
  }

  // ---- 毫米波证据 ----
  if (radar && radar->confidence > 0.3f) {
    // RCS 和速度验证
    float vel = radar->velocity.norm();
    bool vel_ok = vel < 2.0f;
    float vel_score = vel_ok ? 1.0f : std::max(0.0f, 1.0f - (vel - 2.0f) / 3.0f);

    float conf = radar->confidence * vel_score;
    fused += weights.radar * conf;
    weight_sum += weights.radar;
    out_source_mask |= SRC_RADAR;
    if (conf > 0.30f && vel_ok) positive_sources++;
  }

  // ---- 归一化 ----
  if (weight_sum > 1e-6f) {
    fused /= weight_sum;
  } else {
    fused = 0.0f;
  }

  // ---- 多传感器确认 ----
  out_confirmed = (positive_sources >= 2);

  // ---- 分类决策 ----
  if (out_confirmed && fused >= 0.70f) {
    out_class = ObjectClass::PERSON_STANDING;  // 高置信人员 (具体姿态由RGB/热成像确定)
  } else if (fused >= 0.40f) {
    out_class = ObjectClass::PERSON_CANDIDATE;  // 疑似人员
  } else {
    out_class = ObjectClass::OBSTACLE_GENERIC;  // 不确定
  }

  // ---- 极端情况升格 ----
  if (mode >= DegradationMode::HEAVY_SMOKE) {
    if (thermal && radar && radar->velocity.norm() > 0.3f) {
      // 热源 + 毫米波运动 → 至少是人员候选
      if (out_class < ObjectClass::PERSON_CANDIDATE) {
        out_class = ObjectClass::PERSON_CANDIDATE;
      }
      fused = std::max(fused, 0.45f);
    }
    if (thermal && thermal->confidence > 0.4f && thermal->temperature_max >= 28.0f
        && thermal->temperature_max <= 42.0f) {
      // 热源温度在人体范围 → 至少是人员候选
      if (out_class < ObjectClass::PERSON_CANDIDATE) {
        out_class = ObjectClass::PERSON_CANDIDATE;
      }
      fused = std::max(fused, 0.40f);
    }
  }

  return std::clamp(fused, 0.0f, 1.0f);
}

// ═════════════════════════════════════════════════════════════════
// 数据关联: 匈牙利匹配
// ═════════════════════════════════════════════════════════════════
PersonFusion::AssociationResult PersonFusion::associate(
    const std::vector<Detection3D>& candidates,
    const std::vector<TrackedObject>& active_tracks)
{
  AssociationResult result;

  if (candidates.empty()) {
    for (size_t i = 0; i < active_tracks.size(); i++) {
      result.unmatched_tracks.push_back(i);
    }
    return result;
  }
  if (active_tracks.empty()) {
    for (size_t i = 0; i < candidates.size(); i++) {
      result.unmatched_candidates.push_back(i);
    }
    return result;
  }

  // 构建代价矩阵
  int N = candidates.size();
  int M = active_tracks.size();
  std::vector<std::vector<double>> cost(N, std::vector<double>(M));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      cost[i][j] = associationCost(candidates[i], active_tracks[j]);
    }
  }

  // 匈牙利算法求解
  auto assignments = hungarian_solve(cost);

  std::vector<bool> cand_matched(N, false);
  std::vector<bool> track_matched(M, false);

  for (auto [i, j] : assignments) {
    if (cost[i][j] < max_association_cost_) {
      result.matches.push_back({i, j});
      cand_matched[i] = true;
      track_matched[j] = true;
    }
  }

  for (int i = 0; i < N; i++) if (!cand_matched[i]) result.unmatched_candidates.push_back(i);
  for (int j = 0; j < M; j++) if (!track_matched[j]) result.unmatched_tracks.push_back(j);

  return result;
}

float PersonFusion::associationCost(const Detection3D& cand, const TrackedObject& track) {
  // 空间距离
  Vector3d track_pos = track.state.head<3>();
  double dist = (cand.position - track_pos).norm();

  // 如果超出最大关联距离, 代价无穷
  if (dist > max_association_dist_) {
    return std::numeric_limits<float>::max();
  }

  // 时间间隔
  auto now = std::chrono::steady_clock::now();
  double dt = to_seconds(now - track.last_seen);

  // 综合代价
  float cost = w_pos_ * dist
             + w_time_ * std::min(dt, 2.0);  // 时间惩罚上限 2s

  return cost;
}

void PersonFusion::kalmanPredict(TrackedObject& track, double dt) {
  // 8 维状态: [x, y, z, l, w, h, vx, vy]
  // 匀速运动模型
  Eigen::Matrix<double, 8, 8> F = Eigen::Matrix<double, 8, 8>::Identity();
  F(0, 6) = dt;  F(1, 7) = dt;  // x += vx*dt, y += vy*dt

  // 过程噪声
  Eigen::Matrix<double, 8, 8> Q = Eigen::Matrix<double, 8, 8>::Identity() * 0.1;
  Q(6, 6) = 0.5; Q(7, 7) = 0.5;  // 速度噪声更大

  track.state = F * track.state;
  track.cov   = F * track.cov * F.transpose() + Q;
}

void PersonFusion::kalmanUpdate(TrackedObject& track, const Vector3d& measurement) {
  // 观测矩阵: 只观测位置 [x, y, z]
  Eigen::Matrix<double, 3, 8> H = Eigen::Matrix<double, 3, 8>::Zero();
  H(0, 0) = 1; H(1, 1) = 1; H(2, 2) = 1;

  // 观测噪声
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * 0.15;

  // 卡尔曼增益
  Eigen::Matrix<double, 8, 3> K = track.cov * H.transpose()
    * (H * track.cov * H.transpose() + R).inverse();

  // 更新
  Eigen::Vector3d innovation = measurement - H * track.state;
  track.state += K * innovation;
  track.cov   = (Eigen::Matrix<double, 8, 8>::Identity() - K * H) * track.cov;
}

} // namespace rescue_perception
```

---

#### 15.3.3 热成像检测器 —— 温度矩阵处理

```cpp
// ============================================================================
// rescue_perception_core/src/thermal_detector.cpp (核心算法部分)
// ============================================================================

#include "rescue_perception_core/thermal_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <queue>
#include <algorithm>

namespace rescue_perception {

struct ThermalDetectResult {
  std::vector<Detection3D> detections;
  std::vector<ThermalHotspot> hotspots;
  int risk_level = 0;
};

ThermalDetectResult ThermalDetector::detect(
    const cv::Mat& thermal_image,
    const cv::Mat& temperature_matrix)  // CV_32FC1, 640x512
{
  ThermalDetectResult result;

  // ===== 步骤1: 坏点修复 (3x3 中值滤波, 仅针对异常跳变像素) =====
  cv::Mat temp_fixed;
  cv::medianBlur(temperature_matrix, temp_fixed, 3);

  // ===== 步骤2: 更新背景基线 =====
  updateBaseline(temp_fixed);

  // ===== 步骤3: 自适应阈值分割 =====
  float T_bg = bg_mean_;
  float bg_std = bg_std_;
  float threshold = T_bg + std::max(30.0f, 3.0f * bg_std);

  cv::Mat hot_mask;
  cv::threshold(temp_fixed, hot_mask, threshold, 255, cv::THRESH_BINARY);
  hot_mask.convertTo(hot_mask, CV_8U);

  // ===== 步骤4: 连通域分析 =====
  cv::Mat labels, stats, centroids;
  int n_components = cv::connectedComponentsWithStats(
    hot_mask, labels, stats, centroids, 8, CV_32S);

  std::vector<ThermalHotspot> hotspots;

  for (int i = 1; i < n_components; i++) {  // 跳过背景 (label 0)
    int area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (area < 9) continue;  // 最小面积过滤

    // 提取区域温度值
    cv::Mat region_mask = (labels == i);
    std::vector<float> temps;
    for (int r = 0; r < temp_fixed.rows; r++) {
      for (int c = 0; c < temp_fixed.cols; c++) {
        if (labels.at<int>(r, c) == i) {
          temps.push_back(temp_fixed.at<float>(r, c));
        }
      }
    }

    if (temps.empty()) continue;

    // 统计量
    auto [min_it, max_it] = std::minmax_element(temps.begin(), temps.end());
    float T_max = *max_it;
    float T_min = *min_it;
    float T_mean = std::accumulate(temps.begin(), temps.end(), 0.0f) / temps.size();

    // 边界提取 (用于周长和温度梯度)
    cv::Mat edge;
    cv::Canny(region_mask, edge, 1, 1);
    int perimeter = cv::countNonZero(edge);

    ThermalHotspot hs;
    hs.T_max    = T_max;
    hs.T_mean   = T_mean;
    hs.T_min    = T_min;
    hs.area_px  = area;
    hs.perimeter = perimeter;
    hs.centroid_px = {centroids.at<double>(i, 0),
                      centroids.at<double>(i, 1)};
    hs.circularity = (perimeter > 0)
      ? (4.0f * M_PI * area / (perimeter * perimeter)) : 1.0f;

    // ===== 步骤5: 时序匹配 =====
    auto hist = findMatchingHotspot(hs.centroid_px, hs.area_px);
    if (hist.has_value()) {
      float dt = to_seconds(std::chrono::steady_clock::now() - hist->last_seen);
      if (dt > 1e-6f) {
        hs.dT_dt = (hs.T_max - hist->T_max) / dt;
        hs.dA_dt = (hs.area_px - hist->area_px) / (hist->area_px * dt);
      }
      hs.duration_s = hist->duration_s + frame_interval_;
    }

    // ===== 步骤6: 热点分类 =====
    classifyHotspot(hs);

    hotspots.push_back(hs);
  }

  // 更新历史
  hotspot_history_.push_back(hotspots);
  if (hotspot_history_.size() > HISTORY_FRAMES) {
    hotspot_history_.pop_front();
  }

  // ===== 步骤7: 计算热风险等级 =====
  result.risk_level = computeRiskLevel(hotspots, T_bg);
  result.hotspots = hotspots;

  // ===== 步骤8: 生成检测输出 (供融合模块使用) =====
  result.detections = hotspotsToDetections(hotspots);

  return result;
}

void ThermalDetector::classifyHotspot(ThermalHotspot& hs) {
  if (hs.T_max > 200.0f && hs.dT_dt > 5.0f && hs.dA_dt > 0.10f
      && hs.circularity < 0.70f) {
    hs.type = ThermalHotspot::FIRE_SOURCE;
    hs.type_confidence = 0.85f;
  }
  else if (hs.T_max > 150.0f && hs.dT_dt > 3.0f && hs.duration_s > 3.0f) {
    hs.type = ThermalHotspot::FIRE_SUSPECTED;
    hs.type_confidence = 0.60f;
  }
  else if (hs.T_max > 80.0f && hs.circularity > 0.80f && hs.dT_dt < 1.0f) {
    hs.type = ThermalHotspot::HOT_EQUIPMENT;
    hs.type_confidence = 0.75f;
  }
  else if (hs.T_max > 50.0f && hs.dT_dt < 0.5f && hs.circularity > 0.85f) {
    hs.type = ThermalHotspot::WARM_SURFACE;
    hs.type_confidence = 0.70f;
  }
  else {
    hs.type = ThermalHotspot::HOT_ANOMALY;
    hs.type_confidence = 0.50f;
  }
}

int ThermalDetector::computeRiskLevel(
    const std::vector<ThermalHotspot>& hotspots, float T_bg)
{
  if (hotspots.empty()) return 0;

  float max_temp = 0.0f;
  int fire_count = 0;
  for (const auto& h : hotspots) {
    max_temp = std::max(max_temp, h.T_max);
    if (h.type == ThermalHotspot::FIRE_SOURCE
        || h.type == ThermalHotspot::FIRE_SUSPECTED) {
      fire_count++;
    }
  }

  if (max_temp > 500.0f || fire_count >= 3)  return 4;
  if (max_temp > 300.0f || fire_count >= 2)  return 3;
  if (max_temp > 150.0f || fire_count >= 1)  return 2;
  if (max_temp > T_bg + 30.0f)               return 1;
  return 0;
}

void ThermalDetector::updateBaseline(const cv::Mat& temp_fixed) {
  // 滑动窗口: 取最近30帧每像素的最小值作为背景基线
  temp_history_.push_back(temp_fixed.clone());
  if (temp_history_.size() > BASELINE_WINDOW) {
    temp_history_.pop_front();
  }

  if (temp_history_.size() < 2) {
    bg_mean_ = cv::mean(temp_fixed)[0];
    return;
  }

  // 计算每像素最小值图像
  cv::Mat bg_image = temp_history_[0].clone();
  for (size_t i = 1; i < temp_history_.size(); i++) {
    bg_image = cv::min(bg_image, temp_history_[i]);
  }

  // 取中心50%区域的均值作为背景基线
  cv::Rect center_roi(bg_image.cols/4, bg_image.rows/4,
                      bg_image.cols/2, bg_image.rows/2);
  cv::Mat center = bg_image(center_roi);
  cv::Scalar mean, stddev;
  cv::meanStdDev(center, mean, stddev);
  bg_mean_ = mean[0];
  bg_std_  = stddev[0];
}

} // namespace rescue_perception
```

---

#### 15.3.4 火源确认决策矩阵

```cpp
// ============================================================================
// rescue_perception_core/src/fire_fusion.cpp (核心决策矩阵)
// ============================================================================

#include "rescue_perception_core/fire_fusion.hpp"

namespace rescue_perception {

struct FireConfirmationResult {
  int fire_level = 0;        // 0=NONE, 1=SUSPECTED, 2=CONFIRMED, 3=HIGH, 4=CRITICAL
  float confidence = 0.0f;
  bool should_stop = false;
  float hot_zone_radius_m = 1.5f;
};

FireConfirmationResult FireFusion::confirmFire(
    const Detection3D* rgb_flame,
    const ThermalHotspot* hotspot,
    const GasRiskResult& gas_risk,
    const SensorWeights& weights)
{
  FireConfirmationResult result;

  // ===== 步骤1: 量化证据强度 =====
  float rgb_conf     = rgb_flame ? rgb_flame->confidence : 0.0f;
  float thermal_conf = hotspot ? hotspot->type_confidence : 0.0f;
  float gas_conf     = gas_risk.fire_signature ? 0.7f
                      : (gas_risk.risk_level >= GasRiskResult::DANGER ? 0.5f : 0.0f);

  float max_temp     = hotspot ? hotspot->T_max : 0.0f;
  int hotspot_count  = hotspot ? 1 : 0;  // 简化: 实际应统计全部hotspots

  // ===== 步骤2: 决策树 (从高到低优先级) =====

  // 等级4: 致命火灾
  if ((thermal_conf > 0.8f && hotspot_count >= 3 && max_temp > 500.0f)
      || (thermal_conf > 0.7f && gas_conf > 0.7f && gas_risk.asphyxiation_risk)) {
    result.fire_level = 4;
    result.confidence = 0.95f;
  }
  // 等级3: 高危火灾
  else if ((thermal_conf > 0.7f && max_temp > 300.0f && hotspot_count >= 2)
           || (thermal_conf > 0.6f && gas_conf > 0.6f)) {
    result.fire_level = 3;
    result.confidence = 0.85f;
  }
  // 等级2: 确认火灾
  else if ((rgb_conf > 0.7f && thermal_conf > 0.6f)
           || (thermal_conf > 0.6f && gas_conf > 0.5f)
           || (rgb_conf > 0.5f && gas_conf > 0.5f && thermal_conf > 0.4f)) {
    result.fire_level = 2;
    result.confidence = 0.75f;
  }
  // 等级1: 疑似火灾
  else if ((rgb_conf > 0.5f && gas_conf > 0.4f)
           || thermal_conf > 0.5f
           || (rgb_conf > 0.6f && thermal_conf > 0.3f)) {
    result.fire_level = 1;
    result.confidence = 0.50f;
  }
  // 等级0: 无火灾
  else {
    result.fire_level = 0;
    result.confidence = 0.0f;
  }

  // ===== 步骤3: 气体风险修正 =====
  if (gas_risk.fire_signature) {
    result.confidence = std::min(result.confidence + 0.10f, 1.0f);
  }
  if (gas_risk.explosion_risk) {
    result.confidence = std::min(result.confidence + 0.15f, 1.0f);
    result.fire_level = std::max(result.fire_level, 3);
  }

  // ===== 步骤4: 安全建议 =====
  result.should_stop = (result.fire_level >= 3)
                    || (result.fire_level == 2 && max_temp > 300.0f);

  // ===== 步骤5: 高温禁区半径 =====
  if (max_temp > 500.0f)       result.hot_zone_radius_m = 8.0f;
  else if (max_temp > 300.0f)  result.hot_zone_radius_m = 5.0f;
  else if (max_temp > 150.0f)  result.hot_zone_radius_m = 3.0f;
  else                          result.hot_zone_radius_m = 1.5f;

  return result;
}

// ═════════════════════════════════════════════════════════════════
// 主融合接口
// ═════════════════════════════════════════════════════════════════
std::vector<TrackedObject> FireFusion::fuse(
    const std::vector<Detection3D>& rgb_detections,
    const std::vector<Detection3D>& thermal_detections,
    const GasRiskResult& gas_risk,
    const SensorWeights& weights)
{
  std::vector<TrackedObject> results;

  // 从热成像检测结果中提取火焰相关检测
  std::vector<Detection3D> thermal_flames;
  for (const auto& d : thermal_detections) {
    if (d.class_id == ObjectClass::FLAME
        || d.class_id == ObjectClass::FIRE_CANDIDATE
        || d.class_id == ObjectClass::HOTSPOT) {
      thermal_flames.push_back(d);
    }
  }

  // 从RGB检测结果中提取火焰
  std::vector<Detection3D> rgb_flames;
  for (const auto& d : rgb_detections) {
    if (d.class_id == ObjectClass::FLAME || d.class_id == ObjectClass::SMOKE) {
      rgb_flames.push_back(d);
    }
  }

  // 如果既没有RGB火焰也没有热成像热点, 但仍需检查气体fire_signature
  if (rgb_flames.empty() && thermal_flames.empty()) {
    if (gas_risk.fire_signature) {
      TrackedObject gas_only_fire;
      gas_only_fire.track_id   = next_id_++;
      gas_only_fire.class_id   = ObjectClass::FIRE_CANDIDATE;
      gas_only_fire.class_name = "fire_candidate";
      gas_only_fire.confidence = 0.40f;
      gas_only_fire.source_mask = SRC_GAS;
      gas_only_fire.confirmed   = false;
      results.push_back(gas_only_fire);
    }
    return results;
  }

  // 对每个热点/火焰候选执行决策矩阵
  for (const auto& hotspot_det : thermal_flames) {
    // 找同方向的 RGB 火焰 (简化: 实际需要空间关联)
    const Detection3D* matched_rgb = nullptr;
    for (const auto& rgb_f : rgb_flames) {
      double dist = (hotspot_det.position - rgb_f.position).norm();
      if (dist < 2.0) { matched_rgb = &rgb_f; break; }
    }

    auto conf = confirmFire(matched_rgb, nullptr, gas_risk, weights);

    if (conf.fire_level > 0) {
      TrackedObject fire;
      fire.track_id   = next_id_++;
      fire.class_id   = (conf.fire_level >= 2) ? ObjectClass::FLAME
                                                : ObjectClass::FIRE_CANDIDATE;
      fire.class_name = (conf.fire_level >= 2) ? "fire" : "fire_candidate";
      fire.confidence = conf.confidence;
      fire.source_mask = matched_rgb ? (SRC_RGB | SRC_THERMAL) : SRC_THERMAL;
      fire.confirmed   = conf.fire_level >= 2;
      fire.state.head<3>() = hotspot_det.position;
      fire.temperature_max = hotspot_det.temperature_max;

      // 设置风险等级
      switch (conf.fire_level) {
        case 4: fire.risk = EnvironmentRisk::CRITICAL; break;
        case 3: fire.risk = EnvironmentRisk::HIGH;     break;
        case 2: fire.risk = EnvironmentRisk::HIGH;     break;
        case 1: fire.risk = EnvironmentRisk::MEDIUM;   break;
      }

      results.push_back(fire);
    }
  }

  return results;
}

} // namespace rescue_perception
```

---

### 15.4 ROS 2 节点入口 — ComposableNode 模式

使用 ROS 2 组件化节点实现零拷贝通信:

```cpp
// ============================================================================
// rescue_perception_core/src/perception_node.cpp
// ============================================================================

#include <rclcpp/rclcpp.hpp>
#include "rescue_perception_core/perception_pipeline.hpp"

namespace rescue_perception {

// ROS 2 组件化节点: 支持 ComposableNode 零拷贝
class PerceptionNode : public rclcpp::Node {
public:
  explicit PerceptionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
    : rclcpp::Node("perception_node", options)
  {
    // 声明参数
    this->declare_parameter("pipeline_rate", 100.0);
    this->declare_parameter("mode", "auto");  // auto, inspection, rescue, minimal

    // 初始化管线
    pipeline_ = std::make_unique<PerceptionPipeline>(this);
    pipeline_->initialize();

    // 创建定时器驱动管线 (替代 sensor_sync 回调驱动, 简化版)
    double rate = this->get_parameter("pipeline_rate").as_double();
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / rate),
      std::bind(&PerceptionNode::timerCallback, this));

    // 订阅 sensor_sync 的同步输出 (实际应由 sensor_sync 回调驱动)
    // 此处简化: 假设 sensor_sync_node 作为 ComposableNode 在同一进程内运行,
    // 通过 intra_process_comms 零拷贝传递数据
    sync_sub_ = this->create_subscription<SyncedSensorDataMsg>(
      "/perception/synced/data", 10,
      std::bind(&PerceptionNode::syncedDataCallback, this,
                std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "[PerceptionNode] Started at %.0f Hz", rate);
  }

private:
  std::unique_ptr<PerceptionPipeline> pipeline_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<SyncedSensorDataMsg>::SharedPtr sync_sub_;

  // 最新同步数据缓存 (由 syncedDataCallback 更新, timerCallback 消费)
  std::mutex data_mutex_;
  std::optional<SyncedSensorData> latest_synced_;

  void syncedDataCallback(const SyncedSensorDataMsg::SharedPtr msg) {
    SyncedSensorData synced = fromRosMsg(msg);  // ROS消息 → 内部结构
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_synced_ = std::move(synced);
  }

  void timerCallback() {
    SyncedSensorData synced;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (!latest_synced_.has_value()) return;
      synced = latest_synced_.value();
    }
    pipeline_->spinOnce(synced);
  }
};

} // namespace rescue_perception

// ---- main 入口 (独立进程) ----
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rescue_perception::PerceptionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

// ---- 组件注册 (ComposableNode 入口) ----
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(rescue_perception::PerceptionNode)
```

对应的 CMakeLists.txt 配置:

```cmake
# ============================================================================
# rescue_perception_core/CMakeLists.txt
# ============================================================================
cmake_minimum_required(VERSION 3.8)
project(rescue_perception_core)

# C++17
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O3 -march=armv8.2-a+fp16+simd)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(message_filters REQUIRED)

# 如果需要 TensorRT/ONNX Runtime
find_package(onnxruntime REQUIRED)

# ---- 库: 核心类型 (header-only) ----
add_library(rescue_perception_types INTERFACE)
target_include_directories(rescue_perception_types INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(rescue_perception_types INTERFACE
  Eigen3::Eigen
)

# ---- 库: 核心算法 ----
set(CORE_SOURCES
  src/perception_pipeline.cpp
  src/degradation_manager.cpp
  src/quality_monitor.cpp
  src/rgb_detector.cpp
  src/thermal_detector.cpp
  src/lidar_cluster.cpp
  src/radar_tracker.cpp
  src/smoke_estimator.cpp
  src/gas_risk.cpp
  src/person_fusion.cpp
  src/fire_fusion.cpp
  src/vehicle_fusion.cpp
  src/semantic_tracker.cpp
  src/semantic_localizer.cpp
  src/risk_assessor.cpp
  src/sensor_sync.cpp
)

add_library(rescue_perception_core SHARED ${CORE_SOURCES})
target_include_directories(rescue_perception_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(rescue_perception_core PUBLIC
  rescue_perception_types
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs
  geometry_msgs::geometry_msgs
  tf2::tf2
  tf2_ros::tf2_ros
  ${OpenCV_LIBS}
  Eigen3::Eigen
  onnxruntime::onnxruntime
)
ament_target_dependencies(rescue_perception_core
  rclcpp sensor_msgs geometry_msgs std_msgs tf2 tf2_ros cv_bridge
)

# ---- 可执行文件 ----
add_executable(perception_node src/perception_node.cpp)
target_link_libraries(perception_node rescue_perception_core)
ament_target_dependencies(perception_node rclcpp rclcpp_components)

# ---- 组件注册 ----
rclcpp_components_register_node(perception_node
  PLUGIN "rescue_perception::PerceptionNode"
  EXECUTABLE perception_node_component
)

# ---- 安装 ----
install(TARGETS rescue_perception_types rescue_perception_core perception_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)

ament_package()
```

---

### 15.5 Jetson AGX Orin 性能优化要点

```cpp
// ============================================================================
// rescue_perception_core/include/rescue_perception_core/core/jetson_optimize.hpp
// ============================================================================
//
// Jetson AGX Orin 优化关键:
//   1. TensorRT FP16 推理 (ONNX → TensorRT Engine)
//   2. CUDA 流并行 (Detection + Segmentation 不同 stream)
//   3. 零拷贝: 图像数据在 GPU 内存中传递, 不回传 CPU
//   4. 内存池: 预分配大块内存, 避免运行时分配
//   5. CPU 亲和性: 将 pipeline 线程绑定到大核 (Cortex-A78AE)
//   6. DVFS: 锁定 MAXN 电源模式

#pragma once

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <memory>
#include <vector>
#include <thread>

namespace rescue_perception {

// ---- TensorRT 推理引擎封装 ----
class TRTEngine {
public:
  TRTEngine(const std::string& engine_path);
  ~TRTEngine();

  // 异步推理 (不阻塞 CPU)
  bool inferAsync(const std::vector<void*>& inputs,
                  const std::vector<void*>& outputs,
                  cudaStream_t stream);

  // 同步推理 (阻塞等待)
  bool inferSync(const std::vector<void*>& inputs,
                 const std::vector<void*>& outputs);

  void* getInputBuffer(int idx);
  void* getOutputBuffer(int idx);

private:
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<void*> buffers_;
  cudaStream_t stream_ = nullptr;
};

// ---- GPU 内存池 (避免频繁 cudaMalloc/cudaFree) ----
class GpuMemoryPool {
public:
  GpuMemoryPool(size_t pool_size_mb = 512);
  ~GpuMemoryPool();

  void* allocate(size_t bytes);
  void  deallocate(void* ptr);

  // 批量分配: 用于 TensorRT binding buffers
  std::vector<void*> allocateBuffers(const std::vector<size_t>& sizes);

private:
  struct Block {
    void* ptr;
    size_t size;
    bool in_use;
  };
  std::vector<Block> blocks_;
  void* pool_base_ = nullptr;
  std::mutex mutex_;
};

// ---- CPU 亲和性设置 ----
inline void setCpuAffinity(const std::vector<int>& cores) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int core : cores) {
    CPU_SET(core, &cpuset);
  }
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

// ---- 锁频到 MAXN 模式 ----
inline void setMaxPowerMode() {
  // 通过 nvpmodel 设置为 MAXN 模式 (在 launch 脚本中执行)
  // $ sudo nvpmodel -m 0
  // $ sudo jetson_clocks
  // 此时:
  //   - 12x Cortex-A78AE @ 2.2 GHz
  //   - GPU @ 1.3 GHz
  //   - DLA @ 1.4 GHz (如果使用)
  //   - Memory @ 3200 MHz
}

// ---- 零拷贝图像传递 ----
// 使用 cv::cuda::GpuMat 在 GPU 内存中直接传递图像,
// 避免 GPU→CPU→GPU 的冗余拷贝
struct GpuImageBundle {
  cv::cuda::GpuMat rgb_image;
  cv::cuda::GpuMat thermal_image;
  cv::cuda::GpuMat depth_image;
};

// ---- 双缓冲 (生产者-消费者) ----
template<typename T>
class DoubleBuffer {
public:
  void write(const T& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_[write_idx_] = data;
    write_idx_ = 1 - write_idx_;
    has_new_ = true;
  }

  bool read(T& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_new_) return false;
    out = buffers_[1 - write_idx_];
    has_new_ = false;
    return true;
  }

private:
  std::array<T, 2> buffers_;
  int write_idx_ = 0;
  bool has_new_ = false;
  std::mutex mutex_;
};

} // namespace rescue_perception
```

---

### 15.6 传感器同步节点

```cpp
// ============================================================================
// rescue_perception_core/src/sensor_sync.cpp
// ============================================================================

#include "rescue_perception_core/sensor_sync.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

namespace rescue_perception {

class SensorSyncImpl {
public:
  using RGBSub     = message_filters::Subscriber<sensor_msgs::msg::Image>;
  using ThermalSub = message_filters::Subscriber<sensor_msgs::msg::Image>;
  using LidarSub   = message_filters::Subscriber<sensor_msgs::msg::PointCloud2>;
  using RadarSub   = message_filters::Subscriber<RadarTrackArray>;
  using GasSub     = message_filters::Subscriber<GasSensorArray>;
  using OdomSub    = message_filters::Subscriber<nav_msgs::msg::Odometry>;

  // 关键传感器: 10ms 容忍窗口
  using CoreSyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,        // rgb
    sensor_msgs::msg::Image,        // thermal
    sensor_msgs::msg::PointCloud2,  // lidar
    nav_msgs::msg::Odometry         // odometry
  >;

  // 环境传感器: 50ms 容忍窗口
  using EnvSyncPolicy = message_filters::sync_policies::ApproximateTime<
    GasSensorArray,
    PMSensorArray
  >;

  SensorSyncImpl(rclcpp::Node* node)
    : node_(node)
  {
    // 关键传感器同步器
    rgb_sub_   = std::make_unique<RGBSub>(node, "/camera/front/image_raw");
    therm_sub_ = std::make_unique<ThermalSub>(node, "/thermal/image_raw");
    lidar_sub_ = std::make_unique<LidarSub>(node, "/lidar/points_raw");
    odom_sub_  = std::make_unique<OdomSub>(node, "/odometry/filtered");

    core_sync_ = std::make_unique<message_filters::Synchronizer<CoreSyncPolicy>>(
      CoreSyncPolicy(10), *rgb_sub_, *therm_sub_, *lidar_sub_, *odom_sub_);

    core_sync_->registerCallback(
      std::bind(&SensorSyncImpl::coreCallback, this,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4));

    // 发布器
    synced_pub_ = node_->create_publisher<SyncedSensorDataMsg>(
      "/perception/synced/data", 10);
  }

private:
  void coreCallback(
      const sensor_msgs::msg::Image::ConstSharedPtr& rgb,
      const sensor_msgs::msg::Image::ConstSharedPtr& thermal,
      const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar,
      const nav_msgs::msg::Odometry::ConstSharedPtr& odom)
  {
    SyncedSensorData synced;
    synced.stamp = node_->now();

    // 图像转换 (cv_bridge, 零拷贝模式)
    synced.rgb_image     = cv_bridge::toCvCopy(rgb, "bgr8")->image;
    synced.thermal_image = cv_bridge::toCvCopy(thermal, "mono16")->image;

    // 点云转换
    pcl::fromROSMsg(*lidar, synced.lidar_points);

    // 里程计
    synced.odom_pose = odom->pose.pose;
    synced.odom_twist = odom->twist.twist;

    // 发布同步数据
    auto msg = toRosMsg(synced);
    msg.header.stamp = node_->now();
    msg.header.frame_id = "base_link";
    synced_pub_->publish(msg);
  }

  rclcpp::Node* node_;
  std::unique_ptr<RGBSub> rgb_sub_;
  std::unique_ptr<ThermalSub> therm_sub_;
  std::unique_ptr<LidarSub> lidar_sub_;
  std::unique_ptr<OdomSub> odom_sub_;
  std::unique_ptr<message_filters::Synchronizer<CoreSyncPolicy>> core_sync_;
  rclcpp::Publisher<SyncedSensorDataMsg>::SharedPtr synced_pub_;
};

} // namespace rescue_perception
```

---

### 15.7 启动文件与运行时管理

```python
# ============================================================================
# rescue_perception_bringup/launch/perception_bringup.launch.py
# ============================================================================

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_dir = os.path.join(
        get_package_share_directory('rescue_perception_bringup'), 'config')

    # ---- Jetson 性能优化 ----
    set_maxn = ExecuteProcess(
        cmd=['sudo', 'nvpmodel', '-m', '0'],
        name='set_maxn_mode'
    )
    set_clocks = ExecuteProcess(
        cmd=['sudo', 'jetson_clocks'],
        name='set_jetson_clocks'
    )

    # ---- 组件化容器 (零拷贝进程内通信) ----
    container = ComposableNodeContainer(
        name='perception_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',  # 或 component_container_mt (多线程)
        composable_node_descriptions=[

            # 1. 传感器同步 (最先启动)
            ComposableNode(
                package='rescue_perception_core',
                plugin='rescue_perception::SensorSyncNode',
                name='sensor_sync_node',
                parameters=[os.path.join(config_dir, 'sensors.yaml')],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),

            # 2. 质量监控
            ComposableNode(
                package='rescue_perception_core',
                plugin='rescue_perception::QualityMonitorNode',
                name='quality_monitor_node',
                parameters=[os.path.join(config_dir, 'perception_master.yaml')],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),

            # 3. 检测节点 (并行)
            ComposableNode(
                package='rescue_perception_core',
                plugin='rescue_perception::RgbDetectorNode',
                name='rgb_detector_node',
                parameters=[
                    os.path.join(config_dir, 'perception_master.yaml'),
                    os.path.join(config_dir, 'detectors.yaml'),
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='rescue_perception_core',
                plugin='rescue_perception::ThermalDetectorNode',
                name='thermal_detector_node',
                parameters=[
                    os.path.join(config_dir, 'perception_master.yaml'),
                    os.path.join(config_dir, 'detectors.yaml'),
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='rescue_perception_core',
                plugin='rescue_perception::LidarClusterNode',
                name='lidar_cluster_node',
                parameters=[os.path.join(config_dir, 'detectors.yaml')],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='rescue_perception_core',
                plugin='rescue_perception::RadarTrackerNode',
                name='radar_tracker_node',
                parameters=[os.path.join(config_dir, 'detectors.yaml')],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),

            # 4. 融合节点
            ComposableNode(
                package='rescue_perception_core',
                plugin='rescue_perception::PersonFusionNode',
                name='person_fusion_node',
                parameters=[os.path.join(config_dir, 'fusion.yaml')],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='rescue_perception_core',
                plugin='rescue_perception::FireFusionNode',
                name='fire_fusion_node',
                parameters=[os.path.join(config_dir, 'fusion.yaml')],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),

            # 5. 管理节点
            ComposableNode(
                package='rescue_perception_core',
                plugin='rescue_perception::DegradationManagerNode',
                name='degradation_manager_node',
                parameters=[os.path.join(config_dir, 'degradation.yaml')],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='rescue_perception_core',
                plugin='rescue_perception::RiskAssessorNode',
                name='risk_assessor_node',
                parameters=[os.path.join(config_dir, 'risk.yaml')],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
    )

    return LaunchDescription([
        set_maxn,
        set_clocks,
        container,
    ])
```

---

### 15.8 头文件组织一览

```
rescue_perception_core/include/rescue_perception_core/
├── core/
│   ├── types.hpp              # 所有基础类型定义 (§15.1)
│   └── jetson_optimize.hpp    # Jetson 优化工具 (§15.5)
│
├── perception_pipeline.hpp    # 主管线调度器 (§15.2)
│
├── sync/
│   └── sensor_sync.hpp        # 传感器时间同步 (§15.6)
│
├── detectors/
│   ├── rgb_detector.hpp       # 可见光检测 (YOLO11 ONNX)
│   ├── thermal_detector.hpp   # 热成像检测 (§15.3.3)
│   ├── lidar_cluster.hpp      # LiDAR 聚类
│   └── radar_tracker.hpp      # 毫米波跟踪
│
├── estimators/
│   ├── smoke_estimator.hpp    # 烟雾估计
│   └── gas_risk.hpp           # 气体风险评估
│
├── fusion/
│   ├── person_fusion.hpp      # 人员融合 (§15.3.2)
│   ├── fire_fusion.hpp        # 火源融合 (§15.3.4)
│   ├── vehicle_fusion.hpp     # 车辆融合
│   ├── semantic_tracker.hpp   # 统一跟踪
│   └── semantic_localizer.hpp # 坐标转换
│
├── management/
│   ├── quality_monitor.hpp    # 环境质量评估
│   ├── degradation_manager.hpp # 退化管理 (§15.3.1)
│   └── risk_assessor.hpp      # 风险评估
│
└── inspection/                 # Phase 5: 巡检功能
    ├── crack_segmenter.hpp
    ├── water_detector.hpp
    └── facility_inspector.hpp
```

---

### 15.9 构建与运行命令速查

```bash
# ===== 首次构建 =====
cd ~/ros2_ws
colcon build --packages-select rescue_perception_msgs rescue_perception_core rescue_perception_bringup \
  --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_CUDA_ARCHITECTURES=87 \
               -DENABLE_TENSORRT=ON

# ===== 仅编译修改的包 =====
colcon build --packages-select rescue_perception_core --symlink-install

# ===== 运行 =====
# 完整模式
ros2 launch rescue_perception_bringup perception_bringup.launch.py

# 巡检模式 (启裂缝/积水/设施检测)
ros2 launch rescue_perception_bringup perception_inspection.launch.py

# 救援模式 (仅火焰/人员检测)
ros2 launch rescue_perception_bringup perception_rescue.launch.py

# 退化最小模式
ros2 launch rescue_perception_bringup perception_minimal.launch.py

# ===== 调试 =====
# 查看节点列表
ros2 node list

# 查看目标输出
ros2 topic echo /perception/target_observations

# 查看感知状态
ros2 topic echo /perception/status

# 查看热风险
ros2 topic echo /thermal/risk_level

# 可视化
rviz2 -d $(ros2 pkg prefix rescue_perception_bringup)/share/rescue_perception_bringup/rviz/perception.rviz

# ===== 性能分析 =====
# ROS 2 管线延迟
ros2 topic delay /perception/target_observations

# GPU 利用率
tegrastats

# TensorRT 推理时间 (在代码中通过 NvInfer::IExecutionContext::getProfiler 获取)
```

---

### 15.10 从伪代码到 C++ 的映射速查

| 伪代码模式 | C++ 实现 | 涉及节 |
|---|---|---|
| `ParallelLaunch:` | 实际串行调用 (延迟可接受) 或 `std::async` | 11.1, 15.2 |
| `if degradation_mode >= HEAVY_SMOKE` | `if (mode >= DegradationMode::HEAVY_SMOKE)` | 15.3.1 |
| `temp_matrix > threshold` | `cv::threshold(temp_fixed, hot_mask, ...)` | 15.3.3 |
| `connected_components_8(hot_mask)` | `cv::connectedComponentsWithStats(...)` | 15.3.3 |
| `匈牙利匹配` | 第三方库 `hungarian_algorithm.hpp` | 15.3.2 |
| `卡尔曼更新` | `Eigen::Matrix` 手写 8 维卡尔曼 | 15.3.2 |
| `EMA 平滑` | `smoke_score = 0.3*new + 0.7*old` | 11.6 |
| `clamp(x, 0, 1)` | `std::clamp(x, 0.0f, 1.0f)` | 各处 |
| `EMA 平滑` | `smoothed = α * current + (1-α) * previous` | 11.6 |
| `depth_30pct = percentile(points.z, 0.30)` | `std::nth_element` 部分排序 | 11.11 |
| `linear_slope(history)` | `Eigen::MatrixXd` 最小二乘拟合 | 11.12 |

---

## 十六、跨平台开发策略：Windows 编写 → Jetson 部署

核心原则：**算法与框架解耦**。所有感知算法是纯数值计算，与 ROS 2 / 操作系统无关。Windows 上开发和单测算法，Jetson 上集成 ROS 2 基础架构。

---

### 16.1 库分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    部署层 (Jetson Only)                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  ROS 2 Nodes / Launch Files / RViz                    │  │
│  │  - sensor_sync_node:  订阅硬件Topic, 调用core同步      │  │
│  │  - rgb_detector_node: 订阅图像, 调用core推理, 发布结果 │  │
│  │  - person_fusion_node:订阅检测, 调用core融合, 发布结果 │  │
│  │  - ...                                                │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  ROS 2 胶水层 (跨平台)                                 │  │
│  │  - 消息序列化/反序列化                                  │  │
│  │  - Topic 发布/订阅适配器                                │  │
│  │  - TF2 坐标变换适配器                                   │  │
│  │  - 参数服务器适配器                                     │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│              核心算法层 (纯 C++17, 全平台可编译)             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  rescue_perception_core/  (零 ROS 依赖)                │  │
│  │                                                        │  │
│  │  core/types.hpp          基础数据结构                  │  │
│  │  algorithms/                                          │  │
│  │    sensor_credibility.hpp   传感器可信度 (§11.2)       │  │
│  │    sensor_weights.hpp        权重分配 (§11.3)           │  │
│  │    degradation_fsm.hpp       退化状态机 (§11.4)         │  │
│  │    thermal_risk.hpp          热风险等级 (§11.5)         │  │
│  │    smoke_score.hpp           烟雾指数 (§11.6)           │  │
│  │    person_confidence.hpp     人员置信度 (§11.7)         │  │
│  │    fire_decision.hpp         火源决策 (§11.8)           │  │
│  │    environment_risk.hpp      环境风险 (§11.9)           │  │
│  │    safety_mapper.hpp         安全映射 (§11.10)          │  │
│  │    depth_validator.hpp       深度验证 (§11.11)          │  │
│  │    gas_assessor.hpp          气体评估 (§11.12)          │  │
│  │    hungarian_assoc.hpp       匈牙利关联                 │  │
│  │    kalman_8d.hpp             8维卡尔曼                  │  │
│  │  detectors/                                           │  │
│  │    rgb_detector_core.hpp     可见光检测核心              │  │
│  │    thermal_detector_core.hpp 热成像检测核心              │  │
│  │    lidar_cluster_core.hpp    LiDAR聚类核心              │  │
│  │    radar_tracker_core.hpp    毫米波跟踪核心              │  │
│  │  fusion/                                              │  │
│  │    person_fusion_core.hpp    人员融合核心 (§15.3.2)     │  │
│  │    fire_fusion_core.hpp      火源融合核心 (§15.3.4)     │  │
│  │    vehicle_fusion_core.hpp   车辆融合核心               │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  平台抽象层 (极薄)                                      │  │
│  │  - ImageContainer   (cv::Mat / GpuMat 统一包装)       │  │
│  │  - PointCloudContainer                                │  │
│  │  - Timer / Logger                                     │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 第三方依赖 (全平台可用)                        │
│  Eigen3  OpenCV  ONNX Runtime  PCL  gtest                  │
└─────────────────────────────────────────────────────────────┘
```

关键约束：

```
核心算法层:
  ❌ 不包含: rclcpp, std_msgs, sensor_msgs, tf2_ros
  ❌ 不包含: ros::NodeHandle, pub/sub, spin
  ✅ 仅依赖: C++17 STL, Eigen3, OpenCV, ONNX Runtime
  ✅ 函数签名: 纯数据 in → 纯数据 out (无副作用)

胶水层:
  ✅ 负责: ROS消息 ↔ 核心数据结构 互转
  ✅ 负责: Topic回调 → 调用核心算法 → 发布结果
  ✅ 负责: TF查询 → 传入核心算法的坐标变换参数
```

---

### 16.2 Windows 开发环境搭建

```powershell
# ====================================================================
# Windows 10/11 开发环境一键搭建
# ====================================================================

# 1. 安装 vcpkg (C++ 包管理器)
git clone https://github.com/microsoft/vcpkg.git C:\dev\vcpkg
cd C:\dev\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# 2. 安装核心依赖 (Windows x64)
.\vcpkg install `
  eigen3:x64-windows `
  opencv[core,imgproc,dnn,cuda]:x64-windows `
  onnxruntime-gpu:x64-windows `
  pcl:x64-windows `
  gtest:x64-windows `
  spdlog:x64-windows `
  nlohmann-json:x64-windows

# 3. (可选) 如果要测试 TensorRT → 仅 Jetson 可用
#    Windows 上用 ONNX Runtime DirectML 替代
#    .\vcpkg install onnxruntime-directml:x64-windows

# 4. Visual Studio 2022
#    - 工作负荷: "使用C++的桌面开发"
#    - 组件: CMake, C++17, Windows 10 SDK
```

---

### 16.3 跨平台 CMake 结构

```
rescue_perception/
├── CMakeLists.txt                    # 顶层: 分 core / ros2 / tests 三部分
├── cmake/
│   └── CompilerWarnings.cmake        # 统一编译警告配置
│
├── rescue_perception_core/           # ★ 核心算法库 (纯C++, 全平台)
│   ├── CMakeLists.txt                #   无 ROS 依赖
│   ├── include/rescue_perception_core/
│   │   ├── core/
│   │   │   └── types.hpp             #   基础类型
│   │   ├── algorithms/               #   纯算法头文件
│   │   │   ├── sensor_credibility.hpp
│   │   │   ├── degradation_fsm.hpp
│   │   │   ├── thermal_risk.hpp
│   │   │   ├── smoke_score.hpp
│   │   │   ├── person_confidence.hpp
│   │   │   ├── fire_decision.hpp
│   │   │   ├── environment_risk.hpp
│   │   │   ├── safety_mapper.hpp
│   │   │   ├── gas_assessor.hpp
│   │   │   ├── hungarian_assoc.hpp
│   │   │   └── kalman_8d.hpp
│   │   ├── detectors/                #   检测器核心
│   │   │   ├── rgb_detector_core.hpp
│   │   │   ├── thermal_detector_core.hpp
│   │   │   ├── lidar_cluster_core.hpp
│   │   │   └── radar_tracker_core.hpp
│   │   └── fusion/                   #   融合器核心
│   │       ├── person_fusion_core.hpp
│   │       ├── fire_fusion_core.hpp
│   │       └── vehicle_fusion_core.hpp
│   ├── src/                          #   算法实现 (.cpp)
│   │   ├── algorithms/
│   │   │   ├── sensor_credibility.cpp
│   │   │   ├── degradation_fsm.cpp
│   │   │   ├── thermal_risk.cpp
│   │   │   ├── smoke_score.cpp
│   │   │   ├── person_confidence.cpp
│   │   │   ├── fire_decision.cpp
│   │   │   └── ...
│   │   ├── detectors/
│   │   └── fusion/
│   └── tests/                        # ★ 单元测试 (Windows上主力运行)
│       ├── CMakeLists.txt
│       ├── test_thermal_risk.cpp
│       ├── test_degradation_fsm.cpp
│       ├── test_person_confidence.cpp
│       ├── test_fire_decision.cpp
│       ├── test_hungarian_assoc.cpp
│       ├── test_smoke_score.cpp
│       ├── test_safety_mapper.cpp
│       └── test_data/                # 测试用传感器数据
│           ├── sample_temp_matrix.csv
│           ├── sample_detections.json
│           └── sample_env_quality.json
│
├── rescue_perception_ros2/           # ★ ROS 2 胶水层 (仅 Linux/Jetson)
│   ├── CMakeLists.txt                #   需要 ROS 2 Humble
│   ├── package.xml
│   ├── include/rescue_perception_ros2/
│   │   ├── perception_node.hpp       #   主管线 ROS 2 节点
│   │   ├── adapters/                 #   消息适配器
│   │   │   ├── msg_converter.hpp     #     核心类型 ↔ ROS Msg
│   │   │   └── tf_adapter.hpp        #     TF2 ↔ Eigen::Isometry3d
│   │   └── nodes/                    #   各功能 ROS 2 节点
│   │       ├── rgb_detector_node.hpp
│   │       ├── thermal_detector_node.hpp
│   │       ├── person_fusion_node.hpp
│   │       └── ...
│   ├── src/
│   │   ├── perception_node.cpp
│   │   ├── adapters/
│   │   └── nodes/
│   ├── launch/
│   └── config/
│
└── rescue_perception_msgs/           # ROS 2 消息定义 (仅 Linux/Jetson)
    ├── CMakeLists.txt
    ├── package.xml
    └── msg/
```

顶层 `CMakeLists.txt`:

```cmake
# rescue_perception/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(rescue_perception VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- 选项 ----
option(BUILD_CORE_ONLY   "仅构建核心算法库 (Windows开发)" ON)
option(BUILD_ROS2_NODES  "构建 ROS 2 节点 (Jetson部署)"   OFF)
option(BUILD_TESTS       "构建单元测试"                    ON)
option(ENABLE_TENSORRT   "启用 TensorRT (仅 Jetson)"      OFF)

# ---- 核心库 (全平台) ----
add_subdirectory(rescue_perception_core)

# ---- ROS 2 胶水层 (条件编译) ----
if(BUILD_ROS2_NODES)
  add_subdirectory(rescue_perception_ros2)
endif()

# ---- 测试 ----
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(rescue_perception_core/tests)
endif()
```

核心算法库的 CMakeLists.txt（零 ROS 依赖）:

```cmake
# rescue_perception_core/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(rescue_perception_core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# ---- 第三方依赖 (全平台可用) ----
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(onnxruntime REQUIRED)   # Windows: DirectML / Jetson: TensorRT

# ---- 可选依赖 ----
if(ENABLE_TENSORRT)
  find_package(TensorRT REQUIRED)
  add_definitions(-DHAS_TENSORRT)
endif()

# ---- 核心算法库 ----
add_library(rescue_perception_core STATIC
  # 基础类型 (header-only)
  # include/core/types.hpp

  # 算法实现
  src/algorithms/sensor_credibility.cpp
  src/algorithms/degradation_fsm.cpp
  src/algorithms/thermal_risk.cpp
  src/algorithms/smoke_score.cpp
  src/algorithms/person_confidence.cpp
  src/algorithms/fire_decision.cpp
  src/algorithms/environment_risk.cpp
  src/algorithms/safety_mapper.cpp
  src/algorithms/gas_assessor.cpp
  src/algorithms/hungarian_assoc.cpp
  src/algorithms/kalman_8d.cpp

  # 检测器核心
  src/detectors/rgb_detector_core.cpp
  src/detectors/thermal_detector_core.cpp
  src/detectors/lidar_cluster_core.cpp
  src/detectors/radar_tracker_core.cpp

  # 融合器核心
  src/fusion/person_fusion_core.cpp
  src/fusion/fire_fusion_core.cpp
  src/fusion/vehicle_fusion_core.cpp
)

target_include_directories(rescue_perception_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(rescue_perception_core PUBLIC
  Eigen3::Eigen
  ${OpenCV_LIBS}
  onnxruntime::onnxruntime
)

# Windows 特有: 禁用 MSVC 警告
if(MSVC)
  target_compile_options(rescue_perception_core PRIVATE /W4 /wd4100 /wd4244)
endif()
```

---

### 16.4 纯算法接口设计（核心层无 ROS 依赖）

每个算法头文件遵循以下契约：

```cpp
// ====================================================================
// rescue_perception_core/include/rescue_perception_core/algorithms/degradation_fsm.hpp
// ====================================================================
//
// 纯算法接口示例：退化状态机
// - 无 ROS 依赖
// - 无 I/O 操作
// - 无全局状态（所有状态通过参数传入传出）
// - 输入/输出均为纯数据结构
// - 可在任何平台编译和单测

#pragma once

#include "rescue_perception_core/core/types.hpp"

namespace rescue_perception {
namespace algorithms {

// ── 退化状态机配置 ─────────────────────────────────────────
struct DegradationFsmConfig {
  // 进入阈值
  float smoke_enter_lowvis  = 0.15f;
  float smoke_enter_heavy   = 0.35f;
  float rgb_cred_low        = 0.50f;
  float brightness_low      = 30.0f;

  // 退出阈值 (滞后)
  float hysteresis          = 0.20f;   // 退出条件比进入宽松 20%
  float smoke_exit_clear    = 0.10f;   // HEAVY→LOW_VIS 的 smoke 上限
  float smoke_exit_lowvis   = 0.25f;   // LOW_VIS→CLEAR 的 smoke 上限

  // 振荡保护
  float oscillation_window_s  = 3.0f;
  int   oscillation_max_switches = 2;
};

// ── 退化状态机输出 ─────────────────────────────────────────
struct DegradationFsmOutput {
  DegradationMode   mode           = DegradationMode::CLEAR;
  DegradationEvent  event          = DegradationEvent::NONE;
  SensorWeights     weights;
  BehaviorRecommendation behavior;
  bool              is_locked      = false;   // 振荡保护锁定中
  float             lock_remaining_s = 0.0f;
};

// ── 状态机类 (纯计算, 无 I/O) ──────────────────────────────
class DegradationFsm {
public:
  explicit DegradationFsm(const DegradationFsmConfig& cfg = {});

  // ★ 核心接口: 纯函数式, 输入→计算→输出
  //   调用方负责: 传入当前环境质量, 接收输出并行动
  //   本类不负责: 订阅消息、发布消息、访问参数服务器
  DegradationFsmOutput update(
    const EnvironmentQuality& env_quality,
    float smoke_score
  );

  // 直接查询当前状态 (无副作用)
  DegradationMode currentMode() const { return state_.mode; }

  // 重置 (用于测试)
  void reset();

private:
  DegradationFsmConfig cfg_;
  DegradationFsmOutput state_;
  std::vector<std::pair<Timestamp, DegradationMode>> change_log_;
  Timestamp lock_until_{};
};

} // namespace algorithms
} // namespace rescue_perception
```

```cpp
// ====================================================================
// rescue_perception_core/include/rescue_perception_core/algorithms/person_confidence.hpp
// ====================================================================
//
// 纯算法: 人员多源置信度融合 (§11.7)

#pragma once

#include "rescue_perception_core/core/types.hpp"
#include <optional>

namespace rescue_perception {
namespace algorithms {

struct PersonConfidenceInput {
  struct SourceEvidence {
    bool  found      = false;
    float confidence = 0.0f;
    // 热成像专用
    float temp_min   = 0.0f;
    float temp_max   = 0.0f;
    // LiDAR 专用
    Vector3d dimensions = Vector3d::Zero();
    // 毫米波专用
    float velocity_ms = 0.0f;
    bool  has_micro_doppler = false;
    float rcs_dbsm   = 0.0f;
  };

  SourceEvidence rgb;
  SourceEvidence thermal;
  SourceEvidence lidar;
  SourceEvidence radar;

  SensorWeights  weights;
  DegradationMode mode = DegradationMode::CLEAR;
};

struct PersonConfidenceOutput {
  float       fused_confidence = 0.0f;
  ObjectClass person_class     = ObjectClass::OBSTACLE_GENERIC;
  bool        confirmed        = false;    // 多传感器确认
  uint8_t     source_mask      = 0;
  std::string diagnostic;                  // 可读的诊断字符串
};

// 纯函数: 无状态, 可并行调用
PersonConfidenceOutput fusePersonConfidence(const PersonConfidenceInput& input);

} // namespace algorithms
} // namespace rescue_perception
```

---

### 16.5 Windows 上的算法单元测试

Unit test on Windows using GTest, running independently of any ROS infrastructure:

```cpp
// ====================================================================
// rescue_perception_core/tests/test_person_confidence.cpp
// ====================================================================
//
// 在 Windows 上只需:
//   cmake --build . --target rescue_perception_core_tests
//   ctest -V
//
// 无需 ROS, 无需传感器, 纯算法验证

#include <gtest/gtest.h>
#include "rescue_perception_core/algorithms/person_confidence.hpp"

using namespace rescue_perception;
using namespace rescue_perception::algorithms;

// ═════════════════════════════════════════════════════════════════
// 测试1: 正常情况 — RGB + Thermal 双重确认 → 高置信 person
// ═════════════════════════════════════════════════════════════════
TEST(PersonConfidenceTest, RgbThermalConfirmed_Normal) {
  PersonConfidenceInput in;
  in.mode = DegradationMode::CLEAR;
  in.weights = {0.90f, 0.40f, 0.85f, 0.30f, 0.20f, 0.20f};

  // RGB 高置信检测
  in.rgb.found = true;
  in.rgb.confidence = 0.85f;

  // 热成像确认人体温度范围
  in.thermal.found = true;
  in.thermal.confidence = 0.70f;
  in.thermal.temp_min = 30.0f;
  in.thermal.temp_max = 37.0f;

  auto out = fusePersonConfidence(in);

  EXPECT_EQ(out.person_class, ObjectClass::PERSON_STANDING);
  EXPECT_TRUE(out.confirmed);
  EXPECT_GT(out.fused_confidence, 0.70f);
  EXPECT_EQ(out.source_mask & SRC_RGB, SRC_RGB);
  EXPECT_EQ(out.source_mask & SRC_THERMAL, SRC_THERMAL);
}

// ═════════════════════════════════════════════════════════════════
// 测试2: 极端情况 — 仅热成像 + 毫米波 → person_candidate
// ═════════════════════════════════════════════════════════════════
TEST(PersonConfidenceTest, ThermalRadarCandidate_HeavySmoke) {
  PersonConfidenceInput in;
  in.mode = DegradationMode::HEAVY_SMOKE;
  // 极端情况权重: RGB 几乎不可用
  in.weights = {0.03f, 0.85f, 0.10f, 0.85f, 0.50f, 0.60f};

  // RGB 完全不可信
  in.rgb.found = false;

  // 热成像检测到人体热源
  in.thermal.found = true;
  in.thermal.confidence = 0.50f;
  in.thermal.temp_min = 32.0f;
  in.thermal.temp_max = 38.0f;

  // 毫米波检测到同方向运动目标
  in.radar.found = true;
  in.radar.confidence = 0.55f;
  in.radar.velocity_ms = 1.2f;   // 人体步行速度
  in.radar.has_micro_doppler = true;

  auto out = fusePersonConfidence(in);

  EXPECT_EQ(out.person_class, ObjectClass::PERSON_CANDIDATE);
  EXPECT_GT(out.fused_confidence, 0.40f);
  // 极端情况下不要求 confirmed
  EXPECT_EQ(out.source_mask & SRC_THERMAL, SRC_THERMAL);
  EXPECT_EQ(out.source_mask & SRC_RADAR, SRC_RADAR);
}

// ═════════════════════════════════════════════════════════════════
// 测试3: 误检过滤 — 仅 RGB, 无热成像确认, 置信度不足 → unknown
// ═════════════════════════════════════════════════════════════════
TEST(PersonConfidenceTest, RgbOnlyLowConf_Rejected) {
  PersonConfidenceInput in;
  in.mode = DegradationMode::CLEAR;
  in.weights = {0.90f, 0.40f, 0.85f, 0.30f, 0.20f, 0.20f};

  in.rgb.found = true;
  in.rgb.confidence = 0.45f;   // 低于 0.5 的单源阈值

  in.thermal.found = false;    // 没有热成像确认
  in.lidar.found = false;
  in.radar.found = false;

  auto out = fusePersonConfidence(in);

  EXPECT_LT(out.fused_confidence, 0.40f);
  EXPECT_FALSE(out.confirmed);
}

// ═════════════════════════════════════════════════════════════════
// 测试4: 温度异常过滤 — 热源温度超出人体范围
// ═════════════════════════════════════════════════════════════════
TEST(PersonConfidenceTest, ThermalOutOfHumanRange_Downweighted) {
  PersonConfidenceInput in;
  in.mode = DegradationMode::CLEAR;
  in.weights = {0.90f, 0.40f, 0.85f, 0.30f, 0.20f, 0.20f};

  in.thermal.found = true;
  in.thermal.confidence = 0.80f;
  in.thermal.temp_min = 80.0f;  // 明显不是人体 (发热设备)
  in.thermal.temp_max = 120.0f;

  auto out = fusePersonConfidence(in);

  // 温度不合理 → 即使置信度高也是 person_candidate 而非 person
  EXPECT_NE(out.person_class, ObjectClass::PERSON_STANDING);
}
```

```cpp
// ====================================================================
// rescue_perception_core/tests/test_degradation_fsm.cpp
// ====================================================================

#include <gtest/gtest.h>
#include "rescue_perception_core/algorithms/degradation_fsm.hpp"

using namespace rescue_perception;
using namespace rescue_perception::algorithms;

// ── Helper: 创建环境质量 ─────────────────────────────────
static EnvironmentQuality makeEnv(float smoke, float rgb_c,
                                   float lidar_c, float thermal_c,
                                   float brightness = 500.0f) {
  EnvironmentQuality env;
  env.visibility_score = 1.0f - smoke;
  env.illuminance_lux  = brightness;
  env.sensors[0].credibility = rgb_c;       // RGB
  env.sensors[1].credibility = thermal_c;   // Thermal
  env.sensors[2].credibility = lidar_c;     // LiDAR
  env.sensors[3].credibility = 0.9f;        // Radar (通常正常)
  return env;
}

// ── 测试: CLEAR → LOW_VIS 进入 ───────────────────────────
TEST(DegradationFsmTest, ClearToLowVis_OnSmoke) {
  DegradationFsm fsm;
  ASSERT_EQ(fsm.currentMode(), DegradationMode::CLEAR);

  auto env = makeEnv(0.20f, 0.90f, 0.90f, 0.90f);  // smoke=0.20 > 0.15
  auto out = fsm.update(env, 0.20f);

  EXPECT_EQ(out.mode, DegradationMode::LOW_VISIBILITY);
  EXPECT_EQ(out.event, DegradationEvent::ENTER_LOW_VIS);
}

// ── 测试: 滞后 — smoke 回到 0.12 但不够低, 不退出 ─────────
TEST(DegradationFsmTest, Hysteresis_PreventsOscillation) {
  DegradationFsm fsm;

  // 先进入 LOW_VIS
  auto env1 = makeEnv(0.20f, 0.90f, 0.90f, 0.90f);
  fsm.update(env1, 0.20f);
  ASSERT_EQ(fsm.currentMode(), DegradationMode::LOW_VISIBILITY);

  // smoke 降到 0.12 (> 0.10 退出阈值) → 不退出
  auto env2 = makeEnv(0.12f, 0.90f, 0.90f, 0.90f);
  auto out = fsm.update(env2, 0.12f);
  EXPECT_EQ(out.mode, DegradationMode::LOW_VISIBILITY);  // 仍停留

  // smoke 降到 0.08 (< 0.10) → 退出
  auto env3 = makeEnv(0.08f, 0.90f, 0.90f, 0.90f);
  auto out2 = fsm.update(env3, 0.08f);
  EXPECT_EQ(out2.mode, DegradationMode::CLEAR);
}

// ── 测试: 多传感器失效 → PERCEPTION_DEGRADED ─────────────
TEST(DegradationFsmTest, AllSensorsFail_GoesToDegraded) {
  DegradationFsm fsm;

  // LOW_VIS
  auto env1 = makeEnv(0.20f, 0.90f, 0.90f, 0.90f);
  fsm.update(env1, 0.20f);

  // HEAVY_SMOKE
  auto env2 = makeEnv(0.40f, 0.90f, 0.90f, 0.90f);
  fsm.update(env2, 0.40f);
  ASSERT_EQ(fsm.currentMode(), DegradationMode::HEAVY_SMOKE);

  // 全部传感器失效
  auto env3 = makeEnv(0.40f, 0.05f, 0.10f, 0.15f);
  auto out = fsm.update(env3, 0.40f);

  EXPECT_EQ(out.mode, DegradationMode::PERCEPTION_DEGRADED);
}

// ── 测试: 权重随模式变化 ─────────────────────────────────
TEST(DegradationFsmTest, WeightsChangeWithMode) {
  DegradationFsm fsm;

  // CLEAR 权重
  auto env_clear = makeEnv(0.05f, 0.90f, 0.90f, 0.90f);
  auto out_clear = fsm.update(env_clear, 0.05f);
  EXPECT_GT(out_clear.weights.rgb, 0.70f);       // RGB 主导
  EXPECT_GT(out_clear.weights.lidar, 0.50f);     // LiDAR 主导

  // HEAVY_SMOKE 权重
  auto env_smoke = makeEnv(0.50f, 0.10f, 0.20f, 0.80f);
  // 先逐步退化
  fsm.update(makeEnv(0.20f, 0.90f, 0.90f, 0.90f), 0.20f);
  fsm.update(makeEnv(0.40f, 0.90f, 0.90f, 0.90f), 0.40f);
  auto out_heavy = fsm.update(env_smoke, 0.50f);

  EXPECT_LT(out_heavy.weights.rgb, 0.10f);       // RGB 大幅降权
  EXPECT_GT(out_heavy.weights.radar, 0.50f);     // 毫米波提升
}
```

---

### 16.6 Windows → Jetson 迁移路径

```
Phase A: Windows 算法开发 (第1-3周)
┌──────────────────────────────────────────────────────────┐
│  环境: VS 2022 + vcpkg + GTest                           │
│  工作:                                                    │
│    1. 编写所有算法头文件 (algorithms/*.hpp)               │
│    2. 实现所有算法 .cpp                                   │
│    3. 编写单元测试 (tests/test_*.cpp)                     │
│    4. 使用模拟数据跑通全部测试                             │
│    5. cmake --build . && ctest -V  全绿                  │
│                                                          │
│  ✓ 验收: 50+ 个测试用例全部通过                           │
│  ✓ 不依赖: ROS 2, Linux, Jetson, 任何传感器硬件           │
└──────────────────────────────────────────────────────────┘
        │
        │  代码通过 git 提交
        ▼
Phase B: Linux 模拟环境集成 (第4周)
┌──────────────────────────────────────────────────────────┐
│  环境: Ubuntu 22.04 (VM/WSL2) + ROS 2 Humble              │
│  工作:                                                    │
│    1. 编译核心库 (cmake, 不改代码)                        │
│    2. 编写 ROS 2 节点外壳 (§15.4)                         │
│    3. 使用 rosbag 回放数据进行集成测试                     │
│    4. 验证消息发布/订阅、TF 坐标变换                       │
│                                                          │
│  ✓ 验收: rosbag 回放下管线端到端运行                       │
└──────────────────────────────────────────────────────────┘
        │
        │  代码通过 git 提交
        ▼
Phase C: Jetson AGX Orin 部署 (第5-6周)
┌──────────────────────────────────────────────────────────┐
│  环境: Jetson AGX Orin + JetPack 6.0 + ROS 2 Humble       │
│  工作:                                                    │
│    1. 交叉编译 或 本地编译 (colcon)                       │
│    2. ONNX → TensorRT Engine 转换                         │
│    3. 传感器驱动联调                                       │
│    4. 性能调优: FP16, CUDA Stream, 内存池                  │
│    5. 实车测试                                             │
│                                                          │
│  ✓ 验收: 端到端延迟 ≤ 100ms @10Hz                          │
└──────────────────────────────────────────────────────────┘
```

---

### 16.7 Windows 上模拟传感器数据进行测试

```cpp
// ====================================================================
// rescue_perception_core/tests/test_data/sample_generator.hpp
// ====================================================================
//
// 在 Windows 上无真实传感器时, 用模拟数据驱动算法测试

#pragma once

#include "rescue_perception_core/core/types.hpp"
#include <random>
#include <vector>

namespace rescue_perception {
namespace testing {

// ── 模拟温度矩阵 (640x512) ──────────────────────────
inline cv::Mat generateTemperatureMatrix(
    float ambient_temp = 25.0f,
    const std::vector<std::pair<cv::Rect, float>>& hotspots = {})
{
  cv::Mat temp(512, 640, CV_32FC1, cv::Scalar(ambient_temp));

  // 添加随机噪声
  cv::Mat noise(512, 640, CV_32FC1);
  cv::randn(noise, 0.0, 0.5);  // σ=0.5°C
  temp += noise;

  // 添加模拟热点
  for (const auto& [roi, t_max] : hotspots) {
    cv::Mat region = temp(roi);
    // 高斯热源分布
    for (int r = 0; r < roi.height; r++) {
      for (int c = 0; c < roi.width; c++) {
        float dx = (c - roi.width/2.0f) / (roi.width/4.0f);
        float dy = (r - roi.height/2.0f) / (roi.height/4.0f);
        float weight = std::exp(-(dx*dx + dy*dy));
        region.at<float>(r, c) += (t_max - ambient_temp) * weight;
      }
    }
  }

  return temp;
}

// ── 模拟环境质量 ─────────────────────────────────────
inline EnvironmentQuality makeClearEnv() {
  EnvironmentQuality env;
  env.visibility = VisibilityLevel::CLEAR;
  env.visibility_score = 0.95f;
  env.illuminance_lux = 500.0f;
  env.smoke_density = 0.02f;
  for (auto& s : env.sensors) {
    s.credibility = 0.90f;
    s.effective_ratio = 0.95f;
  }
  return env;
}

inline EnvironmentQuality makeHeavySmokeEnv() {
  EnvironmentQuality env;
  env.visibility = VisibilityLevel::HEAVY_SMOKE;
  env.visibility_score = 0.25f;
  env.smoke_density = 0.70f;
  env.smoke_coverage = 0.85f;
  // RGB 大幅退化
  env.sensors[0].credibility = 0.08f;  // RGB
  env.sensors[2].credibility = 0.20f;  // LiDAR
  // Thermal 和 Radar 仍可用
  env.sensors[1].credibility = 0.75f;  // Thermal
  env.sensors[3].credibility = 0.80f;  // Radar
  return env;
}

// ── 模拟 3D 检测结果 ─────────────────────────────────
inline Detection3D makePersonDetection(
    SensorType source, float conf,
    float x, float y, float z,
    float temp_max = 36.0f, float temp_min = 30.0f)
{
  Detection3D det;
  det.source = source;
  det.confidence = conf;
  det.position = Vector3d(x, y, z);
  det.dimensions = Vector3d(0.5, 0.4, 1.7);  // 典型人体尺寸
  det.temperature_max = temp_max;
  det.temperature_min = temp_min;

  switch (source) {
    case SensorType::RGB:
      det.class_id = ObjectClass::PERSON_STANDING;
      break;
    case SensorType::THERMAL:
      det.class_id = ObjectClass::PERSON_STANDING;
      break;
    case SensorType::LIDAR_3D:
      det.class_id = ObjectClass::OBSTACLE_GENERIC;
      break;
    case SensorType::RADAR_4D:
      det.class_id = ObjectClass::OBSTACLE_GENERIC;
      det.velocity = Vector3d(1.0, 0.0, 0.0);  // 1m/s 行走
      break;
    default:
      break;
  }
  return det;
}

} // namespace testing
} // namespace rescue_perception
```

```cpp
// ====================================================================
// rescue_perception_core/tests/test_fire_decision.cpp (模拟数据集成测试)
// ====================================================================

#include <gtest/gtest.h>
#include "rescue_perception_core/algorithms/fire_decision.hpp"
#include "test_data/sample_generator.hpp"

using namespace rescue_perception;
using namespace rescue_perception::algorithms;
using namespace rescue_perception::testing;

TEST(FireDecisionTest, ClearFire_FromTemperatureMatrix) {
  // 生成含有明火的温度矩阵
  // 在图像中央 (300,240) 处有一个 200°C 的高温区域
  cv::Rect fire_roi(280, 220, 40, 40);
  auto temp_mat = generateTemperatureMatrix(25.0f, {
    {fire_roi, 350.0f}
  });

  // 传递到热成像检测核心
  ThermalDetectConfig cfg;
  cfg.min_hotspot_area = 9;
  cfg.bg_window_frames = 30;

  ThermalDetectorCore detector(cfg);
  auto result = detector.process(temp_mat);

  // 验证: 应检测到至少1个热点
  ASSERT_GE(result.hotspots.size(), 1);

  // 验证: 最高温度接近 350°C
  auto& hs = result.hotspots[0];
  EXPECT_GT(hs.T_max, 200.0f);
  EXPECT_EQ(hs.type, ThermalHotspot::FIRE_SOURCE);

  // 验证: 热风险等级 ≥ 3 (高危)
  EXPECT_GE(result.risk_level, 3);
}
```

---

### 16.8 CMake 预设：一键切换平台

```cmake
# rescue_perception/CMakePresets.json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "windows-dev",
      "displayName": "Windows Development (core + tests only)",
      "binaryDir": "${sourceDir}/build/windows",
      "cacheVariables": {
        "BUILD_CORE_ONLY": "ON",
        "BUILD_ROS2_NODES": "OFF",
        "BUILD_TESTS": "ON",
        "ENABLE_TENSORRT": "OFF",
        "CMAKE_TOOLCHAIN_FILE": "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake",
        "VCPKG_TARGET_TRIPLET": "x64-windows"
      }
    },
    {
      "name": "linux-sim",
      "displayName": "Linux Simulation (core + ROS 2 + rosbag test)",
      "binaryDir": "${sourceDir}/build/linux_sim",
      "cacheVariables": {
        "BUILD_CORE_ONLY": "OFF",
        "BUILD_ROS2_NODES": "ON",
        "BUILD_TESTS": "ON",
        "ENABLE_TENSORRT": "OFF"
      }
    },
    {
      "name": "jetson-orin",
      "displayName": "Jetson AGX Orin (full deployment)",
      "binaryDir": "${sourceDir}/build/jetson",
      "cacheVariables": {
        "BUILD_CORE_ONLY": "OFF",
        "BUILD_ROS2_NODES": "ON",
        "BUILD_TESTS": "ON",
        "ENABLE_TENSORRT": "ON",
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CUDA_ARCHITECTURES": "87"
      }
    }
  ]
}
```

使用方式：

```bash
# Windows 开发
cmake --preset windows-dev
cmake --build build/windows --config Release
ctest --test-dir build/windows -V

# Linux 模拟 (WSL2 / VM)
cmake --preset linux-sim
colcon build --symlink-install

# Jetson 部署
cmake --preset jetson-orin
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

---

### 16.9 核心算法可移植性检查清单

在 Windows 上开发时，确保每个 `.hpp/.cpp` 满足以下条件：

| 检查项 | 合规 | 不合规 |
|---|---|---|
| 头文件依赖 | `#include <Eigen/Dense>` | `#include <rclcpp/rclcpp.hpp>` |
| 函数签名 | `float compute(const Input& in)` | `void compute(ros::NodeHandle& nh)` |
| 错误处理 | `std::optional<T>` / 异常 | `ROS_ERROR()` / `ROS_ASSERT()` |
| 日志输出 | `std::cerr` / `spdlog::` | `RCLCPP_INFO()` / `ROS_INFO()` |
| 时间类型 | `std::chrono::steady_clock::time_point` | `ros::Time` / `rclcpp::Time` |
| 坐标类型 | `Eigen::Vector3d` | `geometry_msgs::msg::Point` |
| 参数配置 | 构造函数传入 struct | `node->get_parameter()` |
| 平台特定 | `#ifdef _WIN32` ... `#else` ... | 直接调用 Linux syscall |

---

### 16.10 Windows 开发工作流速查

```powershell
# ====================================================================
# 典型开发循环 (Windows)
# ====================================================================

# 1. 修改算法代码
#    code rescue_perception_core/src/algorithms/thermal_risk.cpp

# 2. 编译 (仅核心库+测试)
cmake --build build\windows --config Release --target rescue_perception_core_tests

# 3. 运行全部测试
ctest --test-dir build\windows -C Release -V

# 4. 运行特定测试
build\windows\rescue_perception_core\tests\Release\test_thermal_risk.exe --gtest_filter="*ClearFire*"

# 5. 如果有失败, 调试
#    VS 2022 → 打开 build\windows\rescue_perception.sln → 设置启动项目 → F5

# 6. 提交 (git)
git add rescue_perception_core/src/algorithms/thermal_risk.cpp
git add rescue_perception_core/tests/test_thermal_risk.cpp
git commit -m "fix(thermal): correct hotspot circularity formula"

# 7. 在 Jetson 上拉取并编译 (另开终端)
ssh jetson@192.168.1.100
cd ~/ros2_ws/src/rescue_perception
git pull
cd ~/ros2_ws
colcon build --packages-select rescue_perception_core --symlink-install
```
