import React from "react";

import {
  DataTable, ReportSection, RichNarrative, SortableItem, SortableRegion, useDataApp,
} from "../../data-app-public.jsx";

const sectionOrder = [
  "system-route", "device-training-matrix", "data-preparation", "training-method",
  "fusion-plan", "verification-plan", "deployment-plan", "delivery-roadmap", "risk-plan",
];

const sourcePreviews = {
  "https://docs.ultralytics.com/models": {
    title: "Ultralytics supported models",
    summary: "YOLO26 is positioned for new edge-oriented projects, while YOLO11 remains the mature production alternative with train, validation, prediction and export support.",
    source: "Ultralytics documentation", date: "2026-09-18", approvedForReport: true,
  },
  "https://oem.flir.com/en-ca/solutions/automotive/adas-dataset-form/": {
    title: "FLIR ADAS thermal dataset",
    summary: "Paired visible and thermal data with people and vehicle annotations, including higher-bit-depth pre-AGC thermal frames.",
    source: "Teledyne FLIR", approvedForReport: true,
  },
  "https://github.com/gaia-solutions-on-demand/DFireDataset": {
    title: "D-Fire dataset",
    summary: "YOLO-format fire and smoke boxes plus a substantial negative-image set, suitable for the first RGB fire baseline.",
    source: "D-Fire maintainers", approvedForReport: true,
  },
  "https://github.com/kennedyk1/MID-3K": {
    title: "MID-3K multimodal dataset",
    summary: "Mobile-robot RGB, thermal, LiDAR-derived depth and intensity scenes with person annotations; useful for person fusion, not vehicle or fire training.",
    source: "MID-3K maintainers", approvedForReport: true,
  },
};

export function ReportContent() {
  const { snapshot, queries, visible, canEdit, mode, appTitle, setAppTitle } = useDataApp();
  const datasets = queries.dataset_catalog?.rows ?? [];
  const roadmap = queries.roadmap?.rows ?? [];
  const subtitle = snapshot.report?.subtitle;

  return <article className="report-content perception-roadmap" aria-label="环境感知系统开发规划">
    <header className="report-hero">
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}>
        {appTitle}
      </h1>
      <RichNarrative id="report-subtitle" value={subtitle} className="report-deck" label="编辑副标题" />
    </header>

    <ReportSection id="executive-summary" queryId="dataset_catalog" sourceRows={datasets}
      title="执行结论" showHeading={false} className="report-summary">
      <RichNarrative id="summary-body" sourcePreviews={sourcePreviews} value={`## 执行结论

- **继续以YOLO11作为当前主线模型。** 现有代码已完成YOLO11推理接口，先用它建立RGB人员车辆、RGB火情和热成像人员车辆三条可复现基线。YOLO26只在数据划分固定后做同条件对照；模型更新本身不作为论文创新。[官方模型定位](https://docs.ultralytics.com/models)
- **近期真正需要训练的模型只有三类：** RGB火焰烟雾、热成像人员车辆，以及按实际场景微调的RGB人员车辆。LiDAR、毫米波、二维LiDAR和环境量先用规则、聚类与跟踪形成闭环。
- **融合采用目标级后融合。** 所有检测器先转换成统一Detection3D，再按人员、车辆、火情分别关联；没有同场景全传感器公开数据时，用小规模自采同步序列完成融合验证。
- **开发顺序由可验证性驱动。** 先冻结数据、类别、标定和单设备指标，再接入多设备；不能在单模型误差不清楚时直接调融合权重。`} />
    </ReportSection>

    <SortableRegion id="report:sections" label="规划章节" variant="stack"
      authoredOrder={sectionOrder} className="report-sortable-sections">

      {visible("system-route") && <SortableItem id="system-route" label="系统实现主线" kind="section">
        <RichNarrative id="system-route-body" value={`## 1. 系统实现主线

系统应保持“独立检测—统一表示—语义融合—统一跟踪—风险决策”的边界：

1. **输入和同步：** RGB、热成像、3D LiDAR、4D毫米波、二维LiDAR、气体、PM、温湿度和位姿统一时间戳。
2. **质量评估：** 先计算亮度、烟雾遮挡、点云有效率、同步误差和设备健康状态。
3. **独立检测：** 每种设备只输出自身能够直接证明的结果，不提前制造“融合结论”。
4. **统一接口：** 目标统一转换为Detection3D，至少包含类别、置信度、位置、速度、尺寸、时间戳、来源和协方差。
5. **目标级融合：** 人员、车辆、火情使用不同的证据组合和确认条件。
6. **跟踪与决策：** 统一目标ID、轨迹、风险等级和限速/停车建议。

现有主管线已经具备这一骨架。后续开发应围绕完善真实后端、标定、数据集和验证，而不是重写架构。`} />
      </SortableItem>}

      {visible("device-training-matrix") && <SortableItem id="device-training-matrix" label="设备训练矩阵" kind="table">
        <ReportSection id="device-training-matrix" queryId="dataset_catalog" sourceRows={datasets}
          title="2. 不同设备应分别训练或标定" showHeading={false}
          description="每行对应一个设备任务、首选数据来源和建议实现方法。">
          <RichNarrative id="device-matrix-intro" sourcePreviews={sourcePreviews} value={`## 2. 不同设备应分别训练或标定

公开数据主要解决预训练和算法可行性，真实设备数据解决域差异、坐标外参、量纲和最终验收。MID-3K只承担人员多模态验证；车辆由FLIR、nuScenes、VoD等承担；火情由D-Fire、热成像辐射数据和环境量承担。`} />
          <DataTable id="device-training-table" queryId="dataset_catalog" rows={datasets}
            columns={[
              { key: "device", label: "设备" }, { key: "task", label: "任务" },
              { key: "primary", label: "首选数据" }, { key: "secondary", label: "补充数据" },
              { key: "method", label: "首版方法" }, { key: "priority", label: "优先级" },
            ]} />
        </ReportSection>
      </SortableItem>}

      {visible("data-preparation") && <SortableItem id="data-preparation" label="数据准备" kind="section">
        <RichNarrative id="data-preparation-body" sourcePreviews={sourcePreviews} value={`## 3. 数据准备与版本管理

### 3.1 统一类别，但不强迫每个数据集拥有全部类别

- RGB人员车辆：person、car、truck、bus、motorcycle。
- RGB火情：flame、smoke。
- 热成像目标：person、car、truck、bus、motorcycle。
- 系统内部再映射为ObjectClass；训练模型的局部类别编号不要直接暴露给融合层。

MID-3K只有person时只保留person标注；FLIR同时提供人员和车辆。禁止为缺失类别人工生成空目标。

### 3.2 按采集序列划分

视频连续帧必须按序列划分为训练、验证和测试，建议70%/15%/15%。同一视频相邻帧不得跨集合，否则mAP会因场景泄漏虚高。公开测试集与自采测试集同时保留：前者用于论文可比性，后者用于设备适用性。

### 3.3 数据质量检查

每次训练前固定检查损坏文件、空标签、越界框、重复图像、类别映射、极小框、标注遗漏和许可证。火情数据还要专门检查灯光、夕阳、蒸汽、灰尘、焊接和红色物体等困难负样本。

### 3.4 建议的数据版本

建立data-v1-baseline、data-v2-domain、data-v3-synchronized三个版本。数据清单记录来源、许可证、哈希、类别数、序列划分和转换脚本；模型权重只引用不可变的数据版本。`} />
      </SortableItem>}

      {visible("training-method") && <SortableItem id="training-method" label="训练方法" kind="section">
        <RichNarrative id="training-method-body" value={`## 4. 分设备训练方法

### 4.1 RGB人员车辆

以yolo11n.pt的COCO权重为起点。第一版直接部署官方权重验证链路，第二版用FLIR RGB、MID-3K RGB和自采机器人视角数据微调。先冻结数据划分，训练YOLO11n；若小目标召回不足，再训练YOLO11s，不先修改网络结构。

### 4.2 RGB火焰烟雾

以D-Fire为主训练集，TunnelFire2024只作为辅助或外部测试，FASDD-CV按场景抽样扩充。训练检测版flame/smoke基线，再单独尝试烟雾分割。火情验收优先看Recall、困难负样本每小时误报数和连续帧确认延迟，而不是只看mAP。

### 4.3 热成像人员车辆

以FLIR ADAS Thermal为主，MID-3K/LLVIP/KAIST补充行人。单通道热图复制为三通道或修改首层卷积，但训练和部署必须使用一致预处理。保留原始高位深数据；AGC显示图只服务检测，不能当作真实摄氏温度。

### 4.4 热成像火源

首版不训练：使用辐射温度矩阵进行坏点修复、背景估计、双阈值、连通域、温升速度和面积增长分析。后续用IGNITE/FLAME3或自采数据训练分割模型，只提供候选区域；最终火源结论仍由真实温度和时序确认。

### 4.5 LiDAR与毫米波

LiDAR首版保留RANSAC地面分割、聚类、PCA包围盒和尺寸规则；毫米波首版保留杂波过滤、卡尔曼跟踪、RCS、速度和微多普勒规则。只有在自采误差分析证明规则上限不足时，才分别引入CenterPoint/PointPillars或雷达点云网络。AWR1843的RA/RD/RAD张量与SR75点云是两条不同技术路线，不共用训练模型。

### 4.6 训练统一规范

每个模型至少训练三次随机种子，保存best/last、训练配置、数据版本和环境锁定文件。基础对照统一使用imgsz=640、相同增强和早停策略；YOLO11/YOLO26对照只改变模型，其余条件保持一致。`} />
      </SortableItem>}

      {visible("fusion-plan") && <SortableItem id="fusion-plan" label="多设备融合" kind="section">
        <RichNarrative id="fusion-plan-body" value={`## 5. 多设备融合实施方案

### 5.1 融合前的四个前置条件

1. 硬件时间同步或已知时间偏差；建议相机/热成像小于50 ms，运动目标关联不超过100 ms。
2. 完成camera、thermal、LiDAR、radar到base_link的外参标定。
3. 每个检测器输出统一坐标和协方差；没有真实深度时必须标记depth_valid=false。
4. 质量监控能动态输出设备可信度，不能把失效传感器继续等权融合。

### 5.2 人员融合

RGB给出语义类别，热成像提供人体温区，LiDAR提供尺寸和三维位置，毫米波提供烟雾中的速度与轨迹。先用空间门控和匈牙利匹配，再按传感器质量加权。至少两类独立证据才确认人员；重烟时允许“热成像+毫米波”替代RGB。

### 5.3 车辆融合

RGB确定car/truck/bus，LiDAR提供车辆尺寸和占道比例，毫米波提供距离、速度和高RCS证据。车辆内部的火焰框不能通过跨类别NMS删除；车辆目标与火情事件是两个可同时存在的语义实体。

### 5.4 火情融合

RGB火焰/烟雾只生成视觉候选；热成像高温和温升提供物理证据；PM、CO和其他气体提供环境趋势。建议采用事件状态机：无事件→疑似→确认→高危→恢复，加入进入和退出迟滞，避免单帧报警抖动。

### 5.5 没有统一公开数据集时如何验证

公开数据分别训练单设备模型；再自采20–50段同步序列，每段30–120秒，覆盖正常、人员、车辆、弱光、遮挡、烟雾、热源、真实小火和传感器掉线。融合参数只在开发集调整，最终测试序列一次性评估。`} />
      </SortableItem>}

      {visible("verification-plan") && <SortableItem id="verification-plan" label="验证指标" kind="section">
        <RichNarrative id="verification-plan-body" value={`## 6. 验证体系与阶段门槛

### 单模型指标

- 检测：mAP50-95、Precision、Recall、F1、按目标尺寸分组的AP。
- 火情：小火焰Recall、烟雾Recall、每小时误报数、首次发现时间。
- 热成像：昼夜、距离、遮挡、AGC模式和不同环境温度的分组结果。
- 点云/雷达：位置误差、速度误差、分类准确率、轨迹中断率和ID切换数。

### 融合指标

- 人员/车辆：目标级Precision、Recall、MOTA/IDF1、三维位置RMSE。
- 火情：事件级Precision/Recall、确认延迟、误报警次数、漏报次数。
- 鲁棒性：关闭任意一个传感器后的性能下降；时间偏移50/100/200 ms敏感性；轻烟/重烟下的性能变化。

### 系统指标

- Jetson AGX Orin上的平均和P95延迟、FPS、显存、功耗与温度。
- 输入丢帧、模型超时、设备掉线和重启恢复时间。
- 安全规则是否在失去定位、重烟和最近障碍过近时正确限速或停车。

进入融合阶段前，单设备模型必须在独立测试集上达到预设门槛，并完成错误样例分类。没有错误分析的高mAP不能作为进入下一阶段的依据。`} />
      </SortableItem>}

      {visible("deployment-plan") && <SortableItem id="deployment-plan" label="部署方案" kind="section">
        <RichNarrative id="deployment-plan-body" sourcePreviews={sourcePreviews} value={`## 7. 部署与模型选择

当前部署建议：RGB人员车辆使用YOLO11n，RGB火情从YOLO11n起步并以YOLO11s作为召回候选，热成像人员车辆使用YOLO11n；烟雾和热源分割放在第二阶段。训练完成后导出ONNX和TensorRT FP16，逐模型验证数值一致性，再做多模型调度。

运行频率可设为：人员车辆10–15 Hz、火情5–10 Hz、热成像10 Hz、LiDAR/雷达跟随设备频率、融合20 Hz。火情模型可复用最近结果，不要求和人员车辆模型每帧同时推理。

YOLO26只在数据与指标固定后加入对照。用相同数据、增强、随机种子、图像尺寸和TensorRT精度比较YOLO11n/s与YOLO26n/s。只有当YOLO26在实机Recall、P95延迟、显存和稳定性上形成明确优势，才替换部署模型。Ultralytics代码和权重涉及AGPL-3.0/企业许可双轨，闭源商业化前需单独处理许可证。`} />
      </SortableItem>}

      {visible("delivery-roadmap") && <SortableItem id="delivery-roadmap" label="24周路线图" kind="table">
        <ReportSection id="delivery-roadmap" queryId="roadmap" sourceRows={roadmap}
          title="8. 建议的24周交付路线" showHeading={false}
          description="阶段周期为建议值；设备到位、标注和算力会改变排期。">
          <RichNarrative id="roadmap-intro" value={`## 8. 建议的24周交付路线

每个阶段都设置进入下一阶段的门槛，避免同时训练全部模型而无法定位问题。`} />
          <DataTable id="roadmap-table" queryId="roadmap" rows={roadmap}
            columns={[
              { key: "phase", label: "阶段" }, { key: "weeks", label: "建议周期" },
              { key: "deliverable", label: "交付物" }, { key: "gate", label: "阶段门槛" },
            ]} />
        </ReportSection>
      </SortableItem>}

      {visible("risk-plan") && <SortableItem id="risk-plan" label="主要风险" kind="section">
        <RichNarrative id="risk-plan-body" value={`## 9. 主要风险与控制办法

- **公开数据域差异：** 用跨数据集测试和实际设备小样本微调处理，不能只报告同分布测试集。
- **无统一多模态数据：** 不阻塞单模型训练；融合必须依靠自采同步序列验证。
- **热图不等于温度：** 保存辐射温度矩阵，AGC图只用于视觉模型。
- **雷达输出格式不一致：** 先确认设备输出是目标列表、4D点云还是ADC/RAD张量，再选择VoD、TJ4DRadSet或CARRADA。
- **重复视频帧导致指标虚高：** 严格按序列划分并做感知哈希去重。
- **双模型算力竞争：** 分频调度、FP16 TensorRT、异步队列和P95延迟监控。
- **论文创新不足：** 不把YOLO版本升级当创新；优先研究退化感知动态权重、热成像温度证据、烟雾时序与多设备失效鲁棒性。

最终验收应交付：数据清单、转换脚本、训练配置、权重、单模型评估、标定文件、同步rosbag、融合消融、TensorRT基准、故障注入结果和完整复现实验说明。`} />
      </SortableItem>}
    </SortableRegion>
  </article>;
}
