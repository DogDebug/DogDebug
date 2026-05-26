# DogDebug

# 🤖 足式机器人智能巡检系统

> **2026年中国高校智能机器人创意大赛 - 足式机器人挑战赛专项赛（预选赛）**

> **参赛队伍：** 中南民族大学 - 修狗不修BUG队

> **参赛组别：** 四足机器人（大型组）

## 🌟 项目简介

本项目针对足式机器人（如“绝影”系列）在复杂工业场景下的应用需求，开发了一套集**高鲁棒视觉感知、ROS节点通信、以及高精度自主导航**于一体的智能巡检系统。

项目涵盖了仪表盘状态检测、多形态物块识别、ROS底层通信机制以及 Gazebo 仿真环境下的定点建图与轨迹规划。所有代码均采用模块化设计，具备极高的稳定性和可迁移性。

## ✨ 核心创新点 (Key Innovations)

1. **YOLOv8 + OpenCV 双轨防抖视觉融合框架**
   - 摒弃了单一传统视觉极易受环境光照、反光干扰的痛点。采用 YOLOv8 负责大范围 ROI 粗定位与分类兜底，OpenCV 负责局部高精度几何/颜色采样。
   - 引入**滑动窗口投票机制 (Sliding Window Voting)**，完美消除了视频帧的瞬间跳变与闪烁，实现了极其沉稳、高置信度的仪表盘状态输出。

2. **基于旋转矩阵与顶点的 3D 几何特征降维识别**
   - 在颜色与形状识别模块中，创新性地结合 `cv2.minAreaRect` (最小外接旋转矩形) 与 `cv2.approxPolyDP` (多边形逼近)。
   - 突破了 2D 图像识别 3D 物体的盲区，完美矫正了倾斜长方体的误判问题，并通过 15x15 椭圆形态学卷积核实现了“连体同色物块”的物理级切割分离。

3. **基于偏航平移解耦的工业级高精度轨迹规划**
   - 在“马”字轨迹复现任务中，放弃了粗糙的摇杆操作。自主编写基于 `move_base_msgs` 和欧拉角四元数解算的高阶导航脚本。
   - 实现了**“先旋转对准，再笔直平移”**的控制解耦，配合 `hector_trajectory_server` 输出工业级平滑、精准的【马】字轨迹。

---

## 环境配置 (Environment Setup)

### 1. 视觉依赖配置 (Conda)
我们提供了完整的跨平台环境配置文件，在根目录下执行以下命令即可一键复刻运行环境：
```bash
conda env create -f environment.yaml
conda activate dial_recognition
```

### 2. ROS 依赖配置 (Ubuntu 20.04 + Noetic)
仿真与通信模块依赖以下官方包及第三方轨迹包：
```bash
sudo apt-get install ros-noetic-ros-tutorials
sudo apt-get install ros-noetic-hector-trajectory-server
```

---

## 📁 目录结构与运行指南 (Project Structure & Usage)

### 模块一：视觉感知 (Task1_Dashboard_Color)
包含仪表盘识别与颜色形状识别的核心算法。
*   `yolo_opencv_fusion.py`：仪表盘融合识别主程序（带时间戳与防抖输出）。
*   `color_shape.py`：抗干扰颜色与形状多边形检测程序。
*   `train.py` / `dial.yaml`：YOLOv8 模型训练脚本与配置。
*   `weights/best.pt`：训练收敛的仪表盘检测最优权重。
**运行方式：**
```bash
python yolo_opencv_fusion.py
python color_shape.py
```

### 模块二：ROS 程序通讯 (Task2_ROS_Service)
包含自定义 `.srv` 格式文件以及服务端、客户端双向通信节点。
*   `server.py`：城市邮编与位置查询服务端。
*   `client.py`：查询请求客户端。
**运行方式：**
```bash
rosrun city_query server.py
rosservice call /query_city "city: '首尔'"  # 或者使用客户端调用
```

### 模块三：ROS 基础操作 (Task3_ROS_Basics)
包含 ROS 基础命令操作凭证。
*   `run.bag`：使用 `rosbag` 记录的 turtlesim 小乌龟 `cmd_vel` 运动轨迹。
*   `commands.txt`：工作空间环境变量覆盖操作的命令记录。
**运行方式：**
```bash
rosservice call /reset
rosbag play run.bag
```

### 模块四：ROS 仿真建图与导航 (Task4_Gazebo_Navigation)
包含基于 TurtleBot3 模型的 Gazebo 环境与定点导航脚本。
*   `task1.world` / `task2.world`：自定义仿真房间与物体。
*   `run_sim.launch`：无显卡冲突的一键式小车降生启动配置。
*   `mymap.pgm` / `mymap.yaml`：Gmapping 算法构建的 2D 栅格地图。
*   `draw_ma_pro.py`：解耦式定点导航“画笔”控制脚本。

**运行方式（轨迹生成）：**
```bash
# 启动环境与导航
roslaunch city_query task1.launch
roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=/path/to/mymap.yaml
# 启动轨迹服务器与画字程序
rosrun hector_trajectory_server hector_trajectory_server _target_frame_name:=map _source_frame_name:=base_link
rosrun city_query draw_ma_pro.py
```

---
*Powered by [修狗不修BUG队] @ 2026*
