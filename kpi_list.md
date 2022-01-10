# KPI List

## case 3 （无损失）

alarm in TOI: External Clock Reference Problem

+ 告警影响：可能导致**TDD**网元系统时钟不可用，可能出现告警小区接入失败、掉话等业务异常或无法提供业务。可能导致邻小区之间干扰变大，影响速率。FDD网元不受影响。
+ 告警对**告警小区**的影响：（若导致小区部分用户迁出）接入成功率下降 -> 掉话率上升 -> 用户数量下降 -> 下行流量下降

### 告警小区kpi list:

#### 主要指标

- L.UL.Interference.Avg(dBm)
- RRC Setup Success Rate(%)
- ERAB Setup Success Rate(%)
- S1Sig Setup Success Rate(%)
- RRC Conn Users Avg
- LTE_User DL Average Throughput(Mbps)
- LTE_User UL Average Throughput(Mbps)
- LTE_DL Traffic Volume(GB)
- LTE_UL Traffic Volume(GB)

#### 完整指标

干扰类指标

- L.UL.Interference.Avg(dBm)

接入成功率、掉话率类指标

- RRC_Setup_Att_Times
- RRC_Setup_Failure_Times
- RRC Setup Success Rate(%)
- ERAB_Setup_Att_Times
- ERAB_Setup_Failure_Times
- ERAB Setup Success Rate(%)
- S1Sig Setup Att Times
- S1Sig Setup Failure Times
- S1Sig Setup Success Rate(%)

用户数

- RRC Conn Users Avg
- UL Activated Users Avg
- DL Activated Users Avg

速率类指标

- LTE_User DL Average Throughput(Mbps)
- LTE_User UL Average Throughput(Mbps)

流量指标

- LTE_DL Traffic Volume(GB)
- LTE_UL Traffic Volume(GB)


### 邻居小区kpi list

+ 如果有断站小区用户迁移入**同站邻区**：随机接入增加  ->  用户数量增加 ->  下行流量增加（用户分布不会变）
+ 如果有断站小区用户迁移入**异站邻区**：可能远端用户占比升高（用户分布变化）-> 随机接入增加  ->  用户数量增加 ->  下行流量增加（紧密共覆盖时，用户离原服务小区和新小区距离相近，分布不会变化）

#### 主要指标

- Average TA
- RRC_Setup_Att_Times
- ERAB_Setup_Att_Times
- S1Sig Setup Att Times
- RRC Conn Users Avg
- LTE_DL Traffic Volume(GB)

#### 完整指标

用户分布指标（适用于用户迁移入异站邻区）

- TAlessthan1ratio(300m)
- Average TA

随机接入类

- RRC_Setup_Att_Times
- RRC_Setup_Failure_Times
- RRC Setup Success Rate(%)
- ERAB_Setup_Att_Times
- ERAB_Setup_Failure_Times
- ERAB Setup Success Rate(%)
- S1Sig Setup Att Times
- S1Sig Setup Failure Times
- S1Sig Setup Success Rate(%)

用户数类

- RRC Conn Users Avg
- UL Activated Users Avg
- DL Activated Users Avg

流量类

- LTE_DL Traffic Volume(GB)




----------

## case 4/13（有损失）

alarm in TOI: Cell Unavailable

+ 告警影响：告警小区不能提供业务=>用户会随机接入邻小区
+ 告警对**告警小区**的影响：1、若告警持续整个小时，可观测到kpi掉0或缺失（断站后无KPI上报，某些KPI系统默认填0），告警基站服务用户在该小时内全部迁移出。2、若告警未持续整个小时，kpi不会掉0，对于均值类kpi，由于仅根据正常时刻数据统计，不会有异常，对于累加类kpi，可能会有掉坑。

### 告警小区kpi list:

流量类指标

- LTE_UL Traffic Volume(GB)
- LTE_DL Traffic Volume(GB)

用户类指标

- RRC Conn Users Avg
- DL Activated Users Avg
- UL Activated Users Avg


### 邻居小区kpi list

+ 如果有告警小区用户迁移入同站邻区：随机接入增加 or 切换入增加  ->  用户数量增加 ->  下行流量增加
（用户分布不会变）
+ 如果有告警小区用户迁移入异站邻区：可能远端用户占比升高（用户分布变化）-> 随机接入增加 or 切换入增加  ->  用户数量增加 ->  下行流量增加（紧密共覆盖时，用户离原服务小区和新小区距离相近，分布不会变化）


#### 主要指标

- Average TA
- RRC_Setup_Att_Times
- ERAB_Setup_Att_Times
- S1Sig Setup Att Times
- HI Succ Times
- RRC Conn Users Avg
- LTE_DL Traffic Volume(GB)

#### 完整指标

用户分布指标（适用于用户迁移入异站邻区）

- TAlessthan1ratio(300m)
- Average TA

随机接入类

- RRC_Setup_Att_Times
- RRC_Setup_Failure_Times
- RRC Setup Success Rate(%)
- ERAB_Setup_Att_Times
- ERAB_Setup_Failure_Times
- ERAB Setup Success Rate(%)
- S1Sig Setup Att Times
- S1Sig Setup Failure Times
- S1Sig Setup Success Rate(%)

切换入类

- HI Succ Times
- HI Failure Times
- HI Succ Rate(%)

用户数类

- RRC Conn Users Avg
- UL Activated Users Avg
- DL Activated Users Avg

流量类

- LTE_DL Traffic Volume(GB)

