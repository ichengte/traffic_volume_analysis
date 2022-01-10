## 包依赖

```
python3.*
tqdm
pandas
natsort
matplotlib
rstl
scipy
numpy
python_dateutil
scikit_learn
```

## 包依赖安装

在项目根目录下使用如下命令进行python环境配置

```
pip3 install -r requirements.txt
```

## 项目目录

```
--traffic_volume_analysis
	--src
		--analysis
			--main
        --traffic_visualize.py # 可视化每个小区的预测值与真实值
        --compute_traffic_loss.py # 计算STL在toi时间段的累计流量损失
        --traffic_predict.py # 通过STL预测每小时的流量残差
		  --3sigma_gmm_plot.py # 使用告警前数据，采用gmm建模，并计算告警期间的流量损失
     	  --result # 生成结果PPT/PDF脚本
		--case_config.json # case配置文件
		--files.py # 文件读取相关
		--utils.py # 工具包
```

## 运行结果说明

**注意：**1、2、3必须按顺序运行

1. traffic_volume_analysis/main/traffic_predict.py会生成如下文件以及文件夹，在stored_data/traffic/loss/data/alarm_cell和stored_data/traffic/loss/data/neighbor目录下有每个小区的STL残差结果csv文件

2. traffic_volume_analysis/main/compute_traffic_loss.py会通过STL每小时的残差计算生成在toi的累计流量损失，结果保存在stored_data/traffic/loss/result下

3. traffic_volume_analysis/main/traffic_visualize.py会可视化告警小区和邻区的STL预测值以及真实值，保存在stored_data/traffic/loss/fig下

5. traffic_volume_analysis/3sigma_gmm_plot.py 使用告警前数据建模，在邻区的流量分布中标出$\mu+\sigma,\mu+2\sigma,\mu+3\sigma$，并通过$3\sigma$异常检测来计算告警小区的减少流量和邻区增加流量，结果保存在stored_data/traffi/3sigma_extend目录下


## NOTICE

需要更改`src/case_config.json`, `src/files.py`中的case数据文件路径

