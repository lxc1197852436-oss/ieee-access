# 数据字段字典

## 标准 VPP 调度数据表

当前系统默认读取：

`data/processed/china_vpp_priority1_guangdong_sample.csv`

该数据表是“国内公开披露数据校准 + 虚拟电厂仿真构造”的 15 分钟级样例数据，不应表述为真实 VPP 运行数据。

| 字段 | 类型 | 单位 | 含义 | 来源/构造方式 | 是否入模型 |
|---|---|---:|---|---|---:|
| `timestamp` | datetime | - | 调度时刻 | 按 15 分钟频率生成 | 是 |
| `region` | string | - | 区域 | 当前为广东省 | 否 |
| `load_mw` | float | MW | 虚拟电厂聚合负荷 | 基于夏季日负荷曲线、晚高峰和随机扰动构造 | 是 |
| `pv_mw` | float | MW | 分布式光伏出力 | 基于日照曲线、云量扰动和 5 MW 装机上限构造 | 是 |
| `price_yuan_mwh` | float | 元/MWh | 电力市场价格 | 由广东公开披露现货价格区间校准 | 是 |
| `temperature_c` | float | 摄氏度 | 环境温度 | 基于夏季日内温度曲线构造 | 是 |
| `event_type` | string | - | 文本事件类别 | 高温预警、需求响应、价格尖峰、新能源消纳等 | 是 |
| `event_text` | string | - | 中文文本事件内容 | 参考市场公告、气象预警和调度通知表达方式构造 | 是 |
| `source_note` | string | - | 数据边界说明 | 标注公开数据校准与仿真构造边界 | 否 |

## 已抓取国内公开数据表

### `nea_2024_power_consumption.csv`

| 字段 | 含义 |
|---|---|
| `year` | 年份 |
| `indicator` | 指标名称 |
| `value_100m_kwh` | 数值，单位为亿千瓦时 |
| `yoy_pct` | 同比增速，百分比 |
| `source_id` | 来源 ID |

用途：论文绪论背景、负荷增长和中国电力需求规模说明。

### `nea_2024_renewable_summary.csv`

| 字段 | 含义 |
|---|---|
| `year` | 年份 |
| `indicator` | 指标名称 |
| `value` | 数值 |
| `unit` | 单位 |
| `source_id` | 来源 ID |

用途：新能源装机、风电/光伏规模、新能源消纳背景说明。

### `guangdong_market_storage_2024.csv`

| 字段 | 含义 |
|---|---|
| `indicator` | 广东市场或储能指标 |
| `value` | 指标值 |
| `unit` | 单位 |
| `source_id` | 来源 ID |

用途：广东现货市场价格范围、储能充放电价差、储能商业模式与实验价格校准。

## 新能源消纳利用率替代数据

### `new_energy_utilization_substitute_2024.csv`

第三方省级新能源消纳表被 403 拦截后，当前采用国家能源局官方全国利用率作为替代背景校准数据。

建议字段：

| 字段 | 单位 | 含义 |
|---|---:|---|
| `region` | - | 省份或区域 |
| `wind_utilization_2024_pct` | % | 2024 年风电利用率 |
| `solar_utilization_2024_pct` | % | 2024 年光伏利用率 |
| `source_id` | - | 来源 ID |
| `note` | - | 数据边界说明 |

当前替代数据包含全国风电平均利用率 95.9%、全国光伏发电利用率 96.8%。省级逐时消纳压力不使用外部表格，而由 VPP 环境内部的剩余光伏功率和弃光惩罚动态刻画。
