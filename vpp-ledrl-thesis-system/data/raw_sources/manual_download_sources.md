# 手动下载/补录数据源清单

## 1. 新能源消纳利用率数据

自动抓取失败 URL：

```text
https://mnewenergy.in-en.com/html/newenergy-2438728.shtml
```

失败原因：

```text
HTTP 403 Forbidden
```

用途：

- 各省风电、光伏利用率。
- 构造“新能源消纳压力”文本事件。
- 选择不同省份场景，例如广东、山东、甘肃、新疆等。

你可以手动下载或截图表格。拿到文件后，建议整理成：

```csv
region,wind_utilization_dec_pct,wind_utilization_2024_pct,solar_utilization_dec_pct,solar_utilization_2024_pct,source_id
广东,99.5,99.5,100.0,99.9,manual_new_energy_utilization_2024
```

保存为：

```text
data/processed/new_energy_utilization_2024.csv
```

## 可替代方案

如果该页面仍然无法下载，可以直接采用替代方案，不影响论文推进：

```text
data/processed/new_energy_utilization_substitute_2024.csv
```

替代逻辑：

- 使用国家能源局《2024年可再生能源并网运行情况》中的全国风电平均利用率和全国光伏发电利用率作为权威背景校准。
- 省级逐时新能源消纳压力不再依赖外部表格，而是在 VPP 环境中通过 `pv_mw - load_mw - charge_power` 的剩余光伏功率与弃光惩罚项动态刻画。
- 论文中表述为“全国利用率用于背景校准，逐时消纳压力由仿真环境生成”，不要写成“已获得省级真实消纳率时序数据”。

## 2. 后续可补充天气数据

若你能申请或下载国内天气数据，建议字段为：

```csv
timestamp,region,temperature_c,humidity_pct,wind_speed_ms,solar_radiation_wm2,weather_text,source_id
```

可选来源：

- 中国气象数据网。
- 和风天气历史天气 API。
- 心知天气 API。
- 地方气象台历史预警文本。

如果暂时没有国内天气 API，可先使用 Open-Meteo 中国坐标历史天气作为开源替代，但论文中应如实说明。
