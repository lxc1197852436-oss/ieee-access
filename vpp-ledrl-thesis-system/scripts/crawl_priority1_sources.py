from __future__ import annotations

import csv
import json
import re
import sys
import urllib.request
from urllib.error import HTTPError, URLError
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_sources"
OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

SOURCES = {
    "nea_power_2024": "https://www.nea.gov.cn/20250120/4f7f249bac714e7693adecac996d742f/c.html",
    "nea_renewable_2024": "https://www.nea.gov.cn/20250221/e10f363cabe3458aaf78ba4558970054/c.html",
    "renewable_utilization_2024": "https://mnewenergy.in-en.com/html/newenergy-2438728.shtml",
    "guangdong_market_2024": "https://www.chujiewang.net/cxw/col151/7300",
    "people_guangdong_spot": "https://paper.people.com.cn/rmrbhwb/html/2024-01/10/content_26036149.htm",
}


def fetch(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/121 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://www.baidu.com/",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def html_to_text(html: str) -> str:
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    text = unescape(html)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*", "\n", text)
    return text


def pct_to_float(value: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%?", value)
    return float(m.group(1)) if m else None


def parse_nea_power(text: str) -> list[dict]:
    patterns = [
        ("全社会用电量", r"全社会用电量(\d+(?:\.\d+)?)亿千瓦时，同比增长(\d+(?:\.\d+)?)%"),
        ("规模以上工业发电量", r"规模以上工业发电量为?(\d+(?:\.\d+)?)亿千瓦时"),
        ("第一产业用电量", r"第一产业用电量(\d+(?:\.\d+)?)亿千瓦时，同比增长(\d+(?:\.\d+)?)%"),
        ("第二产业用电量", r"第二产业用电量(\d+(?:\.\d+)?)亿千瓦时，同比增长(\d+(?:\.\d+)?)%"),
        ("第三产业用电量", r"第三产业用电量(\d+(?:\.\d+)?)亿千瓦时，同比增长(\d+(?:\.\d+)?)%"),
        ("城乡居民生活用电量", r"城乡居民生活用电量(\d+(?:\.\d+)?)亿千瓦时，同比增长(\d+(?:\.\d+)?)%"),
    ]
    rows = []
    for name, pat in patterns:
        m = re.search(pat, text)
        if not m:
            continue
        rows.append(
            {
                "year": 2024,
                "indicator": name,
                "value_100m_kwh": float(m.group(1)),
                "yoy_pct": float(m.group(2)) if len(m.groups()) >= 2 else "",
                "source_id": "nea_power_2024",
            }
        )
    return rows


def parse_nea_renewable(text: str) -> list[dict]:
    metrics = [
        ("renewable_added_capacity", r"可再生能源发电新增装机(\d+(?:\.\d+)?)亿千瓦"),
        ("renewable_total_capacity", r"可再生能源装机达到(\d+(?:\.\d+)?)亿千瓦"),
        ("hydro_total_capacity", r"水电装机(\d+(?:\.\d+)?)亿千瓦"),
        ("wind_total_capacity", r"风电装机(\d+(?:\.\d+)?)亿千瓦"),
        ("solar_total_capacity", r"太阳能发电装机(\d+(?:\.\d+)?)亿千瓦"),
        ("biomass_total_capacity", r"生物质发电装机(\d+(?:\.\d+)?)亿千瓦"),
        ("renewable_generation", r"可再生能源发电量达(\d+(?:\.\d+)?)万亿千瓦时"),
        ("wind_solar_generation", r"风电太阳能发电量合计达(\d+(?:\.\d+)?)万亿千瓦时"),
        ("wind_generation", r"全国风电发电量(\d+(?:\.\d+)?)亿千瓦时"),
        ("wind_utilization_rate", r"全国风电平均利用率(\d+(?:\.\d+)?)%"),
        ("solar_generation", r"全国光伏发电量(\d+(?:\.\d+)?)亿千瓦时"),
        ("solar_utilization_rate", r"全国光伏发电利用率(\d+(?:\.\d+)?)%"),
    ]
    rows = []
    for name, pat in metrics:
        m = re.search(pat, text)
        if not m:
            continue
        if "utilization_rate" in name:
            unit = "pct"
        elif name in {"wind_generation", "solar_generation"}:
            unit = "100m_kwh"
        else:
            unit = "100m_kw" if "capacity" in name else "trillion_kwh"
        rows.append({"year": 2024, "indicator": name, "value": float(m.group(1)), "unit": unit, "source_id": "nea_renewable_2024"})
    return rows


def parse_utilization(text: str) -> list[dict]:
    # The source page exposes rows like: 广东 99.5%99.5%100.0%99.9%
    rows = []
    for line in text.splitlines():
        compact = re.sub(r"\s+", " ", line).strip()
        m = re.match(r"^([\u4e00-\u9fa5]{2,4}|蒙西|蒙东)\s+(\d+(?:\.\d+)?)%?\s*(\d+(?:\.\d+)?)%?\s*(\d+(?:\.\d+)?)%?\s*(\d+(?:\.\d+)?)%?", compact)
        if not m:
            continue
        region = m.group(1)
        if region in {"地区", "来源"}:
            continue
        rows.append(
            {
                "region": region,
                "wind_utilization_dec_pct": float(m.group(2)),
                "wind_utilization_2024_pct": float(m.group(3)),
                "solar_utilization_dec_pct": float(m.group(4)),
                "solar_utilization_2024_pct": float(m.group(5)),
                "source_id": "renewable_utilization_2024",
            }
        )
    return rows


def parse_guangdong_market(text: str) -> list[dict]:
    rows = []
    patterns = [
        ("guangdong_spot_price_min_yuan_kwh", r"现货市场日前价格在(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)元/kWh之间波动", "range"),
        ("storage_day_ahead_spread_yuan_kwh", r"独立储能参与现货日前市场.*?充放电平均价差.*?(\d+(?:\.\d+)?)元/kWh", "single"),
        ("storage_charge_energy_100m_kwh", r"充电电量(\d+(?:\.\d+)?)亿千瓦时，放电电量(\d+(?:\.\d+)?) ?亿千瓦时", "pair"),
        ("spot_market_direct_trade_100m_kwh", r"市场直接交易电量(\d+(?:\.\d+)?)亿千瓦时", "single"),
        ("registered_market_entities", r"市场主体近(\d+)万家", "wan"),
        ("spot_participants_total", r"累计注册登记经营主体共(\d+)家", "single"),
        ("storage_charge_settlement_price_yuan_kwh", r"充电（抽水）电量共计8\.7亿千瓦时，均价(\d+(?:\.\d+)?)元/kWh", "single"),
        ("storage_discharge_settlement_price_yuan_kwh", r"放电（发电）结算电量7\.1亿千瓦时，均价(\d+(?:\.\d+)?)元/kWh", "single"),
    ]
    for name, pat, kind in patterns:
        m = re.search(pat, text, flags=re.S)
        if not m:
            continue
        if kind == "range":
            rows.append({"indicator": "guangdong_spot_price_min_yuan_kwh", "value": float(m.group(1)), "unit": "yuan/kWh", "source_id": "guangdong_market_2024"})
            rows.append({"indicator": "guangdong_spot_price_max_yuan_kwh", "value": float(m.group(2)), "unit": "yuan/kWh", "source_id": "guangdong_market_2024"})
        elif kind == "pair":
            rows.append({"indicator": "storage_charge_energy_100m_kwh", "value": float(m.group(1)), "unit": "100m_kWh", "source_id": "guangdong_market_2024"})
            rows.append({"indicator": "storage_discharge_energy_100m_kwh", "value": float(m.group(2)), "unit": "100m_kWh", "source_id": "guangdong_market_2024"})
        elif kind == "wan":
            rows.append({"indicator": name, "value": float(m.group(1)) * 10000, "unit": "entity", "source_id": "guangdong_market_2024"})
        else:
            rows.append({"indicator": name, "value": float(m.group(1)), "unit": infer_unit(name), "source_id": "guangdong_market_2024"})
    return rows


def infer_unit(name: str) -> str:
    if "yuan_kwh" in name:
        return "yuan/kWh"
    if "100m_kwh" in name:
        return "100m_kWh"
    if "entities" in name:
        return "entity"
    return "value"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    texts: dict[str, str] = {}
    catalog = []
    for source_id, url in SOURCES.items():
        print(f"Fetching {source_id}: {url}")
        try:
            html = fetch(url)
        except (HTTPError, URLError, TimeoutError) as exc:
            err_text = f"FETCH_FAILED: {type(exc).__name__}: {exc}"
            (RAW_DIR / f"{source_id}.error.txt").write_text(err_text, encoding="utf-8")
            texts[source_id] = ""
            catalog.append({"source_id": source_id, "url": url, "raw_html": "", "raw_text": str(RAW_DIR / f"{source_id}.error.txt"), "status": "failed"})
            print(err_text)
            continue
        (RAW_DIR / f"{source_id}.html").write_text(html, encoding="utf-8")
        text = html_to_text(html)
        (RAW_DIR / f"{source_id}.txt").write_text(text, encoding="utf-8")
        texts[source_id] = text
        catalog.append({"source_id": source_id, "url": url, "raw_html": str(RAW_DIR / f"{source_id}.html"), "raw_text": str(RAW_DIR / f"{source_id}.txt"), "status": "ok"})

    write_csv(OUT_DIR / "source_catalog.csv", catalog)
    power_rows = parse_nea_power(texts["nea_power_2024"])
    renewable_rows = parse_nea_renewable(texts["nea_renewable_2024"])
    utilization_rows = parse_utilization(texts["renewable_utilization_2024"])
    market_rows = parse_guangdong_market(texts["guangdong_market_2024"])

    write_csv(OUT_DIR / "nea_2024_power_consumption.csv", power_rows)
    write_csv(OUT_DIR / "nea_2024_renewable_summary.csv", renewable_rows)
    write_csv(OUT_DIR / "new_energy_utilization_2024.csv", utilization_rows)
    write_csv(OUT_DIR / "guangdong_market_storage_2024.csv", market_rows)

    summary = {
        "source_count": len(catalog),
        "power_rows": len(power_rows),
        "renewable_rows": len(renewable_rows),
        "utilization_rows": len(utilization_rows),
        "market_rows": len(market_rows),
        "recommended_region": "广东",
        "notes": "These sources are public aggregate/disclosure data. They are suitable for thesis background, parameter calibration, scenario construction, and event text, but not enough alone for fully real 15-minute RL training.",
    }
    (REPORT_DIR / "priority1_crawl_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"crawl failed: {exc}", file=sys.stderr)
        raise
