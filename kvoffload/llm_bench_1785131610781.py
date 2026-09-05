#!/usr/bin/env python3
"""
LLM API 速度基准测试工具
测量 TTFT、Tokens/s、端到端延迟等关键指标
"""

import asyncio  # 异步 I/O 支持，用于并发请求
import aiohttp  # 异步 HTTP 客户端库
import json     # JSON 解析，处理 API 响应
import os       # 环境变量读取（API Key）
import time     # 时间测量（monotonic 时钟）
import uuid     # 生成唯一 ID，用于随机文本填充
import sys      # 系统标准输入输出
import io       # 文本 I/O 包装，解决中文编码问题
import argparse # 命令行参数解析
import csv      # CSV 导出
import hashlib  # 记录 prompt 指纹，用于成对正确性比较
import random   # 随机数生成，用于构造随机 prompt
import statistics  # 统计数据计算（均值、标准差）
import shutil   # 终端宽度检测，用于进度条自适应
from transformers import AutoTokenizer
from dataclasses import dataclass, asdict  # 数据类，简化结构体定义和序列化
from datetime import datetime  # 时间戳格式化

# 将 stdout 包装为 UTF-8 编码，防止 Windows 终端中文输出乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def _progress_bar(current: int, total: int, bar_width: int = 0) -> str:
    """生成彩色进度条字符串"""
    if total == 0:
        return ""
    if bar_width == 0:
        bar_width = min(shutil.get_terminal_size().columns - 20, 60)
    ratio = current / total
    filled = int(ratio * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    pct = ratio * 100
    # 绿色 (92) → 黄色 (93) → 红色 (91) 渐变
    if pct < 50:
        color = f"\033[92m{bar}\033[0m"
    elif pct < 80:
        color = f"\033[93m{bar}\033[0m"
    else:
        color = f"\033[91m{bar}\033[0m"
    return f"  [{color}] {current:>{len(str(total))}}/{total} ({pct:5.1f}%)"


@dataclass
class RequestResult:
    """单次请求结果"""
    success: bool = False          # 请求是否成功
    status_code: int = 0           # HTTP 状态码
    ttft: float = 0.0              # Time To First Token（秒），首 token 延迟
    total_time: float = 0.0        # 请求总耗时（端到端延迟，秒）
    output_tokens: int = 0         # 模型输出的 token 数
    input_tokens: int = 0          # 输入（prompt）的 token 数
    output_tps: float = 0.0        # 输出速度（output tokens / decode 时间）
    total_tps: float = 0.0         # 总 token 速度（input + output tokens / total_time）
    input_tps: float = 0.0         # 输入吞吐（input tokens / ttft），衡量 prompt 处理速度
    tpot: float = 0.0              # Time Per Output Token（秒/ token）
    error: str = ""                # 错误信息（成功时为空）
    request_index: int = 0         # 请求序号，用于跨实验一一对应
    prompt_sha256: str = ""        # 输入指纹，验证两轮 workload 完全一致
    generated_text: str = ""       # 完整输出，用于正确性比较
    finish_reason: str = ""        # OpenAI 兼容接口返回的结束原因


@dataclass
class BenchReport:
    """聚合统计报告"""
    runs: int = 0                  # 总请求数
    success_count: int = 0         # 成功请求数
    fail_count: int = 0            # 失败请求数
    success_rate: float = 0.0      # 成功率（百分比）
    ttft_avg: float = 0.0          # TTFT 平均值
    ttft_std: float = 0.0          # TTFT 标准差
    ttft_min: float = 0.0          # TTFT 最小值
    ttft_max: float = 0.0          # TTFT 最大值
    ttft_p50: float = 0.0          # TTFT 中位数（P50）
    ttft_p90: float = 0.0          # TTFT P90
    ttft_p99: float = 0.0          # TTFT P99
    tps_avg: float = 0.0           # 输出速度平均值
    tps_std: float = 0.0           # 输出速度标准差
    tps_min: float = 0.0           # 输出速度最小值
    tps_max: float = 0.0           # 输出速度最大值
    tps_p50: float = 0.0           # 输出速度中位数
    tps_p90: float = 0.0           # 输出速度 P90
    tps_p99: float = 0.0           # 输出速度 P99
    total_tps_avg: float = 0.0     # 总 token 速度平均值
    total_tps_std: float = 0.0     # 总 token 速度标准差
    total_tps_min: float = 0.0     # 总 token 速度最小值
    total_tps_max: float = 0.0     # 总 token 速度最大值
    total_tps_p50: float = 0.0     # 总 token 速度中位数
    total_tps_p90: float = 0.0     # 总 token 速度 P90
    total_tps_p99: float = 0.0     # 总 token 速度 P99
    input_tps_avg: float = 0.0     # 输入吞吐平均值（input tokens / ttft）
    input_tps_std: float = 0.0     # 输入吞吐标准差
    input_tps_min: float = 0.0     # 输入吞吐最小值
    input_tps_max: float = 0.0     # 输入吞吐最大值
    input_tps_p50: float = 0.0     # 输入吞吐中位数
    input_tps_p90: float = 0.0     # 输入吞吐 P90
    input_tps_p99: float = 0.0     # 输入吞吐 P99
    latency_avg: float = 0.0       # 端到端延迟平均值
    latency_min: float = 0.0       # 端到端延迟最小值
    latency_max: float = 0.0       # 端到端延迟最大值
    latency_std: float = 0.0       # 端到端延迟标准差
    latency_p50: float = 0.0       # 端到端延迟中位数
    latency_p90: float = 0.0       # 端到端延迟 P90
    latency_p99: float = 0.0       # 端到端延迟 P99
    avg_input_tokens: float = 0.0  # 平均输入 token 数
    avg_output_tokens: float = 0.0 # 平均输出 token 数
    request_throughput: float = 0.0  # 整轮请求吞吐（requests/s）
    aggregate_input_tps: float = 0.0 # 整轮输入 token 吞吐
    aggregate_output_tps: float = 0.0 # 整轮输出 token 吞吐
    aggregate_total_tps: float = 0.0 # 整轮总 token 吞吐

    def to_dict(self):
        """转换为字典，方便 JSON 序列化"""
        return asdict(self)


def percentile(data: list, p: float) -> float:
    """计算百分位数（如 P50、P90、P99）"""
    if not data:
        return 0.0
    sorted_data = sorted(data)  # 先排序
    idx = int(len(sorted_data) * p / 100)  # 计算对应索引
    return sorted_data[min(idx, len(sorted_data) - 1)]  # 防止越界


def compute_report(results: list[RequestResult]) -> BenchReport:
    """根据所有请求结果计算统计报告"""
    report = BenchReport()
    report.runs = len(results)
    successes = [r for r in results if r.success]  # 仅筛选成功请求
    report.success_count = len(successes)
    report.fail_count = report.runs - report.success_count
    report.success_rate = report.success_count / report.runs * 100 if report.runs else 0

    if not successes:
        return report

    # —— TTFT 统计 ——
    ttfts = [r.ttft for r in successes]
    report.ttft_avg = statistics.mean(ttfts)
    report.ttft_std = statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0
    report.ttft_min = min(ttfts)
    report.ttft_max = max(ttfts)
    report.ttft_p50 = percentile(ttfts, 50)
    report.ttft_p90 = percentile(ttfts, 90)
    report.ttft_p99 = percentile(ttfts, 99)

    # —— 输出速度 TPS 统计 ——
    tps_list = [r.output_tps for r in successes if r.output_tps > 0]
    if tps_list:
        report.tps_avg = statistics.mean(tps_list)
        report.tps_std = statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0
        report.tps_min = min(tps_list)
        report.tps_max = max(tps_list)
        report.tps_p50 = percentile(tps_list, 50)
        report.tps_p90 = percentile(tps_list, 90)
        report.tps_p99 = percentile(tps_list, 99)

    # —— 输入吞吐 INPUT TPS 统计 ——
    input_tps_list = [r.input_tps for r in successes if r.input_tps > 0]
    if input_tps_list:
        report.input_tps_avg = statistics.mean(input_tps_list)
        report.input_tps_std = statistics.stdev(input_tps_list) if len(input_tps_list) > 1 else 0.0
        report.input_tps_min = min(input_tps_list)
        report.input_tps_max = max(input_tps_list)
        report.input_tps_p50 = percentile(input_tps_list, 50)
        report.input_tps_p90 = percentile(input_tps_list, 90)
        report.input_tps_p99 = percentile(input_tps_list, 99)

    # —— 总 token 速度统计 ——
    total_tps_list = [r.total_tps for r in successes if r.total_tps > 0]
    if total_tps_list:
        report.total_tps_avg = statistics.mean(total_tps_list)
        report.total_tps_std = statistics.stdev(total_tps_list) if len(total_tps_list) > 1 else 0.0
        report.total_tps_min = min(total_tps_list)
        report.total_tps_max = max(total_tps_list)
        report.total_tps_p50 = percentile(total_tps_list, 50)
        report.total_tps_p90 = percentile(total_tps_list, 90)
        report.total_tps_p99 = percentile(total_tps_list, 99)

    # —— 端到端延迟统计 ——
    latencies = [r.total_time for r in successes]
    report.latency_avg = statistics.mean(latencies)
    report.latency_min = min(latencies)
    report.latency_max = max(latencies)
    report.latency_std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    report.latency_p50 = percentile(latencies, 50)
    report.latency_p90 = percentile(latencies, 90)
    report.latency_p99 = percentile(latencies, 99)

    # —— Token 数统计 ——
    input_tokens = [r.input_tokens for r in successes if r.input_tokens > 0]
    output_tokens = [r.output_tokens for r in successes if r.output_tokens > 0]
    report.avg_input_tokens = statistics.mean(input_tokens) if input_tokens else 0.0
    report.avg_output_tokens = statistics.mean(output_tokens) if output_tokens else 0.0

    return report


def _dw(s: str) -> int:
    """计算字符串在终端中的显示宽度（中文占 2 格，英文/数字占 1 格）"""
    w = 0
    for ch in s:
        if ord(ch) > 0x3000:  # 中文字符（CJK 统一表意文字区起点）
            w += 2
        else:
            w += 1
    return w


def _pad(s: str, width: int) -> str:
    """右填空格使字符串的终端显示宽度达到指定值，用于对齐中英文混排的表格"""
    return s + " " * (width - _dw(s))




def print_report(report: BenchReport, model: str, total_elapsed: float, results: list = None):
    """以对齐表格形式打印统计报告到终端"""
    # 总览行
    print(f"\n  模型: {model}  |  请求: {report.runs}  |  成功: {report.success_count}  |  失败: {report.fail_count}  |  成功率: {report.success_rate:.1f}%  |  耗时: {total_elapsed:.1f}s", flush=True)

    if report.success_count == 0:
        print(f"  所有请求失败，无统计信息\n", flush=True)
        if results:
            for r in results[:5]:
                print(f"  [错误] {r.error}", flush=True)
        return

    # 所有单元格数据（原始字符串，不含中英文混合宽度问题）
    label_data = [
        ("首 Token 延迟",  "s"),
        ("输入速度",     " t/s"),
        ("输出速度",     " t/s"),
        ("全部速度",     " t/s"),
        ("端到端延迟",   "s"),
    ]
    val_rows = [
        [f"{report.ttft_avg:.3f}",  f"{report.ttft_std:.3f}",  f"{report.ttft_p50:.3f}",  f"{report.ttft_p90:.3f}",  f"{report.ttft_p99:.3f}",  f"{report.ttft_min:.3f}",  f"{report.ttft_max:.3f}"],
        [f"{report.input_tps_avg:.1f}", f"{report.input_tps_std:.1f}", f"{report.input_tps_p50:.1f}", f"{report.input_tps_p90:.1f}", f"{report.input_tps_p99:.1f}", f"{report.input_tps_min:.1f}", f"{report.input_tps_max:.1f}"],
        [f"{report.tps_avg:.1f}", f"{report.tps_std:.1f}", f"{report.tps_p50:.1f}", f"{report.tps_p90:.1f}", f"{report.tps_p99:.1f}", f"{report.tps_min:.1f}", f"{report.tps_max:.1f}"],
        [f"{report.total_tps_avg:.1f}", f"{report.total_tps_std:.1f}", f"{report.total_tps_p50:.1f}", f"{report.total_tps_p90:.1f}", f"{report.total_tps_p99:.1f}", f"{report.total_tps_min:.1f}", f"{report.total_tps_max:.1f}"],
        [f"{report.latency_avg:.3f}", f"{report.latency_std:.3f}", f"{report.latency_p50:.3f}", f"{report.latency_p90:.3f}", f"{report.latency_p99:.3f}", f"{report.latency_min:.3f}", f"{report.latency_max:.3f}"],
    ]
    headers = ["指标", "平均", "标准差", "P50", "P90", "P99", "最小", "最大"]

    # 拼上单位：每列数值 + 该行对应的单位
    full_rows = []
    for (label, unit), vals in zip(label_data, val_rows):
        full_rows.append([label] + [v + unit for v in vals])

    # 计算每列中英文混排最大显示宽度（中文 2 格，其他 1 格）
    cjk = lambda s: sum(2 if ord(c) > 0x2e80 else 1 for c in s)
    col_widths = [max(cjk(h), max(cjk(r[i]) for r in full_rows)) for i, h in enumerate(headers)]

    # 构造行：每列左对齐到显示宽度
    def format_row(cells):
        parts = []
        for cell, w in zip(cells, col_widths):
            pad = w - cjk(cell)
            parts.append(cell + " " * pad)
        return "            " + "            ".join(parts)

    # 分隔线
    sep_parts = []
    for w in col_widths:
        sep_parts.append("-" * w)
    sep = "            " + "            ".join(sep_parts)

    # 打印
    print(format_row(headers), flush=True)
    print(sep, flush=True)
    for r in full_rows:
        print(format_row(r), flush=True)
    print(sep, flush=True)
    print(f"  平均输入: {int(report.avg_input_tokens)} token   |   平均输出: {int(report.avg_output_tokens)} token", flush=True)
    print(
        f"  系统吞吐: {report.request_throughput:.3f} req/s"
        f"  |  输入 {report.aggregate_input_tps:.1f} tok/s"
        f"  |  输出 {report.aggregate_output_tps:.1f} tok/s"
        f"  |  总计 {report.aggregate_total_tps:.1f} tok/s",
        flush=True,
    )
    print()


def save_json(report: BenchReport, results: list, path: str, start_time: str = "", end_time: str = ""):
    """将测试报告和所有单次结果保存为 JSON 文件"""
    data = {
        "report": report.to_dict(),
        "start_time": start_time,
        "end_time": end_time,
        "timestamp": datetime.now().isoformat(),
        "individual_results": [asdict(r) for r in results]  # 包含每次请求的详细数据
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(results: list, path: str, model: str = "", start_time: str = "", end_time: str = ""):
    """将单次请求结果保存为 CSV 文件，包含所有指标字段和请求级配置信息"""
    if not results:
        return
    # 中文字段名映射
    cn_fields = {
        "success": "是否成功",
        "status_code": "状态码",
        "ttft": "首 Token 延迟(秒)",
        "total_time": "总耗时(秒)",
        "output_tokens": "输出 Token(tok)",
        "input_tokens": "输入 Token(tok)",
        "output_tps": "输出速度(tok/秒)",
        "total_tps": "全部速度(tok/秒)",
        "input_tps": "输入速度(tok/秒)",
        "tpot": "每 Token 耗时(秒)",
        "error": "错误信息",
    }
    fieldnames = [
        key for key in asdict(results[0]).keys() if key != "generated_text"
    ]
    cn_headers = [cn_fields.get(k, k) for k in fieldnames]
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["# 模型", model, "开始时间", start_time, "结束时间", end_time])
        writer.writerow(["# 请求数", len(results)])
        writer.writerow([])
        writer.writerow(cn_headers)
        for r in results:
            writer.writerow([asdict(r)[k] for k in fieldnames])


async def benchmark_request(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    on_first_token=None,
    estimated_input_tokens: int = 0,
    request_index: int = 0,
    temperature: float = 0.0,
    sampling_seed: int = 1,
    request_timeout: float = 900.0,
    ignore_eos: bool = False,
) -> RequestResult:
    """发送一次流式（SSE）请求并测量各项性能指标"""
    result = RequestResult(
        request_index=request_index,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "stream": True,            # 开启流式输出（SSE）
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": sampling_seed,
        "ignore_eos": ignore_eos,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    }

    start = time.monotonic()  # 请求开始时间（monotonic 不受系统时间调整影响）
    ttft = None               # 首次收到 token 的时间戳，后续用于计算 TTFT
    output_tokens = 0
    usage_output_tokens = None
    generated_chunks = []

    try:
        buffer = bytearray()  # SSE 数据流缓冲区，按 \n 分割处理数据行
        async with session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=request_timeout),
        ) as resp:
            result.status_code = resp.status
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:200]}"
                result.total_time = time.monotonic() - start
                return result

            # 逐块读取流式响应
            async for chunk_bytes in resp.content.iter_any():
                buffer.extend(chunk_bytes)
                # 按换行符拆分成完整行（SSE 协议每行以 \n 分隔）
                while b"\n" in buffer:
                    line_bytes, buffer = buffer.split(b"\n", 1)
                    decoded = line_bytes.decode("utf-8", errors="replace").strip()
                    if not decoded or not decoded.startswith("data: "):
                        continue  # 跳过非 SSE data 行或空行
                    data = decoded[6:]  # 去掉 "data: " 前缀
                    if data == "[DONE]":
                        break  # SSE 结束标志
                    try:
                        chunk = json.loads(data)
                        # —— 从 usage 字段解析准确的 token 计数（OpenAI 兼容格式） ——
                        usage = chunk.get("usage")
                        if usage:
                            if usage.get("prompt_tokens") is not None:
                                result.input_tokens = usage["prompt_tokens"]
                            elif estimated_input_tokens > 0 and result.input_tokens == 0:
                                result.input_tokens = estimated_input_tokens
                            if usage.get("completion_tokens") is not None:
                                usage_output_tokens = usage["completion_tokens"]

                        # 只有实际内容到达时才记录 TTFT；role/usage chunk 不算首 token。
                        choices = chunk.get("choices", [])
                        if choices:
                            if choices[0].get("finish_reason"):
                                result.finish_reason = choices[0]["finish_reason"]
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if ttft is None:
                                    ttft = time.monotonic() - start
                                    if on_first_token:
                                        on_first_token()
                                generated_chunks.append(content)
                    except json.JSONDecodeError:
                        continue  # 跳过非 JSON 行（如 keep-alive）

            elapsed = time.monotonic() - start
            result.success = True
            result.ttft = ttft if ttft is not None else elapsed  # 如果从未收到 token，TTFT 回退为总耗时
            result.total_time = elapsed
            result.generated_text = "".join(generated_chunks)
            if usage_output_tokens is not None:
                output_tokens = usage_output_tokens
            else:
                output_tokens = len(result.generated_text) // 4
                output_tokens += sum(ord(c) > 0x2e80 for c in result.generated_text)
            result.output_tokens = output_tokens

            # 如果 API 未返回 input_tokens，用请求时传入的估算值
            if result.input_tokens == 0 and estimated_input_tokens > 0:
                result.input_tokens = estimated_input_tokens

            # —— 计算衍生指标 ——
            decode_time = elapsed - result.ttft  # 解码阶段时长（首 token 到结束）
            total_tokens = result.input_tokens + output_tokens
            if output_tokens > 0 and decode_time > 0.01:
                result.output_tps = output_tokens / decode_time  # 输出速度：token / 解码秒
            if total_tokens > 0 and elapsed > 0:
                result.total_tps = total_tokens / elapsed  # 总 token 速度：总 token / 请求总耗时
            if output_tokens > 0 and decode_time > 0:
                result.tpot = decode_time / output_tokens  # 每 token 耗时
            if result.input_tokens > 0 and ttft and ttft > 0:
                result.input_tps = result.input_tokens / ttft  # prompt 处理吞吐

    except asyncio.TimeoutError:
        result.error = f"Timeout (>{request_timeout:g}s)"
        result.total_time = time.monotonic() - start
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.total_time = time.monotonic() - start

    return result


# ==================== 随机 Prompt 生成 ====================
# 以下数据用于构造多种多样的人为 prompt，模拟不同问法

_SENTENCES = [
    # 一组中文技术知识句子池，用于拼接成长文本输入
    "人工智能是计算机科学的一个分支，它致力于研究、开发用于模拟和扩展人类智能的理论、方法和技术。",
    "深度学习是人工智能的核心技术之一，通过多层神经网络对数据进行学习和特征提取。",
    "大语言模型是一种基于Transformer架构的人工智能模型，能够理解和生成人类语言。",
    "卡奥斯工业互联网平台是海尔集团打造的具有中国自主知识产权的工业互联网平台。",
    "云计算是一种基于互联网的计算方式，通过网络提供可伸缩的虚拟化资源。",
    "大数据技术用于处理海量数据，从中挖掘有价值的信息和知识。",
    "物联网技术将物理设备与互联网连接，实现智能化识别和管理。",
    "区块链是一种分布式账本技术，具有去中心化、不可篡改的特点。",
    "5G网络提供高速率、低延迟的无线通信服务，支撑万物互联。",
    "量子计算利用量子力学原理，有望在特定领域实现超越经典计算机的性能。",
    "边缘计算将计算资源部署在网络边缘，降低延迟并减轻云端负担。",
    "机器学习是人工智能的子领域，通过算法让计算机从数据中学习并改进性能。",
    "自然语言处理技术使计算机能够理解、分析和生成人类语言。",
    "计算机视觉让机器能够理解和处理图像和视频内容。",
    "强化学习通过与环境交互，让智能体学习最优决策策略。",
]

_TEMPLATES = [
    # 多种提问模板，使 prompt 风格多样化
    "请详细介绍{}",
    "请解释一下{}的概念和应用",
    "{}是什么意思？",
    "关于{}，你能告诉我哪些关键信息？",
    "我想了解{}，请详细说明",
]

_TOPICS = [
    # 候选主题词
    "人工智能", "深度学习", "大语言模型", "卡奥斯平台",
    "云计算", "大数据", "物联网", "区块链", "5G",
    "量子计算", "边缘计算", "机器学习", "自然语言处理"
]


def generate_random_text(target_tokens: int, seed: int = None) -> str:
    """
    生成长文本作为 prompt。
    将随机句子拼接达到 target_tokens 对应的字符数（1 token ≈ 1.5 个中文字符）。
    """
    rng = random.Random(seed)
    target_chars = int(target_tokens * 1.5)  # 中文字符转 token 的粗略比例

    # 用 UUID 作为开头，增加文本随机性
    result = [str(uuid.UUID(int=rng.getrandbits(128)))]
    current_len = 0
    while current_len < target_chars:
        result.append(rng.choice(_SENTENCES))
        current_len += len(result[-1])
    t = "".join(result)[:target_chars]  # 截断到目标长度
    return t


def generate_exact_token_prompt(tokenizer, target_tokens: int, seed: int) -> str:
    """构造应用 chat template 后长度精确等于 target_tokens 的 prompt。"""
    def rendered_length(value) -> int:
        if hasattr(value, "get") and value.get("input_ids") is not None:
            value = value["input_ids"]
        return len(value)

    text = generate_random_text(target_tokens * 2, seed=seed)
    filler_ids = tokenizer.encode(_SENTENCES[0], add_special_tokens=False)
    for _ in range(8):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        delta = target_tokens - rendered_length(rendered)
        if delta == 0:
            return text
        content_ids = tokenizer.encode(text, add_special_tokens=False)
        if delta < 0:
            content_ids = content_ids[:delta]
        else:
            repeats = (delta + len(filler_ids) - 1) // len(filler_ids)
            content_ids.extend((filler_ids * repeats)[:delta])
        text = tokenizer.decode(content_ids, skip_special_tokens=True)
    actual = rendered_length(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    raise RuntimeError(
        f"failed to construct exact prompt length: target={target_tokens}, actual={actual}"
    )


def generate_random_prompt(base_text: str = None, use_long: bool = False, rng: random.Random = None) -> str:
    """
    生成一个随机短 prompt（问句形式）。
    从模板和主题中各随机选取一个组合成问句。
    """
    if rng is None:
        rng = random.Random()
    template = rng.choice(_TEMPLATES)
    topic = rng.choice(_TOPICS)
    if use_long and base_text:
        return template.format(topic) + "\n\n" + base_text  # 在长文本前加问题
    return template.format(topic)


# ==================== 全局请求计数器 ====================
# 用于为每个请求分配唯一 seed，保证随机 prompt 不重复
_request_counter = 0
_progress_lock = asyncio.Lock()
_bench_start_time = 0.0


def _print_progress(completed: int, first_token: int, total: int, results: list = None):
    """打印实时进度条（含已用时间和实时 token 速度）"""
    elapsed = time.time() - _bench_start_time
    started = min(completed + first_token, total)
    avg_total_tps = 0.0
    avg_input_tps = 0.0
    avg_output_tps = 0.0
    if results:
        total_vals = [r.total_tps for r in results if r.success and r.total_tps > 0]
        input_vals = [r.input_tps for r in results if r.success and r.input_tps > 0]
        output_vals = [r.output_tps for r in results if r.success and r.output_tps > 0]
        if total_vals:
            avg_total_tps = sum(total_vals) / len(total_vals)
        if input_vals:
            avg_input_tps = sum(input_vals) / len(input_vals)
        if output_vals:
            avg_output_tps = sum(output_vals) / len(output_vals)
    if started == 0:
        print(f"\r{_progress_bar(0, total)}  已用 {elapsed:.0f}s", flush=True, end="")
    else:
        bar = _progress_bar(started, total)
        info = f"  成功={completed} 解码中={first_token}  输入={avg_input_tps:.1f} t/s  输出={avg_output_tps:.1f} t/s  全部={avg_total_tps:.1f} t/s  已用 {elapsed:.0f}s"
        print(f"\r{bar} {info}", flush=True, end="")


async def _timer_loop(total: int, stop_event: asyncio.Event, completed_ref: list, decoding_ref: list):
    """后台计时器，每秒刷新显示已用时间和实时总 token 速度"""
    while not stop_event.is_set():
        _print_progress(len(completed_ref), decoding_ref[0], total, completed_ref)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=0.2)
        except asyncio.TimeoutError:
            continue


async def run_benchmark(args) -> tuple[BenchReport, list]:
    """
    基准测试主循环。
    根据参数决定并发/串行执行，收集所有结果后生成报告。
    """
    global _request_counter, _bench_start_time
    _request_counter = 0
    _bench_start_time = time.time()
    results: list[RequestResult] = []
    tokenizer = None
    if args.exact_input_tokens and args.input_tokens > 0:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base_prompt = args.prompt
    if args.input_tokens > 0:
        base_prompt = generate_random_text(args.input_tokens, seed=args.seed)

    # 创建 TCP 连接器，limit 控制最大并发连接数
    connector = aiohttp.TCPConnector(limit=args.concurrent + 2, ssl=not args.no_ssl_verify)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def run_single(idx: int, on_first_token=None) -> tuple[int, RequestResult]:
            """单个请求包装函数：生成 prompt → 发起请求 → 返回结果"""
            global _request_counter
            _request_counter += 1
            request_seed = args.seed + idx
            rng = random.Random(request_seed)
            if args.random_prompt:
                if args.input_tokens > 0:
                    if tokenizer is not None:
                        prompt = generate_exact_token_prompt(
                            tokenizer, args.input_tokens, request_seed
                        )
                    else:
                        prompt = generate_random_text(args.input_tokens, seed=request_seed)
                else:
                    # 短文本模式：随机选模板/主题生成问句
                    prompt = generate_random_prompt(use_long=False, rng=rng)
            else:
                prompt = base_prompt  # 所有请求使用相同的固定 prompt
            result = await benchmark_request(
                session,
                args.url,
                args.api_key,
                args.model,
                prompt,
                args.max_tokens,
                on_first_token=on_first_token,
                estimated_input_tokens=args.input_tokens,
                request_index=idx,
                temperature=args.temperature,
                sampling_seed=request_seed,
                request_timeout=args.request_timeout,
                ignore_eos=args.ignore_eos,
            )
            return idx, result

        if args.concurrent > 1:
            # —— 并发模式（信号量控制，不分组批等待） ——
            stop_timer = asyncio.Event()
            decoding_ref = [0]
            timer_task = asyncio.create_task(_timer_loop(args.runs, stop_timer, results, decoding_ref))
            sem = asyncio.Semaphore(args.concurrent)
            results_lock = asyncio.Lock()

            async def run_with_sem(idx: int) -> tuple[int, RequestResult]:
                first_token_seen = [False]
                cb = make_ft_callback(first_token_seen)
                async with sem:
                    idx_r, r = await run_single(idx, on_first_token=cb)
                if first_token_seen[0]:
                    decoding_ref[0] -= 1
                async with results_lock:
                    results.append(r)

            def make_ft_callback(first_token_seen):
                def on_ft():
                    first_token_seen[0] = True
                    decoding_ref[0] += 1
                    _print_progress(len(results), decoding_ref[0], args.runs, results)
                return on_ft

            tasks = []
            arrival_rng = random.Random(args.seed)
            for i in range(args.runs):
                if i > 0 and args.request_rate > 0:
                    await asyncio.sleep(arrival_rng.expovariate(args.request_rate))
                tasks.append(asyncio.create_task(run_with_sem(i)))
            await asyncio.gather(*tasks)
            stop_timer.set()
            await timer_task
            _print_progress(len(results), 0, args.runs, results)
        else:
            # —— 串行模式 ——
            for i in range(args.runs):
                decoded = False
                def on_first():
                    nonlocal decoded
                    decoded = True
                    _print_progress(i, 1, args.runs, results)
                _, result = await run_single(i, on_first_token=on_first)
                results.append(result)
                _print_progress(i + 1, 0, args.runs, results)
                if i < args.runs - 1:
                    await asyncio.sleep(0.5)  # 串行请求间间隔 0.5 秒

        print()  # 进度条结束后换行

    report = compute_report(results)
    return report, results


def main():
    """入口函数：解析参数 → 运行测试 → 输出结果"""
    parser = argparse.ArgumentParser(description="LLM API 速度基准测试工具")
    parser.add_argument("--url", "-u", default="https://gpt.cosmoplat.com/v1/chat/completions")
    parser.add_argument("--api-key", "-k", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--model", "-m", default="glm5")
    parser.add_argument("--prompt", "-p", default="介绍一下卡奥斯平台")
    parser.add_argument("--prompt-file", "-f", type=str)             # 从文件读取 prompt
    parser.add_argument("--max-tokens", "-t", type=int, default=512) # 最大输出 token 数
    parser.add_argument("--input-tokens", "-i", type=int, default=0) # 输入文本目标 token 数（0 表示使用固定 prompt）
    parser.add_argument("--random-prompt", "-r", action="store_true")# 每次请求使用随机 prompt
    parser.add_argument(
        "--exact-input-tokens",
        action="store_true",
        help="按模型 chat template 精确构造 input-tokens",
    )
    parser.add_argument("--runs", "-n", type=int, default=5)         # 总请求次数
    parser.add_argument("--concurrent", "-c", type=int, default=1)   # 并发数
    parser.add_argument(
        "--request-rate",
        type=float,
        default=0.0,
        help="请求到达率（RPS）；0 表示立即提交全部并发请求",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=900.0,
        help="单请求总超时秒数",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="忽略 EOS，强制生成到 max_tokens",
    )
    parser.add_argument("--no-ssl-verify", action="store_true")      # 跳过 SSL 验证（用于自签名证书）
    parser.add_argument("--output-json", "-j", type=str)             # JSON 输出路径
    parser.add_argument("--output-csv", "-s", type=str)              # CSV 输出路径
    parser.add_argument("--note", "-b", type=str, default="")        # 备注信息
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--summary-csv", type=str, default="")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            args.prompt = f.read()

    # 打印测试概况
    rate_text = f"  |  rps={args.request_rate:g}" if args.request_rate > 0 else ""
    print(f"\n  LLM API 速度基准测试  |  {args.model}  |  {args.runs}x{args.concurrent}{rate_text}  |  max_tokens={args.max_tokens}", flush=True)

    start_time = time.time()
    start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  开始时间: {start_str}", flush=True)

    # 运行基准测试（异步事件循环入口）
    report, results = asyncio.run(run_benchmark(args))
    total_elapsed = time.time() - start_time
    successful_results = [result for result in results if result.success]
    if total_elapsed > 0:
        total_input_tokens = sum(result.input_tokens for result in successful_results)
        total_output_tokens = sum(result.output_tokens for result in successful_results)
        report.request_throughput = len(successful_results) / total_elapsed
        report.aggregate_input_tps = total_input_tokens / total_elapsed
        report.aggregate_output_tps = total_output_tokens / total_elapsed
        report.aggregate_total_tps = (
            total_input_tokens + total_output_tokens
        ) / total_elapsed
    end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"  结束时间: {end_str}", flush=True)

    # 打印报告到终端
    print_report(report, args.model, total_elapsed, results)

    # 默认 CSV 导出（文件名包含模型和时间戳）
    csv_path = args.output_csv or os.path.join(
        args.output_dir,
        f"{args.tag or 'benchmark'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    save_csv(results, csv_path, args.model, start_str, end_str)
    print(f"  [保存] CSV: {csv_path}", flush=True)

    # 追加一行汇总记录到默认文件
    summary_path = args.summary_csv or os.path.join(
        args.output_dir, "benchmark_results.csv"
    )
    file_exists = os.path.isfile(summary_path)
    with open(summary_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["模型", "请求数", "并发数", "输入 Token", "最大输出 Token",
                             "开始时间", "结束时间", "耗时(秒)",
                             "首 Token 延迟-平均(秒)", "首 Token 延迟-标准差(秒)", "首 Token 延迟-P50(秒)", "首 Token 延迟-P90(秒)",
                             "输入速度-平均(tok/秒)", "输入速度-标准差(tok/秒)", "输入速度-P50(tok/秒)", "输入速度-P90(tok/秒)",
                             "输出速度-平均(tok/秒)", "输出速度-标准差(tok/秒)", "输出速度-P50(tok/秒)", "输出速度-P90(tok/秒)",
                             "全部速度-平均(tok/秒)", "全部速度-标准差(tok/秒)", "全部速度-P50(tok/秒)", "全部速度-P90(tok/秒)",
                             "端到端延迟-平均(秒)", "端到端延迟-标准差(秒)", "端到端延迟-P50(秒)", "端到端延迟-P90(秒)",
                             "平均输入 Token(tok)", "平均输出 Token(tok)", "成功率(%)", "备注"])
        writer.writerow([
            args.model, args.runs, args.concurrent, args.input_tokens, args.max_tokens,
            start_str, end_str, f"{total_elapsed:.1f}",
            f"{report.ttft_avg:.3f}", f"{report.ttft_std:.3f}", f"{report.ttft_p50:.3f}", f"{report.ttft_p90:.3f}",
            f"{report.input_tps_avg:.1f}", f"{report.input_tps_std:.1f}", f"{report.input_tps_p50:.1f}", f"{report.input_tps_p90:.1f}",
            f"{report.tps_avg:.1f}", f"{report.tps_std:.1f}", f"{report.tps_p50:.1f}", f"{report.tps_p90:.1f}",
            f"{report.total_tps_avg:.1f}", f"{report.total_tps_std:.1f}", f"{report.total_tps_p50:.1f}", f"{report.total_tps_p90:.1f}",
            f"{report.latency_avg:.3f}", f"{report.latency_std:.3f}", f"{report.latency_p50:.3f}", f"{report.latency_p90:.3f}",
            int(report.avg_input_tokens), int(report.avg_output_tokens), f"{report.success_rate:.1f}",
            args.note,
        ])
    print(f"  [记录] 汇总: {summary_path}", flush=True)

    # JSON 导出（如果指定了参数）
    if args.output_json:
        save_json(report, results, args.output_json, start_str, end_str)
        print(f"  [保存] JSON: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
