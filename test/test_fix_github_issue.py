from src.services.llm import LLM
from src.services.logical_tree import LogicalTree
from src.utils.workflow import query_based_on_tool_calls, fix_github_error
from typing import Callable, List, Dict, Any
from src.services.agents import Tool_Calls
from src.services.external_client import ExternalClient
import re

# from webui import launch_web

# from src.services.agents import Agent, Tool
from src.services.custom_tools import custom_tools, clean_training_logs
import glob
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, default=0, help="输入数字参数")
parser.add_argument("--verbose", type=bool, default=False, help="verbose")
args = parser.parse_args()

num = args.num
verbose = args.verbose
print("num =", num, ",verbose =", verbose)

# llm = LLM(model_name="qwen3:8b")
with open(".webui_port", "r", encoding="utf-8") as f:
    port = int(f.read())
if verbose:
    ec = ExternalClient(port=port)
else:
    ec = None
llm = LLM(
    model_name="deepseek-chat",
    llm_url="https://api.deepseek.com/chat/completions",
    api_key="sk-8969134148cb48c88377e5eefc6322c0",
    format="openai",
    # proxies={
    #     "http": "socks5h://127.0.0.1:1081",
    #     "https": "socks5h://127.0.0.1:1081",
    # },
    ec=ec,
)
# llm = LLM(
#     model_name="/data/yuanwen/models/qwen3-32b",
#     llm_url="http://localhost:8000/v1/chat/completions",
#     api_key="",
#     format="openai",
#     ec=ec
# )
# llm = LLM(
#     model_name="gpt-5-2025-08-07",
#     llm_url="https://api.aimlapi.com/v1/chat/completions",
#     api_key="9ce046a9681446c48427b3fe4dd7cdd4",
#     format="openai",
# )
# llm = LLM(
#     model_name="gpt-4o",
#     llm_url="https://api.aimlapi.com/v1/chat/completions",
#     api_key="9ce046a9681446c48427b3fe4dd7cdd4",
#     format="openai",
# )

# print("A")
import os
ct = custom_tools(
    MAIN_DIR=f"/data/yuanwen/workspace/swe-solution/lite/{num}",
    WRITE_DIR=f"/data/yuanwen/workspace/swe-solution/lite/{num}",
    MAIN_DATA_DIR=f"/data/yuanwen/workspace/swe-solution/lite/{num}",
    PYTHON_PATH="python",
    # MATLAB_PATH="/Applications/MATLAB_R2023b.app/bin/matlab",
    LOCAL_TMP_PATH=f"/data/yuanwen/workspace/tmp/{num}",
    CHUNK_SIZE=5000,
    llm=llm,
    verbose=verbose,
)
os.makedirs(ct.MAIN_DIR, exist_ok=True)
from src.services.llm import load_messages, save_messages

LOG_DIR = (
    f"/data/yuanwen/workspace/tmp/{num}"
)

import requests
from bs4 import BeautifulSoup

def fetch_problem_statement(issue_url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SWEAgentFetcher/1.1)"}

    # ✅ 设置代理（可根据你的代理端口修改）
    # proxies = {
    #     "http": "http://127.0.0.1:7897",
    #     "https": "http://127.0.0.1:7897",
    # }

    # ✅ 设置超时防止卡死
    resp = requests.get(issue_url, headers=headers, timeout=15)

    if resp.status_code != 200:
        raise RuntimeError(f"无法访问 {issue_url}, 状态码: {resp.status_code}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # ✅ 标题兼容新版 GitHub 结构
    title_tag = soup.find("bdi", {"class": "js-issue-title"})
    if not title_tag:
        title_tag = soup.find("span", {"class": "js-issue-title"})
    title = title_tag.get_text(strip=True) if title_tag else "(No title found)"

    # ✅ 正文
    body_tag = soup.find("div", {"class": "markdown-body"})
    body = (
        body_tag.get_text("\n", strip=True) if body_tag else "(No body content found)"
    )

    return f"{body}"


def fetch_problem_statement(issue_url: str) -> str:
    """从 GitHub issue 页面抓取标题+正文"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SWEAgentFetcher/1.1)"}
    resp = requests.get(issue_url, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"无法访问 {issue_url}, 状态码: {resp.status_code}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # 标题 (兼容新版 GitHub)
    title_tag = soup.find("bdi", {"class": "js-issue-title"}) or soup.find(
        "span", {"class": "js-issue-title"}
    )
    title = title_tag.get_text(strip=True) if title_tag else "(No title found)"

    # 正文
    body_tag = soup.find("div", {"class": "markdown-body"})
    body = (
        body_tag.get_text("\n", strip=True) if body_tag else "(No body content found)"
    )

    return f"{title}\n\n{body}".strip()


def build_sample_from_github(
    repo_url: str, issue_url: str, base_commit: str = "main"
) -> dict:
    """
    输入:
        repo_url:  仓库主页, 如 "https://github.com/SWE-agent/test-repo"
        issue_url: Issue 链接, 如 "https://github.com/SWE-agent/test-repo/issues/1"
        base_commit: 可选, 默认为 'main'
    输出:
        SWE-bench 风格的 sample 字典
    """
    # 提取 repo 名称，例如 "SWE-agent/test-repo"
    repo_match = re.search(r"github\.com/([^/]+/[^/]+)", repo_url)
    if not repo_match:
        raise ValueError(f"无法从 {repo_url} 提取仓库名称")
    repo = repo_match.group(1)

    # 提取 issue 编号
    issue_match = re.search(r"/issues/(\d+)", issue_url)
    if not issue_match:
        raise ValueError(f"无法从 {issue_url} 提取 issue 编号")
    issue_num = issue_match.group(1)

    # 获取问题描述
    problem_text = fetch_problem_statement(issue_url)

    # 构造 sample
    sample = {
        "repo": repo,
        "repo_clone_url": f"https://github.com/{repo}.git",
        "issue_url": issue_url,
        "base_commit": base_commit,
        "problem_statement": problem_text,
    }
    return sample


# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用清华镜像
from datasets import load_dataset

ds = load_dataset("princeton-nlp/SWE-bench_Lite")

print(ds)
sample = ds["test"][num]
print("A")
# sample = build_sample_from_github(
#     repo_url="https://github.com/SWE-agent/test-repo",
#     issue_url="https://github.com/SWE-agent/test-repo/issues/1",
# )
# sample["instance_id"] = "test"
print(sample)
import time
time.sleep(1)
package_name = "package"
tc = Tool_Calls(LOG_DIR=LOG_DIR, MAX_CHAR=40000)
time.sleep(1)
tc.clear()

# launch_web()
os.environ["NO_PROXY"] = "*"
message = fix_github_error(
    sample=sample,
    ct=ct,
    max_steps=300,
    tc=tc,
    package_name=package_name,
    verbose=verbose,
    whether_recreate=False,
    save_filepath=f"{ct.MAIN_DIR}/fix_issue.patch",
)
# print(ct.func_reflect(text="""
#                       检查补丁
#                       diff --git a/astropy/io/fits/fitsrec.py b/astropy/io/fits/fitsrec.py
# index 574b4073b1..aa0f23cb69 100644
# --- a/astropy/io/fits/fitsrec.py
# +++ b/astropy/io/fits/fitsrec.py
# @@ -1261,7 +1261,7 @@ class FITS_rec(np.recarray):
 
#          # Replace exponent separator in floating point numbers
#          if 'D' in format:
# -            output_field.replace(encode_ascii('E'), encode_ascii('D'))
# +            output_field = output_field.replace(encode_ascii('E'), encode_ascii('D'))
#                       """))
# print(llm.query("你好", verbose=False))
