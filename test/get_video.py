from src.services.llm import LLM
from src.services.logical_tree import LogicalTree
from src.utils.workflow import solve_goal_list
from typing import Callable, List, Dict, Any
from src.services.agents import Tool_Calls

# from src.services.agents import Agent, Tool
from src.services.custom_tools import custom_tools, clean_training_logs
from src.services.external_client import ExternalClient
import glob
import os

os.environ["NO_PROXY"] = "*"

# llm = LLM(model_name="qwen3:8b")

with open(".webui_port", "r", encoding="utf-8") as f:
    port = int(f.read())
ec = ExternalClient(port=port, task_id="get_vedio")
llm = LLM(
    model_name="deepseek-chat",
    llm_url="https://api.deepseek.com/chat/completions",
    api_key="sk-2332c3d16a8d4f4ba1b3503074ba04c5",
    format="openai",
    ec=ec,
)
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


ct = custom_tools(
    MAIN_DIR="/data/yuanwen/workspace/video",
    WRITE_DIR="/data/yuanwen/workspace/video",
    PYTHON_PATH="python",
    MATLAB_PATH="/Applications/MATLAB_R2023b.app/bin/matlab",
    LOCAL_TMP_PATH="/data/yuanwen/workspace/tmp",
    PMC_URL="https://pmc.ncbi.nlm.nih.gov/articles/PMC7567795",
    llm=llm,
)
from src.services.llm import load_messages, save_messages

tool_calls_path = (
    "/data/yuanwen/workspace/tmp/tool_calls_path.json"
)
env_tool_calls_path = "/workspace/tool_calls_path_video.json"

tc = Tool_Calls(PATH=tool_calls_path, ENV_PATH=env_tool_calls_path, MAX_CHAR=50000)
tc.clear()

tools = [
    ct.func_ls,
    ct.func_cat,
    # ct.func_cat_with_llm,
    ct.func_write,
    ct.func_modify,
    ct.func_insert,
    ct.func_append,
    ct.func_prepend,
    ct.func_python,
    ct.func_cmd,
    ct.func_matlab,
    # ct.func_fetch_info_from_pmc_with_llm,
    ct.func_human,
    ct.func_guide,
    ct.func_think,
    ct.add_child,
    ct.return_to_parent,
    # ct.func_youtube_search_with_transcript,
]
goal_text = "在这个文件夹下, 针对每个前沿领域, 请你使用我申请到的是Youtube的API是AIzaSyCrA-ISpOpkZ_hdvrBNeEedEdxQ7HH09hM, 搜索大量领域从而拿到目前高质量, 冲击效果强的视频, 暂时各个领域给我10个左右, 并且整理到这个文件夹下, 我希望每个领域单独一个文件, 每个文件都有对应的每个视频的链接与大致简介, 注意由于IP限制, 有些时候可能搜索收到限流, 但你可以多次重复达到目的, 以及之前已经整理过一些领域你可以直接跳过, 视频必须是高质量的, 比如时长要超过10min, 播放量也得相应较高, 而且尽量同一个领域的视频不要都一样, 视频可以是那种讲述细节的, 也可以是概括的, 且必须是英文视频, 你搜索的主题也都得是English, 你的概括则需要是中文的, 且需要说明你认为高质量的原因和列出上述基本数据"
message = solve_goal_list(
    goal_list=[("goal", goal_text)],
    ct=ct,
    max_steps=10000,
    tc=tc,
    tree_filepath="/data/yuanwen/workspace/tmp/logical_tree_video.json",
    verbose=True,
    human_decision=False,
    tools=tools,
    human_check_before_calling=False,
)
# print(ct.func_ls("tmp"))
