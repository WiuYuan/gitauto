# %% [Setup]
from src.services.llm import LLM
from src.services.custom_tools import custom_tools
from src.services.external_client import ExternalClient
from src.services.agents import Tool_Calls
from src.utils.thought_graph import ThoughtGraph  # 假设你把 ThoughtGraph 放在这里
import os, time, json

os.environ["NO_PROXY"] = "*"

# 读取 WebUI 端口并连接 ExternalClient
with open(".webui_port", "r", encoding="utf-8") as f:
    port = int(f.read())
ec = ExternalClient(port=port)

# === 初始化 LLM ===
llm = LLM(
    model_name="deepseek-chat",
    llm_url="https://api.deepseek.com/chat/completions",
    api_key="sk-2332c3d16a8d4f4ba1b3503074ba04c5",
    format="openai",
    ec=ec,
)

# === 初始化 custom tools（保持一致）===
ct = custom_tools(
    MAIN_DIR="/data/yuanwen/workspace/gitauto",
    PYTHON_PATH="python",
    MATLAB_PATH="/Applications/MATLAB_R2023b.app/bin/matlab",
    LOCAL_TMP_PATH="/data/yuanwen/workspace/tmp",
    PMC_URL="https://pmc.ncbi.nlm.nih.gov/articles/PMC7567795",
    llm=llm,
)

# === 初始化 Tool Calls（保持一致）===
tool_calls_path = "/data/yuanwen/workspace/tmp/tool_calls_path.json"
env_tool_calls_path = "/workspace/tool_calls_path.json"
tc = Tool_Calls(PATH=tool_calls_path, ENV_PATH=env_tool_calls_path, MAX_CHAR=50000)
tc.clear()

# %% [ThoughtGraph test]
from src.utils.thought_graph import ThoughtGraph

# 初始化思维图
tg = ThoughtGraph(llm=llm, verbose=True)

# 添加结点（可以自行指定初始 instruction）
tg.add_node("check", instruction="这一步, 你需要假设前面对于问题的分析和解决办法是错误的, 找到错误的地方, 你不需要提供任何的理由")
tg.add_node("solve", instruction="这一步你需要生成可能的解决方案, 你不能输出任何的思考内容")
tg.add_node("think", instruction="这一步你需要仔细思考每一个细节, 注意你需要放慢步骤, 问题不需要你一次性解决, 你不需要提供任何的解决方案, 只有思考细节, 而且不能超过100个字符")
tg.add_node("long think", instruction="这一步你需要仔细思考每一个细节, 注意你需要放慢步骤, 问题不需要你一次性解决, 你不需要提供任何的解决方案, 只有思考细节")

# 添加边：Goal → Analysis → Implementation
tg.add_edge("think", "solve")
tg.add_edge("solve", "check")
tg.add_edge("check", "think")
tg.add_edge("solve", "think")
tg.add_edge("think", "long think")
tg.add_edge("long think", "think")

question = """
(venv) (testbed) root@17c306c8458b:/workspace/django# cat fix_full_corrected.patch
diff --git a/django/db/models/fields/__init__.py b/django/db/models/fields/__init__.py
index 1234567..abcdefg 100644
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -1664,6 +1664,8 @@ class FilePathField(Field):
     def __init__(self, verbose_name=None, name=None, path='', match=None,
                  recursive=False, allow_files=True, allow_folders=False, **kwargs):
         self.path, self.match, self.recursive = path, match, recursive
+        if callable(self.path):
+            self.path = self.path()
         self.allow_files, self.allow_folders = allow_files, allow_folders
         kwargs.setdefault('max_length', 100)
         super().__init__(verbose_name, name, **kwargs)
@@ -1688,7 +1690,7 @@ class FilePathField(Field):
 
     def deconstruct(self):
         name, path, args, kwargs = super().deconstruct()
-        if self.path != '':
+        if self.path != '' and not callable(self.path):
             kwargs['path'] = self.path
         if self.match is not None:
             kwargs['match'] = self.match
@@ -1709,7 +1711,10 @@ class FilePathField(Field):
 
     def formfield(self, **kwargs):
         return super().formfield(**{
-            'path': self.path,
+            'path': self.path() if callable(self.path) else self.path,
             'match': self.match,
             'recursive': self.recursive,
             'form_class': forms.FilePathField,
             'allow_files': self.allow_files,
             'allow_folders': self.allow_folders,
             **kwargs,
         })
(venv) (testbed) root@17c306c8458b:/workspace/django# patch --dry-run -p1 < fix_full_corrected.patch
checking file django/db/models/fields/__init__.py
Hunk #2 succeeded at 1689 (offset -1 lines).
patch: **** malformed patch at line 32:              'allow_files': self.allow_files,

(venv) (testbed) root@17c306c8458b:/workspace/django#
以上是问题, 你需要使用**中文**分析问题, 你必须依靠自己解决, 我不会提供任何帮助比如运行程序等
"""
# 初次计算（可以让 LLM 生成 summary）
list = ["solve", "think", "check"]
for m in range(3):
    tg.recompute_node("long think", question=question)
    tg.recompute_node("think", question=question)
    tg.recompute_node("solve", question=question)
    tg.recompute_node("check", question=question)

# 输出整个图的 JSON
# print(tg.export_json())

# 打印最终结果
# for node_id in tg.G.nodes:
#     s = tg.G.nodes[node_id].get("state")
#     print(f"\n🧩 Node: {node_id}\nSummary: {s.summary_text if s else '[Empty]'}")