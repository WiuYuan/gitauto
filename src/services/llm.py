# src/services/llm.py

import requests
import json
import inspect
from typing import Callable, List, Dict, Any, Union, Optional
from src.utils.prompts import get_prompt
from src.services.agents import Tool_Calls, get_func_tool_call
import copy
import base64

# from src.services.agents import Tool
from src.services.summary_attention_dag import SummaryAttentionDAG
import os
import uuid
from pathlib import Path
from collections import defaultdict
from src.services.external_client import ExternalClient


os.environ["NO_PROXY"] = "*"


class LLM:
    def __init__(
        self,
        api_key: str = "",
        llm_url: str = "http://localhost:11434/api/chat",
        model_name: str = "qwen3:8b",
        remove_think: str = True,
        proxies: dict = None,
        format: str = "ollama",
        system_prompt: str = "",
        ec: Optional[ExternalClient] = None,
    ):
        """
        Initialize the LLM instance.

        Parameters:
        llm_url (str): The URL of the LLM service (e.g., Ollama API endpoint).
        model_name (str): The model to use. Default is "qwen3:32b".
        remove_think (bool): Whether to remove <think>...</think> sections from the response. Default is True.
        """
        self.api_key = api_key
        self.llm_url = llm_url
        self.model_name = model_name
        self.remove_think_enabled = remove_think
        self.proxies = proxies or {"http": None, "https": None}
        self.format = format
        self.system_prompt = system_prompt
        self.ec = ec

    def remove_think(self, text: str) -> str:
        """
        Remove <think>...</think> sections from the text and trim surrounding whitespace.

        Parameters:
        text (str): The input text containing potential <think> sections.

        Returns:
        str: Cleaned text without <think> blocks.
        """
        start_tag = "<think>"
        end_tag = "</think>"

        start_idx = text.find(start_tag)
        if start_idx != -1:
            end_idx = text.find(end_tag, start_idx)
            if end_idx != -1:
                # Remove the entire <think> block including the tags
                text = text[:start_idx] + text[end_idx + len(end_tag) :]

        # Trim whitespace at the start and end
        return text.strip()

    def query(self, prompt: str, verbose: bool = True) -> str:
        if verbose:
            # print("[LLM] Query")
            self.ec.send_message(
                {
                    "type": "info",
                    "data": {
                        "category": "LLM",
                        "content": "Query",
                        "level_delta": 1,
                    },
                }
            )
            formatted_prompt = prompt.replace("\n", "\n    ")
            formatted_system_prompt = self.system_prompt.replace("\n", "\n    ")
            # print(f"  [SYSTEM PROMPT]\n    {formatted_system_prompt}")
            self.ec.send_message(
                {
                    "type": "info",
                    "data": {
                        "category": "SYSTEM PROMPT",
                        "content": formatted_system_prompt,
                        "level_delta": 0,
                    },
                }
            )
            # print(f"  [INPUT]\n    {formatted_prompt}")
            self.ec.send_message(
                {
                    "type": "info",
                    "data": {
                        "category": "INPUT",
                        "content": formatted_prompt,
                        "level_delta": 0,
                    },
                }
            )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        result = self.query_messages(messages, verbose=verbose)
        if verbose:
            self.ec.send_message(
                {
                    "type": "info",
                    "data": {
                        "category": "",
                        "content": "",
                        "level_delta": -1,
                    },
                }
            )
        return result

    def query_with_tools(
        self,
        prompt: Union[str, Callable[[], str]],
        max_steps: int,
        tc: Tool_Calls,
        extra_guide_tool_call: List[Dict[str, Any]] = [],
        tools: List[Callable] = None,
        verbose: bool = True,
        stop_condition: Callable[..., bool] = None,
        check_start_prompt: str = None,
    ) -> str:
        # sample_size = 5
        # if verbose:
        #     print(prompt() if callable(prompt) else prompt)
        func_dict = {func.__name__: func for func in tools}
        messages = [
            {"role": "user", "content": prompt() if callable(prompt) else prompt}
        ]
        all_texts = []
        for step in range(max_steps):
            # resample = False
            print(f"\n=== Prompt step {step+1} ===\n")
            if callable(prompt):
                messages[0]["content"] = prompt()
                # if verbose:
                #     print(prompt())
            text = ""
            if check_start_prompt is None:
                text, tool_calls = self.query_messages_with_tools(
                    messages + tc.get_value() + extra_guide_tool_call,
                    tools=tools,
                    verbose=verbose,
                )
            else:
                while not text.startswith(check_start_prompt):
                    text, tool_calls = self.query_messages_with_tools(
                        messages + tc.get_value() + extra_guide_tool_call,
                        tools=tools,
                        verbose=verbose,
                    )
                    if not text.startswith(check_start_prompt):
                        print(f"\nCheck Failed, Generate Again!\n")

            all_texts.append(text)
            new_tool_calls = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                }
            ]

            for call in tool_calls:
                if self.format == "ollama":
                    call = call["function"]
                func_name = call["name"]
                args = call["function"]["arguments"]

                if func_name in func_dict:
                    if isinstance(args, str):
                        args = json.loads(args)
                    if verbose:
                        formatted_args = self._format_arguments_for_display(
                            func_name, args
                        )
                        formatted_args = formatted_args.replace("\n", "\n    ")
                        print(
                            f"\nCalling function '{func_name}' with arguments:\n{formatted_args}"
                        )
                    result = func_dict[func_name](**args)
                else:
                    result = f"Function {func_name} not found"

                new_tool_calls.append(
                    {
                        "role": "tool",
                        "name": func_name,
                        "tool_call_id": call.get("id", ""),
                        "content": str(result),
                    }
                )
            tc.extend(new_tool_calls)
            if tc.UpdataFunc is not None:
                tc.UpdataFunc()
            if stop_condition and stop_condition(tool_calls=tool_calls):
                if verbose:
                    print(f"\n=== Stopping at step {step+1} ===\n")
                break
        return "\n".join(all_texts)

    def query_messages(self, messages: str, verbose: bool = True) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            # "logprobs": True,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        # print(messages)

        text_accumulate = ""

        # if self.model_name == "human":
        #     open(
        #         "/Users/yuanwen/Desktop/Docker_Environment/intern2/2/test_prompt.txt",
        #         "w",
        #         encoding="utf-8",
        #     ).write(prompt)
        #     option = input()
        #     if option == "1":
        #         raise RuntimeError("exit")
        #     text_accumulate = open(
        #         "/Users/yuanwen/Desktop/Docker_Environment/intern2/2/test_answer.txt",
        #         "r",
        #     ).read()
        #     return text_accumulate

        # Make a streaming POST request
        if verbose:
            # print(f"  [OUTPUT]\n    ", end="", flush=True)
            self.ec.send_message(
                {
                    "type": "info",
                    "data": {
                        "category": "OUTPUT",
                        "content": "",
                        "level_delta": 1,
                    },
                }
            )
        with requests.post(
            self.llm_url,
            headers=headers,
            json=payload,
            proxies=self.proxies,
            stream=True,
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                # print(line)
                line_str = line.decode("utf-8").strip()
                if self.format == "ollama":
                    chunk = json.loads(line_str)
                    token = None
                    if "message" in chunk and "content" in chunk["message"]:
                        token = chunk["message"]["content"]

                    if token:
                        text_accumulate += token
                        if verbose:
                            # print(token.replace("\n", "\n    "), end="", flush=True)
                            self.ec.send_message(
                                {
                                    "type": "info",
                                    "data": {
                                        "category": "",
                                        "content": token,
                                        "level_delta": 0,
                                    },
                                }
                            )

                if self.format == "openai":
                    line_str = line_str[len("data: ") :]
                    if line_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line_str)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON from line: {line_str}")
                        if line_str.startswith("-alive"):
                            continue
                        print(messages)
                        raise
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            token = delta["content"]

                    if token:
                        text_accumulate += token
                        if verbose:
                            # print(token.replace("\n", "\n    "), end="", flush=True)
                            self.ec.send_message(
                                {
                                    "type": "info",
                                    "data": {
                                        "category": "",
                                        "content": token,
                                        "level_delta": 0,
                                    },
                                }
                            )

        # Optionally remove <think> blocks
        if self.remove_think_enabled:
            text_accumulate = self.remove_think(text_accumulate)

        if verbose:
            # print("")
            self.ec.send_message(
                {
                    "type": "info",
                    "data": {
                        "category": "",
                        "content": "",
                        "level_delta": -1,
                    },
                }
            )

        return text_accumulate

    def _format_arguments_for_display(self, func_name: str, args: dict) -> str:
        """
        Format function arguments for better display, especially for code content.

        Parameters:
            func_name (str): The name of the function being called
            args (dict): The arguments dictionary

        Returns:
            str: Formatted string representation of arguments
        """
        formatted_args = []

        for key, value in args.items():
            if isinstance(value, str):
                # Special formatting for text/code content in write functions
                if key in ["text", "new_text", "old_text", "info"]:
                    # If it looks like code (contains newlines), format it nicely
                    if "\n" in value:
                        formatted_value = f"\n{'-'*40}\n{value}\n{'-'*40}"
                    else:
                        formatted_value = repr(value)
                else:
                    formatted_value = repr(value)
            else:
                formatted_value = repr(value)

            formatted_args.append(f"  {key}: {formatted_value}")

        return "\n".join(formatted_args)

    def query_messages_with_tools(
        self,
        messages: str,
        tools: Union[str, Callable[[], str]] = None,
        verbose: bool = True,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        tools = tools or []
        tools = self.create_tools(tools)

        for msg in messages:
            if "tool_calls" in msg:
                for call in msg["tool_calls"]:
                    if "function" in call and "arguments" in call["function"]:
                        args = call["function"]["arguments"]
                        if not isinstance(args, str):
                            # 只有在 dict/非 str 的时候才转为 JSON 字符串
                            call["function"]["arguments"] = json.dumps(args)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "stream": True,
            # "logprobs": True,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        text_accumulate = ""

        # Make a streaming POST request
        tool_calls = []
        if verbose:
            self.ec.send_message(
                {
                    "type": "info",
                    "data": {
                        "category": "OUTPUT",
                        "content": "",
                        "level_delta": 1,
                    },
                }
            )
            # print(f"  [OUTPUT]\n    ", end="", flush=True)
        with requests.post(
            self.llm_url,
            headers=headers,
            json=payload,
            proxies=self.proxies,
            stream=True,
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                # print(line)
                line_str = line.decode("utf-8").strip()
                if self.format == "ollama":
                    chunk = json.loads(line_str)
                    token = None
                    if "message" in chunk and "content" in chunk["message"]:
                        token = chunk["message"]["content"]

                    if token:
                        text_accumulate += token
                        if verbose:
                            self.ec.send_message(
                                {
                                    "type": "info",
                                    "data": {
                                        "category": "",
                                        "content": token,
                                        "level_delta": 0,
                                    },
                                }
                            )
                            # print(token.replace("\n", "\n    "), end="", flush=True)

                    if "message" in chunk and "tool_calls" in chunk["message"]:
                        tool_calls.extend(chunk["message"]["tool_calls"])

                if self.format == "openai":
                    line_str = line_str[len("data: ") :]
                    if line_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line_str)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON from line: {line_str}")
                        if line_str.startswith("-alive"):
                            continue
                        print(messages)
                        raise
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            token = delta["content"]
                        if "tool_calls" in delta:
                            tool_calls.extend(delta["tool_calls"])
                            # print(chunk)

                    if token:
                        text_accumulate += token
                        if verbose:
                            self.ec.send_message(
                                {
                                    "type": "info",
                                    "data": {
                                        "category": "",
                                        "content": token,
                                        "level_delta": 0,
                                    },
                                }
                            )
                        # print(token.replace("\n", "\n    "), end="", flush=True)

        # Optionally remove <think> blocks
        if self.remove_think_enabled:
            text_accumulate = self.remove_think(text_accumulate)

        if self.format == "openai":
            grouped = defaultdict(list)
            for call in tool_calls:
                idx = call.get("index", 0)
                args = call.get("function", {}).get("arguments", "")
                grouped[idx].append(args)

            tool_calls_clean = []
            for idx, parts in grouped.items():
                full_args_str = "".join(parts).strip()
                try:
                    full_args = json.loads(full_args_str) if full_args_str else {}
                except json.JSONDecodeError:
                    full_args = full_args_str

                func_name = None
                fields = ["id", "type", "function"]
                extracted = {}
                for call in tool_calls:
                    if call.get("index") == idx:
                        if "name" in call.get("function", {}):
                            func_name = call["function"]["name"]
                        for field in fields:
                            if field in call:
                                extracted[field] = call.get(field)
                        break

                new_tool_call = {
                    **extracted,
                    "index": idx,
                    "name": func_name,
                }
                new_tool_call["function"]["arguments"] = full_args

                tool_calls_clean.append(new_tool_call)
            tool_calls = tool_calls_clean

        if verbose:
            self.ec.send_message(
                {
                    "type": "info",
                    "data": {
                        "category": "",
                        "content": "",
                        "level_delta": -1,
                    },
                }
            )
        # print("")

        return text_accumulate, tool_calls

    @classmethod
    def create_tools(cls, func_list: List[Callable]) -> List[Dict[str, Any]]:
        tools = []

        for func in func_list:
            sig = inspect.signature(func)
            func_name = func.__name__
            func_description = func.__doc__ or f"Function {func_name}"
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_type = "string"
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation in (int, float):
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                    else:
                        param_type = "string"

                param_description = f"Parameter {param_name}"

                properties[param_name] = {
                    "type": param_type,
                    "description": param_description,
                }

                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            tool = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": func_description.strip(),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
            tools.append(tool)

        return tools

    def query_with_tools_by_attention(
        self,
        prompt: Union[str, Callable[[], str]],
        max_steps: int,
        tc: Tool_Calls,
        extra_guide_tool_call: List[Dict[str, Any]] = [],
        tools: List[Callable] = None,
        verbose: bool = True,
        stop_condition: Callable[..., bool] = None,
        check_start_prompt: str = None,
        human_check_before_calling: bool = False,
    ) -> str:
        dag = SummaryAttentionDAG(m_layers=1, llm=self, parent_window=5, verbose=False)
        # 记录每个step生成的输出
        all_texts = []
        func_dict = {func.__name__: func for func in tools} if tools else {}
        max_chunk_id = 0

        for step in range(max_steps):
            print(f"\n[INFO] === Prompt step {step+1} ===\n")
            if verbose:
                self.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": f"STEP {step + 1}",
                            "content": f"=== Prompt step {step+1} ===",
                            "level_delta": 1,
                        },
                    }
                )
            if self.ec is not None:
                guidance = self.ec.get_all_guidance()
                if len(guidance) != 0:
                    guidance = (
                        guidance + "Above is the guidance from human, please follow it."
                    )
                    tc.extend(
                        get_func_tool_call(
                            func_name="func_guide",
                            result=guidance,
                            guidance="Will Return Guidance",
                        )
                    )
                    self.ec.send_message(
                        {
                            "type": "info",
                            "data": {
                                "category": f"GUIDANCE",
                                "content": guidance,
                                "level_delta": 0,
                            },
                        }
                    )
            previous_tool_calls = tc.get_value()
            previous_tool_calls_str = str(previous_tool_calls[max_chunk_id:])
            # 将chunk添加进DAG
            if len(previous_tool_calls_str) > tc.MAX_CHAR:
                new_chunk = ""
                backward_len = 0
                for id in range(max_chunk_id - 1, 0, -1):
                    backward_len += len(str(previous_tool_calls[id]))
                    new_chunk = str(previous_tool_calls[id]) + new_chunk
                    if previous_tool_calls[id].get("role") != "assistant":
                        continue
                    if backward_len > tc.MAX_CHAR / 5:
                        break
                forward_len = 0
                for id in range(max_chunk_id, len(previous_tool_calls)):
                    forward_len += len(str(previous_tool_calls[id]))
                    new_chunk = new_chunk + str(previous_tool_calls[id])
                    if (
                        id < len(previous_tool_calls) - 1
                        and previous_tool_calls[id + 1].get("role") != "assistant"
                    ):
                        continue
                    if forward_len > tc.MAX_CHAR / 2:
                        max_chunk_id = id + 1
                        break
                if verbose:
                    log_webui(
                        "DAG",
                        [
                            f"[STEP] {step}",
                            f"[CHUNK LENGTH] {len(new_chunk)} chars",
                            f"[BACKWARD LEN] {backward_len}, [FORWARD LEN] {forward_len}",
                            f"[CHUNK BUILT FROM] tool_calls[{max_chunk_id - int(forward_len / len(str(previous_tool_calls[max_chunk_id])))}..{max_chunk_id}]",
                            f"[PREVIOUS TOTAL NODES] {dag.N}",
                        ],
                        ec=self.ec,
                    )
                    # print("\n[INFO] ======== DAG Chunk Append ========")
                    # print(f"  [STEP] {step}")
                    # print(f"  [CHUNK LENGTH] {len(new_chunk)} chars")
                    # print(
                    #     f"  [BACKWARD LEN] {backward_len}, [FORWARD LEN] {forward_len}"
                    # )
                    # print(
                    #     f"  [CHUNK BUILT FROM] tool_calls[{max_chunk_id - int(forward_len / len(str(previous_tool_calls[max_chunk_id])))}..{max_chunk_id}]"
                    # )
                    # print(f"  [PREVIOUS TOTAL NODES] {dag.N}")
                    # print("----------------------------------------")
                dag.add_chunk(new_chunk)

            # 使用LLM执行主要任务（类似原来的 query_messages_with_tools）
            guidance = dag.get_last_neuron_value() or ""
            if guidance == "":
                pre_guide_tool_call = []
            else:
                guidance = guidance + "Above is the summary of previous tool calls"
                pre_guide_tool_call = get_func_tool_call(
                    func_name="func_guide",
                    result=guidance,
                    guidance="Will Return Guidance",
                )
            self.system_prompt = get_prompt("llm_system_prompt.txt").format(
                problem_description=prompt(),
                tool_calls_summary=Tool_Calls.summarize_tool_calls(
                    pre_guide_tool_call + previous_tool_calls[max_chunk_id:]
                ),
            )
            # dag.recompute_all_states(verbose=verbose)

            guidance = dag.get_last_neuron_value() or ""
            if guidance == "":
                pre_guide_tool_call = []
            else:
                guidance = guidance + "Above is the summary of previous tool calls"
                pre_guide_tool_call = get_func_tool_call(
                    func_name="func_guide",
                    result=guidance,
                    guidance="Will Return Guidance",
                )

            messages = [
                {
                    "role": "user",
                    "content": (prompt() if callable(prompt) else prompt),
                }
            ]
            if verbose:
                self.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": "LLM",
                            "content": "Query with Tools",
                            "level_delta": 1,
                        },
                    }
                )
                # print("\n[LLM] Query with Tools")
                format_prompt = prompt() + (dag.get_last_neuron_value() or "").replace(
                    "\n", "\n\t"
                )
                # format_prompt = format_prompt.replace("\n", "\n    ")
                if verbose:
                    self.ec.send_message(
                        {
                            "type": "info",
                            "data": {
                                "category": "INPUT",
                                "content": format_prompt,
                                "level_delta": 0,
                            },
                        }
                    )
                # print(f"  [INPUT]\n    {format_prompt}\n")
                format_tool_calls = Tool_Calls.summarize_tool_calls(
                    previous_tool_calls[max_chunk_id:]
                )
                # self.ec.send_message(
                #     {
                #         "type": "info",
                #         "data": {
                #             "category": "TOOL CALLS",
                #             "content": format_tool_calls,
                #             "level_delta": 0,
                #         },
                #     }
                # )
                # format_tool_calls = format_tool_calls.replace("\n", "\n    ")
                # print(f"  [TOOL CALLS]\n    {format_tool_calls}\n")
            if check_start_prompt is None:
                out_text, tool_calls = self.query_messages_with_tools(
                    messages
                    + pre_guide_tool_call
                    + previous_tool_calls[max_chunk_id:]
                    + extra_guide_tool_call,
                    tools=tools,
                    verbose=verbose,
                )
            else:
                out_text = ""
                tool_calls = []
                while not out_text.startswith(check_start_prompt):
                    out_text, tool_calls = self.query_messages_with_tools(
                        messages
                        + pre_guide_tool_call
                        + previous_tool_calls[max_chunk_id:]
                        + extra_guide_tool_call,
                        tools=tools,
                        verbose=verbose,
                    )
                    if not out_text.startswith(check_start_prompt):
                        print("\n[WARN] Check failed, regenerating...\n")

            if verbose:
                self.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": "",
                            "content": "",
                            "level_delta": -1,
                        },
                    }
                )

            all_texts.append(out_text)

            # 工具调用部分保留
            new_tool_calls = [
                {"role": "assistant", "content": "", "tool_calls": tool_calls}
            ]
            for call in tool_calls:
                if self.format == "ollama":
                    call = call["function"]
                func_name = call["name"]
                args = call["function"]["arguments"]

                if func_name in func_dict:
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError as e:
                            print("[ERROR] JSON 解析失败，原始 args 内容如下：")
                            print(repr(args))  # 用 repr 保留转义字符，防止打印不完整
                            print(f"[ERROR] 解析错误信息: {e}")
                            raise e
                    if verbose:
                        formatted_args = self._format_arguments_for_display(
                            func_name, args
                        )
                        formatted_args = formatted_args.replace("\n", "\n    ")
                        self.ec.send_message(
                            {
                                "type": "info",
                                "data": {
                                    "category": "CALL",
                                    "content": f"Calling '{func_name}' with arguments:\n{formatted_args}: ",
                                    "level_delta": 1,
                                },
                            }
                        )
                        # print(
                        #     f"\n[CALL] '{func_name}' with arguments:\n{formatted_args}"
                        # )
                    if human_check_before_calling and func_name not in [
                        "func_ls",
                        "func_cat",
                        "help_add_child",
                        "help_return_to_parent",
                        "add_child",
                        "return_to_parent",
                    ]:
                        human_response = self.ec.get_choice_response(
                            question=f"Whether call'{func_name}' with arguments:\n{formatted_args}",
                            options=["Yes", "No"],
                        )
                        if human_response == "Yes":
                            result = func_dict[func_name](**args)
                        else:
                            result = "Human prevents this operation"
                    else:
                        result = func_dict[func_name](**args)
                    if verbose:
                        self.ec.send_message(
                            {
                                "type": "info",
                                "data": {
                                    "category": "RESULT",
                                    "content": result,
                                    "level_delta": 0,
                                },
                            }
                        )
                        self.ec.send_message(
                            {
                                "type": "info",
                                "data": {
                                    "category": "",
                                    "content": "",
                                    "level_delta": -1,
                                },
                            }
                        )
                else:
                    result = f"[Error] Function {func_name} not found"

                new_tool_calls.append(
                    {
                        "role": "tool",
                        "name": func_name,
                        "tool_call_id": call.get("id", ""),
                        "content": str(result),
                    }
                )

            tc.extend(new_tool_calls)
            if verbose:
                self.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": "INFO",
                            "content": f"",
                            "level_delta": -1,
                        },
                    }
                )
            if tc.UpdataFunc is not None:
                tc.UpdataFunc()

            if stop_condition and stop_condition(tool_calls=tool_calls):
                if verbose:
                    print(f"\n=== Stopping early at chunk {step+1} ===\n")
                break

        # 输出结果：把所有step的文本连接
        final_text = "\n".join(all_texts)

        # if verbose:
        #     print("\n=== Summary DAG Final Answer ===\n")
        #     print(dag.answer("总结全部chunk内容"))
        #     print("\n=== Exported JSON Structure ===\n")
        #     print(dag.export_json())

        return final_text


def save_messages(messages: str, filepath: str):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def load_messages(filepath: str):
    filepath = Path(filepath)
    if not filepath.exists():
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_tool_calls(main_dir: str, exts=None):
    if exts is None:
        exts = [".py", ".m"]  # 可根据需要扩展

    main_path = Path(main_dir)
    all_calls = []

    # 根目录本身 → func_ls(".")
    all_calls.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": "func_ls",
                        "arguments": json.dumps({"filepath": "."}),
                    },
                }
            ],
        }
    )

    def recurse(path: Path):
        calls = []
        for child in path.iterdir():
            rel_path = str(child.relative_to(main_dir))
            if child.is_dir():
                # 目录 → func_ls
                calls.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": "func_ls",
                                    "arguments": json.dumps({"filepath": rel_path}),
                                },
                            }
                        ],
                    }
                )
                calls.extend(recurse(child))
            elif child.is_file() and child.suffix in exts:
                # 文件 → func_cat
                calls.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": "func_cat",
                                    "arguments": json.dumps({"filepath": rel_path}),
                                },
                            }
                        ],
                    }
                )
        return calls

    all_calls.extend(recurse(main_path))
    return all_calls


def generate_call_id():
    random_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("utf-8").rstrip("=")
    return f"call_00_{random_id[:22]}"


def log_webui(category: str, lines: list, ec: ExternalClient, start=True, end=True):
    if start:
        ec.send_message(
            {
                "type": "info",
                "data": {
                    "category": category,
                    "content": f"======== {category} ========",
                    "level_delta": 1,
                },
            }
        )
    for line in lines:
        ec.send_message(
            {
                "type": "info",
                "data": {"category": "", "content": line, "level_delta": 0},
            }
        )
    if end:
        ec.send_message(
            {
                "type": "info",
                "data": {
                    "category": "",
                    "content": "----------------------------------------",
                    "level_delta": -1,
                },
            }
        )
