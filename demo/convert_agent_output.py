import json
import os
from typing import List
import uuid
from schemas import AgentOutputItem, ToolUse, ReasoningStep

def convert_agent_outputs():
    # 获取results目录下所有json文件
    results_dir = "results"
    json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    
    converted_outputs = []
    
    for json_file in json_files:
        file_path = os.path.join(results_dir, json_file)
        with open(file_path, "r") as f:
            results = json.load(f)
            
        for result in results:
            if result["success"] and isinstance(result["result"], list):
                # 提取answer (result[0])
                answer = result["result"][0]
                
                # 提取tool_use_list (从result[1]中获取)
                tool_use_list = []
                if len(result["result"]) > 1 and isinstance(result["result"][1], dict):
                    session_data = result["result"][1]
                    for turn in session_data.get("turns", []):
                        for tool_call in turn.get("tool_calls", []):
                            tool_use = ToolUse(
                                call_id=str(uuid.uuid4()),
                                tool_name=tool_call["tool_name"],
                                tool_description=tool_call["tool_description"],
                                tool_output=tool_call["tool_output"],
                                tool_input=str(tool_call["arguments"])
                            )
                            tool_use_list.append(tool_use)
                
                # 创建AgentOutputItem
                output_item = AgentOutputItem(
                    task_id=result["question"],  # 使用question作为task_id
                    answer=answer,
                    tool_use_list=tool_use_list,
                    reasoning_list=[]  # 空列表
                )
                converted_outputs.append(output_item)
    
    # 保存转换后的结果
    output_file = "converted_agent_outputs.json"
    with open(output_file, "w") as f:
        json.dump([item.dict() for item in converted_outputs], f, indent=2, ensure_ascii=False)
    
    print(f"已转换 {len(converted_outputs)} 条记录并保存到 {output_file}")

if __name__ == "__main__":
    convert_agent_outputs()
