import json
import os
from typing import List
import uuid
from schemas import AgentOutputItem, ToolUse, ReasoningStep

def convert_agent_outputs():
    # Get all json files in the results directory
    results_dir = "results"
    output_dir = "converted_agent_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]

    total_count = 0

    for json_file in json_files:
        file_path = os.path.join(results_dir, json_file)
        with open(file_path, "r") as f:
            results = json.load(f)

        converted_outputs = []

        for result in results:
            if result.get("success") and isinstance(result.get("result"), list):
                # Extract answer (result[0])
                answer = result["result"][0]

                # Extract tool_use_list (from result[1])
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

                # Create AgentOutputItem
                output_item = AgentOutputItem(
                    task_id=result["task_id"],
                    answer=answer,
                    tool_use_list=tool_use_list,
                    reasoning_list=[]
                )
                converted_outputs.append(output_item)

        # Save the converted results for each file to the new directory
        output_file = os.path.join(output_dir, f"converted_{json_file}")
        with open(output_file, "w") as f:
            json.dump([item.dict() for item in converted_outputs], f, indent=2, ensure_ascii=False)

        print(f"Converted {len(converted_outputs)} records and saved to {output_file}")
        total_count += len(converted_outputs)

    print(f"Total {total_count} records converted from all files and saved to {output_dir}.")

if __name__ == "__main__":
    convert_agent_outputs()
