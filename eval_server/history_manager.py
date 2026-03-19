"""
History Manager for Multi-Step Reasoning

核心功能:
- 维护确定性的累积history,避免模型幻觉
- 从模型输出的完整history中提取新步骤(最后一行)
- 提供格式化的history字符串用于prompt构建
"""

import re
from typing import List, Optional


class HistoryManager:
    """
    外挂History管理器

    关键机制:
    1. 框架维护确定性的accumulated_history
    2. 从模型输出中提取最后一行(新的第n步)
    3. 累积更新: 将新步骤添加到accumulated_history
    4. 下一步使用框架维护的history,确保确定性
    """

    def __init__(self):
        """初始化History管理器"""
        self.accumulated_history: List[str] = []
        self.step_num = 0

    def initialize_with_task_analysis(self):
        """
        初始化step 0的固定内容

        训练数据中,step 0的History固定为:
        "0) I have already received the detailed task instruction and am ready to start."
        """
        self.step_num = 0
        self.accumulated_history = [
            "0) I have already received the detailed task instruction and am ready to start."
        ]

    def extract_new_step_from_gpt_output(self, gpt_history: str) -> Optional[str]:
        """
        从GPT输出的完整history中提取最后一行(新步骤)

        Args:
            gpt_history: 模型输出的完整History字符串,格式如:
                "1) I have completed step 1.\n2) I have completed step 2."

        Returns:
            提取的新步骤描述(不包含序号),如: "I have completed step 2."
            如果提取失败,返回原始最后一行

        示例:
            输入: "1) Task analysis done.\n2) Navigated to desk."
            输出: "Navigated to desk."
        """
        lines = gpt_history.strip().split('\n')
        if not lines:
            return None

        last_line = lines[-1].strip()

        # 尝试匹配 "序号) 描述" 的格式
        match = re.match(r'^\d+\)\s*(.+)$', last_line)
        if match:
            return match.group(1).strip()

        # 如果无法匹配,返回原始最后一行
        return last_line

    def add_step(self, description: str):
        """
        添加新步骤到累积history

        Args:
            description: 新步骤的描述(不包含序号)

        示例:
            输入: "Successfully picked up the clock."
            累积后: "3) Successfully picked up the clock."
        """
        self.step_num += 1
        formatted_step = f"{self.step_num}) {description.strip()}"
        self.accumulated_history.append(formatted_step)

    def get_formatted_history(self) -> str:
        """
        获取格式化的history字符串,用于构建prompt

        Returns:
            格式化的history字符串,每步一行

        示例:
            "0) I have already received the detailed task instruction and am ready to start.\n1) Navigated to desk_1.\n2) Picked up the clock."
        """
        return '\n'.join(self.accumulated_history)

    def get_current_step_num(self) -> int:
        """获取当前步骤编号"""
        return self.step_num

    def get_history_list(self) -> List[str]:
        """获取累积的history列表"""
        return self.accumulated_history.copy()

    def __repr__(self):
        return f"HistoryManager(step_num={self.step_num}, history_length={len(self.accumulated_history)})"
