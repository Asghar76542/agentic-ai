"""Generic tool base class used by various agents."""

from __future__ import annotations

import os
import configparser
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

from sources.logger import Logger


class Tools(ABC):
    """Abstract base class for all tools."""

    def __init__(self) -> None:
        self.tag: str = "undefined"
        self.name: str = "undefined"
        self.description: str = "undefined"
        self.client = None
        self.messages: List[str] = []
        self.logger = Logger("tools.log")
        self.config = configparser.ConfigParser()
        self.work_dir = self.create_work_dir()
        self._executable_blocks_found = False
        self.safe_mode = True
        self.allow_language_exec_bash = False

    def get_work_dir(self) -> str:
        return self.work_dir

    def set_allow_language_exec_bash(self, value: bool) -> None:
        self.allow_language_exec_bash = value

    def config_exists(self) -> bool:
        return os.path.exists("./config.ini")

    def safe_get_work_dir_path(self) -> str:
        """Return the work directory path from env or config or default."""
        path = os.getenv("WORK_DIR")
        if not path and self.config_exists():
            self.config.read("./config.ini")
            path = self.config.get("MAIN", "work_dir", fallback="")
        if not path:
            print("No work directory specified, using default.")
            path = os.path.dirname(os.getcwd())
        return path

    def create_work_dir(self) -> str:
        path = self.safe_get_work_dir_path()
        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Abstract methods implemented by concrete tools
    @abstractmethod
    def execute(self, blocks: List[str], safety: bool) -> str:
        pass

    @abstractmethod
    def execution_failure_check(self, output: str) -> bool:
        pass

    @abstractmethod
    def interpreter_feedback(self, output: str) -> str:
        pass

    # ------------------------------------------------------------------
    def save_block(self, blocks: List[str], save_path: str) -> None:
        if save_path is None:
            return
        self.logger.info(f"Saving blocks to {save_path}")
        directory = os.path.join(self.work_dir, os.path.dirname(save_path))
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, os.path.basename(save_path))
        with open(file_path, "w") as f:
            for block in blocks:
                f.write(block)

    def get_parameter_value(self, block: str, parameter_name: str) -> Optional[str]:
        """Return the value of ``parameter_name`` within ``block``.

        Parameters are expected in the form ``<name> = <value>`` but extra
        whitespace is tolerated.
        """
        prefix = parameter_name.strip()
        for line in block.splitlines():
            line = line.strip()
            if not line or not line.startswith(prefix):
                continue
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == prefix:
                return value.strip()
        return None

    def found_executable_blocks(self) -> bool:
        result = self._executable_blocks_found
        self._executable_blocks_found = False
        return result

    # ------------------------------------------------------------------
    def load_exec_block(
        self, llm_text: str
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        """Extract executable code blocks from LLM text."""
        assert self.tag != "undefined", "Tag not defined"
        start_tag = f"```{self.tag}"
        end_tag = "```"
        if start_tag not in llm_text:
            return None, None

        blocks: List[str] = []
        save_path: Optional[str] = None
        index = 0
        while True:
            start_pos = llm_text.find(start_tag, index)
            if start_pos == -1:
                break
            end_pos = llm_text.find(end_tag, start_pos + len(start_tag))
            if end_pos == -1:
                # incomplete block
                return [], None
            content = llm_text[start_pos + len(start_tag) : end_pos]
            content = self._dedent_block(content)
            if not content.startswith("\n"):
                content = "\n" + content
            if not content.endswith("\n"):
                content = content + "\n"
            blocks.append(content)
            self._executable_blocks_found = True
            index = end_pos + len(end_tag)
        self.logger.info(f"Found {len(blocks)} blocks to execute")
        return blocks, save_path

    def _dedent_block(self, block: str) -> str:
        """Remove common leading whitespace from block."""
        lines = block.splitlines()
        if not lines:
            return ""
        # Determine minimum indentation
        indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        indent = min(indents) if indents else 0
        dedented = "\n".join(
            line[indent:] if len(line) >= indent else line for line in lines
        )
        return dedented


if __name__ == "__main__":
    # Simple manual test
    t = Tools()  # type: ignore  # abstract base instantiation for manual run
    t.tag = "python"
    code, _ = t.load_exec_block("""```python\nprint('hello')\n```""")
    print(code)
