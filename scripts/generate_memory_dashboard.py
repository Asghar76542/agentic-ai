#!/usr/bin/env python3
"""Generate memory analytics dashboard for AgenticSeek."""

import argparse
from pathlib import Path
import webbrowser

from sources.knowledge.memoryIntegration import MemorySystemConfig, EnhancedMemorySystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an interactive memory dashboard")
    parser.add_argument("-o", "--output", default="./data/dashboard", help="Directory where dashboard files are saved")
    parser.add_argument("--open", action="store_true", help="Open the generated dashboard in your browser")
    args = parser.parse_args()

    config = MemorySystemConfig(dashboard_path=args.output, enable_background_processing=False)
    memory_system = EnhancedMemorySystem(config=config)

    dashboard_path = memory_system.generate_dashboard()
    memory_system.shutdown()

    if dashboard_path:
        print(f"Dashboard created at {dashboard_path}")
        if args.open:
            webbrowser.open(f"file://{Path(dashboard_path).resolve()}")
    else:
        print("Failed to generate dashboard")


if __name__ == "__main__":
    main()
