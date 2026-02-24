# -*- coding: utf-8 -*-
"""
nsfw-checker-pro — Multi-Engine NSFW Analyzer
Entry point for the application.
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Main entry point."""
    from gui.app import launch
    launch()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
