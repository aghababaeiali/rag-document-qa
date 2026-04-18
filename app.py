# app.py — HuggingFace Spaces entry point
# This simply imports and launches the Gradio demo from demo.py

from demo import demo

if __name__ == "__main__":
    demo.launch()