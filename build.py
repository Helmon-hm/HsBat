"""
HsBat - PyInstaller 打包脚本
运行: python build.py [--debug]
"""

import os
import sys

try:
    import PyInstaller.__main__
except ImportError:
    print("请先安装 pyinstaller: pip install pyinstaller")
    sys.exit(1)


def build():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    debug_mode = "--debug" in sys.argv

    args = [
        "--name=HsBat",
        "--onefile",
        "--noconfirm",
        f"--distpath={os.path.join(root_dir, 'dist')}",
        f"--workpath={os.path.join(root_dir, 'build_temp')}",
        f"--specpath={root_dir}",
        "--add-data", f"config.yaml{os.pathsep}.",
        "--add-data", f"templates{os.pathsep}templates",
        "--hidden-import=src.state_recognizer",
        "--hidden-import=src.decision_maker",
        "--hidden-import=src.action_executor",
        "--hidden-import=src.game_controller",
        "--hidden-import=src.logger",
        "--hidden-import=src.gui_app",
        "--hidden-import=src.paths",
        "--hidden-import=pytesseract",
        "--hidden-import=pyautogui",
        "--hidden-import=pystray",
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=tkinter.ttk",
        "--hidden-import=tkinter.scrolledtext",
        "--hidden-import=tkinter.messagebox",
        "--hidden-import=tkinter.filedialog",
        "--exclude-module=torch",
        "--exclude-module=pandas",
        "--exclude-module=matplotlib",
        "--exclude-module=scipy",
        "--exclude-module=sklearn",
        "--exclude-module=openpyxl",
        "--exclude-module=xlrd",
        "--exclude-module=sqlalchemy",
        "--exclude-module=pygments",
        "--exclude-module=boto3",
        "--exclude-module=botocore",
        "--exclude-module=zmq",
        "--exclude-module=notebook",
        "--exclude-module=jupyter",
        "--exclude-module=IPython",
        "--exclude-module=prometheus",
        "--exclude-module=flask",
        "--exclude-module=django",
        "--exclude-module=plotly",
        "--exclude-module=tensorflow",
        "--exclude-module=Qt",
        "--exclude-module=sympy",
        "--exclude-module=stack_data",
    ]

    if not debug_mode:
        args.append("--windowed")

    args.append(os.path.join(root_dir, "main.py"))

    print("=" * 60)
    print("HsBat - 正在打包为 EXE...")
    print("=" * 60)
    print(f"模式: {'调试 (带控制台)' if debug_mode else '发布 (无控制台)'}")
    print("=" * 60)

    PyInstaller.__main__.run(args)

    exe_path = os.path.join(root_dir, "dist", "HsBat.exe")
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / 1024 / 1024
        print("=" * 60)
        print(f"打包成功!")
        print(f"EXE 路径: {exe_path}")
        print(f"文件大小: {size_mb:.1f} MB")
        print("=" * 60)
    else:
        print("=" * 60)
        print("打包失败! 请检查上方错误信息。")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    build()
