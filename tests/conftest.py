"""pytest 配置：把项目根和 src/ 都加进 import 路径。

项目里混用了两种 import 风格：
    from src.retriever import ...   需要「项目根」
    from retriever import ...       需要「src/」
所以两个都要加，否则 pytest 收集用例时就报 No module named 'src'。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)
