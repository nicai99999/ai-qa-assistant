#!/bin/sh
set -e

# 首次启动（volume 为空）时自动建索引
if [ ! -f /app/chroma_db/chroma.sqlite3 ]; then
    echo ">>> 未发现索引，开始构建（首次启动会慢，要调 embedding 接口）..."
    python src/reindex.py
    echo ">>> 索引构建完成"
else
    echo ">>> 已发现现有索引，跳过构建"
fi

# 启动 Streamlit
# --server.address=0.0.0.0 是必须的，否则容器外访问不到
exec streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true