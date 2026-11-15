# ============ 基础镜像 ============
# slim 版比完整版小很多；不要用 alpine（编译依赖缺，装包容易失败）
FROM python:3.12-slim

# ============ 环境变量 ============
# PYTHONUNBUFFERED      日志立即输出，不被缓冲
# PYTHONDONTWRITEBYTECODE 不生成 .pyc
# TZ                    容器时区
# UV_PROJECT_ENVIRONMENT 让 uv 装到 /opt/venv，而不是 /app/.venv
# PYTHONPATH            两个路径都要：/app 给 from src.xxx，/app/src 给 from xxx
#
# 注意：注释都写在 ENV 之前，续行里不能夹空行和注释
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONPATH=/app:/app/src

# ============ 装 uv ============
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# ============ 依赖层（利用缓存）============
# 先只拷依赖声明：代码改了不会让依赖层重新构建
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ============ 代码层 ============
COPY src ./src
COPY app.py ./
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# ============ 数据层 ============
# 语料打进镜像（体积小）；chroma_db 用 volume 挂载，不打进镜像
COPY data ./data

# ============ 启动 ============
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8501

ENTRYPOINT ["./docker-entrypoint.sh"]