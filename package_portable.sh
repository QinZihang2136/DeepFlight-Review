#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APP_DIR="$DIST_DIR/LogCortex-Portable"

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR"

cp -a "$ROOT_DIR/app.py" "$APP_DIR/"
cp -a "$ROOT_DIR/modules" "$APP_DIR/"
if [ -d "$ROOT_DIR/.streamlit" ]; then
  cp -a "$ROOT_DIR/.streamlit" "$APP_DIR/"
fi
cp -a "$ROOT_DIR/venv" "$APP_DIR/"

# 清理缓存，减小包体积
find "$APP_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + || true
find "$APP_DIR" -type f -name "*.pyc" -delete || true

cat > "$APP_DIR/run_logcortex.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$BASE_DIR/venv/bin/activate"

# 可选：从环境变量注入 API Key（推荐，不写进代码）
# export LOGCORTEX_API_KEY="your_api_key_here"

echo "Starting LogCortex..."
echo "Open in browser: http://localhost:8501"
exec streamlit run "$BASE_DIR/app.py" --server.address 127.0.0.1 --server.port 8501
RUN
chmod +x "$APP_DIR/run_logcortex.sh"

cat > "$APP_DIR/README_PORTABLE.md" <<'MD'
# LogCortex Portable

## 启动
```bash
./run_logcortex.sh
```

启动后在浏览器访问：
- http://localhost:8501

## API Key 注入（推荐）
不要把 Key 写进代码。可在启动前设置：
```bash
export LOGCORTEX_API_KEY="your_api_key"
./run_logcortex.sh
```

## 说明
- 此包包含独立 `venv`，无需重新安装依赖。
- 本包仅适用于与打包机器相同架构/系统（当前为 Linux x86_64）。
MD

mkdir -p "$DIST_DIR"
tar -czf "$DIST_DIR/LogCortex-Portable-linux-x86_64.tar.gz" -C "$DIST_DIR" LogCortex-Portable

echo "Portable package created: $DIST_DIR/LogCortex-Portable-linux-x86_64.tar.gz"
