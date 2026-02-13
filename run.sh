#!/bin/bash
# LogCortex V3 启动脚本

cd /home/qinzihang/Code/DeepFlight-Review

# 激活虚拟环境
source venv/bin/activate

# 设置 GLM API Key
export LOGCORTEX_API_KEY="b00e23d740524abba55a3072d10bda47.Mno0AlrtfrkG18I8"

# 清除代理（避免连接问题）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

echo "=========================================="
echo "  LogCortex V3 启动中..."
echo "=========================================="
echo ""
echo "  日志目录: ~/Code/FlightLog/"
echo "  API: GLM (已配置)"
echo ""
echo "  访问地址: http://localhost:8501"
echo ""
echo "=========================================="

# 启动 Streamlit
streamlit run app.py
