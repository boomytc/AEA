#!/bin/bash

# 音频事件分析训练脚本
# 该脚本用于执行特征提取、标准化和模型训练的完整流程

# 显示彩色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 记录开始时间
start_time=$(date +%s)

# 创建必要的目录
mkdir -p models
mkdir -p datasets

# 检查数据列表文件是否存在
DATA_LIST="datasets/data_list.txt"
if [ ! -f "$DATA_LIST" ]; then
    echo -e "${RED}错误: 数据列表文件 $DATA_LIST 不存在!${NC}"
    echo "请创建数据列表文件，格式为: <音频文件路径> <标签>"
    exit 1
fi

# 显示帮助信息
show_help() {
    echo -e "${BLUE}音频事件分析训练脚本${NC}"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help        显示此帮助信息"
    echo "  -m, --model TYPE  指定要训练的模型类型: rf (随机森林) 或 xgb (XGBoost)"
    echo "  -f, --force       强制重新提取特征和训练模型，即使已存在"
    echo ""
    echo "示例:"
    echo "  $0 -m rf          训练随机森林模型"
    echo "  $0 -m xgb         训练XGBoost模型"
    echo "  $0 -m all         训练所有模型"
}

# 解析命令行参数
MODEL_TYPE="all"
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 验证模型类型
if [[ "$MODEL_TYPE" != "rf" && "$MODEL_TYPE" != "xgb" && "$MODEL_TYPE" != "all" ]]; then
    echo -e "${RED}错误: 无效的模型类型 '$MODEL_TYPE'${NC}"
    echo "有效的模型类型: rf, xgb, all"
    exit 1
fi

# 训练随机森林模型
train_rf() {
    echo -e "${YELLOW}开始训练随机森林模型...${NC}"
    python train_randomforest.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}随机森林模型训练完成!${NC}"
    else
        echo -e "${RED}随机森林模型训练失败!${NC}"
        exit 1
    fi
}

# 训练XGBoost模型
train_xgb() {
    echo -e "${YELLOW}开始训练XGBoost模型...${NC}"
    python train_xgboost.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}XGBoost模型训练完成!${NC}"
    else
        echo -e "${RED}XGBoost模型训练失败!${NC}"
        exit 1
    fi
}

# 根据选择的模型类型执行训练
case $MODEL_TYPE in
    "rf")
        train_rf
        ;;
    "xgb")
        train_xgb
        ;;
    "all")
        echo -e "${BLUE}将训练所有模型类型...${NC}"
        train_rf
        train_xgb
        ;;
esac

# 计算总耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo -e "${GREEN}训练完成!${NC}"
echo -e "总耗时: ${minutes}分${seconds}秒"
