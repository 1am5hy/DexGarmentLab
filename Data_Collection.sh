#!/bin/bash

# 获取参数
task_name=$1
demo_num=$2
isaac_path=~/isaacsim_4.5.0/python.sh

export ISAAC_PATH=$isaac_path

# 创建目录和文件
base_dir="Data/${task_name}"
mkdir -p "${base_dir}/final_state_pic"
mkdir -p "${base_dir}/train_data"
mkdir -p "${base_dir}/vedio"
touch "${base_dir}/data_collection_log.txt"

# 获取当前数据数量
current_num=$(ls "${base_dir}/train_data" | wc -l)

# 进度条函数（写 stderr）
print_progress() {
    local current=$1
    local total=$2
    local task=$3
    local width=50
    local percent=$((100 * current / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))

    local bar=$(printf "%0.s█" $(seq 1 $filled))
    bar+=$(printf "%0.s " $(seq 1 $empty))

    # 输出任务名和进度条
    printf "\rTask: %-20s |%s| %3d%% (%d/%d)" "$task" "$bar" "$percent" "$current" "$total" >&2
}

# 数据采集循环
while [ "$current_num" -lt "$demo_num" ]; do

    # 打印进度条
    print_progress "$current_num" "$demo_num" "$task_name"

    # 执行 isaac 命令（stdout 保留）
    $ISAAC_PATH Env_StandAlone/${task_name}_Env.py \
        --env_random_flag True \
        --garment_random_flag True \
        --data_collection_flag True \
        > /dev/null 2>&1

    # 更新数量
    current_num=$(ls "${base_dir}/train_data" | wc -l)

    sleep 5
done

# 打印进度条
print_progress "$current_num" "$demo_num" "$task_name"

# 完成后换行
echo >&2
