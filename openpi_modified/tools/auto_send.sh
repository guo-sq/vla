#!/bin/bash

# --- 配置部分 ---
REMOTE_USER="root"
REMOTE_IP="39.96.194.254"
REMOTE_PORT="1314"
REMOTE_DEST="/mnt/shared/datasets/anyverse_human_data_record_raw/arxx5_bimanual/fold_box"

# 定义关联数组
# 格式: ["基础名字"]="起始编号 终止编号"
declare -A TASK_DICT
TASK_DICT=(
    ["record-arxx5_bimanual-fold-box-0129-"]="1 15"   # 从 1 传到 15
)

# --- 执行部分 ---

echo "🚀 开始执行多组自定义范围批量传输任务..."
echo "------------------------------------------"

for base_name in "${!TASK_DICT[@]}"; do
    # 解析起始和终止值
    read -r start_idx end_idx <<< "${TASK_DICT[$base_name]}"
    
    echo "📂 处理任务组: $base_name (范围: $start_idx 到 $end_idx)"
    
    # 使用 seq 生成自定义范围序列
    for i in $(seq "$start_idx" "$end_idx"); do
        folder_name="${base_name}${i}"
        local_path="./heyuan1993/${folder_name}"

        if [ -d "$local_path/images" ]; then
            echo "✅ 目录存在: $local_path/images, 马上删除"
            # 删除 images 目录
            rm -rf "$local_path/images"
        fi

        if [ ! -d "$local_path" ]; then
            echo "⚠️  跳过: '$local_path' 不存在"
            continue
        fi

        echo "正在传输: $folder_name ..."
        
        # 执行 scp
        scp -P "$REMOTE_PORT" -r "$local_path" "$REMOTE_USER@$REMOTE_IP:$REMOTE_DEST"

        if [ $? -eq 0 ]; then
            echo "✅ '$folder_name' 成功"
        else
            echo "❌ '$folder_name' 失败"
        fi
    done
    echo "------------------------------------------"
done

echo "🎉 所有指定范围的任务已处理完毕。"