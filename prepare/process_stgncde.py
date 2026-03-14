import numpy as np
import pandas as pd

def process_fire_data(csv_path):
    """
    处理火灾数据，提取THCP、CO、Soot特征
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path, skiprows=[0], header=0)  # 使用相同的读取方式

    # 使用与split_data相同的列名匹配模式
    thcp_cols = [col for col in df.columns if 'THCP' in col]
    co_cols = [col for col in df.columns if 'CO_' in col]
    soot_cols = [col for col in df.columns if 'Soot_' in col]

    # 在转换数组之前添加这些打印语句
    print(f"找到的THCP列数量: {len(thcp_cols)}")
    print(f"找到的CO列数量: {len(co_cols)}")
    print(f"找到的Soot列数量: {len(soot_cols)}")
    print("\n前几个列名示例:")
    print("THCP列:", thcp_cols[:3])
    print("CO列:", co_cols[:3])
    print("Soot列:", soot_cols[:3])

    # 转换为numpy数组
    thcp_data = df[thcp_cols].values
    co_data = df[co_cols].values * 1000000  # 转换为ppm，保持一致
    soot_data = df[soot_cols].values * 490.8  # 转换为μg/m³，保持一致

    # 组合成最终的形状 (时间步, 节点数, 3)
    time_steps = len(df)
    num_nodes = len(thcp_cols)
    final_data = np.zeros((time_steps, num_nodes, 3))

    final_data[:, :, 0] = thcp_data  # THCP特征
    final_data[:, :, 1] = co_data    # CO特征
    final_data[:, :, 2] = soot_data  # Soot特征

    # 保存为npz文件
    np.savez('fire_data.npz', data=final_data)

    print(f"数据形状: {final_data.shape}")
    print(f"时间步数: {time_steps}")
    print(f"节点数量: {num_nodes}")
    print(f"特征维度: 3 (THCP, CO, Soot)")

    print(final_data.shape)

    return final_data