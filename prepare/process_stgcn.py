import pandas as pd
import os


def split_data(input_file):
    """
    将CSV数据分割为温度、CO和SOOT三个文件
    Args:
        input_file: 输入文件名（用于创建保存目录）
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)

    # 获取不带后缀的文件名
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    # 在上一级目录的data文件夹下创建以base_name命名的文件夹
    save_dir = os.path.join(parent_dir, 'data', base_name)
    # 如果文件夹不存在就创建
    os.makedirs(save_dir, exist_ok=True)

    # 读取CSV文件，使用第二行作为列名
    df = pd.read_csv(input_file, skiprows=[0], header=0)  # 这里修改为使用input_file参数

    # 分离温度数据
    temp_cols = [col for col in df.columns if 'THCP' in col]
    temp_df = df[temp_cols]
    temp_df.to_csv(os.path.join(save_dir, 'temperature.csv'), index=False, header=False)

    # 分离CO数据
    co_cols = [col for col in df.columns if 'CO_' in col]
    co_df = df[co_cols]
    co_df = co_df * 1000000  # 转换为ppm
    co_df.to_csv(os.path.join(save_dir, 'co_ppm.csv'), index=False, header=False)

    # 分离Soot数据
    soot_cols = [col for col in df.columns if 'Soot_' in col]
    soot_df = df[soot_cols]
    soot_df = soot_df * 490.8  # 转换为μg/m³
    soot_df.to_csv(os.path.join(save_dir, 'soot_ugm3.csv'), index=False, header=False)

    print(f"数据已保存到 {save_dir} 文件夹中")
    return save_dir