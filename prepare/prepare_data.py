
from IFCGraphProcessor import Processor
from process_stgcn import split_data
from process_stgncde import process_fire_data
import numpy as np
input_file = "full_model_10_devc.csv"
split_data(input_file)
# process_fire_data(input_file)

def prepare():
    # 文件路径
    input_ifc_path = "Institute.ifc"
    ontology_output_path = "evacuation_ontology.owl"


    # 将device分成三个文件

    try:
        # 2. 使用Processor处理修改后的IFC文件创建空间图
        print("\n=== 创建空间图 ===")
        processor = Processor(input_ifc_path)  # 使用修改后的IFC文件
        processor.process_and_build_topology()
        processor.build_distance_matrix()

        processor.build_final_adjacency_matrix()

        # processor.export_for_origin()
        # 添加可视化调用
        # 1. 折线图展示
        # processor.visualize_weight_changes_line()
        #
        # # 2. 热力图展示
        # processor.visualize_weight_changes_heatmap()
        #

        processor.apply_gaussian_threshold()
        processor.network_visualization()
        # processor.matrix_visualization()
        # 验证空间图是否正确创建
        print(f"空间图节点数: {len(processor.G.nodes())}")
        print(f"空间图边数: {len(processor.G.edges())}")

        print(f"space节点数: {sum(1 for node in processor.G.nodes() if processor.G.nodes[node]['type'] == 'space')}")
        print(f"door节点数: {sum(1 for node in processor.G.nodes() if processor.G.nodes[node]['type'] == 'door')}")
        print(f"stair节点数: {sum(1 for node in processor.G.nodes() if processor.G.nodes[node]['type'] == 'stair')}")
        # 调用方法生成配置文件
        processor.generate_fire_sources("fire_sources.txt")
        if len(processor.G.nodes()) == 0:
            print("错误: 空间图创建失败")
            return
        # true_values = np.load('true_values.npy')
        # processor.visualize_temperature_with_slider(true_values, "Temperature Distribution")

    except Exception as e:
        print(f"主程序发生错误: {str(e)}")
        raise


# prepare()