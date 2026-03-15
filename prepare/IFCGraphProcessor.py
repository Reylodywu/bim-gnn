import networkx as nx
import ifcopenshell
import ifcopenshell.geom
import matplotlib.pyplot as plt
import numpy as np
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
from OCC.Core.BRepGProp import brepgprop
import logging
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopoDS import topods
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.AIS import AIS_Shape
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.SimpleGui import init_display
import seaborn as sns
import scipy.sparse as sp
import os
from OCC.Display.backend import load_backend
import pandas as pd
from matplotlib.widgets import Slider



class Processor:
    def __init__(self, ifc_file_path):
        self.ifc_file_path = ifc_file_path
        self.ifc_file = ifcopenshell.open(ifc_file_path)
        self.settings = ifcopenshell.geom.settings()
        self.settings.set(self.settings.USE_PYTHON_OPENCASCADE, True)
        self.settings.set(self.settings.USE_WORLD_COORDS, True)
        self.G = nx.Graph()
        self.intersection_points = []
        self.fire_index = 1
        self.opened_windows = {}
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        self.display = None
        self.start_display = None

    def optimize_graph(self, threshold=None):
        #threshold 代表两边之和与直接距离的比值，如果过小说明三点共线
        def distance(p1, p2):
            # 确保我们使用的是坐标而不是节点索引
            if isinstance(p1, tuple) and isinstance(p2, tuple):
                return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
            else:
                # 如果输入不是坐标，则获取节点的坐标
                pos1 = self.G.nodes[p1]['pos'] if 'pos' in self.G.nodes[p1] else p1
                pos2 = self.G.nodes[p2]['pos'] if 'pos' in self.G.nodes[p2] else p2
                return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

        nodes = list(self.G.nodes())
        adj_list = {i: set() for i in range(len(nodes))}

        # 构建邻接表时保存实际的节点标识符
        for edge in self.G.edges():
            n1, n2 = nodes.index(edge[0]), nodes.index(edge[1])
            adj_list[n1].add(n2)
            adj_list[n2].add(n1)

        edge_weights = {}
        for edge in self.G.edges():
            n1, n2 = nodes.index(edge[0]), nodes.index(edge[1])
            weight = distance(edge[0], edge[1])  # 使用实际节点而不是索引
            edge_weights[(n1, n2)] = weight
            edge_weights[(n2, n1)] = weight

        edges_to_remove = set()

        # print("Analyzing paths...")
        for node in range(len(nodes)):
            neighbors = adj_list[node]
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 >= n2:
                        continue

                    # 使用实际节点计算距离
                    direct_path = distance(nodes[n1], nodes[n2])
                    indirect_path = edge_weights[(node, n1)] + edge_weights[(node, n2)]

                    # print(f"Checking path: {nodes[n1]} -> {nodes[node]} -> {nodes[n2]}")
                    # print(
                    #     f"Direct: {direct_path:.2f}, Indirect: {indirect_path:.2f}, Diff: {abs(indirect_path - direct_path):.2f}")

                    if indirect_path < threshold * direct_path:
                        # 保存实际的节点对而不是索引
                        edge_to_remove = tuple(sorted([nodes[n1], nodes[n2]]))
                        edges_to_remove.add(edge_to_remove)
                        # print(f"Marking edge {edge_to_remove} for removal")

        for edge in edges_to_remove:
            if self.G.has_edge(*edge):
                # print(f"Removing edge {edge}")
                self.G.remove_edge(*edge)
            # else:
            #     print(f"Edge {edge} not found in graph")

    def get_ifc_spaces_on_specified_floor(self, storey_name):
        ifc_spaces = []
        ifc_building_storeys = self.ifc_file.by_type('IfcBuildingStorey')

        for storey in ifc_building_storeys:
            if storey.Name == storey_name:
                # 检查 IsDecomposedBy 是否存在且不为空
                if hasattr(storey, 'IsDecomposedBy') and storey.IsDecomposedBy:
                    for decomposition in storey.IsDecomposedBy:
                        if decomposition.RelatedObjects:
                            for obj in decomposition.RelatedObjects:
                                if obj.is_a('IfcSpace'):
                                    ifc_spaces.append(obj)

                # 如果没有找到空间，尝试使用 ContainsElements 关系
                if not ifc_spaces:
                    if hasattr(storey, 'ContainsElements') and storey.ContainsElements:
                        for rel in storey.ContainsElements:
                            for obj in rel.RelatedElements:
                                if obj.is_a('IfcSpace'):
                                    ifc_spaces.append(obj)

        if not ifc_spaces:
            print(f"Warning: No spaces found on floor '{storey_name}'")

        return ifc_spaces

    def get_furniture_dimensions(self, element):
        """
        通过boundingbox获取家具的尺寸信息（长宽高和体积）
        """
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            bbox = Bnd_Box()
            brepbndlib.Add(shape.geometry, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

            # 计算尺寸
            length = abs(xmax - xmin)  # X方向长度
            width = abs(ymax - ymin)  # Y方向宽度
            height = abs(zmax - zmin)  # Z方向高度

            # 计算面积和体积
            area = length * width
            volume = area * height

            return area, volume, length, width, height
        except Exception as e:
            print(f"计算家具 {element.GlobalId} 尺寸时出错: {str(e)}")
            return None, None, None, None, None

    def calculate_total_furniture_metrics(self, furniture_list):
        """
        计算一个空间内所有家具的总面积和总体积
        """
        total_area = 0
        total_volume = 0
        valid_furniture_count = 0

        for furniture in furniture_list:
            if furniture[1] and furniture[2]:  # area和volume都不为None
                total_area += furniture[1]
                total_volume += furniture[2]
                valid_furniture_count += 1

        return total_area, total_volume, valid_furniture_count

    def get_furnishing_elements(self, space):
        """获取指定IfcSpace中的所有IfcFurnishingElement"""
        furnishing_elements = []
        if space.ContainsElements:
            for rel in space.ContainsElements:
                for element in rel.RelatedElements:
                    if element.is_a('IfcFurnishingElement'):
                        # 获取尺寸信息
                        area, volume, length, width, height = self.get_furniture_dimensions(element)
                        name = element.Name if element.Name else "未命名家具"
                        type_info = element.ObjectType if hasattr(element, 'ObjectType') else "未知类型"

                        furniture_info = (name, area, volume, length, width, height, type_info)
                        furnishing_elements.append(furniture_info)
        return furnishing_elements

    def process_spaces_and_furniture(self, storey_name):
        """处理指定楼层的空间和家具"""
        spaces = self.get_ifc_spaces_on_specified_floor(storey_name)
        result = {}

        for space_obj in spaces:
            # 获取空间名称
            space_name = space_obj.Name if hasattr(space_obj, 'Name') and space_obj.Name else space_obj.GlobalId

            # 获取家具列表
            furniture_list = self.get_furnishing_elements(space_obj)
            total_area, total_volume, furniture_count = self.calculate_total_furniture_metrics(furniture_list)

            # print(f"空间: {space_name}")
            # if furniture_list:
            #     print(f"家具总数: {furniture_count}")
            #     print(f"家具总占地面积: {total_area:.2f} 平方米")
            #     print(f"家具总体积: {total_volume:.2f} 立方米")
            #     print("包含的家具:")
            #     for furniture in furniture_list:
            #         print(f"  - 名称: {furniture[0]}")  # name
            #         print(f"    类型: {furniture[6]}")  # type_info
            #         if furniture[1] and furniture[2]:  # area和volume不为None
            #             print(f"    占地面积: {furniture[1]:.2f} 平方米")
            #             print(f"    体积: {furniture[2]:.2f} 立方米")
            #             print(
            #                 f"    尺寸: {furniture[3]:.2f}m x {furniture[4]:.2f}m x {furniture[5]:.2f}m")  # length x width x height
            # else:
            #     print("  该空间没有家具")

            result[space_name] = {
                'furniture_list': furniture_list,
                'total_area': total_area,
                'total_volume': total_volume,
                'furniture_count': furniture_count
            }

        return result

    def get_doors_on_specified_floor(self, storey_name):
        doors = []
        for door in self.ifc_file.by_type('IfcDoor'):
            decomposed_by = door.ContainedInStructure
            if decomposed_by:
                for rel in decomposed_by:
                    if hasattr(rel.RelatingStructure, 'Name') and rel.RelatingStructure.Name == storey_name:
                        doors.append(door)
                        break
        return doors

    def get_stairs_on_specified_floor(self, storey_name):
        """获取指定楼层的楼梯"""
        stairs = []
        for stair in self.ifc_file.by_type('IfcStair'):
            decomposed_by = stair.ContainedInStructure
            if decomposed_by:
                for rel in decomposed_by:
                    if hasattr(rel.RelatingStructure, 'Name') and rel.RelatingStructure.Name == storey_name:
                        stairs.append(stair)
                        break
        return stairs

    def get_walls_on_specified_floor(self, storey_name):
        walls = []
        for wall in self.ifc_file.by_type('IfcWall'):
            decomposed_by = wall.ContainedInStructure
            if decomposed_by:
                for rel in decomposed_by:
                    if hasattr(rel.RelatingStructure, 'Name') and rel.RelatingStructure.Name == storey_name:
                        walls.append(wall)
                        break
        return walls

    def get_center(self, ifc_element):
        try:
            # 首先尝试使用几何表示获取中心点
            try:
                shape = ifcopenshell.geom.create_shape(self.settings, ifc_element).geometry
                exp_face = TopExp_Explorer(shape, TopAbs_FACE)
                min_x, min_y = float('inf'), float('inf')
                max_x, max_y = float('-inf'), float('-inf')

                while exp_face.More():
                    face = topods.Face(exp_face.Current())
                    exp_edge = TopExp_Explorer(face, TopAbs_EDGE)
                    while exp_edge.More():
                        edge = topods.Edge(exp_edge.Current())
                        exp_vertex = TopExp_Explorer(edge, TopAbs_VERTEX)
                        while exp_vertex.More():
                            vertex = topods.Vertex(exp_vertex.Current())
                            point = BRep_Tool.Pnt(vertex)
                            min_x = min(min_x, point.X())
                            min_y = min(min_y, point.Y())
                            max_x = max(max_x, point.X())
                            max_y = max(max_y, point.Y())
                            exp_vertex.Next()
                        exp_edge.Next()
                    exp_face.Next()

                if min_x < max_x and min_y < max_y:
                    # print(self.current_elevation)
                    return ((min_x + max_x) / 2, (min_y + max_y) / 2, self.current_elevation)
            except:
                # 如果几何表示获取失败，使用ObjectPlacement
                if hasattr(ifc_element, 'ObjectPlacement'):
                    location = self.get_placement_location(ifc_element)
                    if location:
                        return location
                return None

        except Exception as e:
            print(f"Error getting center for element {ifc_element.GlobalId}: {str(e)}")
            return None

    def get_placement_location(self, element):
        """从ObjectPlacement获取位置信息"""
        try:
            if element.ObjectPlacement:
                placement = element.ObjectPlacement

                # 处理IfcLocalPlacement
                if placement.is_a('IfcLocalPlacement'):
                    relative_placement = placement.RelativePlacement
                    if relative_placement:
                        location = relative_placement.Location
                        if location:
                            # 获取坐标
                            x = location.Coordinates[0]
                            y = location.Coordinates[1]
                            z = location.Coordinates[2] if len(location.Coordinates) > 2 else self.current_elevation
                            return (x, y, z)

                # 如果有RelativeTo属性，递归获取相对位置
                if hasattr(placement, 'RelativeTo') and placement.RelativeTo:
                    parent_location = self.get_placement_location(placement.RelativeTo)
                    if parent_location:
                        # 合并相对位置
                        relative_location = self.get_placement_location(placement)
                        if relative_location:
                            return (
                                parent_location[0] + relative_location[0],
                                parent_location[1] + relative_location[1],
                                parent_location[2] + relative_location[2]
                            )

        except Exception as e:
            print(f"Error getting placement location: {str(e)}")
        return None

    def line_intersects_wall(self, line_start, line_end, wall_shape):
        TOLERANCE = 1e-7  # 可以适当调大，比如 1e-5
        start_point = gp_Pnt(*line_start)
        end_point = gp_Pnt(*line_end)
        direction = gp_Vec(start_point, end_point)
        direction_magnitude = direction.Magnitude()

        if direction_magnitude < TOLERANCE:
            return False

        valid_faces = []
        explorer = TopExp_Explorer(wall_shape, TopAbs_FACE)

        while explorer.More():
            face = topods.Face(explorer.Current())
            face_surface = BRepAdaptor_Surface(face)

            if face_surface.GetType() == GeomAbs_Plane:
                plane = face_surface.Plane()
                normal = plane.Axis().Direction()

                # 调整法向量的判断阈值，使其能捕获到更多的垂直面
                if abs(normal.Z()) < 0.1:  # 原来是 TOLERANCE
                    props = GProp_GProps()
                    brepgprop.SurfaceProperties(face, props)
                    area = props.Mass()

                    face_bbox = Bnd_Box()
                    brepbndlib.Add(face, face_bbox)
                    xmin, ymin, zmin, xmax, ymax, zmax = face_bbox.Get()

                    height = zmax - zmin
                    width = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5

                    wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
                    wire_count = 0
                    while wire_explorer.More():
                        wire_count += 1
                        wire_explorer.Next()

                    valid_faces.append({
                        'face': face,
                        'area': area,
                        'height': height,
                        'width': width,
                        'bounds': (xmin, ymin, zmin, xmax, ymax, zmax),
                        'normal': normal,
                        'plane': plane,
                        'wire_count': wire_count
                    })

            explorer.Next()

        if not valid_faces:
            return False

        valid_faces.sort(key=lambda x: x['area'], reverse=True)
        max_area = valid_faces[0]['area']

        for face_data in valid_faces:
            # 调整面积筛选的阈值，降低以包含更多小面
            if face_data['area'] < max_area * 0.005:  # 原来是 0.1
                continue

            # 调整带孔面的面积阈值
            # if face_data['wire_count'] > 1 and face_data['area'] < max_area * 0.3:  # 原来是 0.5
            #     continue

            if face_data['wire_count'] > 1:  # 原来是 0.5
                continue

            normal = face_data['normal']
            plane = face_data['plane']
            xmin, ymin, zmin, xmax, ymax, zmax = face_data['bounds']

            normal_vec = gp_Vec(normal.X(), normal.Y(), normal.Z())
            Q = plane.Location()
            denominator = direction.Dot(normal_vec)

            # 调整平行判断的阈值
            if abs(denominator) < 0.01:  # 原来是 TOLERANCE
                continue

            QP0 = gp_Vec(Q.X() - start_point.X(),
                         Q.Y() - start_point.Y(),
                         Q.Z() - start_point.Z())

            t = QP0.Dot(normal_vec) / denominator

            if 0 <= t <= 1:
                intersection_x = start_point.X() + t * direction.X()
                intersection_y = start_point.Y() + t * direction.Y()
                intersection_z = start_point.Z() + t * direction.Z()

                # 调整边界检查的容差
                boundary_tolerance = 1e-5  # 可以适当调大
                if (xmin - boundary_tolerance <= intersection_x <= xmax + boundary_tolerance and
                        ymin - boundary_tolerance <= intersection_y <= ymax + boundary_tolerance and
                        zmin - boundary_tolerance <= intersection_z <= zmax + boundary_tolerance):
                    # if (xmin - boundary_tolerance <= intersection_x <= xmax + boundary_tolerance and
                    #         ymin - boundary_tolerance <= intersection_y <= ymax + boundary_tolerance):

                    point_on_face = gp_Pnt(intersection_x, intersection_y, intersection_z)
                    projector = BRepExtrema_DistShapeShape(
                        BRepBuilderAPI_MakeVertex(point_on_face).Vertex(),
                        face_data['face']
                    )

                    # 调整投影点判断的容差
                    if projector.Value() <= 0.1:  # 原来是 TOLERANCE
                        intersection_coords = (intersection_x, intersection_y, intersection_z)
                        # self.G.add_node(intersection_coords, type='intersection')
                        self.intersection_points.append(intersection_coords)
                        return True

        return False

    def get_space_bbox(self, space):
        """
        获取空间的边界框
        返回格式: [(xmin, ymin, zmin), (xmax, ymax, zmax)]
        """
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, space)
            bbox = Bnd_Box()
            brepbndlib.Add(shape.geometry, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

            return [(xmin, ymin, zmin), (xmax, ymax, zmax)]
        except Exception as e:
            print(f"计算空间 {space.GlobalId} 边界框时出错: {str(e)}")
            return None

    def get_space_doors(self, ifc_space):
        """
        获取与给定空间相连的所有门
        通过RelSpaceBoundary或空间边界关系获取相关的门
        """
        connected_doors = []

        # 方法1：通过RelSpaceBoundary获取
        for rel in self.ifc_file.by_type('IfcRelSpaceBoundary'):
            if rel.RelatingSpace == ifc_space:
                element = rel.RelatedBuildingElement
                if element and element.is_a('IfcDoor'):
                    connected_doors.append(element)

        # # 方法2：通过IfcRelContainedInSpatialStructure获取
        # if not connected_doors:
        #     for rel in self.ifc_file.by_type('IfcRelContainedInSpatialStructure'):
        #         if rel.RelatingStructure == ifc_space:
        #             for element in rel.RelatedElements:
        #                 if element.is_a('IfcDoor'):
        #                     connected_doors.append(element)
        #
        # # 方法3：通过ProvidesBoundaries关系获取
        # if not connected_doors:
        #     for rel in self.ifc_file.by_type('IfcRelSpaceProvidesBoundaries'):
        #         if rel.RelatingSpace == ifc_space:
        #             boundary_element = rel.RelatedBuildingElement
        #             if boundary_element and boundary_element.is_a('IfcDoor'):
        #                 connected_doors.append(boundary_element)

        # # 打印调试信息
        # print(f"Space: {ifc_space.Name}, Found doors: {len(connected_doors)}")
        # for door in connected_doors:
        #     print(f"  - Door: {door.Name}")

        return connected_doors

    def subdivide_corridor_space(self, ifc_space):
        """
        基于空间几何形状的分段处理和门的数量判断,返回centers和是否为走廊
        """
        bbox_points = self.get_space_bbox(ifc_space)
        if not bbox_points:
            return [self.get_center(ifc_space)], False

        # 计算长宽比
        length = max(bbox_points[1][0] - bbox_points[0][0],
                     bbox_points[1][1] - bbox_points[0][1])
        width = min(bbox_points[1][0] - bbox_points[0][0],
                    bbox_points[1][1] - bbox_points[0][1])
        ratio = length / width

        # 获取空间相关的门的数量
        connected_doors = self.get_space_doors(ifc_space)
        # print('connected_doors:', connected_doors)
        door_count = len(connected_doors) if connected_doors else 0

        # 判断是否为走廊：长宽比大于4 或者 门的数量大于等于4
        is_corridor = ratio > 4 or door_count >= 4
        # is_corridor = ratio > 4

        if is_corridor:
            num_segments = max(int(0.5 * ratio), 1)  # 确保至少分为2段
            segment_length = length / num_segments
            segment_centers = []

            is_x_direction = (bbox_points[1][0] - bbox_points[0][0]) > (bbox_points[1][1] - bbox_points[0][1])

            for i in range(num_segments):
                if is_x_direction:
                    x = bbox_points[0][0] + (i + 0.5) * segment_length
                    y = (bbox_points[0][1] + bbox_points[1][1]) / 2
                else:
                    x = (bbox_points[0][0] + bbox_points[1][0]) / 2
                    y = bbox_points[0][1] + (i + 0.5) * segment_length
                segment_centers.append((x, y, self.current_elevation))

            return segment_centers, True  # 返回centers和走廊标志

        return [self.get_center(ifc_space)], False  # 返回center和非走廊标志

    def get_space_volume(self,space):
        """获取IfcSpace的NetVolume"""
        # 获取所有的PropertySet和QuantitySet
        quantity_sets = space.IsDefinedBy

        for quantity_set in quantity_sets:
            # 确保是数量集
            if quantity_set.is_a('IfcRelDefinesByProperties'):
                property_set = quantity_set.RelatingPropertyDefinition

                # 检查是否是BaseQuantities
                if property_set.is_a('IfcElementQuantity') and property_set.Name == 'BaseQuantities':
                    quantities = property_set.Quantities

                    # 查找NetVolume
                    for quantity in quantities:
                        if quantity.is_a('IfcQuantityVolume') and quantity.Name == 'NetVolume':
                            return quantity.VolumeValue

        return 0  # 如果没找到则返回0

    def get_door_width(self, door):
        """获取IfcDoor的宽度"""
        # 方法1：直接从对象属性获取
        if hasattr(door, 'OverallWidth'):
            return door.OverallWidth
        else:
            return 0

    def get_door_is_external(self, door):
        """获取IfcDoor的IsExternal属性"""
        for definition in door.IsDefinedBy:
            if definition.is_a('IfcRelDefinesByProperties'):
                property_set = definition.RelatingPropertyDefinition
                if property_set.Name == 'Pset_DoorCommon':
                    for prop in property_set.HasProperties:
                        if prop.Name == 'IsExternal':
                            return prop.NominalValue.wrappedValue
        return False  # 如果没有找到IsExternal属性，默认返回False

    def get_window_open_states(self, spaces):
        """
        返回所有空间的窗户状态
        :param spaces: IFC spaces列表
        :param opened_space_windows: 指定哪些空间的窗户是打开的，例如 {3} 表示空间3的窗户是打开的
        """
        # 使用spaces的数量
        num_spaces = len(spaces)
        print('num_spaces:', num_spaces)
        # 默认所有空间的窗户都是关闭的
        window_states = {i: False for i in range(num_spaces)}

        # 如果指定了要打开的窗户
        if self.opened_windows:
            for space_idx in self.opened_windows:
                print('space_idx:', space_idx)
                if not isinstance(space_idx, int):
                    raise TypeError(f"Space index must be an integer, got {type(space_idx)}")
                if space_idx < 0:
                    raise ValueError(f"Space index must be non-negative, got {space_idx}")
                if space_idx >= num_spaces:
                    raise ValueError(f"Space index {space_idx} out of range (total spaces: 0-{num_spaces - 1})")
                window_states[space_idx] = True

        return window_states

        # except Exception as e:
        #     print(f"Error in get_window_open_states: {str(e)}")
        #     # 出错时返回默认状态（所有窗户关闭）
        #     return {i: True for i in range(num_spaces)}

    def process_and_build_topology(self, save_views=True, output_dir='topology_views'):
        load_backend("pyqt5")  # 使用OpenGL后端


        display, start_display, _, _ = init_display()
        # 在显示设置中启用硬件加速

        # 创建输出目录
        if save_views:
            import os
            os.makedirs(output_dir, exist_ok=True)
        ifc_building_storeys = self.ifc_file.by_type('IfcBuildingStorey')

        # 按照标高排序楼层
        sorted_storeys = sorted(ifc_building_storeys, key=lambda x: float(x.Elevation))
        all_processed_walls = []

        # 遍历每一层
        for storey in sorted_storeys:
            process_storey = storey.Name
            self.current_elevation = float(storey.Elevation)
            print(f"Processing storey: {process_storey}")

            space_centers = []
            door_centers = []
            stair_centers = []
            count = 0

            # 获取空间家具信息
            space_furniture_info = self.process_spaces_and_furniture(process_storey)

            # 处理墙体
            floor_walls = self.get_walls_on_specified_floor(process_storey)
            for wall in floor_walls:
                try:
                    shape = ifcopenshell.geom.create_shape(self.settings, wall)
                    processed_shape = shape.geometry
                    if processed_shape:
                        all_processed_walls.append(processed_shape)
                        display.DisplayShape(processed_shape, color=Quantity_Color(0, 0, 1, Quantity_TOC_RGB),
                                             transparency=0.5)
                except Exception as e:
                    print(f"Error processing wall {wall.GlobalId}: {str(e)}")
                    continue

            # 处理spaces
            spaces = self.get_ifc_spaces_on_specified_floor(process_storey)
            print(f"Original spaces found on floor {process_storey}: {len(spaces)}")

            # 处理所有spaces，获取centers和相关信息
            space_info = {}  # 用于存储每个space的信息
            space_centers = []
            for ifc_space in spaces:
                centers, is_corridor = self.subdivide_corridor_space(ifc_space)  # 获取centers和走廊标志
                valid_centers = [c for c in centers if c]
                space_centers.extend(valid_centers)

                # 计算空间体积
                total_volume = self.get_space_volume(ifc_space)
                sub_volume = total_volume / len(centers) if centers else 0

                # 获取家具体积信息
                space_name = ifc_space.Name if hasattr(ifc_space, 'Name') and ifc_space.Name else ifc_space.GlobalId
                furniture_volume = space_furniture_info.get(space_name, {}).get('total_volume', 0)

                # 存储信息
                space_info[ifc_space] = {
                    'centers': valid_centers,
                    'volume': sub_volume,
                    'is_corridor': is_corridor,
                    'furniture_volume': furniture_volume
                }

            # 获取窗户状态
            window_open_states = self.get_window_open_states(space_centers)
            space_window_states = self.get_room_window_states(spaces, window_open_states)

            # 添加节点到图中
            for ifc_space, info in space_info.items():
                # 获取这个space的window_state
                centers = info['centers']
                window_state = next((space_window_states[c] for c in centers if c in space_window_states), 0)

                # 计算每个子空间的家具体积（平均分配）
                sub_furniture_volume = info['furniture_volume'] / len(centers) if centers else 0

                # 为所有centers添加节点
                for center in centers:
                    self.G.add_node(center,
                                    type='space',
                                    window_state=window_state,
                                    volume=info['volume'],
                                    is_corridor=info['is_corridor'],
                                    furniture_volume=sub_furniture_volume)

            # 处理门
            for door in self.get_doors_on_specified_floor(process_storey):
                door_center = self.get_center(door)
                if door_center and door_center not in door_centers:
                    # 获取门的宽度
                    width = self.get_door_width(door)
                    is_external = self.get_door_is_external(door)
                    self.G.add_node(door_center,
                                    type='door',
                                    width=width,
                                    is_external=is_external)  # 添加width属性
                    door_centers.append(door_center)

            # 处理楼梯
            for stair in self.get_stairs_on_specified_floor(process_storey):
                stair_center = self.get_center(stair)
                if stair_center and stair_center not in stair_centers:
                    self.G.add_node(stair_center, type='stair')
                    stair_centers.append(stair_center)

            # 连接当前楼层的节点
            all_points = space_centers + door_centers + stair_centers
            for i, point1 in enumerate(all_points):
                for point2 in all_points[i + 1:]:
                    if point1 != point2:
                        # 只连接同一楼层的节点（z坐标相同）
                        if abs(point1[2] - point2[2]) < 0.001:
                            distance = ((point1[0] - point2[0]) ** 2 +
                                        (point1[1] - point2[1]) ** 2) ** 0.5

                            intersects_wall = False
                            for wall_shape in all_processed_walls:
                                if self.line_intersects_wall(point1, point2, wall_shape):
                                    intersects_wall = True
                                    break

                            if not intersects_wall:
                                self.G.add_edge(point1, point2, weight=distance)
                                count += 1


            print(f"Floor {process_storey} - Number of edges added: {count}")
        # 连接不同楼层的楼梯节点
        stair_nodes = [node for node in self.G.nodes() if self.G.nodes[node]['type'] == 'stair']
        # 按z坐标（楼层高度）排序
        sorted_stair_nodes = sorted(stair_nodes, key=lambda x: x[2])

        # 只连接相邻楼层的楼梯节点
        for i in range(len(sorted_stair_nodes) - 1):
            current_stair = sorted_stair_nodes[i]
            next_stair = sorted_stair_nodes[i + 1]
            # 确保是相邻楼层（可以根据实际情况调整判断条件）
            if current_stair[2] != next_stair[2]:
                vertical_distance = abs(next_stair[2] - current_stair[2])
                self.G.add_edge(current_stair, next_stair, weight=vertical_distance)


        self.optimize_graph(threshold=1.1)
        print(f"Total number of edges after optimization: {self.G.number_of_edges()}")

        # 可视化所有边
        for edge in self.G.edges:
            vertex1 = gp_Pnt(edge[0][0], edge[0][1], edge[0][2])
            vertex2 = gp_Pnt(edge[1][0], edge[1][1], edge[1][2])
            edge_shape = BRepBuilderAPI_MakeEdge(vertex1, vertex2).Edge()
            display.DisplayShape(edge_shape, color='RED', update=True)
        # 在显示和保存之前，确保所有图形都已经更新
        display.FitAll()

        # 定义要保存的视角
        views = [
            ("iso", "iso"),
            ("front", "front"),
            ("top", "top"),
            ("left", "left"),
            ("right", "right")
        ]

        if save_views:
            # 保存不同视角的图像
            for view_name, view_function in views:
                # 设置视角
                if view_function == "iso":
                    display.View_Iso()
                elif view_function == "front":
                    display.View_Front()
                elif view_function == "top":
                    display.View_Top()
                elif view_function == "left":
                    display.View_Left()
                elif view_function == "right":
                    display.View_Right()

                # 确保视图更新
                display.FitAll()

                # 保存高分辨率图像
                filename = os.path.join(output_dir, f"topology_{view_name}.png")
                self.save_display_high_quality(display, filename)
                print(f"Saved view: {filename}")

        # 显示交互式视图
        display.View_Iso()
        start_display()
        return display, start_display

    def save_display_high_quality(self, display, filename, width=1920, height=1080):
        """保存高质量的显示图像"""
        try:
            # 设置白色背景
            display.View.SetBackgroundColor(Quantity_TOC_RGB, 0.1, 0.1, 0.1)

            # 确保所有内容都在视图中
            display.FitAll()

            # 保存图像
            display.View.Dump(filename)

            return True
        except Exception as e:
            print(f"Error saving display: {str(e)}")
            return False

    def save_display_with_transparency(self, display, filename, width=1920, height=1080):
        """保存带透明背景的显示图像"""
        try:
            import numpy as np
            from PIL import Image

            # 保存白底和黑底的图像
            white_filename = filename + "_white.png"
            black_filename = filename + "_black.png"

            # 白底
            display.View.SetBackgroundColor(1, 1, 1)
            display.FitAll()
            display.View.Dump(white_filename, width, height)

            # 黑底
            display.View.SetBackgroundColor(0, 0, 0)
            display.View.Dump(black_filename, width, height)

            # 使用PIL处理图像
            white_img = Image.open(white_filename)
            black_img = Image.open(black_filename)

            # 计算alpha通道
            white_pixels = np.array(white_img)
            black_pixels = np.array(black_img)

            alpha = white_pixels - black_pixels
            alpha = np.mean(alpha, axis=2)
            alpha = (alpha * 255.0 / alpha.max()).astype(np.uint8)

            # 创建RGBA图像
            rgba = np.dstack((white_pixels[:, :, :3], alpha))
            result = Image.fromarray(rgba)

            # 保存最终结果
            result.save(filename)

            # 清理临时文件
            import os
            os.remove(white_filename)
            os.remove(black_filename)

            return True
        except Exception as e:
            print(f"Error saving transparent display: {str(e)}")
            return False

    def get_layout_positions(self):
        """获取节点的布局位置"""
        return {node: (node[0], node[1]) for node in self.G.nodes()}

    def get_room_window_states(self, spaces, window_open_states):
        """
        返回一个字典，key是space的center坐标，value是window_state
        只考虑外窗(IsExternal=True)的状态
        """
        space_window_states = {}
        node_idx = 0  # 用于追踪节点索引

        for space in spaces:
            centers, _ = self.subdivide_corridor_space(space)
            external_windows = []

            # 从space获取墙，再从墙获取窗
            for ifc_rel_space_boundary in space.BoundedBy:
                if (ifc_rel_space_boundary.RelatedBuildingElement is not None and
                        ifc_rel_space_boundary.RelatedBuildingElement.is_a('IfcWallStandardCase')):
                    wall = ifc_rel_space_boundary.RelatedBuildingElement

                    # 从墙获取窗
                    if wall.HasOpenings is not None:
                        for ifc_rel_voids_element in wall.HasOpenings:
                            ifc_opening_element = ifc_rel_voids_element.RelatedOpeningElement
                            if ifc_opening_element.HasFillings is not None:
                                for ifc_rel_fills_element in ifc_opening_element.HasFillings:
                                    window = ifc_rel_fills_element.RelatedBuildingElement
                                    if window.is_a('IfcWindow'):
                                        # 检查窗户是否是外窗
                                        for definition in window.IsDefinedBy:
                                            if definition.is_a('IfcRelDefinesByProperties'):
                                                property_set = definition.RelatingPropertyDefinition
                                                if property_set.Name == 'Pset_WindowCommon':
                                                    for prop in property_set.HasProperties:
                                                        if prop.Name == 'IsExternal' and prop.NominalValue.wrappedValue:
                                                            external_windows.append(window)
                                                            break

            # 设置窗户状态
            has_external_window = len(external_windows) > 0

            # 对这个space的所有center分别设置window_state
            for center in centers:
                if center:
                    is_window_open = window_open_states.get(node_idx, False)  # 使用节点索引获取窗户状态
                    window_state = 1 if (has_external_window and is_window_open) else 0  # 有外窗且窗是开着的才为1
                    space_window_states[center] = window_state
                    node_idx += 1  # 更新节点索引

        return space_window_states

    def build_adjacency_matrix(self):
        """构建包含原始距离信息的邻接矩阵"""
        nodes = list(self.G.nodes())
        n = len(nodes)
        adj_matrix = np.zeros((n, n))

        # 建立节点索引映射并保存为类属性
        self.node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        print("节点映射",self.node_to_idx)
        self.idx_to_node = {idx: node for idx, node in enumerate(nodes)}

        # 遍历所有边，保存原始距离
        for u, v, data in self.G.edges(data=True):
            i = self.node_to_idx[u]
            j = self.node_to_idx[v]
            distance = data['weight']
            # 直接存储距离值
            adj_matrix[i][j] = distance
            adj_matrix[j][i] = distance

        # 添加自环
        np.fill_diagonal(adj_matrix, 1)

        self.adj_matrix = adj_matrix
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        print(f"节点总数: {len(self.G.nodes())}")
        print("邻接矩阵",adj_matrix.shape)
        return adj_matrix

    def build_distance_matrix(self):
        """构建最短路径距离矩阵"""
        import networkx as nx

        # 直接使用all_pairs_shortest_path_length获取距离
        dist_matrix = nx.floyd_warshall_numpy(self.G, weight='weight')

        self.dist_matrix = dist_matrix

        return dist_matrix

    def calculate_specific_weight_origin(self, node1, node2, fire_source_id):
        """
        计算直接相连节点间的传播权重，考虑六个影响因素并记录每个阶段的权重变化：
        1. 基础权重
        2. 空间体积比例影响
        3. 空间体积大小影响（门-空间连接）
        4. 窗户状态影响
        5. 家具影响
        6. 外门影响
        """
        node1_idx = self.node_to_idx[node1]
        node2_idx = self.node_to_idx[node2]
        fire_idx = self.node_to_idx[fire_source_id]

        if self.adj_matrix[node1_idx, node2_idx] == 0:
            return float('inf')

        base_weight = self.dist_matrix[node1_idx, node2_idx]
        weight_modifier = 1.0
        current_weight = base_weight

        # 存储权重变化
        weight_changes = [base_weight]  # 基础权重

        node1_data = self.G.nodes[node1]
        node2_data = self.G.nodes[node2]

        type1 = node1_data['type']
        type2 = node2_data['type']

        # 1. 空间体积比例影响
        if type1 == 'space' and type2 == 'space':
            volume1 = node1_data.get('volume', 0)
            volume2 = node2_data.get('volume', 0)

            if abs(volume1 - volume2) > 1e-6:
                dist1_to_source = self.dist_matrix[fire_idx, node1_idx]
                dist2_to_source = self.dist_matrix[fire_idx, node2_idx]

                if dist1_to_source < dist2_to_source:
                    if volume1 < volume2:
                        weight_modifier *= 1.2
                    else:
                        weight_modifier *= 0.8
                else:
                    if volume2 < volume1:
                        weight_modifier *= 1.2
                    else:
                        weight_modifier *= 0.8

                # print(f"\n1. Volume Ratio Effect ({node1_idx}-{node2_idx}):")
                # print(f"Volumes: node{node1_idx}={volume1:.2f}, node{node2_idx}={volume2:.2f}")
                # print(f"Current modifier: {weight_modifier:.2f}")

        current_weight = base_weight * weight_modifier
        weight_changes.append(current_weight)  # 记录空间体积比例影响后的权重

        # 2. 空间体积大小影响（门-空间连接）
        if (type1 == 'door' and type2 == 'space') or (type1 == 'space' and type2 == 'door'):
            space_data = node1_data if type1 == 'space' else node2_data
            space_volume = space_data.get('volume', 0)
            base_volume = 50

            if space_volume > base_volume:
                volume_ratio = space_volume / base_volume
                volume_modifier = 1 + 0.15 * min(volume_ratio - 1, 5)
                weight_modifier *= volume_modifier

                # print(f"\n2. Volume Size Effect (Door-Space):")
                # print(f"Space volume: {space_volume:.2f}")
                # print(f"Volume modifier: {volume_modifier:.2f}")
                # print(f"Current modifier: {weight_modifier:.2f}")

        current_weight = base_weight * weight_modifier
        weight_changes.append(current_weight)  # 记录空间体积大小影响后的权重

        # 3. 窗户状态影响
        if node1_data.get('window_state') or node2_data.get('window_state'):
            fire_height = fire_source_id[2]
            HEIGHT_TOLERANCE = 0.2

            same_floor_nodes = [idx for idx, node_id in enumerate(self.node_to_idx)
                                if abs(node_id[2] - fire_height) <= HEIGHT_TOLERANCE]

            max_floor_dist = 0
            for i in same_floor_nodes:
                for j in same_floor_nodes:
                    if i != j and self.dist_matrix[i][j] != float('inf'):
                        max_floor_dist = max(max_floor_dist, self.dist_matrix[i][j])

            avg_distance = (self.dist_matrix[node1_idx, fire_idx] +
                            self.dist_matrix[node2_idx, fire_idx]) / 2

            node1_height = node1[2]
            node2_height = node2[2]

            if abs(fire_height - node1_height) > HEIGHT_TOLERANCE or abs(fire_height - node2_height) > HEIGHT_TOLERANCE:
                window_modifier = 1.1
            else:
                if fire_idx in (node1_idx, node2_idx):
                    window_modifier = 0.6
                elif max_floor_dist > 0:
                    relative_distance = avg_distance / max_floor_dist
                    if relative_distance <= 0.3:
                        window_modifier = 0.8
                    elif relative_distance >= 0.6:
                        window_modifier = 1.2
                    else:
                        window_modifier = 1.0

                weight_modifier *= window_modifier

                # print(f"\n3. Window Effect:")
                # print(f"Window modifier: {window_modifier:.2f}")
                # print(f"Current modifier: {weight_modifier:.2f}")

        current_weight = base_weight * weight_modifier
        weight_changes.append(current_weight)  # 记录窗户影响后的权重

        # 4. 家具（可燃物）影响
        if type1 == 'space' or type2 == 'space':
            space_data = node1_data if type1 == 'space' else node2_data
            furniture_vol = space_data.get('furniture_volume', 0)

            if furniture_vol > 0:
                furniture_modifier = 0.8
                weight_modifier *= furniture_modifier

                # print(f"\n4. Furniture Effect:")
                # print(f"Furniture modifier: {furniture_modifier:.2f}")
                # print(f"Current modifier: {weight_modifier:.2f}")

        current_weight = base_weight * weight_modifier
        weight_changes.append(current_weight)  # 记录家具影响后的权重

        # 5. 外门影响
        if type1 == 'door' or type2 == 'door':
            door_data = node1_data if type1 == 'door' else node2_data
            is_external = door_data.get('is_external', False)

            if is_external:
                door_idx = node1_idx if type1 == 'door' else node2_idx
                door_node = node1 if type1 == 'door' else node2

                fire_height = fire_source_id[2]
                door_height = door_node[2]

                HEIGHT_TOLERANCE = 0.2
                if abs(fire_height - door_height) > HEIGHT_TOLERANCE:
                    door_modifier = 1.2
                else:
                    same_floor_nodes = [idx for idx, node_id in enumerate(self.node_to_idx)
                                        if abs(node_id[2] - fire_height) <= HEIGHT_TOLERANCE]

                    max_floor_dist = 0
                    for i in same_floor_nodes:
                        for j in same_floor_nodes:
                            if i != j and self.dist_matrix[i][j] != float('inf'):
                                max_floor_dist = max(max_floor_dist, self.dist_matrix[i][j])

                    dist_to_fire = self.dist_matrix[door_idx][fire_idx]

                    if max_floor_dist > 0:
                        relative_distance = dist_to_fire / max_floor_dist
                        if relative_distance <= 0.3:
                            door_modifier = 0.8
                        elif relative_distance >= 0.6:
                            door_modifier = 1.2
                        else:
                            door_modifier = 1.0

                    weight_modifier *= door_modifier

                    # print(f"\n5. External Door Effect:")
                    # print(f"Door modifier: {door_modifier:.2f}")
                    # print(f"Current modifier: {weight_modifier:.2f}")

        current_weight = base_weight * weight_modifier
        weight_changes.append(current_weight)  # 记录外门影响后的权重

        # 存储该边的权重变化
        # 修改存储边的方式，使用节点索引
        edge = (node1_idx, node2_idx)  # 使用节点索引而不是坐标
        if not hasattr(self, 'weight_changes'):
            self.weight_changes = {}
        self.weight_changes[edge] = weight_changes

        return base_weight,current_weight

    def calculate_specific_weight(self, node1, node2, fire_source_id):
        """
        基于物理感知机制计算传播权重 (Final Optimized Version)

        物理逻辑修正：
        1. Volume Factor: 仅在填充空间(Target=Space)时激活。Door->Space阻力>1 (突扩)，Space->Door阻力=1 (瓶颈主导)。
        2. Ventilation Factor: 区分流体力学状态。直管(D->D) > 突扩(D->S) > 突缩(S->D)。
        3. Capacity Factor: 燃料主导逻辑 (Fuel-Controlled)，燃料越多阻力越小。
        4. Environmental Factor: 随火源距离衰减。
        """
        node1_idx = self.node_to_idx[node1]
        node2_idx = self.node_to_idx[node2]
        fire_idx = self.node_to_idx[fire_source_id]

        if self.adj_matrix[node1_idx, node2_idx] == 0:
            return float('inf'), float('inf')

        d_ij = self.dist_matrix[node1_idx, node2_idx]
        data_i, data_j = self.G.nodes[node1], self.G.nodes[node2]
        type1, type2 = data_i['type'], data_j['type']

        # 参数定义
        ALPHA = 0.5  # 体积敏感度
        BETA = 1.5  # 通风敏感度
        GAMMA = 0.8  # 燃料敏感度
        DELTA = 0.3  # 环境影响幅度
        LAMBDA = 20.0  # 距离衰减特征长度

        # --- 1. Volume Factor (容量/突扩阻力) ---
        # 逻辑：只有"进入并填满"一个大空间时，体积才构成阻力
        if type2 == 'space':
            vol_j = max(data_j.get('volume', 10.0), 10.0)
            if type1 == 'door':
                # Door->Space: 突扩效应 + 填充时间。门效体积设为10，计算倍率。
                # 限制最大倍率为6倍，防止大厅阻力过大。
                ratio = min(vol_j / 10.0, 6.0)
                phi_vol = ratio ** ALPHA
            elif type1 == 'space':
                # Space->Space: 标准体积热容比
                vol_i = max(data_i.get('volume', 10.0), 0.1)
                phi_vol = (vol_j / vol_i) ** ALPHA
            else:
                phi_vol = 1.0
        else:
            # Target is Door/Stair: 体积不构成阻力 (Space->Door 由瓶颈主导)
            phi_vol = 1.0

        # --- 2. Ventilation Factor (气动/接口阻力) ---
        # 逻辑：描述接口的几何通畅度和流体动能损失
        if type1 == 'door' and type2 == 'door':
            opening_ratio = 0.95  # 管流：极度通畅
        elif type1 == 'door' and type2 == 'space':
            opening_ratio = 0.75  # 突扩：有湍流损失
        elif type1 == 'space' and type2 == 'door':
            opening_ratio = 0.50  # 突缩：瓶颈限制 (几何约束最强)
        elif type1 == 'stair' and type2 == 'stair':
            opening_ratio = 1.0  # 烟囱效应：垂直贯通
        else:
            opening_ratio = 0.1  # 普通隔墙

        phi_vent = np.exp(-BETA * opening_ratio)

        # --- 3. Capacity Factor (燃料荷载) ---
        # 逻辑：燃料密度越高 -> 火势越猛 -> 阻力越小
        if type2 == 'space':
            rho = data_j.get('furniture_volume', 0) / max(data_j.get('volume', 1.0), 1.0)
            phi_cap = 1.0 / (1.0 + GAMMA * rho)
        else:
            phi_cap = 1.0

        # --- 4. Environmental Factor (距离衰减) ---
        # 逻辑：外窗/门的影响随火源距离指数衰减
        has_ext = data_j.get('is_external', False) or data_j.get('window_state', False)
        dist_fire = self.dist_matrix[node2_idx, fire_idx]

        decay = 0.0 if dist_fire == float('inf') else np.exp(-dist_fire / LAMBDA)
        phi_env = 1 - (DELTA * (1.0 if has_ext else 0.0) * decay)

        # --- Final Calculation ---
        total_modifier = phi_vol * phi_vent * phi_cap * phi_env
        final_weight = d_ij * total_modifier

        return d_ij, final_weight

    def visualize_weight_impact(self, save_path=None):
        """
        可视化：对比 Base Weight (几何距离) 与 Final Weight (物理修正) 的差异
        """
        print("开始生成权重影响分析图...")

        # 确保矩阵已加载
        if not hasattr(self, 'final_adj_matrix') or not hasattr(self, 'original_adj_matrix'):
            print("错误：矩阵尚未生成，请先运行 build_final_adjacency_matrix()")
            return

        n = len(self.G.nodes())
        ratios = []  # 存储比率: Final / Original
        edges_info = []  # 存储边的数据用于绘图

        # 收集数据
        for i in range(n):
            for j in range(n):
                # 只看有效连接，且忽略对角线
                if i != j and self.original_adj_matrix[i][j] > 0:
                    w_base = self.original_adj_matrix[i][j]
                    w_final = self.final_adj_matrix[i][j]

                    # 避免除以0
                    if w_base == 0: continue

                    ratio = w_final / w_base
                    ratios.append(ratio)

                    edges_info.append({
                        'u': self.idx_to_node[i],
                        'v': self.idx_to_node[j],
                        'ratio': ratio
                    })

        if not ratios:
            print("警告：未找到有效边数据进行可视化。")
            return

        ratios = np.array(ratios)

        # --- 创建画布 ---
        fig = plt.figure(figsize=(18, 8), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])

        # --- 图 1: 权重变化分布直方图 ---
        ax_hist = fig.add_subplot(gs[0])
        ax_hist.axvline(1.0, color='black', linestyle='--', linewidth=2, label='No Change')
        sns.histplot(ratios, bins=30, kde=True, ax=ax_hist, color='skyblue', edgecolor='black')
        mean_ratio = np.mean(ratios)
        ax_hist.axvline(mean_ratio, color='red', linestyle=':', linewidth=2, label=f'Mean: {mean_ratio:.2f}x')

        ax_hist.set_title('Distribution of Weight Modification Factors ($\phi_{total}$)', fontsize=14)
        ax_hist.set_xlabel('Ratio (Final / Base)', fontsize=12)
        ax_hist.legend()

        # --- 图 2: 空间网络热力图 ---
        ax_net = fig.add_subplot(gs[1])

        # 获取节点位置
        pos = {}
        # 尝试从 G.nodes 属性中获取 'pos' (x,y,z) 并转为 (x,y)
        # 如果你的节点是 (x,y,z) 元组作为ID，也可以直接用
        for node in self.G.nodes():
            # 方案A: 节点ID本身就是坐标 (x,y,z) -> 这种最方便
            if isinstance(node, tuple) and len(node) >= 2:
                pos[node] = (node[0], node[1])
            # 方案B: 节点属性中有 'pos'
            elif 'pos' in self.G.nodes[node]:
                p = self.G.nodes[node]['pos']
                pos[node] = (p[0], p[1])
            else:
                # 方案C: 弹簧布局兜底
                if not pos:  # 只计算一次
                    pos = nx.spring_layout(self.G, seed=42)
                if node not in pos:  # 防止有的节点没位置
                    pos[node] = (0, 0)

        # 1. 绘制普通节点
        nx.draw_networkx_nodes(self.G, pos, node_size=20, node_color='lightgray', alpha=0.6, ax=ax_net)

        # 2. 绘制火源节点 (修复部分：移除 marker 参数，改用 node_shape)
        fire_node = self.idx_to_node[self.fire_index]
        if fire_node in pos:
            nx.draw_networkx_nodes(
                self.G, pos,
                nodelist=[fire_node],
                node_size=200,  # 把火源画大一点
                node_color='red',
                node_shape='*',  # 这里使用 node_shape 来指定五角星
                label='Fire Source',
                ax=ax_net
            )

        # 3. 绘制边 (根据 Ratio 上色)
        edge_colors = [d['ratio'] for d in edges_info]
        edge_list = [(d['u'], d['v']) for d in edges_info]

        if edge_list:
            # 定义颜色映射和范围
            cmap = plt.cm.RdYlBu
            vmin = 0.5
            vmax = 1.5

            # 绘制边 (注意：我们不再依赖它的返回值来做 colorbar)
            nx.draw_networkx_edges(
                self.G, pos,
                edgelist=edge_list,
                edge_color=edge_colors,
                edge_cmap=cmap,
                edge_vmin=vmin,
                edge_vmax=vmax,
                width=1.5,
                alpha=0.8,
                ax=ax_net,
                arrows=True
            )

            # --- 修复部分：手动创建 ScalarMappable 用于 Colorbar ---
            # 这是一个虚拟对象，专门告诉 colorbar 颜色范围和色卡是什么
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])  # 必须设置一个空数组，否则 matplotlib 会报错

            # 使用 sm 来生成 colorbar
            cbar = plt.colorbar(sm, ax=ax_net, fraction=0.046, pad=0.04)
            cbar.set_label('Physics Modifier Factor ($\phi$)\nRed=Faster Spread, Blue=Slower Spread', rotation=270,
                           labelpad=20)

        ax_net.set_title(f'Spatial Weight Impact (Fire Node: {self.fire_index})', fontsize=14)
        ax_net.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()
    def get_weight_changes(self):
        """
        返回所有边的权重变化记录
        格式：{(node1, node2): [base_weight, volume_ratio_weight, volume_size_weight,
                               window_weight, furniture_weight, external_door_weight]}
        """
        if hasattr(self, 'weight_changes'):
            return self.weight_changes
        return {}

    def build_final_adjacency_matrix(self):
        """构建考虑火灾蔓延因素后的最终邻接矩阵，并保存 Original 和 Final 两个版本（均包含楼梯修正）"""
        self.build_adjacency_matrix()
        fire_source_node = self.idx_to_node[self.fire_index]
        n = len(self.G.nodes())

        # 初始化两个矩阵
        final_adj_matrix = np.zeros((n, n))
        original_adj_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    node1 = self.idx_to_node[i]
                    node2 = self.idx_to_node[j]

                    if self.adj_matrix[i][j] > 0:
                        # 检查是否是楼梯连接
                        is_stair_connection = (self.G.nodes[node1].get('type') == 'stair' and
                                               self.G.nodes[node2].get('type') == 'stair')

                        # 获取基础权重(base_weight)和受火灾影响权重(weight)
                        base_weight, weight = self.calculate_specific_weight(node1, node2, fire_source_node)

                        # 使用临时变量存储即将赋值的权重
                        val_original = base_weight
                        val_final = weight

                        # --- 统一应用楼梯修正 ---
                        if is_stair_connection:
                            stair_factor = self.calculate_stair_factor(node1, node2, fire_source_node)
                            # Original 和 Final 都乘上楼梯修正系数
                            val_original *= stair_factor
                            val_final *= stair_factor

                        # 赋值到对应矩阵
                        original_adj_matrix[i][j] = val_original
                        final_adj_matrix[i][j] = val_final
                    else:
                        final_adj_matrix[i][j] = 0
                        original_adj_matrix[i][j] = 0

        # 填充对角线
        np.fill_diagonal(final_adj_matrix, 1)
        np.fill_diagonal(original_adj_matrix, 1)

        self.original_adj_matrix = original_adj_matrix
        self.final_adj_matrix = final_adj_matrix

        # --- 保存矩阵 ---
        save_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. 保存 Final 版本 (weight * stair_factor)
        save_path_final = os.path.join(save_dir, f'adj_final{self.fire_index}.npz')
        sp.save_npz(save_path_final, sp.csc_matrix(final_adj_matrix))

        # 2. 保存 Original 版本 (base_weight * stair_factor)
        save_path_original = os.path.join(save_dir, f'adj_original{self.fire_index}.npz')
        sp.save_npz(save_path_original, sp.csc_matrix(original_adj_matrix))

        # 同时也保存边列表格式
        self.matrix_to_edge_list()

        return final_adj_matrix

    def apply_gaussian_threshold(self):
        """对矩阵应用高斯核和阈值处理 (保存 Final 和 Original 两个版本)"""
        save_dir = os.path.dirname(os.path.abspath(__file__))

        # 定义要处理的目标：(名称标识, 对应的矩阵数据)
        targets = [
            ('final', self.final_adj_matrix),
            ('original', self.original_adj_matrix)
        ]

        for name, matrix in targets:
            n = matrix.shape[0]
            result = np.zeros((n, n))

            # 应用高斯核
            for i in range(n):
                for j in range(n):
                    if i != j and matrix[i][j] > 0:
                        result[i][j] = np.exp(-matrix[i][j] ** 2 / 20)

            # 保持对角线为1
            np.fill_diagonal(result, 1.0)

            # 保存文件：分别保存为 adj_gaussian_final_... 和 adj_gaussian_original_...
            save_path = os.path.join(save_dir, f'adj_gaussian_{name}_{self.fire_index}.npz')
            sp.save_npz(save_path, sp.csc_matrix(result))

            # 更新类属性 (self.gaussian_matrix 默认指向 final 以保持后续代码兼容)
            if name == 'final':
                self.gaussian_matrix = result

        return self.gaussian_matrix

    def matrix_to_edge_list(self):
        """
        将最终邻接矩阵转换为边列表格式的CSV
        """
        n = len(self.final_adj_matrix)
        edges = []

        # 遍历矩阵中的每个非零元素
        for i in range(n):
            for j in range(n):
                if self.final_adj_matrix[i][j] > 0 and i != j:  # 排除对角线上的1
                    edges.append({
                        'from': i,
                        'to': j,
                        'cost': self.final_adj_matrix[i][j]  # 保留一位小数
                    })

        # 创建DataFrame
        df = pd.DataFrame(edges)

        # 保存为CSV
        save_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(save_dir, f'edge_list{self.fire_index}.csv')
        df.to_csv(save_path, index=False)

        return df

    def calculate_stair_factor(self, node1, node2, fire_source_node):
        """计算楼梯连接的修正因子(越小表示越容易蔓延)"""
        z1, z2 = node1[2], node2[2]
        fire_z = fire_source_node[2]

        base_factor = 1.0

        # 1. 考虑火源位置的影响
        if min(z1, z2) <= fire_z <= max(z1, z2):
            # 火源在两个楼梯节点之间,更容易蔓延
            base_factor *= 1.0
        elif fire_z > max(z1, z2):
            # 火源在楼梯节点上方,不太容易向下蔓延
            base_factor *= 1.2
        elif fire_z < min(z1, z2):
            # 火源在下方,容易通过烟囱效应向上蔓延
            base_factor *= 0.8

        # 2. 考虑蔓延方向
        if z2 > z1:  # 向上蔓延
            base_factor *= 0.8  # 由于热浮力,更容易向上蔓延

        # 3. 考虑高度差的影响(可选)
        height_diff = abs(z2 - z1)
        if height_diff > 3:
            # 距离过大时蔓延会有所减弱
            base_factor *= (1 + 0.05 * (height_diff - 3))

        return base_factor


    def _base_visualize_network(self, graph, weights_dict, title):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 定义颜色
        COLOR_SPACE = '#87CEEB'  # 浅蓝色
        COLOR_DOOR = '#FF0000'  # 红色
        COLOR_STAIR = '#FFD700'  # 金黄色
        COLOR_FIRE = '#FFFF00'  # 亮黄色

        # 绘制节点和节点编号
        for i, node in enumerate(graph.nodes()):
            node_type = graph.nodes[node].get('type', '')
            # 设置fire_source
            if i == self.fire_index:
                color = COLOR_FIRE
                node_type = 'fire_source'
            elif node_type == 'space':
                color = COLOR_SPACE
            elif node_type == 'door':
                color = COLOR_DOOR
            elif node_type == 'stair':
                color = COLOR_STAIR
            else:
                color = 'gray'

            # 绘制节点，添加透明度
            ax.scatter(node[0], node[1], node[2], c=color, s=100, alpha=0.5)

            # 添加节点编号，稍微偏移位置以增加可见性
            ax.text(node[0] + 0.1, node[1] + 0.1, node[2] + 0.1, f'{i}', fontsize=8)

        # 绘制边和权重
        for (node1, node2, data) in graph.edges(data=True):
            x = [node1[0], node2[0]]
            y = [node1[1], node2[1]]
            z = [node1[2], node2[2]]
            ax.plot(x, y, z, 'gray', alpha=0.3)  # 降低边的透明度

            # 显示边的权重
            weight = weights_dict.get((node1, node2))
            if weight is not None:
                mid_x = (node1[0] + node2[0]) / 2
                mid_y = (node1[1] + node2[1]) / 2
                mid_z = (node1[2] + node2[2]) / 2
                ax.text(mid_x, mid_y, mid_z, f'{weight:.2f}', fontsize=8)

        # 设置标题和轴标签
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_SPACE, label='Spaces', markersize=10,
                       alpha=0.5),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_DOOR, label='Doors', markersize=10,
                       alpha=0.5),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_STAIR, label='Stairs', markersize=10,
                       alpha=0.5),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_FIRE, label='Fire Source', markersize=10,
                       alpha=0.5)
        ]
        ax.legend(handles=legend_elements)

        # 设置等比例
        ax.set_box_aspect([1, 1, 1])

        plt.show()

    def visualize_network(self):
        """显示原始网络图"""
        weights = {(u, v): data['weight'] for u, v, data in self.G.edges(data=True)}
        self._base_visualize_network(self.G, weights, 'Building Graph')

    def visualize_final_network(self):
        """显示基于final_adj_matrix的网络图"""
        if not hasattr(self, 'final_adj_matrix'):
            print("未找到final_adj_matrix，请先计算调整后的邻接矩阵")
            return

        # 创建新图并添加调整后的边
        G_final = nx.Graph()
        G_final.add_nodes_from(self.G.nodes(data=True))

        nodes_list = list(self.G.nodes())
        for i, node1 in enumerate(nodes_list):
            for j, node2 in enumerate(nodes_list):
                if i != j:
                    weight = self.final_adj_matrix[i][j]
                    if weight > 0:
                        G_final.add_edge(node1, node2, weight=weight)

        weights = {(u, v): float(data['weight']) for u, v, data in self.G.edges(data=True)}
        self._base_visualize_network(G_final, weights, 'Building Graph (Final Weights)')

    def visualize_matrix(self, matrix, title="Adjacency Matrix", cmap='viridis', tick_step=20):
        """
        可视化邻接矩阵
        Args:
            matrix: 要可视化的矩阵
            title: 图表标题
            cmap: 颜色映射
            tick_step: 坐标轴标签的显示间隔
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # 创建图形
        plt.figure(figsize=(10, 8))

        # 处理无穷大值，将其替换为较大的有限值以便可视化
        matrix_vis = matrix.copy()
        matrix_vis[matrix_vis == float('inf')] = np.nanmax(matrix_vis[matrix_vis != float('inf')]) * 1.5

        # 生成间隔标签位置
        n = len(matrix)
        tick_positions = np.arange(0, n, tick_step)

        # 创建热力图
        sns.heatmap(
            matrix_vis,
            annot=False,
            cmap=cmap,
            square=True,
            cbar=True
        )

        # 设置标题和轴标签
        plt.title(title, fontsize=40, pad=20)
        plt.xlabel('Node Index', fontsize=25, labelpad=10)
        plt.ylabel('Node Index', fontsize=25, labelpad=10)

        # 设置刻度位置和标签，增大字体大小
        plt.xticks(tick_positions, tick_positions, fontsize=25)
        plt.yticks(tick_positions, tick_positions, fontsize=25)

        # 调整colorbar的字体大小
        cbar = plt.gca().collections[0].colorbar
        cbar.ax.tick_params(labelsize=30)

        # 调整布局
        plt.tight_layout()

        # 显示图像
        plt.show()

    def network_visualization(self):
        """运行完整的可视化流程"""
        # 验证节点属性
        print("\n=== 验证图节点属性 ===")
        # space_nodes = [node for node, attrs in self.G.nodes(data=True)
        #                if attrs.get('type') == 'space']
        # print(f"space节点总数: {len(space_nodes)}")

        # # 打印几个示例节点的属性
        # print("\n示例节点属性:")
        # for node in space_nodes[:3]:  # 打印前3个space节点的属性
        #     print(f"节点 {node}: {self.G.nodes[node]}")

        self.pos = self.get_layout_positions()  # 使用函数获取位置信息
        print("节点位置信息",self.pos)
        self.visualize_network()
        self.visualize_final_network()
        return self.display, self.start_display, self.pos

    def matrix_visualization(self):
        """运行完整的矩阵可视化流程"""
        self.visualize_matrix(self.adj_matrix, title="Adjacency_Original Matrix")
        self.visualize_matrix(self.dist_matrix, title="Distance Matrix")
        self.visualize_matrix(self.final_adj_matrix, title="Final Adjacency Matrix")
        self.visualize_matrix(self.gaussian_matrix,title='Gaussian Matrix')

    def generate_fire_sources(self, output_file='fire_sources.txt'):
        """
        在space节点放置火源OBST、VENT(顶面)和CTRL，在所有节点放置三种DEVC（温度、CO、烟）
        对于每个space节点：
        - 如果该位置有家具，使用家具体积计算火源尺寸
        - 如果该位置没有家具，基于房间体积计算火源尺寸（参考有家具房间的比例）
        """
        fire_commands = []
        vent_commands = []
        ctrl_commands = []
        temp_devc_commands = []
        co_devc_commands = []
        soot_devc_commands = []
        solid_devc_commands = []
        node_comments = []

        volume_reduction_factor = 0.6
        spread_rate = 2.0E-3

        # 获取所有节点和space节点
        all_nodes = list(self.G.nodes(data=True))
        space_nodes = [(node, data) for node, data in self.G.nodes(data=True)
                       if data.get('type') == 'space'
                       and not data.get('is_corridor', False)]

        # 分析有家具房间的家具体积与房间体积的比例
        furniture_room_ratios = []
        for node, data in space_nodes:
            if data.get('furniture_volume', 0) > 0:
                room_volume = data.get('volume', 0)
                furniture_volume = data.get('furniture_volume', 0)
                if room_volume > 0:
                    furniture_room_ratios.append(furniture_volume / room_volume)

        # 计算平均比例，如果没有参考数据，使用一个默认值
        if furniture_room_ratios:
            avg_furniture_ratio = sum(furniture_room_ratios) / len(furniture_room_ratios)
        else:
            avg_furniture_ratio = 0.15  # 默认家具占房间体积的15%

        # 获取起火点信息
        initial_fire_node = list(self.G.nodes())[self.fire_index]
        initial_fire_data = self.G.nodes[initial_fire_node]

        # 确保起火点在space_nodes中
        fire_node_tuple = (initial_fire_node, initial_fire_data)
        if fire_node_tuple not in space_nodes:
            space_nodes.append(fire_node_tuple)

        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入文件头部信息
            f.write(f"! Initial fire point at node {self.fire_index}\n")
            f.write(f"! Fire sources size calculated from:\n")
            f.write(f"!   - With furniture: volume * {volume_reduction_factor} then cube root\n")
            f.write(
                f"!   - Without furniture: room volume * {avg_furniture_ratio:.3f} * {volume_reduction_factor} then cube root\n")
            f.write("!" + "=" * 80 + "\n\n")

            # 生成所有节点的注释
            for i, (node, data) in enumerate(all_nodes):
                x_center, y_center, z_center = node
                node_type = data.get('type', 'unknown')
                furniture_vol = data.get('furniture_volume', 0)
                room_vol = data.get('volume', 0)
                is_initial_fire = (i == self.fire_index)

                # 计算火源尺寸
                if furniture_vol > 0:
                    combustible_vol = furniture_vol
                    vol_source = "furniture"
                else:
                    combustible_vol = room_vol * avg_furniture_ratio if room_vol > 0 else 0
                    vol_source = "room"

                reduced_vol = combustible_vol * volume_reduction_factor
                size = round((reduced_vol) ** (1 / 3), 2) if reduced_vol > 0 else 0

                if is_initial_fire:
                    note = " (INITIAL FIRE POINT)"
                else:
                    note = ""

                node_info = f"! Node {i}{note} ({node_type}) at coordinates: ({x_center}, {y_center}, {z_center})"
                if size > 0:
                    node_info += f", Fire Source Size: {size}m (Based on {vol_source} volume: {combustible_vol:.2f}, Reduced: {reduced_vol:.2f})"
                else:
                    node_info += ", No valid volume for fire source calculation"

                node_comments.append(node_info)

            # 写入所有注释
            f.write("\n".join(node_comments) + "\n\n")

            # 为所有节点生成基础探测器
            for node_id, (node, data) in enumerate(all_nodes):
                x_center, y_center, z_center = node
                temp_devc_commands.append(
                    f"&DEVC ID='THCP{node_id}', QUANTITY='TEMPERATURE', XYZ={x_center},{y_center},{z_center + 1.9}/")
                co_devc_commands.append(
                    f"&DEVC ID='CO_{node_id}', QUANTITY='VOLUME FRACTION', SPEC_ID='CARBON MONOXIDE', XYZ={x_center},{y_center},{z_center + 1.9}/")
                soot_devc_commands.append(
                    f"&DEVC ID='Soot_{node_id}', QUANTITY='VOLUME FRACTION', SPEC_ID='SOOT', XYZ={x_center},{y_center},{z_center + 1.9}/")

            # 生成火源相关命令
            f.write("! OBST commands (Fire sources)\n")
            for node, data in space_nodes:
                x_center, y_center, z_center = node
                node_id = list(self.G.nodes()).index(node)
                furniture_vol = data.get('furniture_volume', 0)
                room_vol = data.get('volume', 0)

                # 计算火源尺寸
                if furniture_vol > 0:
                    combustible_vol = furniture_vol
                else:
                    combustible_vol = room_vol * avg_furniture_ratio if room_vol > 0 else 0

                reduced_volume = combustible_vol * volume_reduction_factor
                if reduced_volume > 0:
                    size = (reduced_volume) ** (1 / 3)
                    half_size = size / 2

                    # 计算坐标
                    xb = [
                        x_center - half_size,
                        x_center + half_size,
                        y_center - half_size,
                        y_center + half_size,
                        z_center,
                        z_center + size
                    ]

                    # 生成命令
                    fire_commands.append(
                        f"&OBST ID='Fire_Source_{node_id}', XB={xb[0]:.2f},{xb[1]:.2f},{xb[2]:.2f},{xb[3]:.2f},{xb[4]:.2f},{xb[5]:.2f}, SURF_ID='INERT'/")
                    vent_commands.append(
                        f"&VENT ID='Vent{node_id}', SURF_ID='ignition', XB={xb[0]:.2f},{xb[1]:.2f},{xb[2]:.2f},{xb[3]:.2f},{xb[5]:.2f},{xb[5]:.2f}, SPREAD_RATE={spread_rate}, XYZ={x_center},{y_center},{xb[5]:.2f}, CTRL_ID='{node_id}'/")
                    ctrl_commands.append(
                        f"&CTRL ID='{node_id}', FUNCTION_TYPE='DEADBAND', SETPOINT=20.0,330.0, ON_BOUND='UPPER', LATCH=.FALSE., INPUT_ID='solid{node_id}'/")
                    solid_devc_commands.append(
                        f"&DEVC ID='solid{node_id}', QUANTITY='ADIABATIC SURFACE TEMPERATURE', XYZ={x_center},{y_center},{xb[5]:.2f}, IOR=3/")

            # 写入所有命令
            f.write("\n".join(fire_commands) + "\n\n")
            f.write("! VENT commands (on OBST top faces)\n")
            f.write("\n".join(vent_commands) + "\n\n")
            f.write("! CTRL commands\n")
            f.write("\n".join(ctrl_commands) + "\n\n")
            f.write("! Solid surface temperature DEVC commands (for VENTs)\n")
            f.write("\n".join(solid_devc_commands) + "\n\n")
            f.write("! Temperature DEVC commands (for all nodes)\n")
            f.write("\n".join(temp_devc_commands) + "\n\n")
            f.write("! Carbon Monoxide DEVC commands (for all nodes)\n")
            f.write("\n".join(co_devc_commands) + "\n\n")
            f.write("! Soot DEVC commands (for all nodes)\n")
            f.write("\n".join(soot_devc_commands) + "\n\n")

            # 写入统计信息
            f.write("!" + "=" * 80 + "\n")
            f.write(f"! Average furniture to room volume ratio: {avg_furniture_ratio:.3f}\n")
            f.write(
                f"! Initial fire point (Node {self.fire_index}): {'Based on furniture' if data.get('furniture_volume', 0) > 0 else 'Based on room volume'}\n")
            f.write(f"! Total number of nodes: {len(all_nodes)}\n")
            f.write(f"! Number of potential fire sources: {len(space_nodes)}\n")

        print(f"Fire source, vent, control and device commands have been saved to {output_file}")
        devc_commands = temp_devc_commands + co_devc_commands + soot_devc_commands + solid_devc_commands
        return fire_commands, vent_commands, ctrl_commands, devc_commands

    def _get_changed_edges(self):
        """
        辅助函数：获取权重发生显著变化的边
        Returns:
            dict: 只包含权重发生显著变化的边的字典
        """
        changed_edges = {}
        for edge, weights in self.weight_changes.items():
            # 使用更大的阈值来判断显著变化
            if max(weights) - min(weights) > 0.1:  # 增加阈值到0.1或更大
                changed_edges[edge] = weights

            # 或者使用相对变化来判断
            # base_weight = weights[0]
            # relative_change = max([abs(w - base_weight) / base_weight for w in weights]) if base_weight != 0 else 0
            # if relative_change > 0.1:  # 10%的相对变化
            #     changed_edges[edge] = weights

        return changed_edges

    def export_for_origin(self, filename='weight_changes_origin.csv'):
        """
        导出适合Origin绘图的CSV格式数据
        """
        import pandas as pd

        changed_edges = self._get_changed_edges()
        if not changed_edges:
            print("没有边的权重发生变化")
            return

        # 准备数据
        stages = ['Base', 'Volume_Ratio', 'Volume_Size', 'Window', 'Furniture', 'External_Door']

        # 创建数据框架
        df = pd.DataFrame(index=stages)

        # 添加每条边的数据作为新的列
        for edge, weights in changed_edges.items():
            node1, node2 = edge
            col_name = f"Edge_{node1}_{node2}"
            df[col_name] = weights

        # 保存为CSV
        df.to_csv(filename)
        print(f"数据已保存到 {filename}")

    def visualize_weight_changes_line(self):
        """
        使用折线图展示权重变化过程，只显示发生变化的边
        """
        import matplotlib.pyplot as plt

        stages = ['Base', 'Volume Ratio', 'Volume Size', 'Window', 'Furniture', 'External Door']
        changed_edges = self._get_changed_edges()
        print(f"发生权重变化的边: {changed_edges}")
        if not changed_edges:
            print("没有边的权重发生变化")
            return

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)

        # 绘制变化的边
        for edge, weights in changed_edges.items():
            node1, node2 = edge
            label = f"Edge ({node1}-{node2})"
            ax.plot(stages, weights, marker='o', label=label)

        plt.xlabel('Modification Stages')
        plt.ylabel('Weight Value')
        plt.title('Weight Changes Through Different Stages (Changed Edges Only)')
        plt.xticks(rotation=45)

        # 优化图例显示
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # 缩小主图，为图例留空间
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.grid(True)
        plt.show()

    def visualize_weight_changes_heatmap(self):
        """
        使用热力图展示权重变化，只显示发生变化的边
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        stages = ['Base', 'Volume Ratio', 'Volume Size', 'Window', 'Furniture', 'External Door']
        changed_edges = self._get_changed_edges()

        if not changed_edges:
            print("没有边的权重发生变化")
            return

        # 创建数据矩阵
        edges = list(changed_edges.keys())
        data = np.array([changed_edges[edge] for edge in edges])

        plt.figure(figsize=(12, len(edges) * 0.5 + 2))
        sns.heatmap(data,
                    xticklabels=stages,
                    yticklabels=[f"({e[0]}-{e[1]})" for e in edges],
                    annot=True,
                    fmt='.3f',
                    cmap='YlOrRd')

        plt.xlabel('Modification Stages')
        plt.ylabel('Edges')
        plt.title('Weight Changes Heatmap (Changed Edges Only)')
        plt.tight_layout()
        plt.show()

    def visualize_temperature_with_slider(self, temperature_data, title="Temperature Distribution", threshold=100):
        """
        可视化图中节点的温度分布，带时间滑块

        参数:
        temperature_data: numpy数组，形状为(time_steps, n_nodes)
        title: 图表标题
        threshold: 温度阈值，超过此值的节点使用热色显示
        """
        fig = plt.figure(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25, right=0.9)  # 为颜色条留出空间
        ax = fig.add_subplot(111, projection='3d')

        x_coords = []
        y_coords = []
        z_coords = []
        for node in self.G.nodes():
            x_coords.append(node[0])
            y_coords.append(node[1])
            z_coords.append(node[2])

        time_idx = 0

        # 创建自定义颜色映射
        def custom_color_map(temperatures, threshold):
            colors = np.zeros((len(temperatures), 4))  # RGBA
            # 默认颜色：灰色
            colors[:, 0] = 0.7  # R
            colors[:, 1] = 0.7  # G
            colors[:, 2] = 0.7  # B
            colors[:, 3] = 0.7  # Alpha

            # 对超过阈值的温度使用红色到黄色的渐变
            high_temp_mask = temperatures > threshold
            if np.any(high_temp_mask):
                max_temp = np.max(temperatures)
                normalized_temps = (temperatures[high_temp_mask] - threshold) / (max_temp - threshold)

                # 红色到黄色的渐变
                colors[high_temp_mask, 0] = 1.0  # R
                colors[high_temp_mask, 1] = normalized_temps  # G
                colors[high_temp_mask, 2] = 0.0  # B
                colors[high_temp_mask, 3] = 1.0  # Alpha

            return colors

        # 初始散点图
        colors = custom_color_map(temperature_data[time_idx], threshold)
        scatter = ax.scatter(x_coords, y_coords, z_coords,
                             c=colors,
                             s=100)

        # 绘制边
        for edge in self.G.edges():
            node1, node2 = edge
            x = [node1[0], node2[0]]
            y = [node1[1], node2[1]]
            z = [node1[2], node2[2]]
            ax.plot(x, y, z, color='gray', alpha=0.3, linewidth=0.5)

        # 添加节点编号
        for i, node in enumerate(self.G.nodes()):
            ax.text(node[0] + 0.1, node[1] + 0.1, node[2] + 0.1,
                    f'{i}', fontsize=8)

        # 创建自定义颜色条
        import matplotlib.colors as mcolors
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom',
                                                                [(0, 'gray'),
                                                                 (0.5, 'red'),
                                                                 (1, 'yellow')])
        norm = mcolors.Normalize(vmin=threshold, vmax=np.max(temperature_data))
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])  # 设置一个空数组

        # 添加颜色条，指定位置
        cax = plt.axes([0.92, 0.25, 0.02, 0.6])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)

        # 设置标题和轴标签
        ax.set_title(f'{title} - Time Step: {time_idx}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 设置等比例
        ax.set_box_aspect([1, 1, 1])

        # 添加时间滑块
        ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
        slider = Slider(ax_slider, 'Time Step', 0, len(temperature_data) - 1,
                        valinit=0, valfmt='%d')

        def update(val):
            time_idx = int(slider.val)
            colors = custom_color_map(temperature_data[time_idx], threshold)
            scatter.set_facecolors(colors)
            ax.set_title(f'{title} - Time Step: {time_idx}')
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()
