import time
import json
import struct
import mmap
import os
import threading
import socket

import pyrealsense2 as rs
import numpy as np
import cv2
import pupil_apriltags
import logging
from scipy.spatial.transform import Rotation as R  

config = {
    'ipc': {
        'tcp_socket': {
            'port': 8081
        }
    }
}

APRILTAG_CONFIG = {
    'quad_decimate': 1.0,
    'quad_sigma': 0.0,
    'refine_edges': True,
    'decode_sharpening': 0.25
}

APRILTAG_PRESETS = {
    'high_accuracy': {  
        'quad_decimate': 0.5,
        'quad_sigma': 0.0,
        'refine_edges': True,
        'decode_sharpening': 0.5
    },
    'balanced': { 
        'quad_decimate': 1.0,
        'quad_sigma': 0.2,
        'refine_edges': True,
        'decode_sharpening': 0.25
    },
    'high_speed': {  
        'quad_decimate': 2.0,
        'quad_sigma': 0.5,
        'refine_edges': False,
        'decode_sharpening': 0.1
    },
    'noise_resistant': {  
        'quad_decimate': 1.0,
        'quad_sigma': 0.8,
        'refine_edges': True,
        'decode_sharpening': 0.4
    }
}


USE_PRESET = None  


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)


def retry_on_error(max_retries=3, retry_delay=1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(retry_delay)
            return None
        return wrapper
    return decorator

class IPCManager:
    
    def __init__(self, method='shared_memory', buffer_size=4096):
        self.method = method
        self.buffer_size = buffer_size
        self.is_running = False
        self.lock = threading.Lock()
        self.connection_lock = threading.Lock()
        
        self.shared_memory_file = None
        self.memory_map = None
        
        self.tcp_socket = None
        self.tcp_port = config.get('ipc', {}).get('tcp_socket', {}).get('port', 8081)
        self.client_connections = []  
        
        self._initialize_communication()
    
    @retry_on_error(max_retries=3, retry_delay=1.0)
    def _initialize_communication(self):
        try:
            if self.method == 'shared_memory':
                self._init_shared_memory()
            elif self.method == 'tcp_socket':
                self._init_tcp_socket()
            
            self.is_running = True
            logging.info(f"IPC通信初始化成功，方式: {self.method}")
            
        except Exception as e:
            logging.error(f"IPC通信初始化失败: {e}")
            raise
    
    @retry_on_error(max_retries=3, retry_delay=1.0)
    def _init_shared_memory(self):
        try:
            self.shared_memory_file = 'apriltag_pose_data.tmp'
            
            if os.path.exists(self.shared_memory_file):
                os.remove(self.shared_memory_file)
            
            with open(self.shared_memory_file, 'wb') as f:
                f.write(b'\x00' * self.buffer_size)
            
            with open(self.shared_memory_file, 'r+b') as f:
                self.memory_map = mmap.mmap(f.fileno(), self.buffer_size)
            
            logging.info(f"共享内存初始化成功，文件: {self.shared_memory_file}")
            
        except Exception as e:
            logging.error(f"共享内存初始化失败: {e}")
            raise
    
    @retry_on_error(max_retries=3, retry_delay=1.0)
    def _init_tcp_socket(self):
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_socket.settimeout(5.0)
            self.tcp_socket.bind(('localhost', self.tcp_port))
            self.tcp_socket.listen(5)
            
            self.server_thread = threading.Thread(target=self._tcp_server_loop, daemon=True)
            self.server_thread.start()
            
            logging.info(f"TCP Socket服务器启动成功，端口: {self.tcp_port}")
            
        except Exception as e:
            logging.error(f"TCP Socket初始化失败: {e}")
            raise
    

    
    def _tcp_server_loop(self):
        print(f"TCP服务器开始监听端口 {self.tcp_port}，等待客户端连接...")
        while self.is_running:
            try:
                self.tcp_socket.settimeout(1.0)  
                client_socket, addr = self.tcp_socket.accept()
                print(f"✓ 客户端已连接: {addr}")
                logging.info(f"客户端连接: {addr}")
                
                with self.connection_lock:
                    if not hasattr(self, 'client_connections'):
                        self.client_connections = []
                    self.client_connections.append((client_socket, addr))
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"TCP服务器错误: {e}")
                    logging.error(f"TCP服务器错误: {e}")
    
    def _handle_tcp_client(self, client_socket):
        send_count = 0
        check_count = 0
        client_addr = client_socket.getpeername()
        print(f"开始处理客户端 {client_addr} 的数据传输")
        
        try:
            while self.is_running:
                check_count += 1
                with self.lock:
                    if self.latest_pose_data:
                        json_data = json.dumps(self.latest_pose_data, separators=(',', ':')).encode('utf-8')
                        current_timestamp = int(self.latest_pose_data['t'] * 1000000)
                        
                        send_count += 1
                        client_socket.send(struct.pack('I', len(json_data)))  
                        client_socket.send(struct.pack('Q', current_timestamp))  
                        client_socket.send(json_data)  
                        
                        self.latest_pose_data = None
                    else:
                        if check_count % 4000 == 0:  
                            print(f"TCP客户端 {client_addr} 等待新数据，已检查: {check_count}次，已发送: {send_count}个数据包")
                
                time.sleep(0.001) 
                
        except Exception as e:
            print(f"TCP客户端 {client_addr} 处理错误: {e}")
            logging.error(f"TCP客户端处理错误: {e}")
        finally:
            print(f"客户端 {client_addr} 连接已断开，总共发送了 {send_count} 个数据包")
            client_socket.close()
    
    def send_pose_data(self, cube_pose_matrix, timestamp=None):
        if not self.is_running:
            print("⚠ IPC管理器未运行，无法发送数据")
            return
        
        try:
            if timestamp is None:
                timestamp = time.time()
        
            pose_matrix_mm = cube_pose_matrix.copy()
            pose_matrix_mm[:3, 3] *= 1000  
        
            pose_data = {
                't': timestamp,
                'pose_matrix': pose_matrix_mm.tolist()  
            }
            
            
            if self.method == 'shared_memory':
                self._send_via_shared_memory(pose_data)       
            elif self.method == 'tcp_socket':
                self._send_via_tcp(pose_data)
            
        except Exception as e:
            logging.error(f"发送位姿数据失败: {e}")
    
    def _send_via_shared_memory(self, pose_data):
        try:
            if self.memory_map:
                json_data = json.dumps(pose_data, separators=(',', ':')).encode('utf-8')
                
                header_size = 12  
                if len(json_data) > self.buffer_size - header_size:
                    logging.warning(f"数据过大，截断发送: {len(json_data)} > {self.buffer_size - header_size}")
                    json_data = json_data[:self.buffer_size - header_size]
                
                self.memory_map.seek(0)
                self.memory_map.write(struct.pack('I', len(json_data))) 
                timestamp_us = int(pose_data['t'] * 1000000)
                self.memory_map.write(struct.pack('Q', timestamp_us))  
                self.memory_map.write(json_data)
                self.memory_map.flush()
                
        except Exception as e:
            logging.error(f"共享内存发送失败: {e}")
            import traceback
            logging.error(f"详细错误信息: {traceback.format_exc()}")

    def _send_via_tcp(self, pose_data):
        try:
            with self.connection_lock:
                if not hasattr(self, 'client_connections') or not self.client_connections:
                    return  
                
                json_data = json.dumps(pose_data, separators=(',', ':')).encode('utf-8')
                
                timestamp_us = int(pose_data['t'] * 1000000)
                header = struct.pack('IQ', len(json_data), timestamp_us)
                
                disconnected = []
                
                for i, (client_socket, client_addr) in enumerate(self.client_connections):
                    try:
                        client_socket.sendall(header + json_data)
                    except Exception as e:
                        logging.error(f"TCP客户端发送错误: {e}")
                        disconnected.append(i)
                
                for idx in sorted(disconnected, reverse=True):
                    try:
                        client_socket, client_addr = self.client_connections[idx]
                        client_socket.close()
                        print(f"关闭断开的客户端连接: {client_addr}")
                    except Exception as e:
                        logging.error(f"关闭断开的客户端连接时出错: {e}")
                    del self.client_connections[idx]
                    
        except Exception as e:
            logging.error(f"TCP发送失败: {e}")
            import traceback
            logging.error(f"详细错误信息: {traceback.format_exc()}")    
    

    



    

    

    
    def get_connection_info(self):
        if self.method == 'shared_memory':
            return {
                'method': 'shared_memory',
                'file_path': os.path.abspath(self.shared_memory_file),
                'buffer_size': self.buffer_size
            }
        elif self.method == 'tcp_socket':
            return {
                'method': 'tcp_socket',
                'host': 'localhost',
                'port': self.tcp_port
            }
        return {}
    
    def close(self):
        self.is_running = False
        
        try:
            with self.connection_lock:
                if hasattr(self, 'client_connections'):
                    for client_socket, client_addr in self.client_connections:
                        try:
                            print(f"关闭客户端连接: {client_addr}")
                            client_socket.close()
                        except Exception as e:
                            logging.error(f"关闭客户端连接 {client_addr} 时出错: {e}")
                    self.client_connections = []
            
            if self.memory_map:
                self.memory_map.close()
            
            if self.tcp_socket:
                self.tcp_socket.close()
            
            if self.shared_memory_file and os.path.exists(self.shared_memory_file):
                os.remove(self.shared_memory_file)
            
            logging.info("IPC通信已关闭")
            
        except Exception as e:
            logging.error(f"关闭IPC通信时出错: {e}")


def setup_realsense_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) 
    
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    pipeline.start(config)
    return pipeline, align

def setup_apriltag_detector(quad_decimate=1.0, quad_sigma=0.0, refine_edges=True, decode_sharpening=0.25):
    detector = pupil_apriltags.Detector(
        families='tag36h11',
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening
    )
    
    rgb_camera_matrix = np.array([[646.71386719, 0, 645.97967529], [0, 645.81292725, 373.04150391], [0, 0, 1]], dtype=np.float64)
    rgb_dist_coeffs = np.array([-0.05683016777038574, 0.0678257942199707, 0.00020736592705361545, 0.0006015175022184849, -0.022939417511224747], dtype=np.float64)
    
    depth_camera_matrix = np.array([[653.67602539, 0, 642.39532471], [0, 653.67602539, 365.88021851], [0, 0, 1]], dtype=np.float64)
    depth_dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # 深度相机无畸变
    
    return detector, rgb_camera_matrix, rgb_dist_coeffs, depth_camera_matrix, depth_dist_coeffs

def write_pose_matrix_to_file(pose_matrix, filename="pose_matrix3.txt"):
    try:
        pose_matrix_mm = pose_matrix.copy()
        pose_matrix_mm[:3, 3] *= 1000  
        
        output_lines = []
        
        for i in range(4):
            row_str = " ".join([f"{pose_matrix_mm[i, j]:12.6f}" for j in range(4)])
            output_lines.append(row_str)
        
        output_lines.append("")  
        
        with open(filename, 'a', encoding='utf-8') as f:
            f.write("\n".join(output_lines) + "\n")
        
        try:
            file_size = os.path.getsize(filename)
            if file_size > 10 * 1024 * 1024:  
                with open(filename, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                keep_lines = lines[len(lines)//2:]
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.writelines(keep_lines)
                
                logging.info(f"位姿文件 {filename} 已清理，保留最新数据")
        
        except Exception as e:
            logging.warning(f"文件大小检查失败: {e}")
            
    except Exception as e:
        logging.error(f"写入位姿矩阵到文件失败: {e}")
        raise



def estimate_pose_with_depth(corners, tag_size, camera_matrix, dist_coeffs, depth_frame, depth_scale=0.0001):
    try:
        depths = []
        valid_corners = []
        
        center_x = int(np.mean([corner[0] for corner in corners]))
        center_y = int(np.mean([corner[1] for corner in corners]))
        
        for i, corner in enumerate(corners):
            x, y = int(corner[0]), int(corner[1])
            if 0 <= x < depth_frame.get_width() and 0 <= y < depth_frame.get_height():
                depth_value = depth_frame.get_distance(x, y)  
                if depth_value > 0:  
                    depths.append(depth_value * 1000)  
                    valid_corners.append(corner)
        
        center_depth = None
        if 0 <= center_x < depth_frame.get_width() and 0 <= center_y < depth_frame.get_height():
            center_depth_value = depth_frame.get_distance(center_x, center_y)
            if center_depth_value > 0:
                center_depth = center_depth_value * 1000  
                depths.append(center_depth)
        
        if len(depths) < 2: 
            return None
            
        avg_depth = np.median(depths)
        
        

        half_size = tag_size / 2
        object_points = np.array([
            [-half_size, -half_size, 0],  
            [half_size, -half_size, 0],    
            [half_size, half_size, 0],    
            [-half_size, half_size, 0]    
        ], dtype=np.float64)
        
        image_points = np.array(corners, dtype=np.float64)
        
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
            
        rotation_matrix, _ = cv2.Rodrigues(rvec)
            
        estimated_depth = tvec[2][0]  

        fusion_weight = 0.000000001  
        
        corrected_depth = estimated_depth * (1 - fusion_weight) + avg_depth * fusion_weight
        
        depth_correction_factor = corrected_depth / estimated_depth
        tvec[2][0] = corrected_depth
        
        tvec[0][0] *= depth_correction_factor
        tvec[1][0] *= depth_correction_factor
        
        pose_matrix = np.eye(4, dtype=np.float64)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = tvec.flatten()
        
        return pose_matrix
        
    except Exception as e:
        return None


previous_rotation = None
previous_face_set = set()
rotation_history = []
rotation_direction = None  # 'clockwise', 'counterclockwise', None

def calculate_cube_orientation_from_faces(face_orientations, face_ids, weights):

    global previous_rotation, previous_face_set, rotation_history, rotation_direction
    if not face_orientations:
        return np.eye(3, dtype=np.float64)
    
    current_face_set = set(face_ids)
    

    face_to_cube_transforms = {
        0: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),    
        1: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),    
        2: np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),   
        3: np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),   
        4: np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),   
        5: np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])   
    }
    
    print(f"开始统一坐标系旋转融合，检测到的面: {face_ids}")
    
    unified_rotations = []
    valid_weights = []
    valid_face_ids = []
    
    for i, (face_id, face_rotation) in enumerate(zip(face_ids, face_orientations)):
        if face_id in face_to_cube_transforms:
            transform_matrix = face_to_cube_transforms[face_id]
            unified_rotation = transform_matrix @ face_rotation @ transform_matrix.T
            
            unified_rotations.append(unified_rotation)
            valid_weights.append(weights[i])
            valid_face_ids.append(face_id)
            
            print(f"面{face_id}已转换到统一坐标系，权重={weights[i]:.3f}")
        else:
            print(f"跳过未知面ID {face_id}")
    
    if not unified_rotations:
        print("没有有效的面旋转数据，返回单位矩阵")
        return np.eye(3)
    
    if len(unified_rotations) == 1:
        final_rotation = unified_rotations[0]
        print(f"单面检测：使用面{valid_face_ids[0]}的统一坐标系旋转")
    else:
        print(f"多面融合：{len(unified_rotations)}个面在统一坐标系中融合")
        final_rotation = calculate_weighted_rotation_average(unified_rotations, valid_weights)
    
    if previous_rotation is not None:
        final_rotation = apply_rotation_continuity(final_rotation, previous_rotation)
        print("应用了旋转连续性约束")
    
    previous_rotation = final_rotation.copy()
    previous_face_set = current_face_set.copy()
    
    print(f"统一坐标系旋转融合完成")
    return final_rotation

def detect_rotation_direction(previous_faces, current_faces):
    clockwise_transitions = {
        frozenset([1]): frozenset([2]),  
        frozenset([2]): frozenset([3]),  
        frozenset([3]): frozenset([4]),  
        frozenset([4]): frozenset([1]),  
        frozenset([1, 2]): frozenset([2, 3]),  
        frozenset([2, 3]): frozenset([3, 4]),  
        frozenset([3, 4]): frozenset([4, 1]),  
        frozenset([4, 1]): frozenset([1, 2])   
    }
    
    counterclockwise_transitions = {
        frozenset([1]): frozenset([4]),  
        frozenset([4]): frozenset([3]), 
        frozenset([3]): frozenset([2]),  
        frozenset([2]): frozenset([1]),  
        frozenset([1, 4]): frozenset([4, 3]),  
        frozenset([4, 3]): frozenset([3, 2]),  
        frozenset([3, 2]): frozenset([2, 1]), 
        frozenset([2, 1]): frozenset([1, 4])   
    }
    
    prev_set = frozenset(previous_faces)
    curr_set = frozenset(current_faces)

    if prev_set in clockwise_transitions and clockwise_transitions[prev_set] == curr_set:
        return 'clockwise'
    
    if prev_set in counterclockwise_transitions and counterclockwise_transitions[prev_set] == curr_set:
        return 'counterclockwise'
    
    return None

def handle_top_face_only(top_face_rotation, rotation_direction, previous_rotation):

    face_to_cube_top = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    base_rotation = top_face_rotation @ face_to_cube_top.T
    
    if previous_rotation is None or rotation_direction is None:
        return base_rotation
    
    if rotation_direction == 'clockwise':
        additional_rotation = R.from_euler('y', 2, degrees=True).as_matrix()  
    elif rotation_direction == 'counterclockwise':
        additional_rotation = R.from_euler('y', -2, degrees=True).as_matrix()  
    else:
        additional_rotation = np.eye(3, dtype=np.float64)
    
    adjusted_rotation = base_rotation @ additional_rotation
    
    return adjusted_rotation

def apply_rotation_continuity(current_rotation, previous_rotation, max_angle_diff=30):

    try:
        rotation_diff = previous_rotation.T @ current_rotation
        angle_diff = np.arccos(np.clip((np.trace(rotation_diff) - 1) / 2, -1, 1))
        angle_diff_degrees = np.degrees(angle_diff)
        
        if angle_diff_degrees > max_angle_diff:
            print(f"旋转跳跃过大 ({angle_diff_degrees:.1f}°)，应用平滑处理")
            
            prev_quat = R.from_matrix(previous_rotation).as_quat()
            curr_quat = R.from_matrix(current_rotation).as_quat()
            
            if np.dot(prev_quat, curr_quat) < 0:
                curr_quat = -curr_quat
            
            interpolation_factor = 0.3
            smoothed_quat = (1 - interpolation_factor) * prev_quat + interpolation_factor * curr_quat
            smoothed_quat = smoothed_quat / np.linalg.norm(smoothed_quat)
            
            return R.from_quat(smoothed_quat).as_matrix()
        
        return current_rotation
        
    except Exception as e:
        print(f"连续性处理失败: {e}，使用原始旋转")
        return current_rotation

def calculate_weighted_rotation_average(rotations, weights):

    try:
        quaternions = []
        for rotation in rotations:
            if np.linalg.det(rotation) < 0.5:
                print(f"Warning: 旋转矩阵行列式异常，跳过")
                continue
            
            r = R.from_matrix(rotation)
            quat = r.as_quat()  # [x, y, z, w]
            quaternions.append(quat)
        
        if not quaternions:
            return np.eye(3, dtype=np.float64)
        
        if len(quaternions) == 1:
            return R.from_quat(quaternions[0]).as_matrix()
        
        weights = np.array(weights[:len(quaternions)], dtype=np.float64)
        weights = weights / np.sum(weights)
        
        q_ref = quaternions[0]
        
        aligned_quaternions = [q_ref]
        for q in quaternions[1:]:
            if np.dot(q_ref, q) < 0:
                aligned_quaternions.append(-q)  
            else:
                aligned_quaternions.append(q)
        
        avg_quat = np.zeros(4, dtype=np.float64)
        for q, w in zip(aligned_quaternions, weights):
            avg_quat += w * q
        
        avg_quat = avg_quat / np.linalg.norm(avg_quat)
        
        avg_rotation = R.from_quat(avg_quat).as_matrix()
        
        print(f"四元数加权平均完成，权重分布: {weights}")
        return avg_rotation
        
    except Exception as e:
        print(f"四元数平均计算失败: {e}，使用第一个旋转矩阵")
        return rotations[0]



def calculate_cube_center(tag_poses, cube_size=0.057):
    if not tag_poses:
        return np.eye(4, dtype=np.float64)
    
    print(f"开始统一坐标系位姿融合，检测到的面: {list(tag_poses.keys())}")
    print(f"正方体边长: {cube_size:.3f}m，建立中心坐标系")
    
    unified_centers = []
    unified_orientations = []
    weights = []
    tag_ids = []
    
    L = cube_size / 2
    
    face_info = {
        0: {'name': '上面(Z+)', 'offset': np.array([0, 0, L], dtype=np.float64), 'stability': 1.0, 'priority': 1},     
        1: {'name': '前面(X+)', 'offset': np.array([L, 0, 0], dtype=np.float64), 'stability': 1.0, 'priority': 1},     
        2: {'name': '右面(Y+)', 'offset': np.array([0, L, 0], dtype=np.float64), 'stability': 1.0, 'priority': 1},     
        3: {'name': '后面(X-)', 'offset': np.array([-L, 0, 0], dtype=np.float64), 'stability': 1.0, 'priority': 1},    
        4: {'name': '左面(Y-)', 'offset': np.array([0, -L, 0], dtype=np.float64), 'stability': 1.0, 'priority': 1},    
        5: {'name': '下面(Z-)', 'offset': np.array([0, 0, -L], dtype=np.float64), 'stability': 1.0, 'priority': 1}     
    }
    

    face_to_cube_transforms = {
        0: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),     
        1: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),    
        2: np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),    
        3: np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),    
        4: np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64),    
        5: np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)    
    }
    
    print(f"开始统一坐标系变换，检测到的面: {[face_info.get(tid, {}).get('name', f'未知面{tid}') for tid in tag_poses.keys()]}")
    
    for tag_id, pose_matrix in tag_poses.items():
        if tag_id not in face_info:
            print(f"Warning: 未知面ID {tag_id}，跳过处理")
            continue
            
        face_name = face_info[tag_id]['name']
        
        R_face_camera = pose_matrix[:3, :3] 
        t_face_camera = pose_matrix[:3, 3]  
        
        det_R = np.linalg.det(R_face_camera)
        if abs(det_R - 1.0) > 0.1:
            print(f"Warning: {face_name}的旋转矩阵行列式异常: {det_R:.3f}，跳过")
            continue
        
        if tag_id in face_to_cube_transforms:
            offset_in_face_coords = np.array([0, 0, -L], dtype=np.float64)  
            
            offset_in_camera = R_face_camera @ offset_in_face_coords
            
            cube_center_camera = t_face_camera + offset_in_camera
            
            unified_centers.append(cube_center_camera)
            
            transform_matrix = face_to_cube_transforms[tag_id]
            unified_rotation = transform_matrix @ R_face_camera @ transform_matrix.T
            unified_orientations.append(unified_rotation)
            
            tag_ids.append(tag_id)
            
            distance = np.linalg.norm(t_face_camera)  
            final_weight = 1.0
            weights.append(final_weight)
            
            print(f"{face_name}(ID:{tag_id}): 距离={distance:.3f}m, 中心位置=({cube_center_camera[0]:.3f}, {cube_center_camera[1]:.3f}, {cube_center_camera[2]:.3f}), 权重={final_weight:.3f}")
        else:
            print(f"跳过未知面ID {tag_id}的变换")
    
    if not unified_centers:
        print("没有有效的面数据，返回单位矩阵")
        return np.eye(4, dtype=np.float64)
    
    unified_centers = np.array(unified_centers)
    weights = np.array(weights)
    
    if len(unified_centers) > 2:
        median_center = np.median(unified_centers, axis=0)
        distances_to_median = np.linalg.norm(unified_centers - median_center, axis=1)
        distance_threshold = np.median(distances_to_median) + 2 * np.std(distances_to_median)
        
        valid_indices = distances_to_median <= distance_threshold
        if np.sum(valid_indices) >= 1:  # 至少保留一个点
            unified_centers = unified_centers[valid_indices]
            unified_orientations = [unified_orientations[i] for i in range(len(unified_orientations)) if valid_indices[i]]
            weights = weights[valid_indices]
            tag_ids = [tag_ids[i] for i in range(len(tag_ids)) if valid_indices[i]]
            print(f"异常值检测: 保留{np.sum(valid_indices)}/{len(valid_indices)}个中心点")
        else:
            print("Warning: 所有点都被标记为异常值，保留原始数据")
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)  
    
    avg_center_camera = np.average(unified_centers, axis=0, weights=weights)
    
   
    print(f"\n=== 统一坐标系旋转融合 ===")
    avg_rotation_unified = calculate_cube_orientation_from_faces(
        unified_orientations, tag_ids, weights
    )
    print(f"统一坐标系旋转融合完成，参与的面: {[face_info[tid]['name'] for tid in tag_ids]}")
    
    cube_pose_camera = np.eye(4, dtype=np.float64)
    cube_pose_camera[:3, :3] = avg_rotation_unified
    cube_pose_camera[:3, 3] = avg_center_camera
    
    final_distance = np.linalg.norm(avg_center_camera)
    
    print(f"\n=== 统一坐标系多面融合完成 ===")
    print(f"参与融合的面: {[face_info[tid]['name'] for tid in tag_ids]}")
    print(f"正方体中心位置（相机坐标系）: X={cube_pose_camera[0,3]:.3f}m, Y={cube_pose_camera[1,3]:.3f}m, Z={cube_pose_camera[2,3]:.3f}m")
    
    try:
        rotation_obj = R.from_matrix(avg_rotation_unified)
        euler_angles = rotation_obj.as_euler('xyz', degrees=True)
        print(f"统一坐标系旋转角度: Roll={euler_angles[0]:.1f}°, Pitch={euler_angles[1]:.1f}°, Yaw={euler_angles[2]:.1f}°")
        print(f"坐标系定义: X轴=前面(1号), Y轴=右面(2号), Z轴=上面(0号)")
    except Exception as e:
        print(f"旋转角度计算失败: {e}")
    
    print(f"统一坐标系多面融合算法执行完成\n")
    return cube_pose_camera



def estimate_pose_traditional(corners, tag_size, camera_matrix, dist_coeffs):
    try:
        half_size = tag_size / 2
        object_points = np.array([
            [-half_size, -half_size, 0],  # 左上角
            [half_size, -half_size, 0],   # 右上角  
            [half_size, half_size, 0],    # 右下角
            [-half_size, half_size, 0]    # 左下角
        ], dtype=np.float64)
        
        image_points = np.array(corners, dtype=np.float64)
        
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            print("Warning: PnP求解失败")
            return None
        
        rvec_matrix, _ = cv2.Rodrigues(rvec)
        
        pose_matrix = np.eye(4, dtype=np.float64)
        pose_matrix[:3, :3] = rvec_matrix
        pose_matrix[:3, 3] = tvec.flatten()
        
        return pose_matrix
        
    except Exception as e:
        print(f"传统位姿估计失败: {e}")
        return None

def process_frame(frames, detector, rgb_camera_matrix, rgb_dist_coeffs, depth_camera_matrix, depth_dist_coeffs, tag_size, align, ipc_manager=None):
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    
    if not color_frame or not depth_frame:
        return None, None
    
    color_image = np.asanyarray(color_frame.get_data())
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    height, width = color_image.shape[:2]
    if height != 720 or width != 1280:
        print(f"Warning: 图像尺寸不匹配，期望1280x720，实际{width}x{height}")
    
    frame_info_text = f"RGB: {width}x{height} | Depth: {depth_frame.get_width()}x{depth_frame.get_height()}"
    cv2.putText(color_image, frame_info_text, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    results = detector.detect(gray_image, estimate_tag_pose=False)
    
    tag_poses = {}

    for r in results:
        if 0 <= r.tag_id <= 4:
            depth_enhanced_pose = estimate_pose_with_depth(
                r.corners, tag_size, rgb_camera_matrix, rgb_dist_coeffs, depth_frame
            )
            
            if depth_enhanced_pose is not None:
                pose_matrix = depth_enhanced_pose
                tag_poses[r.tag_id] = pose_matrix
                print(f"Tag {r.tag_id}: 使用深度增强位姿估计（权重0.1）")
            else:
                print(f"Tag {r.tag_id}: 回退到传统PnP位姿估计")
                pose_matrix = estimate_pose_traditional(
                    r.corners, tag_size, rgb_camera_matrix, rgb_dist_coeffs
                )
                
                if pose_matrix is not None:
                    tag_poses[r.tag_id] = pose_matrix
                else:
                    print(f"Tag {r.tag_id}: 位姿估计失败")
                    continue  

    for r in results:
        if 0 <= r.tag_id <= 4 and r.tag_id in tag_poses:
            pose_matrix = tag_poses[r.tag_id]
            
            corners = np.array(r.corners, dtype=np.int32)
            for i in range(4):
                cv2.line(color_image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
            
            cv2.putText(color_image, str(r.tag_id), (int(r.center[0]), int(r.center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            try:
                rvec_matrix = pose_matrix[:3, :3]
                tvec = pose_matrix[:3, 3].reshape(3, 1)
                
                rvec_rodrigues, _ = cv2.Rodrigues(rvec_matrix)
                
                axis_length = tag_size * 1.5  
                
                print(f"Tag {r.tag_id}: 绘制坐标轴，长度={axis_length:.3f}m, 位置=({tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f})m")
                try:
                    cv2.drawFrameAxes(color_image, rgb_camera_matrix, rgb_dist_coeffs, 
                                    rvec_rodrigues, tvec, axis_length)
                except AttributeError:
                    axis_points = np.float64([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]).reshape(-1,3)
                    imgpts, _ = cv2.projectPoints(axis_points, rvec_rodrigues, tvec, rgb_camera_matrix, rgb_dist_coeffs)
                    imgpts = np.int32(imgpts).reshape(-1,2)
                    
                    if len(imgpts) >= 4:
                        origin = tuple(imgpts[0])
                        cv2.arrowedLine(color_image, origin, tuple(imgpts[1]), (0, 0, 255), 3)
                        cv2.arrowedLine(color_image, origin, tuple(imgpts[2]), (0, 255, 0), 3)
                        cv2.arrowedLine(color_image, origin, tuple(imgpts[3]), (255, 0, 0), 3)
                
                axis_points = np.float64([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]).reshape(-1,3)
                imgpts, _ = cv2.projectPoints(axis_points, rvec_rodrigues, tvec, rgb_camera_matrix, rgb_dist_coeffs)
                imgpts = np.int32(imgpts).reshape(-1,2)
                
                if len(imgpts) >= 4:
                    cv2.putText(color_image, 'X', tuple(imgpts[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(color_image, 'Y', tuple(imgpts[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(color_image, 'Z', tuple(imgpts[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
            except Exception as e:
                print(f"绘制Tag {r.tag_id}坐标轴失败: {e}")
                import traceback
                traceback.print_exc()

    if tag_poses:
        cube_pose = calculate_cube_center(tag_poses)
        
        center_text = f"Cube: X={cube_pose[0,3]:.3f}m Y={cube_pose[1,3]:.3f}m Z={cube_pose[2,3]:.3f}m"
        cv2.putText(color_image, center_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        try:
            rotation_obj = R.from_matrix(cube_pose[:3, :3])
            euler_angles = rotation_obj.as_euler('xyz', degrees=True)
            rotation_text = f"Rot: R={euler_angles[0]:.1f} P={euler_angles[1]:.1f} Y={euler_angles[2]:.1f}"
            cv2.putText(color_image, rotation_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            cv2.putText(color_image, "Rot: Error", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        tags_text = f"Tags: {list(tag_poses.keys())}"
        cv2.putText(color_image, tags_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        try:
            write_pose_matrix_to_file(cube_pose)
        except Exception as e:
            logging.error(f"写入位姿矩阵到文件失败: {e}")
        
        if ipc_manager:
            try:
               
                ipc_manager.send_pose_data(cube_pose)
            except Exception as e:
                logging.error(f"发送位姿数据失败: {e}")

    return color_image, tag_poses if tag_poses else None










def main(show_image, enable_ipc=True, ipc_method='shared_memory'):
    
    tag_size = 0.043

    try:
        pipeline, align = setup_realsense_pipeline()
    except Exception as e:
        logging.error(f"相机初始化失败: {e}")
        return
    if USE_PRESET and USE_PRESET in APRILTAG_PRESETS:
        apriltag_params = APRILTAG_PRESETS[USE_PRESET]
        logging.info(f"使用AprilTag预设配置: {USE_PRESET}")
    else:
        apriltag_params = APRILTAG_CONFIG
        logging.info("使用自定义AprilTag配置")
    
    logging.info(f"AprilTag检测参数: {apriltag_params}")
    
    detector, rgb_camera_matrix, rgb_dist_coeffs, depth_camera_matrix, depth_dist_coeffs = setup_apriltag_detector(
        quad_decimate=apriltag_params['quad_decimate'],
        quad_sigma=apriltag_params['quad_sigma'],
        refine_edges=apriltag_params['refine_edges'],
        decode_sharpening=apriltag_params['decode_sharpening']
    )
    
    ipc_manager = None
    if enable_ipc:
        try:
            ipc_manager = IPCManager(method=ipc_method)
        except Exception as e:
            logging.error(f"IPC初始化失败: {e}")
            ipc_manager = None
        
        if ipc_manager:
            connection_info = ipc_manager.get_connection_info()
            print(f"\n=== IPC通信已启动 ===")
            print(f"通信方式: {connection_info.get('method', 'unknown')}")
            if ipc_method == 'shared_memory':
                print(f"共享内存文件: {connection_info.get('file_path', 'unknown')}")
                print(f"缓冲区大小: {connection_info.get('buffer_size', 'unknown')} bytes")
            elif ipc_method == 'tcp_socket':
                print(f"TCP服务器地址: {connection_info.get('host', 'unknown')}:{connection_info.get('port', 'unknown')}")
            print("======================\n")
            logging.info(f"IPC通信已启动: {connection_info}")
        else:
            logging.warning("IPC初始化失败，继续运行但不进行IPC通信")

    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
                if not frames:
                    continue

                color_image, tag_poses = process_frame(frames, detector, rgb_camera_matrix, rgb_dist_coeffs, depth_camera_matrix, depth_dist_coeffs, tag_size, align, ipc_manager)
                
                if show_image and color_image is not None:
                    cv2.imshow('AprilTag Detection', color_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                
                if frame_count % 300 == 0: 次
                    elapsed_time = time.time() - start_time
                    avg_fps = frame_count / elapsed_time
                    logging.info(f"处理了 {frame_count} 帧，平均FPS: {avg_fps:.1f}")
            except Exception as e:
                logging.error(f"帧处理错误: {e}")
                continue

    except KeyboardInterrupt:
        logging.info("程序被用户中断")
    
    except Exception as e:
        logging.error(f"程序运行错误: {e}")
    
    finally:
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()
        
        if ipc_manager:
            ipc_manager.close()
            print("IPC通信已关闭")

        logging.info("Program terminated")

def main_with_mode_selection():
    print(f"显示图像? 按Enter确认, 输入'y'显示, 其他键隐藏: ", end="")
    show_image_input = input().strip().lower()
    show_image = show_image_input == "" or show_image_input == "y"
    
    print(f"启用IPC通信? (y/n, 默认y): ", end="")
    enable_ipc_input = input().strip().lower()
    enable_ipc = enable_ipc_input != "n"
    
    ipc_method = "shared_memory"  
    if enable_ipc:
        print(f"选择IPC通信方式: 1-共享内存(默认), 2-TCP Socket: ", end="")
        ipc_choice = input().strip()
        if ipc_choice == "2":
            ipc_method = "tcp_socket"
        
        print(f"\n将使用 {ipc_method} 方式进行IPC通信\n")
    
    main(show_image=show_image, enable_ipc=enable_ipc, ipc_method=ipc_method)

if __name__ == "__main__":
    mode = config.get('mode', 'normal')
    show_image = config.get('show_image', True)
    enable_ipc = config.get('enable_ipc', True)
    ipc_method = config.get('ipc_method', 'tcp_socket')
    
    print("=== AprilTag实时检测系统 ===")
    print(f"运行模式: {mode}")
    print(f"显示图像: {'是' if show_image else '否'}")
    print(f"IPC通信: {ipc_method if enable_ipc else '禁用'}")
    print("\n按Ctrl+C停止程序\n")
    
    print("快速选项 (可选，直接按Enter使用默认配置):")
    print("1-正常检测(默认) s-共享内存 t-TCP Socket n-禁用IPC m-完整模式选择")
    quick_choice = input("选择: ").strip().lower()
    
    if quick_choice == "s":
        ipc_method = "shared_memory"
        print("切换到: 共享内存IPC通信")
        main(show_image=show_image, enable_ipc=enable_ipc, ipc_method=ipc_method)
    elif quick_choice == "t":
        ipc_method = "tcp_socket"
        print("切换到: TCP Socket IPC通信")
        main(show_image=show_image, enable_ipc=enable_ipc, ipc_method=ipc_method)
    elif quick_choice == "n":
        enable_ipc = False
        print("禁用: IPC通信")
        main(show_image=show_image, enable_ipc=enable_ipc, ipc_method=ipc_method)
    elif quick_choice == "m":
        main_with_mode_selection()
    else:
        if enable_ipc:
            print(f"\n将使用 {ipc_method} 方式进行IPC通信\n")
        main(show_image=show_image, enable_ipc=enable_ipc, ipc_method=ipc_method)
