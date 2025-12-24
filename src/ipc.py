import mmap
import os
import json
import struct
import socket
import threading
import time
import logging
import traceback
from .utils import retry_on_error

class IPCManager:
    def __init__(self, config, method='shared_memory'):
        self.config = config
        self.method = method
        self.buffer_size = config['ipc']['shared_memory']['buffer_size']
        self.is_running = False
        self.lock = threading.Lock()
        self.connection_lock = threading.Lock()
        self.shared_memory_file = None
        self.memory_map = None
        self.tcp_socket = None
        self.tcp_port = config['ipc']['tcp_socket']['port']
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
            self.shared_memory_file = self.config['ipc']['shared_memory']['file_name']
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
        while self.is_running:
            try:
                self.tcp_socket.settimeout(1.0)
                client_socket, addr = self.tcp_socket.accept()
                logging.info(f"客户端连接: {addr}")
                with self.connection_lock:
                    if not hasattr(self, 'client_connections'):
                        self.client_connections = []
                    self.client_connections.append((client_socket, addr))
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    logging.error(f"TCP服务器错误: {e}")

    def send_pose_data(self, cube_pose_matrix, timestamp=None):
        if not self.is_running:
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
                    except Exception as e:
                        logging.error(f"关闭断开的客户端连接时出错: {e}")
                    del self.client_connections[idx]
        except Exception as e:
            logging.error(f"TCP发送失败: {e}")
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
