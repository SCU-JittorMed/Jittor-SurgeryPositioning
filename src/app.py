import time
import cv2
import numpy as np
import pupil_apriltags
import logging
from scipy.spatial.transform import Rotation as R
from .camera import setup_realsense_pipeline
from .ipc import IPCManager
from .pose import CubePoseEstimator
from .utils import setup_logging

class PositioningApp:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.align = None
        self.detector = None
        self.ipc_manager = None
        self.pose_estimator = None
        self.is_running = False

    def setup(self):
        setup_logging(self.config)
        try:
            self.pipeline, self.align = setup_realsense_pipeline(self.config)
        except Exception as e:
            logging.error(f"相机初始化失败: {e}")
            raise

        self.detector = pupil_apriltags.Detector(
            families=self.config['apriltag']['families'],
            quad_decimate=self.config['apriltag']['quad_decimate'],
            quad_sigma=self.config['apriltag']['quad_sigma'],
            refine_edges=self.config['apriltag']['refine_edges'],
            decode_sharpening=self.config['apriltag']['decode_sharpening']
        )
        
        self.pose_estimator = CubePoseEstimator(self.config)

    def run(self, show_image=True, enable_ipc=True, ipc_method='shared_memory'):
        if enable_ipc:
            try:
                self.ipc_manager = IPCManager(self.config, method=ipc_method)
                connection_info = self.ipc_manager.get_connection_info()
                logging.info(f"IPC通信已启动: {connection_info}")
            except Exception as e:
                logging.error(f"IPC初始化失败: {e}")
                self.ipc_manager = None
        else:
            self.ipc_manager = None

        self.is_running = True
        frame_count = 0
        start_time = time.time()
        prev_frame_time = time.time()
        fps = 0.0

        try:
            while self.is_running:
                try:
                    curr_frame_time = time.time()
                    time_diff = curr_frame_time - prev_frame_time
                    if time_diff > 0:
                        fps = 1.0 / time_diff
                    prev_frame_time = curr_frame_time
                    
                    frames = self.pipeline.wait_for_frames()
                    if not frames:
                        continue
                    
                    color_image, tag_poses = self.process_frame(frames)
                    
                    if show_image and color_image is not None:
                        height, width = color_image.shape[:2]
                        cv2.putText(color_image, f"FPS: {fps:.1f}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.imshow('AprilTag Detection', color_image)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    frame_count += 1
                    if frame_count % 100 == 0:
                        elapsed_time = time.time() - start_time
                        avg_fps = frame_count / elapsed_time
                        logging.info(f"Frame: {frame_count}, Instant FPS: {fps:.1f}, Avg FPS: {avg_fps:.1f}")
                except Exception as e:
                    logging.error(f"帧处理错误: {e}")
                    continue
        except KeyboardInterrupt:
            logging.info("程序被用户中断")
        except Exception as e:
            logging.error(f"程序运行错误: {e}")
        finally:
            self.stop()

    def stop(self):
        self.is_running = False
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()
        if self.ipc_manager:
            self.ipc_manager.close()
        logging.info("Program terminated")

    def process_frame(self, frames):
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
            
        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        height, width = color_image.shape[:2]
        frame_info_text = f"RGB: {width}x{height} | Depth: {depth_frame.get_width()}x{depth_frame.get_height()}"
        cv2.putText(color_image, frame_info_text, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        results = self.detector.detect(gray_image, estimate_tag_pose=False)
        tag_poses = {}
        
        for r in results:
            if 0 <= r.tag_id <= 4:
                depth_enhanced_pose = self.pose_estimator.estimate_pose_with_depth(
                    r.corners, depth_frame
                )
                if depth_enhanced_pose is not None:
                    tag_poses[r.tag_id] = depth_enhanced_pose
                else:
                    pose_matrix = self.pose_estimator.estimate_pose_traditional(r.corners)
                    if pose_matrix is not None:
                        tag_poses[r.tag_id] = pose_matrix
                    else:
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
                    axis_length = self.pose_estimator.tag_size * 1.5
                    rgb_camera_matrix = self.pose_estimator.rgb_camera_matrix
                    rgb_dist_coeffs = self.pose_estimator.rgb_dist_coeffs
                    
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
                    pass
                    
        if tag_poses:
            cube_pose = self.pose_estimator.calculate_cube_center(tag_poses)
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
                self.pose_estimator.write_pose_matrix_to_file(cube_pose)
            except Exception as e:
                logging.error(f"写入位姿矩阵到文件失败: {e}")
                
            if self.ipc_manager:
                try:
                    self.ipc_manager.send_pose_data(cube_pose)
                except Exception as e:
                    logging.error(f"发送位姿数据失败: {e}")
                    
        return color_image, tag_poses if tag_poses else None
