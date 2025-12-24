import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import os
import logging

class CubePoseEstimator:
    def __init__(self, config):
        self.config = config
        self.cube_size = config['cube']['size']
        self.tag_size = config['apriltag']['size']
        self.rgb_camera_matrix = np.array(config['camera']['rgb']['matrix'], dtype=np.float64)
        self.rgb_dist_coeffs = np.array(config['camera']['rgb']['dist_coeffs'], dtype=np.float64)
        
        self.previous_rotation = None
        self.previous_face_set = set()
        self.rotation_history = []
        self.rotation_direction = None

    def estimate_pose_with_depth(self, corners, depth_frame, depth_scale=0.0001):
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
            half_size = self.tag_size / 2
            object_points = np.array([
                [-half_size, -half_size, 0],
                [half_size, -half_size, 0],
                [half_size, half_size, 0],
                [-half_size, half_size, 0]
            ], dtype=np.float64)
            image_points = np.array(corners, dtype=np.float64)
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points, self.rgb_camera_matrix, self.rgb_dist_coeffs,
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

    def estimate_pose_traditional(self, corners):
        try:
            half_size = self.tag_size / 2
            object_points = np.array([
                [-half_size, -half_size, 0],
                [half_size, -half_size, 0],
                [half_size, half_size, 0],
                [-half_size, half_size, 0]
            ], dtype=np.float64)
            image_points = np.array(corners, dtype=np.float64)
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points, self.rgb_camera_matrix, self.rgb_dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                return None
            rvec_matrix, _ = cv2.Rodrigues(rvec)
            pose_matrix = np.eye(4, dtype=np.float64)
            pose_matrix[:3, :3] = rvec_matrix
            pose_matrix[:3, 3] = tvec.flatten()
            return pose_matrix
        except Exception as e:
            return None

    def calculate_cube_center(self, tag_poses):
        if not tag_poses:
            return np.eye(4, dtype=np.float64)
        unified_centers = []
        unified_orientations = []
        weights = []
        tag_ids = []
        L = self.cube_size / 2
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
        for tag_id, pose_matrix in tag_poses.items():
            if tag_id not in face_info:
                continue
            R_face_camera = pose_matrix[:3, :3]
            t_face_camera = pose_matrix[:3, 3]
            det_R = np.linalg.det(R_face_camera)
            if abs(det_R - 1.0) > 0.1:
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
                weights.append(1.0)
        if not unified_centers:
            return np.eye(4, dtype=np.float64)
        unified_centers = np.array(unified_centers)
        weights = np.array(weights)
        if len(unified_centers) > 2:
            median_center = np.median(unified_centers, axis=0)
            distances_to_median = np.linalg.norm(unified_centers - median_center, axis=1)
            distance_threshold = np.median(distances_to_median) + 2 * np.std(distances_to_median)
            valid_indices = distances_to_median <= distance_threshold
            if np.sum(valid_indices) >= 1:
                unified_centers = unified_centers[valid_indices]
                unified_orientations = [unified_orientations[i] for i in range(len(unified_orientations)) if valid_indices[i]]
                weights = weights[valid_indices]
                tag_ids = [tag_ids[i] for i in range(len(tag_ids)) if valid_indices[i]]
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        avg_center_camera = np.average(unified_centers, axis=0, weights=weights)
        avg_rotation_unified = self._calculate_cube_orientation_from_faces(
            unified_orientations, tag_ids, weights
        )
        cube_pose_camera = np.eye(4, dtype=np.float64)
        cube_pose_camera[:3, :3] = avg_rotation_unified
        cube_pose_camera[:3, 3] = avg_center_camera
        
        print(f"最终定位位姿 (多面体融合后):")
        print(f"位置 (X, Y, Z): [{avg_center_camera[0]:.6f}, {avg_center_camera[1]:.6f}, {avg_center_camera[2]:.6f}]")
        print(f"旋转矩阵:")
        for i in range(3):
            print(f"  [{avg_rotation_unified[i, 0]:.6f}, {avg_rotation_unified[i, 1]:.6f}, {avg_rotation_unified[i, 2]:.6f}]")
            
        return cube_pose_camera

    def _calculate_cube_orientation_from_faces(self, face_orientations, face_ids, weights):
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
        if not unified_rotations:
            return np.eye(3)
        if len(unified_rotations) == 1:
            final_rotation = unified_rotations[0]
        else:
            final_rotation = self._calculate_weighted_rotation_average(unified_rotations, valid_weights)
        if self.previous_rotation is not None:
            final_rotation = self._apply_rotation_continuity(final_rotation, self.previous_rotation)
        self.previous_rotation = final_rotation.copy()
        self.previous_face_set = current_face_set.copy()
        return final_rotation

    def _apply_rotation_continuity(self, current_rotation, previous_rotation, max_angle_diff=30):
        try:
            rotation_diff = previous_rotation.T @ current_rotation
            angle_diff = np.arccos(np.clip((np.trace(rotation_diff) - 1) / 2, -1, 1))
            angle_diff_degrees = np.degrees(angle_diff)
            if angle_diff_degrees > max_angle_diff:
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
            return current_rotation

    def _calculate_weighted_rotation_average(self, rotations, weights):
        try:
            quaternions = []
            for rotation in rotations:
                if np.linalg.det(rotation) < 0.5:
                    continue
                r = R.from_matrix(rotation)
                quat = r.as_quat()
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
            return avg_rotation
        except Exception as e:
            return rotations[0]
            
    def write_pose_matrix_to_file(self, pose_matrix, filename="pose_matrix3.txt"):
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

    def _detect_rotation_direction(self, previous_faces, current_faces):
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

    def _handle_top_face_only(self, top_face_rotation, rotation_direction, previous_rotation):
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
