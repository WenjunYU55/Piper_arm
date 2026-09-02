#!/usr/bin/env python3
import json
import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from piper_mobile_manipulation.msg import Detection2D, Target3D
from piper_mobile_manipulation.perception.target_envelope import (
    clipped_shape_rejection,
    TargetSilhouetteClippedError,
    trusted_silhouette_measurement,
)
from piper_mobile_manipulation.utils.target_depth import (
    select_target_depth_component,
)


def depth_jump_reacquisition(
        previous_depth, candidate_depth, maximum_jump, pending_depth,
        pending_count, required_samples, consistency_tolerance):
    """Reject one-off depth jumps but accept a short consistent new depth."""
    if (
            previous_depth is None
            or maximum_jump <= 0.0
            or abs(candidate_depth - previous_depth) <= maximum_jump):
        return True, None, 0, False

    required = max(int(required_samples), 1)
    tolerance = max(float(consistency_tolerance), 0.0)
    if (
            pending_depth is None
            or abs(candidate_depth - pending_depth) > tolerance):
        if required == 1:
            return True, None, 0, True
        return False, float(candidate_depth), 1, False

    count = int(pending_count) + 1
    mean_depth = (
        float(pending_depth) * float(max(count - 1, 1))
        + float(candidate_depth)
    ) / float(count)
    if count >= required:
        return True, None, 0, True
    return False, mean_depth, count, False


def primary_depth_component(
        valid, depth_m, seed_u, seed_v, absolute_band_m,
        neighbour_jump_m):
    """Compatibility wrapper for the shared seed-independent selector."""
    support = np.asarray(valid, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    if support.shape != depth.shape or support.ndim != 2:
        raise ValueError('valid support and depth must be matching 2D arrays')
    if not np.any(support):
        return np.zeros_like(support), math.nan
    band = max(float(absolute_band_m), 0.0)
    if band <= 0.0:
        rows, columns = np.nonzero(support)
        nearest = int(np.argmin(
            (columns - float(seed_u)) ** 2 + (rows - float(seed_v)) ** 2))
        return support.copy(), float(depth[rows[nearest], columns[nearest]])
    selected, report = select_target_depth_component(
        support, depth, center_u=seed_u, center_v=seed_v,
        depth_band_m=band,
        minimum_peak_separation_m=max(float(neighbour_jump_m) * 2.0, 0.025),
        minimum_points=1, minimum_support_fraction=0.0,
        ambiguity_margin=0.0)
    return selected, float(report['selected_depth_m'])


def median_component_camera_point(depth_m, support, camera_matrix):
    """
    Return the robust 3-D centre of one qualified visible component.

    The previous implementation selected a robust Z value but projected it
    through the 2-D detector centre.  A slightly asymmetric SAM mask could
    therefore move X/Y independently of the depth samples that had actually
    passed qualification.  Use the same qualified pixels for all three axes.
    This remains a visible-surface centroid, not a fitted cube centre.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    selected = np.asarray(support, dtype=bool)
    intrinsic = np.asarray(camera_matrix, dtype=np.float64).reshape(-1)
    if depth.ndim != 2 or selected.shape != depth.shape:
        raise ValueError('qualified depth and support must be matching 2D arrays')
    if intrinsic.size != 9 or not np.all(np.isfinite(intrinsic)):
        raise ValueError('camera matrix must contain nine finite values')
    fx, fy, cx, cy = (
        float(intrinsic[0]), float(intrinsic[4]),
        float(intrinsic[2]), float(intrinsic[5]))
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError('camera focal lengths must be positive')
    rows, columns = np.nonzero(
        selected & np.isfinite(depth) & (depth > 0.0))
    if rows.size == 0:
        raise ValueError('qualified target component is empty')
    z = depth[rows, columns]
    points = np.column_stack((
        (columns.astype(np.float64) - cx) * z / fx,
        (rows.astype(np.float64) - cy) * z / fy,
        z,
    ))
    point = np.median(points, axis=0)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError('qualified target component centroid is non-finite')
    return point


class DepthTo3DNode(Node):
    def __init__(self):
        super().__init__('depth_to_3d_node')
        self.declare_parameter('detection_topic', '/piper/sam2_detection_2d')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('mask_topic', '/piper/sam2_target_mask')
        self.declare_parameter('target_topic', '/piper/target_3d')
        self.declare_parameter(
            'target_shape_topic', '/piper/target_shape_measurement')
        self.declare_parameter('depth_min_m', 0.25)
        self.declare_parameter('depth_max_m', 3.0)
        self.declare_parameter('min_depth_m', 0.25)
        self.declare_parameter('max_depth_m', 3.0)
        self.declare_parameter('use_median_depth', True)
        self.declare_parameter('min_valid_depth_ratio', 0.4)
        self.declare_parameter('min_valid_depth_pixels', 20)
        self.declare_parameter('crop_half_size_px', 8)
        self.declare_parameter('roi_half_size_px', 10)
        self.declare_parameter('use_detection_bbox', True)
        self.declare_parameter('bbox_scale', 0.8)
        self.declare_parameter('depth_percentile', 50.0)
        self.declare_parameter('max_depth_stddev_m', 0.08)
        self.declare_parameter('confidence_depth_stddev_m', 0.15)
        self.declare_parameter('max_depth_jump_m', 0.20)
        self.declare_parameter('depth_jump_reacquire_samples', 3)
        self.declare_parameter('depth_jump_reacquire_tolerance_m', 0.03)
        self.declare_parameter('smoothing_alpha', 0.2)
        self.declare_parameter('use_mask_depth', True)
        self.declare_parameter('mask_max_age_s', 0.20)
        self.declare_parameter('mask_erode_px', 2)
        self.declare_parameter('primary_depth_band_m', 0.03)
        self.declare_parameter('primary_depth_neighbour_jump_m', 0.015)
        self.declare_parameter('primary_depth_minimum_fraction', 0.15)
        self.declare_parameter('primary_depth_ambiguity_margin', 0.08)
        self.declare_parameter('debug', True)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop_sec', 0.08)

        self.bridge = CvBridge()
        self.latest_mask_msg = None
        self.previous_depth = None
        self.pending_jump_depth = None
        self.pending_jump_count = 0
        self.refresh_runtime_params()
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value)
        self.sync_slop_sec = float(self.get_parameter('sync_slop_sec').value)

        self.pub = self.create_publisher(Target3D, self.get_parameter('target_topic').value, 10)
        self.shape_pub = self.create_publisher(
            String, self.get_parameter('target_shape_topic').value,
            qos_profile_sensor_data)
        self.det_sub = Subscriber(
            self, Detection2D, self.get_parameter('detection_topic').value, qos_profile=10
        )
        self.depth_sub = Subscriber(
            self, Image, self.get_parameter('depth_topic').value,
            qos_profile=qos_profile_sensor_data
        )
        self.info_sub = Subscriber(
            self, CameraInfo, self.get_parameter('camera_info_topic').value,
            qos_profile=qos_profile_sensor_data
        )
        self.mask_sub = self.create_subscription(
            Image,
            self.get_parameter('mask_topic').value,
            self.mask_cb,
            qos_profile_sensor_data,
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.det_sub, self.depth_sub, self.info_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_sec,
        )
        self.sync.registerCallback(self.synced_cb)
        self.get_logger().info(
            'Depth projection synchronizing Detection2D, depth image, and CameraInfo slop=%.3fs'
            % self.sync_slop_sec
        )

    def mask_cb(self, mask_msg):
        self.latest_mask_msg = mask_msg

    def synced_cb(self, detection_msg, depth_msg, camera_info):
        self.refresh_runtime_params()
        out = Target3D()
        out.header = depth_msg.header
        out.source_u = float(detection_msg.u)
        out.source_v = float(detection_msg.v)
        out.detection_width = float(detection_msg.width)
        out.detection_height = float(detection_msg.height)
        if not detection_msg.valid:
            self.previous_depth = None
            self.pending_jump_depth = None
            self.pending_jump_count = 0
            out.valid = False
            self.pub.publish(out)
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as exc:
            out.valid = False
            self.pub.publish(out)
            self.get_logger().warn('depth cv_bridge failed: %s' % exc)
            return

        depth_m = self.depth_image_to_meters(depth, depth_msg.encoding)
        h, w = depth_m.shape[:2]
        u = int(round(detection_msg.u))
        v = int(round(detection_msg.v))
        if u < 0 or u >= w or v < 0 or v >= h:
            out.valid = False
            self.pub.publish(out)
            self.get_logger().warn(
                'Detection rejected outside depth image u=%d v=%d size=%dx%d'
                % (u, v, w, h))
            return

        mask, mask_rejection = self.mask_for_depth(depth_msg, w, h)
        if mask is None:
            # Target3D is the semantic target-measurement path.  A rectangle
            # around a stale/missing mask can contain a very clean background
            # depth layer and must never be reinterpreted as the target.
            out.depth_source = mask_rejection
            out.valid = False
            self.pub.publish(out)
            self.log_debug(
                'Target3D rejected before depth projection: %s'
                % mask_rejection,
                warn=True,
            )
            return

        x0, x1, y0, y1 = self.bbox_bounds(detection_msg, u, v, w, h)
        roi_mask = mask[y0:y1, x0:x1] > 0
        crop = depth_m[y0:y1, x0:x1]
        valid = (
            roi_mask & np.isfinite(crop)
            & (crop > self.depth_min) & (crop < self.depth_max))
        out.depth_source = 'mask'

        out.roi_width = float(x1 - x0)
        out.roi_height = float(y1 - y0)
        raw_valid_count = int(np.count_nonzero(valid))
        total_count = int(crop.size)
        raw_ratio = float(raw_valid_count) / float(max(total_count, 1))
        out.valid_depth_ratio = raw_ratio

        if (
                raw_valid_count < self.min_valid_depth_pixels
                or raw_ratio < self.min_ratio):
            out.valid = False
            self.pub.publish(out)
            self.log_debug(
                'Depth rejected source=%s valid_pixels=%d ratio=%.2f roi=%dx%d'
                % (out.depth_source, raw_valid_count, raw_ratio,
                   x1 - x0, y1 - y0),
                warn=True,
            )
            return

        try:
            valid, component_report = select_target_depth_component(
                valid, crop,
                center_u=float(detection_msg.u) - float(x0),
                center_v=float(detection_msg.v) - float(y0),
                depth_band_m=self.primary_depth_band_m,
                minimum_peak_separation_m=max(
                    self.primary_depth_neighbour_jump_m * 2.0, 0.025),
                minimum_points=self.min_valid_depth_pixels,
                minimum_support_fraction=(
                    self.primary_depth_minimum_fraction),
                ambiguity_margin=self.primary_depth_ambiguity_margin,
                preferred_depth_m=self.previous_depth,
            )
        except ValueError as exc:
            out.valid = False
            self.pub.publish(out)
            self.log_debug(
                'Depth rejected source=%s layer_selection=%s'
                % (out.depth_source, exc), warn=True)
            return
        valid_count = int(np.count_nonzero(valid))
        primary_fraction = float(
            component_report['selected_support_fraction'])
        ratio = float(valid_count) / float(max(total_count, 1))
        out.valid_depth_ratio = ratio

        if valid_count < self.min_valid_depth_pixels or ratio < self.min_ratio:
            out.valid = False
            self.pub.publish(out)
            self.log_debug(
                'Depth rejected source=%s primary_component=%d/%d roi=%dx%d'
                % (out.depth_source, valid_count, raw_valid_count,
                   x1 - x0, y1 - y0),
                warn=True,
            )
            return

        valid_depths = crop[valid]
        depth_stddev = float(np.std(valid_depths))
        out.depth_stddev = depth_stddev
        if self.max_depth_stddev > 0.0 and depth_stddev > self.max_depth_stddev:
            out.valid = False
            self.pub.publish(out)
            self.get_logger().warn(
                'Depth rejected stddev=%.3f roi=%dx%d ratio=%.2f'
                % (depth_stddev, x1 - x0, y1 - y0, ratio)
            )
            return

        if self.use_median_depth:
            z = float(np.median(valid_depths))
        else:
            percentile = float(np.clip(self.depth_percentile, 0.0, 100.0))
            z = float(np.percentile(valid_depths, percentile))
        accept_depth, self.pending_jump_depth, self.pending_jump_count, _resynced = \
            depth_jump_reacquisition(
                self.previous_depth,
                z,
                self.max_depth_jump,
                self.pending_jump_depth,
                self.pending_jump_count,
                self.depth_jump_reacquire_samples,
                self.depth_jump_reacquire_tolerance,
            )
        if not accept_depth:
            out.valid = False
            self.pub.publish(out)
            self.log_debug(
                'Depth jump awaiting consistent reacquisition %.3fm -> %.3fm '
                'max=%.3fm sample=%d/%d'
                % (
                    self.previous_depth,
                    z,
                    self.max_depth_jump,
                    self.pending_jump_count,
                    self.depth_jump_reacquire_samples,
                ),
                warn=True,
            )
            return
        try:
            new_point = median_component_camera_point(
                crop, valid, [
                    float(camera_info.k[0]), 0.0,
                    float(camera_info.k[2]) - float(x0),
                    0.0, float(camera_info.k[4]),
                    float(camera_info.k[5]) - float(y0),
                    0.0, 0.0, 1.0,
                ])
        except (TypeError, ValueError) as exc:
            out.valid = False
            self.pub.publish(out)
            self.log_debug(
                'Depth rejected qualified component geometry: %s' % exc,
                warn=True)
            return
        # Do not average camera-frame coordinates across eye-in-hand motion.
        # The timestamped base-frame Kalman tracker is the sole temporal
        # estimator; blending here would mix coordinates expressed at
        # different camera poses and create an artificial target displacement.
        filtered_point = new_point
        self.previous_depth = z

        out.point.x = float(filtered_point[0])
        out.point.y = float(filtered_point[1])
        out.point.z = float(filtered_point[2])
        out.depth = float(filtered_point[2])
        out.measurement_confidence = self.measurement_confidence(
            detection_msg.confidence, valid_count, depth_stddev)
        out.valid = True
        try:
            shape = trusted_silhouette_measurement(
                mask,
                valid,
                (x0, y0),
                crop,
                np.asarray(camera_info.k, dtype=float).reshape(3, 3),
                depth_msg.header,
                out.measurement_confidence,
            )
        except TargetSilhouetteClippedError as exc:
            rejection_msg = String()
            rejection_msg.data = json.dumps(
                clipped_shape_rejection(depth_msg.header, exc.near_depth_m),
                sort_keys=True)
            # Publish before Target3D so the planner can correlate the exact
            # source stamp and fail without freezing an undersized envelope.
            self.shape_pub.publish(rejection_msg)
            self.log_debug(
                'Target shape measurement rejected: %s' % exc,
                warn=True)
        except (TypeError, ValueError) as exc:
            self.log_debug(
                'Target shape measurement rejected: %s' % exc,
                warn=True)
        else:
            shape_msg = String()
            shape_msg.data = json.dumps(shape, sort_keys=True)
            # Publish first so a tracked target carrying the same source stamp
            # can freeze this private planning measurement without a race.
            self.shape_pub.publish(shape_msg)
        self.pub.publish(out)
        self.log_debug(
            'Target3D source=%s camera_frame=(%.3f, %.3f, %.3f) '
            'raw_z=%.3f ratio=%.2f valid=%d primary=%.2f roi=%dx%d '
            'std=%.3f det_conf=%.2f depth_conf=%.2f conf=%.2f'
            % (
                out.depth_source,
                out.point.x,
                out.point.y,
                out.point.z,
                z,
                ratio,
                valid_count,
                primary_fraction,
                x1 - x0,
                y1 - y0,
                depth_stddev,
                detection_msg.confidence,
                self.depth_confidence(valid_count, depth_stddev),
                out.measurement_confidence,
            )
        )

    def bbox_bounds(self, detection_msg, u, v, image_width, image_height):
        det_width = max(float(detection_msg.width), 1.0)
        det_height = max(float(detection_msg.height), 1.0)
        half_w = max(1, int(round(det_width * 0.5)))
        half_h = max(1, int(round(det_height * 0.5)))
        return (
            max(0, u - half_w),
            min(image_width, u + half_w + 1),
            max(0, v - half_h),
            min(image_height, v + half_h + 1),
        )

    def depth_roi(self, detection_msg, u, v, image_width, image_height):
        if not self.use_detection_bbox:
            return (
                max(0, u - self.crop_half),
                min(image_width, u + self.crop_half + 1),
                max(0, v - self.crop_half),
                min(image_height, v + self.crop_half + 1),
            )

        det_width = max(float(detection_msg.width), 1.0)
        det_height = max(float(detection_msg.height), 1.0)
        scale = float(np.clip(self.bbox_scale, 0.05, 1.0))
        half_w = max(1, int(round(det_width * scale * 0.5)))
        half_h = max(1, int(round(det_height * scale * 0.5)))
        return (
            max(0, u - half_w),
            min(image_width, u + half_w + 1),
            max(0, v - half_h),
            min(image_height, v + half_h + 1),
        )

    def mask_for_depth(self, depth_msg, image_width, image_height):
        """Return one fresh, frame-correlated nonempty semantic target mask."""
        if not self.use_mask_depth:
            return None, 'mask_depth_disabled'
        if self.latest_mask_msg is None:
            return None, 'mask_missing'
        age = abs(
            (
                self.stamp_to_seconds(depth_msg.header.stamp)
                - self.stamp_to_seconds(self.latest_mask_msg.header.stamp)
            )
        )
        if age > self.mask_max_age_s:
            return None, 'mask_stale'
        depth_frame = str(getattr(depth_msg.header, 'frame_id', ''))
        mask_frame = str(getattr(self.latest_mask_msg.header, 'frame_id', ''))
        if depth_frame and mask_frame and depth_frame != mask_frame:
            return None, 'mask_frame_mismatch'
        try:
            mask = self.bridge.imgmsg_to_cv2(self.latest_mask_msg, desired_encoding='mono8')
        except Exception as exc:
            self.log_debug('mask cv_bridge failed: %s' % exc, warn=True)
            return None, 'mask_decode_failed'
        if mask.shape[0] != image_height or mask.shape[1] != image_width:
            return None, 'mask_shape_mismatch'
        if self.mask_erode_px > 0:
            import cv2

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * self.mask_erode_px + 1, 2 * self.mask_erode_px + 1)
            )
            mask = cv2.erode(mask, kernel)
        if int(np.count_nonzero(mask)) == 0:
            return None, 'mask_empty'
        return mask, ''

    @staticmethod
    def stamp_to_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def depth_image_to_meters(depth, encoding):
        arr = np.asarray(depth)
        if encoding in ('16UC1', 'mono16'):
            return arr.astype(np.float32) * 0.001
        return arr.astype(np.float32)

    def measurement_confidence(self, detection_confidence, valid_count, depth_stddev):
        return float(
            np.clip(detection_confidence, 0.0, 1.0)
            * self.depth_confidence(valid_count, depth_stddev))

    def depth_confidence(self, valid_count, depth_stddev):
        pixel_quality = float(np.clip(
            valid_count / float(max(self.min_valid_depth_pixels * 2, 1)),
            0.0, 1.0))
        std_quality = 1.0
        if self.confidence_depth_stddev > 0.0:
            std_ratio = max(float(depth_stddev), 0.0) / self.confidence_depth_stddev
            std_quality = 1.0 / (1.0 + std_ratio * std_ratio)
        return float(np.clip(0.35 + 0.65 * pixel_quality * std_quality, 0.0, 1.0))

    def refresh_runtime_params(self):
        self.depth_min = float(self.get_parameter('min_depth_m').value)
        self.depth_max = float(self.get_parameter('max_depth_m').value)
        self.use_median_depth = bool(self.get_parameter('use_median_depth').value)
        self.min_ratio = float(self.get_parameter('min_valid_depth_ratio').value)
        self.min_valid_depth_pixels = int(self.get_parameter('min_valid_depth_pixels').value)
        self.crop_half = int(self.get_parameter('roi_half_size_px').value)
        self.use_detection_bbox = bool(self.get_parameter('use_detection_bbox').value)
        self.bbox_scale = float(self.get_parameter('bbox_scale').value)
        self.depth_percentile = float(self.get_parameter('depth_percentile').value)
        self.max_depth_stddev = float(self.get_parameter('max_depth_stddev_m').value)
        self.confidence_depth_stddev = float(self.get_parameter('confidence_depth_stddev_m').value)
        self.max_depth_jump = float(self.get_parameter('max_depth_jump_m').value)
        self.depth_jump_reacquire_samples = max(
            1, int(self.get_parameter('depth_jump_reacquire_samples').value))
        self.depth_jump_reacquire_tolerance = max(
            0.0,
            float(self.get_parameter(
                'depth_jump_reacquire_tolerance_m').value),
        )
        self.use_mask_depth = bool(self.get_parameter('use_mask_depth').value)
        self.mask_max_age_s = float(self.get_parameter('mask_max_age_s').value)
        self.mask_erode_px = max(0, int(self.get_parameter('mask_erode_px').value))
        self.primary_depth_band_m = max(
            0.0, float(self.get_parameter('primary_depth_band_m').value))
        self.primary_depth_neighbour_jump_m = max(
            0.0,
            float(self.get_parameter(
                'primary_depth_neighbour_jump_m').value),
        )
        self.primary_depth_minimum_fraction = float(np.clip(
            self.get_parameter('primary_depth_minimum_fraction').value,
            0.0, 1.0))
        self.primary_depth_ambiguity_margin = max(
            0.0, float(self.get_parameter(
                'primary_depth_ambiguity_margin').value))
        self.debug = bool(self.get_parameter('debug').value)

    def log_debug(self, message, warn=False):
        if warn:
            self.get_logger().warn(message)
        elif self.debug:
            self.get_logger().info(message)


def main(args=None):
    rclpy.init(args=args)
    node = DepthTo3DNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
