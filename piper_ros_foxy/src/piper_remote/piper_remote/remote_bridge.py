#!/usr/bin/env python3
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from piper_msgs.msg import PosCmd


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def require_number(data: Dict[str, Any], name: str, default: float = 0.0) -> float:
    value = data.get(name, default)
    if not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    return float(value)


class PiperRemoteBridge(Node):
    def __init__(self) -> None:
        super().__init__('piper_remote_bridge')

        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 8080)
        self.declare_parameter('max_linear_m', 0.6)
        self.declare_parameter('min_z_m', 0.02)
        self.declare_parameter('max_z_m', 0.8)
        self.declare_parameter('max_rotation_rad', 3.14159)
        self.declare_parameter('max_gripper_m', 0.08)
        self.declare_parameter('default_joint_speed', 30.0)
        self.declare_parameter('default_gripper_effort', 1.0)

        self.host = self.get_parameter('host').get_parameter_value().string_value
        self.port = self.get_parameter('port').get_parameter_value().integer_value
        self.max_linear_m = self.get_parameter('max_linear_m').get_parameter_value().double_value
        self.min_z_m = self.get_parameter('min_z_m').get_parameter_value().double_value
        self.max_z_m = self.get_parameter('max_z_m').get_parameter_value().double_value
        self.max_rotation_rad = self.get_parameter('max_rotation_rad').get_parameter_value().double_value
        self.max_gripper_m = self.get_parameter('max_gripper_m').get_parameter_value().double_value
        self.default_joint_speed = self.get_parameter('default_joint_speed').get_parameter_value().double_value
        self.default_gripper_effort = self.get_parameter('default_gripper_effort').get_parameter_value().double_value

        self.pos_pub = self.create_publisher(PosCmd, 'pos_cmd', 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_ctrl_single', 10)
        self.enable_pub = self.create_publisher(Bool, 'enable_flag', 10)

        self.httpd = self._make_server()
        self.http_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.http_thread.start()
        self.get_logger().info(f'Piper remote bridge listening on http://{self.host}:{self.port}')

    def destroy_node(self) -> bool:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.http_thread.join(timeout=2.0)
        return super().destroy_node()

    def _make_server(self) -> ThreadingHTTPServer:
        bridge = self

        class RequestHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                bridge.get_logger().info('%s - %s' % (self.address_string(), fmt % args))

            def do_OPTIONS(self) -> None:
                self._send_json({'ok': True})

            def do_GET(self) -> None:
                if self.path == '/health':
                    self._send_json({'ok': True, 'node': bridge.get_name()})
                    return
                self._send_json({
                    'ok': True,
                    'endpoints': ['/health', '/enable', '/disable', '/pose', '/joints'],
                })

            def do_POST(self) -> None:
                try:
                    data = self._read_json()
                    if self.path == '/enable':
                        bridge.publish_enable(True)
                        self._send_json({'ok': True, 'enabled': True})
                    elif self.path == '/disable':
                        bridge.publish_enable(False)
                        self._send_json({'ok': True, 'enabled': False})
                    elif self.path == '/pose':
                        msg = bridge.publish_pose(data)
                        self._send_json({'ok': True, 'published': bridge.pose_to_dict(msg)})
                    elif self.path == '/joints':
                        msg = bridge.publish_joints(data)
                        self._send_json({'ok': True, 'published': bridge.joints_to_dict(msg)})
                    else:
                        self._send_json({'ok': False, 'error': 'unknown endpoint'}, status=404)
                except ValueError as exc:
                    self._send_json({'ok': False, 'error': str(exc)}, status=400)
                except Exception as exc:
                    bridge.get_logger().error(f'HTTP bridge request failed: {exc}')
                    self._send_json({'ok': False, 'error': str(exc)}, status=500)

            def _read_json(self) -> Dict[str, Any]:
                length = int(self.headers.get('Content-Length', '0'))
                if length == 0:
                    return {}
                raw = self.rfile.read(length).decode('utf-8')
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError('JSON body must be an object')
                return data

            def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
                body = json.dumps(data).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.end_headers()
                self.wfile.write(body)

        return ThreadingHTTPServer((self.host, self.port), RequestHandler)

    def publish_enable(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = enabled
        self.enable_pub.publish(msg)
        self.get_logger().info(f'Published enable_flag={enabled}')

    def publish_pose(self, data: Dict[str, Any]) -> PosCmd:
        msg = PosCmd()
        msg.x = clamp(require_number(data, 'x'), -self.max_linear_m, self.max_linear_m)
        msg.y = clamp(require_number(data, 'y'), -self.max_linear_m, self.max_linear_m)
        msg.z = clamp(require_number(data, 'z'), self.min_z_m, self.max_z_m)
        msg.roll = clamp(require_number(data, 'roll'), -self.max_rotation_rad, self.max_rotation_rad)
        msg.pitch = clamp(require_number(data, 'pitch'), -self.max_rotation_rad, self.max_rotation_rad)
        msg.yaw = clamp(require_number(data, 'yaw'), -self.max_rotation_rad, self.max_rotation_rad)
        msg.gripper = clamp(require_number(data, 'gripper'), 0.0, self.max_gripper_m)
        msg.mode1 = int(require_number(data, 'mode1', 0.0))
        msg.mode2 = int(require_number(data, 'mode2', 0.0))
        self.pos_pub.publish(msg)
        return msg

    def publish_joints(self, data: Dict[str, Any]) -> JointState:
        positions = data.get('positions')
        if not isinstance(positions, list) or len(positions) < 6:
            raise ValueError('positions must be a list with at least 6 joint values in radians')

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']
        msg.position = self._number_list(positions, 'positions', 8, default=0.0)

        speed = clamp(require_number(data, 'speed', self.default_joint_speed), 0.0, 100.0)
        msg.velocity = [speed] * 8

        gripper_effort = clamp(require_number(data, 'gripper_effort', self.default_gripper_effort), 0.5, 3.0)
        msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_effort, gripper_effort]

        self.joint_pub.publish(msg)
        return msg

    def _number_list(self, values: Iterable[Any], name: str, length: int, default: float) -> List[float]:
        result = list(values)
        while len(result) < length:
            result.append(default)
        result = result[:length]
        if not all(isinstance(value, (int, float)) for value in result):
            raise ValueError(f'{name} must only contain numbers')
        return [float(value) for value in result]

    def pose_to_dict(self, msg: PosCmd) -> Dict[str, Any]:
        return {
            'x': msg.x,
            'y': msg.y,
            'z': msg.z,
            'roll': msg.roll,
            'pitch': msg.pitch,
            'yaw': msg.yaw,
            'gripper': msg.gripper,
            'mode1': msg.mode1,
            'mode2': msg.mode2,
        }

    def joints_to_dict(self, msg: JointState) -> Dict[str, Any]:
        return {
            'name': list(msg.name),
            'position': list(msg.position),
            'velocity': list(msg.velocity),
            'effort': list(msg.effort),
        }


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PiperRemoteBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
