#!/usr/bin/env python3
# flake8: noqa
"""
Standalone, read-only PyQt5/VTK mission Ray Review window.

The process accepts one JSON object per stdin line:

``{"command":"open","report":"/validated/report.json"}``
``{"command":"shutdown"}``

It intentionally contains no robot middleware imports or command APIs.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import vtk
# Ubuntu/Foxy's VTK Python bridge still references the removed NumPy alias.
# Keep the compatibility local to this viewer process.
if 'bool' not in np.__dict__:
    np.bool = np.bool_
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # noqa: E402
from vtk.util.numpy_support import numpy_to_vtk  # noqa: E402

from piper_gui.ray_review_model import (
    capability_ray_overlay,
    FREE,
    STAGE_LABELS,
    SURFACE,
    UNKNOWN,
    UrdfAssembly,
    event_cycles,
    filter_rays,
    load_capability_view,
    load_coverage_snapshot,
    load_diagnostic_document,
    load_optical_registration,
    revolved_envelope_mesh,
    state_at_event,
)


BACKGROUND = (0.055, 0.071, 0.090)
SURVIVOR = (0.18, 0.78, 0.94)
CULLED = (0.36, 0.40, 0.45)
NEWLY_CULLED = (0.95, 0.43, 0.30)
SELECTED = (1.0, 0.77, 0.20)
CAPTURED = (0.31, 0.91, 0.48)
REEVALUATED = (0.72, 0.38, 1.0)
RANK_PALETTE = (
    (0.12, 0.40, 1.00),  # best: blue
    (0.00, 0.78, 1.00),  # cyan
    (0.12, 0.84, 0.43),  # green
    (1.00, 0.86, 0.10),  # yellow
    (1.00, 0.46, 0.06),  # orange
    (0.90, 0.12, 0.10),  # lowest: red
)
GROUND_Z_M = -0.466
REVOLVED_MODEL = (0.10, 0.72, 0.68)
PLANNING_BOX = (0.15, 0.92, 0.86)
SOURCE_OUTLINE = (1.00, 0.55, 0.16)


def _vtk_matrix(array):
    result = vtk.vtkMatrix4x4()
    values = np.asarray(array, dtype=float)
    for row in range(4):
        for column in range(4):
            result.SetElement(row, column, float(values[row, column]))
    return result


def _line_actor(start, end, color, opacity=1.0, width=2.0, dashed=False):
    source = vtk.vtkLineSource()
    source.SetPoint1(*start)
    source.SetPoint2(*end)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetLineWidth(width)
    if dashed:
        actor.GetProperty().SetLineStipplePattern(0x00FF)
        actor.GetProperty().SetLineStippleRepeatFactor(1)
    return actor


def _point_actor(position, color, radius=0.006, opacity=1.0):
    source = vtk.vtkSphereSource()
    source.SetCenter(*position)
    source.SetRadius(radius)
    source.SetThetaResolution(12)
    source.SetPhiResolution(8)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    return actor


def _is_culled(ray):
    return bool(ray.get('culled') or ray.get('status') == 'culled'
                or 'culled' in str(ray.get('planner_status', '')))


def _numeric_rank(ray):
    try:
        return int(ray['rank'])
    except (KeyError, TypeError, ValueError):
        return None


def _rank_color(rank, best_rank, worst_rank):
    """Map an event-time rank through a blue-to-red quality spectrum."""
    if rank is None or best_rank is None or worst_rank is None:
        return SURVIVOR
    if worst_rank <= best_rank:
        return RANK_PALETTE[0]
    position = max(0.0, min(
        1.0, float(rank - best_rank) / float(worst_rank - best_rank)))
    scaled = position * (len(RANK_PALETTE) - 1)
    start_index = min(int(scaled), len(RANK_PALETTE) - 2)
    amount = scaled - start_index
    start = RANK_PALETTE[start_index]
    end = RANK_PALETTE[start_index + 1]
    return tuple(start[index] + amount * (end[index] - start[index])
                 for index in range(3))


class MissionScene:
    def __init__(self, widget, project_root):
        self.widget = widget
        self.project_root = Path(project_root)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*BACKGROUND)
        widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = widget.GetRenderWindow().GetInteractor()
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor_style.SetMouseWheelMotionFactor(1.2)
        self.interactor.SetInteractorStyle(self.interactor_style)
        self.dynamic = []
        self.robot_actors = []
        self.camera_source = None
        self.reference_joints = {}
        self.assembly = None
        self.ground_z_m = GROUND_Z_M
        self.last_valid_camera = None
        self._load_robot()
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(0.07, 0.07, 0.07)
        axes.SetNormalizedShaftLength(0.76, 0.76, 0.76)
        axes.SetNormalizedTipLength(0.24, 0.24, 0.24)
        axes.SetCylinderRadius(0.025)
        axes.SetConeRadius(0.08)
        for caption in (
                axes.GetXAxisCaptionActor2D(),
                axes.GetYAxisCaptionActor2D(),
                axes.GetZAxisCaptionActor2D()):
            caption.GetTextActor().SetTextScaleModeToNone()
            caption.GetCaptionTextProperty().SetFontSize(11)
            caption.SetWidth(0.025)
            caption.SetHeight(0.025)
        self.axes_actor = axes
        self.renderer.AddActor(axes)
        # Fit to the robot before the broad ground plane enters the scene.
        self.renderer.ResetCamera()
        self._add_ground()
        self.renderer.ResetCameraClippingRange()
        self._remember_camera()
        self.interactor.AddObserver(
            'InteractionEvent', self._keep_camera_above_ground)

    def _add_ground(self):
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(-1.5, -1.5, self.ground_z_m)
        plane.SetPoint1(1.5, -1.5, self.ground_z_m)
        plane.SetPoint2(-1.5, 1.5, self.ground_z_m)
        plane.SetXResolution(30)
        plane.SetYResolution(30)

        solid_mapper = vtk.vtkPolyDataMapper()
        solid_mapper.SetInputConnection(plane.GetOutputPort())
        solid = vtk.vtkActor()
        solid.SetMapper(solid_mapper)
        solid.GetProperty().SetColor(0.105, 0.125, 0.14)
        solid.PickableOff()
        self.renderer.AddActor(solid)
        self.ground_actor = solid

        grid_mapper = vtk.vtkPolyDataMapper()
        grid_mapper.SetInputConnection(plane.GetOutputPort())
        grid = vtk.vtkActor()
        grid.SetMapper(grid_mapper)
        grid.GetProperty().SetRepresentationToWireframe()
        grid.GetProperty().SetColor(0.24, 0.29, 0.33)
        grid.GetProperty().SetOpacity(0.72)
        grid.GetProperty().SetLineWidth(1.0)
        grid.PickableOff()
        self.renderer.AddActor(grid)

    def _remember_camera(self):
        camera = self.renderer.GetActiveCamera()
        self.last_valid_camera = (
            tuple(camera.GetPosition()),
            tuple(camera.GetFocalPoint()),
            tuple(camera.GetViewUp()),
        )

    def _keep_camera_above_ground(self, _caller=None, _event=None):
        camera = self.renderer.GetActiveCamera()
        position = camera.GetPosition()
        minimum_z = self.ground_z_m + 0.015
        if position[2] >= minimum_z:
            self._remember_camera()
            return
        if self.last_valid_camera is not None:
            previous_position, previous_focal, previous_up = (
                self.last_valid_camera)
            camera.SetPosition(*previous_position)
            camera.SetFocalPoint(*previous_focal)
            camera.SetViewUp(*previous_up)
        else:
            camera.SetPosition(position[0], position[1], minimum_z)
        self.renderer.ResetCameraClippingRange()
        self.widget.GetRenderWindow().Render()

    def focus_target(self, target):
        values = np.asarray(target, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            return
        camera = self.renderer.GetActiveCamera()
        position = np.asarray(camera.GetPosition(), dtype=float)
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        offset = position - focal
        if float(np.linalg.norm(offset)) < 1e-6:
            offset = np.asarray([0.9, -1.1, 0.65], dtype=float)
        camera.SetFocalPoint(*values)
        camera.SetPosition(*(values + offset))
        self.renderer.ResetCameraClippingRange()
        self._keep_camera_above_ground()
        self._remember_camera()
        self.widget.GetRenderWindow().Render()

    def _load_robot(self):
        urdf = self.project_root / (
            'piper_ros_foxy/src/piper_description/urdf/'
            'piper_description.xacro')
        try:
            home_path = self.project_root / 'piper_home_pose.json'
            if home_path.is_file():
                with home_path.open('r', encoding='utf-8') as stream:
                    home = json.load(stream)
                names = home.get('joint_names', [])
                values = home.get('positions_rad', [])
                if len(names) >= 6 and len(values) >= 6:
                    self.reference_joints = dict(zip(names[:6], values[:6]))
            self.assembly = UrdfAssembly(urdf)
            for visual, transform in self.assembly.visual_transforms({}):
                reader = vtk.vtkSTLReader()
                reader.SetFileName(str(visual.mesh_path))
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(reader.GetOutputPort())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.SetUserMatrix(_vtk_matrix(transform))
                actor.SetScale(*visual.scale)
                actor.GetProperty().SetColor(*visual.color[:3])
                self.renderer.AddActor(actor)
                self.robot_actors.append((visual, actor))
                if visual.link == 'bunker_chassis_collision':
                    reader.Update()
                    bounds = actor.GetBounds()
                    if bounds and np.all(np.isfinite(bounds)):
                        self.ground_z_m = float(bounds[4])
                if visual.link == 'l515_visual':
                    calibration = self.project_root / (
                        'L515_camera/calibration/hand_eye/'
                        'session_20260808_straight_mount/'
                        'calibration_result.yaml')
                    visual_from_color = load_optical_registration(calibration)
                    mesh_scale = np.diag(list(visual.scale) + [1.0])
                    # Glyph direction mode aligns source +X to the requested
                    # optical direction. Express the calibrated colour-frame
                    # mesh with its optical +Z rotated onto source +X.
                    source_axis = np.asarray([
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ])
                    source_transform = source_axis.dot(
                        np.linalg.inv(visual_from_color)).dot(
                            visual.origin).dot(mesh_scale)
                    scale = vtk.vtkTransform()
                    scale.SetMatrix(_vtk_matrix(source_transform))
                    scaled = vtk.vtkTransformPolyDataFilter()
                    scaled.SetTransform(scale)
                    scaled.SetInputConnection(reader.GetOutputPort())
                    scaled.Update()
                    self.camera_source = scaled.GetOutput()
        except (OSError, ValueError):
            self.assembly = None

    def _clear_dynamic(self):
        for actor in self.dynamic:
            self.renderer.RemoveActor(actor)
        self.dynamic = []

    def _update_robot(self, robot_pose):
        if self.assembly is None:
            return
        joints = dict(self.reference_joints)
        if isinstance(robot_pose, dict):
            names = robot_pose.get('joint_names', [])
            values = robot_pose.get(
                'joint_positions_rad', robot_pose.get('positions', []))
            if names and len(names) == len(values):
                joints = dict(zip(names, values))
            elif len(values) >= 6:
                joints = {'joint%d' % (i + 1): values[i] for i in range(6)}
            gripper_names = robot_pose.get('gripper_joint_names', [])
            gripper_values = robot_pose.get('gripper_joint_positions', [])
            if len(gripper_names) == len(gripper_values):
                joints.update(zip(gripper_names, gripper_values))
        transforms = dict((item.link, transform) for item, transform in
                          self.assembly.visual_transforms(joints))
        for visual, actor in self.robot_actors:
            actor.SetUserMatrix(_vtk_matrix(transforms[visual.link]))

    def _add_target(self, target):
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*target)
        sphere.SetRadius(0.009)
        sphere.SetThetaResolution(24)
        sphere.SetPhiResolution(16)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.1, 0.9, 0.78)
        self.renderer.AddActor(actor)
        self.dynamic.append(actor)

    def _add_revolved_model(self, envelope):
        vertices, faces = revolved_envelope_mesh(envelope)
        if not len(vertices) or not len(faces):
            return
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(vertices, deep=True))
        polygons = vtk.vtkCellArray()
        for face in faces:
            triangle = vtk.vtkTriangle()
            for index, point_id in enumerate(face):
                triangle.GetPointIds().SetId(index, int(point_id))
            polygons.InsertNextCell(triangle)
        surface = vtk.vtkPolyData()
        surface.SetPoints(points)
        surface.SetPolys(polygons)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(surface)
        normals.ConsistencyOn()
        normals.SplittingOff()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*REVOLVED_MODEL)
        actor.GetProperty().SetOpacity(0.24)
        actor.GetProperty().SetInterpolationToPhong()
        actor.PickableOff()
        self.renderer.AddActor(actor)
        self.dynamic.append(actor)

    def _add_source_outline(self, envelope):
        try:
            outline = np.asarray(
                envelope['visible_silhouette_points_m'], dtype=float)
        except (KeyError, TypeError, ValueError):
            return
        if (
                outline.ndim != 2 or outline.shape[1] != 3
                or outline.shape[0] < 3
                or not np.all(np.isfinite(outline))):
            return
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(outline, deep=True))
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(outline) + 1)
        for index in range(len(outline)):
            polyline.GetPointIds().SetId(index, index)
        polyline.GetPointIds().SetId(len(outline), 0)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(polyline)
        source = vtk.vtkPolyData()
        source.SetPoints(points)
        source.SetLines(lines)
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(source)
        tube.SetRadius(0.0015)
        tube.SetNumberOfSides(8)
        tube.CappingOn()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*SOURCE_OUTLINE)
        actor.PickableOff()
        self.renderer.AddActor(actor)
        self.dynamic.append(actor)

    def _add_target_envelope(
            self, envelope, show_revolved_model=True,
            show_planning_boxes=True, show_source_outline=True):
        """Render independently selectable recorded target evidence."""
        if not isinstance(envelope, dict):
            return
        if show_revolved_model:
            self._add_revolved_model(envelope)
        if show_source_outline:
            self._add_source_outline(envelope)
        if not show_planning_boxes:
            return
        boxes = envelope.get('collision_boxes', [])
        if not isinstance(boxes, list):
            return
        for box in boxes:
            if not isinstance(box, dict):
                continue
            try:
                minimum = np.asarray(box['minimum_m'], dtype=float)
                maximum = np.asarray(box['maximum_m'], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            if (minimum.shape != (3,) or maximum.shape != (3,)
                    or not np.all(np.isfinite(minimum))
                    or not np.all(np.isfinite(maximum))
                    or np.any(maximum <= minimum)):
                continue
            cube = vtk.vtkCubeSource()
            cube.SetBounds(
                float(minimum[0]), float(maximum[0]),
                float(minimum[1]), float(maximum[1]),
                float(minimum[2]), float(maximum[2]))
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(cube.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*PLANNING_BOX)
            actor.GetProperty().SetOpacity(0.075)
            actor.GetProperty().SetEdgeVisibility(True)
            actor.GetProperty().SetEdgeColor(0.15, 0.92, 0.86)
            actor.GetProperty().SetLineWidth(1.0)
            actor.PickableOff()
            self.renderer.AddActor(actor)
            self.dynamic.append(actor)

    def _add_coverage(self, model):
        if not isinstance(model, dict):
            return
        path = model.get('snapshot_path')
        if not path:
            return
        try:
            snapshot = load_coverage_snapshot(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return
        states = snapshot['states']
        centers = snapshot['voxel_centers_m']
        groups = [
                (SURFACE, (0.06, 0.75, 0.69), 0.92),
                (UNKNOWN, (0.64, 0.67, 0.70), 0.16)]
        if model.get('show_free'):
            groups.append((FREE, (0.20, 0.34, 0.44), 0.08))
        for state, color, opacity in groups:
            points_array = centers[states == state]
            if not len(points_array):
                continue
            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(points_array, deep=True))
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            cube = vtk.vtkCubeSource()
            size = float(snapshot['metadata']['voxel_size_m']) * 0.88
            cube.SetXLength(size); cube.SetYLength(size); cube.SetZLength(size)
            mapper = vtk.vtkGlyph3DMapper()
            mapper.SetInputData(poly)
            mapper.SetSourceConnection(cube.GetOutputPort())
            mapper.ScalingOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetOpacity(opacity)
            self.renderer.AddActor(actor)
            self.dynamic.append(actor)

    def render(self, state, visible_rays, show_free=False,
               show_past_culled=False, show_standoff_bounds=False,
               show_revolved_model=True, show_planning_boxes=True,
               show_source_outline=True):
        self._clear_dynamic()
        event = state.get('event') or {}
        target = state.get('target_center_m')
        if target is None:
            target = [0.0, 0.0, 0.0]
        self._add_target(target)
        self._add_target_envelope(
            state.get('target_envelope'), show_revolved_model,
            show_planning_boxes, show_source_outline)
        self._update_robot(state.get('robot_pose'))
        event_ranks = [
            _numeric_rank(ray) for ray in state.get('rays', {}).values()
            if not _is_culled(ray) and not ray.get('captured')]
        event_ranks = [rank for rank in event_ranks if rank is not None]
        best_rank = min(event_ranks) if event_ranks else None
        worst_rank = max(event_ranks) if event_ranks else None
        rendered_rays = []
        for ray in visible_rays:
            culled = _is_culled(ray)
            newly_culled = bool(ray.get('newly_culled_at_event'))
            # A rejected candidate flashes red on its cull event, then leaves
            # the scene. Its audit evidence remains available in the table.
            if (culled and not newly_culled and not ray.get('captured')
                    and not show_past_culled):
                continue
            direction = np.asarray(ray.get('direction', [0, 0, 0]), dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                continue
            direction /= norm
            minimum = float(ray.get('minimum_standoff_m', 0.28))
            maximum = float(ray.get('maximum_standoff_m', 0.80))
            requested_minimum = float(ray.get(
                'requested_minimum_standoff_m', minimum))
            requested_maximum = float(ray.get(
                'requested_maximum_standoff_m', maximum))
            # The primary ray is always visible from target to camera range.
            # Its previous near-to-far-only form collapsed to a point whenever
            # diagnostics recorded one exact standoff (minimum == maximum).
            start = np.asarray(target, dtype=float)
            end = np.asarray(target) + direction * maximum
            rank_color = _rank_color(
                _numeric_rank(ray), best_rank, worst_rank)
            if newly_culled:
                color, opacity, width = NEWLY_CULLED, 1.0, 2.5
            elif ray.get('reevaluated_at_event'):
                color, opacity, width = REEVALUATED, 1.0, 1.5
            elif ray.get('captured'):
                color, opacity, width = CAPTURED, 1.0, 3.0
            elif ray.get('selected_at_event'):
                color, opacity, width = rank_color, 1.0, 3.0
            elif culled:
                color, opacity, width = CULLED, 0.22, 1.0
            else:
                color, opacity, width = rank_color, 0.82, 1.0
            actor = _line_actor(
                start, end, color, opacity, width,
                dashed=culled and not newly_culled)
            self.renderer.AddActor(actor)
            self.dynamic.append(actor)
            if show_standoff_bounds:
                requested_near = (
                    np.asarray(target, dtype=float)
                    + direction * requested_minimum)
                requested_far = (
                    np.asarray(target, dtype=float)
                    + direction * requested_maximum)
                requested = _line_actor(
                    requested_near, requested_far,
                    (0.72, 0.75, 0.80), 0.32, 1.0, dashed=True)
                self.renderer.AddActor(requested)
                self.dynamic.append(requested)
                intervals = ray.get('capability_intervals_m', [])
                for interval in intervals:
                    if not isinstance(interval, (list, tuple)) \
                            or len(interval) != 2:
                        continue
                    near = (np.asarray(target, dtype=float)
                            + direction * float(interval[0]))
                    far = (np.asarray(target, dtype=float)
                           + direction * float(interval[1]))
                    bounded = _line_actor(
                        near, far, (0.20, 0.82, 0.90), 0.90, 2.5)
                    self.renderer.AddActor(bounded)
                    self.dynamic.append(bounded)
                for position in (
                        (requested_near,) if np.allclose(
                            requested_near, requested_far)
                        else (requested_near, requested_far)):
                    marker = _point_actor(
                        position, (0.92, 0.95, 1.0), 0.006, 0.9)
                    self.renderer.AddActor(marker)
                    self.dynamic.append(marker)
            rendered_rays.append(ray)
        # Candidate cameras are useful only after this population has been
        # culled.  Before then, one mesh per provisional ray creates a false
        # solid-looking shell around the target.  Past culled rays remain as
        # dashed audit lines, but only current survivors receive a camera
        # glyph.
        population_has_culls = any(
            _is_culled(ray) for ray in state.get('rays', {}).values())
        camera_rays = [
            ray for ray in rendered_rays
            if not _is_culled(ray) and not ray.get('captured')]
        if (self.camera_source is not None
                and population_has_culls and camera_rays):
            camera_positions = []
            camera_directions = []
            for ray in camera_rays:
                position = ray.get('camera_position_m',
                                   ray.get('representative_position_m'))
                if isinstance(position, (list, tuple)) and len(position) == 3:
                    camera_positions.append(position)
                    direction = -np.asarray(
                        ray.get('direction', [1.0, 0.0, 0.0]), dtype=float)
                    direction /= max(1e-12, float(np.linalg.norm(direction)))
                    camera_directions.append(direction)
            if camera_positions:
                points = vtk.vtkPoints()
                points.SetData(numpy_to_vtk(
                    np.asarray(camera_positions, dtype=float), deep=True))
                poly = vtk.vtkPolyData(); poly.SetPoints(points)
                orientations = numpy_to_vtk(
                    np.asarray(camera_directions, dtype=float), deep=True)
                orientations.SetName('optical_direction')
                poly.GetPointData().AddArray(orientations)
                mapper = vtk.vtkGlyph3DMapper()
                mapper.SetInputData(poly); mapper.SetSourceData(self.camera_source)
                mapper.SetOrientationArray('optical_direction')
                mapper.SetOrientationModeToDirection()
                mapper.OrientOn()
                mapper.ScalingOff()
                cameras = vtk.vtkActor(); cameras.SetMapper(mapper)
                cameras.GetProperty().SetColor(0.28, 0.65, 0.82)
                cameras.GetProperty().SetOpacity(0.60)
                self.renderer.AddActor(cameras); self.dynamic.append(cameras)
                highlighted = [
                    (ray.get('camera_position_m',
                             ray.get('representative_position_m')),
                     -np.asarray(ray.get('direction', [1.0, 0.0, 0.0]),
                                 dtype=float))
                    for ray in rendered_rays
                    if ray.get('selected_at_event') or ray.get('captured')]
                highlighted = [
                    value for value in highlighted
                    if isinstance(value[0], (list, tuple))
                    and len(value[0]) == 3]
                if highlighted:
                    selected_points = vtk.vtkPoints()
                    selected_points.SetData(numpy_to_vtk(
                        np.asarray([value[0] for value in highlighted],
                                   dtype=float), deep=True))
                    selected_poly = vtk.vtkPolyData()
                    selected_poly.SetPoints(selected_points)
                    selected_directions = np.asarray([
                        value[1] / max(1e-12, float(np.linalg.norm(value[1])))
                        for value in highlighted], dtype=float)
                    selected_orientations = numpy_to_vtk(
                        selected_directions, deep=True)
                    selected_orientations.SetName('optical_direction')
                    selected_poly.GetPointData().AddArray(
                        selected_orientations)
                    selected_mapper = vtk.vtkGlyph3DMapper()
                    selected_mapper.SetInputData(selected_poly)
                    selected_mapper.SetSourceData(self.camera_source)
                    selected_mapper.SetOrientationArray('optical_direction')
                    selected_mapper.SetOrientationModeToDirection()
                    selected_mapper.OrientOn()
                    selected_mapper.ScalingOff()
                    selected_actor = vtk.vtkActor()
                    selected_actor.SetMapper(selected_mapper)
                    selected_actor.GetProperty().SetColor(*SELECTED)
                    self.renderer.AddActor(selected_actor)
                    self.dynamic.append(selected_actor)
        model = dict(state.get('target_model') or {})
        model['show_free'] = bool(show_free)
        self._add_coverage(model)
        self.widget.GetRenderWindow().Render()


class Inspector(QtWidgets.QWidget):
    filters_changed = QtCore.pyqtSignal()
    ray_selected = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel('Ray / ranking inspector')
        title.setObjectName('sectionTitle')
        layout.addWidget(title)
        self.show_revolved_model = QtWidgets.QCheckBox(
            'Estimated revolved model')
        self.show_planning_boxes = QtWidgets.QCheckBox(
            'Conservative planning boxes')
        self.show_source_outline = QtWidgets.QCheckBox(
            'Original mask/depth outline')
        self.show_revolved_model.setChecked(True)
        self.show_planning_boxes.setChecked(True)
        self.show_source_outline.setChecked(True)
        target_layers = QtWidgets.QGroupBox('Target layers')
        target_layers_layout = QtWidgets.QVBoxLayout(target_layers)
        target_layers_layout.setContentsMargins(8, 5, 8, 5)
        target_layers_layout.addWidget(self.show_revolved_model)
        target_layers_layout.addWidget(self.show_planning_boxes)
        target_layers_layout.addWidget(self.show_source_outline)
        layout.addWidget(target_layers)
        form = QtWidgets.QFormLayout()
        self.stage = QtWidgets.QComboBox(); self.stage.addItem('All stages', '')
        for value in ('history', 'information', 'prequalification', 'bridge',
                      'tesseract'):
            self.stage.addItem(value.title(), value)
        self.reason = QtWidgets.QLineEdit(); self.reason.setPlaceholderText('reason contains…')
        self.rank_min = QtWidgets.QSpinBox(); self.rank_min.setRange(0, 100000)
        self.rank_max = QtWidgets.QSpinBox(); self.rank_max.setRange(0, 100000); self.rank_max.setValue(100000)
        self.ray_id = QtWidgets.QLineEdit(); self.ray_id.setPlaceholderText('ray ID')
        self.only_key = QtWidgets.QCheckBox('Selected / captured only')
        self.show_culled = QtWidgets.QCheckBox('Keep culled rays visible')
        self.show_culled.setChecked(True)
        self.show_bounds = QtWidgets.QCheckBox('Show standoff bounds')
        self.show_free = QtWidgets.QCheckBox('Show FREE target voxels')
        form.addRow('Cull stage', self.stage); form.addRow('Reason', self.reason)
        form.addRow('Minimum rank', self.rank_min); form.addRow('Maximum rank', self.rank_max)
        form.addRow('Ray ID', self.ray_id); form.addRow('', self.only_key)
        form.addRow('', self.show_culled)
        form.addRow('', self.show_bounds)
        form.addRow('', self.show_free)
        layout.addLayout(form)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ['Ray', 'Rank', 'State', 'Score', 'Stage', 'Reason'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)
        self.evidence = QtWidgets.QPlainTextEdit(); self.evidence.setReadOnly(True)
        self.evidence.setMaximumHeight(160)
        layout.addWidget(self.evidence)
        for control in (self.stage, self.rank_min, self.rank_max,
                        self.only_key, self.show_culled, self.show_bounds,
                        self.show_free, self.show_revolved_model,
                        self.show_planning_boxes,
                        self.show_source_outline):
            if isinstance(control, QtWidgets.QComboBox):
                control.currentIndexChanged.connect(self.filters_changed)
            elif isinstance(control, QtWidgets.QCheckBox):
                control.toggled.connect(self.filters_changed)
            else:
                control.valueChanged.connect(self.filters_changed)
        self.reason.textChanged.connect(self.filters_changed)
        self.ray_id.textChanged.connect(self.filters_changed)
        self.table.itemSelectionChanged.connect(self._selection)

    def _selection(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.ray_selected.emit(int(self.table.item(rows[0].row(), 0).text()))

    def values(self):
        maximum_rank = self.rank_max.value()
        return dict(
            cull_stage=self.stage.currentData(), reason=self.reason.text(),
            rank_min=self.rank_min.value() or None,
            rank_max=(None if maximum_rank == self.rank_max.maximum()
                      else maximum_rank),
            selected_captured=self.only_key.isChecked(),
            ray_id=self.ray_id.text())

    def populate(self, rays, event):
        self.table.setRowCount(len(rays))
        for row, ray in enumerate(rays):
            reasons = ray.get('reasons', ray.get('planner_reasons', []))
            if ray.get('reevaluated_at_event'):
                state_label = 'REEVALUATED'
                previous = ray.get('previous_cull_reasons', [])
                reasons = list(reasons) + ([
                    'previously culled: ' + '; '.join(
                        str(value) for value in previous)] if previous else [])
            elif _is_culled(ray) and ray.get(
                    'cull_disposition') == 'permanent':
                state_label = 'PERMANENT'
            elif _is_culled(ray) and ray.get(
                    'cull_disposition') == 'retry_eligible':
                state_label = 'RETRY ELIGIBLE'
            else:
                state_label = ray.get(
                    'status', ray.get('planner_status', 'visible'))
            values = (
                ray.get('ray_id', ''), ray.get('rank', ''),
                'CAPTURED' if ray.get('captured') else
                ('SELECTED' if ray.get('selected_at_event') else
                 state_label),
                ray.get('nbv_rank_score', ''), ray.get('cull_stage', ''),
                '; '.join(str(value) for value in reasons),
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        self.evidence.setPlainText(json.dumps(event or {}, indent=2, sort_keys=True))


class MissionTab(QtWidgets.QWidget):
    state_changed = QtCore.pyqtSignal(object)
    selected_ray_changed = QtCore.pyqtSignal(int)

    def __init__(self, project_root):
        super().__init__()
        self.document = {'events': []}
        self.current_state = {}
        self.follow_latest = True
        self.speed = 1.0
        self.full_replay_index = None
        self.focused_view = None
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.next_event)
        outer = QtWidgets.QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        toolbar = QtWidgets.QHBoxLayout()
        self.previous = QtWidgets.QToolButton(); self.previous.setText('◀')
        self.play = QtWidgets.QToolButton(); self.play.setText('▶'); self.play.setCheckable(True)
        self.next = QtWidgets.QToolButton(); self.next.setText('▶|')
        self.speed_box = QtWidgets.QComboBox(); self.speed_box.addItems(['0.5×', '1×', '2×']); self.speed_box.setCurrentIndex(1)
        self.follow = QtWidgets.QCheckBox('Follow latest'); self.follow.setChecked(True)
        self.display_mode = QtWidgets.QComboBox()
        self.display_mode.addItems(['Focused review', 'Full ray lifecycle'])
        self.full_replay = QtWidgets.QToolButton()
        self.full_replay.setText('Restart lifecycle')
        self.full_replay.setDisabled(True)
        self.position = QtWidgets.QLabel('No report')
        self.navigation_help = QtWidgets.QLabel(
            'Left drag: rotate · Middle drag: pan · Wheel: zoom · Q/E: step')
        for item in (self.previous, self.play, self.next, self.speed_box,
                     self.follow, self.display_mode, self.full_replay,
                     self.position):
            toolbar.addWidget(item)
        toolbar.addStretch(1)
        toolbar.addWidget(self.navigation_help)
        outer.addLayout(toolbar)
        self.ray_legend = QtWidgets.QLabel(
            'Rank: <span style="color:#1f66ff">&#9632; best</span> '
            '&#8594; <span style="color:#00c7ff">&#9632;</span> '
            '&#8594; <span style="color:#1fd66e">&#9632;</span> '
            '&#8594; <span style="color:#ffdb1a">&#9632;</span> '
            '&#8594; <span style="color:#ff750f">&#9632;</span> '
            '&#8594; <span style="color:#e61f1a">&#9632; lowest</span> &nbsp; '
            '<span style="color:#f26e4d">&#9632; culled now</span> &nbsp; '
            '<span style="color:#5c6673">&#9632; past cull</span> &nbsp; '
            '<span style="color:#b761ff">&#9632; re-evaluated</span> &nbsp; '
            '<span style="color:#4fe87a">&#9632; captured</span> &nbsp; '
            '<span style="color:#19e6c2">&#9679; target/ray centre</span> &nbsp; '
            'target envelope: '
            '<span style="color:#1ab8ad">&#9632; revolved estimate</span> &nbsp; '
            '<span style="color:#26ebdb">&#9633; planning boxes</span> &nbsp; '
            '<span style="color:#ff8c29">&#9633; source outline</span> &nbsp; '
            '<b>thick line = selected</b> &nbsp; Bounds: '
            '<span style="color:#b8beca">grey dashed = requested</span>, '
            '<span style="color:#33d1e6">cyan = capability-supported</span>'
            ' &nbsp; camera meshes = post-cull survivors only')
        self.ray_legend.setAlignment(QtCore.Qt.AlignCenter)
        outer.addWidget(self.ray_legend)
        self.population = QtWidgets.QLabel('No ray population loaded')
        self.population.setAlignment(QtCore.Qt.AlignCenter)
        outer.addWidget(self.population)
        vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        horizontal = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.scene = MissionScene(self.vtk_widget, project_root)
        self.inspector = Inspector()
        horizontal.addWidget(self.vtk_widget); horizontal.addWidget(self.inspector)
        horizontal.setStretchFactor(0, 4); horizontal.setStretchFactor(1, 1)
        bottom = QtWidgets.QWidget(); bottom_layout = QtWidgets.QVBoxLayout(bottom)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.events = QtWidgets.QListWidget(); self.events.setFlow(QtWidgets.QListView.LeftToRight)
        self.events.setWrapping(False); self.events.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.events.setMaximumHeight(115)
        bottom_layout.addWidget(self.slider); bottom_layout.addWidget(self.events)
        vertical.addWidget(horizontal); vertical.addWidget(bottom)
        vertical.setStretchFactor(0, 5); vertical.setStretchFactor(1, 1)
        outer.addWidget(vertical, 1)
        self.slider.valueChanged.connect(self.set_event)
        self.events.currentRowChanged.connect(self.slider.setValue)
        self.previous.clicked.connect(self.previous_event); self.next.clicked.connect(self.next_event)
        self.play.toggled.connect(self._toggle_play); self.speed_box.currentIndexChanged.connect(self._speed)
        self.follow.toggled.connect(self._follow)
        self.display_mode.currentIndexChanged.connect(
            self._display_mode_changed)
        self.full_replay.clicked.connect(self.replay_full_lifecycle)
        self.inspector.filters_changed.connect(self.refresh)
        self.inspector.ray_selected.connect(self.selected_ray_changed)
        self.installEventFilter(self)
        self.vtk_widget.installEventFilter(self)
        self.events.installEventFilter(self)
        self.slider.installEventFilter(self)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def load(self, document):
        self.document = document
        self.focused_view = None
        self.full_replay_index = next((
            index for index, event in enumerate(document.get('events', []))
            if event.get('stage') == 'generate'), None)
        complete = bool(document.get('journal_complete', True))
        available = complete and self.full_replay_index is not None
        model_item = self.display_mode.model().item(1)
        if model_item is not None:
            model_item.setFlags(QtCore.Qt.ItemFlags(33 if available else 0))
            model_item.setToolTip(
                'Replay every recorded ray through the complete mission'
                if available else
                'Unavailable because the original full ray population was '
                'not recorded')
        if not available and self.display_mode.currentIndex() == 1:
            self.display_mode.setCurrentIndex(0)
        self.full_replay.setDisabled(
            not (available and self.display_mode.currentIndex() == 1))
        self.full_replay.setToolTip(
            'Restart with every recorded generated ray'
            if available else
            'Unavailable: this historical report does not contain its full '
            'generated ray population')
        self.events.clear()
        for cycle in event_cycles(document):
            for event in cycle['events']:
                label = STAGE_LABELS.get(event.get('stage'), event.get('stage', '?'))
                metrics = event.get('metrics') or {}
                eliminated = int(metrics.get(
                    'eliminated_ray_count',
                    len(event.get('newly_culled_ray_ids', []))))
                surviving = metrics.get('surviving_ray_count')
                if event.get('stage') == 'generate':
                    detail = '%d rays' % int(metrics.get(
                        'generated_ray_count', len(event.get(
                            'ray_deltas', {}))))
                elif eliminated:
                    detail = '-%d eliminated%s' % (
                        eliminated,
                        '' if surviving is None else ' · %d remain' % int(
                            surviving))
                elif event.get('selected_ray_ids'):
                    detail = 'selected %s' % ', '.join(
                        '#%d' % int(value)
                        for value in event['selected_ray_ids'])
                elif surviving is not None and 'rank' in str(
                        event.get('stage', '')):
                    detail = '%d ranked' % int(surviving)
                else:
                    detail = ''
                item = QtWidgets.QListWidgetItem(
                    'Cycle %d\n%s%s' % (
                        cycle['cycle'], label,
                        '' if not detail else ' · ' + detail))
                item.setToolTip(str(event.get('message', '')))
                self.events.addItem(item)
        count = len(document.get('events', []))
        self.slider.setRange(0, max(0, count - 1))
        self.slider.setValue(max(0, count - 1) if self.follow_latest else 0)
        self.set_event(self.slider.value())
        if available and self.display_mode.currentIndex() == 1:
            self.replay_full_lifecycle()

    def _follow(self, value): self.follow_latest = bool(value)
    def _speed(self, index): self.speed = (0.5, 1.0, 2.0)[index]

    def _filter_controls(self):
        return (
            self.inspector.stage, self.inspector.reason,
            self.inspector.rank_min, self.inspector.rank_max,
            self.inspector.ray_id, self.inspector.only_key)

    def _display_mode_changed(self, index):
        full = int(index) == 1
        available = self.full_replay_index is not None and bool(
            self.document.get('journal_complete', True))
        if full and not available:
            self.display_mode.setCurrentIndex(0)
            return
        if full:
            self.focused_view = {
                'event': self.slider.value(),
                'follow': self.follow.isChecked(),
                'stage': self.inspector.stage.currentIndex(),
                'reason': self.inspector.reason.text(),
                'rank_min': self.inspector.rank_min.value(),
                'rank_max': self.inspector.rank_max.value(),
                'ray_id': self.inspector.ray_id.text(),
                'only_key': self.inspector.only_key.isChecked(),
            }
            for control in self._filter_controls():
                control.setDisabled(True)
            self.full_replay.setDisabled(False)
            self.replay_full_lifecycle()
            return
        self.play.setChecked(False)
        self.full_replay.setDisabled(True)
        for control in self._filter_controls():
            control.setDisabled(False)
        if self.focused_view is not None:
            view = self.focused_view
            self.follow.setChecked(view['follow'])
            self.inspector.stage.setCurrentIndex(view['stage'])
            self.inspector.reason.setText(view['reason'])
            self.inspector.rank_min.setValue(view['rank_min'])
            self.inspector.rank_max.setValue(view['rank_max'])
            self.inspector.ray_id.setText(view['ray_id'])
            self.inspector.only_key.setChecked(view['only_key'])
            self.slider.setValue(min(view['event'], self.slider.maximum()))
            self.focused_view = None

    def replay_full_lifecycle(self):
        if (self.full_replay_index is None
                or not self.document.get('journal_complete', True)):
            return
        self.play.setChecked(False)
        self.follow.setChecked(False)
        self.inspector.stage.setCurrentIndex(0)
        self.inspector.reason.clear()
        self.inspector.rank_min.setValue(0)
        self.inspector.rank_max.setValue(100000)
        self.inspector.ray_id.clear()
        self.inspector.only_key.setChecked(False)
        self.inspector.show_culled.setChecked(True)
        self.slider.setValue(self.full_replay_index)
        target = self.current_state.get('target_center_m')
        if target is not None:
            self.scene.focus_target(target)
        self.play.setChecked(True)

    def _toggle_play(self, playing):
        self.play.setText('❚❚' if playing else '▶')
        if playing: self.timer.start(max(50, round(650 / self.speed)))
        else: self.timer.stop()
    def previous_event(self): self.slider.setValue(max(0, self.slider.value() - 1))
    def next_event(self):
        if self.slider.value() >= self.slider.maximum():
            self.play.setChecked(False); return
        self.slider.setValue(self.slider.value() + 1)

    def set_event(self, index):
        if not self.document.get('events'): return
        self.current_state = state_at_event(self.document, index)
        event = self.current_state['event']
        self.position.setText('%d / %d · %s' % (
            index + 1, len(self.document['events']),
            STAGE_LABELS.get(event.get('stage'), event.get('stage', '')))
            + ('' if self.current_state.get('robot_pose') else
               ' · configured home reference'))
        self._update_population_label()
        self.events.setCurrentRow(index)
        self.refresh()
        self.state_changed.emit(self.current_state)

    def _update_population_label(self):
        rays = list(self.current_state.get('rays', {}).values())
        known = len(rays)
        active = sum(not _is_culled(ray) and not ray.get('captured')
                     for ray in rays)
        culled = sum(_is_culled(ray) and not ray.get('captured')
                     for ray in rays)
        captured = sum(bool(ray.get('captured')) for ray in rays)
        event = self.current_state.get('event') or {}
        phase = str(event.get('ray_population_phase', '')).strip()
        phase_label = (
            ' · %s population' % phase if phase else '')
        generation = event.get('snapshot_generation')
        if not isinstance(generation, dict):
            cycle = int(event.get('accepted_view_cycle', 0))
            generation = next((item for item in self.document.get(
                'generations', []) if int(item.get('generation', -1)) == cycle), {})
        expected = int(generation.get('generated_ray_count', known))
        eliminated_now = len(event.get('newly_culled_ray_ids', []))
        if not self.document.get('journal_complete', True):
            self.population.setText(
                'Partial historical evidence: %d of %d rays recorded · '
                'full lifecycle unavailable' % (known, max(known, expected)))
        else:
            self.population.setText(
                '%s%s · recorded population: %d · active: %d · culled: %d · '
                'captured: %d%s' % (
                    self.display_mode.currentText(), phase_label,
                    known, active, culled,
                    captured,
                    '' if not eliminated_now else
                    ' · <span style="color:#f26e4d"><b>eliminated now: '
                    '%d</b></span>' % eliminated_now))

    def refresh(self):
        if not self.current_state: return
        rays = filter_rays(self.current_state, **self.inspector.values())
        self.inspector.populate(rays, self.current_state.get('event'))
        self.scene.render(
            self.current_state, rays, self.inspector.show_free.isChecked(),
            self.inspector.show_culled.isChecked(),
            self.inspector.show_bounds.isChecked(),
            self.inspector.show_revolved_model.isChecked(),
            self.inspector.show_planning_boxes.isChecked(),
            self.inspector.show_source_outline.isChecked())

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key == QtCore.Qt.Key_Home: self.slider.setValue(0); return True
            if key == QtCore.Qt.Key_End: self.slider.setValue(self.slider.maximum()); return True
            if key == QtCore.Qt.Key_Left: self.previous_event(); return True
            if key == QtCore.Qt.Key_Right: self.next_event(); return True
            if key == QtCore.Qt.Key_Q: self.previous_event(); return True
            if key == QtCore.Qt.Key_E: self.next_event(); return True
            if key == QtCore.Qt.Key_Space: self.play.toggle(); return True
        return super().eventFilter(watched, event)


class CapabilityTab(QtWidgets.QWidget):
    process_step_requested = QtCore.pyqtSignal(int)

    def __init__(self, project_root):
        super().__init__()
        self.project_root = Path(project_root)
        self.view = None
        self.mission_state = None
        self.selected_ray_id = None
        self.base_info = ''
        layout = QtWidgets.QHBoxLayout(self)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer(); self.renderer.SetBackground(*BACKGROUND)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor_style.SetMouseWheelMotionFactor(1.2)
        self.interactor.SetInteractorStyle(self.interactor_style)
        self.vtk_widget.installEventFilter(self)
        self.vtk_widget.setFocusPolicy(QtCore.Qt.StrongFocus)
        panel = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(panel)
        navigation = QtWidgets.QLabel(
            'Left drag: rotate\nMiddle drag: pan\nWheel: zoom\nQ/E: process step')
        self.color = QtWidgets.QComboBox(); self.color.addItems(['Direction density', 'Maximum floor / clearance'])
        self.axis = QtWidgets.QComboBox(); self.axis.addItems(['All', 'X', 'Y', 'Z'])
        self.slice = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slice.setRange(0, 1000); self.slice.setValue(500)
        self.info = QtWidgets.QPlainTextEdit(); self.info.setReadOnly(True)
        self.heatmap = QtWidgets.QLabel(); self.heatmap.setMinimumSize(280, 150); self.heatmap.setScaledContents(True)
        form.addRow(navigation)
        form.addRow('Color', self.color); form.addRow('Slice axis', self.axis); form.addRow('Slice', self.slice)
        form.addRow('Azimuth/elevation support', self.heatmap); form.addRow(self.info)
        layout.addWidget(self.vtk_widget, 4); layout.addWidget(panel, 1)
        self.color.currentIndexChanged.connect(self.render)
        self.axis.currentIndexChanged.connect(self.render); self.slice.valueChanged.connect(self.render)

    def eventFilter(self, watched, event):
        if watched is self.vtk_widget \
                and event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Q:
                self.process_step_requested.emit(-1)
                return True
            if event.key() == QtCore.Qt.Key_E:
                self.process_step_requested.emit(1)
                return True
        return super().eventFilter(watched, event)

    def load(self):
        path = self.project_root / (
            'piper_ros_foxy/src/piper_mobile_manipulation/config/'
            'piper_camera_capability_map.npz')
        self.view = load_capability_view(path)
        metadata = self.view.metadata
        self.base_info = json.dumps({
            'artifact_sha256': self.view.sha256,
            'source_validation': metadata.get('_viewer_source_validation'),
            'enforcement_qualified': metadata.get(
                'qualified_for_enforcement'),
            'position_voxel_m': metadata.get('position_voxel_m'),
            'direction_bin_deg': metadata.get('direction_bin_deg'),
            'unique_camera_position_cells': len(self.view.positions_m),
            'occupied_pose_direction_bins': self.view.occupied_pose_direction_bins,
            'stores_joint_configurations': False,
            'notice': 'Capability support only; this is not an IK solution or trajectory.',
            'saved_convergence_checkpoints': {
                'checkpoint_samples': metadata.get('checkpoint_samples'),
                'reference_checkpoint_samples': metadata.get(
                    'reference_checkpoint_samples'),
                'selected_checkpoint_samples': metadata.get(
                    'selected_checkpoint_samples'),
                'convergence_summary': metadata.get('convergence_summary'),
            },
        }, indent=2, sort_keys=True)
        self.info.setPlainText(self.base_info)
        histogram = self.view.direction_histogram.astype(float)
        histogram /= max(1.0, histogram.max())
        image = QtGui.QImage(histogram.shape[1], histogram.shape[0], QtGui.QImage.Format_RGB32)
        for y_value in range(histogram.shape[0]):
            for x_value in range(histogram.shape[1]):
                value = histogram[y_value, x_value]
                image.setPixelColor(x_value, histogram.shape[0] - 1 - y_value,
                                    QtGui.QColor.fromHsvF(0.65 - 0.65 * value, 0.85, 0.25 + 0.75 * value))
        self.heatmap.setPixmap(QtGui.QPixmap.fromImage(image))
        self.render(); self.renderer.ResetCamera()

    def render(self):
        if self.view is None: return
        self.info.setPlainText(self.base_info)
        self.renderer.RemoveAllViewProps()
        positions = self.view.positions_m
        values = self.view.direction_density.astype(float) if self.color.currentIndex() == 0 else self.view.maximum_floor_m.astype(float)
        axis_index = self.axis.currentIndex() - 1
        if axis_index >= 0:
            low, high = positions[:, axis_index].min(), positions[:, axis_index].max()
            center = low + (high - low) * self.slice.value() / 1000.0
            width = float(self.view.metadata['position_voxel_m']) * 0.55
            keep = np.abs(positions[:, axis_index] - center) <= width
            positions, values = positions[keep], values[keep]
        points = vtk.vtkPoints(); points.SetData(numpy_to_vtk(positions, deep=True))
        poly = vtk.vtkPolyData(); poly.SetPoints(points)
        scalars = numpy_to_vtk(values.astype(np.float32), deep=True); scalars.SetName('support')
        poly.GetPointData().SetScalars(scalars)
        cube = vtk.vtkCubeSource(); size = float(self.view.metadata['position_voxel_m']) * 0.72
        cube.SetXLength(size); cube.SetYLength(size); cube.SetZLength(size)
        mapper = vtk.vtkGlyph3DMapper(); mapper.SetInputData(poly); mapper.SetSourceConnection(cube.GetOutputPort()); mapper.ScalingOff()
        mapper.SetScalarRange(float(values.min(initial=0)), float(values.max(initial=1)))
        actor = vtk.vtkActor(); actor.SetMapper(mapper); self.renderer.AddActor(actor)
        self._render_mission_overlay()
        self.vtk_widget.GetRenderWindow().Render()

    def set_mission_state(self, state):
        self.mission_state = state
        if self.view is not None:
            self.render()

    def set_selected_ray(self, ray_id):
        self.selected_ray_id = int(ray_id)
        if self.view is not None:
            self.render()

    def _render_mission_overlay(self):
        if not self.mission_state:
            return
        event = self.mission_state.get('event') or {}
        target = self.mission_state.get('target_center_m')
        rays = self.mission_state.get('rays', {})
        ray = rays.get(self.selected_ray_id)
        if target is None or not isinstance(ray, dict):
            return
        overlay = capability_ray_overlay(
            self.view, target, ray.get('direction', [0, 0, 0]),
            ray.get('requested_minimum_standoff_m',
                    ray.get('minimum_standoff_m', 0.28)),
            ray.get('requested_maximum_standoff_m',
                    ray.get('maximum_standoff_m', 0.80)))
        start = np.asarray(target, dtype=float)
        direction = np.asarray(ray.get('direction'), dtype=float)
        direction /= max(1e-12, float(np.linalg.norm(direction)))
        end = start + direction * float(ray.get(
            'requested_maximum_standoff_m',
            ray.get('maximum_standoff_m', 0.80)))
        line = _line_actor(start, end, SELECTED, 1.0, 2.5)
        self.renderer.AddActor(line)
        for key, color in (('matched', CAPTURED), ('unsupported', NEWLY_CULLED)):
            values = np.asarray(overlay[key], dtype=float)
            if not len(values):
                continue
            points = vtk.vtkPoints(); points.SetData(numpy_to_vtk(values, deep=True))
            poly = vtk.vtkPolyData(); poly.SetPoints(points)
            sphere = vtk.vtkSphereSource(); sphere.SetRadius(0.007); sphere.SetThetaResolution(10); sphere.SetPhiResolution(8)
            glyph = vtk.vtkGlyph3DMapper(); glyph.SetInputData(poly); glyph.SetSourceConnection(sphere.GetOutputPort()); glyph.ScalingOff()
            marker = vtk.vtkActor(); marker.SetMapper(glyph); marker.GetProperty().SetColor(*color)
            self.renderer.AddActor(marker)
        self.info.appendPlainText('\nSelected ray query\n' + json.dumps({
            'ray_id': self.selected_ray_id,
            'checked_cells': overlay['checked_count'],
            'matching_cells': overlay['matching_count'],
            'reason': overlay['reason'],
        }, sort_keys=True))


class RayReviewWindow(QtWidgets.QMainWindow):
    def __init__(self, project_root):
        super().__init__()
        self.project_root = Path(project_root).resolve()
        self.setWindowTitle('Ray Review')
        self.resize(1500, 900); self.setMinimumSize(620, 420)
        self.tabs = QtWidgets.QTabWidget(); self.setCentralWidget(self.tabs)
        self.mission = MissionTab(self.project_root); self.capability = CapabilityTab(self.project_root)
        self.tabs.addTab(self.mission, 'Mission Process'); self.tabs.addTab(self.capability, 'Capability Map')
        self.status = self.statusBar(); self.capability_loaded = False
        self.current_report = None
        self.report_mtime_ns = 0
        self.report_timer = QtCore.QTimer(self)
        self.report_timer.setInterval(500)
        self.report_timer.timeout.connect(self._refresh_live_report)
        self.report_timer.start()
        self.tabs.currentChanged.connect(self._tab_changed)
        self.mission.state_changed.connect(self.capability.set_mission_state)
        self.mission.selected_ray_changed.connect(
            self.capability.set_selected_ray)
        self.capability.process_step_requested.connect(self._step_process)
        self.setStyleSheet("""
        QWidget { background:#101820; color:#e7edf4; font-size:13px; }
        QToolButton,QComboBox,QLineEdit,QSpinBox { background:#1b2835; border:1px solid #34485b; padding:5px; }
        QTableWidget,QPlainTextEdit,QListWidget { background:#111c26; border:1px solid #2b3d4e; }
        QHeaderView::section { background:#203141; padding:5px; }
        QTabBar::tab { background:#182633; padding:9px 18px; } QTabBar::tab:selected { background:#2c526b; }
        QSplitter::handle { background:#263847; } #sectionTitle { font-size:17px; font-weight:600; }
        """)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            self.mission.previous_event()
            return
        if event.key() == QtCore.Qt.Key_E:
            self.mission.next_event()
            return
        if event.key() == QtCore.Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
            return
        super().keyPressEvent(event)

    def _step_process(self, direction):
        if int(direction) < 0:
            self.mission.previous_event()
        else:
            self.mission.next_event()

    def _tab_changed(self, index):
        if index == 1 and not self.capability_loaded:
            try:
                self.capability.load(); self.capability_loaded = True
            except Exception as error:
                self.status.showMessage('Capability map unavailable: %s' % error)

    def open_report(self, path):
        try:
            source = Path(path).resolve()
            previous_index = self.mission.slider.value()
            document = load_diagnostic_document(path)
            self.mission.load(document)
            self.current_report = source
            self.report_mtime_ns = source.stat().st_mtime_ns
            if not self.mission.follow_latest:
                self.mission.slider.setValue(min(
                    previous_index, self.mission.slider.maximum()))
            partial = ' · partial historical evidence' if not document.get('journal_complete', True) else ''
            self.setWindowTitle('Ray Review — %s%s' % (
                document.get('mission_id') or document.get('artifact_id', Path(path).parent.name), partial))
            self.status.showMessage('%d recorded events%s' % (len(document.get('events', [])), partial))
            self.show(); self.raise_(); self.activateWindow()
        except Exception as error:
            self.status.showMessage('Could not load report: %s' % error)

    def _refresh_live_report(self):
        if self.current_report is None:
            return
        try:
            modified = self.current_report.stat().st_mtime_ns
        except OSError:
            return
        if modified != self.report_mtime_ns:
            self.open_report(self.current_report)


class StdinProtocol(QtCore.QObject):
    def __init__(self, window):
        super().__init__(window); self.window = window
        self.notifier = QtCore.QSocketNotifier(sys.stdin.fileno(), QtCore.QSocketNotifier.Read, self)
        self.notifier.activated.connect(self.read)

    def read(self):
        line = sys.stdin.readline()
        if line == '': QtWidgets.QApplication.quit(); return
        try: message = json.loads(line)
        except (TypeError, json.JSONDecodeError): return
        if message.get('command') == 'shutdown': QtWidgets.QApplication.quit()
        elif message.get('command') == 'open': self.window.open_report(message.get('report', ''))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read-only mission Ray Review')
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--report')
    parser.add_argument('--full-lifecycle', action='store_true')
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('Ray Review')
    window = RayReviewWindow(args.project_root)
    protocol = StdinProtocol(window) if not args.report else None
    if args.report:
        window.open_report(args.report)
        if (args.full_lifecycle
                and window.mission.full_replay_index is not None
                and window.mission.document.get('journal_complete', True)):
            window.mission.speed_box.setCurrentIndex(0)
            window.mission.display_mode.setCurrentIndex(1)
    else: window.show()
    # Keep the protocol alive for the entire event loop.
    window._stdin_protocol = protocol
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(main())
