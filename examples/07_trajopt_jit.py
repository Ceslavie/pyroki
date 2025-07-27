"""Trajectory Optimization

Basic Trajectory Optimization using PyRoKi.

Robot going over a wall, while avoiding world-collisions.
"""

import time
from typing import Literal

import numpy as np
import pyroki as pk
import trimesh
import tyro
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroki_snippets as pks


def main(robot_name: Literal["ur5", "panda"] = "panda"):
    if robot_name == "ur5":
        urdf = load_robot_description("ur5_description")
        down_wxyz = np.array([0.707, 0, 0.707, 0])
        target_link_name = "ee_link"

        # For UR5 it's important to initialize the robot in a safe configuration;
        # the zero-configuration puts the robot aligned with the wall obstacle.
        default_cfg = np.zeros(6)
        default_cfg[1] = -1.308
        robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_cfg)

    elif robot_name == "panda":
        urdf = load_robot_description("panda_description")
        target_link_name = "panda_hand"
        down_wxyz = np.array([0, 0, 1, 0])  # for panda!
        robot = pk.Robot.from_urdf(urdf)

    else:
        raise ValueError(f"Invalid robot: {robot_name}")

    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    # Define the trajectory problem:
    # - number of timesteps, timestep size
    timesteps, dt = 25, 0.02
    # - the start and end poses.
    start_pos, end_pos = np.array([0.5, -0.3, 0.2]), np.array([0.5, 0.3, 0.2])

    # Define the obstacles:
    # - Ground
    ground_coll = pk.collision.HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    # - Sphere Obstacle (参考 06_online_planning.py)
    sphere_coll = pk.collision.Sphere.from_center_and_radius(
        np.array([0.5, 0.0, 0.2]), np.array([0.05])
    )
    # - Capsule Obstacle
    capsule_center = np.array([0.5, 0.0, 0.2])
    capsule_radius = 0.05
    capsule_height = 0.4
    # 绕y轴旋转90度对应四元数为 [cos(θ/2), 0, sin(θ/2), 0]，θ=90°
    from math import radians, cos, sin
    theta = radians(90)
    w = cos(theta/2)
    y = sin(theta/2)
    # capsule_wxyz = np.array([w, 0.0, y, 0.0])
    capsule_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    capsule_coll = pk.collision.Capsule.from_radius_height(
        radius=capsule_radius,
        height=capsule_height,
        position=capsule_center,
        wxyz=capsule_wxyz,
    )
    # world_coll = [ground_coll, sphere_coll, capsule_coll]
    world_coll = [ground_coll, capsule_coll]


    # 第一次调用
    t0 = time.time()
    traj = pks.solve_trajopt(
        robot,
        robot_coll,
        world_coll,
        target_link_name,
        start_pos,
        down_wxyz,
        end_pos,
        down_wxyz,
        timesteps,
        dt,
    )
    t1 = time.time()
    elapsed1 = t1 - t0

    # 第二次调用
    t0 = time.time()
    traj = pks.solve_trajopt(
        robot,
        robot_coll,
        world_coll,
        target_link_name,
        start_pos,
        down_wxyz,
        end_pos,
        down_wxyz,
        timesteps,
        dt,
    )
    t1 = time.time()
    elapsed2 = t1 - t0

    # # - Wall
    # wall_height = 0.4
    # wall_width = 0.1
    # wall_length = 0.4
    # wall_intervals = np.arange(start=0.2, stop=wall_length + 0.2, step=0.05)
    # translation = np.concatenate(
    #     [
    #         wall_intervals.reshape(-1, 1),
    #         np.full((wall_intervals.shape[0], 1), 0.0),
    #         np.full((wall_intervals.shape[0], 1), wall_height/2),
    #     ],
    #     axis=1,
    # )
    # wall_coll = pk.collision.Capsule.from_radius_height(
    #     position=translation,
    #     radius=np.full((translation.shape[0], 1), wall_width / 2),
    #     height=np.full((translation.shape[0], 1), wall_height),
    # )
    # sphere_coll = pk.collision.Sphere.from_center_and_radius(
    #     np.array([0.55, 0.0, 0.2]), np.array([0.05])
    # )
    # capsule_center = np.array([0.7, 0.0, 0.30])
    # capsule_coll = pk.collision.Capsule.from_radius_height(
    #     radius=capsule_radius,
    #     height=capsule_height,
    #     position=capsule_center,
    #     wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    # )
    # world_coll = [ground_coll, sphere_coll, capsule_coll]
    # world_coll = [ground_coll, capsule_coll]

    # 第三次调用
    # start_pos, end_pos = np.array([0.4, -0.3, 0.3]), np.array([0.4, 0.3, 0.3])
    t0 = time.time()
    traj = pks.solve_trajopt(
        robot,
        robot_coll,
        world_coll,
        target_link_name,
        start_pos,
        down_wxyz,
        end_pos,
        down_wxyz,
        timesteps,
        dt,
    )
    t1 = time.time()
    elapsed3 = t1 - t0

    traj = np.array(traj)

    # 汇总彩色打印
    print(f"\033[92m第一次调用耗时: {elapsed1:.3f} 秒\033[0m")
    print(f"\033[93m第二次调用耗时: {elapsed2:.3f} 秒\033[0m")
    print(f"\033[91m第三次调用耗时: {elapsed3:.3f} 秒\033[0m")

    # Visualize!
    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf)
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)
    # server.scene.add_mesh_trimesh(
    #     "sphere_obstacle",
    #     trimesh.creation.icosphere(
    #         radius=0.05,
    #         subdivisions=3,
    #     ).apply_translation(np.array([0.55, 0.0, 0.2])),
    # )
    import trimesh.transformations as tra
    # 绕y轴旋转90度
    capsule_mesh = trimesh.creation.capsule(
        radius=capsule_radius,
        height=capsule_height,
        count=[32, 32]
    )
    # rot_mat = tra.rotation_matrix(np.pi/2, [0,1,0], point=None)
    rot_mat = np.eye(4)
    capsule_mesh.apply_transform(rot_mat)
    capsule_mesh.apply_translation(capsule_center)
    server.scene.add_mesh_trimesh(
        "capsule_obstacle",
        capsule_mesh,
    )
    for name, pos in zip(["start", "end"], [start_pos, end_pos]):
        server.scene.add_frame(
            f"/{name}",
            position=pos,
            wxyz=down_wxyz,
            axes_length=0.05,
            axes_radius=0.01,
        )

    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps

        urdf_vis.update_cfg(traj[slider.value])
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    tyro.cli(main)
