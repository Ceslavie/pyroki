from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Float, Array

from ._geometry import HalfSpace, Sphere, Capsule, Box, Heightmap
from . import _utils


# --- HalfSpace Collision Implementations ---


def _halfspace_sphere_dist(
    halfspace_normal: Float[Array, "*batch 3"],
    halfspace_point: Float[Array, "*batch 3"],
    sphere_pos: Float[Array, "*batch 3"],
    sphere_radius: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """Helper: Calculates distance between a halfspace boundary plane and sphere center, minus radius."""
    dist = (
        jnp.einsum("...i,...i->...", sphere_pos - halfspace_point, halfspace_normal)
        - sphere_radius
    )
    return dist


def halfspace_sphere(halfspace: HalfSpace, sphere: Sphere) -> Float[Array, "*batch"]:
    """Calculates distance between a halfspace and a sphere."""
    dist = _halfspace_sphere_dist(
        halfspace.normal,
        halfspace.pose.translation(),
        sphere.pose.translation(),
        sphere.radius,
    )
    return dist


def halfspace_capsule(halfspace: HalfSpace, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculates distance between halfspace and capsule (closest end)."""
    halfspace_normal = halfspace.normal
    halfspace_point = halfspace.pose.translation()
    cap_center = capsule.pose.translation()
    cap_radius = capsule.radius
    cap_axis = capsule.axis
    segment_offset = cap_axis * capsule.length[..., None]
    dist1 = _halfspace_sphere_dist(
        halfspace_normal, halfspace_point, cap_center + segment_offset, cap_radius
    )
    dist2 = _halfspace_sphere_dist(
        halfspace_normal, halfspace_point, cap_center - segment_offset, cap_radius
    )
    final_dist = jnp.minimum(dist1, dist2)
    return final_dist


def halfspace_box(halfspace: HalfSpace, box: Box) -> Float[Array, "*batch"]:
    """Calculates distance between halfspace and box."""
    halfspace_normal = halfspace.normal
    halfspace_point = halfspace.pose.translation()

    box_pose = box.pose
    box_extents = box.extents
    half_extents = box_extents / 2.0

    local_vertices = (
        jnp.array(
            [
                [-1, -1, -1],
                [-1, -1, +1],
                [-1, +1, -1],
                [-1, +1, +1],
                [+1, -1, -1],
                [+1, -1, +1],
                [+1, +1, -1],
                [+1, +1, +1],
            ]
        )
        * half_extents[..., None, :]
    )
    world_vertices = box_pose.apply(local_vertices)
    hs_normal_b = halfspace_normal[..., None, :]
    hs_point_b = halfspace_point[..., None, :]
    vertex_distances = jnp.einsum(
        "...vi,...vi->...v", world_vertices - hs_point_b, hs_normal_b
    )
    min_vertex_distance = jnp.min(vertex_distances, axis=-1)
    return min_vertex_distance


# --- Sphere/Capsule/Box Collision Implementations ---


def _sphere_sphere_dist(
    pos1: Float[Array, "*batch 3"],
    radius1: Float[Array, "*batch"],
    pos2: Float[Array, "*batch 3"],
    radius2: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """Helper: Calculates distance between two spheres."""
    _, dist_center = _utils.normalize_with_norm(pos2 - pos1)
    dist = dist_center - (radius1 + radius2)
    return dist


def sphere_sphere(sphere1: Sphere, sphere2: Sphere) -> Float[Array, "*batch"]:
    """Calculate distance between two spheres."""
    dist = _sphere_sphere_dist(
        sphere1.pose.translation(),
        sphere1.radius,
        sphere2.pose.translation(),
        sphere2.radius,
    )
    return dist


def sphere_capsule(sphere: Sphere, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculate distance between sphere and capsule."""
    cap_pos = capsule.pose.translation()
    sphere_pos = sphere.pose.translation()
    cap_axis = capsule.axis
    segment_offset = cap_axis * capsule.length[..., None]
    cap_a = cap_pos - segment_offset
    cap_b = cap_pos + segment_offset
    pt_on_axis = _utils.closest_segment_point(cap_a, cap_b, sphere_pos)
    dist = _sphere_sphere_dist(sphere_pos, sphere.radius, pt_on_axis, capsule.radius)
    return dist


def sphere_box(sphere: Sphere, box: Box) -> Float[Array, "*batch"]:
    """Calculate distance between sphere and box."""
    sphere_center_world = sphere.pose.translation()
    sphere_radius = sphere.radius
    box_pose = box.pose
    box_half_extents = box.extents / 2.0
    sphere_center_local = box_pose.inverse().apply(sphere_center_world)
    closest_point_local = jnp.clip(
        sphere_center_local, -box_half_extents, box_half_extents
    )
    closest_point_world = box_pose.apply(closest_point_local)
    dist_center_to_box = jnp.linalg.norm(
        sphere_center_world - closest_point_world, axis=-1
    )
    dist = dist_center_to_box - sphere_radius
    return dist


def capsule_capsule(capsule1: Capsule, capsule2: Capsule) -> Float[Array, "*batch"]:
    """Calculate distance between two capsules."""
    pos1 = capsule1.pose.translation()
    axis1 = capsule1.axis
    length1 = capsule1.length
    radius1 = capsule1.radius
    segment1_offset = axis1 * length1[..., None]
    a1 = pos1 - segment1_offset
    b1 = pos1 + segment1_offset

    pos2 = capsule2.pose.translation()
    axis2 = capsule2.axis
    length2 = capsule2.length
    radius2 = capsule2.radius
    segment2_offset = axis2 * length2[..., None]
    a2 = pos2 - segment2_offset
    b2 = pos2 + segment2_offset

    pt1_on_axis, pt2_on_axis = _utils.closest_segment_to_segment_points(a1, b1, a2, b2)
    dist = _sphere_sphere_dist(pt1_on_axis, radius1, pt2_on_axis, radius2)
    return dist


def capsule_box(capsule: Capsule, box: Box) -> Float[Array, "*batch"]:
    """Calculate approximate distance between capsule and box."""
    cap_center_world = capsule.pose.translation()
    cap_radius = capsule.radius
    cap_axis = capsule.axis
    cap_length = capsule.length  # Half-length
    box_pose = box.pose
    box_half_extents = box.extents / 2.0
    cap_center_local = box_pose.inverse().apply(cap_center_world)
    closest_point_on_box_local = jnp.clip(
        cap_center_local, -box_half_extents, box_half_extents
    )
    closest_point_on_box_world = box_pose.apply(closest_point_on_box_local)
    segment_offset = cap_axis * cap_length[..., None]
    cap_a = cap_center_world - segment_offset
    cap_b = cap_center_world + segment_offset
    pt_on_axis = _utils.closest_segment_point(cap_a, cap_b, closest_point_on_box_world)
    dist_axis_to_box = jnp.linalg.norm(closest_point_on_box_world - pt_on_axis, axis=-1)
    dist = dist_axis_to_box - cap_radius
    return dist


# --- Heightmap Collision Implementations ---


def heightmap_sphere(heightmap: Heightmap, sphere: Sphere) -> Float[Array, "*batch"]:
    """Calculate approximate distance between heightmap and sphere.

    Approximation: Considers the heightmap point directly below the sphere center
    using bilinear interpolation and calculates vertical distance minus radius.
    """
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), sphere.get_batch_axes()
    )

    sphere_pos_w = sphere.pose.translation()
    sphere_radius = sphere.radius
    interpolated_local_z = heightmap._interpolate_height_at_coords(sphere_pos_w)
    sphere_pos_h = heightmap.pose.inverse().apply(sphere_pos_w)
    sphere_local_z = sphere_pos_h[..., 2]
    dist = sphere_local_z - interpolated_local_z - sphere_radius

    assert dist.shape == batch_axes
    return dist


def heightmap_capsule(heightmap: Heightmap, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculate approximate distance between heightmap and capsule.

    Approximation: Considers the heightmap point directly below each of the
    capsule's two end-sphere centers. Calculates the vertical distance for each
    (minus capsule radius) and returns the minimum.
    Uses the Heightmap._interpolate_height_at_coords helper.

    Limitation: Does not accurately capture collisions involving the cylindrical
    body of the capsule if both endpoints are above the heightmap.
    """
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), capsule.get_batch_axes()
    )

    cap_pos_w = capsule.pose.translation()
    cap_radius = capsule.radius
    cap_axis_w = capsule.axis  # World frame axis
    segment_offset_w = cap_axis_w * capsule.length[..., None]

    # Calculate world positions of the two end-sphere centers.
    p1_w = cap_pos_w + segment_offset_w
    p2_w = cap_pos_w - segment_offset_w

    # Interpolate heightmap surface height (local Z) below each end-sphere center.
    h_surf1_local = heightmap._interpolate_height_at_coords(p1_w)
    h_surf2_local = heightmap._interpolate_height_at_coords(p2_w)

    # Get end-sphere centers Z coordinates in heightmap's local frame.
    p1_h = heightmap.pose.inverse().apply(p1_w)
    p2_h = heightmap.pose.inverse().apply(p2_w)
    z1_local = p1_h[..., 2]
    z2_local = p2_h[..., 2]

    # Calculate vertical distance for each end sphere.
    dist1 = z1_local - h_surf1_local - cap_radius
    dist2 = z2_local - h_surf2_local - cap_radius

    # Return the minimum distance.
    min_dist = jnp.minimum(dist1, dist2)
    assert min_dist.shape == batch_axes
    return min_dist


def heightmap_halfspace(
    heightmap: Heightmap, halfspace: HalfSpace
) -> Float[Array, "*batch"]:
    """Calculate approximate distance between heightmap and halfspace.

    Approximation: Finds the minimum signed distance between any heightmap vertex
    and the halfspace plane.
    """
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), halfspace.get_batch_axes()
    )

    # Heightmap vertices in world frame.
    verts_local = heightmap._get_vertices_local()  # (*batch, N, 3), N=H*W
    verts_world = heightmap.pose.apply(verts_local)  # (*batch, N, 3)

    # Halfspace plane properties (world frame).
    hs_normal_w = halfspace.normal  # (*batch, 3)
    hs_point_w = halfspace.pose.translation()  # (*batch, 3)

    # Ensure batch dimensions are compatible for broadcasting.
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), halfspace.get_batch_axes()
    )
    # Expand dims for broadcasting against vertices.
    hs_normal_w = jnp.broadcast_to(hs_normal_w, batch_axes + (3,))[..., None, :]
    hs_point_w = jnp.broadcast_to(hs_point_w, batch_axes + (3,))[..., None, :]
    verts_world = jnp.broadcast_to(verts_world, batch_axes + verts_world.shape[-2:])

    # Calculate signed distance for each vertex to the plane:
    # dist = dot(vertex - plane_point, plane_normal)
    vertex_distances = jnp.einsum(
        "...vi,...i->...v", verts_world - hs_point_w, hs_normal_w.squeeze(-2)
    )

    # Find the minimum distance among all vertices.
    min_dist = jnp.min(vertex_distances, axis=-1)
    assert min_dist.shape == batch_axes
    return min_dist


def heightmap_box(heightmap: Heightmap, box: Box) -> Float[Array, "*batch"]:
    """Calculate approximate distance between heightmap and box.

    Approximation: Finds the minimum vertical distance between the heightmap
    surface (interpolated) and each of the box's 8 corner vertices.
    Uses the Heightmap._interpolate_height_at_coords helper.

    Limitation: Very coarse approximation, ignores box faces and edges.
    """
    batch_axes = jnp.broadcast_shapes(heightmap.get_batch_axes(), box.get_batch_axes())

    # Box properties.
    box_pose = box.pose
    box_extents = box.extents
    box_half_extents = box_extents / 2.0

    # 1. Calculate Box corners.
    local_corners = (
        jnp.array(
            [
                [-1, -1, -1],
                [-1, -1, +1],
                [-1, +1, -1],
                [-1, +1, +1],
                [+1, -1, -1],
                [+1, -1, +1],
                [+1, +1, -1],
                [+1, +1, +1],
            ]
        )
        * box_half_extents[..., None, :]
    )
    world_corners = box_pose.apply(local_corners)
    assert world_corners.shape == (*batch_axes, 8, 3)

    # 2. Interpolate heightmap height below each corner
    h_surf_local_corners = heightmap._interpolate_height_at_coords(world_corners)
    assert h_surf_local_corners.shape == (*batch_axes, 8)

    # 3. Get corner Z coordinates in heightmap local frame
    corners_h = heightmap.pose.inverse().apply(world_corners)
    corners_local_z = corners_h[..., 2]
    assert corners_local_z.shape == (*batch_axes, 8)

    # 4. Calculate vertical distances for each corner.
    dists_corners = corners_local_z - h_surf_local_corners
    assert dists_corners.shape == (*batch_axes, 8)

    # 5. Return minimum distance.
    min_dist = jnp.min(dists_corners, axis=-1)
    assert min_dist.shape == batch_axes
    return min_dist
