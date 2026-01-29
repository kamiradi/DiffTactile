"""
Sanity check: Learn FEM material parameters (E, nu) from tactile marker displacements.

Setup:
- SOURCE sensor: Ground truth parameters, generates observed marker displacements
- TARGET sensor: Learnable parameters, generates predicted marker displacements
- Same trajectory and object for both
- Loss: ||predicted_displacement - observed_displacement||²
- Backprop through Taichi autodiff to learn E, nu
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from difftactile.sensor_model.fem_sensor import FEMDomeSensor
from difftactile.object_model.rigid_dynamic import RigidObj

TI_TYPE = ti.f32
NP_TYPE = np.float32

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Module-level logger
logger = logging.getLogger(__name__)


@ti.data_oriented
class SensorParamEstimation:
    """
    Two-sensor setup for parameter estimation:
    - source_sensor: GT parameters, frozen
    - target_sensor: learnable parameters
    """

    def __init__(
        self,
        dt=5e-5,
        total_steps=100,
        sub_steps=50,
        obj_name=None,
        visualize=False,
        grad_check=False,
    ):
        self.dt = dt
        self.total_steps = total_steps
        self.sub_steps = sub_steps
        self.dim = 3
        self.visualize = visualize
        self.grad_check = grad_check

        # GUI windows for visualization (created later if needed)
        self.gui_gt = None
        self.gui_pred = None
        self.gui_contact = None

        # View angles for 3D perspective projection
        self.view_phi = 0
        self.view_theta = 0

        # Initialize Taichi fields for learnable parameters BEFORE creating sensors
        # These will be used by target sensor
        self.E_target = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.nu_target = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.mu_target = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.lam_target = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        # Two sensors: source (GT) and target (learnable)
        init_img_path = os.path.join(SCRIPT_DIR, "init.png")
        self.source_sensor = FEMDomeSensor(dt, sub_steps, init_img_path=init_img_path)
        self.target_sensor = FEMDomeSensor(dt, sub_steps, init_img_path=init_img_path)

        # Single object (both sensors interact with same object)
        self.space_scale = 10.0
        self.obj_scale = 4.0
        self.mpm_object = RigidObj(
            dt=dt,
            sub_steps=sub_steps,
            obj_name=obj_name,
            space_scale=self.space_scale,
            obj_scale=self.obj_scale,
            density=1.50,
            rho=0.3,
        )

        # Contact parameters (frozen, same for both sensors)
        self.kn = 55.0
        self.kd = 269.44
        self.kt = 108.72
        self.friction_coeff = 14.16
        self.norm_eps = 1e-11

        # Trajectory (same for both sensors)
        self.p_sensor = ti.Vector.field(
            self.dim, dtype=ti.f32, shape=(self.total_steps)
        )
        self.o_sensor = ti.Vector.field(
            self.dim, dtype=ti.f32, shape=(self.total_steps)
        )

        # Marker displacements storage
        self.num_markers = self.source_sensor.num_markers
        self.observed_displacement = ti.Vector.field(
            2, float, shape=(self.total_steps, self.num_markers)
        )
        self.predicted_displacement = ti.Vector.field(
            2, float, shape=(self.total_steps, self.num_markers), needs_grad=True
        )

        # Loss
        self.loss = ti.field(float, (), needs_grad=True)

        # Contact detection index
        self.contact_idx = ti.Vector.field(
            1,
            dtype=int,
            shape=(
                self.sub_steps,
                self.mpm_object.n_grid,
                self.mpm_object.n_grid,
                self.mpm_object.n_grid,
            ),
        )

        self.init_trajectory()

        # Visualization fields for 3D perspective projection (adapted from object_repose.py)
        # These need to be created after sensors and object are initialized
        self.draw_pos2 = ti.Vector.field(
            2, float, self.source_sensor.n_verts
        )  # sensor vertices
        self.draw_pos3 = ti.Vector.field(
            2, float, self.mpm_object.n_particles
        )  # object particles

        # Store GT marker data for visualization (numpy arrays, set after generate_observations)
        self.gt_init_markers = None
        self.gt_cur_markers = None

        # Initialize GUI windows if visualization is enabled
        if self.visualize:
            self.gui_gt = ti.GUI("GT Markers", res=(640, 480))
            self.gui_pred = ti.GUI("Predicted Markers", res=(640, 480))
            self.gui_contact = ti.GUI("Contact Viz", res=(512, 512))

    def draw_markers(self, init_markers, cur_markers, gui, title=None):
        """Draw markers with displacement arrows on GUI window.

        Adapted from object_repose.py draw_markers method.
        """
        img_height = 480
        img_width = 640
        scale = img_width
        rescale = 1.8
        draw_points = rescale * (init_markers - [320, 240]) / scale + [0.5, 0.5]
        offset = rescale * (cur_markers - init_markers) / scale
        gui.circles(draw_points, radius=2, color=0xF542A1)
        gui.arrows(draw_points, 10.0 * offset, radius=2, color=0xE6C949)
        if title:
            gui.text(title, pos=(0.05, 0.95), color=0xFFFFFF)
        gui.show()

    def update_gt_visualization(self):
        """Update GT marker visualization window."""
        if not self.visualize or self.gui_gt is None:
            return

        # Use stored GT marker data (set during generate_observations)
        if self.gt_init_markers is not None and self.gt_cur_markers is not None:
            self.draw_markers(
                self.gt_init_markers, self.gt_cur_markers, self.gui_gt, "GT Markers"
            )
        else:
            # Fallback: try to extract from source sensor
            self.source_sensor.extract_markers(self.sub_steps - 2)
            init_markers = self.source_sensor.virtual_markers.to_numpy()
            cur_markers = self.source_sensor.predict_markers.to_numpy()
            self.draw_markers(init_markers, cur_markers, self.gui_gt, "GT Markers")

    def update_pred_visualization(self):
        """Update predicted marker visualization window."""
        if not self.visualize or self.gui_pred is None:
            return

        # Get current markers from target sensor
        self.target_sensor.extract_markers(self.sub_steps - 2)
        init_markers = self.target_sensor.virtual_markers.to_numpy()
        cur_markers = self.target_sensor.predict_markers.to_numpy()

        self.draw_markers(
            init_markers,
            cur_markers,
            self.gui_pred,
            f"Predicted (E={self.E_target[None]:.0f}, nu={self.nu_target[None]:.3f})",
        )

    @ti.kernel
    def draw_perspective(self, sensor: ti.template(), f: ti.i32):
        """Project 3D sensor and object positions to 2D for visualization.

        Adapted from object_repose.py draw_perspective method.
        """
        phi = ti.math.radians(self.view_phi)
        theta = ti.math.radians(self.view_theta)
        c_p, s_p = ti.math.cos(phi), ti.math.sin(phi)
        c_t, s_t = ti.math.cos(theta), ti.math.sin(theta)
        offset = 0.2

        # Project sensor vertices
        for i in range(sensor.n_verts):
            x = sensor.pos[f, i][0] - offset
            y = sensor.pos[f, i][1] - offset
            z = sensor.pos[f, i][2] - offset
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            u, v = xx, y * c_t + zz * s_t
            self.draw_pos2[i][0] = u + 0.2
            self.draw_pos2[i][1] = v + 0.5

        # Project object particles
        for i in range(self.mpm_object.n_particles):
            x = self.mpm_object.x_0[f, i][0] - offset
            y = self.mpm_object.x_0[f, i][1] - offset
            z = self.mpm_object.x_0[f, i][2] - offset
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            u, v = xx, y * c_t + zz * s_t
            self.draw_pos3[i][0] = u + 0.2
            self.draw_pos3[i][1] = v + 0.5

    def update_contact_visualization(self, sensor, title=None):
        """Update contact visualization window showing sensor-object interaction.

        Adapted from object_repose.py visualization code.
        """
        if not self.visualize or self.gui_contact is None:
            return

        viz_scale = 0.1
        viz_offset = np.array([0.0, 0.0])

        # Project 3D positions to 2D
        self.draw_perspective(sensor, 0)

        # Draw object particles (blue)
        self.gui_contact.circles(
            viz_scale * self.draw_pos3.to_numpy() + viz_offset, radius=2, color=0x039DFC
        )
        # Draw sensor vertices (yellow)
        self.gui_contact.circles(
            viz_scale * self.draw_pos2.to_numpy() + viz_offset, radius=2, color=0xE6C949
        )

        if title:
            self.gui_contact.text(title, pos=(0.05, 0.95), color=0xFFFFFF)

        self.gui_contact.show()

    @ti.kernel
    def init_trajectory(self):
        """Pressing trajectory - velocity commands."""
        for i in range(self.total_steps):
            # Pressing motion matching object_repose.py
            self.p_sensor[i] = ti.Vector([0.0, 1.5, 0.0])
            self.o_sensor[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def compute_lame_params(self):
        """Compute Lame parameters from E and nu - THIS IS DIFFERENTIABLE."""
        # Dummy loop required for tape autodiff compatibility
        # (Taichi can't mix looping and non-looping kernels in the same tape)
        for _ in range(1):
            E = self.E_target[None]
            nu = self.nu_target[None]
            self.mu_target[None] = E / (2.0 * (1.0 + nu))
            self.lam_target[None] = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @ti.kernel
    def apply_target_params(self):
        """Copy computed Lame params to target sensor."""
        # Dummy loop required for tape autodiff compatibility
        for _ in range(1):
            self.target_sensor.mu[None] = self.mu_target[None]
            self.target_sensor.lam[None] = self.lam_target[None]

    def set_source_params(self, E, nu):
        """Set source sensor parameters (GT, frozen)."""
        self.source_sensor.set_material_params(E, nu)

    def set_target_params(self, E, nu):
        """Set target sensor learnable parameters."""
        # Check for nu approaching singularity (1 - 2*nu -> 0)
        if nu > 0.48:
            logger.warning(
                f"nu={nu:.4f} is close to 0.5, Lame parameter lambda will be very large (singularity at nu=0.5)"
            )

        self.E_target[None] = E
        self.nu_target[None] = nu
        self.compute_lame_params()
        self.apply_target_params()

    def _check_nan(self, name, value):
        """Check if value is NaN and log warning."""
        if np.isnan(value):
            logger.warning(f"NaN detected in {name}: {value}")
            return True
        return False

    def _check_grads_for_nan(self):
        """Check all gradient fields for NaN values and log warnings."""
        has_nan = False
        has_nan |= self._check_nan("grad_E", self.E_target.grad[None])
        has_nan |= self._check_nan("grad_nu", self.nu_target.grad[None])
        has_nan |= self._check_nan("grad_mu", self.mu_target.grad[None])
        has_nan |= self._check_nan("grad_lam", self.lam_target.grad[None])
        has_nan |= self._check_nan("loss", self.loss[None])
        return has_nan

    def _check_field_grad_nan(self, field, name, ts=None):
        """Check a field's gradient for NaN values.

        Args:
            field: Taichi field with .grad attribute
            name: Name of the field for logging
            ts: Optional timestep index to check specific slice

        Returns:
            True if NaN detected, False otherwise
        """
        if ts is not None:
            grad_np = field.grad.to_numpy()[ts]
        else:
            grad_np = field.grad.to_numpy()

        if np.isnan(grad_np).any():
            nan_count = np.isnan(grad_np).sum()
            logger.debug(
                f"NaN in {name}.grad at ts={ts}: {nan_count}/{grad_np.size} values"
            )
            logger.debug(f"  NaN indices: {np.argwhere(np.isnan(grad_np)).tolist()}")
            return True
        return False

    def _debug_vertex_projection(self, sensor, vertex_idx, f, ts):
        """Debug helper to check if a vertex causes acos singularity in projection.

        Args:
            sensor: The FEMDomeSensor instance
            vertex_idx: Index of the vertex to debug
            f: Substep index for position lookup
            ts: Outer timestep (for logging context)
        """
        surface_id_np = sensor.surface_id.to_numpy()

        if vertex_idx not in surface_id_np:
            logger.debug(f"Vertex {vertex_idx} is NOT a surface vertex")
            return

        surface_idx = np.where(surface_id_np == vertex_idx)[0][0]

        # Get position
        pos = sensor.pos.to_numpy()[f, vertex_idx]

        # Get transformation matrix
        inv_trans_h = sensor.inv_trans_h.to_numpy()

        # Compute cam_pos like extract_markers does
        hom_pos = np.array([pos[0], pos[1], pos[2], 1.0])
        inv_pos = inv_trans_h @ hom_pos
        cam_pos = np.array([inv_pos[0], inv_pos[2], inv_pos[1]])

        # Store pre-offset cam_pos for logging
        cam_pos_before = cam_pos.copy()

        # Add offset like project_3d_2d does: a[2] += 2.0*0.01
        cam_pos[2] += 0.02
        a_norm = np.linalg.norm(cam_pos)
        cos_val = cam_pos[2] / (a_norm + 1e-10)

        # Check for singularity
        singularity_warning = ""
        if abs(cos_val) >= 0.9999:
            singularity_warning = " (WARNING: near acos singularity at ±1!)"
        elif abs(cos_val) >= 0.99:
            singularity_warning = " (CAUTION: approaching acos singularity)"

        logger.debug(f"Vertex {vertex_idx} projection debug at outer_ts={ts}:")
        logger.debug(f"  Is surface vertex: True (surface_idx={surface_idx})")
        logger.debug(f"  pos[{f}, {vertex_idx}] = {pos}")
        logger.debug(f"  cam_pos (after transform) = {cam_pos_before}")
        logger.debug(f"  cam_pos (after z-offset) = {cam_pos}")
        logger.debug(f"  a_norm = {a_norm:.6f}")
        logger.debug(f"  cos = {cos_val:.6f}{singularity_warning}")

    def _debug_grad_summary(self, name, grad_np, ts=None):
        """Log compact stats for a gradient array."""
        if grad_np.size == 0:
            logger.debug(f"{name}.grad empty at ts={ts}")
            return
        max_abs = float(np.max(np.abs(grad_np)))
        mean_abs = float(np.mean(np.abs(grad_np)))
        any_nonzero = bool(np.any(grad_np != 0))
        logger.debug(
            f"{name}.grad summary at ts={ts}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, any_nonzero={any_nonzero}"
        )

    def init_scene(self, sensor):
        """Initialize object and sensor positions."""
        # Object
        ball_pos = [3.2, 1.0, 5.0]
        ball_ori = [0.0, 0.0, 90.0]
        ball_vel = [0.0, 0.0, 0.0]
        self.mpm_object.init(ball_pos, ball_ori, ball_vel)

        # Sensor
        rx, ry, rz = 0.0, 0.0, 90.0
        tx, ty, tz = 7.0, 1.5, 5.0
        sensor.init(rx, ry, rz, tx, ty, tz)

    @ti.kernel
    def set_pos_control(self, sensor: ti.template(), f: ti.i32):
        # Dummy loop required for tape autodiff compatibility
        for _ in range(1):
            sensor.d_pos[None] = self.p_sensor[f]
            sensor.d_ori[None] = self.o_sensor[f]

    @ti.func
    def calculate_contact_force(self, sdf, norm_v, relative_v):
        """Penalty-based contact (frozen params)."""
        shear_factor = ti.Vector([0.0, 0.0, 0.0])
        normal_vel = ti.max(norm_v.dot(relative_v), 0.0)
        normal_factor = -(self.kn + self.kd * normal_vel) * sdf * norm_v

        shear_vel = relative_v - norm_v.dot(relative_v) * norm_v
        shear_vel_norm = shear_vel.norm(self.norm_eps)

        if shear_vel_norm > 1e-4:
            max_shear = self.friction_coeff * normal_factor.norm(self.norm_eps)
            shear_factor = (shear_vel / shear_vel_norm) * ti.min(
                self.kt * shear_vel_norm, max_shear
            )

        return normal_factor + shear_factor

    @ti.kernel
    def check_collision(self, sensor: ti.template(), f: ti.i32):
        for i, j, k in ti.ndrange(
            self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid
        ):
            if self.mpm_object.grid_occupy[f, i, j, k] == 1:
                cur_p = ti.Vector(
                    [
                        (i + 0.5) * self.mpm_object.dx_0,
                        (j + 0.5) * self.mpm_object.dx_0,
                        (k + 0.5) * self.mpm_object.dx_0,
                    ]
                )
                min_idx = sensor.find_closest(cur_p, f)
                self.contact_idx[f, i, j, k][0] = min_idx

    @ti.kernel
    def collision(self, sensor: ti.template(), f: ti.i32):
        for i, j, k in ti.ndrange(
            self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid
        ):
            if self.mpm_object.grid_occupy[f, i, j, k] == 1:
                cur_p = ti.Vector(
                    [
                        (i + 0.5) * self.mpm_object.dx_0,
                        (j + 0.5) * self.mpm_object.dx_0,
                        (k + 0.5) * self.mpm_object.dx_0,
                    ]
                )
                cur_v = self.mpm_object.grid_v_in[f, i, j, k] / (
                    self.mpm_object.grid_m[f, i, j, k] + self.mpm_object.eps
                )
                min_idx = self.contact_idx[f, i, j, k][0]
                sdf, norm_v, relative_v, contact_flag = sensor.find_sdf(
                    cur_p, cur_v, min_idx, f
                )

                if contact_flag:
                    ext_force = self.calculate_contact_force(sdf, -norm_v, -relative_v)
                    self.mpm_object.update_contact_force(ext_force, f, i, j, k)
                    sensor.update_contact_force(min_idx, -ext_force, f)

    def update_step(self, sensor, f):
        """One simulation substep."""
        self.mpm_object.compute_new_F(f)
        self.mpm_object.svd(f)
        self.mpm_object.p2g(f)
        sensor.update(f)
        self.mpm_object.check_grid_occupy(f)
        self.check_collision(sensor, f)
        self.collision(sensor, f)
        self.mpm_object.grid_op(f)
        self.mpm_object.g2p(f)
        self.mpm_object.compute_COM(f)
        self.mpm_object.compute_H(f)
        self.mpm_object.compute_H_svd(f)
        self.mpm_object.compute_R(f)
        sensor.update2(f)

    def update_step_grad(self, sensor, f, debug_substep=False):
        """Backward through one substep."""
        sensor.update2.grad(f)

        if debug_substep:
            vel_grad = sensor.vel.grad.to_numpy()[f]
            logger.debug(f"      After update2.grad: vel.grad[{f}] max={np.max(np.abs(vel_grad)):.6e}, any_nonzero={np.any(vel_grad != 0)}")

        self.mpm_object.compute_R.grad(f)
        self.mpm_object.compute_H_svd_grad(f)
        self.mpm_object.compute_H.grad(f)
        self.mpm_object.compute_COM.grad(f)
        self.mpm_object.g2p.grad(f)
        self.mpm_object.grid_op.grad(f)
        self.collision.grad(sensor, f)
        sensor.update.grad(f)

        if debug_substep:
            logger.debug(f"      After update.grad: mu.grad={sensor.mu.grad[None]:.6e}, lam.grad={sensor.lam.grad[None]:.6e}")

        self.mpm_object.p2g.grad(f)
        self.mpm_object.svd_grad(f)
        self.mpm_object.compute_new_F.grad(f)

    @ti.kernel
    def extract_displacement_source(self, t: ti.i32):
        """Extract marker displacement for source sensor -> observed."""
        for i in range(self.num_markers):
            self.observed_displacement[t, i] = (
                self.source_sensor.predict_markers[i]
                - self.source_sensor.virtual_markers[i]
            )

    @ti.kernel
    def extract_displacement_target(self, t: ti.i32):
        """Extract marker displacement for target sensor -> predicted."""
        for i in range(self.num_markers):
            self.predicted_displacement[t, i] = (
                self.target_sensor.predict_markers[i]
                - self.target_sensor.virtual_markers[i]
            )  # virtual markers is not changing, so no grad needed

    @ti.kernel
    def compute_loss_at_timestep(self, t: ti.i32):
        """MSE loss between predicted and observed displacements."""
        for i in range(self.num_markers):
            diff = self.predicted_displacement[t, i] - self.observed_displacement[t, i]
            self.loss[None] += diff.dot(diff)
            # ti.print("-"*2 + f"Forward loss for marker {i}:{diff.dot(diff):.6e}")

    @ti.kernel
    def compute_total_loss(self):
        """Compute total MSE loss across all timesteps and markers (for tape autodiff)."""
        for t, i in ti.ndrange(self.total_steps, self.num_markers):
            diff = self.predicted_displacement[t, i] - self.observed_displacement[t, i]
            self.loss[None] += diff.dot(diff)

    def reset_sensor(self, sensor):
        """Reset sensor and object state."""
        sensor.reset_contact()
        self.mpm_object.reset()
        self.contact_idx.fill(-1)

    def memory_to_cache(self, sensor, t):
        sensor.memory_to_cache(t)
        self.mpm_object.memory_to_cache(t)

    def memory_from_cache(self, sensor, t):
        sensor.memory_from_cache(t)
        self.mpm_object.memory_from_cache(t)

    def clear_grads(self):
        """Clear all gradients."""
        self.loss[None] = 0.0
        self.loss.grad[None] = 1.0
        self.E_target.grad[None] = 0.0
        self.nu_target.grad[None] = 0.0
        self.mu_target.grad[None] = 0.0
        self.lam_target.grad[None] = 0.0
        self.predicted_displacement.grad.fill(0.0)
        self.target_sensor.clear_loss_grad()
        self.mpm_object.clear_loss_grad()
        self.target_sensor.clear_step_grad(self.sub_steps)
        self.mpm_object.clear_step_grad(self.sub_steps)

    def generate_observations(self, E_gt, nu_gt):
        """Run source sensor with GT params to generate observations."""
        print(
            f"Generating observations with source sensor (E={E_gt:.1f}, nu={nu_gt:.3f})"
        )
        self.set_source_params(E_gt, nu_gt)
        self.init_scene(self.source_sensor)

        # loop to move the tactile sensor/robot
        for ts in range(self.total_steps):
            self.set_pos_control(self.source_sensor, ts)
            self.source_sensor.set_pose_control()
            self.source_sensor.set_control_vel(0)
            self.source_sensor.set_vel(0)
            self.reset_sensor(self.source_sensor)

            # loop to advance FEM simulation --- computes marker deformation
            for ss in range(self.sub_steps - 1):
                self.update_step(self.source_sensor, ss)

            # Extract observed marker displacement
            self.source_sensor.extract_markers(self.sub_steps - 2)
            self.extract_displacement_source(ts)
            self.gt_init_markers = self.source_sensor.virtual_markers.to_numpy().copy()
            self.gt_cur_markers = self.source_sensor.predict_markers.to_numpy().copy()
            self.update_gt_visualization()
            self.update_contact_visualization(self.source_sensor, "GT Contact")

            self.memory_to_cache(self.source_sensor, ts)

        # Store GT marker data for visualization (from final timestep)
        # self.source_sensor.extract_markers(self.sub_steps - 2)
        # self.gt_init_markers = self.source_sensor.virtual_markers.to_numpy().copy()
        # self.gt_cur_markers = self.source_sensor.predict_markers.to_numpy().copy()

        # Report
        final_disp = self.observed_displacement.to_numpy()[self.total_steps - 1]
        mean_disp = np.mean(np.linalg.norm(final_disp, axis=1))
        max_disp = np.max(np.linalg.norm(final_disp, axis=1))
        print(f"  Mean displacement at final step: {mean_disp:.4f} pixels")
        print(f"  Max displacement at final step: {max_disp:.4f} pixels")

        # Update GT visualization
        # self.update_gt_visualization()

        # Update contact visualization (showing GT sensor-object interaction)
        # self.update_contact_visualization(self.source_sensor, "GT Contact")

    def forward_target(self):
        """Run target sensor forward pass."""
        logger.debug(
            f"Forward pass starting with E={self.E_target[None]:.1f}, nu={self.nu_target[None]:.4f}"
        )

        # Apply current E, nu to compute mu, lam
        self.compute_lame_params()
        self.apply_target_params()

        logger.debug(
            f"Lame params: mu={self.mu_target[None]:.4f}, lam={self.lam_target[None]:.4f}"
        )

        self.init_scene(self.target_sensor)

        for ts in range(self.total_steps):
            self.set_pos_control(self.target_sensor, ts)
            self.target_sensor.set_pose_control()
            self.target_sensor.set_control_vel(0)
            self.target_sensor.set_vel(0)
            self.reset_sensor(self.target_sensor)

            for ss in range(self.sub_steps - 1):
                self.update_step(self.target_sensor, ss)

            # Extract predicted marker displacement
            self.target_sensor.extract_markers(self.sub_steps - 2)
            self.extract_displacement_target(ts)
            self.update_pred_visualization()
            # Update contact visualization (showing predicted sensor-object interaction)
            self.update_contact_visualization(
                self.target_sensor,
                f"Predicted Contact (E={self.E_target[None]:.0f}, nu={self.nu_target[None]:.3f})",
            )

            # Compute loss for this timestep
            self.compute_loss_at_timestep(ts)

            self.memory_to_cache(self.target_sensor, ts)

        logger.debug(f"Forward pass complete. Loss={self.loss[None]:.6f}")

    def backward_target(self, debug=False):
        """Backward pass through target sensor."""
        if debug:
            logger.debug("Starting backward pass")

        for ts in range(self.total_steps - 1, -1, -1):
            if debug:
                logger.debug("-" * 2 + f"Backward timestep ts={ts}")

            # Backward through loss and marker extraction
            self.compute_loss_at_timestep.grad(ts)
            if self.grad_check:
                grad_np = self.predicted_displacement.grad.to_numpy()[ts]
                self._debug_grad_summary("predicted_displacement", grad_np, ts=ts)
                self._check_field_grad_nan(
                    self.predicted_displacement, "predicted_displacement", ts=ts
                )
            self.extract_displacement_target.grad(ts)
            if self.grad_check:
                grad_np = self.target_sensor.predict_markers.grad.to_numpy()
                self._debug_grad_summary(
                    "target_sensor.predict_markers", grad_np, ts=ts
                )
                self._check_field_grad_nan(
                    self.target_sensor.predict_markers,
                    "target_sensor.predict_markers",
                    ts=None,
                )
            self.target_sensor.extract_markers.grad(self.sub_steps - 2)
            if self.grad_check:
                grad_np = self.target_sensor.pos.grad.to_numpy()[self.sub_steps - 2]
                self._debug_grad_summary(
                    "target_sensor.pos", grad_np, ts=self.sub_steps - 2
                )
                if self._check_field_grad_nan(
                    self.target_sensor.pos, "target_sensor.pos", ts=self.sub_steps - 2
                ):
                    # Debug vertex 1110 which consistently shows NaN
                    self._debug_vertex_projection(
                        self.target_sensor, 1110, self.sub_steps - 2, ts
                    )

            # CRITICAL: Save pos.grad before restore/re-run (it will be cleared)
            saved_pos_grad = self.target_sensor.pos.grad.to_numpy().copy()

            # Restore state
            if ts > 0:
                self.memory_from_cache(self.target_sensor, ts - 1)
            else:
                self.init_scene(self.target_sensor)

            self.set_pos_control(self.target_sensor, ts)
            self.target_sensor.set_pose_control()
            self.target_sensor.set_control_vel(0)
            self.target_sensor.set_vel(0)
            self.reset_sensor(self.target_sensor)

            # Re-run forward
            for ss in range(self.sub_steps - 1):
                self.update_step(self.target_sensor, ss)

            # CRITICAL: Restore pos.grad after re-run forward
            self.target_sensor.pos.grad.from_numpy(saved_pos_grad)

            # Backward through physics
            for ss in range(self.sub_steps - 2, -1, -1):
                # Debug first few substeps and last substep of the last timestep
                is_last_ts = ts == self.total_steps - 1
                is_debug_substep = ss >= self.sub_steps - 4 or ss == 0  # First 3 and last substep
                debug_this_substep = self.grad_check and is_last_ts and is_debug_substep

                if debug_this_substep:
                    logger.debug(f"    Before update_step_grad(ss={ss}):")
                    vel_grad_before = self.target_sensor.vel.grad.to_numpy()[ss]
                    pos_grad_before = self.target_sensor.pos.grad.to_numpy()[ss]
                    logger.debug(f"      vel.grad[{ss}]: max={np.max(np.abs(vel_grad_before)):.6e}, any_nonzero={np.any(vel_grad_before != 0)}")
                    logger.debug(f"      pos.grad[{ss}]: max={np.max(np.abs(pos_grad_before)):.6e}, any_nonzero={np.any(pos_grad_before != 0)}")
                    logger.debug(f"      mu.grad={self.target_sensor.mu.grad[None]:.6e}, lam.grad={self.target_sensor.lam.grad[None]:.6e}")

                self.update_step_grad(self.target_sensor, ss, debug_substep=debug_this_substep)

                if debug_this_substep:
                    logger.debug(f"    After update_step_grad(ss={ss}):")
                    vel_grad_after = self.target_sensor.vel.grad.to_numpy()[ss]
                    pos_grad_after = self.target_sensor.pos.grad.to_numpy()[ss]
                    logger.debug(f"      vel.grad[{ss}]: max={np.max(np.abs(vel_grad_after)):.6e}, any_nonzero={np.any(vel_grad_after != 0)}")
                    logger.debug(f"      pos.grad[{ss}]: max={np.max(np.abs(pos_grad_after)):.6e}, any_nonzero={np.any(pos_grad_after != 0)}")
                    logger.debug(f"      mu.grad={self.target_sensor.mu.grad[None]:.6e}, lam.grad={self.target_sensor.lam.grad[None]:.6e}")

            if self.grad_check:
                mu_grad = self.target_sensor.mu.grad.to_numpy()
                lam_grad = self.target_sensor.lam.grad.to_numpy()
                self._debug_grad_summary("target_sensor.mu", mu_grad, ts=ts)
                self._debug_grad_summary("target_sensor.lam", lam_grad, ts=ts)

            self.target_sensor.set_vel.grad(0)
            self.target_sensor.set_control_vel.grad(0)
            self.target_sensor.set_pose_control.grad()

            # Check for NaN in intermediate gradients
            if debug:
                grad_E_intermediate = self.E_target.grad[None]
                grad_nu_intermediate = self.nu_target.grad[None]
                logger.debug(
                    "-" * 4
                    + f"ts={ts}: grad_E={grad_E_intermediate:.6e}, grad_nu={grad_nu_intermediate:.6e}"
                )
                if np.isnan(grad_E_intermediate) or np.isnan(grad_nu_intermediate):
                    logger.warning(f"NaN gradient detected at timestep ts={ts}")

        # Backward through Lame parameter computation
        if self.grad_check:
            logger.debug(f"Before apply_target_params.grad():")
            logger.debug(f"  target_sensor.mu.grad={self.target_sensor.mu.grad[None]:.6e}")
            logger.debug(f"  target_sensor.lam.grad={self.target_sensor.lam.grad[None]:.6e}")
            logger.debug(f"  mu_target.grad={self.mu_target.grad[None]:.6e}")
            logger.debug(f"  lam_target.grad={self.lam_target.grad[None]:.6e}")

        self.apply_target_params.grad()

        if self.grad_check:
            logger.debug(f"After apply_target_params.grad():")
            logger.debug(f"  mu_target.grad={self.mu_target.grad[None]:.6e}")
            logger.debug(f"  lam_target.grad={self.lam_target.grad[None]:.6e}")

        self.compute_lame_params.grad()

        if self.grad_check:
            logger.debug(f"After compute_lame_params.grad():")
            logger.debug(f"  E_target.grad={self.E_target.grad[None]:.6e}")
            logger.debug(f"  nu_target.grad={self.nu_target.grad[None]:.6e}")

            # logger.debug(
            #     f"Backward pass complete. grad_E={self.E_target.grad[None]:.6e}, grad_nu={self.nu_target.grad[None]:.6e}"
            # )

    def train_step(self, debug=False):
        """One training iteration: forward, loss, backward, return gradients."""
        if debug:
            logger.debug("=" * 40)
            logger.debug("Starting train_step")

        self.clear_grads()
        self.forward_target()
        self.backward_target(debug=debug)

        # Check for NaN gradients after backward pass
        if debug and self._check_grads_for_nan():
            logger.error("NaN detected in gradients after backward pass!")

        result = {
            "loss": self.loss[None],
            "E": self.E_target[None],
            "nu": self.nu_target[None],
            "grad_E": self.E_target.grad[None],
            "grad_nu": self.nu_target.grad[None],
            "grad_mu": self.mu_target.grad[None],
            "grad_lam": self.lam_target.grad[None],
        }

        if debug:
            logger.debug(
                f"train_step complete: loss={result['loss']:.6f}, grad_E={result['grad_E']:.6e}, grad_nu={result['grad_nu']:.6e}"
            )

        return result

    def train_step_tape(self, debug=False):
        """One training iteration using Taichi tape autodiff."""
        logger.debug("=" * 40)
        logger.debug("Starting train_step_tape (using ti.ad.Tape)")

        # Clear gradients and loss
        self.loss[None] = 0.0
        self.E_target.grad[None] = 0.0
        self.nu_target.grad[None] = 0.0
        self.mu_target.grad[None] = 0.0
        self.lam_target.grad[None] = 0.0
        self.predicted_displacement.grad.fill(0.0)
        self.target_sensor.clear_loss_grad()
        self.mpm_object.clear_loss_grad()

        # Use Taichi's tape for automatic differentiation
        with ti.ad.Tape(loss=self.loss):
            # Apply current E, nu to compute mu, lam
            self.compute_lame_params()
            self.apply_target_params()

            logger.debug(
                f"Lame params: mu={self.mu_target[None]:.4f}, lam={self.lam_target[None]:.4f}"
            )

            self.init_scene(self.target_sensor)

            for ts in range(self.total_steps):
                self.set_pos_control(self.target_sensor, ts)
                self.target_sensor.set_pose_control()
                self.target_sensor.set_control_vel(0)
                self.target_sensor.set_vel(0)
                self.reset_sensor(self.target_sensor)

                for ss in range(self.sub_steps - 1):
                    self.update_step(self.target_sensor, ss)

                # Extract predicted marker displacement
                self.target_sensor.extract_markers(self.sub_steps - 2)
                self.extract_displacement_target(ts)

            # Compute loss in single kernel call (required for tape autodiff)
            self.compute_total_loss()

        # Tape automatically computes gradients when context exits

        # Visualization (outside tape to avoid recording non-essential kernels)
        if self.visualize:
            self.target_sensor.extract_markers(self.sub_steps - 2)
            self.update_pred_visualization()
            self.update_contact_visualization(
                self.target_sensor,
                f"Predicted Contact (E={self.E_target[None]:.0f}, nu={self.nu_target[None]:.3f})",
            )

        # Check for NaN gradients
        if self._check_grads_for_nan():
            logger.error("NaN detected in gradients after tape autodiff!")

        result = {
            "loss": self.loss[None],
            "E": self.E_target[None],
            "nu": self.nu_target[None],
            "grad_E": self.E_target.grad[None],
            "grad_nu": self.nu_target.grad[None],
            "grad_mu": self.mu_target.grad[None],
            "grad_lam": self.lam_target.grad[None],
        }

        logger.debug(
            f"train_step_tape complete: loss={result['loss']:.6f}, grad_E={result['grad_E']:.6e}, grad_nu={result['grad_nu']:.6e}"
        )

        return result


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run_experiment(cfg: DictConfig):
    # Print config for reproducibility
    print(OmegaConf.to_yaml(cfg))

    # Configure logging
    log_level = logging.DEBUG if cfg.debug.enabled else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(log_level)

    if cfg.debug.enabled:
        logger.info("Debug mode enabled - verbose logging active")

    ti.init(arch=ti.gpu, device_memory_GB=4)

    # Ground truth parameters (from config)
    E_gt = cfg.params.E_gt
    nu_gt = cfg.params.nu_gt

    # Initial parameters (from config)
    E_init = cfg.params.E_init
    nu_init = cfg.params.nu_init

    print("=" * 60)
    print("FEM Parameter Identification - Two Sensor Setup")
    print("=" * 60)
    print(f"Source (GT):  E = {E_gt:.1f}, nu = {nu_gt:.3f}")
    print(f"Target (init): E = {E_init:.1f}, nu = {nu_init:.3f}")
    if cfg.debug.visualize:
        print("Visualization: ENABLED (three GUI windows will open)")
    print("=" * 60)

    estimator = SensorParamEstimation(
        dt=cfg.sim.dt,
        total_steps=cfg.sim.total_steps,
        sub_steps=cfg.sim.sub_steps,
        obj_name="block-10.stl",
        visualize=cfg.debug.visualize,
        grad_check=cfg.debug.grad_check,
    )

    # Generate observations with source sensor
    estimator.generate_observations(E_gt, nu_gt)

    # Initialize target sensor with initial params
    estimator.set_target_params(E_init, nu_init)
    print(
        f"\nTarget sensor initialized: E={estimator.E_target[None]:.1f}, nu={estimator.nu_target[None]:.4f}"
    )

    # Learning rates (from config)
    lr_E = cfg.optim.lr_E
    lr_nu = cfg.optim.lr_nu

    # History
    history = {"loss": [], "E": [], "nu": [], "grad_E": [], "grad_nu": []}

    autodiff_method = "tape" if cfg.debug.use_tape else "manual"
    print(f"\nStarting optimization (lr_E={lr_E}, lr_nu={lr_nu}, autodiff={autodiff_method})")
    print("-" * 60)

    for opt_iter in range(cfg.optim.num_iters):
        if cfg.debug.use_tape:
            result = estimator.train_step_tape()
        else:
            result = estimator.train_step(debug=cfg.debug.enabled)

        # Store history
        history["loss"].append(result["loss"])
        history["E"].append(result["E"])
        history["nu"].append(result["nu"])
        history["grad_E"].append(result["grad_E"])
        history["grad_nu"].append(result["grad_nu"])

        if opt_iter % cfg.output.print_every == 0:
            print(
                f"Iter {opt_iter:3d} | Loss: {result['loss']:.6f} | E: {result['E']:.1f} | nu: {result['nu']:.4f}"
            )
            print(
                f"         | grad_E: {result['grad_E']:.6e} | grad_nu: {result['grad_nu']:.6e}"
            )
            print(
                f"         | grad_mu: {result['grad_mu']:.6e} | grad_lam: {result['grad_lam']:.6e}"
            )

        # Check for NaN before gradient descent
        grad_E_nan = np.isnan(result["grad_E"])
        grad_nu_nan = np.isnan(result["grad_nu"])

        if grad_E_nan or grad_nu_nan:
            logger.error(
                f"Iter {opt_iter}: Skipping gradient update due to NaN gradients"
            )
            logger.error(f"  grad_E={result['grad_E']}, grad_nu={result['grad_nu']}")
            logger.error(f"  Current params: E={result['E']}, nu={result['nu']}")
            logger.error(f"  Loss={result['loss']}")
            continue

        # Gradient descent
        new_E = estimator.E_target[None] + lr_E * result["grad_E"]
        new_nu = estimator.nu_target[None] + lr_nu * result["grad_nu"]

        # Clamp to valid ranges
        new_E = max(new_E, 100.0)
        new_nu = max(min(new_nu, 0.49), 0.01)

        logger.debug(
            f"Gradient descent: E {result['E']:.1f} -> {new_E:.1f}, nu {result['nu']:.4f} -> {new_nu:.4f}"
        )

        estimator.set_target_params(new_E, new_nu)

    # Final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Ground truth:  E = {E_gt:.1f}, nu = {nu_gt:.4f}")
    print(
        f"Learned:       E = {estimator.E_target[None]:.1f}, nu = {estimator.nu_target[None]:.4f}"
    )
    print(
        f"Error:         E = {abs(estimator.E_target[None] - E_gt) / E_gt * 100:.1f}%"
    )
    print(
        f"               nu = {abs(estimator.nu_target[None] - nu_gt) / nu_gt * 100:.1f}%"
    )
    print("=" * 60)

    # Plot (save to Hydra output directory)
    if cfg.output.plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        axes[0, 0].semilogy(history["loss"])
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Loss")
        axes[0, 0].grid(True)

        axes[0, 1].plot(history["E"], label="Learned")
        axes[0, 1].axhline(y=E_gt, color="r", linestyle="--", label="GT")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("E")
        axes[0, 1].set_title("Young's Modulus")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[0, 2].plot(history["nu"], label="Learned")
        axes[0, 2].axhline(y=nu_gt, color="r", linestyle="--", label="GT")
        axes[0, 2].set_xlabel("Iteration")
        axes[0, 2].set_ylabel("nu")
        axes[0, 2].set_title("Poisson's Ratio")
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        axes[1, 0].plot(history["grad_E"])
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("grad_E")
        axes[1, 0].set_title("Gradient w.r.t. E")
        axes[1, 0].grid(True)

        axes[1, 1].plot(history["grad_nu"])
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("grad_nu")
        axes[1, 1].set_title("Gradient w.r.t. nu")
        axes[1, 1].grid(True)

        E_err = [abs(e - E_gt) / E_gt for e in history["E"]]
        nu_err = [abs(n - nu_gt) / nu_gt for n in history["nu"]]
        axes[1, 2].semilogy(E_err, label="E error")
        axes[1, 2].semilogy(nu_err, label="nu error")
        axes[1, 2].set_xlabel("Iteration")
        axes[1, 2].set_ylabel("Relative Error")
        axes[1, 2].set_title("Parameter Error")
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        plt.tight_layout()
        # Save to Hydra output directory
        output_dir = HydraConfig.get().runtime.output_dir
        plot_path = os.path.join(output_dir, "convergence.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")

        if not cfg.output.no_show:
            plt.show()


if __name__ == "__main__":
    run_experiment()
