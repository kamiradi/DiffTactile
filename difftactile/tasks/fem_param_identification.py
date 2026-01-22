"""
Sanity check: Learn FEM material parameters (E, nu) from tactile marker displacements.

Experiment:
1. Run simulation with ground-truth E, nu → record marker displacements
2. Re-initialize with wrong E, nu (contact params frozen)
3. Optimize E, nu to match observed marker displacements
4. Check if we recover ground-truth parameters

This validates gradients flow correctly through FEM → markers.
"""
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from difftactile.sensor_model.fem_sensor import FEMDomeSensor
from difftactile.object_model.rigid_dynamic import RigidObj

TI_TYPE = ti.f32
NP_TYPE = np.float32

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@ti.data_oriented
class FEMParamLearner:
    def __init__(self, dt=5e-5, total_steps=50, sub_steps=50, obj=None):
        self.dt = dt
        self.total_steps = total_steps
        self.sub_steps = sub_steps
        self.dim = 3

        # Sensor and object - pass the init image path
        init_img_path = os.path.join(SCRIPT_DIR, "init.png")
        self.fem_sensor = FEMDomeSensor(dt, sub_steps, init_img_path=init_img_path)
        self.space_scale = 10.0
        self.obj_scale = 4.0

        self.mpm_object = RigidObj(
            dt=dt,
            sub_steps=sub_steps,
            obj_name=obj,
            space_scale=self.space_scale,
            obj_scale=self.obj_scale,
            density=1.50,
            rho=0.3
        )

        # Contact parameters - FROZEN (not learned)
        self.kn = 55.0
        self.kd = 269.44
        self.kt = 108.72
        self.friction_coeff = 14.16
        self.norm_eps = 1e-11

        # Sensor trajectory (fixed pressing motion)
        self.p_sensor = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.total_steps))
        self.o_sensor = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.total_steps))

        # Observations: marker displacements at each timestep
        self.num_markers = self.fem_sensor.num_markers
        self.observed_displacement = ti.Vector.field(2, float, shape=(self.total_steps, self.num_markers))
        self.predicted_displacement = ti.Vector.field(2, float, shape=(self.total_steps, self.num_markers), needs_grad=True)

        # Loss
        self.loss = ti.field(float, (), needs_grad=True)

        # Contact detection
        self.contact_idx = ti.Vector.field(
            1, dtype=int,
            shape=(self.sub_steps, self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid)
        )

        self.init_scene()
        self.init_trajectory()

    def init_scene(self):
        """Initialize object and sensor positions - matches object_repose.py."""
        # Object position (same as object_repose.py)
        ball_pos = [3.2, 1.0, 5.0]
        ball_ori = [0.0, 0.0, 90.0]
        ball_vel = [0.0, 0.0, 0.0]
        self.mpm_object.init(ball_pos, ball_ori, ball_vel)

        # Sensor position (same as object_repose.py)
        rx, ry, rz = 0.0, 0.0, 90.0
        tx, ty, tz = 7.0, 1.5, 5.0
        self.fem_sensor.init(rx, ry, rz, tx, ty, tz)

    @ti.kernel
    def init_trajectory(self):
        """Pressing trajectory - velocity commands matching object_repose.py."""
        # These are VELOCITIES, not positions!
        # First phase: press into object (positive y velocity)
        for i in range(self.total_steps):
            # Pressing motion: velocity in y direction
            self.p_sensor[i] = ti.Vector([0.0, 1.5, 0.0])  # Velocity
            self.o_sensor[i] = ti.Vector([0.0, 0.0, 0.0])  # No rotation

    def set_fem_params(self, E, nu):
        """Set FEM material parameters."""
        self.fem_sensor.E_init[None] = E
        self.fem_sensor.nu_init[None] = nu
        # Update Lame parameters
        self.fem_sensor.mu[None] = E / (2.0 * (1.0 + nu))
        self.fem_sensor.lam[None] = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @ti.kernel
    def set_pos_control(self, f: ti.i32):
        self.fem_sensor.d_pos[None] = self.p_sensor[f]
        self.fem_sensor.d_ori[None] = self.o_sensor[f]

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
            shear_factor = (shear_vel / shear_vel_norm) * ti.min(self.kt * shear_vel_norm, max_shear)

        return normal_factor + shear_factor

    @ti.kernel
    def check_collision(self, f: ti.i32):
        for i, j, k in ti.ndrange(self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid):
            if self.mpm_object.grid_occupy[f, i, j, k] == 1:
                cur_p = ti.Vector([
                    (i + 0.5) * self.mpm_object.dx_0,
                    (j + 0.5) * self.mpm_object.dx_0,
                    (k + 0.5) * self.mpm_object.dx_0
                ])
                min_idx = self.fem_sensor.find_closest(cur_p, f)
                self.contact_idx[f, i, j, k][0] = min_idx

    @ti.kernel
    def collision(self, f: ti.i32):
        for i, j, k in ti.ndrange(self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid):
            if self.mpm_object.grid_occupy[f, i, j, k] == 1:
                cur_p = ti.Vector([
                    (i + 0.5) * self.mpm_object.dx_0,
                    (j + 0.5) * self.mpm_object.dx_0,
                    (k + 0.5) * self.mpm_object.dx_0
                ])
                cur_v = self.mpm_object.grid_v_in[f, i, j, k] / (self.mpm_object.grid_m[f, i, j, k] + self.mpm_object.eps)
                min_idx = self.contact_idx[f, i, j, k][0]
                sdf, norm_v, relative_v, contact_flag = self.fem_sensor.find_sdf(cur_p, cur_v, min_idx, f)

                if contact_flag:
                    ext_force = self.calculate_contact_force(sdf, -norm_v, -relative_v)
                    self.mpm_object.update_contact_force(ext_force, f, i, j, k)
                    self.fem_sensor.update_contact_force(min_idx, -ext_force, f)

    def update(self, f):
        """One simulation substep."""
        self.mpm_object.compute_new_F(f)
        self.mpm_object.svd(f)
        self.mpm_object.p2g(f)
        self.fem_sensor.update(f)  # FEM stress uses mu, lam (derived from E, nu)
        self.mpm_object.check_grid_occupy(f)
        self.check_collision(f)
        self.collision(f)
        self.mpm_object.grid_op(f)
        self.mpm_object.g2p(f)
        self.mpm_object.compute_COM(f)
        self.mpm_object.compute_H(f)
        self.mpm_object.compute_H_svd(f)
        self.mpm_object.compute_R(f)
        self.fem_sensor.update2(f)

    def update_grad(self, f):
        """Backward through one substep."""
        self.fem_sensor.update2.grad(f)
        self.mpm_object.compute_R.grad(f)
        self.mpm_object.compute_H_svd_grad(f)
        self.mpm_object.compute_H.grad(f)
        self.mpm_object.compute_COM.grad(f)
        self.mpm_object.g2p.grad(f)
        self.mpm_object.grid_op.grad(f)
        self.collision.grad(f)
        self.fem_sensor.update.grad(f)
        self.mpm_object.p2g.grad(f)
        self.mpm_object.svd_grad(f)
        self.mpm_object.compute_new_F.grad(f)

    @ti.kernel
    def extract_displacement(self, t: ti.i32):
        """Compute marker displacement = current - reference."""
        for i in range(self.num_markers):
            self.predicted_displacement[t, i] = self.fem_sensor.predict_markers[i] - self.fem_sensor.virtual_markers[i]

    @ti.kernel
    def compute_loss_kernel(self, t: ti.i32):
        """MSE loss between predicted and observed marker displacements."""
        for i in range(self.num_markers):
            diff = self.predicted_displacement[t, i] - self.observed_displacement[t, i]
            self.loss[None] += diff.dot(diff)

    def reset(self):
        """Reset simulation state."""
        self.fem_sensor.reset_contact()
        self.mpm_object.reset()
        self.contact_idx.fill(-1)

    def clear_grads(self):
        """Clear all gradients."""
        self.loss[None] = 0.0
        self.loss.grad[None] = 1.0
        self.predicted_displacement.grad.fill(0.0)
        self.fem_sensor.clear_loss_grad()
        self.mpm_object.clear_loss_grad()
        self.fem_sensor.clear_step_grad(self.sub_steps)
        self.mpm_object.clear_step_grad(self.sub_steps)

    def memory_to_cache(self, t):
        self.fem_sensor.memory_to_cache(t)
        self.mpm_object.memory_to_cache(t)

    def memory_from_cache(self, t):
        self.fem_sensor.memory_from_cache(t)
        self.mpm_object.memory_from_cache(t)

    def forward(self, record_observations=False):
        """Run full forward simulation."""
        self.init_scene()

        for ts in range(self.total_steps):
            self.set_pos_control(ts)
            self.fem_sensor.set_pose_control()
            self.fem_sensor.set_control_vel(0)
            self.fem_sensor.set_vel(0)
            self.reset()

            # Run substeps
            for ss in range(self.sub_steps - 1):
                self.update(ss)

            # Extract marker displacement - use the final substep frame
            # After substeps, frame 0 has been updated via copy_frame in memory_to_cache
            # But we need to call extract_markers BEFORE memory_to_cache
            self.fem_sensor.extract_markers(self.sub_steps - 2)
            self.extract_displacement(ts)

            if record_observations:
                # Copy to observations
                self.copy_to_observations(ts)

            # Cache for backward
            self.memory_to_cache(ts)

    @ti.kernel
    def copy_to_observations(self, t: ti.i32):
        for i in range(self.num_markers):
            self.observed_displacement[t, i] = self.predicted_displacement[t, i]

    def generate_observations(self, E_gt, nu_gt):
        """Generate synthetic observations with ground-truth parameters."""
        print(f"Generating observations with GT params: E={E_gt:.1f}, nu={nu_gt:.3f}")
        self.set_fem_params(E_gt, nu_gt)
        self.forward(record_observations=True)

        # Report final displacement magnitude
        final_disp = self.predicted_displacement.to_numpy()[self.total_steps - 1]
        mean_disp = np.mean(np.linalg.norm(final_disp, axis=1))
        max_disp = np.max(np.linalg.norm(final_disp, axis=1))
        print(f"  Mean marker displacement at final step: {mean_disp:.4f} pixels")
        print(f"  Max marker displacement at final step: {max_disp:.4f} pixels")

        # Store a copy of observations for verification
        self.obs_copy = self.observed_displacement.to_numpy().copy()
        print(f"  Observations stored. Shape: {self.obs_copy.shape}")

    def compute_total_loss(self):
        """Compute loss over all timesteps."""
        self.loss[None] = 0.0
        for ts in range(self.total_steps):
            self.compute_loss_kernel(ts)
        return self.loss[None]

    def backward_and_get_grads(self):
        """Run forward, compute loss, backward, return gradients."""
        self.clear_grads()

        # Forward pass
        self.init_scene()

        for ts in range(self.total_steps):
            self.set_pos_control(ts)
            self.fem_sensor.set_pose_control()
            self.fem_sensor.set_control_vel(0)
            self.fem_sensor.set_vel(0)
            self.reset()

            for ss in range(self.sub_steps - 1):
                self.update(ss)

            self.fem_sensor.extract_markers(self.sub_steps - 2)
            self.extract_displacement(ts)
            self.memory_to_cache(ts)

        # Compute loss
        self.compute_total_loss()

        # Backward pass
        for ts in range(self.total_steps - 1, -1, -1):
            # Backward through loss for this timestep
            self.compute_loss_kernel.grad(ts)
            self.extract_displacement.grad(ts)
            self.fem_sensor.extract_markers.grad(self.sub_steps - 2)

            # Reload state
            if ts > 0:
                self.memory_from_cache(ts - 1)
            else:
                self.init_scene()

            self.set_pos_control(ts)
            self.fem_sensor.set_pose_control()
            self.fem_sensor.set_control_vel(0)
            self.fem_sensor.set_vel(0)
            self.reset()

            # Re-run forward for this step (needed for grad computation)
            for ss in range(self.sub_steps - 1):
                self.update(ss)

            # Backward through physics
            for ss in range(self.sub_steps - 2, -1, -1):
                self.update_grad(ss)

            self.fem_sensor.set_vel.grad(0)
            self.fem_sensor.set_control_vel.grad(0)
            self.fem_sensor.set_pose_control.grad()

        # Return gradients for E and nu
        # Note: Gradients accumulate in mu and lam, need to chain rule to E, nu
        grad_mu = self.fem_sensor.mu.grad[None]
        grad_lam = self.fem_sensor.lam.grad[None]

        E = self.fem_sensor.E_init[None]
        nu = self.fem_sensor.nu_init[None]

        # Debug: print intermediate gradients
        print(f"    DEBUG: grad_mu={grad_mu:.6e}, grad_lam={grad_lam:.6e}")
        print(f"    DEBUG: E={E:.1f}, nu={nu:.4f}")

        # Check for NaN in raw gradients
        if np.isnan(grad_mu) or np.isnan(grad_lam):
            print("    WARNING: NaN in mu/lam gradients!")
            return {'E': 0.0, 'nu': 0.0, 'mu': grad_mu, 'lam': grad_lam}

        # Chain rule: d_loss/d_E = d_loss/d_mu * d_mu/d_E + d_loss/d_lam * d_lam/d_E
        # mu = E / (2*(1+nu))  =>  d_mu/d_E = 1 / (2*(1+nu))
        # lam = E*nu / ((1+nu)*(1-2*nu))  =>  d_lam/d_E = nu / ((1+nu)*(1-2*nu))
        dmu_dE = 1.0 / (2.0 * (1.0 + nu))
        dlam_dE = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        grad_E = grad_mu * dmu_dE + grad_lam * dlam_dE

        # d_mu/d_nu = -E / (2*(1+nu)^2)
        dmu_dnu = -E / (2.0 * (1.0 + nu) ** 2)
        # For lam: lam = E*nu / ((1+nu)*(1-2*nu))
        # d_lam/d_nu using quotient rule
        numer = E * nu
        denom = (1.0 + nu) * (1.0 - 2.0 * nu)
        d_numer = E
        d_denom = (1.0 - 2.0 * nu) + (1.0 + nu) * (-2.0)  # = 1 - 2*nu - 2 - 2*nu = -1 - 4*nu
        dlam_dnu = (d_numer * denom - numer * d_denom) / (denom ** 2)
        grad_nu = grad_mu * dmu_dnu + grad_lam * dlam_dnu

        print(f"    DEBUG: grad_E={grad_E:.6e}, grad_nu={grad_nu:.6e}")

        return {
            'E': grad_E,
            'nu': grad_nu,
            'mu': grad_mu,
            'lam': grad_lam
        }


def run_experiment(args):
    ti.init(arch=ti.gpu, device_memory_GB=4)

    # Ground truth FEM parameters
    E_gt = 0.8e4   # 8000 Pa
    nu_gt = 0.43

    # Initial (wrong) parameters
    E_init = 0.4e4   # 4000 Pa (50% of GT)
    nu_init = 0.35   # (lower than GT)

    print("=" * 60)
    print("FEM Parameter Identification Sanity Check")
    print("=" * 60)
    print(f"Ground truth:  E = {E_gt:.1f}, nu = {nu_gt:.3f}")
    print(f"Initial guess: E = {E_init:.1f}, nu = {nu_init:.3f}")
    print("=" * 60)

    learner = FEMParamLearner(
        dt=5e-5,
        total_steps=args.total_steps,
        sub_steps=50,
        obj="block-10.stl"
    )

    # Step 1: Generate observations with GT params
    learner.generate_observations(E_gt, nu_gt)

    # Step 2: Initialize with wrong params
    learner.set_fem_params(E_init, nu_init)

    # Verify params were actually changed
    print(f"\nAfter resetting to wrong params:")
    print(f"  E = {learner.fem_sensor.E_init[None]:.1f} (should be {E_init})")
    print(f"  nu = {learner.fem_sensor.nu_init[None]:.4f} (should be {nu_init})")
    print(f"  mu = {learner.fem_sensor.mu[None]:.4f}")
    print(f"  lam = {learner.fem_sensor.lam[None]:.4f}")

    # Learning rates
    lr_E = args.lr_E
    lr_nu = args.lr_nu

    # History for plotting
    history = {
        'loss': [],
        'E': [],
        'nu': [],
        'grad_E': [],
        'grad_nu': []
    }

    print(f"\nStarting optimization with lr_E={lr_E}, lr_nu={lr_nu}")
    print("-" * 60)

    for opt_iter in range(args.num_iters):
        # Get current params
        E_cur = learner.fem_sensor.E_init[None]
        nu_cur = learner.fem_sensor.nu_init[None]

        # Debug: Check if observations changed
        if opt_iter == 0:
            obs_now = learner.observed_displacement.to_numpy()
            obs_diff = np.abs(obs_now - learner.obs_copy).max()
            print(f"  DEBUG: Max diff in observations since storage: {obs_diff:.6e}")

            # Check predictions vs observations at a specific timestep
            ts_check = min(30, learner.total_steps - 1)
            pred = learner.predicted_displacement.to_numpy()[ts_check]
            obs = learner.observed_displacement.to_numpy()[ts_check]
            diff = np.linalg.norm(pred - obs, axis=1)
            print(f"  DEBUG: At timestep {ts_check}:")
            print(f"    Pred mean norm: {np.mean(np.linalg.norm(pred, axis=1)):.6f}")
            print(f"    Obs mean norm: {np.mean(np.linalg.norm(obs, axis=1)):.6f}")
            print(f"    Diff mean: {np.mean(diff):.6f}, max: {np.max(diff):.6f}")

        # Compute gradients
        grads = learner.backward_and_get_grads()
        loss = learner.loss[None]

        # Store history
        history['loss'].append(loss)
        history['E'].append(E_cur)
        history['nu'].append(nu_cur)
        history['grad_E'].append(grads['E'])
        history['grad_nu'].append(grads['nu'])

        # Print progress
        if opt_iter % args.print_every == 0:
            print(f"Iter {opt_iter:3d} | Loss: {loss:.6f} | E: {E_cur:.1f} | nu: {nu_cur:.4f}")
            print(f"         | grad_E: {grads['E']:.6f} | grad_nu: {grads['nu']:.6f}")

        # Gradient descent update
        E_new = E_cur - lr_E * grads['E']
        nu_new = nu_cur - lr_nu * grads['nu']

        # Clamp to valid ranges
        E_new = max(E_new, 100.0)  # E must be positive
        nu_new = max(min(nu_new, 0.49), 0.01)  # nu in (0, 0.5) for stability

        learner.set_fem_params(E_new, nu_new)

    # Final results
    E_final = learner.fem_sensor.E_init[None]
    nu_final = learner.fem_sensor.nu_init[None]

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Ground truth:  E = {E_gt:.1f}, nu = {nu_gt:.4f}")
    print(f"Learned:       E = {E_final:.1f}, nu = {nu_final:.4f}")
    print(f"Error:         E = {abs(E_final - E_gt) / E_gt * 100:.1f}%, nu = {abs(nu_final - nu_gt) / nu_gt * 100:.1f}%")
    print(f"Final loss:    {history['loss'][-1]:.6f}")
    print("=" * 60)

    # Plot results
    if args.plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Loss
        axes[0, 0].semilogy(history['loss'])
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Convergence')
        axes[0, 0].grid(True)

        # E
        axes[0, 1].plot(history['E'], label='Learned')
        axes[0, 1].axhline(y=E_gt, color='r', linestyle='--', label='GT')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('E (Pa)')
        axes[0, 1].set_title("Young's Modulus (E)")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # nu
        axes[0, 2].plot(history['nu'], label='Learned')
        axes[0, 2].axhline(y=nu_gt, color='r', linestyle='--', label='GT')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('nu')
        axes[0, 2].set_title("Poisson's Ratio (nu)")
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Gradient E
        axes[1, 0].plot(history['grad_E'])
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('grad_E')
        axes[1, 0].set_title('Gradient w.r.t. E')
        axes[1, 0].grid(True)

        # Gradient nu
        axes[1, 1].plot(history['grad_nu'])
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('grad_nu')
        axes[1, 1].set_title('Gradient w.r.t. nu')
        axes[1, 1].grid(True)

        # Parameter error
        E_error = [abs(e - E_gt) / E_gt for e in history['E']]
        nu_error = [abs(n - nu_gt) / nu_gt for n in history['nu']]
        axes[1, 2].semilogy(E_error, label='E error')
        axes[1, 2].semilogy(nu_error, label='nu error')
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Relative Error')
        axes[1, 2].set_title('Parameter Recovery Error')
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        plt.tight_layout()

        save_dir = "fem_param_identification_results"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "convergence.png"), dpi=150)
        print(f"\nPlot saved to {save_dir}/convergence.png")

        if not args.no_show:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FEM parameter identification from tactile markers")
    parser.add_argument("--total_steps", type=int, default=30, help="Simulation timesteps")
    parser.add_argument("--num_iters", type=int, default=50, help="Optimization iterations")
    parser.add_argument("--lr_E", type=float, default=1e2, help="Learning rate for E")
    parser.add_argument("--lr_nu", type=float, default=1e-4, help="Learning rate for nu")
    parser.add_argument("--print_every", type=int, default=5, help="Print frequency")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--no_show", action="store_true", help="Don't display plot (just save)")

    args = parser.parse_args()

    # Change to the tasks directory so relative paths work
    os.chdir(SCRIPT_DIR)

    run_experiment(args)
