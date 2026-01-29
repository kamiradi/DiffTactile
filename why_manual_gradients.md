# Why Manual Gradients and Checkpointing in DiffTactile

This document explains the design decisions behind the manual backward pass implementation in `fem_param_identification.py`.

## The Memory Problem

In reverse-mode automatic differentiation (backpropagation), computing gradients requires intermediate values from the forward pass. For this differentiable physics simulation:

- `total_steps` outer timesteps (e.g., 200)
- `sub_steps` inner substeps each (e.g., 50)
- Each substep has multiple fields: `pos`, `vel`, `F` (deformation gradient), `stress`, etc.

Storing ALL intermediate values would require enormous memory - potentially gigabytes for a single simulation.

## Why Not Use Taichi's Tape Autodiff?

Taichi provides `ti.ad.Tape()` for automatic differentiation, but it has limitations:

1. **Kernel restrictions**: Tape autodiff cannot handle kernels that mix for-loops with non-looping statements. Many physics kernels (like `set_pose_control`, `compute_lame_params`) violate this.

2. **Memory**: Tape records the entire computation graph, which is memory-intensive for long simulations.

3. **Complex physics**: The FEM sensor and MPM object have many interdependent kernels that were designed for manual autodiff.

## The Checkpointing Solution

Instead of storing everything, we use **gradient checkpointing**:

### Forward Pass
```python
for ts in range(total_steps):
    # Run physics substeps
    for ss in range(sub_steps - 1):
        update_step(sensor, ss)

    # Save checkpoint at outer timestep boundary
    memory_to_cache(sensor, ts)
```

### Backward Pass
```python
for ts in range(total_steps - 1, -1, -1):
    # 1. Backward through loss and marker extraction
    compute_loss_at_timestep.grad(ts)
    extract_displacement_target.grad(ts)
    extract_markers.grad(sub_steps - 2)

    # 2. Restore checkpoint from START of this timestep
    memory_from_cache(sensor, ts - 1)  # or init_scene for ts=0

    # 3. Re-run forward to reconstruct intermediate substep values
    for ss in range(sub_steps - 1):
        update_step(sensor, ss)

    # 4. Backward through physics (now with correct intermediate values)
    for ss in range(sub_steps - 2, -1, -1):
        update_step_grad(sensor, ss)
```

## Why Re-run Forward is Necessary

When calling `sensor.update.grad(f)`, Taichi needs the field values (`pos[f]`, `vel[f]`, etc.) to be exactly as they were during forward. The gradient computation uses these values:

```python
# In update kernel (forward):
F_i = D_i @ self.B[i]  # Deformation gradient depends on pos
IC = (F_i.transpose() @ F_i).trace()
stress = self.mu[None] * (1 - 1/(IC+1)) * F_i + self.lam[None] * (J - alpha) * dJdF

# In update.grad (backward):
# d(loss)/d(mu) requires F_i, IC, J, dJdF, etc.
# These all depend on pos[f] being correct
```

Without the correct `pos[f]` values, the gradient computation gives wrong results.

## Gradient Flow

The expected gradient flow is:

```
loss
  ↓ compute_loss_at_timestep.grad
predicted_displacement.grad
  ↓ extract_displacement_target.grad
predict_markers.grad
  ↓ extract_markers.grad
pos.grad[sub_steps-2]
  ↓ update2.grad (for each substep, backward)
vel.grad[f]
  ↓ update.grad
mu.grad, lam.grad
  ↓ apply_target_params.grad
mu_target.grad, lam_target.grad
  ↓ compute_lame_params.grad
E_target.grad, nu_target.grad
```

## Known Issue: Gradient Loss During Checkpointing

The checkpointing correctly restores **field values**, but **gradients** can be lost:

```
extract_markers.grad  →  sets pos.grad[48]  ✓
memory_from_cache     →  restores values, may clear gradients
re-run forward        →  overwrites fields, may clear gradients
update_step_grad      →  pos.grad[48] is now ZERO  ✗
```

The fix is to save gradient fields before restore/re-run and restore them after:

```python
# Save gradients before checkpoint restore
saved_pos_grad = sensor.pos.grad.to_numpy().copy()

# Restore state and re-run forward
memory_from_cache(sensor, ts - 1)
for ss in range(sub_steps - 1):
    update_step(sensor, ss)

# Restore gradients
sensor.pos.grad.from_numpy(saved_pos_grad)

# Now backward through physics works correctly
for ss in range(sub_steps - 2, -1, -1):
    update_step_grad(sensor, ss)
```

## References

- [Taichi Differentiable Programming Docs](https://docs.taichi-lang.org/docs/differentiable_programming)
- Gradient checkpointing is a standard technique in deep learning (see PyTorch's `torch.utils.checkpoint`)
