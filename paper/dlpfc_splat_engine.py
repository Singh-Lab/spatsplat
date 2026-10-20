"""Scoped copy of the established MERFISH 2D fitting/registration routines.

No notebook is imported or executed. Each engine has private configuration,
device and AE state. DLPFC orchestration lives in dlpfc_registration.py.
"""
import random
import numpy as np
import torch
from torch import nn
from scipy.ndimage import distance_transform_edt
from spatsplat import (AmplitudeSplatField, SplatConfig, capture_initial_splat_state,
    composition_cosine_loss, create_amplitude_splat_optimizer,
    farthest_point_sample_indices, log_total_count_loss, mae_count_loss,
    sample_tissue_support)
from .registration_baselines import ObjectiveCheckpoint, OrientationGuard

ENGINE_DEFAULTS = {'registration_query_count': 4096, 'registration_epochs': 500, 'registration_lr': 0.01, 'registration_log_every': 50, 'registration_rigid_reg_weight': 0.01, 'nonaffine_grid_size': 8, 'nonaffine_epochs': 400, 'nonaffine_lr': 0.005, 'nonaffine_smoothness_weight': 0.01, 'nonaffine_splat_match_weight': 1.0, 'nonaffine_min_valid_fraction': 0.25, 'init_log_amplitude': 6.0, 'section_splat_epochs': 150, 'section_splat_batch_size': 2048, 'section_splat_means_lr': 0.005, 'section_splat_log_scales_lr': 0.001, 'section_splat_rotation_lr': 0.001, 'section_splat_opacity_lr': 0.001, 'section_splat_latents_lr': 0.001, 'section_splat_decoder_lr': 0.001, 'section_splat_amplitude_lr': 0.001, 'section_composition_loss_weight': 1.0, 'section_total_count_loss_weight': 0.2, 'section_mae_refine_weight': 1.0, 'section_mae_refine_fraction': 0.25, 'seed_neighbor_k': 16, 'section_mask_grid_size': 128, 'section_cell_mask_min_radius': 1.5, 'section_off_tissue_batch_size': 2048, 'section_off_tissue_weight': 0.01, 'section_center_support_weight': 0.001, 'display_sig_digits': 4, 'nonaffine_inverse_steps': 80, 'nonaffine_inverse_tolerance': 1e-06, 'nonaffine_contraction_limit': 0.8, 'nonaffine_displacement_weight': 1.0, 'nonaffine_support_weight': 10.0, 'nonaffine_support_margin': 0.02}

def build_engine(options, shared_profile_autoencoder, device):
    unknown = set(options) - set(ENGINE_DEFAULTS) - {"splat_config"}
    if unknown:
        raise ValueError(f"Unknown engine settings: {unknown}")
    settings = {**ENGINE_DEFAULTS, **options}
    splat_cfg = SplatConfig(**settings["splat_config"])
    registration_query_count = settings['registration_query_count']
    registration_epochs = settings['registration_epochs']
    registration_lr = settings['registration_lr']
    registration_log_every = settings['registration_log_every']
    registration_rigid_reg_weight = settings['registration_rigid_reg_weight']
    nonaffine_grid_size = settings['nonaffine_grid_size']
    nonaffine_epochs = settings['nonaffine_epochs']
    nonaffine_lr = settings['nonaffine_lr']
    nonaffine_smoothness_weight = settings['nonaffine_smoothness_weight']
    nonaffine_splat_match_weight = settings['nonaffine_splat_match_weight']
    nonaffine_min_valid_fraction = settings['nonaffine_min_valid_fraction']
    init_log_amplitude = settings['init_log_amplitude']
    section_splat_epochs = settings['section_splat_epochs']
    section_splat_batch_size = settings['section_splat_batch_size']
    section_splat_means_lr = settings['section_splat_means_lr']
    section_splat_log_scales_lr = settings['section_splat_log_scales_lr']
    section_splat_rotation_lr = settings['section_splat_rotation_lr']
    section_splat_opacity_lr = settings['section_splat_opacity_lr']
    section_splat_latents_lr = settings['section_splat_latents_lr']
    section_splat_decoder_lr = settings['section_splat_decoder_lr']
    section_splat_amplitude_lr = settings['section_splat_amplitude_lr']
    section_composition_loss_weight = settings['section_composition_loss_weight']
    section_total_count_loss_weight = settings['section_total_count_loss_weight']
    section_mae_refine_weight = settings['section_mae_refine_weight']
    section_mae_refine_fraction = settings['section_mae_refine_fraction']
    seed_neighbor_k = settings['seed_neighbor_k']
    section_mask_grid_size = settings['section_mask_grid_size']
    section_cell_mask_min_radius = settings['section_cell_mask_min_radius']
    section_off_tissue_batch_size = settings['section_off_tissue_batch_size']
    section_off_tissue_weight = settings['section_off_tissue_weight']
    section_center_support_weight = settings['section_center_support_weight']
    display_sig_digits = settings['display_sig_digits']
    nonaffine_inverse_steps = settings['nonaffine_inverse_steps']
    nonaffine_inverse_tolerance = settings['nonaffine_inverse_tolerance']
    nonaffine_contraction_limit = settings['nonaffine_contraction_limit']
    nonaffine_displacement_weight = settings['nonaffine_displacement_weight']
    nonaffine_support_weight = settings['nonaffine_support_weight']
    nonaffine_support_margin = settings['nonaffine_support_margin']

    def format_scalar(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            value = float(value)
            if np.isnan(value) or np.isinf(value):
                return value
            return float(f'{value:.{display_sig_digits}g}')
        return value

    def format_for_display(obj):
        if isinstance(obj, dict):
            return {k: format_for_display(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [format_for_display(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(format_for_display(v) for v in obj)
        return format_scalar(obj)

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def build_section_tissue_support(xy_norm: np.ndarray, grid_size: int = section_mask_grid_size, min_radius: float = section_cell_mask_min_radius):
        mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        x_idx = np.clip(np.rint(xy_norm[:, 0] * (grid_size - 1)).astype(int), 0, grid_size - 1)
        y_idx = np.clip(np.rint(xy_norm[:, 1] * (grid_size - 1)).astype(int), 0, grid_size - 1)
        mask[y_idx, x_idx] = 1
        dist = distance_transform_edt(1 - mask)
        support = (dist <= float(min_radius)).astype(np.float32)
        support_tensor = torch.tensor(support, dtype=torch.float32).view(1, 1, grid_size, grid_size)

        grid_lin = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
        xx, yy = np.meshgrid(grid_lin, grid_lin, indexing='xy')
        off_mask = support < 0.5
        off_points = np.stack([xx[off_mask], yy[off_mask]], axis=1) if np.any(off_mask) else np.empty((0, 2), dtype=np.float32)
        return support_tensor, torch.tensor(off_points, dtype=torch.float32)

    def init_section_splat_model(section_xy: torch.Tensor, section_profiles: torch.Tensor, section_total_counts: torch.Tensor, train_idx: torch.Tensor):
        with torch.no_grad():
            seed_idx = farthest_point_sample_indices(section_xy.to(device), splat_cfg.num_splats)
            seed_xy = section_xy.to(device)[seed_idx]
            train_idx = train_idx.to(device)
            pairwise_dist = torch.cdist(seed_xy, section_xy.to(device)[train_idx])
            neighbor_k = min(seed_neighbor_k, len(train_idx))
            _, neighbor_idx = torch.topk(pairwise_dist, k=neighbor_k, largest=False, dim=1)

            neighbor_profiles = section_profiles.to(device)[train_idx][neighbor_idx]
            neighbor_totals = section_total_counts.to(device)[train_idx][neighbor_idx]
            avg_profiles = neighbor_profiles.mean(dim=1)
            avg_profiles = avg_profiles / avg_profiles.sum(dim=1, keepdim=True).clamp_min(1e-8)
            avg_totals = neighbor_totals.mean(dim=1)
            encoded_seed_latents = shared_profile_autoencoder.encoder(avg_profiles)

        model = AmplitudeSplatField(
            num_splats=splat_cfg.num_splats,
            latent_dim=splat_cfg.latent_dim,
            hidden_dim=splat_cfg.hidden_dim,
            num_genes=section_profiles.shape[1],
            init_log_scale=splat_cfg.init_log_scale,
            kernel_sharpness=splat_cfg.kernel_sharpness,
            init_log_amplitude=init_log_amplitude,
        ).to(device)

        with torch.no_grad():
            model.means.copy_(seed_xy)
            model.latents.copy_(encoded_seed_latents)
            model.log_amplitudes.copy_(avg_totals.clamp_min(1.0).log())
            model.log_scales.fill_(splat_cfg.init_log_scale)
            model.rotation.zero_()
            model.opacity_logits.zero_()
            model.decoder.load_state_dict(shared_profile_autoencoder.decoder.state_dict())

        return model

    def clone_splat_model(model):
        cloned = AmplitudeSplatField(
            num_splats=model.num_splats,
            latent_dim=model.latent_dim,
            hidden_dim=model.hidden_dim,
            num_genes=model.num_genes,
            init_log_scale=model.init_log_scale,
            kernel_sharpness=model.kernel_sharpness,
            init_log_amplitude=float(model.log_amplitudes.detach().mean().item()),
        ).to(model.means.device)
        cloned.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
        cloned.eval()
        return cloned

    def evaluate_section_splat_model(model, section_xy: torch.Tensor, section_counts: torch.Tensor, split_idx: torch.Tensor, batch_size: int = 2048):
        was_training = model.training
        model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_seen = 0
        with torch.no_grad():
            for start in range(0, len(split_idx), batch_size):
                idx = split_idx[start:start + batch_size].to(device)
                pred_log_rate = model(section_xy.to(device)[idx])
                loss = mae_count_loss(pred_log_rate, section_counts.to(device)[idx])
                pred_counts = torch.exp(pred_log_rate)
                mae = (pred_counts - section_counts.to(device)[idx]).abs().mean()
                total_loss += loss.item() * len(idx)
                total_mae += mae.item() * len(idx)
                total_seen += len(idx)
        if was_training:
            model.train()
        return {'loss': total_loss / total_seen, 'mae': total_mae / total_seen, 'n': total_seen}

    def train_section_splat_model(section_xy: torch.Tensor, section_counts: torch.Tensor, split_idx: dict[str, torch.Tensor], initial_model, *, label: str, seed: int, off_tissue_points: torch.Tensor, tissue_mask_tensor: torch.Tensor, min_steps: int = 6000):
        set_seed(seed)
        model = clone_splat_model(initial_model)
        optimizer = create_amplitude_splat_optimizer(
            model,
            means_lr=section_splat_means_lr,
            log_scales_lr=section_splat_log_scales_lr,
            rotation_lr=section_splat_rotation_lr,
            opacity_lr=section_splat_opacity_lr,
            latents_lr=section_splat_latents_lr,
            decoder_lr=section_splat_decoder_lr,
            amplitude_lr=section_splat_amplitude_lr,
        )
        history = {'train_joint_loss': [], 'train_composition_loss': [], 'train_total_count_loss': [], 'train_cell_loss': [], 'train_off_tissue_loss': [], 'train_center_support_loss': [], 'val_loss': [], 'val_mae': [], 'val_phase': []}
        train_idx = split_idx['train'].to(device)
        val_idx = split_idx['val']
        xy = section_xy.to(device)
        counts = section_counts.to(device)
        if min(len(train_idx), section_splat_batch_size, section_splat_epochs) < 1 or min_steps < 0:
            raise ValueError('Invalid splat training size, batch size, epoch count, or minimum steps')
        if not 0.0 <= section_mae_refine_fraction <= 1.0:
            raise ValueError('MAE refinement fraction must be between zero and one')
        updates_per_epoch = (len(train_idx) + section_splat_batch_size - 1) // section_splat_batch_size
        effective_epochs = max(section_splat_epochs, (min_steps + updates_per_epoch - 1) // updates_per_epoch)
        optimizer_steps = 0
        # Keep the existing curriculum proportions over the expanded training budget.
        mae_refine_start_epoch = max(0, int(round((1.0 - section_mae_refine_fraction) * effective_epochs)))
        print(f'{label} | training budget:', {
            'training_measurements': len(train_idx), 'updates_per_epoch': updates_per_epoch,
            'epochs': effective_epochs, 'planned_optimizer_steps': effective_epochs * updates_per_epoch,
            'mae_refine_start_epoch': mae_refine_start_epoch,
        })

        for epoch in range(effective_epochs):
            model.train()
            mae_refine_active = epoch >= mae_refine_start_epoch
            perm = train_idx[torch.randperm(len(train_idx), device=device)]
            off_perm = torch.randperm(len(off_tissue_points), device=device) if len(off_tissue_points) > 0 else None
            running = {'joint_loss': 0.0, 'cell_loss': 0.0, 'composition_loss': 0.0, 'total_count_loss': 0.0, 'off_tissue_loss': 0.0, 'center_support_loss': 0.0}
            steps = 0
            for step, start in enumerate(range(0, len(perm), section_splat_batch_size)):
                idx = perm[start:start + section_splat_batch_size]
                pred_log_rate = model(xy[idx])
                cell_loss = mae_count_loss(pred_log_rate, counts[idx]) if mae_refine_active else pred_log_rate.new_tensor(0.0)
                composition_loss = composition_cosine_loss(torch.exp(pred_log_rate), counts[idx])
                total_count_loss = log_total_count_loss(pred_log_rate, counts[idx])
                if off_perm is not None and len(off_tissue_points) > 0:
                    off_start = (step * section_off_tissue_batch_size) % len(off_perm)
                    off_idx = off_perm[off_start:off_start + section_off_tissue_batch_size]
                    if len(off_idx) == 0:
                        off_idx = off_perm[:section_off_tissue_batch_size]
                    off_tissue_loss = model.total_intensity(off_tissue_points[off_idx]).mean()
                else:
                    off_tissue_loss = pred_log_rate.new_tensor(0.0)
                center_support_loss = (1.0 - sample_tissue_support(model.means, tissue_mask_tensor)).mean()
                loss = section_mae_refine_weight * cell_loss + section_composition_loss_weight * composition_loss + section_total_count_loss_weight * total_count_loss + section_off_tissue_weight * off_tissue_loss + section_center_support_weight * center_support_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer_steps += 1
                with torch.no_grad():
                    model.means.clamp_(0.0, 1.0)
                    model.log_scales.clamp_(-8.0, -2.0)
                running['joint_loss'] += loss.item()
                running['cell_loss'] += cell_loss.item()
                running['composition_loss'] += composition_loss.item()
                running['total_count_loss'] += total_count_loss.item()
                running['off_tissue_loss'] += off_tissue_loss.item()
                running['center_support_loss'] += center_support_loss.item()
                steps += 1

            train_metrics = {k: v / max(steps, 1) for k, v in running.items()}
            val_metrics = evaluate_section_splat_model(model, section_xy, section_counts, val_idx)
            history['train_joint_loss'].append(train_metrics['joint_loss'])
            history['train_composition_loss'].append(train_metrics['composition_loss'])
            history['train_total_count_loss'].append(train_metrics['total_count_loss'])
            history['train_cell_loss'].append(train_metrics['cell_loss'])
            history['train_off_tissue_loss'].append(train_metrics['off_tissue_loss'])
            history['train_center_support_loss'].append(train_metrics['center_support_loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_mae'].append(val_metrics['mae'])
            history['val_phase'].append('mae_refine' if mae_refine_active else 'composition')

            if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == effective_epochs - 1:
                print(f'{label} | epoch {epoch}:', format_for_display({'optimizer_steps': optimizer_steps, 'train': train_metrics, 'val': val_metrics, 'phase': history['val_phase'][-1]}))

        summary = {
            'train': evaluate_section_splat_model(model, section_xy, section_counts, split_idx['train']),
            'val': evaluate_section_splat_model(model, section_xy, section_counts, split_idx['val']),
            'test': evaluate_section_splat_model(model, section_xy, section_counts, split_idx['test']),
        }
        return {'model': model, 'history': history, 'summary': summary, 'initial_state': capture_initial_splat_state(model),
                'training_budget': {'epochs': effective_epochs, 'optimizer_steps': optimizer_steps,
                                    'updates_per_epoch': updates_per_epoch, 'minimum_steps': min_steps,
                                    'mae_refine_start_epoch': mae_refine_start_epoch,
                                    'composition_steps': mae_refine_start_epoch * updates_per_epoch,
                                    'mae_refine_steps': (effective_epochs - mae_refine_start_epoch) * updates_per_epoch}}

    class Affine2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Parameter(torch.eye(2, dtype=torch.float32))
            self.translation = nn.Parameter(torch.zeros(2, dtype=torch.float32))

        def forward(self, xy: torch.Tensor) -> torch.Tensor:
            return xy @ self.linear.T + self.translation

        def inverse(self, xy: torch.Tensor) -> torch.Tensor:
            linear_inv = torch.linalg.inv(self.linear)
            return (xy - self.translation) @ linear_inv.T

    def sample_registration_queries(xy: torch.Tensor, n: int = registration_query_count) -> torch.Tensor:
        n = min(n, xy.shape[0])
        idx = torch.randperm(xy.shape[0])[:n]
        return xy[idx]

    def in_bounds_mask(xy: torch.Tensor) -> torch.Tensor:
        return ((xy[:, 0] >= 0.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= 0.0) & (xy[:, 1] <= 1.0))

    def registration_loss_from_models(model_a, model_b, query_xy_a: torch.Tensor, transform: Affine2D, *, min_valid_fraction: float = 0.25):
        query_xy_b = transform(query_xy_a)
        valid_mask = in_bounds_mask(query_xy_b)
        valid_fraction = valid_mask.float().mean()

        if valid_mask.sum() == 0:
            invalid_loss = query_xy_b.new_tensor(1e3)
            return invalid_loss, valid_fraction

        rate_a = torch.exp(model_a(query_xy_a[valid_mask]))
        rate_b = torch.exp(model_b(query_xy_b[valid_mask]))
        field_loss = composition_cosine_loss(rate_a, rate_b)

        if valid_fraction < min_valid_fraction:
            field_loss = field_loss + (min_valid_fraction - valid_fraction) * 10.0

        return field_loss, valid_fraction

    def affine_regularization(transform: Affine2D) -> torch.Tensor:
        linear = transform.linear
        translation = transform.translation
        return (linear - torch.eye(2, device=linear.device)).pow(2).mean() + translation.pow(2).mean()

    def run_affine_registration(model_a, model_b, query_xy_a: torch.Tensor, *, epochs: int = registration_epochs, lr: float = registration_lr, rigid_reg_weight: float = registration_rigid_reg_weight, allow_reflection: bool = False):
        transform = Affine2D().to(device)
        opt = torch.optim.Adam(transform.parameters(), lr=lr)
        history = {'loss': [], 'field_loss': [], 'reg_loss': [], 'valid_fraction': []}
        query_xy_a = query_xy_a.to(device)

        checkpoint = ObjectiveCheckpoint()
        for epoch in range(epochs + 1):
            field_loss, valid_fraction = registration_loss_from_models(model_a, model_b, query_xy_a, transform)
            reg_loss = affine_regularization(transform)
            loss = field_loss + rigid_reg_weight * reg_loss

            if allow_reflection or torch.linalg.det(transform.linear).detach().item() > 0:
                checkpoint.consider(transform, loss.detach(), epoch)
            if epoch == epochs:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

            history['loss'].append(float(loss.item()))
            history['field_loss'].append(float(field_loss.item()))
            history['reg_loss'].append(float(reg_loss.item()))
            history['valid_fraction'].append(float(valid_fraction.item()))

            if epoch == 0 or (epoch + 1) % registration_log_every == 0 or epoch == epochs - 1:
                print(f'registration epoch {epoch}:', format_for_display({
                    'loss': loss.item(),
                    'field_loss': field_loss.item(),
                    'reg_loss': reg_loss.item(),
                    'valid_fraction': valid_fraction.item(),
                }))

        checkpoint.restore(transform)
        print('Selected registration step:', checkpoint.step, '| objective:', format_scalar(checkpoint.objective))
        return {'transform': transform, 'history': history, 'selected_step': checkpoint.step, 'selected_objective': checkpoint.objective}

    class NonAffineWarp2D(nn.Module):
        def __init__(self, base_affine: Affine2D, grid_size: int = nonaffine_grid_size):
            super().__init__()
            self.base_affine = Affine2D().to(device)
            with torch.no_grad():
                self.base_affine.linear.copy_(base_affine.linear.detach())
                self.base_affine.translation.copy_(base_affine.translation.detach())
            for param in self.base_affine.parameters():
                param.requires_grad_(False)
            self.grid_size = grid_size
            self.displacement = nn.Parameter(torch.zeros(1, 2, grid_size, grid_size, dtype=torch.float32, device=device))

        def sample_displacement(self, xy: torch.Tensor) -> torch.Tensor:
            grid = xy.view(1, -1, 1, 2) * 2.0 - 1.0
            sampled = torch.nn.functional.grid_sample(
                self.displacement,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True,
            )
            return sampled.view(2, -1).T

        def forward(self, xy: torch.Tensor) -> torch.Tensor:
            affine_xy = self.base_affine(xy)
            return affine_xy + self.sample_displacement(xy)

        @torch.no_grad()
        def project_contraction(self):
            # Bound the Jacobian of A^{-1}d using the bilinear grid edge slopes.
            inv = torch.linalg.inv(self.base_affine.linear)
            residual = torch.einsum('ij,bjhw->bihw', inv, self.displacement)
            dx = torch.linalg.vector_norm(residual[:, :, :, 1:] - residual[:, :, :, :-1], dim=1).max()
            dy = torch.linalg.vector_norm(residual[:, :, 1:, :] - residual[:, :, :-1, :], dim=1).max()
            bound = (self.grid_size - 1) * torch.sqrt(dx.square() + dy.square())
            self.displacement.mul_(torch.clamp(nonaffine_contraction_limit / bound.clamp_min(1e-12), max=1.0))

        def inverse_approx(self, xy: torch.Tensor, steps=None) -> torch.Tensor:
            steps = nonaffine_inverse_steps if steps is None else steps
            estimate = self.base_affine.inverse(xy)
            for _ in range(steps):
                updated = self.base_affine.inverse(xy - self.sample_displacement(estimate))
                delta = torch.linalg.vector_norm(updated - estimate, dim=1).max()
                estimate = updated
                if float(delta.detach()) <= nonaffine_inverse_tolerance:
                    break
            return estimate

        @torch.no_grad()
        def inverse_diagnostics(self, xy):
            short = self.inverse_approx(xy, steps=5)
            mapped = self.inverse_approx(xy)
            error = torch.linalg.vector_norm(self(mapped) - xy, dim=1)
            short_error = torch.linalg.vector_norm(self(short) - xy, dim=1)
            return dict(roundtrip_max=float(error.max()),
                        roundtrip_q99=float(torch.quantile(error, 0.99)),
                        five_step_roundtrip_max=float(short_error.max()),
                        inverse_change_max=float(torch.linalg.vector_norm(mapped - short, dim=1).max()),
                        unconverged_fraction=float((error > 10 * nonaffine_inverse_tolerance).float().mean()))

    def tissue_support_loss(mapped, support):
        # Select neighbors without a large autograd distance matrix; distances retain gradients.
        losses = []
        for chunk in mapped.split(512):
            with torch.no_grad():
                nearest = torch.cdist(chunk.detach(), support).argmin(dim=1)
            distance = torch.linalg.vector_norm(chunk - support[nearest], dim=1)
            losses.append(torch.relu(distance - nonaffine_support_margin).square())
        return torch.cat(losses).mean()

    def warp_smoothness_loss(warp: NonAffineWarp2D) -> torch.Tensor:
        disp = warp.displacement
        dx = disp[:, :, 1:, :] - disp[:, :, :-1, :]
        dy = disp[:, :, :, 1:] - disp[:, :, :, :-1]
        return dx.pow(2).mean() + dy.pow(2).mean()

    def nonaffine_registration_loss(model_a, model_b, query_xy_a: torch.Tensor, warp: NonAffineWarp2D, *, min_valid_fraction: float = nonaffine_min_valid_fraction):
        query_xy_b = warp(query_xy_a)
        valid_mask = in_bounds_mask(query_xy_b)
        valid_fraction = valid_mask.float().mean()
        if valid_mask.sum() == 0:
            return 1e3 + (torch.relu(-query_xy_b).square() + torch.relu(query_xy_b - 1).square()).mean(), valid_fraction
        rate_a = torch.exp(model_a(query_xy_a[valid_mask]))
        rate_b = torch.exp(model_b(query_xy_b[valid_mask]))
        field_loss = composition_cosine_loss(rate_a, rate_b)
        if valid_fraction < min_valid_fraction:
            field_loss = field_loss + (min_valid_fraction - valid_fraction) * 10.0
        bounds_loss = (torch.relu(-query_xy_b).square() + torch.relu(query_xy_b - 1).square()).mean()
        return field_loss + nonaffine_support_weight * bounds_loss, valid_fraction

    def run_nonaffine_registration(model_a, model_b, query_xy_a: torch.Tensor, *, affine_result, anchor_means_b: torch.Tensor, anchor_means_a: torch.Tensor, anchor_weights: torch.Tensor, allow_reflection: bool = False,
                                   epochs: int = nonaffine_epochs, lr: float = nonaffine_lr,
                                   smoothness_weight: float = nonaffine_smoothness_weight,
                                   anchor_weight: float = nonaffine_splat_match_weight, orientation_xy_b=None, support_xy_a=None):
        warp = NonAffineWarp2D(affine_result['transform'], grid_size=nonaffine_grid_size).to(device)
        opt = torch.optim.Adam([warp.displacement], lr=lr)
        history = {'loss': [], 'field_loss': [], 'smoothness_loss': [], 'anchor_loss': [], 'valid_fraction': []}
        query_xy_a = query_xy_a.to(device)
        anchor_means_b = anchor_means_b.to(device)
        anchor_means_a = anchor_means_a.to(device)
        anchor_weights = anchor_weights.to(device)

        checkpoint = ObjectiveCheckpoint()
        orientation_points = query_xy_a if orientation_xy_b is None else orientation_xy_b.to(device)
        orientation_guard = None if allow_reflection else OrientationGuard(orientation_points)
        support_a = query_xy_a.detach() if support_xy_a is None else support_xy_a.to(device).detach()
        support_b = None if orientation_xy_b is None else orientation_xy_b.to(device).detach()
        history.update(displacement_loss=[], support_loss=[])
        print('Initial inverse diagnostics:', format_for_display(warp.inverse_diagnostics(orientation_points)))
        for epoch in range(epochs + 1):
            field_loss, valid_fraction = nonaffine_registration_loss(model_a, model_b, query_xy_a, warp)
            smoothness_loss = warp_smoothness_loss(warp)
            if anchor_means_b.shape[0] > 0:
                mapped_anchor = warp.inverse_approx(anchor_means_b)
                anchor_dist = torch.linalg.vector_norm(mapped_anchor - anchor_means_a, dim=1)
                anchor_loss = (anchor_dist * anchor_weights).mean()
            else:
                anchor_loss = field_loss.new_tensor(0.0)
            displacement_loss = warp.displacement.square().mean()
            support_loss = field_loss.new_tensor(0.0)
            if support_b is not None:
                support_loss = (tissue_support_loss(warp(query_xy_a), support_b)
                                + tissue_support_loss(warp.inverse_approx(support_b), support_a)) / 2
            loss = (field_loss + smoothness_weight * smoothness_loss + anchor_weight * anchor_loss
                    + nonaffine_displacement_weight * displacement_loss
                    + nonaffine_support_weight * support_loss)

            with torch.no_grad():
                orientation_mapped = warp(query_xy_a) if orientation_xy_b is None else warp.inverse_approx(orientation_points)
                orientation_ok = orientation_guard is None or orientation_guard(orientation_mapped)
                roundtrip_ok = True
                if support_b is not None:
                    roundtrip = torch.linalg.vector_norm(warp(orientation_mapped) - support_b, dim=1)
                    roundtrip_ok = bool(torch.isfinite(roundtrip).all() and roundtrip.max() <= 10 * nonaffine_inverse_tolerance)
            if orientation_ok and roundtrip_ok:
                checkpoint.consider(warp, loss.detach(), epoch)
            if epoch == epochs:
                break

            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite non-affine objective')
            opt.zero_grad()
            loss.backward()
            if not torch.isfinite(warp.displacement.grad).all():
                raise RuntimeError('Non-finite non-affine gradient')
            opt.step()
            warp.project_contraction()

            history['support_loss'].append(float(support_loss.detach()))
            history['displacement_loss'].append(float(displacement_loss.detach()))
            history['loss'].append(float(loss.item()))
            history['field_loss'].append(float(field_loss.item()))
            history['smoothness_loss'].append(float(smoothness_loss.item()))
            history['anchor_loss'].append(float(anchor_loss.item()))
            history['valid_fraction'].append(float(valid_fraction.item()))

            if epoch == 0 or (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                print(f'nonaffine epoch {epoch}:', format_for_display({
                    'loss': loss.item(),
                    'field_loss': field_loss.item(),
                    'smoothness_loss': smoothness_loss.item(),
                    'anchor_loss': anchor_loss.item(),
                    'valid_fraction': valid_fraction.item(),
                }))

        checkpoint.restore(warp)
        print('Selected registration step:', checkpoint.step, '| objective:', format_scalar(checkpoint.objective))
        diagnostics = warp.inverse_diagnostics(orientation_points)
        if support_b is not None:
            with torch.no_grad():
                mapped = warp.inverse_approx(support_b)
                distances = torch.cat([torch.cdist(chunk, support_a).min(dim=1).values for chunk in mapped.split(512)])
                diagnostics.update(nearest_support_distance_max=float(distances.max()),
                                   nearest_support_distance_q99=float(torch.quantile(distances, 0.99)),
                                   outside_support_margin_fraction=float((distances > nonaffine_support_margin).float().mean()))
        print('Selected inverse diagnostics:', format_for_display(diagnostics))
        return {'warp': warp, 'history': history, 'inverse_diagnostics': diagnostics, 'selected_step': checkpoint.step, 'selected_objective': checkpoint.objective}

    return {
        'build_section_tissue_support': build_section_tissue_support,
        'init_section_splat_model': init_section_splat_model,
        'train_section_splat_model': train_section_splat_model,
        'sample_registration_queries': sample_registration_queries,
        'run_affine_registration': run_affine_registration,
        'run_nonaffine_registration': run_nonaffine_registration,
    }
