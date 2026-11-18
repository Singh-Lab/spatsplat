"""Pilot registration against frozen or jointly fitted serial-section fields.

All coordinates use one shared isotropic scale. Ground-truth coordinates enter
evaluation only; transform fitting uses supplied observations and expression.
"""
import copy
import math

import numpy as np
import torch
from torch import nn

from spatsplat import VolumetricAmplitudeSplatField, farthest_point_sample_indices


def profiles(values):
    return values / values.sum(-1, keepdim=True).clamp_min(1e-8)


def expression_loss(prediction, target):
    cosine = 1 - nn.functional.cosine_similarity(prediction, target).mean()
    total = (prediction.sum(1).clamp_min(1e-8).log() - target.sum(1).clamp_min(1e-8).log()).abs().mean()
    return cosine + 0.1 * total


class SectionRigid(nn.Module):
    """In-plane correction about the common volume center; z stays fixed."""
    def __init__(self, device='cpu'):
        super().__init__()
        self.angle = nn.Parameter(torch.tensor(0., device=device))
        self.translation = nn.Parameter(torch.zeros(2, device=device))

    def forward(self, xyz):
        c, s = self.angle.cos(), self.angle.sin()
        rotation = torch.stack((c, -s, s, c)).reshape(2, 2)
        xy = (xyz[:, :2]-.5) @ rotation.T + .5 + self.translation
        return torch.cat((xy, xyz[:, 2:]), dim=1)


def perturb(xyz, degrees, translation):
    transform = SectionRigid(xyz.device)
    with torch.no_grad():
        transform.angle.fill_(math.radians(degrees))
        transform.translation.copy_(xyz.new_tensor(translation))
        return transform(xyz)


def initialize_field(xyz, values, autoencoder, settings, max_tilt_degrees, seed):
    torch.manual_seed(seed)
    count = settings['num_splats']
    if len(xyz) < count:
        raise ValueError('More splats than training measurements')
    model = VolumetricAmplitudeSplatField(
        count, settings['latent_dim'], settings['hidden_dim'], values.shape[1],
        init_log_scale_xy=math.log(settings['scale_xy']),
        init_log_scale_z=math.log(settings['spacing']),
        kernel_sharpness=settings['sharpness'], top_k_splats=settings['top_k'],
        max_tilt_degrees=max_tilt_degrees).to(xyz.device)
    idx = farthest_point_sample_indices(xyz, count, candidate_limit=min(len(xyz), 30000), random_state=seed)
    centers = xyz[idx].clone()
    centers[:, 2] = torch.linspace(float(xyz[:, 2].min()), float(xyz[:, 2].max()), count, device=xyz.device)
    with torch.no_grad():
        model.means.copy_(centers)
        for start in range(0, count, 128):
            neighbors = torch.cdist(centers[start:start+128], xyz).topk(min(16, len(xyz)), largest=False).indices
            profile = profiles(profiles(values)[neighbors].mean(1))
            model.latents[start:start+128].copy_(autoencoder.encoder(profile))
            model.log_amplitudes[start:start+128].copy_(values[neighbors].sum(-1).mean(1, keepdim=True).clamp_min(1e-8).log())
        model.decoder.load_state_dict(autoencoder.decoder.state_dict())
    model.decoder.requires_grad_(False)
    return model


def project_field(model, settings, lower, upper):
    model.project_geometry_(min_scale_xy=.005, max_scale_xy=.2,
                            min_scale_z=.5*settings['spacing'], max_scale_z=2*settings['spacing'])
    with torch.no_grad():
        model.means.copy_(torch.maximum(lower, torch.minimum(upper, model.means)))


def tilt_penalty(model):
    return model.splat_field.tilt_vectors().square().sum(1).mean()


@torch.no_grad()
def evaluate_field(model, xyz, values, batch=512):
    losses, cosines, maes = [], [], []
    for x, y in zip(xyz.split(batch), values.split(batch)):
        p = model(x).exp()
        losses.append(float(expression_loss(p, y))*len(x))
        cosines.append(float(nn.functional.cosine_similarity(p, y).sum()))
        maes.append(float((p-y).abs().mean(1).sum()))
    return dict(loss=sum(losses)/len(xyz), cosine=sum(cosines)/len(xyz), mae=sum(maes)/len(xyz))


def fit_field(model, sections, validation, settings, seed):
    """Section-balanced batches; checkpoint on flanking-section validation only."""
    torch.manual_seed(seed)
    model.requires_grad_(True)
    model.decoder.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=settings['field_lr'])
    all_xyz = torch.cat([x for x, _ in sections])
    lower, upper = all_xyz.amin(0), all_xyz.amax(0)
    best, state, history = float('inf'), None, []
    for epoch in range(settings['field_epochs']+1):
        if epoch % settings['validation_every'] == 0 or epoch == settings['field_epochs']:
            metrics = evaluate_field(model, *validation)
            history.append(dict(epoch=epoch, **metrics))
            if metrics['loss'] < best:
                best, state = metrics['loss'], copy.deepcopy(model.state_dict())
            print(f"Field {epoch}: validation loss {metrics['loss']:.4g}, cosine {metrics['cosine']:.4g}")
        if epoch == settings['field_epochs']:
            break
        for _ in range(settings['steps_per_epoch']):
            loss = 0
            for xyz, values in sections:
                idx = torch.randint(len(xyz), (min(settings['batch_size'], len(xyz)),), device=xyz.device)
                loss = loss + expression_loss(model(xyz[idx]).exp(), values[idx])/len(sections)
            loss = loss + settings['tilt_weight']*tilt_penalty(model)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            project_field(model, settings, lower, upper)
    model.load_state_dict(state)
    model.requires_grad_(False)
    return history


def support_loss(xyz, support, margin):
    with torch.no_grad():
        nearest = torch.cdist(xyz[:, :2].detach(), support[:, :2]).argmin(1)
    return torch.relu(torch.linalg.vector_norm(xyz[:, :2]-support[nearest, :2], dim=1)-margin).square().mean()


def fit_transform(model, observed, values, fit_idx, val_idx, support, settings, seed):
    """No original coordinates or known inverse are inputs to optimization."""
    torch.manual_seed(seed)
    model.requires_grad_(False)
    transform = SectionRigid(observed.device)
    opt = torch.optim.Adam(transform.parameters(), lr=settings['registration_lr'])
    best, state, history = float('inf'), None, []
    def objective(idx):
        mapped = transform(observed[idx])
        return (expression_loss(model(mapped).exp(), values[idx])
                + settings['support_weight']*support_loss(mapped, support, settings['support_margin'])
                + settings['transform_weight']*(transform.translation.square().sum()+transform.angle.square()))
    for step in range(settings['registration_steps']+1):
        if step % 10 == 0 or step == settings['registration_steps']:
            with torch.no_grad():
                loss = sum(float(objective(chunk))*len(chunk) for chunk in val_idx.split(settings['batch_size']))/len(val_idx)
            history.append(dict(step=step, validation_objective=loss))
            if loss < best:
                best, state = loss, copy.deepcopy(transform.state_dict())
        if step == settings['registration_steps']:
            break
        idx = fit_idx[torch.randint(len(fit_idx), (min(settings['batch_size'], len(fit_idx)),), device=observed.device)]
        loss = objective(idx)
        opt.zero_grad(); loss.backward(); opt.step()
    transform.load_state_dict(state)
    return transform, history


@torch.no_grad()
def recovery_metrics(model, mapped, truth, values, unit):
    distance = torch.linalg.vector_norm(mapped[:, :2]-truth[:, :2], dim=1)
    return dict(**evaluate_field(model, mapped, values),
                rms_coordinate_error=float(distance.square().mean().sqrt())*unit,
                q95_coordinate_error=float(torch.quantile(distance, .95))*unit)


def fit_joint(model, observed, values, train_indices, validation_indices, settings, seed, update_transforms=True):
    """Optional alternating fit; middle transform is fixed. Truth is never supplied."""
    torch.manual_seed(seed)
    transforms = [SectionRigid(observed[0].device) for _ in observed]
    transforms[1].requires_grad_(False)
    model.requires_grad_(True); model.decoder.requires_grad_(False)
    field_opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=settings['field_lr'])
    pose_opt = torch.optim.Adam(list(transforms[0].parameters())+list(transforms[2].parameters()), lr=settings['registration_lr'])
    best, checkpoint, history = float('inf'), None, []
    # Anchor only to observed middle-section support, not undistorted outer coordinates.
    support = observed[1][train_indices[1]]
    for step in range(settings['joint_steps']+1):
        if step % 20 == 0 or step == settings['joint_steps']:
            with torch.no_grad():
                val = np.mean([evaluate_field(model, t(x[idx]), y[idx])['loss']
                               for t,x,y,idx in zip(transforms,observed,values,validation_indices)])
            history.append(dict(step=step, validation_loss=float(val)))
            if val < best:
                best = val
                checkpoint = (copy.deepcopy(model.state_dict()), [copy.deepcopy(t.state_dict()) for t in transforms])
        if step == settings['joint_steps']:
            break
        pose_step = update_transforms and step % 2 == 1
        field_opt.zero_grad(); pose_opt.zero_grad()
        loss = 0
        for t,x,y,idx in zip(transforms,observed,values,train_indices):
            sampled = idx[torch.randint(len(idx), (min(settings['batch_size'],len(idx)),), device=x.device)]
            mapped = t(x[sampled])
            if not pose_step:
                mapped = mapped.detach()
            loss = loss + expression_loss(model(mapped).exp(), y[sampled])/3
            if pose_step:
                loss = loss + settings['support_weight']*support_loss(mapped, support, settings['support_margin'])/3
                loss = loss + settings['transform_weight']*(t.translation.square().sum()+t.angle.square())
        loss = loss + settings['tilt_weight']*tilt_penalty(model)
        loss.backward()
        if pose_step:
            pose_opt.step()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            field_opt.step()
            xyz = torch.cat(observed)
            lower, upper = xyz.amin(0), xyz.amax(0)
            lower[:2]=0; upper[:2]=1
            project_field(model, settings, lower, upper)
    model.load_state_dict(checkpoint[0]); model.requires_grad_(False)
    for t,state in zip(transforms,checkpoint[1]):
        t.load_state_dict(state)
    return transforms, history
