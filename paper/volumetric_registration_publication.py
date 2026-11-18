"""Static overview and equal-scale tilt insets for fitted volumetric splats."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import proj3d


@torch.no_grad()
def tilt_publication_figure(model, xyz, *, section_labels, top_usage_percent=5.,
                            z_stretch=10., batch_size=512):
    """Select examples by explicit rules; never change the model or fit it."""
    if not 0 < top_usage_percent <= 100 or not np.isfinite(z_stretch) or z_stretch <= 0:
        raise ValueError('Invalid percentage or display stretch')
    if len(xyz) != len(section_labels) or not xyz or any(len(x) == 0 for x in xyz):
        raise ValueError('Nonempty sections and matching labels required')
    field = model.splat_field
    means = model.means.detach().cpu().numpy()
    scales = model.log_scales.exp().detach().cpu().numpy()
    tilts = field.tilt_matrices().detach().cpu().numpy()
    yaw = model.rotation.detach().cpu().numpy().ravel()
    angles = torch.rad2deg(field.tilt_vectors().norm(dim=1)).cpu().numpy()
    usage = torch.zeros(len(means), device=model.means.device)
    for points in xyz:
        for chunk in points.split(batch_size):
            usage += field.compute_weights(chunk.to(model.means.device)).sum(0)/(len(points)*len(xyz))
    usage = usage.cpu().numpy()
    count = max(1, int(np.ceil(len(means)*top_usage_percent/100)))
    high_usage = np.argsort(-usage, kind='stable')[:count]
    largest = int(np.argmax(angles))
    ordered = high_usage[np.argsort(angles[high_usage], kind='stable')]
    median = int(ordered[(len(ordered)-1)//2])
    examples = [largest, median]
    shown = np.unique(np.concatenate([high_usage, [largest]]))
    sections = [x.detach().cpu().numpy() for x in xyz]
    bounds = np.concatenate(sections)
    low, high = bounds.min(0)-.03, bounds.max(0)+.03
    fig = plt.figure(figsize=(12, 8), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1])
    overview = fig.add_subplot(grid[:, 0], projection='3d')
    colors = ['#0072b2', '#777777', '#e69f00', '#a656a0']
    section_handles = []
    for i, (points, label) in enumerate(zip(sections, section_labels)):
        sample = points[::max(1, len(points)//1500)]
        color = colors[i % len(colors)]
        overview.scatter(*sample.T, color=color, s=1, alpha=.12, rasterized=True)
        section_handles.append(Line2D([], [], marker='o', linestyle='', color=color,
                                      markersize=4, label=f'Section {label}'))
    norm = plt.Normalize(0, max(float(angles.max()), 1.))
    cmap = plt.get_cmap('viridis')
    scatter = overview.scatter(*means[shown].T, c=angles[shown], cmap=cmap, norm=norm,
                               s=22, edgecolors='white', linewidths=.35, depthshade=False)
    overview.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
                 xlabel='x', ylabel='y', zlabel=f'z (display exaggerated {z_stretch:g}x)',
                 title=f'A  Spatial context\nTop {top_usage_percent:g}% by usage + maximum-tilt splat')
    overview.set_box_aspect((high-low)*np.array([1., 1., z_stretch]))
    overview.view_init(elev=20, azim=-55)
    overview.tick_params(labelsize=8)
    overview.legend(handles=section_handles, loc='upper center', bbox_to_anchor=(.5, -.12),
                    ncol=len(sections), fontsize=8, frameon=False)
    bar = fig.colorbar(scatter, ax=overview, orientation='horizontal', shrink=.7, pad=.18)
    bar.set_label('True out-of-plane tilt (degrees)')
    details = []
    inset_axes = []
    theta = np.linspace(0, 2*np.pi, 160)
    # A shared equal-unit inset scale permits direct size comparison.
    extent = float(scales[examples].max())*1.25
    for row, (k, label, rule) in enumerate(zip(examples, ['B', 'C'],
                                              ['Maximum tilt', 'Median tilt among high-usage splats'])):
        ax = fig.add_subplot(grid[row, 1])
        inset_axes.append(ax)
        c, s = np.cos(yaw[k]), np.sin(yaw[k])
        rotation = tilts[k] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        depth = rotation[:, 2]
        horizontal = depth[:2]
        horizontal = horizontal/np.linalg.norm(horizontal) if np.linalg.norm(horizontal) > 1e-8 else np.array([1., 0.])
        projection = np.array([[horizontal[0], horizontal[1], 0], [0, 0, 1]])
        covariance = rotation @ np.diag(scales[k]**2) @ rotation.T
        eigenvalues, eigenvectors = np.linalg.eigh(projection @ covariance @ projection.T)
        outline = np.stack((np.cos(theta), np.sin(theta)), 1) @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        color = cmap(norm(angles[k]))
        ax.fill(*outline.T, color=color, alpha=.2)
        ax.plot(*outline.T, color=color, linewidth=2)
        axis = np.array([-extent*.85, extent*.85])[:, None]*depth
        ax.plot(*(axis @ projection.T).T, color='#c44e19', linewidth=2)
        ax.axvline(0, color='#555555', linestyle='--', linewidth=1)
        ax.scatter([0], [0], color='black', s=15, zorder=5)
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect='equal',
               xlabel='Horizontal offset along tilt direction', ylabel='z offset',
               title=f'{label}  {rule}\nSplat {k}: {angles[k]:.2f} degrees')
        ax.grid(alpha=.15)
        ax.tick_params(labelsize=8)
        details.append(dict(panel=label, splat=int(k), selection=rule,
                            tilt_degrees=float(angles[k]), mean_responsibility=float(usage[k]),
                            center=means[k].tolist(), scales=scales[k].tolist()))
    inset_axes[-1].legend(handles=[
        Line2D([], [], color='#c44e19', label='Depth-axis direction'),
        Line2D([], [], color='#555555', linestyle='--', label='Vertical reference')],
        loc='upper center', bbox_to_anchor=(.5, -.22), ncol=2, frameon=False, fontsize=8)
    fig.suptitle('Volumetric splats: spatial context and true geometry\n'
                 'B-C: equal-scale side-view silhouettes; coordinates relative to each center',
                 fontsize=13)
    # Project endpoints after layout settles; connectors live outside the layout calculation.
    fig.canvas.draw()
    for label, k, inset in zip(['B', 'C'], examples, inset_axes):
        px, py, _ = proj3d.proj_transform(*means[k], overview.get_proj())
        annotation = overview.annotate(label, (px, py), xytext=(-12, 10), textcoords='offset points',
            fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor='#555555', pad=2),
            arrowprops=dict(arrowstyle='-', color='#555555'))
        annotation.set_in_layout(False)
        connector = ConnectionPatch(xyA=(px, py), coordsA=overview.transData,
                                    xyB=(0, .5), coordsB=inset.transAxes,
                                    color='#888888', linewidth=.7, linestyle=':', alpha=.7)
        connector.set_in_layout(False)
        fig.add_artist(connector)
    return fig, details
