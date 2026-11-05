"""Interactive diagnostics only; does not fit or modify volumetric fields."""
import numpy as np
import torch
import plotly.graph_objects as go


def highest_tilt_figure(model, xyz, *, name, top_percent=5., z_stretch=10.,
                        section_labels=None):
    """Match the static three-plane view, selecting by actual learned tilt."""
    if not 0 < top_percent <= 100:
        raise ValueError('top_percent must be in (0, 100]')
    if not np.isfinite(z_stretch) or z_stretch <= 0:
        raise ValueError('z_stretch must be positive and finite')
    if not xyz or any(len(points) == 0 for points in xyz):
        raise ValueError('Nonempty section coordinates required')
    field = model.splat_field
    with torch.no_grad():
        means = model.means.detach().cpu().numpy()
        scales = model.log_scales.exp().detach().cpu().numpy()
        tilt = field.tilt_matrices().detach().cpu().numpy()
        yaw = model.rotation.detach().cpu().numpy().ravel()
        angles = torch.rad2deg(field.tilt_vectors().norm(dim=1)).cpu().numpy()
    sections = [points.detach().cpu().numpy() for points in xyz]
    labels = list(section_labels) if section_labels is not None else list(range(1, len(sections)+1))
    if len(labels) != len(sections):
        raise ValueError('One label per section required')
    count = max(1, int(np.ceil(len(means)*top_percent/100)))
    selected = np.argsort(-angles, kind='stable')[:count]
    fig = go.Figure()
    colors = ['#0072b2', '#777777', '#e69f00', '#a656a0']
    for i, (points, label) in enumerate(zip(sections, labels)):
        shown = points[::max(1, len(points)//1500)]
        fig.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2],
            mode='markers', marker=dict(size=2, color=colors[i % len(colors)], opacity=.12),
            name=f'Section {label}', hoverinfo='skip'))
    theta = np.linspace(0, 2*np.pi, 60)
    for k in selected:
        c, s = np.cos(yaw[k]), np.sin(yaw[k])
        rotation = tilt[k] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        contours = []
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            circle = np.zeros((len(theta), 3))
            circle[:, a] = np.cos(theta)*scales[k, a]
            circle[:, b] = np.sin(theta)*scales[k, b]
            contours.extend((circle @ rotation.T + means[k], np.full((1, 3), np.nan)))
        points = np.concatenate(contours)
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='lines', line=dict(color='#238b8b', width=2), opacity=.45,
            name=f'Splat {k}', legendgroup='splats', showlegend=False,
            hovertemplate=f'Splat {k}<br>Tilt: {angles[k]:.4g} degrees'
                          '<br>x: %{x:.4f}<br>y: %{y:.4f}<br>z: %{z:.4f}<extra></extra>'))
    fig.add_trace(go.Scatter3d(x=means[selected, 0], y=means[selected, 1], z=means[selected, 2],
        mode='markers', marker=dict(size=3, color='#176565'), name='Selected splat centers',
        customdata=np.column_stack((selected, angles[selected])),
        hovertemplate='Splat %{customdata[0]:.0f}<br>Tilt: %{customdata[1]:.4g} degrees'
                      '<br>x: %{x:.4f}<br>y: %{y:.4f}<br>z: %{z:.4f}<extra></extra>'))
    points = np.concatenate(sections)
    low, high = points.min(0)-.03, points.max(0)+.03
    span = high-low

    def aspect(stretch):
        ratio = span*np.array([1., 1., stretch])
        ratio /= ratio.max()
        return dict(zip(('x', 'y', 'z'), ratio.tolist()))

    elevation, azimuth = np.deg2rad([20, -55])
    eye = dict(x=1.8*np.cos(elevation)*np.cos(azimuth),
               y=1.8*np.cos(elevation)*np.sin(azimuth), z=1.8*np.sin(elevation))
    fig.update_layout(
        title=dict(text=f'{name}: top {top_percent:g}% by learned tilt ({count}/{len(means)})'
                   f'<br><sup>{angles[selected].min():.2f} to {angles[selected].max():.2f} degrees</sup>'),
        font=dict(family='Helvetica, Nimbus Sans, Arial, sans-serif'),
        height=780, margin=dict(l=0, r=0, b=10, t=110), template='plotly_white',
        scene=dict(xaxis=dict(title='x', range=[low[0], high[0]]),
                   yaxis=dict(title='y', range=[low[1], high[1]]),
                   zaxis=dict(title=f'z (display stretched {z_stretch:g}x)', range=[low[2], high[2]]),
                   aspectmode='manual', aspectratio=aspect(z_stretch),
                   camera=dict(eye=eye), dragmode='orbit'),
        legend=dict(orientation='h', y=-.04, x=.5, xanchor='center'),
        uirevision='highest-tilt-view',
        updatemenus=[dict(type='buttons', direction='right', x=0, y=1.07, buttons=[
            dict(label=f'z stretch {z_stretch:g}x', method='relayout',
                 args=[{'scene.aspectratio': aspect(z_stretch),
                        'scene.zaxis.title.text': f'z (display stretched {z_stretch:g}x)'}]),
            dict(label='Equal units', method='relayout',
                 args=[{'scene.aspectratio': aspect(1.), 'scene.zaxis.title.text': 'z (equal units)'}])])])
    return fig
