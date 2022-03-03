import logging
from typing import List, Tuple, Dict, Optional
import io
import numpy as np
from scipy.stats import binom
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px


pio.templates.default = 'plotly_white'
PLOT_WIDTH = 1000
PLOT_HEIGHT = 500


def image_data(fig) -> Image.Image:
    img_bytes = fig.to_image(format="png", width=PLOT_WIDTH, height=PLOT_HEIGHT)
    img = Image.open(io.BytesIO(img_bytes))
    return img


def pxbar(data, x, y, layout: Optional[dict] = None, **kwargs) -> Image.Image:
    if layout is None:
        layout = {}
    fig = px.bar(data, x=x, y=y, text='text', **kwargs)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, **layout)
    fig.update_traces(
        marker_color='rgba(117, 156, 108, 1.0)',
        textfont_color='white',
    )
    return image_data(fig)


def pxlinepct(data, x, y, ylim=None,  **kwargs) -> Image.Image:
    if ylim is None:
        ylim = (0, 1)
    fig = px.line(data, x=x, y=y, text='text', **kwargs)
    fig.update_traces(
        mode='markers+lines+text',
        line_color='#86b596',
        marker_size=12,
        marker_color='rgba(117, 156, 108, 1.0)',
        textfont_color='rgba(117, 156, 108, 1.0)',
        textposition="top center",
    )
    fig.update_layout(yaxis_tickformat='%', width=PLOT_WIDTH, height=PLOT_HEIGHT)
    fig.update_yaxes(range=ylim)
    return image_data(fig)


def class_balance(target_stats: dict) -> Image.Image:
    data_labels = [f"{target_stats[x]:.1%}" for x in ["negative_rate", "positive_rate"]]
    data = pd.DataFrame({
        "class": ["0", "1"],
        "count": [target_stats["num_negative"], target_stats["num_positive"]],
        "frequency": data_labels,
    })
    fig = px.bar(
        data,
        x="class",
        y="count",
        color="class",
        color_discrete_sequence=["red", "green"],
        text="frequency",
    )
    fig.update_layout(
        title={
            'text': 'Target classes balance',
            'x': 0.5,
            'xanchor': 'center',
        },
    )
    return image_data(fig)


def feature_importance(featimp: List[Tuple[str, float]]) -> Image.Image:
    nf = len(featimp)
    data = pd.DataFrame(featimp, columns=["feature_name", "feature_importance"])
    data = data.sort_values(by="feature_importance")
    fig = go.Figure()
    annotations = [f"{v:.3g}" for v in data["feature_importance"]]
    fig.add_trace(go.Bar(
        x=data['feature_importance'],
        y=data['feature_name'],
        orientation='h',
        name='feature importance',
        marker=dict(color='rgba(117, 156, 108, 1.0)'),
        text=annotations,
        textposition="inside",
        textfont=dict(
            # family="sans serif",
            size=10,
            color="white"
        ),
    ))
    fig.update_layout(
        width=PLOT_WIDTH,
        height=60*nf,
        barmode='stack',
        title={
            'text': f'Feature importance for {nf} features',
            'x': 0.5,
            'xanchor': 'center',
        },
        xaxis_title="Feature importance",
        yaxis_title="Feature",
        template='plotly_white',
    )
    return image_data(fig)


def gauge(
        label: str,
        value: float,
        value_range: Tuple[float, float],
        color_ranges: Dict[str, Tuple[float, float]],
        **kwargs,
) -> Image.Image:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': label, 'font': {'size': 24}},
        gauge={
            'axis': {'range': value_range, 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': rng, 'color': color}
                for color, rng in color_ranges.items()
            ],
        },
        **kwargs
    ))
    return image_data(fig)


def binom_ci_scalar(nall: int, npos: int, confidence: float = 0.90) -> tuple:
    assert nall >= 0
    assert npos >= 0
    assert npos <= nall
    if nall == 0:
        lb = 0.0
        ub = 0.0
    else:
        if npos == 0:
            prob = 1 / (nall + 1)
        elif npos == nall:
            prob = nall / (nall + 1)
        else:
            prob = npos / nall
        alpha = 1 - confidence
        lb = binom.ppf(alpha/2, n=nall, p=prob) / nall
        ub = binom.ppf(1 - alpha/2, n=nall, p=prob) / nall
    return lb, ub


def univariate(
        data,
        yticks=None,
        title: str = None,
        y_label: str = None,
        width: int = PLOT_WIDTH,
        height: int = PLOT_HEIGHT,
) -> Image.Image:
    """
    Plot univariate analysis graph.

    :param data: list of dict, where each dict is with the following keys
        bin
        num_observations
        num_positive
        positive_rate
        frequency
    :param yticks: optional vector of yticks (between 0 and 1)
    :param title: optional title
    :param y_label: optional label for the vertical axis.
    :param width: width in pixels
    :param height: height in pixels
    :return:
    """
    if data:
        decimal_places = 2
        posrate = np.array([np.nan if v['positive_rate'] is None else v['positive_rate'] for v in data])
        cis = [binom_ci_scalar(v["num_observations"], v["num_positive"]) for v in data]
        posrate_lb = np.array([ci[0] for ci in cis])
        posrate_ub = np.array([ci[1] for ci in cis])
        w = np.array([np.nan if v['frequency'] is None else v['frequency'] for v in data])
        posrate_total = np.nansum(w * posrate)
        if yticks is None:
            maxub = np.max(posrate_ub)
            maxy = np.ceil(1.02*100*maxub)/100
            yticks = np.arange(start=0.0, stop=maxy+0.01, step=0.1)
        bins = [v["bin"] for v in data]
        posrate_annotations = [
            f"{r:.1%}&plusmn;{(ci[1] - ci[0])/2:.1%}"
            for r, ci in zip(posrate, cis)
        ]
        upper_bound = go.Scatter(
            name='Upper Bound',
            x=bins,
            y=100*posrate_ub,
            mode='lines+text',
            marker=dict(color="#444"),
            line=dict(width=0),
            text=posrate_annotations,
            textposition="top center",
            textfont=dict(
                # family="sans serif",
                size=10,
                color="red"
            ),
            hovertemplate='%{y:.' + str(decimal_places) + 'f}',
            fillcolor='rgba(220, 0, 0, 0.3)',
            fill='tonexty')

        trace = go.Scatter(
            name=y_label,
            x=bins,
            y=100*posrate,
            mode='lines',
            line=dict(color='red', width=3),
            fillcolor='rgba(220, 0, 0, 0.3)',
            hovertemplate='%{y:.' + str(decimal_places) + 'f}',
            fill='tonexty')

        lower_bound = go.Scatter(
            name='Lower Bound',
            x=bins,
            y=100*posrate_lb,
            marker=dict(color="#444"),
            line=dict(width=0),
            hovertemplate='%{y:.' + str(decimal_places) + 'f}',
            mode='lines')

        reference_line = go.Scatter(
            name='reference_Line',
            x=[bins[0], bins[-1]],
            y=[100*posrate_total, 100*posrate_total],
            mode='lines',
            line=dict(color='red', width=1, dash="dash"),
        )

        # Trace order can be important
        # with continuous error bars
        figure_data = [lower_bound, trace, upper_bound, reference_line]

        layout = go.Layout(
            showlegend=False)

        annotations = [dict(
            x=row["bin"],
            y=1.1,
            xref="x",
            yref="paper",
            text=f"{100*row['frequency']:.0f}% ({row['num_observations']})",
            showarrow=False,
            ax=0,
        ) for row in data]

        fig = go.Figure(data=figure_data, layout=layout)
        # logging.debug(f"yticks={yticks}")
        fig.update_layout(
            width=width,
            height=height,
            yaxis=dict(
                title=y_label,
                tickmode='array',
                tickvals=100*yticks,
                ticktext=[f"{100*t:.0f}%" for t in yticks]
            ),
            hovermode="x",
            annotations=annotations,
            template='plotly_white',
        )
        fig.update_yaxes(range=[0, 100 * max(yticks)])
    else:
        fig = go.Figure(data=None, layout=go.Layout(template='plotly_white'))
    if title:
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center')
        )
    return fig


def predictions_histogram(histogram_data: List[dict]) -> Image.Image:
    data = pd.DataFrame({
        "predicted_probability": [(x['lb'] + x['ub'])/2 for x in histogram_data],
        "pdf": [x['pdf'] for x in histogram_data],
        "n": [x['n'] for x in histogram_data]
    })
    fig = px.bar(
        data,
        x="predicted_probability",
        y="pdf",
        text="n",
    )
    fig.update_layout(
        xaxis_title="predicted probability",
        yaxis_title="probability density function estimate",
        title={
            'text': 'Predicted probability histogram',
            'x': 0.5,
            'xanchor': 'center',
        },
    )
    return image_data(fig)
