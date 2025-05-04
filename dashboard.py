import dash
from dash import dcc, html
import os

app = dash.Dash(__name__)

output_dir = 'data_archive'  # 替换为实际路径
chart_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.html')])

app.layout = html.Div([
    html.Div(
        html.H1("专业级股票分析看板", 
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'fontFamily': 'SimHei',
                    'fontSize': '28px',
                    'marginBottom': '20px'
                }),
        style={'backgroundColor': '#f8f9fa', 'padding': '20px'}
    ),
    dcc.Tabs(
        id="chart-tabs",
        value=chart_files[0],
        children=[
            dcc.Tab(
                label=os.path.splitext(f)[0].replace('_analysis', ''),
                value=f,
                children=[
                    html.Iframe(
                        srcDoc=open(os.path.join(output_dir, f), 'r').read(),
                        style={
                            'width': '100%',
                            'height': '2200px',  # 与图表高度匹配
                            'border': 'none',
                            'background': '#FFFFFF'
                        }
                    )
                ],
                style={
                    'backgroundColor': '#e9ecef',
                    'fontFamily': 'SimHei',
                    'fontWeight': 'bold'
                },
                selected_style={
                    'backgroundColor': '#FFFFFF',
                    'borderTop': '3px solid #2c3e50'
                }
            ) for f in chart_files
        ],
        colors={
            "border": "#dee2e6",
            "primary": "#2c3e50",
            "background": "#f8f9fa"
        }
    )
], style={'backgroundColor': '#FFFFFF'})

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)