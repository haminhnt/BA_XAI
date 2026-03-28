import pandas as pd
import requests
import re
import random
from sklearn.ensemble import RandomForestRegressor
from explainerdashboard import RegressionExplainer, ExplainerDashboard
from explainerdashboard.custom import *
from dash import html, dcc, ctx
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

# --- 1. DATA & MODEL PREPARATION ---
df = pd.read_csv('Walmart_Sales.csv')
# Create a readable Date string for the dropdown
df['Date_Str'] = pd.to_datetime(df['Date'], dayfirst=True).dt.strftime('%B %d, %Y')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Rename features to professional Business KPIs
X = df.drop(['Weekly_Sales', 'Date', 'Date_Str'], axis=1).rename(columns={
    'Holiday_Flag': 'Holiday Event', 
    'Temperature': 'Weather Impact',
    'Fuel_Price': 'Logistics Cost', 
    'CPI': 'Consumer Price Index', 
    'Unemployment': 'Labor Market Rate',
    'Store': 'Store ID'
})
y = df['Weekly_Sales']

# Training a quick model 
model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42).fit(X, y)
explainer = RegressionExplainer(model, X, y)

# --- 2. INTEGRATED EXECUTIVE COMPONENT ---
class ExecutiveStrategyComponent(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Supermarket Executive Strategy Hub")
        self.analysis_df = df.head(6436).copy()

        # Use unique dates and hide the earliest week.
        unique_dates = list(self.analysis_df['Date_Str'].drop_duplicates())
        if len(unique_dates) > 1:
            unique_dates = unique_dates[1:]
        self.date_options = [{'label': d, 'value': d} for d in unique_dates]

        self.default_date = unique_dates[0] if unique_dates else self.analysis_df.iloc[0]['Date_Str']
        default_store_options = self._store_options_for_date(self.default_date)
        self.default_store = default_store_options[0]['value'] if default_store_options else int(self.analysis_df.iloc[0]['Store'])

        # Fixed overall population baseline from the explainer background set (not per-store).
        self.global_population_avg = self._compute_global_population_avg()

        # Keep a row-level selector for Step 2 internal sync.
        self.row_options = [
            {'label': f"{row['Date_Str']} - Store {int(row['Store'])}", 'value': int(i)}
            for i, row in self.analysis_df.iterrows()
        ]

    def _compute_global_population_avg(self):
        for i in self.analysis_df.index:
            contrib_df = self.explainer.get_contrib_df(index=int(i), topx=None, sort='abs')
            if not contrib_df.empty and 'col' in contrib_df.columns and 'contribution' in contrib_df.columns:
                base_val = contrib_df.loc[
                    contrib_df['col'].astype(str).str.upper().isin(['BASE', '_BASE']),
                    'contribution'
                ].sum()
                if pd.notna(base_val):
                    return float(base_val)
        return 0.0

    def _store_options_for_date(self, date_str):
        stores = self.analysis_df.loc[self.analysis_df['Date_Str'] == date_str, 'Store'].drop_duplicates().tolist()
        return [{'label': f"Store {int(s)}", 'value': int(s)} for s in stores]

    def _get_row_index(self, date_str, store_id):
        matched = self.analysis_df[
            (self.analysis_df['Date_Str'] == date_str) & (self.analysis_df['Store'] == int(store_id))
        ]
        if not matched.empty:
            return int(matched.index[0])

        matched_date = self.analysis_df[self.analysis_df['Date_Str'] == date_str]
        if not matched_date.empty:
            return int(matched_date.index[0])

        return int(self.analysis_df.index[0])
        
    def layout(self):
        return html.Div([
                html.H1("Supermarket Sales Insights", 
                    style={'textAlign': 'center', 'color': '#306578', 'padding': '20px', 'fontFamily': 'Georgia, serif', 'fontSize': '2.5rem', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            
            # SECTION 1: REPORTING SELECTION
            html.Div([
                html.H4("📅 1. Choose a week and a store to get KPI reporting"),
                html.Div([
                    html.Div([
                        html.Label("Week:", style={'fontWeight': 'bold', 'marginBottom': '6px'}),
                        dcc.Dropdown(
                            id='date-selector',
                            options=self.date_options,
                            value=self.default_date,
                            style={'width': '350px', 'color': 'blue'}
                        )
                    ]),
                    html.Div([
                        html.Label("Store:", style={'fontWeight': 'bold', 'marginBottom': '6px'}),
                        dcc.Dropdown(
                            id='store-selector',
                            options=self._store_options_for_date(self.default_date),
                            value=self.default_store,
                            clearable=False,
                            style={'width': '300px'}
                        )
                    ]),
                    html.Div([
                        html.Button(
                            "Random week & store",
                            id='random-select-btn',
                            n_clicks=0,
                            style={
                                'backgroundColor': '#306578',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 16px',
                                'borderRadius': '8px',
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'height': '38px'
                            }
                        )
                    ])
                ], style={'display': 'flex', 'gap': '16px', 'alignItems': 'end', 'flexWrap': 'wrap'}),
                dcc.Loading(
                    id='kpi-loading',
                    type='dot',
                    color='#306578',
                    children=html.Div(id='kpi-summary-cards', style={
                        'display': 'flex', 'justifyContent': 'space-around', 'marginTop': '25px'
                    })
                ),
                html.Div([
                    html.Button(
                        "Show sales drivers",
                        id='reveal-step2-btn',
                        n_clicks=0,
                        disabled=True,
                        style={
                            'backgroundColor': '#306578',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 18px',
                            'borderRadius': '8px',
                            'cursor': 'pointer',
                            'fontWeight': 'bold'
                        }
                    )
                ], style={'marginTop': '20px'})
            ], style={'padding': '30px', 'backgroundColor': '#f8f9fa', 'borderRadius': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),

            # SECTION 2: ROOT CAUSE ANALYSIS (XAI)
            html.Div([
                html.H4("🔍 2. Why did sales go up or down?"),
                html.Div([
                    html.Label("Week to explain: ", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='waterfall-date-selector',
                        options=self.row_options,
                        value=self._get_row_index(self.default_date, self.default_store),
                        clearable=False,
                        disabled=True,
                        style={'width': '420px', 'display': 'inline-block'}
                    )
                ], style={'marginBottom': '20px'}),
                dcc.Graph(id='waterfall-graph', config={'displayModeBar': False}),
                html.Div([
                    html.Button(
                        "Show explanation",
                        id='reveal-step3-btn',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#306578',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 18px',
                            'borderRadius': '8px',
                            'cursor': 'pointer',
                            'fontWeight': 'bold'
                        }
                    )
                ], style={'marginTop': '10px'})
            ], id='step2-container', style={'marginTop': '40px', 'display': 'none'}),

            # SECTION 3: AI-GENERATED INSIGHTS (NLG)
            html.Div([
                html.H4("🤖 3. AI-generated summary"),
                dcc.Loading(
                    id='ai-loading',
                    type='circle',
                    color='#306578',
                    children=dcc.Markdown(
                        id='ai-narrative-box',
                        children='Choose a week to generate a simple explanation.',
                        dangerously_allow_html=True,
                        style={
                            'padding': '30px', 'backgroundColor': '#e3f2fd', 
                            'borderLeft': '10px solid #306578', 'fontSize': '14px',
                            'lineHeight': '1.6', 'borderRadius': '10px', 'color': '#306578',
                            'fontStyle': 'italic', 'minHeight': '90px', 'fontFamily': 'Georgia, serif'
                        }
                    )
                )
            ], id='step3-container', style={'marginTop': '40px', 'marginBottom': '100px', 'display': 'none'})
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'fontFamily': 'Georgia, serif', 'backgroundColor': '#f0efd8', 'padding': '20px', 'borderRadius': '10px'})

    def component_callbacks(self, app):
        @app.callback(
            Output('step2-container', 'style'),
            [Input('reveal-step2-btn', 'n_clicks')]
        )
        def reveal_step2(n_clicks):
            if n_clicks and n_clicks > 0:
                return {'marginTop': '40px', 'display': 'block'}
            return {'marginTop': '40px', 'display': 'none'}

        @app.callback(
            [Output('step3-container', 'style'),
             Output('reveal-step3-btn', 'children'),
             Output('reveal-step3-btn', 'disabled')],
            [Input('reveal-step3-btn', 'n_clicks')]
        )
        def reveal_step3(n_clicks):
            if n_clicks and n_clicks > 0:
                return (
                    {'marginTop': '40px', 'marginBottom': '100px', 'display': 'block'},
                    'Explanation shown',
                    True
                )
            return (
                {'marginTop': '40px', 'marginBottom': '100px', 'display': 'none'},
                'Show explanation',
                False
            )

        @app.callback(
            Output('date-selector', 'value'),
            Input('random-select-btn', 'n_clicks'),
            State('date-selector', 'value')
        )
        def select_random_date_store(n_clicks, current_date):
            if not n_clicks:
                return current_date

            if not self.date_options:
                return current_date

            random_date = random.choice([opt['value'] for opt in self.date_options])
            return random_date

        @app.callback(
            [Output('store-selector', 'options'),
             Output('store-selector', 'value')],
            Input('date-selector', 'value'),
            Input('random-select-btn', 'n_clicks'),
            State('store-selector', 'value')
        )
        def sync_store_selector(selected_date, random_clicks, current_store):
            if selected_date is None:
                selected_date = self.default_date

            options = self._store_options_for_date(selected_date)
            if not options:
                return [], None

            valid_values = [opt['value'] for opt in options]
            if ctx.triggered_id == 'random-select-btn':
                next_store = random.choice(valid_values)
            else:
                next_store = current_store if current_store in valid_values else valid_values[0]
            return options, next_store

        @app.callback(
            Output('waterfall-date-selector', 'value'),
            Input('date-selector', 'value'),
            Input('store-selector', 'value')
        )
        def sync_waterfall_date_selector(date_value, store_value):
            if date_value is None:
                date_value = self.default_date
            if store_value is None:
                store_value = self.default_store
            return self._get_row_index(date_value, store_value)

        @app.callback(
            Output('reveal-step2-btn', 'n_clicks'),
            Input('date-selector', 'value'),
            Input('store-selector', 'value')
        )
        def reset_step2_on_date_change(date_value, store_value):
            return 0

        @app.callback(
            Output('reveal-step3-btn', 'n_clicks'),
            Input('date-selector', 'value'),
            Input('store-selector', 'value')
        )
        def reset_step3_on_date_change(date_value, store_value):
            return 0

        @app.callback(
            Output('waterfall-graph', 'figure'),
            Input('waterfall-date-selector', 'value')
        )
        def update_waterfall_graph(selected_idx):
            if selected_idx is None:
                selected_idx = 1
            selected_idx = int(selected_idx)
            if selected_idx == 0:
                selected_idx = 1
            fig = self.explainer.plot_contributions(index=selected_idx)

            title_text = ''
            if fig.layout.title and fig.layout.title.text:
                title_text = str(fig.layout.title.text)
            if title_text and 'Contribution to prediction' in title_text:
                title_text = title_text.replace('Contribution to prediction', 'Contribution to Prediction<br><sup>') + '</sup>'

            fig.update_layout(
                font={'family': 'Georgia, serif'},
                title={
                    'text': (
                        title_text if title_text else 'Sales Driver Contribution'
                    ) + f"<br><sup>Population Average (overall baseline): ${self.global_population_avg:,.0f}</sup>",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                margin=dict(l=40, r=20, t=110, b=110),
                xaxis=dict(automargin=True, tickangle=-25),
                yaxis=dict(automargin=True)
            )
            return fig

        @app.callback(
            [Output('kpi-summary-cards', 'children'),
             Output('ai-narrative-box', 'children'),
             Output('reveal-step2-btn', 'disabled')],
            [Input('date-selector', 'value'),
             Input('store-selector', 'value')]
        )
        def update_dashboard(selected_date_value, selected_store_value):
            if selected_date_value is None:
                selected_date_value = self.default_date
            if selected_store_value is None:
                selected_store_value = self.default_store

            idx = self._get_row_index(selected_date_value, selected_store_value)
            can_reveal_step2 = False
            
            try:
                # Extract Metrics for the selected week
                actual_sales = self.explainer.y[idx]
                selected_date = df.iloc[idx]['Date_Str']
                selected_date_dt = df.iloc[idx]['Date']
                store_number = df.iloc[idx]['Store']
                
                # Calculate performance metrics
                avg_sales = self.explainer.y.mean()
                predictions = self.explainer.model.predict(self.explainer.X)
                predicted_sales = predictions[idx]
                variance = actual_sales - predicted_sales
                variance_pct = (variance / predicted_sales * 100) if predicted_sales != 0 else 0.0

                # Actual vs Predicted chart 
                comparison_fig = go.Figure(data=[
                    go.Bar(name='Actual Sales', x=['Sales'], y=[float(actual_sales)], marker_color='#1f77b4'),
                    go.Bar(name='Predicted Sales', x=['Sales'], y=[float(predicted_sales)], marker_color='#ff7f0e')
                ])
                comparison_fig.update_layout(
                    barmode='group',
                    title={
                        'text': 'Actual vs Predicted',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20}
                    },
                    font={'family': 'Georgia, serif'},
                    height=360,
                    margin=dict(l=30, r=20, t=80, b=40),
                    yaxis_title='Sales ($)',
                    yaxis=dict(automargin=True),
                    xaxis=dict(automargin=True)
                )

                # Feature comparison chart:
                # - Store ID: avg previous weekly sales of this store vs avg previous weekly sales of all stores.
                # - Other indicators: selected value vs avg indicator value in all weeks before selected week.
                historical_df = df[df['Date'] < selected_date_dt].copy()
                if historical_df.empty:
                    # Fallback for earliest week with no prior history.
                    historical_df = df.copy()

                selected_row = df.iloc[idx]
                feature_rows = []

                # Store revenue comparison (business replacement for Store ID).
                overall_avg_revenue = float(historical_df['Weekly_Sales'].mean())
                store_avg_revenue = float(
                    historical_df.loc[historical_df['Store'] == store_number, 'Weekly_Sales'].mean()
                )
                if pd.isna(store_avg_revenue):
                    store_avg_revenue = overall_avg_revenue
                feature_rows.append({
                    'Feature': 'Store Revenue',
                    'SelectedValue': store_avg_revenue,
                    'AverageValue': overall_avg_revenue
                })

                indicator_map = {
                    'Holiday Event': 'Holiday_Flag',
                    'Weather Impact': 'Temperature',
                    'Logistics Cost': 'Fuel_Price',
                    'Consumer Price Index': 'CPI',
                    'Labor Market Rate': 'Unemployment'
                }

                for display_name, raw_col in indicator_map.items():
                    if raw_col == 'Holiday_Flag':
                        # Keep Holiday Event as binary Yes/No for readability.
                        selected_val = 1.0 if float(selected_row[raw_col]) >= 0.5 else 0.0
                        avg_val = 1.0
                        selected_display = 'Yes' if selected_val >= 0.5 else 'No'
                        average_display = 'Yes'
                    else:
                        selected_val = float(selected_row[raw_col])
                        avg_val = float(historical_df[raw_col].mean())
                        selected_display = f"{selected_val:,.2f}"
                        average_display = f"{avg_val:,.2f}"
                    feature_rows.append({
                        'Feature': display_name,
                        'SelectedValue': selected_val,
                        'AverageValue': avg_val,
                        'SelectedDisplay': selected_display,
                        'AverageDisplay': average_display
                    })

                feature_compare_df = pd.DataFrame(feature_rows)

                feature_compare_df['PctOfAverage'] = (
                    feature_compare_df['SelectedValue'] / feature_compare_df['AverageValue'].replace(0, pd.NA)
                ).fillna(0) * 100

                feature_compare_fig = go.Figure(data=[
                    go.Bar(
                        x=feature_compare_df['Feature'],
                        y=feature_compare_df['PctOfAverage'],
                        marker_color=['#2ca02c' if v >= 100 else '#d62728' for v in feature_compare_df['PctOfAverage']],
                        customdata=feature_compare_df[['SelectedDisplay', 'AverageDisplay']].values,
                        text=[
                            ('Yes' if row['SelectedValue'] >= 0.5 else 'No') if row['Feature'] == 'Holiday Event' else f"{row['PctOfAverage']:.0f}%"
                            for _, row in feature_compare_df.iterrows()
                        ],
                        textposition='auto',
                        cliponaxis=False,
                        hovertemplate=(
                            '<b>%{x}</b><br>'
                            'Selected: %{customdata[0]}<br>'
                            'Average: %{customdata[1]}<br>'
                            'Selected vs Avg: %{y:.1f}%<extra></extra>'
                        )
                    )
                ])
                feature_compare_fig.update_layout(
                    title={
                        'text': 'Selected Week vs Average by Feature<br><sup>(100% = Average)</sup>',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 18}
                    },
                    font={'family': 'Georgia, serif'},
                    height=430,
                    margin=dict(l=30, r=20, t=110, b=110),
                    yaxis_title='Selected as % of Feature Average',
                    xaxis_title='Feature',
                    xaxis_tickangle=-25,
                    yaxis=dict(automargin=True),
                    xaxis=dict(automargin=True)
                )
                feature_compare_fig.add_hline(y=100, line_dash='dash', line_color='#306578')
                
                # Simple traditional KPI report
                cards = html.Div([
                    html.Div([
                        html.Div([
                            html.H5("📊 Weekly Sales Report", style={'color': '#0071ce', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                            html.P([html.Strong("Date: "), selected_date]),
                            html.P([html.Strong("Store ID: "), str(store_number)]),
                            html.P([html.Strong("Actual Sales: "), f"${actual_sales:,.0f}"]),
                            html.P([html.Strong("Predicted Sales: "), f"${predicted_sales:,.0f}"]),
                            html.P([html.Strong("Variance: "), f"${variance:,.0f} ({variance_pct:+.2f}%)"], style={'color': '#d32f2f' if variance < 0 else '#388e3c'}),
                            html.P([html.Strong("Store Average: "), f"${avg_sales:,.0f}"]),
                        ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '8px', 'lineHeight': '2', 'fontSize': '16px'}),
                        html.Div([
                            dcc.Graph(figure=comparison_fig, config={'displayModeBar': False})
                        ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '8px'})
                    ], style={
                        'display': 'grid',
                        'gridTemplateColumns': 'minmax(320px, 420px) minmax(320px, 1fr)',
                        'gap': '16px',
                        'alignItems': 'start',
                        'width': '100%'
                    }),
                    html.Div([
                        dcc.Graph(figure=feature_compare_fig, config={'displayModeBar': False})
                    ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '8px'})
                ], style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'gap': '16px',
                    'width': '100%'
                })
                can_reveal_step2 = True
                
            except Exception as e:
                can_reveal_step2 = False
                cards = html.Div([
                    html.P(f"Error generating KPI report: {str(e)[:200]}", style={'color': 'red'})
                ])
            
            # AI Prompting 
            top_feature = "N/A"
            ai_output = "⏳ Generating AI insight..."
            try:
                # Re-extract for safety
                actual_sales = self.explainer.y[idx]
                selected_date = df.iloc[idx]['Date_Str']
                store_number = df.iloc[idx]['Store']
                
                # Use local row-level contributions so the reason matches this specific prediction
                contrib_df = self.explainer.get_contrib_df(index=idx, topx=8, sort='abs')
                population_avg = float(self.global_population_avg)
                if not contrib_df.empty and 'col' in contrib_df.columns:
                    display_contrib_df = contrib_df[
                        ~contrib_df['col'].astype(str).str.upper().isin(['BASE', '_BASE', '_REST', 'REST'])
                    ].head(4)
                else:
                    display_contrib_df = contrib_df.head(0)

                if not display_contrib_df.empty:
                    top_feature = str(display_contrib_df.iloc[0]['col'])
                
                reason_lines = []
                for _, row in display_contrib_df.iterrows():
                    col = str(row.get('col', 'Unknown factor'))
                    val = float(row.get('contribution', 0.0))
                    col_label = col
                    if col == 'Store ID':
                        col_label = 'Store profile (historical pattern of this store)'
                    direction = 'UP' if val >= 0 else 'DOWN'
                    reason_lines.append(f"- Driver: {col_label} | Contribution: {val:+,.0f} | Direction: {direction}")
                reasons_text = "\n".join(reason_lines) if reason_lines else "- No detailed drivers were available."
                
                prompt = (
                    "You are a retail analyst explaining the variance for a non-technical store manager.\n"
                    "Use warm, natural, manager-friendly English. No AI-style phrases.\n"
                    "Start exactly with: Hi manager, I reviewed this week's sales in detail and here is what I found:\n"
                    "Never mention BASE, _BASE, REST, or _REST.\n"
                    "CRITICAL LOGIC RULE: if contribution is positive (+), describe it as helping/increasing sales.\n"
                    "If contribution is negative (-), describe it as reducing/pressuring sales. Never mix these.\n"
                    "RULES:\n"
                    "1. Start with: 'Hi manager, I reviewed this week's sales in detail and here is what I found:'\n"
                    "2. ALWAYS match the EXACT label with its EXACT contribution value and EXACT direction from the data provided.\n"
                    "3. If contribution is positive (+), use words like: 'boosted', 'helped', 'increased', 'positive contribution'.\n"
                    "4. If contribution is negative (-), use words like: 'dragged down', 'reduced', 'pressure', 'negative impact'.\n"
                    "5. NEVER invent facts outside the data\n\n"

                    "INDICATOR EXPLANATIONS (use these meanings in your wording):\n"
                    "- Population Baseline: the overall starting sales level before week-specific drivers are applied.\n"
                    "- Store profile (historical pattern of this store): recurring store-level sales pattern seen in past data, not store number quality.\n"
                    "- Consumer Price Index: inflation pressure; higher CPI usually reduces purchasing power.\n"
                    "- Weather Impact: local temperature effect on store traffic and shopping behavior.\n"
                    "- Logistics Cost: actual fuel cost pressure that can affect pricing and margin.\n"
                    "- Labor Market Rate: local unemployment conditions that influence customer spending.\n"
                    "- Holiday Event: holiday-week effect that can increase or shift demand.\n\n"
                    
                    "STRUCTURE (4 bullet points + 1 final overall advice line):\n"
                    "- Population Baseline: Mention the starting average ($1,046,956) and the meaning that the store may reach this number without any external influences.\n"
                    "- Top Positive Driver: Identify the factor with the highest positive (+) value and its specific impact.\n"
                    "- Top Negative Driver: Identify the factor with the highest negative (-) value and its specific impact.\n"
                    "- Secondary Factor: Mention one more meaningful factor from the list.\n"
                    "Final line (NOT a bullet): **Overall Advice:** one concise, actionable management recommendation that summarizes the week.\n\n"

                    "MANAGERIAL ADVICE RULES (global, adaptive):\n"
                    "- Advice must be derived from the strongest negative and positive drivers in this specific week only.\n"
                    "- Use contribution sign and magnitude correctly: negative drivers -> mitigation actions; positive drivers -> reinforcement actions.\n"
                    "- Never recommend an action that contradicts the direction of a driver.\n"
                    "- Provide one concrete action with owner + timeline + KPI in one sentence (WHO does WHAT by WHEN, and HOW success is measured).\n"
                    "- Keep actions practical and store-operational (pricing, promotion timing, inventory, staffing, display, checkout flow).\n"
                    "- If a contribution value is positive, do not describe it as a negative impact; if negative, do not describe it as a gain.\n\n"
                    

                    f"DATA FOR THE WEEK OF {selected_date}:\n"
                    f"- Global Baseline (Starting point): ${population_avg:,.0f}\n"
                    f"RAW DRIVERS DATA:\n"
                    f"{reasons_text}\n"
                    "\n"
                    "Write clearly, professionally, and double-check the math logic."
                    
                    f"Store: {store_number}\n"
                    f"Week: {selected_date}\n"
                    f"Population average (baseline): ${population_avg:,.0f}\n"
                    "Main drivers (structured):\n"
                    f"{reasons_text}\n"
                )
                
                r = requests.post('http://localhost:11434/api/generate', 
                                 json={"model": "llama3.2", "prompt": prompt, "stream": False}, 
                                 timeout=45)
                response_data = r.json()
                ai_output = response_data.get('response', 'No response generated').strip()
                ai_output = re.sub(r'(?i)\b_?base\b|\b_?rest\b', 'other factors', ai_output)
                ai_output = re.sub(r'(?i)^\s*(based on the data[:,]?\s*|this means that\s*)', '', ai_output)
                ai_output = re.sub(r'\s-\s(?=[A-Za-z\[])', '\n- ', ai_output)

                # Keep a manager-friendly multiline structure with bullets.
                lines = [ln.strip() for ln in ai_output.splitlines() if ln.strip()]
                if len(lines) <= 1:
                    text = lines[0] if lines else ai_output.strip()
                    if text.lower().startswith("hi manager, i reviewed this week's sales in detail and here is what i found:"):
                        rest = text[len("Hi manager, I reviewed this week's sales in detail and here is what I found:"):].strip()
                    elif text.lower().startswith('hi manager,'):
                        rest = text[len('Hi manager,'):].strip()
                    else:
                        rest = text
                    chunks = [c.strip() for c in re.split(r'(?<=[.!?])\s+', rest) if c.strip()]
                    lines = ["Hi manager, I reviewed this week's sales in detail and here is what I found:"] + [f"- {c}" for c in chunks]

                if lines and not lines[0].lower().startswith("hi manager, i reviewed this week's sales in detail and here is what i found:"):
                    lines.insert(0, "Hi manager, I reviewed this week's sales in detail and here is what I found:")

                formatted_lines = [f"<div style='margin:0'>{lines[0]}</div>"]
                body_lines = []
                for ln in lines[1:]:
                    cleaned = ln.lstrip('-*• ').strip()
                    body_lines.append(cleaned)

                overall_idx = -1
                for i, ln in enumerate(body_lines):
                    if ln.lower().startswith('overall advice:') or ln.lower().startswith('**overall advice:**'):
                        overall_idx = i
                        break

                for i, ln in enumerate(body_lines):
                    if i == overall_idx:
                        advice_text = ln.replace('**', '').strip()
                        if advice_text.lower().startswith('overall advice:'):
                            advice_text = advice_text[len('overall advice:'):].strip()
                        formatted_lines.append(
                            f"<div style='margin:0'><strong>Overall Advice:</strong> {advice_text}</div>"
                        )
                    else:
                        formatted_lines.append(f"<div style='margin:0 0 4px 0'>- {ln}</div>")

                if overall_idx == -1 and len(formatted_lines) > 2:
                    last_bullet = formatted_lines.pop()
                    advice_text = re.sub(r'<[^>]+>', '', last_bullet).lstrip('- ').strip()
                    formatted_lines.append(
                        f"<div style='margin:0'><strong>Overall Advice:</strong> {advice_text}</div>"
                    )
                ai_output = "".join(formatted_lines)
                
            except requests.exceptions.Timeout:
                ai_output = f"⏱️ Response timed out (Ollama might be slow). Top driver: {top_feature}"
            except requests.exceptions.ConnectionError:
                ai_output = "❌ Cannot connect to Ollama. Run: ollama serve"
            except Exception as e:
                ai_output = f"⚠️ AI Error: {str(e)[:100]}"
                
            return cards, ai_output, (not can_reveal_step2)

# --- 3. LAUNCH DASHBOARD ---
db = ExplainerDashboard(explainer, [ExecutiveStrategyComponent], 
                        title="",
                        header_hide_selector=True,
                        max_idxs_in_dropdown=7000)

if __name__ == "__main__":
    db.run(port=8081)