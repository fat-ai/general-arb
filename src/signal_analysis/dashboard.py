import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.express as px

# PAGE CONFIG
st.set_page_config(layout="wide", page_title="Strategy Dashboard")
DATA_PATH = "simulation_results.parquet"

# --- LAZY DATA LOADERS ---
@st.cache_data
def get_market_list():
    return pl.scan_parquet(DATA_PATH).select("fpmm").unique().collect()["fpmm"].to_list()

@st.cache_data
def get_market_data(market_name):
    return pl.scan_parquet(DATA_PATH).filter(pl.col("fpmm") == market_name).sort("timestamp").collect()

@st.cache_data
def get_aggregated_stats():
    # Helper for the scatter plot
    return pl.scan_parquet(DATA_PATH).group_by("fpmm").agg([
        pl.col("signal_strength").max().alias("max_signal"),
        pl.col("outcome").max().alias("outcome"),
        pl.col("trade_volume").sum().alias("total_vol")
    ]).collect()

@st.cache_data
def get_temporal_aggregation(time_bin="1h"):
    # 1. Round timestamps to the nearest bin (e.g., 1 hour)
    # 2. Group by (TimeBin, Outcome)
    # 3. Calculate Median, P25, P75
    
    q = pl.scan_parquet(DATA_PATH).with_columns(
        pl.col("timestamp").dt.truncate(time_bin).alias("time_bin")
    ).group_by(["time_bin", "outcome"]).agg([
        pl.col("signal_strength").median().alias("median_sig"),
        pl.col("signal_strength").quantile(0.25).alias("p25_sig"),
        pl.col("signal_strength").quantile(0.75).alias("p75_sig"),
        pl.len().alias("count")
    ]).sort("time_bin")
    
    return q.collect()

# --- SIDEBAR ---
st.sidebar.title("🔍 Navigation")
view_mode = st.sidebar.radio("Go to:", ["Market Inspector", "Aggregated Analysis", "Temporal Analysis"])

# --- VIEW 1: MARKET INSPECTOR ---
if view_mode == "Market Inspector":
    st.title("📈 Single Market Deep Dive")
    all_markets = get_market_list()
    selected_market = st.selectbox("Search Market:", all_markets)
    
    if selected_market:
        df = get_market_data(selected_market)
        outcome_val = df["outcome"][0]
        outcome_lbl = "YES" if outcome_val == 1 else "NO"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Outcome", outcome_lbl, delta_color="normal" if outcome_val==1 else "inverse")
        c2.metric("Max Signal", f"{df['signal_strength'].max():.2f}")
        c3.metric("Data Points", len(df))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["signal_strength"], mode='lines', name='Signal', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["trade_price"], mode='lines', name='Price', line=dict(color='gray', dash='dot'), yaxis="y2"))
        
        fig.update_layout(height=600, title=f"Timeline: {selected_market}", 
                          yaxis2=dict(title="Price", overlaying="y", side="right", range=[0,1]))
        st.plotly_chart(fig, use_container_width=True)

# --- VIEW 2: AGGREGATED ANALYSIS ---
elif view_mode == "Aggregated Analysis":
    st.title("🦅 All Markets Overview")
    min_vol = st.slider("Filter: Min Volume ($)", 0, 10000, 100)
    stats = get_aggregated_stats().filter(pl.col("total_vol") > min_vol)
    
    st.subheader("Signal Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=stats.filter(pl.col("outcome")==1)["max_signal"], name="YES", marker_color="green", opacity=0.7))
    fig_hist.add_trace(go.Histogram(x=stats.filter(pl.col("outcome")==0)["max_signal"], name="NO", marker_color="red", opacity=0.7))
    fig_hist.update_layout(barmode='overlay', title="Distribution of Max Signals")
    st.plotly_chart(fig_hist, use_container_width=True)

# --- VIEW 3: TEMPORAL ANALYSIS (New Tab) ---
elif view_mode == "Temporal Analysis":
    st.title("⏳ Signal Evolution Over Time")
    st.write("Comparing the **Median Signal Strength** of winning vs. losing markets over time.")
    
    time_bin = st.selectbox("Time Binning:", ["1h", "4h", "1d"], index=0)
    
    with st.spinner("Aggregating millions of rows..."):
        # This reduces 8GB of data down to just a few hundred rows for plotting
        agg_df = get_temporal_aggregation(time_bin)
    
    # Split into YES and NO series
    yes_series = agg_df.filter(pl.col("outcome") == 1)
    no_series = agg_df.filter(pl.col("outcome") == 0)
    
    fig = go.Figure()
    
    # --- YES MARKETS (Green) ---
    # 1. The Median Line
    fig.add_trace(go.Scatter(
        x=yes_series["time_bin"], y=yes_series["median_sig"],
        mode='lines', name='YES (Median)', line=dict(color='green', width=3)
    ))
    # 2. The Upper/Lower Bounds (Shaded Area)
    fig.add_trace(go.Scatter(
        x=yes_series["time_bin"].to_list() + yes_series["time_bin"].to_list()[::-1],
        y=yes_series["p75_sig"].to_list() + yes_series["p25_sig"].to_list()[::-1],
        fill='toself', fillcolor='rgba(0,255,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False
    ))
    
    # --- NO MARKETS (Red) ---
    # 1. The Median Line
    fig.add_trace(go.Scatter(
        x=no_series["time_bin"], y=no_series["median_sig"],
        mode='lines', name='NO (Median)', line=dict(color='red', width=3)
    ))
    # 2. The Upper/Lower Bounds
    fig.add_trace(go.Scatter(
        x=no_series["time_bin"].to_list() + no_series["time_bin"].to_list()[::-1],
        y=no_series["p75_sig"].to_list() + no_series["p25_sig"].to_list()[::-1],
        fill='toself', fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False
    ))
    
    fig.update_layout(
        title="Average Signal Strength Over Time (with 25-75% Confidence Bands)",
        xaxis_title="Date",
        yaxis_title="Signal Strength",
        height=600,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
