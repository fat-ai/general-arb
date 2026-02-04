import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.express as px

# PAGE CONFIG
st.set_page_config(layout="wide", page_title="Strategy Dashboard")
DATA_PATH = "simulation_results.parquet"

# --- LAZY DATA LOADER ---
# This function caches the *list* of markets so we don't scan the file every time
@st.cache_data
def get_market_list():
    # We scan the parquet, select unique markets, and collect only that small list
    return pl.scan_parquet(DATA_PATH).select("fpmm").unique().collect()["fpmm"].to_list()

@st.cache_data
def get_market_data(market_name):
    # Fetches only rows for the specific market
    return pl.scan_parquet(DATA_PATH).filter(pl.col("fpmm") == market_name).sort("timestamp").collect()

@st.cache_data
def get_aggregated_stats():
    # scans the file to create the summary scatter plot
    # Group by market and get Max Signal + Final Outcome
    q = pl.scan_parquet(DATA_PATH).group_by("fpmm").agg([
        pl.col("signal_strength").max().alias("max_signal"),
        pl.col("outcome").max().alias("outcome"), # 1 or 0
        pl.col("trade_volume").sum().alias("total_vol")
    ])
    return q.collect()

# --- SIDEBAR ---
st.sidebar.title("🔍 Navigation")
view_mode = st.sidebar.radio("Go to:", ["Market Inspector", "Aggregated Analysis"])

# --- VIEW 1: MARKET INSPECTOR ---
if view_mode == "Market Inspector":
    st.title("📈 Single Market Deep Dive")
    
    # 1. Select Market
    all_markets = get_market_list()
    selected_market = st.selectbox("Search Market:", all_markets)
    
    if selected_market:
        # 2. Load Data (Lazy)
        df = get_market_data(selected_market)
        
        # 3. Stats Header
        outcome_val = df["outcome"][0]
        outcome_color = "green" if outcome_val == 1 else "red"
        outcome_lbl = "YES" if outcome_val == 1 else "NO"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Outcome", outcome_lbl, delta_color="normal" if outcome_val==1 else "inverse")
        c2.metric("Max Signal", f"{df['signal_strength'].max():.2f}")
        c3.metric("Data Points", len(df))
        
        # 4. Main Chart (Signal vs Price)
        fig = go.Figure()
        
        # Signal Line
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["signal_strength"],
            mode='lines', name='Signal Strength',
            line=dict(color='blue', width=2)
        ))
        
        # Price Line (Secondary Axis)
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["trade_price"],
            mode='lines', name='Price',
            line=dict(color='gray', dash='dot'),
            yaxis="y2"
        ))
        
        fig.update_layout(
            title=f"Timeline: {selected_market}",
            height=600,
            yaxis=dict(title="Signal"),
            yaxis2=dict(title="Price ($)", overlaying="y", side="right", range=[0,1]),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- VIEW 2: AGGREGATED ANALYSIS ---
elif view_mode == "Aggregated Analysis":
    st.title("🦅 All Markets Overview")
    st.write("Does High Signal correlate with YES outcomes?")
    
    # Load Stats
    with st.spinner("Crunching data..."):
        stats = get_aggregated_stats()
        
    # Filter for cleaner viz
    min_vol = st.slider("Filter: Min Volume ($)", 0, 10000, 100)
    filtered_stats = stats.filter(pl.col("total_vol") > min_vol)
    
    # 1. Jitter Plot (Outcome vs Signal)
    fig_agg = px.strip(
        filtered_stats.to_pandas(), # Convert tiny Agg result to pandas for Plotly
        x="outcome", 
        y="max_signal", 
        color="outcome",
        hover_data=["fpmm", "total_vol"],
        stripmode="overlay",
        title="Max Signal vs Outcome (Strip Plot)"
    )
    
    fig_agg.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['NO', 'YES']))
    st.plotly_chart(fig_agg, use_container_width=True)
    
    # 2. Histogram Comparison
    st.subheader("Signal Distribution")
    fig_hist = go.Figure()
    
    yes_sigs = filtered_stats.filter(pl.col("outcome")==1)["max_signal"]
    no_sigs = filtered_stats.filter(pl.col("outcome")==0)["max_signal"]
    
    fig_hist.add_trace(go.Histogram(x=yes_sigs, name="YES Outcomes", marker_color="green", opacity=0.7))
    fig_hist.add_trace(go.Histogram(x=no_sigs, name="NO Outcomes", marker_color="red", opacity=0.7))
    
    fig_hist.update_layout(barmode='overlay', title="Distribution of Max Signals")
    st.plotly_chart(fig_hist, use_container_width=True)
