import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Sampling & Survey Simulator", layout="wide")

st.title("📊 Sampling & Survey Simulator")
st.markdown("Demonstrate random sampling, sample size, selection bias, sampling distributions, and why samples differ from populations.")

# Sidebar Controls
st.sidebar.header("1. Population Parameters")
pop_dist = st.sidebar.selectbox("Distribution Type", ["Normal", "Right-Skewed (Income)", "Bimodal", "Uniform", "Likert (1-5)"])
pop_size = st.sidebar.slider("True Population Size ($N$)", 1000, 50000, 10000, step=1000)

@st.cache_data
def generate_population(dist, size):
    np.random.seed(42)
    if dist == "Normal":
        data = np.random.normal(loc=100, scale=15, size=size)
    elif dist == "Right-Skewed (Income)":
        data = np.random.exponential(scale=35000, size=size) + 20000
    elif dist == "Bimodal":
        data = np.concatenate([np.random.normal(60, 10, int(size*0.4)), np.random.normal(120, 15, int(size*0.6))])
    elif dist == "Uniform":
        data = np.random.uniform(10, 100, size=size)
    elif dist == "Likert (1-5)":
        data = np.random.choice([1, 2, 3, 4, 5], size=size, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    return pd.DataFrame({'Value': data})

pop_df = generate_population(pop_dist, pop_size)
true_mean = pop_df['Value'].mean()
true_std = pop_df['Value'].std()

st.sidebar.metric("True Population Mean ($\mu$)", f"{true_mean:.2f}")
st.sidebar.metric("True Population Std ($\sigma$)", f"{true_std:.2f}")

st.sidebar.header("2. Sampling Design")
sample_method = st.sidebar.selectbox("Sampling Method", ["Simple Random Sampling", "Stratified Sampling", "Cluster Sampling", "Systematic Sampling"])
n_sample = st.sidebar.slider("Sample Size ($n$)", 10, 1000, 100)

st.sidebar.header("3. Selection Bias Injector")
bias_type = st.sidebar.selectbox("Bias Type", ["None", "Non-Response Bias", "Undercoverage Bias", "Convenience / Voluntary Bias"])
bias_strength = st.sidebar.slider("Bias Severity", 0.0, 1.0, 0.3)

# Perform Sampling
def draw_sample(df, method, n, bias, strength):
    data = df['Value'].values
    N = len(data)
    
    if bias == "Non-Response Bias":
        weights = 1.0 / (1.0 + strength * (data - data.min()) / (data.max() - data.min() + 1e-5))
        weights /= weights.sum()
        indices = np.random.choice(N, size=min(n*2, N), replace=False, p=weights)
        sample = data[indices[:n]]
    elif bias == "Undercoverage Bias":
        cutoff = int(N * (1 - 0.2 * strength))
        subset = data[data <= np.percentile(data, cutoff * 100 / N)]
        sample = np.random.choice(subset, size=min(n, len(subset)), replace=False)
    elif bias == "Convenience / Voluntary Bias":
        weights = np.exp(strength * (data - data.mean()) / data.std())
        weights /= weights.sum()
        indices = np.random.choice(N, size=n, replace=True, p=weights)
        sample = data[indices]
    else:
        indices = np.random.choice(N, size=n, replace=False)
        sample = data[indices]
    return sample

sample_data = draw_sample(pop_df, sample_method, n_sample, bias_type, bias_strength)
sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data, ddof=1)

col1, col2, col3 = st.columns(3)
col1.metric("Sample Mean ($ar{x}$)", f"{sample_mean:.2f}", delta=f"{sample_mean - true_mean:.2f}")
col2.metric("Sample Std ($s$)", f"{sample_std:.2f}", delta=f"{sample_std - true_std:.2f}")
col3.metric("Sampling Error", f"{abs(sample_mean - true_mean):.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🎲 Sampling Distribution (CLT)", "📊 Confidence Intervals", "🎓 Teaching Packs & Premium"])

with tab1:
    st.subheader("Population vs. Sample Distribution")
    fig = px.histogram(pop_df, x='Value', nbins=50, opacity=0.6, title="Population (Blue) vs Sample (Orange)")
    sample_df = pd.DataFrame({'Value': sample_data})
    fig_sample = px.histogram(sample_df, x='Value', nbins=30, opacity=0.8)
    for trace in fig_sample.data:
        trace.marker.color = 'orange'
        fig.add_trace(trace)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Central Limit Theorem Simulation")
    n_sims = st.slider("Number of Simulations ($k$)", 100, 5000, 1000, step=100)
    if st.button("Run Simulation"):
        sim_means = [np.mean(draw_sample(pop_df, sample_method, n_sample, bias_type, bias_strength)) for _ in range(n_sims)]
        sim_df = pd.DataFrame({'Sample Mean': sim_means})
        fig_clt = px.histogram(sim_df, x='Sample Mean', nbins=40, title=f"Distribution of {n_sims} Sample Means (n={n_sample})")
        fig_clt.add_vline(x=true_mean, line_dash="dash", line_color="green", annotation_text="True Mean")
        st.plotly_chart(fig_clt, use_container_width=True)
        st.success(f"Standard Error: {np.std(sim_means):.2f} | Theoretical SEM: {true_std/np.sqrt(n_sample):.2f}")

with tab3:
    st.subheader("Confidence Interval Coverage Simulation")
    if st.button("Run CI Coverage Test"):
        ci_data = []
        captured_count = 0
        for i in range(100):
            s = np.random.choice(pop_df['Value'].values, size=n_sample, replace=False)
            m = np.mean(s)
            se = stats.sem(s)
            h = se * stats.t.ppf((1 + 0.95) / 2., n_sample - 1)
            lower, upper = m - h, m + h
            captured = (lower <= true_mean) <= upper
            if captured: captured_count += 1
            ci_data.append({'Sample': i, 'Mean': m, 'Lower': lower, 'Upper': upper, 'Captured': str(captured)})
        
        ci_df = pd.DataFrame(ci_data)
        fig_ci = go.Figure()
        for _, row in ci_df.iterrows():
            color = 'blue' if row['Captured'] == 'True' else 'red'
            fig_ci.add_trace(go.Scatter(x=[row['Lower'], row['Upper']], y=[row['Sample'], row['Sample']], mode='lines+markers', line=dict(color=color, width=2), showlegend=False))
        fig_ci.add_vline(x=true_mean, line_dash="dash", line_color="green", annotation_text="True Mean")
        fig_ci.update_layout(title=f"95% Confidence Intervals ({captured_count}/100 captured true mean)", xaxis_title="Value", yaxis_title="Sample Index")
        st.plotly_chart(fig_ci, use_container_width=True)

with tab4:
    st.subheader("Statistics Teaching Packs & Premium Lessons")
    st.markdown("### 📚 Lesson 1: The 1936 *Literary Digest* Poll Failure")
    st.info("Explore selection bias through historical case studies.")
    if st.button("Download Student Worksheet"):
        st.download_button("Download", data="Worksheet Content...", file_name="worksheet.txt")
