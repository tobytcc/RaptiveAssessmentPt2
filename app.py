import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st

st.set_page_config(page_title="Negative Binomial vs. Poisson", layout="wide")

st.title("Exploring Poisson and Negative Binomial Approximations")
st.write(
    """
This app shows how a Negative Binomial (NB) distribution can both approximate a Poisson
curve and offer extra flexibility when modeling ad conversion with overdispersion.
Use the sliders in each section to see how parameter choices shape the distributions.
"""
)

# Section 1: NB approximating Poisson
st.header("1) Pushing the Negative Binomial toward Poisson")
st.write(
    """
When the number of required successes **r** is large and the success probability **p** is close
 to 1, the Negative Binomial PMF approaches a Poisson PMF. Here we tie the Poisson rate to
 the NB parameters with **λ = r · (1 - p)** and compare their shapes.
"""
)

col1, col2 = st.columns(2)
with col1:
    r_poisson = st.slider("r (number of successes)", min_value=1, max_value=200, value=80)
with col2:
    p_poisson = st.slider("p (success probability)", min_value=0.50, max_value=0.99, value=0.9)

lambda_poisson = r_poisson * (1 - p_poisson)
poisson_dist = stats.poisson(mu=lambda_poisson)
nb_dist = stats.nbinom(n=r_poisson, p=p_poisson)

x_max_part1 = int(max(poisson_dist.ppf(0.999), nb_dist.ppf(0.999)))
x_values_part1 = np.arange(0, max(x_max_part1, 1) + 1)

poisson_pmf = poisson_dist.pmf(x_values_part1)
nb_pmf = nb_dist.pmf(x_values_part1)
mae = float(np.mean(np.abs(poisson_pmf - nb_pmf)))

fig_part1 = go.Figure()
fig_part1.add_trace(
    go.Scatter(x=x_values_part1, y=poisson_pmf, mode="lines+markers", name="Poisson PMF")
)
fig_part1.add_trace(
    go.Scatter(x=x_values_part1, y=nb_pmf, mode="lines+markers", name="NB PMF")
)
fig_part1.update_layout(
    title="Poisson vs. Negative Binomial", xaxis_title="k", yaxis_title="Probability"
)

st.plotly_chart(fig_part1, use_container_width=True)
st.metric("Mean Absolute Error (Poisson vs. NB)", f"{mae:.4f}")

# Section 2: Overdispersion story
st.header("2) Modeling overdispersion for ad conversion")
st.write(
    """
A Gamma-Poisson (Negative Binomial) model separates the mean and variance, letting us account
for traffic or ad-fit variability across sites. Use the sliders to compare a simple Poisson
model against a Poisson-Gamma mixture with dispersion **k**.
"""
)

col3, col4, col5 = st.columns(3)
with col3:
    sites = st.slider("S: number of sites", min_value=1, max_value=250, value=50)
with col4:
    mu = st.slider("μ: mean conversions per site", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
with col5:
    dispersion = st.slider("k: dispersion (higher = less overdispersion)", min_value=0.1, max_value=80.0, value=10.0)

combined_mean = sites * mu
poisson_mean = combined_mean
poisson_variance = combined_mean

nb_p = dispersion / (dispersion + combined_mean)
nb_dist_over = stats.nbinom(n=dispersion, p=nb_p)
nb_mean = combined_mean
nb_variance = combined_mean + (combined_mean ** 2) / dispersion

x_max_part2 = int(max(stats.poisson(mu=poisson_mean).ppf(0.999), nb_dist_over.ppf(0.999)))
x_values_part2 = np.arange(0, max(x_max_part2, 1) + 1)

poisson_pmf_part2 = stats.poisson(mu=poisson_mean).pmf(x_values_part2)
nb_pmf_part2 = nb_dist_over.pmf(x_values_part2)

fig_part2 = go.Figure()
fig_part2.add_trace(
    go.Scatter(x=x_values_part2, y=poisson_pmf_part2, mode="lines+markers", name="Poisson PMF")
)
fig_part2.add_trace(
    go.Scatter(x=x_values_part2, y=nb_pmf_part2, mode="lines+markers", name="Gamma-Poisson (NB) PMF")
)
fig_part2.update_layout(
    title="Poisson vs. Gamma-Poisson for Ad Conversions", xaxis_title="Conversions", yaxis_title="Probability"
)

st.plotly_chart(fig_part2, use_container_width=True)

rng = np.random.default_rng(7)
sample_poisson = rng.poisson(lam=poisson_mean, size=5000)
sample_nb = nb_dist_over.rvs(size=5000, random_state=rng)

col6, col7 = st.columns(2)
with col6:
    st.subheader("Sample stats: Poisson")
    st.write(f"Mean: {np.mean(sample_poisson):.2f}")
    st.write(f"Variance: {np.var(sample_poisson):.2f}")
with col7:
    st.subheader("Sample stats: Gamma-Poisson")
    st.write(f"Mean: {np.mean(sample_nb):.2f}")
    st.write(f"Variance: {np.var(sample_nb):.2f}")

st.info(
    "With dispersion **k**, the Gamma-Poisson variance is μ + μ²/k. Smaller k boosts variance"
    " without changing the mean, making the curve fatter where sites have uneven traffic or"
    " ad relevance."
)
