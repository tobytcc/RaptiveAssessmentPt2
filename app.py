import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st

st.set_page_config(page_title="Negative Binomial is a better Poisson Distribution", layout="wide")

st.title("Negative Binomial is a \"better\" Poisson Distribution")
st.write(
    """
A binomial distribution models the number of successes from a discrete number of events. By extension, a negative binomial distribution models the number of attempts required to get r successes, given 
the probability of success being p in any given event, modelled as $X_{i}\sim\mathrm{NB}(r,p)$ \n
A poisson distribution models the discrete number of events happening over a fixed time period, modelled as $X_j\sim\mathrm{Poisson}(\lambda)$. \n\n

Despite these two distributions being quite different, we can actually use a negative binomial distribution to approximate Poisson.
"""
)

# Section 1: NB approximating Poisson
st.header("1) How can a Negative Binomial be related to a Poisson?")
st.write(
    """
When the number of required successes **r** is large and the success probability **p** is close
 to 1, the Negative Binomial PMF approaches a Poisson PMF. Here we tie the Poisson rate to
 the NB parameters with **λ = r · (1 - p)** and compare their shapes. \n

 On a theoretical level, as we approach the limit,the NB curve imitates a Poisson curve with a rare number of failures. If the rare failures are independent, 
 the occurances behave similarly to conditions required for a Poisson distribution. 

 *Try changing the parameters using the sliders - as you push r to a larger number and p towards 1, the difference (MAE in this case) shrinks as you approach the optimal r and p values.*
"""
)

col1, col2 = st.columns(2)
with col1:
    r_poisson = st.slider("r (no. of successes)", min_value=1, max_value=500, value=200)
with col2:
    p_poisson = st.slider("p (success probability)", min_value=0.5, max_value=0.99, value=0.9)

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
    go.Scatter(x=x_values_part1, y=nb_pmf, mode="lines+markers", name="Negative Binomial PMF")
)
fig_part1.update_layout(
    title="Poisson vs. Negative Binomial", xaxis_title="k", yaxis_title="Probability"
)

st.plotly_chart(fig_part1, use_container_width=True)
st.metric("Mean Absolute Error (Poisson vs. NB)", f"{mae:.4f}")


# Section 2: Overdispersion story
st.header("2) How is Negative Binomial a better Poisson?")
st.write(
    """
In a pure Poisson model, the mean is equal to the variance, coupling both moments together. A Negative Binomial distribution with distinct mean and variance values are no longer constrained to be equal. **This means the Negative Binomial distribution can not only imitate a Poisson distribution (as shown above), it can also be thought of as a more flexible Poisson distribution.** \n

Let's illustrate with an example: Let's say we want to model the no. of click-throughs from a banner ad across 500 websites. We can use a known average click-through rate - this means setting a $\mu$ value and also setting a variance as a result. \n

In real life though, not every site will provide the same click-through with the same banner ad. Some sites see more/less foot traffic, and some sites are more/less related to the banner ad - we might see an increased variation in click-through (compared to the mean) depending on the site.
This is where introducing a parameter for **dispersion** would help - we can tune how much variation we can expect between sites when modelling for a variable, which is where the flexibility of a NB distribution would be helpful.

 *Try changing the parameters using the sliders - changing the level of dispersion captures variance in conversion across sites not possible in a vanilla Poisson distribution.*
"""
)

sites = 500

col3, col4 = st.columns(2)
with col3:
    mu = st.slider("μ: mean conversions per site", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
with col4:
    dispersion = st.slider("k: dispersion (higher = less overdispersion)", min_value=100, max_value=10000, value=1000)

combined_mean = sites * mu
poisson_mean = combined_mean
poisson_variance = combined_mean

nb_p = dispersion / (dispersion + combined_mean)
nb_dist_over = stats.nbinom(n=dispersion, p=nb_p)
nb_mean = combined_mean
nb_variance = combined_mean + (combined_mean ** 2) / dispersion

x_values_part2 = np.arange(0, 2000 + 1)

poisson_pmf_part2 = stats.poisson(mu=poisson_mean).pmf(x_values_part2)
nb_pmf_part2 = nb_dist_over.pmf(x_values_part2)

fig_part2 = go.Figure()
fig_part2.add_trace(
    go.Scatter(x=x_values_part2, y=poisson_pmf_part2, mode="lines+markers", name="Poisson PMF")
)
fig_part2.add_trace(
    go.Scatter(x=x_values_part2, y=nb_pmf_part2, mode="lines+markers", name="Negative Binomial PMF")
)
fig_part2.update_layout(
    title="Poisson vs. Negative Binomial for Banner Ad Conversion across sites",
    xaxis_title="Conversions",
    yaxis_title="Probability",
    xaxis_range=[0, 2000],
)

st.plotly_chart(fig_part2, use_container_width=True)

rng = np.random.default_rng(7)
sample_poisson = rng.poisson(lam=poisson_mean, size=5000)
sample_nb = nb_dist_over.rvs(size=5000, random_state=rng)

col6, col7 = st.columns(2)
with col6:
    st.subheader("Poisson:")
    st.write(f"Mean: {np.mean(sample_poisson):.0f}")
    st.write(f"Variance: {np.var(sample_poisson):.0f}")
with col7:
    st.subheader("Negative Binomial:")
    st.write(f"Mean: {np.mean(sample_nb):.0f}")
    st.write(f"Variance: {np.var(sample_nb):.0f}")

st.info(
    "The Negative Binomial Distribution can not only imitate a Poisson Distribution, it can also be more flexible by introducing a new parameter to decouple mean and variance, which is useful in many real-life scenarios where variance does not equal mean."

)
