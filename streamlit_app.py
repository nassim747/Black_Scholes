import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp  # Make sure to import these
import matplotlib.pyplot as plt
import seaborn as sns

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")


# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)

# (Include the BlackScholes class definition here)

class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(
        self,
    ):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        d1 = (
            log(current_price / strike) +
            (interest_rate + 0.5 * volatility ** 2) * time_to_maturity
            ) / (
                volatility * sqrt(time_to_maturity)
            )
        d2 = d1 - volatility * sqrt(time_to_maturity)

        call_price = current_price * norm.cdf(d1) - (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2)
        )
        put_price = (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)
        ) - current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        # GREEKS
        # Delta
        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)

        # Gamma
        self.call_gamma = norm.pdf(d1) / (
            strike * volatility * sqrt(time_to_maturity)
        )
        self.put_gamma = self.call_gamma

        return call_price, put_price

# Add MonteCarloSimulation class after BlackScholes class
class MonteCarloSimulation:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
        n_simulations: int = 10000,
        n_steps: int = 252,  # Daily steps for 1 year
        use_antithetic: bool = True
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.use_antithetic = use_antithetic
        self.dt = time_to_maturity / n_steps

    def generate_paths(self):
        # Generate random numbers for the simulation
        if self.use_antithetic:
            # Generate half the number of random numbers and use antithetic variates
            n_sims = self.n_simulations // 2
            z = np.random.normal(0, 1, (n_sims, self.n_steps))
            z = np.vstack((z, -z))  # Stack with antithetic variates
        else:
            z = np.random.normal(0, 1, (self.n_simulations, self.n_steps))

        # Calculate drift and diffusion terms
        drift = (self.interest_rate - 0.5 * self.volatility ** 2) * self.dt
        diffusion = self.volatility * np.sqrt(self.dt)

        # Generate price paths
        price_paths = np.zeros((self.n_simulations, self.n_steps + 1))
        price_paths[:, 0] = self.current_price

        for t in range(1, self.n_steps + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(drift + diffusion * z[:, t-1])

        return price_paths

    def calculate_prices(self):
        # Generate price paths
        price_paths = self.generate_paths()
        final_prices = price_paths[:, -1]

        # Calculate payoffs
        call_payoffs = np.maximum(final_prices - self.strike, 0)
        put_payoffs = np.maximum(self.strike - final_prices, 0)

        # Discount payoffs
        discount_factor = np.exp(-self.interest_rate * self.time_to_maturity)
        call_price = np.mean(call_payoffs) * discount_factor
        put_price = np.mean(put_payoffs) * discount_factor

        # Calculate standard errors
        call_std_err = np.std(call_payoffs) * discount_factor / np.sqrt(self.n_simulations)
        put_std_err = np.std(put_payoffs) * discount_factor / np.sqrt(self.n_simulations)

        return call_price, put_price, call_std_err, put_std_err

# Function to generate heatmaps
# ... your existing imports and BlackScholes class definition ...


# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/nassim-a-265944286/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Ameur, Nassim`</a>', unsafe_allow_html=True)

    current_price = st.number_input("Current Asset Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

    st.markdown("---")
    st.subheader("Monte Carlo Parameters")
    n_simulations = st.number_input("Number of Simulations", min_value=1000, value=10000, step=1000)
    use_antithetic = st.checkbox("Use Antithetic Variates", value=True)

    st.markdown("---")
    calculate_btn = st.button('Heatmap Parameters')
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)



def plot_heatmap(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_prices()
            call_prices[i, j] = bs_temp.call_price
            put_prices[i, j] = bs_temp.put_price
    
    # Plotting Call Price Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_call)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    # Plotting Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_put)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put


# Main Page for Output Display
st.title("Black-Scholes Pricing Model")

# Table of Inputs
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Calculate Call and Put values
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.title("Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1,1], gap="small")

with col1:
    st.subheader("Call Price Heatmap")
    heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, strike)
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put Price Heatmap")
    _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, strike)
    st.pyplot(heatmap_fig_put)

# After the existing Black-Scholes calculation, add Monte Carlo results
st.markdown("---")
st.title("Monte Carlo vs. Analytical Comparison")

# Calculate both analytical and Monte Carlo prices
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
analytical_call, analytical_put = bs_model.calculate_prices()

mc_model = MonteCarloSimulation(
    time_to_maturity=time_to_maturity,
    strike=strike,
    current_price=current_price,
    volatility=volatility,
    interest_rate=interest_rate,
    n_simulations=n_simulations,
    use_antithetic=use_antithetic
)
mc_call, mc_put, mc_call_std_err, mc_put_std_err = mc_model.calculate_prices()

# Create comparison table
comparison_data = {
    "Method": ["Analytical (Black-Scholes)", "Monte Carlo"],
    "Call Price": [analytical_call, mc_call],
    "Put Price": [analytical_put, mc_put],
    "Call Std Error": ["N/A", f"±{mc_call_std_err:.4f}"],
    "Put Std Error": ["N/A", f"±{mc_put_std_err:.4f}"]
}
comparison_df = pd.DataFrame(comparison_data)
st.table(comparison_df)

# Add visualization of price paths
st.subheader("Sample Price Paths")
price_paths = mc_model.generate_paths()
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(min(100, len(price_paths))):  # Plot first 100 paths
    ax.plot(np.linspace(0, time_to_maturity, mc_model.n_steps + 1), price_paths[i], alpha=0.1)
ax.set_title("Sample Price Paths")
ax.set_xlabel("Time (Years)")
ax.set_ylabel("Asset Price")
st.pyplot(fig)