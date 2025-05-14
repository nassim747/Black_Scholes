# Black-Scholes Option Pricing simulator
An interactive web-based application for pricing European options using both the analytical Black-Scholes formula and Monte Carlo simulation. This tool allows users to visualize and compare theoretical and numerical results, explore the sensitivity of option prices to key parameters, and apply variance reduction techniques in a user-friendly environment.

# Overview
This project provides a streamlined and educational interface for understanding option pricing theory in practice. It supports two core pricing methods:

    Black-Scholes Analytical Solution: A closed-form model for pricing European call and put options.

    Monte Carlo Simulation: A stochastic modeling approach that simulates asset paths under geometric Brownian motion to estimate option prices numerically.

The application is built with Python and powered by Streamlit, making it accessible directly in the browser with no advanced setup required.


Key Features
  Pricing Models

    Analytical Black-Scholes Formula for European call and put options.

    Monte Carlo Simulation with support for:

        Antithetic Variates (to reduce variance through symmetry)

        Control Variates (leveraging the analytical solution to reduce bias)

Customizable Parameters

    Spot price of the underlying asset

    Strike price

    Time to maturity

    Volatility (σ)

    Risk-free interest rate (r)

    Number of simulations and time steps

    Optional toggles for variance reduction techniques

Visual Output

    Option Price Heatmaps: Visualize how call and put prices vary with spot price and volatility.

    Simulation Path Plots: Display multiple simulated asset paths to illustrate underlying stochastic behavior.

    Comparison Charts: Bar plots comparing analytical and Monte Carlo pricing results, with standard error bars.

To run the application locally:

    git clone https://github.com/nassim747/Black_Scholes.git

    cd Black_Scholes

    pip install -r requirements.txt

    streamlit run streamlit_app.py
