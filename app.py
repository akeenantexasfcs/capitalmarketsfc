#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def get_value_safely(df, key):
    try:
        return df.loc[key].iloc[0] if key in df.index else 0
    except Exception:
        st.warning(f"Unable to retrieve {key}. Using 0 instead.")
        return 0

def plot_sankey(ticker):
    try:
        stock = yf.Ticker(ticker)
        income_statement = stock.financials
        
        # Retrieve values safely
        total_revenue = get_value_safely(income_statement, 'Total Revenue')
        cost_of_revenue = get_value_safely(income_statement, 'Cost Of Revenue')
        gross_profit = get_value_safely(income_statement, 'Gross Profit')
        operating_expenses = get_value_safely(income_statement, 'Total Operating Expenses')
        operating_income = get_value_safely(income_statement, 'Operating Income')
        net_income = get_value_safely(income_statement, 'Net Income')
        
        # Additional breakdowns
        product_costs = cost_of_revenue * 0.9  # Estimate
        service_costs = cost_of_revenue * 0.1  # Estimate
        rnd = get_value_safely(income_statement, 'Research And Development')
        sga = get_value_safely(income_statement, 'Selling General And Administration')
        other_expenses = operating_expenses - rnd - sga
        tax = get_value_safely(income_statement, 'Tax Provision')
        
        # Prepare labels and values
        label = [
            f"Revenue ${total_revenue/1e9:.1f}B",
            f"Cost of revenue ${cost_of_revenue/1e9:.1f}B",
            f"Gross profit ${gross_profit/1e9:.1f}B",
            f"Operating expenses ${operating_expenses/1e9:.1f}B",
            f"Operating profit ${operating_income/1e9:.1f}B",
            f"Net profit ${net_income/1e9:.1f}B",
            f"Product costs ${product_costs/1e9:.1f}B",
            f"Service costs ${service_costs/1e9:.1f}B",
            f"R&D ${rnd/1e9:.1f}B",
            f"SG&A ${sga/1e9:.1f}B",
            f"Other ${other_expenses/1e9:.1f}B",
            f"Tax ${tax/1e9:.1f}B"
        ]
        
        source = [0, 0, 2, 2, 4, 1, 1, 3, 3, 3, 4]
        target = [2, 1, 4, 3, 5, 6, 7, 8, 9, 10, 11]
        values = [
            gross_profit, cost_of_revenue, operating_income, operating_expenses, net_income,
            product_costs, service_costs, rnd, sga, other_expenses, tax
        ]
        
        # Create the Sankey Diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
              pad=15,
              thickness=20,
              line=dict(color="black", width=0.5),
              label=label,
              color=["gray", "pink", "lightgreen", "pink", "green", "darkgreen", 
                     "red", "red", "red", "red", "red", "red"]
            ),
            link=dict(
              source=source,
              target=target,
              value=values,
              color=["lightgray", "pink", "lightgreen", "pink", "green", 
                     "pink", "pink", "pink", "pink", "pink", "red"]
          ))])

        # Update layout
        fig.update_layout(
            title_text=f"Financial Breakdown for {ticker}",
            font_size=10,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        # Display the diagram
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while plotting the Sankey diagram: {str(e)}")

# Streamlit app section
st.title('Financial Breakdown Sankey Diagram')

# User input for stock ticker
ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, MSFT, GOOGL):', 'AAPL')

if st.button('Generate Sankey Diagram'):
    plot_sankey(ticker)

