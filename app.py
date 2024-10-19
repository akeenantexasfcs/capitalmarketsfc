#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

def get_value_safely(df, key):
    try:
        return df.loc[key].iloc[0] if key in df.index else 0
    except Exception:
        st.warning(f"Unable to retrieve {key}. Using 0 instead.")
        return 0

def plot_sankey(ticker, report_type, year):
    try:
        stock = yf.Ticker(ticker)
        
        if report_type == 'Annual':
            financials = stock.financials
        else:  # Quarterly
            financials = stock.quarterly_financials
        
        # Filter for the selected year
        financials = financials.loc[:, financials.columns.year == year]
        
        if financials.empty:
            st.error(f"No financial data available for {ticker} in {year}")
            return

        # Retrieve values safely
        total_revenue = get_value_safely(financials, 'Total Revenue')
        cost_of_revenue = get_value_safely(financials, 'Cost Of Revenue')
        gross_profit = get_value_safely(financials, 'Gross Profit')
        operating_expense = get_value_safely(financials, 'Operating Expense')
        operating_income = get_value_safely(financials, 'Operating Income')
        net_income = get_value_safely(financials, 'Net Income')
        rnd = get_value_safely(financials, 'Research And Development')
        sga = get_value_safely(financials, 'Selling General And Administration')
        
        # Calculate other values
        other_expenses = operating_expense - rnd - sga
        tax = get_value_safely(financials, 'Tax Provision')
        
        # Prepare labels and values
        labels = [
            f"Revenue\n${total_revenue/1e9:.1f}B",
            f"Cost of Revenue\n${cost_of_revenue/1e9:.1f}B",
            f"Gross Profit\n${gross_profit/1e9:.1f}B",
            f"Operating Expense\n${operating_expense/1e9:.1f}B",
            f"Operating Income\n${operating_income/1e9:.1f}B",
            f"Net Income\n${net_income/1e9:.1f}B",
            f"R&D\n${rnd/1e9:.1f}B",
            f"SG&A\n${sga/1e9:.1f}B",
            f"Other Expenses\n${other_expenses/1e9:.1f}B",
            f"Tax\n${tax/1e9:.1f}B"
        ]
        
        source = [0, 0, 2, 2, 4, 3, 3, 3, 4]
        target = [2, 1, 4, 3, 5, 6, 7, 8, 9]
        values = [
            gross_profit, cost_of_revenue, operating_income, operating_expense, 
            net_income, rnd, sga, other_expenses, tax
        ]
        
        # Create the Sankey Diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
              pad=15,
              thickness=20,
              line=dict(color="black", width=0.5),
              label=labels,
              color=["#87CEEB", "#FFB6C1", "#98FB98", "#FFB6C1", "#32CD32", "#006400", 
                     "#FF69B4", "#FF69B4", "#FF69B4", "#FF0000"]
            ),
            link=dict(
              source=source,
              target=target,
              value=values,
              color=["#98FB98", "#FFB6C1", "#32CD32", "#FFB6C1", "#006400", 
                     "#FF69B4", "#FF69B4", "#FF69B4", "#FF0000"]
          ))])

        # Update layout for better readability
        fig.update_layout(
            title_text=f"Financial Breakdown for {ticker} ({report_type} {year})",
            font_size=12,
            paper_bgcolor='white',
            plot_bgcolor='white',
            width=1000,
            height=600
        )

        # Display the diagram
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while plotting the Sankey diagram: {str(e)}")

# Streamlit app section
st.title('Financial Breakdown Sankey Diagram')

# User input for stock ticker
ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, MSFT, GOOGL):', 'AAPL')

# Select report type
report_type = st.selectbox('Select Report Type', ['Annual', 'Quarterly'])

# Select year
current_year = pd.Timestamp.now().year
year = st.selectbox('Select Year', range(current_year, current_year-5, -1))

if st.button('Generate Sankey Diagram'):
    plot_sankey(ticker, report_type, year)

