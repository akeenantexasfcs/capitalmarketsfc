#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

def get_value_safely(df, key):
    try:
        return df.loc[key].iloc[0] if key in df.index else 0
    except Exception as e:
        st.warning(f"Unable to retrieve {key}. Using 0 instead. Error: {str(e)}")
        return 0

def format_value(value, unit):
    if unit == 'Billions':
        return f"${value/1e9:,.1f}B"
    elif unit == 'Millions':
        return f"${value/1e6:,.1f}M"
    else:  # Thousands
        return f"${value/1e3:,.1f}K"

def plot_sankey(ticker, report_type, year, quarter, unit):
    try:
        st.write(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        
        if report_type == 'Annual':
            financials = stock.financials
            financials = financials.loc[:, financials.columns.year == year]
        else:  # Quarterly
            financials = stock.quarterly_financials
            financials = financials.loc[:, (financials.columns.year == year) & (financials.columns.quarter == quarter)]
        
        if financials.empty:
            st.error(f"No financial data available for {ticker} in the selected period")
            return

        st.write("Processing financial data...")
        # Retrieve values safely
        total_revenue = get_value_safely(financials, 'Total Revenue')
        cost_of_revenue = get_value_safely(financials, 'Cost Of Revenue')
        gross_profit = total_revenue - cost_of_revenue
        operating_expense = get_value_safely(financials, 'Operating Expense')
        operating_income = gross_profit - operating_expense
        net_income = get_value_safely(financials, 'Net Income')
        rnd = get_value_safely(financials, 'Research And Development')
        sga = get_value_safely(financials, 'Selling General And Administration')
        
        # Calculate other values
        other_expenses = operating_expense - rnd - sga
        non_operating = net_income - operating_income

        st.write("Preparing Sankey diagram data...")
        # Prepare labels and values
        labels = [
            f"Revenue<br>{format_value(total_revenue, unit)}",
            f"Cost of Revenue<br>{format_value(cost_of_revenue, unit)}",
            f"Gross Profit<br>{format_value(gross_profit, unit)}",
            f"Operating Expense<br>{format_value(operating_expense, unit)}",
            f"Operating Income<br>{format_value(operating_income, unit)}",
            f"Net Income<br>{format_value(net_income, unit)}",
            f"R&D<br>{format_value(rnd, unit)}",
            f"SG&A<br>{format_value(sga, unit)}",
            f"Other Expenses<br>{format_value(other_expenses, unit)}",
            f"Non-Operating<br>{format_value(non_operating, unit)}"
        ]
        
        source = [0, 0, 2, 2, 4, 3, 3, 3, 4]
        target = [2, 1, 4, 3, 5, 6, 7, 8, 9]
        values = [
            gross_profit, cost_of_revenue, operating_income, operating_expense, 
            net_income, rnd, sga, other_expenses, non_operating
        ]
        
        # Ensure all values are positive for Sankey diagram
        values = [abs(v) for v in values]
        
        st.write("Creating Sankey diagram...")
        # Create the Sankey Diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
              pad=20,
              thickness=30,
              line=dict(color="black", width=1),
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
        title = f"Financial Breakdown for {ticker} ({report_type} {year}"
        title += f" Q{quarter}" if report_type == 'Quarterly' else ")"
        fig.update_layout(
            title_text=title,
            font=dict(size=16, color="black"),
            paper_bgcolor='white',
            plot_bgcolor='white',
            width=1200,
            height=800
        )

        # Update node properties for better visibility
        fig.update_traces(
            node=dict(
                pad=20,
                thickness=30,
                line=dict(color="black", width=1),
            ),
            selector=dict(type='sankey')
        )

        st.write("Displaying Sankey diagram...")
        # Display the diagram
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while plotting the Sankey diagram: {str(e)}")
        st.exception(e)

# Streamlit app section
st.title('Financial Breakdown Sankey Diagram')

# User input for stock ticker
ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, MSFT, GOOGL):', 'AAPL')

# Select report type
report_type = st.selectbox('Select Report Type', ['Annual', 'Quarterly'])

# Get current year and quarter
current_year = datetime.now().year
current_quarter = (datetime.now().month - 1) // 3 + 1

# Select year
year = st.selectbox('Select Year', range(current_year, current_year-5, -1))

quarter = None
if report_type == 'Quarterly':
    # Select quarter for quarterly reports
    quarter = st.selectbox('Select Quarter', [1, 2, 3, 4], index=current_quarter-1)

# Select unit for financial figures
unit = st.selectbox('Select Unit', ['Billions', 'Millions', 'Thousands'])

if st.button('Generate Sankey Diagram'):
    st.write("Button clicked! Generating Sankey Diagram...")
    try:
        plot_sankey(ticker, report_type, year, quarter, unit)
        st.success("Sankey diagram generated successfully!")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

st.write("App is running. If you can see this, Streamlit is working correctly.")

