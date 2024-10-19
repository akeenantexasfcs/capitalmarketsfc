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
    except Exception:
        st.warning(f"Unable to retrieve {key}. Using 0 instead.")
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
                pad=15,
                thickness=25,
                line=dict(color="black", width=1),
            ),
            selector=dict(type='sankey')
        )

        # Display the diagram
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while plotting the Sankey diagram: {str(e)}")

# The rest of the Streamlit app code remains the same

