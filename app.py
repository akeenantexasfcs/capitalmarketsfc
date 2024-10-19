#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from io import BytesIO
import re

# Function to get futures data
def get_futures_data(ticker_symbol, start_date, end_date):
    ticker_data = yf.Ticker(ticker_symbol)
    ticker_df = ticker_data.history(period='1d', start=start_date, end=end_date)
    return ticker_df

# Altman Z-Score Calculation Functions
def ratio_x_1(ticker):
    df = ticker.balance_sheet
    working_capital = df.loc['Current Assets'].iloc[0] - df.loc['Current Liabilities'].iloc[0]
    total_assets = df.loc['Total Assets'].iloc[0]
    return working_capital / total_assets

def ratio_x_2(ticker):
    df = ticker.balance_sheet
    retained_earnings = df.loc['Retained Earnings'].iloc[0]
    total_assets = df.loc['Total Assets'].iloc[0]
    return retained_earnings / total_assets

def ratio_x_3(ticker):
    df = ticker.financials
    ebit = df.loc['EBIT'].iloc[0]
    total_assets = ticker.balance_sheet.loc['Total Assets'].iloc[0]
    return ebit / total_assets

def ratio_x_4(ticker):
    equity_market_value = ticker.info['sharesOutstanding'] * ticker.info['currentPrice']
    total_liabilities = ticker.balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
    return equity_market_value / total_liabilities

def ratio_x_5(ticker):
    df = ticker.financials
    sales = df.loc['Total Revenue'].iloc[0]
    total_assets = ticker.balance_sheet.loc['Total Assets'].iloc[0]
    return sales / total_assets

def z_score(ticker):
    try:
        x1 = ratio_x_1(ticker)
        x2 = ratio_x_2(ticker)
        x3 = ratio_x_3(ticker)
        x4 = ratio_x_4(ticker)
        x5 = ratio_x_5(ticker)
        zscore = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        return zscore, x1, x2, x3, x4, x5
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

# Custom formatter that checks for numeric values
def format_score(val):
    try:
        return '{:.2f}'.format(float(val))
    except (ValueError, TypeError):
        return val

# Styling for Altman Z Score table
def highlight_grey(val):
    return 'background-color: grey' if not pd.isna(val) else ''

def highlight_safe(val):
    return 'background-color: green' if not pd.isna(val) else ''

def highlight_distress(val):
    return 'background-color: indianred' if not pd.isna(val) else ''

# Updated make_pretty function with column existence checks
def make_pretty(styler):
    # No index
    styler.hide(axis='index')
    
    # Check which columns exist before applying formatting
    available_columns = set(styler.data.columns)

    # Conditional formatting for existing columns
    if {'Distress Zone', 'Grey Zone', 'Safe Zone'}.issubset(available_columns):
        # Column formatting
        styler.format(format_score, subset=['Distress Zone', 'Grey Zone', 'Safe Zone'])
        
        # Left text alignment for the specific columns
        styler.set_properties(subset=['Symbol', 'Distress Zone', 'Grey Zone', 'Safe Zone'], **{'text-align': 'center', 'width': '100px'})

        # Apply highlight methods to columns
        styler.applymap(highlight_grey, subset=['Grey Zone'])
        styler.applymap(highlight_safe, subset=['Safe Zone'])
        styler.applymap(highlight_distress, subset=['Distress Zone'])
    
    return styler

# Define a function to generate the Sankey diagram
def plot_sankey(income_statement):
    # Prepare labels and values from the income statement
    label = [
        f"Revenue ${income_statement['Total Revenue']}",
        f"Cost of Revenue(-${income_statement['Cost Of Revenue']})",
        f"Gross Profit ${income_statement['Gross Profit']}",
        f"Operating Expenses(-${income_statement['Total Operating Expenses']})",
        f"Operating Profit ${income_statement['Operating Income']}",
        f"Net Income ${income_statement['Net Income']}",
    ]
    
    source = [0, 2, 2, 4]
    target = [2, 3, 4, 5]
    values = [
        income_statement['Total Revenue'],
        income_statement['Cost Of Revenue'],
        income_statement['Total Operating Expenses'],
        income_statement['Operating Income']
    ]
    
    # Create the Sankey Diagram using Plotly
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30,
            thickness=20,
            line=dict(color="white", width=0),
            label=label,
            color=["#49a2eb", "#BC271B", '#519E3F', '#BC271B', '#519E3F', '#519E3F']
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color=["#96cded", "#D58A87", "#D58A87", "#A4CC9E"]
        )
    )])

    # Add title to the diagram
    fig.update_layout(
        title={
            'text': 'Income Statement Sankey Diagram',
            'y': 0.95, 'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font_size': 30
        },
        paper_bgcolor='rgb(248,248,255)',
        plot_bgcolor='rgb(248,248,255)'
    )
    
    st.plotly_chart(fig)

# Streamlit app
st.sidebar.title('Navigation')
option = st.sidebar.radio('Select a section:', ['Sankey Trial', 'Altman Z Score', 'Futures Pricing', 'Other Sections'])

if option == 'Sankey Trial':
    st.title('Income Statement Sankey Diagram Generator')

    # User selects a stock ticker
    ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, MSFT, META):', key='sankey_ticker')

    # User selects a financial period
    period = st.selectbox('Select Financial Period:', ['Quarterly', 'Yearly'], key='sankey_period')

    # When user presses button
    if st.button('Generate Sankey Diagram', key='sankey_button'):
        if ticker:
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(ticker)
            
            if period == 'Quarterly':
                income_data = stock.quarterly_financials
            else:
                income_data = stock.financials

            if not income_data.empty:
                # Convert the data to more usable format
                income_statement = {
                    'Total Revenue': income_data.loc['Total Revenue'].max(),
                    'Cost Of Revenue': income_data.loc['Cost Of Revenue'].max(),
                    'Gross Profit': income_data.loc['Gross Profit'].max(),
                    'Total Operating Expenses': income_data.loc['Total Operating Expenses'].max(),
                    'Operating Income': income_data.loc['Operating Income'].max(),
                    'Net Income': income_data.loc['Net Income'].max(),
                }
                
                # Plot Sankey Diagram
                plot_sankey(income_statement)
            else:
                st.error('Unable to retrieve financial data for the selected stock.')
        else:
            st.warning('Please enter a valid stock ticker.')

elif option == 'Altman Z Score':
    st.title('Altman Z-Score Calculator')
    
    # Define the number of input slots
    NUM_INPUTS = 4
    
    # Input fields for ticker symbols
    tickers = []
    for i in range(NUM_INPUTS):
        ticker = st.text_input(f'Ticker {i+1}', '')
        if ticker:
            tickers.append(ticker.upper())

    # Dictionary to hold scores and components for each symbol
    symbol_to_data = {}
    distress = []
    grey = []
    safe = []

    # Calculate Z-Scores
    if st.button('Calculate Z-Scores'):
        for symbol in tickers:
            ticker = yf.Ticker(symbol)
            zscore, x1, x2, x3, x4, x5 = z_score(ticker)
            symbol_to_data[symbol] = {
                'Z-Score': zscore,
                'X1': x1,
                'X2': x2,
                'X3': x3,
                'X4': x4,
                'X5': x5
            }

            # Classify Z-Scores for the styled table
            if zscore <= 1.8:
                distress.append(zscore)
                grey.append(None)
                safe.append(None)
            elif 1.8 < zscore <= 2.99:
                distress.append(None)
                grey.append(zscore)
                safe.append(None)
            else:
                distress.append(None)
                grey.append(None)
                safe.append(zscore)

        # Table 1: X1, X2, X3, X4, X5 (Raw Z-Score Data)
        df1 = pd.DataFrame.from_dict(symbol_to_data, orient='index')
        df1.index.name = 'Symbol'
        df1 = df1.reset_index()

        # Display Table 1 with custom formatter
        st.write("Raw Z-Score Data:")
        st.dataframe(df1.style.format(format_score))

        # Table 2: Styled Distress, Grey, Safe Zone table
        data_dict = {'Symbol': tickers, 'Distress Zone': distress, 'Grey Zone': grey, 'Safe Zone': safe}
        df2 = pd.DataFrame.from_dict(data_dict)

        # Apply styles to DataFrame
        styles = [
            dict(selector='td', props=[('font-size', '10pt'), ('border-style', 'solid'), ('border-width', '1px')]),
            dict(selector='th.col_heading', props=[('font-size', '11pt'), ('text-align', 'center')]),
            dict(selector='caption', props=[('text-align', 'center'), ('font-size', '14pt'), ('font-weight', 'bold')])
        ]

        df_styled = df2.style.pipe(make_pretty).set_caption('Altman Z Score').set_table_styles(styles)

        # Display Table 2
        st.write("Styled Z-Score Classification:")
        st.dataframe(df_styled)

elif option == 'Futures Pricing':
    st.title('Futures Pricing')
    ticker = st.text_input('Enter the Futures Ticker Symbol (e.g., CL=F):', 'CL=F')
    start_date = st.date_input('Start Date', value=pd.to_datetime('2014-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('2024-04-08'))
    
    if st.button('Get Data'):
        data = get_futures_data(ticker, start_date, end_date)
        st.write(data)
        
        # Export to CSV
        csv_file_name = f'historical_prices_{ticker}.csv'
        data.to_csv(csv_file_name)
        st.write(f'Data exported to {csv_file_name}')
        st.download_button(
            label="Download CSV",
            data=data.to_csv().encode('utf-8'),
            file_name=csv_file_name,
            mime='text/csv',
        )

