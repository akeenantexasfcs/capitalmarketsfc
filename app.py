#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import yfinance as yf
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
        ratio_1 = ratio_x_1(ticker)
        ratio_2 = ratio_x_2(ticker)
        ratio_3 = ratio_x_3(ticker)
        ratio_4 = ratio_x_4(ticker)
        ratio_5 = ratio_x_5(ticker)
        zscore = 1.2 * ratio_1 + 1.4 * ratio_2 + 3.3 * ratio_3 + 0.6 * ratio_4 + 1.0 * ratio_5
        return zscore, ratio_1, ratio_2, ratio_3, ratio_4, ratio_5
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

# Loan Pricing Calculator (Add error handling and state management)
def create_loan_calculator():
    st.title("Loan Pricing Calculator")
    
    # Initialize default values for the session state
    if 'loans' not in st.session_state:
        st.session_state.loans = [{
            'Loan Type': "Insert Loan Type",
            'PD/LGD': "Insert PD/LGD",
            'Company Name': "Insert Company Name",
            'Eligibility': "Directly Eligible",
            'Patronage': "Non-Patronage",
            'Revolver': "No",
            'Direct Note Patronage (%)': 0.40,
            'Fee in lieu (%)': 0.00,
            'SPREAD (%)': 0.00,
            'CSA (%)': 0.00,
            'SOFR (%)': 0.00,
            'COFs (%)': 0.00,
            'Upfront Fee (%)': 0.00,
            'Servicing Fee (%)': 0.15,
            'Years to Maturity': 5.0,
            'Unused Fee (%)': 0.00
        }]
        st.session_state.current_loan_count = 1

    # Show inputs for each loan
    for i in range(st.session_state.current_loan_count):
        with st.expander(f"Loan {i + 1} Details", expanded=True):
            loan_data = st.session_state.loans[i]
            # Loan Type Input
            loan_data['Loan Type'] = st.text_input(f"Loan Type {i + 1}", value=loan_data['Loan Type'], key=f'Loan Type {i}')
            loan_data['PD/LGD'] = st.text_input(f"PD/LGD {i + 1}", value=loan_data['PD/LGD'], key=f'PD/LGD {i}')
            loan_data['Company Name'] = st.text_input(f"Company Name {i + 1}", value=loan_data['Company Name'], key=f'Company Name {i}')
            
            # Eligibility Radio
            eligibility_options = ["Directly Eligible", "Similar Entity"]
            loan_data['Eligibility'] = st.radio(f"Eligibility {i + 1}", options=eligibility_options, index=eligibility_options.index(loan_data['Eligibility']), key=f'Eligibility {i}')

            # Patronage Radio Button
            patronage_options = ["Patronage", "Non-Patronage"]
            loan_data['Patronage'] = st.radio(f"Patronage {i + 1}", options=patronage_options, index=patronage_options.index(loan_data['Patronage']), key=f'Patronage {i}')

            # Revolver Radio Button
            revolver_options = ["Yes", "No"]
            loan_data['Revolver'] = st.radio(f"Revolver {i + 1}", options=revolver_options, index=revolver_options.index(loan_data['Revolver']), key=f'Revolver {i}')
            
            # Direct Note Patronage Input
            loan_data['Direct Note Patronage (%)'] = st.number_input(f"Direct Note Patronage (%) {i + 1}", value=loan_data['Direct Note Patronage (%)'], step=0.01, format="%.2f", key=f'Direct Note Patronage {i}')

            # Upfront Fee, Servicing Fee, and Years to Maturity Inputs
            loan_data['Upfront Fee (%)'] = st.number_input(f"Upfront Fee (%) {i + 1}", value=loan_data['Upfront Fee (%)'], step=0.01, format="%.2f", key=f'Upfront Fee {i}')
            loan_data['Servicing Fee (%)'] = st.number_input(f"Servicing Fee (%) {i + 1}", value=loan_data['Servicing Fee (%)'], step=0.01, format="%.2f", key=f'Servicing Fee {i}')
            loan_data['Years to Maturity'] = st.slider(f"Years to Maturity {i + 1}", 0.0, 30.0, value=loan_data['Years to Maturity'], step=0.5, key=f'Years to Maturity {i}')

    # Add a new loan button if less than 4 loans
    if st.session_state.current_loan_count < 4:
        if st.button("Add Another Loan"):
            st.session_state.current_loan_count += 1

# Streamlit App
st.sidebar.title('Navigation')
option = st.sidebar.radio('Select a section:', ['Altman Z Score', 'Futures Pricing', 'JSON Conversion', 'Loan Pricing Calculator'])

if option == 'Altman Z Score':
    st.title('Altman Z-Score Calculator')
    
    # Define the number of input slots
    NUM_INPUTS = 10
    
    # Input fields for ticker symbols
    tickers = []
    for i in range(NUM_INPUTS):
        ticker = st.text_input(f'Ticker {i+1}', '')
        if ticker:
            tickers.append(ticker.upper())
    
    # Dictionary to hold scores and ratios for each symbol
    symbol_to_data = {}

    # Calculate Z-Scores
    if st.button('Calculate Z-Scores'):
        for symbol in tickers:
            ticker = yf.Ticker(symbol)
            zscore, r1, r2, r3, r4, r5 = z_score(ticker)
            symbol_to_data[symbol] = {
                'Z-Score': zscore,
                'Ratio X1': r1,
                'Ratio X2': r2,
                'Ratio X3': r3,
                'Ratio X4': r4,
                'Ratio X5': r5
            }

        # Create DataFrame
        data_dict = {'Symbol': tickers, 'Z-Score': [symbol_to_data[symbol]['Z-Score'] for symbol in tickers]}
        df = pd.DataFrame(data_dict)

        st.write(df)

        # Add "Audit" button for CSV export
        if st.button('Audit'):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for symbol, data in symbol_to_data.items():
                    df_audit = pd.DataFrame({
                        'Ticker': [symbol],
                        'X1': [data['Ratio X1']],
                        'X2': [data['Ratio X2']],
                        'X3': [data['Ratio X3']],
                        'X4': [data['Ratio X4']],
                        'X5': [data['Ratio X5']],
                        'Total Z-Score': [data['Z-Score']]
                    })
                    df_audit.to_excel(writer, sheet_name=symbol)
                writer.save()
                output.seek(0)
            st.download_button(label="Download Audit Excel file", data=output.getvalue(), file_name="z_score_audit.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif option == 'Loan Pricing Calculator':
    create_loan_calculator()

