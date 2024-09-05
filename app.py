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

# JSON Conversion Functionality
def json_conversion():
    st.title("JSON Conversion")
    uploaded_file = st.file_uploader("Choose a JSON file", type="json", key='json_uploader')
    if uploaded_file is not None:
        try:
            file_contents = uploaded_file.read().decode('utf-8')
            data = json.loads(file_contents)
            st.text(f"File size: {len(file_contents)} bytes")
            tables = []
            for block in data['Blocks']:
                if block['BlockType'] == 'TABLE':
                    table = {}
                    if 'Relationships' in block:
                        for relationship in block['Relationships']:
                            if relationship['Type'] == 'CHILD':
                                for cell_id in relationship['Ids']:
                                    cell_block = next((b for b in data['Blocks'] if b['Id'] == cell_id), None)
                                    if cell_block:
                                        row_index = cell_block.get('RowIndex', 0)
                                        col_index = cell_block.get('ColumnIndex', 0)
                                        if row_index not in table:
                                            table[row_index] = {}
                                        cell_text = ''
                                        if 'Relationships' in cell_block:
                                            for rel in cell_block['Relationships']:
                                                if rel['Type'] == 'CHILD':
                                                    for word_id in rel['Ids']:
                                                        word_block = next((w for w in data['Blocks'] if w['Id'] == word_id), None)
                                                        if word_block and word_block['BlockType'] == 'WORD':
                                                            cell_text += ' ' + word_block.get('Text', '')
                                        table[row_index][col_index] = cell_text.strip()
                    table_df = pd.DataFrame.from_dict(table, orient='index').sort_index()
                    table_df = table_df.sort_index(axis=1)
                    tables.append(table_df)
            all_tables = pd.concat(tables, axis=0, ignore_index=True)
            if len(all_tables.columns) == 0:
                st.error("No columns found in the uploaded JSON file.")
                return
            st.subheader("Data Preview")
            st.dataframe(all_tables)
            numerical_columns = []
            for col in all_tables.columns:
                if st.checkbox(f"Numerical column '{col}'", value=False, key=f"num_{col}"):
                    numerical_columns.append(col)
            if st.button("Convert and Download Excel", key="convert_download"):
                for col in numerical_columns:
                    if col in all_tables.columns:
                        all_tables[col] = all_tables[col].apply(lambda x: float(re.sub(r'[$,()]', '', x.strip().replace(')', '').replace('(', '-')) if x.strip() else 0))
                excel_data = to_excel(all_tables)
                st.download_button(label='📥 Download Excel file', data=excel_data, file_name='converted_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except json.JSONDecodeError:
            st.error("The uploaded file is not a valid JSON.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

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

        # Categorize
        distress = [''] * len(tickers)
        grey = [''] * len(tickers)
        safe = [''] * len(tickers)

        for idx, symbol in enumerate(tickers):
            zscore = symbol_to_data[symbol]['Z-Score']
            if pd.notna(zscore):
                if zscore <= 1.8:
                    distress[idx] = zscore
                elif zscore > 1.8 and zscore <= 2.99:
                    grey[idx] = zscore
                else:
                    safe[idx] = zscore

        # Create DataFrame
        data_dict = {'Symbol': tickers, 'Distress Zone': distress, 'Grey Zone': grey, 'Safe Zone': safe}
        df = pd.DataFrame.from_dict(data_dict)

        # Apply styles
        def highlight_distress(val):
            return 'background-color: indianred' if val != '' else ""

        def highlight_grey(val):
            return 'background-color: grey' if val != '' else ""

        def highlight_safe(val):
            return 'background-color: green' if val != '' else ""

        def format_score(val):
            try:
                return '{:.2f}'.format(float(val))
            except:
                return ''

        def make_pretty(styler):
            # No index
            styler.hide(axis='index')
            
            # Column formatting
            styler.format(format_score, subset=['Distress Zone', 'Grey Zone', 'Safe Zone'])

            # Left text alignment for some columns
            styler.set_properties(subset=['Symbol', 'Distress Zone', 'Grey Zone', 'Safe Zone'], **{'text-align': 'center', 'width': '100px'})

            # Apply highlight methods to columns
            styler.applymap(highlight_grey, subset=['Grey Zone'])
            styler.applymap(highlight_safe, subset=['Safe Zone'])
            styler.applymap(highlight_distress, subset=['Distress Zone'])
            return styler

        # Apply styles to DataFrame
        styles = [
            dict(selector='td', props=[('font-size', '10pt'), ('border-style', 'solid'), ('border-width', '1px')]),
            dict(selector='th.col_heading', props=[('font-size', '11pt'), ('text-align', 'center')]),
            dict(selector='caption', props=[('text-align', 'center'), ('font-size', '14pt'), ('font-weight', 'bold')])
        ]

        df_styled = df.style.pipe(make_pretty).set_caption('Altman Z Score').set_table_styles(styles)
        st.dataframe(df_styled)

        # Add "Audit" button for CSV export
        if st.button('Audit'):
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
                
                # Convert DataFrame to CSV format
                csv = df_audit.to_csv(index=False)
                
                # Create a download button for each ticker
                st.download_button(
                    label=f"Download {symbol} Audit CSV",
                    data=csv,
                    file_name=f"{symbol}_zscore_audit.csv",
                    mime='text/csv',
                )

    # Add footer
    st.markdown("---")
    st.markdown("Credit for this implementation goes to Sugath Mudali. Very slight changes were made from the original Medium blog post.")

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

elif option == 'JSON Conversion':
    json_conversion()

elif option == 'Loan Pricing Calculator':
    create_loan_calculator()

