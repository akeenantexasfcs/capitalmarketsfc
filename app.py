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

# Loan Pricing Calculator Placeholder
def create_loan_calculator():
    st.title("Loan Pricing Calculator")
    st.write("Loan calculator functionality goes here.")

# Streamlit App
st.sidebar.title('Navigation')
option = st.sidebar.radio('Select a section:', ['Altman Z Score', 'Futures Pricing', 'JSON Conversion', 'Loan Pricing Calculator'])

if option == 'Altman Z Score':
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

        # Export only the Raw Z-Score Data table (df1) to Excel
        if st.button('Audit'):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df1.to_excel(writer, sheet_name='Z-Scores', index=False)
                workbook = writer.book

                # Add formatting for Z-Scores
                zscore_worksheet = writer.sheets['Z-Scores']
                cell_format = workbook.add_format({'border': 1})
                zscore_worksheet.set_column('A:G', 15, cell_format)

            output.seek(0)
            st.download_button(
                label="Download Z-Score Audit Excel file",
                data=output.getvalue(),
                file_name="z_score_audit.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

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

