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
        return zscore
    except Exception as e:
        return np.nan

# JSON Conversion Functionality
def json_conversion():
    st.title("JSON Conversion")
    uploaded_file = st.file_uploader("Choose a JSON file", type="json", key='json_uploader')
    if uploaded_file is not None:
        try:
            # Read the uploaded file as a string
            file_contents = uploaded_file.read().decode('utf-8')
            # Load the JSON data
            data = json.loads(file_contents)
            # Display file size for debugging
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

            st.subheader("Select numerical columns")
            numerical_columns = []
            for col in all_tables.columns:
                if st.checkbox(f"Numerical column '{col}'", value=False, key=f"num_{col}"):
                    numerical_columns.append(col)

            def clean_numeric_value(value):
                value_str = str(value).strip()
                if value_str.startswith('(') and value_str.endswith(')'):
                    value_str = '-' + value_str[1:-1]
                cleaned_value = re.sub(r'[$,]', '', value_str)
                try:
                    return float(cleaned_value)
                except ValueError:
                    return 0

            if st.button("Convert and Download Excel", key="convert_download"):
                for col in numerical_columns:
                    if col in all_tables.columns:
                        all_tables[col] = all_tables[col].apply(clean_numeric_value)

                def to_excel(df):
                    output = BytesIO()
                    writer = pd.ExcelWriter(output, engine='xlsxwriter')
                    df.to_excel(writer, index=False, sheet_name='Sheet1')
                    writer.close()
                    processed_data = output.getvalue()
                    return processed_data

                excel_data = to_excel(all_tables)
                st.download_button(label='📥 Download Excel file', data=excel_data, file_name='converted_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except json.JSONDecodeError:
            st.error("The uploaded file is not a valid JSON.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Loan Calculator
def create_loan_calculator():
    st.title("Loan Calculator")

    # Loan Type Input
    loan_type = st.text_input("Loan Type", "Revolver")

    # PD/LGD, Company Name, and Eligibility Inputs
    pd_lgd = st.text_input("PD/LGD", "PD")
    company_name = st.text_input("Company Name", "Dead River Company")
    eligibility = st.radio("Eligibility", ["Directly Eligible", "Similar Entity"])

    # Patronage Radio Button
    patronage = st.radio("Patronage", ["Patronage", "Non-Patronage"])

    # Years to Maturity Slider
    years_to_maturity = st.slider("Years to Maturity", 0.0, 30.0, 5.0, 0.5)

    # Revolver Radio Button
    revolver = st.radio("Revolver", ["Yes", "No"])

    # Direct Note Patronage Input
    direct_note_patronage = st.number_input("Direct Note Patronage (%)", value=0.40, step=0.01, format="%.2f")

    # Fee in lieu Input
    fee_in_lieu = st.number_input("Fee in lieu (%)", value=0.00, step=0.01, format="%.2f")

    # SPREAD, CSA, SOFR, and COFs Inputs
    spread = st.number_input("SPREAD (%)", value=0.00, step=0.01, format="%.2f")
    csa = st.number_input("CSA (%)", value=0.00, step=0.01, format="%.2f")
    sofr = st.number_input("SOFR (%)", value=0.00, step=0.01, format="%.2f")
    cofs = st.number_input("COFs (%)", value=0.00, step=0.01, format="%.2f")

    # Upfront Fee Input
    upfront_fee = st.number_input("Upfront Fee (%)", value=0.00, step=0.01, format="%.2f")

    # Servicing Fee Input
    servicing_fee = st.number_input("Servicing Fee (%)", value=0.15, step=0.01, format="%.2f")

    # Calculate Association Spread
    assoc_spread = spread + csa + sofr - cofs

    # Create DataFrame
    data = {
        'Component': ['Assoc Spread', 'Patronage', 'Fee in lieu', 'Servicing Fee', 'Upfront Fee', 'Years to Maturity', 'Direct Note Pat'],
        loan_type: [f"{assoc_spread:.2f}%", 
                    f"{0.00:.2f}%" if patronage == "Non-Patronage" else "0.00%", 
                    f"{fee_in_lieu:.2f}%", 
                    f"-{servicing_fee:.2f}%", 
                    f"{upfront_fee/years_to_maturity:.2f}%", 
                    f"{years_to_maturity:.1f} years", 
                    f"{direct_note_patronage:.2f}%"]
    }
    df = pd.DataFrame(data)

    # Calculate Income and Capital Yield
    income_yield = assoc_spread + (0 if patronage == "Non-Patronage" else 0) + fee_in_lieu - servicing_fee + (upfront_fee/years_to_maturity)
    capital_yield = income_yield

    # Add Income and Capital Yield to DataFrame
    df = df.append({'Component': 'Income Yield', loan_type: f"{income_yield:.2f}%"}, ignore_index=True)
    df = df.append({'Component': 'Capital Yield', loan_type: f"{capital_yield:.2f}%"}, ignore_index=True)

    # Add PD and Name to DataFrame
    df = df.append({'Component': 'PD', loan_type: pd_lgd}, ignore_index=True)
    df = df.append({'Component': 'Name', loan_type: company_name}, ignore_index=True)
    df = df.append({'Component': 'Eligibility', loan_type: eligibility}, ignore_index=True)

    # Style the DataFrame
    styled_df = df.style.set_properties(**{
        'background-color': 'white',
        'color': 'black',
        'border-color': 'black',
        'border-style': 'solid',
        'border-width': '1px'
    })
    styled_df = styled_df.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
        {'selector': 'th:first-child, td:first-child', 'props': [('text-align', 'left')]}
    ])

    # Display the styled DataFrame
    st.write(styled_df)

    # Export to Excel
    if st.button("Export to Excel"):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Loan Calculation', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Loan Calculation']
            
            # Add formatting
            header_format = workbook.add_format({'bold': True, 'bg_color': '#f0f0f0', 'border': 1})
            cell_format = workbook.add_format({'border': 1})
            
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            for row_num in range(1, len(df) + 1):
                for col_num in range(len(df.columns)):
                    worksheet.write(row_num, col_num, df.iloc[row_num-1, col_num], cell_format)
            
            worksheet.set_column(0, 0, 20)
            worksheet.set_column(1, 1, 15)

        output.seek(0)
        st.download_button(
            label="Download Excel file",
            data=output.getvalue(),
            file_name="loan_calculation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Clear button
    if st.button("Clear"):
        st.experimental_rerun()

# Streamlit App
st.sidebar.title('Navigation')
option = st.sidebar.radio('Select a section:', ['Altman Z Score', 'Futures Pricing', 'JSON Conversion', 'Loan Calculator'])

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
    
    # Dictionary to hold scores for each symbol
    symbol_to_score = {}

    # Calculate Z-Scores
    if st.button('Calculate Z-Scores'):
        for symbol in tickers:
            ticker = yf.Ticker(symbol)
            symbol_to_score[symbol] = z_score(ticker)

        # Categorize
        distress = [''] * len(tickers)
        grey = [''] * len(tickers)
        safe = [''] * len(tickers)

        for idx, symbol in enumerate(tickers):
            zscore = symbol_to_score[symbol]
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

elif option == 'Loan Calculator':
    create_loan_calculator()

