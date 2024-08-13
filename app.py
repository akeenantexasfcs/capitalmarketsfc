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

# Initialize default values for the session state
import streamlit as st
import pandas as pd
from io import BytesIO


# Initialize default values for the session state
def initialize_defaults():
    st.session_state.default_values = {
        'Loan Type': "Insert Loan Type",
        'PD/LGD': "Insert PD/LGD",
        'Company Name': "Insert Company Name",
        'Eligibility': "Directly Eligible",
        'Patronage': "Non-Patronage",
        'Revolver': "No",
        'Direct Note Patronage (%)': 0.40,  # Correct default value
        'Fee in lieu (%)': 0.00,
        'SPREAD (%)': 0.00,
        'CSA (%)': 0.00,
        'SOFR (%)': 0.00,
        'COFs (%)': 0.00,
        'Upfront Fee (%)': 0.00,
        'Servicing Fee (%)': 0.15,
        'Years to Maturity': 5.0,
        'Unused Fee (%)': 0.00
    }

# Reset callback function
def reset_defaults():
    if 'default_values' not in st.session_state:
        initialize_defaults()
    st.session_state.loans = [st.session_state.default_values.copy() for _ in range(4)]
    st.session_state.current_loan_count = 1

# Initialize session state
if 'loans' not in st.session_state:
    initialize_defaults()
    reset_defaults()

# Place Loan Calculator Code here at the bottom to make it easier to update
def create_loan_calculator():
    st.title("Loan Pricing Calculator")

    # Show inputs for each loan
    for i in range(st.session_state.current_loan_count):
        with st.expander(f"Loan {i + 1} Details", expanded=True):
            loan_data = st.session_state.loans[i]
            # Loan Type Input
            loan_data['Loan Type'] = st.text_input(f"Loan Type {i + 1}", value=loan_data['Loan Type'], key=f'Loan Type {i}')

            # PD/LGD, Company Name, and Eligibility Inputs at the top
            loan_data['PD/LGD'] = st.text_input(f"PD/LGD {i + 1}", value=loan_data['PD/LGD'], key=f'PD/LGD {i}')
            loan_data['Company Name'] = st.text_input(f"Company Name {i + 1}", value=loan_data['Company Name'], key=f'Company Name {i}')
            eligibility_options = ["Directly Eligible", "Similar Entity"]
            loan_data['Eligibility'] = st.radio(f"Eligibility {i + 1}", options=eligibility_options, index=eligibility_options.index(loan_data['Eligibility']), key=f'Eligibility {i}')

            # Patronage Radio Button
            patronage_options = ["Patronage", "Non-Patronage"]
            loan_data['Patronage'] = st.radio(f"Patronage {i + 1}", options=patronage_options, index=patronage_options.index(loan_data['Patronage']), key=f'Patronage {i}')

            # Revolver Radio Button
            revolver_options = ["Yes", "No"]
            loan_data['Revolver'] = st.radio(f"Revolver {i + 1}", options=revolver_options, index=revolver_options.index(loan_data['Revolver']), key=f'Revolver {i}')

            # Unused Fee Input (shown if Revolver is "Yes")
            if loan_data['Revolver'] == "Yes":
                loan_data['Unused Fee (%)'] = st.number_input(f"Unused Fee (%) {i + 1}", value=loan_data['Unused Fee (%)'], step=0.01, format='%.2f', key=f'Unused Fee {i}')
            else:
                loan_data['Unused Fee (%)'] = 0.00

            # Direct Note Patronage Input
            loan_data['Direct Note Patronage (%)'] = st.number_input(f"Direct Note Patronage (%) {i + 1}", value=loan_data['Direct Note Patronage (%)'], step=0.01, format="%.2f", key=f'Direct Note Patronage {i}')

            # Fee in lieu Input
            loan_data['Fee in lieu (%)'] = st.number_input(f"Fee in lieu (%) {i + 1}", value=loan_data['Fee in lieu (%)'], step=0.01, format="%.2f", key=f'Fee in lieu {i}')

            # SPREAD, CSA, SOFR, and COFs Inputs
            loan_data['SPREAD (%)'] = st.number_input(f"SPREAD (%) {i + 1}", value=loan_data['SPREAD (%)'], step=0.01, format="%.2f", key=f'SPREAD {i}')
            loan_data['CSA (%)'] = st.number_input(f"CSA (%) {i + 1}", value=loan_data['CSA (%)'], step=0.01, format="%.2f", key=f'CSA {i}')
            loan_data['SOFR (%)'] = st.number_input(f"SOFR (%) {i + 1}", value=loan_data['SOFR (%)'], step=0.01, format="%.2f", key=f'SOFR {i}')
            loan_data['COFs (%)'] = st.number_input(f"COFs (%) {i + 1}", value=loan_data['COFs (%)'], step=0.01, format="%.2f", key=f'COFs {i}')

            # Upfront Fee Input
            loan_data['Upfront Fee (%)'] = st.number_input(f"Upfront Fee (%) {i + 1}", value=loan_data['Upfront Fee (%)'], step=0.01, format="%.2f", key=f'Upfront Fee {i}')

            # Servicing Fee Input
            loan_data['Servicing Fee (%)'] = st.number_input(f"Servicing Fee (%) {i + 1}", value=loan_data['Servicing Fee (%)'], step=0.01, format="%.2f", key=f'Servicing Fee {i}')

            # Years to Maturity Slider
            loan_data['Years to Maturity'] = st.slider(f"Years to Maturity {i + 1}", 0.0, 30.0, value=loan_data['Years to Maturity'], step=0.5, key=f'Years to Maturity {i}')

            # Calculate Association Spread
            assoc_spread = loan_data['SPREAD (%)'] + loan_data['CSA (%)'] + loan_data['SOFR (%)'] - loan_data['COFs (%)']

            # Calculate Income and Capital Yield
            income_yield = assoc_spread + loan_data['Direct Note Patronage (%)'] + (loan_data['Upfront Fee (%)'] / loan_data['Years to Maturity']) - loan_data['Servicing Fee (%)']
            patronage_value = 0.71 if loan_data['Patronage'] == "Patronage" else 0
            capital_yield = income_yield - patronage_value

            # Create DataFrame for main components and a separate one for PD, Name, and Eligibility
            data_main = {
                'Component': ['Assoc Spread', 'Patronage', 'Fee in lieu', 'Servicing Fee', 'Upfront Fee', 'Direct Note Pat', 'Income Yield', 'Capital Yield'],
                f"Loan {i + 1}": [f"{assoc_spread:.2f}%", f"-{patronage_value:.2f}%", f"{loan_data['Fee in lieu (%)']:.2f}%", f"-{loan_data['Servicing Fee (%)']:.2f}%", f"{loan_data['Upfront Fee (%)'] / loan_data['Years to Maturity']:.2f}%", f"{loan_data['Direct Note Patronage (%)']:.2f}%", f"{income_yield:.2f}%", f"{capital_yield:.2f}%"]
            }
            data_secondary = {
                'ID': ['PD', 'Name', 'Eligibility', 'Years to Maturity', 'Unused Fee'],
                'Value': [loan_data['PD/LGD'], loan_data['Company Name'], loan_data['Eligibility'], f"{loan_data['Years to Maturity']:.1f} years", f"{loan_data['Unused Fee (%)']:.2f}%"]
            }
            df_main = pd.DataFrame(data_main)
            df_secondary = pd.DataFrame(data_secondary)

            # Styling for the main table
            def apply_main_table_styles(row):
                return ['background-color: rgb(94, 151, 50); color: white; font-weight: bold' if row['Component'] in ['Income Yield', 'Capital Yield'] else ''] * 2

            styled_df_main = df_main.style.apply(apply_main_table_styles, axis=1)
            styled_df_secondary = df_secondary.style.set_properties(**{'background-color': 'white', 'color': 'black'})

            # Display the styled DataFrame
            st.write("Pricing Information:")
            st.dataframe(styled_df_main)
            st.write("Details:")
            st.dataframe(styled_df_secondary)

    # Add a new loan button if less than 4 loans
    if st.session_state.current_loan_count < 4:
        if st.button("Add Another Loan"):
            st.session_state.current_loan_count += 1

    # Export to Excel with space between tables
    if st.button("Export to Excel"):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for i in range(st.session_state.current_loan_count):
                df_main = pd.DataFrame.from_dict(st.session_state.loans[i], orient='index', columns=[f"Loan {i + 1}"])
                df_main.to_excel(writer, sheet_name=f'Loan {i + 1}', startrow=1, header=False)

                # Write custom headers
                worksheet = writer.sheets[f'Loan {i + 1}']
                worksheet.write(0, 0, 'Component')
                worksheet.write(0, 1, f'Loan {i + 1}')

                # Add formatting
                workbook = writer.book
                header_format = workbook.add_format({'bold': True, 'bg_color': '#f0f0f0', 'border': 1})
                cell_format = workbook.add_format({'border': 1})
                worksheet.set_column('A:B', 20, cell_format)

        output.seek(0)
        st.download_button(
            label="Download Excel file",
            data=output.getvalue(),
            file_name="loan_pricing_calculations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Clear button with a callback to reset defaults
    st.button("Reset", on_click=reset_defaults)


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

elif option == 'Loan Pricing Calculator':
    create_loan_calculator()

