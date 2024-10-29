#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
from io import BytesIO
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# Commodity dictionary
COMMODITIES = {
    # Precious Metals
    'GC=F': {'name': 'Gold', 'color': '#FFD700'},
    'SI=F': {'name': 'Silver', 'color': '#C0C0C0'},
    'PL=F': {'name': 'Platinum', 'color': '#E5E4E2'},
    
    # Energy
    'CL=F': {'name': 'Crude Oil', 'color': '#8B4513'},
    'NG=F': {'name': 'Natural Gas', 'color': '#4169E1'},
    'HO=F': {'name': 'Heating Oil', 'color': '#CD853F'},
    'RB=F': {'name': 'RBOB Gasoline', 'color': '#FF7F50'},
    
    # Base Metals
    'HG=F': {'name': 'Copper', 'color': '#CD7F32'},
    'ALI=F': {'name': 'Aluminum', 'color': '#A0A0A0'},
    
    # Grains
    'ZC=F': {'name': 'Corn', 'color': '#FFE4B5'},
    'ZS=F': {'name': 'Soybeans', 'color': '#90EE90'},
    'ZW=F': {'name': 'Wheat', 'color': '#F4A460'},
    'ZR=F': {'name': 'Rough Rice', 'color': '#FFDAB9'},
    'ZO=F': {'name': 'Oats', 'color': '#DEB887'},
    
    # Softs
    'KC=F': {'name': 'Coffee', 'color': '#8B4513'},
    'CT=F': {'name': 'Cotton', 'color': '#FFFAF0'},
    'SB=F': {'name': 'Sugar', 'color': '#FFFFFF'},
    'CC=F': {'name': 'Cocoa', 'color': '#D2691E'},
    'OJ=F': {'name': 'Orange Juice', 'color': '#FFA500'},
    
    # Livestock
    'LE=F': {'name': 'Live Cattle', 'color': '#FA8072'},
    'HE=F': {'name': 'Lean Hogs', 'color': '#FFC0CB'},
    'GF=F': {'name': 'Feeder Cattle', 'color': '#E9967A'},
    
    # Building Materials
    'LBS=F': {'name': 'Lumber', 'color': '#DEB887'},
    
    # Oilseeds
    'ZL=F': {'name': 'Soybean Oil', 'color': '#98FB98'},
    'ZM=F': {'name': 'Soybean Meal', 'color': '#3CB371'}
}

# Function to calculate annual returns
def calculate_annual_returns(data):
    """Calculate annual returns using first and last trading day of each year"""
    annual_returns = {}
    verification_data = []
    
    for year in sorted(data.index.year.unique()):
        year_data = data[data.index.year == year]
        if len(year_data) > 0:
            first_day = year_data.index[0]
            last_day = year_data.index[-1]
            
            for ticker in data.columns:
                first_price = year_data[ticker].iloc[0]
                last_price = year_data[ticker].iloc[-1]
                return_val = ((last_price - first_price) / first_price * 100)
                
                if year not in annual_returns:
                    annual_returns[year] = {}
                annual_returns[year][ticker] = return_val
                
                verification_data.append({
                    'Year': year,
                    'Ticker': ticker,
                    'Name': COMMODITIES[ticker]['name'],
                    'Start Date': first_day.strftime('%Y-%m-%d'),
                    'End Date': last_day.strftime('%Y-%m-%d'),
                    'Start Price': round(first_price, 2),
                    'End Price': round(last_price, 2),
                    'Return': round(return_val, 1)
                })
    
    return pd.DataFrame(annual_returns).round(1), pd.DataFrame(verification_data)

# Function to create the periodic table
def create_periodic_table(data, annual_returns, start_year, end_year):
    """Create and return the periodic table figure"""
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Filter years
    annual_returns = annual_returns.loc[:, start_year:end_year]
    
    # Define grid parameters
    boxes_per_column = len(data.columns)
    box_width = 0.85
    box_height = 0.85
    spacing = 0.15
    
    # Add title
    plt.text(-0.5, boxes_per_column * (box_height + spacing) + 0.5,
             f"Annual Commodity Returns ({start_year}-{end_year})",
             ha='left', va='center', fontsize=20, fontweight='bold', family='Arial')
    
    for year_idx, year in enumerate(sorted(annual_returns.columns)):
        year_returns = annual_returns[year].sort_values(ascending=False)
        
        for rank, (ticker, return_value) in enumerate(year_returns.items()):
            # Calculate position
            row = boxes_per_column - rank - 1
            col = year_idx
            
            # Get box position
            x = col * (box_width + spacing)
            y = row * (box_height + spacing)
            
            # Get commodity color and name
            commodity_info = COMMODITIES[ticker]
            bg_color = commodity_info['color']
            commodity_name = commodity_info['name']
            
            # Draw box with white edge
            rect = plt.Rectangle((x, y), box_width, box_height,
                               facecolor=bg_color,
                               edgecolor='black',
                               linewidth=1,
                               alpha=0.9)
            ax.add_patch(rect)
            
            # Set text color (red for negative returns, black for positive)
            text_color = 'red' if return_value < 0 else 'black'
            
            # Add commodity name with improved clarity
            plt.text(x + box_width/2, y + box_height*0.65,
                    commodity_name,
                    ha='center', va='center',
                    color=text_color, fontsize=10,
                    fontweight='bold', family='Arial',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # Add return value with improved clarity
            return_text = f"{return_value:.1f}%" if not pd.isna(return_value) else "N/A"
            plt.text(x + box_width/2, y + box_height*0.35,
                    return_text,
                    ha='center', va='center',
                    color=text_color, fontsize=12,
                    fontweight='bold', family='Arial',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Add year labels
    for i, year in enumerate(sorted(annual_returns.columns)):
        plt.text(i * (box_width + spacing) + box_width/2,
                boxes_per_column * (box_height + spacing) + 0.2,
                str(year),
                ha='center', va='bottom',
                fontsize=12,
                fontweight='bold', family='Arial')
    
    # Set plot limits
    plt.xlim(-0.5, len(annual_returns.columns) * (box_width + spacing))
    plt.ylim(-0.5, (boxes_per_column + 0.5) * (box_height + spacing))
    
    # Remove axes
    ax.axis('off')
    
    plt.tight_layout()
    return fig

# Existing functions and code
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

# Initialize default values for the Loan Pricing Calculator
def initialize_defaults():
    st.session_state.default_values = {
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
    }

# Reset callback function for Loan Pricing Calculator
def reset_defaults():
    if 'default_values' not in st.session_state:
        initialize_defaults()
    st.session_state.loans = [st.session_state.default_values.copy() for _ in range(4)]
    st.session_state.current_loan_count = 1

# Initialize session state for Loan Pricing Calculator
if 'loans' not in st.session_state:
    initialize_defaults()
    reset_defaults()

# Initialize session state variables
if 'current_loan_count' not in st.session_state:
    st.session_state.current_loan_count = 1

if 'loans' not in st.session_state:
    st.session_state.loans = [{'Loan Type': 'TL', 'PD/LGD': '', 'Company Name': '', 'Eligibility': 'Directly Eligible', 
                               'Patronage': 'Non-Patronage', 'Revolver': 'No', 'Unused Fee (%)': 0.00,
                               'Direct Note Patronage (%)': 0.40, 'Fee in lieu (%)': 0.00, 'SPREAD (%)': 0.00,
                               'CSA (%)': 0.00, 'SOFR (%)': 0.00, 'COFs (%)': 0.00, 'Upfront Fee (%)': 0.00,
                               'Servicing Fee (%)': 0.15, 'Years to Maturity': 5.0} for _ in range(4)]

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

            # Updated Patronage Radio Button
            patronage_options = ["Patronage 75 bps", "Patronage 71 bps", "Non-Patronage"]
            loan_data['Patronage'] = st.radio(f"Patronage {i + 1}", options=patronage_options, index=patronage_options.index(loan_data['Patronage']) if loan_data['Patronage'] in patronage_options else 2, key=f'Patronage {i}')

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
            
            # Updated patronage value calculation
            if loan_data['Patronage'] == "Patronage 75 bps":
                patronage_value = 0.75
            elif loan_data['Patronage'] == "Patronage 71 bps":
                patronage_value = 0.71
            else:  # Non-Patronage
                patronage_value = 0
            
            capital_yield = income_yield - patronage_value

            # Create DataFrame for main components and a separate one for details
            data_main = {
                'Component': ['Assoc Spread', 'Patronage', 'Fee in lieu', 'Servicing Fee', 'Upfront Fee', 'Direct Note Pat', 'Income Yield', 'Capital Yield'],
                f"{loan_data['Loan Type']}": [f"{assoc_spread:.2f}%", f"-{patronage_value:.2f}%", f"{loan_data['Fee in lieu (%)']:.2f}%", f"-{loan_data['Servicing Fee (%)']:.2f}%", f"{loan_data['Upfront Fee (%)'] / loan_data['Years to Maturity']:.2f}%", f"{loan_data['Direct Note Patronage (%)']:.2f}%", f"{income_yield:.2f}%", f"{capital_yield:.2f}%"]
            }
            data_secondary = {
                'ID': ['PD', 'Name', 'Eligibility', 'Years to Maturity', 'Unused Fee'],
                'Value': [loan_data['PD/LGD'], loan_data['Company Name'], loan_data['Eligibility'], f"{loan_data['Years to Maturity']:.1f} years", f"{loan_data['Unused Fee (%)']:.2f}%"]
            }
            df_main = pd.DataFrame(data_main)
            df_secondary = pd.DataFrame(data_secondary)

            # Display the DataFrames
            st.write("Pricing Information:")
            st.dataframe(df_main)
            st.write("Details:")
            st.dataframe(df_secondary)

    # Add a new loan button if less than 4 loans
    if st.session_state.current_loan_count < 4:
        if st.button("Add Another Loan"):
            st.session_state.current_loan_count += 1

    # Export to Excel with all information
    if st.button("Export to Excel"):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for i in range(st.session_state.current_loan_count):
                loan_data = st.session_state.loans[i]
                
                # Calculate values
                assoc_spread = loan_data['SPREAD (%)'] + loan_data['CSA (%)'] + loan_data['SOFR (%)'] - loan_data['COFs (%)']
                patronage_value = 0.71 if loan_data['Patronage'] == "Patronage" else 0
                income_yield = assoc_spread + loan_data['Direct Note Patronage (%)'] + (loan_data['Upfront Fee (%)'] / loan_data['Years to Maturity']) - loan_data['Servicing Fee (%)']
                capital_yield = income_yield - patronage_value

                # Create DataFrame for main pricing information
                data_main = {
                    'Component': ['Assoc Spread', 'Patronage', 'Fee in lieu', 'Servicing Fee', 'Upfront Fee', 'Direct Note Pat', 'Income Yield', 'Capital Yield'],
                    f"{loan_data['Loan Type']}": [f"{assoc_spread:.2f}%", f"-{patronage_value:.2f}%", f"{loan_data['Fee in lieu (%)']:.2f}%", f"-{loan_data['Servicing Fee (%)']:.2f}%", f"{loan_data['Upfront Fee (%)'] / loan_data['Years to Maturity']:.2f}%", f"{loan_data['Direct Note Patronage (%)']:.2f}%", f"{income_yield:.2f}%", f"{capital_yield:.2f}%"]
                }
                df_main = pd.DataFrame(data_main)

                # Create DataFrame for additional details
                data_details = {
                    'ID': ['Loan Type', 'PD/LGD', 'Company Name', 'Eligibility', 'Patronage', 'Revolver', 'Direct Note Patronage (%)', 'Fee in lieu (%)', 'SPREAD (%)', 'CSA (%)', 'SOFR (%)', 'COFs (%)', 'Upfront Fee (%)', 'Servicing Fee (%)', 'Years to Maturity', 'Unused Fee (%)'],
                    'Value': [loan_data['Loan Type'], loan_data['PD/LGD'], loan_data['Company Name'], loan_data['Eligibility'], loan_data['Patronage'], loan_data['Revolver'], f"{loan_data['Direct Note Patronage (%)']:.2f}%", f"{loan_data['Fee in lieu (%)']:.2f}%", f"{loan_data['SPREAD (%)']:.2f}%", f"{loan_data['CSA (%)']:.2f}%", f"{loan_data['SOFR (%)']:.2f}%", f"{loan_data['COFs (%)']:.2f}%", f"{loan_data['Upfront Fee (%)']:.2f}%", f"{loan_data['Servicing Fee (%)']:.2f}%", f"{loan_data['Years to Maturity']:.1f} years", f"{loan_data['Unused Fee (%)']:.2f}%"]
                }
                df_details = pd.DataFrame(data_details)

                # Write to Excel
                df_main.to_excel(writer, sheet_name=f'Loan {i + 1}', startrow=1, index=False)
                df_details.to_excel(writer, sheet_name=f'Loan {i + 1}', startrow=len(df_main) + 3, index=False)

                # Add basic formatting
                workbook = writer.book
                worksheet = writer.sheets[f'Loan {i + 1}']
                header_format = workbook.add_format({'bold': True})
                
                # Apply formatting to main pricing information
                for col_num, value in enumerate(df_main.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Apply formatting to additional details
                details_start_row = len(df_main) + 3
                for col_num, value in enumerate(df_details.columns.values):
                    worksheet.write(details_start_row - 1, col_num, value, header_format)

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
option = st.sidebar.radio('Select a section:', ['Altman Z Score', 'Futures Pricing', 'JSON Conversion', 'Loan Pricing Calculator', 'Periodic Table of Commodity Returns'])

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

elif option == 'Periodic Table of Commodity Returns':
    st.title('Periodic Table of Commodity Returns')

    # Add title with custom styling
    st.markdown("""
        <h1 style='text-align: center; color: #1E88E5;'>Commodity Returns Dashboard</h1>
        <hr style='margin: 1em 0;'>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    st.sidebar.header("Dashboard Controls")
    
    # Date range selection
    current_year = datetime.now().year
    year_range = range(2005, current_year + 1)
    
    col1, col2 = st.sidebar.columns(2)
    start_year = col1.selectbox("Start Year", year_range, index=9)
    end_year = col2.selectbox("End Year", year_range, index=len(year_range)-1)
    
    if start_year >= end_year:
        st.error("Start year must be before end year!")
    else:
        # Commodity selection
        st.sidebar.subheader("Select Commodities")
        commodity_options = {f"{info['name']} ({ticker})": ticker 
                            for ticker, info in COMMODITIES.items()}
        
        selected_commodities = st.sidebar.multiselect(
            "Choose commodities to display",
            options=list(commodity_options.keys()),
            default=list(commodity_options.keys())[:10]
        )
        
        # Generate button
        if st.sidebar.button("Generate Table", type="primary"):
            if not selected_commodities:
                st.warning("Please select at least one commodity.")
            else:
                # Show loading message
                with st.spinner("Fetching data and generating visualization..."):
                    # Get selected tickers
                    selected_tickers = [commodity_options[comm] for comm in selected_commodities]
                    
                    # Download data
                    data = yf.download(selected_tickers,
                                     start=f"{start_year}-01-01",
                                     end=f"{end_year+1}-01-01",
                                     interval="1d")['Adj Close']
                    
                    # Calculate returns
                    annual_returns, verification_df = calculate_annual_returns(data)
                    
                    # Create tabs for visualization and audit
                    tab1, tab2 = st.tabs(["Periodic Table", "Audit Data"])
                    
                    with tab1:
                        # Display periodic table
                        fig = create_periodic_table(data, annual_returns, start_year, end_year)
                        st.pyplot(fig)
                    
                    with tab2:
                        # Display audit data with formatting
                        st.subheader("Price and Return Verification Data")
                        
                        # Style the dataframe
                        st.dataframe(
                            verification_df.style
                            .format({
                                'Start Price': '${:.2f}',
                                'End Price': '${:.2f}',
                                'Return': '{:.1f}%'
                            })
                            .background_gradient(subset=['Return'], cmap='RdYlGn'),
                            use_container_width=True
                        )
                        
                        # Add download button for audit data
                        csv = verification_df.to_csv(index=False)
                        st.download_button(
                            label="Download Audit Data",
                            data=csv,
                            file_name=f"commodity_returns_audit_{start_year}-{end_year}.csv",
                            mime="text/csv"
                        )

