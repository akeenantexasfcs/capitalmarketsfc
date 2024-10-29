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

# Streamlit App
st.sidebar.title('Navigation')
option = st.sidebar.radio('Select a section:', ['Altman Z Score', 'Futures Pricing', 'JSON Conversion', 'Loan Pricing Calculator', 'Periodic Table of Commodity Returns'])

if option == 'Periodic Table of Commodity Returns':
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
    end_year = col2.selectbox("End Year", year_range, index=len(year_range) - 1)
    
    if start_year >= end_year:
        st.error("Start year must be before end year!")
    else:
        # Commodity selection
        st.sidebar.subheader("Select Commodities")
        commodity_options = {f"{info['name']} ({ticker})": ticker for ticker, info in COMMODITIES.items()}
        
        selected_commodities = st.sidebar.multiselect(
            "Choose commodities to display",
            options=list(commodity_options.keys()),
            default=list(commodity_options.keys())[:10]
        )
        
        # Generate button
        if st.sidebar.button("Generate Table"):
            if not selected_commodities:
                st.warning("Please select at least one commodity.")
            else:
                # Show loading message
                with st.spinner("Fetching data and generating visualization..."):
                    try:
                        # Get selected tickers
                        selected_tickers = [commodity_options[comm] for comm in selected_commodities]
                        
                        # Download data
                        data = yf.download(selected_tickers,
                                           start=f"{start_year}-01-01",
                                           end=f"{end_year + 1}-01-01",
                                           interval="1d")['Adj Close']
                        
                        if data.empty:
                            st.error("No data available for the selected tickers and dates. Please try different options.")
                        else:
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
                    except Exception as e:
                        st.error(f"An error occurred while fetching data: {e}")

