# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:42:50 2021
@author: fdelgado

"""
###############################################################################
# PYTHON INDIVIDUAL PROJECT - FERNANDO DELGADO
###############################################################################


#=========#
# Imports #
#=========#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yahoo_fin.stock_info as si
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

#set the page layout to wide
st.set_page_config(page_title = "IndividualProjectFinance", layout = "wide")

#==============================================================================
# Tab 1 - Summary
#==============================================================================


def tab1():
    
 
    #Tab Titles 
    st.header("Summary:")
    st.subheader(ticker)
    
    #Split the Tab into 2 Columns
    col1, col2 = st.columns((2,2))   
    
    #Tab information
    @st.cache
    def GetSummary(ticker):
        
        #This gets the summary table from Yahoo Fin
        return si.get_quote_table(ticker, dict_result = False)
    

    if ticker != '-':
        
        #Fills table
        info = GetSummary(ticker)
        info = info.set_index("attribute")
        info["value"] = info["value"].astype(str)
        col1.dataframe(info, height=1000)
    
        #Create Graph Selectbox
        timeselect_list = ['1mo', '3mo', '6mo', 'ytd', '1y', '3y', '5y','max']
        
        global timeselect
        timeselect = col2.selectbox("Select a time period:",timeselect_list)
        
        #Select our data
        #NOTE: i use yfinance library that allows me to use "period" 
        #instead of "start_date" and "end_date"
        chart_summary = yf.download(ticker, period = timeselect, interval = "1d")
        chart_summary.reset_index(inplace=True)
        
        
        # The following plot is inspired by this thread in Stack Overflow:
        # https://stackoverflow.com/questions/64074854/how-to-enable-secondary-y-axis-in-plotly-get-better-visuals
        #However, it has been adjusted extensively to my needs
        
        #Set two Y axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])    
            
            
        #Plot Volume
        fig.add_trace(go.Bar(x=chart_summary['Date'], y=chart_summary['Volume'], 
                                     yaxis='y1',name='Volume'))
            
        #Plot Area Chart
        fig.add_trace(go.Scatter(x=chart_summary['Date'], 
                                 y=chart_summary['Close'],
                                 yaxis='y2', name='Close Price', 
                                 fill='tozeroy', 
                                 line=dict(color='darkturquoise')))
            
        # Change Graph Dimensions
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          showlegend=False)        
                
        # Set y-axes titles
        fig.update_yaxes(showgrid = False, secondary_y=True)
        fig.update_yaxes(showticklabels=False, showgrid = False, 
                         range=[0, chart_summary['Volume'].max()*5], secondary_y=False)
            
        #Show plot on streamlit
        col2.plotly_chart(fig, use_container_width=True)
        
    st.write("Data source: Yahoo Finance") 

#==============================================================================
# Tab 2 - Chart
#==============================================================================


def tab2():
    
    #Tab Titles 
    st.header("Chart:")
    st.subheader(ticker)
    
    #Split the Tab into 3 Columns
    col1, col2, col3= st.columns((2,2,2)) 
    
    
    if ticker != '-':
        
        #Create time period drop-down list
        timeselect_list = ['-','1mo', '3mo', '6mo', 'ytd', '1y', '3y', '5y','max']
        global timeselect
        timeselect = col1.selectbox("Period:",timeselect_list)
        
        #Create graph style drop-down list
        graphstyle_list = ['Candle', 'Line']
        global graphstyle
        graphstyle = col2.selectbox("Graph:", graphstyle_list)
        
        
        #Create time interval drop-down list
        timeinterval_list = ['1d','1wk', '1mo','3mo']
        global timeinterval
        timeinterval = col3.selectbox("Interval:", timeinterval_list)
        
        
        if timeselect != '-':
            
            #get data using yf
            df_chart = yf.download(ticker, period = timeselect, interval = timeinterval)
            df_chart.reset_index(inplace=True)
                
            # create the moving average
            #this code is retrieved from: 
            #https://www.statology.org/exponential-moving-average-pandas/
            ma50 = df_chart['Close'].ewm(span=50, adjust=False).mean()
                
                
            # The wolloing plot is inspired by this thread in Stack Overflow:
            # https://stackoverflow.com/questions/64074854/how-to-enable-secondary-y-axis-in-plotly-get-better-visuals
            #However, it has been adjusted extensively to my needs
                    
            # Create a secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
                
            if graphstyle == 'Candle':
                       
            #plot Candlestick
                fig.add_trace(go.Candlestick(x=df_chart['Date'], 
                                             open=df_chart['Open'], 
                                             high=df_chart['High'],
                                             low=df_chart['Low'], 
                                             close=df_chart['Close'],
                                             yaxis='y2', name='Candle'))
            elif graphstyle == 'Line':
                        
                #Plot Line
                fig.add_trace(go.Line(x=df_chart['Date'], 
                                      y=df_chart['Close'],
                                      yaxis='y2', 
                                      name='Close Price'))
                    
            #Plot Moving Average
            fig.add_trace(go.Scatter(x=df_chart['Date'], 
                                     y=ma50, yaxis = 'y2', 
                                     name='Moving Avg 50', 
                                     line=dict(color='blue')))
                    
            #Plot Volume
            fig.add_trace(go.Bar(x=df_chart['Date'], 
                                 y=df_chart['Volume'], 
                                 yaxis='y1',
                                 name='Volume'))
                
            # Change Graph Dimensions
            fig.update_layout(height=600, 
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')        
                    
            # Set y-axes titles
            fig.update_yaxes(showgrid = False, 
                             secondary_y=True)
                    
            fig.update_yaxes(showticklabels=False, 
                             showgrid = False, 
                             range=[0, df_chart['Volume'].max()*5], 
                             secondary_y=False)
                
            #Show plot on streamlit
            st.plotly_chart(fig, use_container_width=True)

        
        
#==============================================================================
# Tab 3 - Statistics
#==============================================================================


def tab3():
    
    #Tab Titles 
    st.header("Statistics:")
    st.subheader(ticker)
    
    #Split the Tab into 2 Columns
    col1, col2 = st.columns((2,2)) 
    
    if ticker != '-':
        #Get data
        df = si.get_stats(ticker)
        vm = si.get_stats_valuation(ticker)
        
        #Subset
        #Valuation Measures
        col1.subheader('Valuation Measures')
        #vm.index = vm.iloc[:, 0]
        #vm = vm.drop(0, 1)
        vm.columns = [' ','Value']
        col1.table(vm)
        
        #Financial Highlights
        col1.subheader('Financial Highlights')
        
        #Fiscal Year
        col1.write('Fiscal Year')
        fh = df.iloc[29:31]
        fh.index = fh['Attribute']
        fh = fh.drop('Attribute', 1)
        col1.table(fh)
           
        #Profitability
        col1.write('Profitability')
        p = df.iloc[31:33]
        p.index = p['Attribute']
        p = p.drop('Attribute', 1)
        col1.table(p)
        
        #Management Efectiveness
        col1.write('Management Effectiveness')
        mgmt = df.iloc[33:35]
        mgmt.index = mgmt['Attribute']
        mgmt = mgmt.drop('Attribute', 1)
        col1.table(mgmt)
        
        #Income Statement
        col1.write('Income Statement')
        ii = df.iloc[35:43]
        ii.index = ii['Attribute']
        ii = ii.drop('Attribute', 1)
        col1.table(ii)
        
        #Balance Sheet
        col1.write('Balance Sheet')
        bs = df.iloc[43:49]
        bs.index = bs['Attribute']
        bs = bs.drop('Attribute', 1)
        col1.table(bs)
        
        #Cash Flow
        col1.write('Cash Flow Statement')
        cf = df.iloc[49:]
        cf.index = cf['Attribute']
        cf = cf.drop('Attribute', 1)
        col1.table(cf)
    
        #Trading Information 
        col2.subheader('Trading Information')
        
        #Stock Price History
        col2.write('Stock Price History')
        sph = df.iloc[0:7]
        sph.index = sph['Attribute']
        sph = sph.drop('Attribute', 1)
        col2.table(sph)
        
        #Share Statistics
        col2.write('Share Statistics')
        ss = df.iloc[7:19]
        ss.index = ss['Attribute']
        ss = ss.drop('Attribute', 1)
        col2.table(ss)
        
        #Dividend & Splits
        col2.write('Dividend & Splits')
        dds = df.iloc[19:29]
        dds.index = dds['Attribute']
        dds = dds.drop('Attribute', 1)
        col2.table(dds)   
        
    st.write("Data source: Yahoo Finance") 
    
#==============================================================================
# Tab 4 - Financials
#==============================================================================


def tab4():
    
    #Tab Titles 
    st.header("Financials:")
    st.subheader(ticker)
    
    col1, col2, col3 = st.columns((2,4,1))
    
    if ticker != '-':
        
        #Create dropdown list
        fs_list = ['Income Statement','Balance Sheet', 'Cash Flow']
        global fs
        fs = col1.selectbox("Time Interval:", fs_list)
        
        periods_list = ['Yearly', 'Quarterly']
        global periods
        periods = col3.selectbox('Period:', periods_list)
        
        col3.write('_Currency in USD_')
    
        #If period is Yearly
        if periods == 'Yearly':
    
            #Get IS
            if fs == 'Income Statement':
                fs_is = si.get_income_statement(ticker)
                
                #Set Correct Order of Income Statement
                fs_is.reset_index(inplace=True)
                fs_is = fs_is.reindex([15,17,6,5,0,9,18,8,10,2,14,4,20,21,16,11,12,13,19,7,1,3])
                fs_is.set_index('Breakdown')
                st.table(fs_is)

            
            #Get BS
            elif fs == 'Balance Sheet':
                fs_bs = si.get_balance_sheet(ticker)
                st.table(fs_bs)
            
            #Get CF
            else:
                fs_cf = si.get_cash_flow(ticker)
                st.table(fs_cf)
                
        #If period is Quarterly        
        else:
            
            #Get IS
            if fs == 'Income Statement':
                fs_is = si.get_income_statement(ticker, yearly = False)
                st.table(fs_is)
            
            #Get BS
            elif fs == 'Balance Sheet':
                fs_bs = si.get_balance_sheet(ticker, yearly = False)
                st.table(fs_bs)
            
            #Get CF
            else:
                fs_cf = si.get_cash_flow(ticker, yearly = False)
                st.table(fs_cf)
                
    st.write("Data source: Yahoo Finance") 

#==============================================================================
# Tab 5 - Analysis
#==============================================================================


def tab5():
    
    #Tab Titles 
    st.header("Analysis:")
    st.subheader(ticker)
    
    #Split the Tab into Columns
    col1, col2, col3 = st.columns((2,4,1)) 
    
    col3.write('_Currency in USD_')
    
    if ticker != '-':
        
        #Get Data
        analysis = si.get_analysts_info(ticker)
        
        #Earnings Estimate
        st.write('Earnings Estimate')
        EE = analysis['Earnings Estimate']
        EE = EE.set_index('Earnings Estimate')
        st.table(EE)
        
        #Revenue Estimate
        st.write('Revenue Estimate')
        RE = analysis['Revenue Estimate']
        RE = RE.set_index('Revenue Estimate')
        st.table(RE)
        
        #Earnings History
        st.write('Earnings History')
        EH = analysis['Earnings History']
        EH = EH.set_index('Earnings History')
        st.table(EH)
        
        #EPS Trend
        st.write('EPS Trend')
        EPS = analysis['EPS Trend']
        EPS = EPS.set_index('EPS Trend')
        st.table(EPS)
        
        #EPS Revisions
        st.write('EPS Revisions')
        EPSr = analysis['EPS Revisions']
        EPSr = EPSr.set_index('EPS Revisions')
        st.table(EPSr)
        
        #Growth Estimates
        st.write('Growth Estimates')
        GE = analysis['Growth Estimates']
        GE = GE.set_index('Growth Estimates')
        st.table(GE)
        
    st.write("Data source: Yahoo Finance") 
    
#==============================================================================
# Tab 6 - Monte Carlo Simulation
#==============================================================================


def tab6():
    
    #Tab Titles 
    st.header("Monte Carlo Simulation:")
    st.subheader(ticker)
    
    #Split the Tab into 2 Columns
    col1, col2 = st.columns((2,2)) 
    
    if ticker != '-':
        
        #Set Dropdown Lists
        simulation_list = [200, 500, 1000]
        global simulationnbr
        simulationnbr = col1.selectbox("Choose the number of simulations:", simulation_list)
        
        days_list = [30, 60, 90]
        global daysnbr
        daysnbr = col2.selectbox("Choose the number of days:", days_list)
        
        #Get Data
        stock_price = yf.download(ticker, period = '1y', interval = "1d")
        
        
        #=====================================================================
        #The following code is the same as we saw in class:
        
        # Take the close price
        close_price = stock_price['Close']
        
        # The returns ((today price - yesterday price) / yesterday price)
        daily_return = close_price.pct_change()
        
        # The volatility (high value, high risk)
        daily_volatility = np.std(daily_return)
        
        # Setup the Monte Carlo simulation
        np.random.seed(123)
        simulations = int(simulationnbr)
        time_horizone = int(daysnbr)
        
        # Run the simulation
        simulation_df = pd.DataFrame()
        
        for i in range(simulations):
            
            # The list to store the next stock price
            next_price = []
            
            # Create the next stock price
            last_price = close_price[-1]
            
            for j in range(time_horizone):
                # Generate the random percentage change around the mean (0) and std (daily_volatility)
                future_return = np.random.normal(0, daily_volatility)
        
                # Generate the random future price
                future_price = last_price * (1 + future_return)
        
                # Save the price and go next
                next_price.append(future_price)
                last_price = future_price
            
            # Store the result of the simulation
            simulation_df[i] = next_price
        
        #VAR
        ending_price = simulation_df.iloc[-1:, :].values[0, ]
        future_price_95ci = np.percentile(ending_price, 5)
        VaR = close_price[-1] - future_price_95ci
        st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')
        
        #plot
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10, forward=True)
        
        plt.plot(simulation_df)
        plt.title('Monte Carlo simulation for ' + ticker +' stock price in next ' + str(daysnbr) + ' days')
        plt.xlabel('Day')
        plt.ylabel('Price')
        
        plt.axhline(y=close_price[-1], color='red')
        plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
        ax.get_legend().legendHandles[0].set_color('red')
        
        st.pyplot(fig)

    st.write("Data source: Yahoo Finance")
   
#==============================================================================
# Tab 7 - Historical Data & Download
#==============================================================================

def tab7():
       
    #Tab Titles 
    st.header("Historical Data:")
    st.subheader(ticker)
    
    #Split the Tab into Columns
    col1, col2, col3, col4 = st.columns((2,1,3,1))
    
    #Create period drop-down list
    timeselect_list = ['1mo', '3mo', '6mo', 'ytd', '1y', '3y', '5y','max']
    global timeselect
    timeselect = col1.selectbox("Choose date period:",timeselect_list)
        
    #Create time interval drop-down list
    timeinterval_list = ['1d','1wk', '1mo']
    global timeinterval
    timeinterval = col2.selectbox("Frequency:", timeinterval_list)
    
    
    #Get Data
    df = yf.download(ticker, period = timeselect, interval = timeinterval)
    
    
    #Download to excel file
    #Retrieved from:
    #https://discuss.streamlit.io/t/download-button-for-csv-or-xlsx-file/17385/2
    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        format1 = workbook.add_format({'num_format': '0.00'}) 
        worksheet.set_column('A:A', None, format1)  
        writer.save()
        processed_data = output.getvalue()
        return processed_data
    
    df2 = yf.download(ticker, period = timeselect, interval = timeinterval)
    df2.reset_index(inplace=True)
    df_xlsx = to_excel(df2)
    col4.download_button(label='👽 Download Current Result',
                                    data=df_xlsx ,
                                    file_name= ticker + '_'+ timeselect + '_' + timeinterval + '.xlsx')
    #Show Table
    st.table(df)
        
#==============================================================================
# Main body
#==============================================================================

def run():
    #Sidebar design
    st.sidebar.header("FINANCIAL DASHBOARD")
    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    ticker_list = ['-'] + si.tickers_sp500()
    

     
    # Add selection box
    global ticker
    ticker = st.sidebar.selectbox("Select a ticker", ticker_list)
    
    
    if ticker != '-':
        st.sidebar.button('👽 Refresh Data')
            
    # Add a radio box|
    select_tab = st.sidebar.radio("Select tab:", ['Summary', 
                                                 'Chart', 
                                                 'Statistics', 
                                                 'Financials', 
                                                 'Analysis', 
                                                 'Monte Carlo Simulation', 
                                                 'Historical Data & Download'])
            
    # Show the selected tab
    if select_tab == 'Summary':
        # Run tab 1
        tab1()
    elif select_tab == 'Chart':
        #Run tab 2
        tab2()
    elif select_tab == 'Statistics':
        #Run tab 3
        tab3()
    elif select_tab == 'Financials':
        #Run tab 4
        tab4()
    elif select_tab == 'Analysis':
        #Run tab 5
        tab5()
    elif select_tab == 'Monte Carlo Simulation':
        #Run tab 6
        tab6()
    elif select_tab == 'Historical Data & Download':
        #Run tab 7
        tab7()
    
    st.sidebar.caption("_DASHBOARD BY: FERNANDO DELGADO_")
    
if __name__ == "__main__":
    run()
 
###############################################################################
# END
###############################################################################