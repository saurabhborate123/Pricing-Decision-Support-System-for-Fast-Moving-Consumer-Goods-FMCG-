# ------------------------------------------------------------------------------------------------------
#                                   Import Libraries
# ------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import altair as alt
import pickle
from scipy import stats
from datetime import timedelta
from sklearn.preprocessing import OrdinalEncoder
from stable_baselines3 import TD3


# ------------------------------------------------------------------------------------------------------
#                                   Load Models
# ------------------------------------------------------------------------------------------------------
# Load Demand Forecasting model
with open('best_xgb_2.pkl', 'rb') as file:
    xgb = pickle.load(file)

# Load Price Optimization model
td3_5 = TD3.load("TD3_5")

# Layout Settings
#st. set_page_config(layout="centered")
st. set_page_config(layout="wide")
custom_css = """
<style>
    .main .block-container {
        max-width: 50%;  /* Adjust this value to your desired width */
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 2rem;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------------
#                                   Load Data Files
# ------------------------------------------------------------------------------------------------------
@st.cache_data   # To cache data and avoid re-computation

def load_data():
    product = pd.read_excel('Dataset\Dataset.xlsx' , sheet_name="dh Products Lookup")
    store = pd.read_excel('Dataset\Dataset.xlsx' , sheet_name="dh Store Lookup")
    upc_encoding = pd.read_excel('Dataset/StoreID UPC Encoding.xlsx', sheet_name='Product')
    store_encoding = pd.read_excel('Dataset/StoreID UPC Encoding.xlsx', sheet_name='Store')
    data = pd.read_excel('Dataset/FINAL SUBSET_preprocessed_data.xlsx')   
    elasticity = pd.read_excel('Dataset/Price Elasticity.xlsx')
    subset = pd.read_excel('Dataset/Subset Data Random Price Optimization.xlsx')
    return product, store, upc_encoding, store_encoding, data, elasticity, subset

product, store, upc_encoding, store_encoding, data, elasticity, subset = load_data()

# ------------------------------------------------------------------------------------------------------
#                                   Functions for Model Operations
# ------------------------------------------------------------------------------------------------------

# To get subset data based on store-upc combination
def get_store_upc_data(store_id, upc):
    store_upc_data = data[(data['UPC'] == upc) & (data['STORE_ID'] == store_id)]
    return store_upc_data

# To pre-process user inputs for demand forecasting
def processing_pipeline(input_data):
    processed_input = input_data.copy()
    store_id = processed_input['STORE_ID'].iloc[0]
    upc = processed_input['UPC'].iloc[0]

    store_upc_data = get_store_upc_data(store_id, upc)
    store_upc_data = store_upc_data.sort_values(by='WEEK_END_DATE', ascending=False)

    # Get the next week date of last date using  timedelta(days=7) 
    processed_input['WEEK_END_DATE'] = store_upc_data['WEEK_END_DATE'].max() + timedelta(days=7) 

    # Extract YEAR and WEEK_NUM 
    processed_input['YEAR'] = processed_input['WEEK_END_DATE'].dt.year
    processed_input['WEEK_NUM'] = processed_input['WEEK_END_DATE'].dt.isocalendar().week

    # Frequecy encoding of STORE_ID
    processed_input = processed_input.merge(store_encoding, on='STORE_ID', how='left')
    # Frequecy encoding of UPC
    processed_input = processed_input.merge(upc_encoding, on='UPC', how='left')

    # Get SEG_VALUE_NAME based on STORE_ID
    processed_input = processed_input.merge(store[['STORE_ID', 'SEG_VALUE_NAME']], on='STORE_ID', how='left')

    # Ordinal encoding of SEG_VALUE_NAME
    encoder = OrdinalEncoder(categories=[['VALUE', 'MAINSTREAM', 'UPSCALE']]) 
    processed_input['SEG_VALUE_NAME_ORDINAL'] = encoder.fit_transform(processed_input[['SEG_VALUE_NAME']]).astype(int) 

    # Get SALES_AREA_SIZE_NUM based on STORE_ID
    processed_input = processed_input.merge(store[['STORE_ID', 'SALES_AREA_SIZE_NUM']], on='STORE_ID', how='left')

    # Get Category based on UPC 
    processed_input = processed_input.merge(product[['UPC', 'CATEGORY']], on='UPC', how='left')

    # One-hot encoding of CATEGORY
    product_categories = ['BAG SNACKS', 'COLD CEREAL', 'FROZEN PIZZA', 'ORAL HYGIENE PRODUCTS']
    for cat in product_categories:
        processed_input.loc[:,cat] = 0
    category = processed_input['CATEGORY']
    processed_input[category] = 1

    # Get UNIT_SALES_LOG_LAG1 and UNIT_SALES_LOG_LAG2
    last_data = store_upc_data.head(1)     # Get the last week of sales data (first row)
    last2_data = store_upc_data.iloc[[1]]  # Get the second last week of sales data (second row)
    processed_input['UNIT_SALES_LOG_LAG1'] = last_data['UNIT_SALES_LOG'].iloc[0]
    processed_input['UNIT_SALES_LOG_LAG2'] = last2_data['UNIT_SALES_LOG'].iloc[0]

    # Get the needed features for forecasting
    features = ['YEAR', 'WEEK_NUM', 'STORE_ID_COUNT', 'UPC_COUNT', 'UNIT_SALES_LOG_LAG1', 'UNIT_SALES_LOG_LAG2',
                'PRICE', 'FEATURE', 'DISPLAY', 'SEG_VALUE_NAME_ORDINAL', 'SALES_AREA_SIZE_NUM',
                'BAG SNACKS', 'COLD CEREAL', 'FROZEN PIZZA', 'ORAL HYGIENE PRODUCTS']
    return processed_input[features]

# To forecast unit sales using xgb
def forecast_unit_sales(processed_input_df):
    pred_unit_sales = xgb.predict(processed_input_df)
    pred_unit_sales = np.exp(pred_unit_sales[0])
    pred_unit_sales = np.round(pred_unit_sales).astype(int)
    return pred_unit_sales

def show_feature_importance(X):
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns, 
        'Importance Score': xgb.feature_importances_}
    ).sort_values(by='Importance Score', ascending=False)  
    feature_importance_df['Importance Score'] = feature_importance_df['Importance Score'].apply(lambda x: f"{x:.4f}")
    return feature_importance_df


def get_idx_store_id(store_id):
    store_id_to_idx = {store_id: idx for idx, store_id in enumerate(subset['STORE_ID'].unique())}
    store_id_idx = store_id_to_idx.get(store_id, -1)
    return store_id_idx 

def get_idx_upc(upc):
    upc_to_idx = {upc: idx for idx, upc in enumerate(subset['UPC'].unique())}
    upc_idx = upc_to_idx.get(upc, -1)
    return upc_idx 

def optimize_price(store_id, upc, feature, display, elasticity_value, forecasted_unit_sales):
    if store_id in [2277, 15755, 25253]:  # These 3 STORE_IDs are used in training
        store_id_idx = get_idx_store_id(store_id)  
    else:
        store_seg = store.loc[store['STORE_ID'] == store_id, 'SEG_VALUE_NAME'].values[0]   
        if store_seg == 'UPSCALE':
            store_id_idx = 0
        elif store_seg == 'VALUE':
            store_id_idx = 1
        else: 
            store_id_idx = 2   
    upc_idx = get_idx_upc(upc)
    state = {
        'store_id': store_id_idx,                    
        'upc': upc_idx,                           
        'feature': feature,                       
        'display': display,                      
        'elasticity': np.array(np.float32([elasticity_value])),  
        'forecasted_unit_sales': np.array(np.int32([forecasted_unit_sales]))  
    }
    print('state: ', state)
    normalized_price_action, _ = td3_5.predict(state)
    return normalized_price_action[0]
    

# ------------------------------------------------------------------------------------------------------
#                                   Functions for Interface Operations
# ------------------------------------------------------------------------------------------------------

# To extract unique STORE_ID
def get_store_id(store_seg_options):
    if store_seg_options:
        store_data = store[(store['SEG_VALUE_NAME'].isin(store_seg_options))]
        store_ids = sorted(store_data['STORE_ID'].unique())
    else:
        store_ids = sorted(elasticity['STORE_ID'].unique())
    return store_ids

# To extract UPCs based on the selected store
def get_upcs(store_id, sub_category_options):
    if sub_category_options: 
        store_id_data = data[(data['STORE_ID'] == store_id) & (data['SUB_CATEGORY'].isin(sub_category_options))]
    else:  
        store_id_data = data[data['STORE_ID'] == store_id]
    upcs = sorted(store_id_data['UPC'].unique())
    return upcs

# To extract price range for the selected store ID and UPC
def get_price_range(store_id, upc): 
    subset = get_store_upc_data(store_id, upc)
    min_price = subset['PRICE'].min()
    max_price = subset['PRICE'].max()
    median_price = subset['PRICE'].median()
    return min_price, max_price, median_price

# To extract store details based on store ID
def get_store_details(store_id):
    store_details_df = store[store['STORE_ID'] == store_id]
    store_details_df = store_details_df[['STORE_NAME', 'ADDRESS_CITY_NAME', 'ADDRESS_STATE_PROV_CODE', 
                                         'SEG_VALUE_NAME', 'SALES_AREA_SIZE_NUM']].transpose()
    store_details_df.columns = [store_id]
    store_details_df = store_details_df.astype(str)
    return store_details_df

# To extract product details based on UPC
def get_produt_details(upc):
    product_details_df = product[product['UPC'] == upc]
    product_details_df = product_details_df.drop(columns=['UPC']).transpose()
    product_details_df.columns = [upc]
    product_details_df = product_details_df.astype(str)
    return product_details_df

# To get price elasticity of demand for selected store id and upc
def get_store_upc_elasticity(store_id, upc):
    store_upc_elasticity_df = elasticity[(elasticity['STORE_ID'] == store_id) & (elasticity['UPC'] == upc)]
    elasticity_value = store_upc_elasticity_df['ABS_PRICE_ELASTICITY'].iloc[0]
    return elasticity_value 

# To visualize price elasticity
def show_elasticity_info(elasticity_value):
    if 0 < elasticity_value <= 0.1:
        color = '#90be6d'  # light green
        elasticity_type = 'Highly Inelastic'  
    elif 0.1 < elasticity_value <= 0.5:
        color = '#f9c74f'  # yellow
        elasticity_type = 'Moderately Inelastic' 
    elif 0.5 < elasticity_value <= 1:
        color = '#f3722c'  # orange
        elasticity_type = 'Slightly Inelastic'
    else:
        color = '#f94144'  # red
        elasticity_type = 'Elastic'

    card_css = """
        <style>
        .card {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
            margin: 10px 0;
        }
        .card-header {
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }
        .card-content {
            padding-top: 20px;
        }
        </style>
    """
    st.markdown(card_css, unsafe_allow_html=True)
    st.markdown(f"""<div class="card"><div class="card-header">{elasticity_type} Demand </div><div>""", unsafe_allow_html=True)
    data = pd.DataFrame({
        'Type': ['Elasticity'],
        'Value': [elasticity_value]
    })
    end = 2 if elasticity_value < 1 else 8.5
    chart = alt.Chart(data).mark_bar(size=20).encode(
        x=alt.X('Value:Q', title='Elasticity Value', scale=alt.Scale(domain=(0, end), nice=False, padding=0.5)),
        color=alt.ColorValue(color),
        tooltip=['Value']
    ).properties(
        width=400,
        height=100
    )
    st.altair_chart(chart, use_container_width=True)


# ------------------------------------------------------------------------------------------------------
#                                   Functions to Plot Graphs
# ------------------------------------------------------------------------------------------------------
# To visualzie unit sales/price and revenue over time
def plot_unit_sales_revenue(store_id, upc, var):
    subset = get_store_upc_data(store_id, upc).copy()
    subset['REVENUE'] = subset['UNIT_SALES'] * subset['PRICE']
    color = '#a4ac86' if var == 'UNIT_SALES' else '#9a8c98'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset['WEEK_END_DATE'], y=subset[var], 
                             mode='lines', name=var, 
                             hovertemplate=f'<b>WEEK_END_DATE</b>: %{{x}}<br><b>{var}</b>: %{{y}}', 
                             line=dict(color=color)))  
    fig.add_trace(go.Scatter(x=subset['WEEK_END_DATE'], y=subset['REVENUE'], 
                             mode='lines', name='REVENUE', yaxis='y2', 
                             hovertemplate='<b>WEEK_END_DATE</b>: %{x}<br><b>REVENUE</b>: %{y}',
                             line=dict(color='#CAB047')))   
    fig.update_layout(
        yaxis=dict(
            title=f'{var}'
        ),
        yaxis2=dict(
            title='REVENUE',
            overlaying='y',
            side='right'
        ),
        xaxis=dict(
            title='WEEK_END_DATE'
        ),
        title={
            'text': var.lower().capitalize() + ' and Revenue Over Time',
            'x': 0.3, 
            'y': 0.9  
        },
        legend=dict(
            x=0,
            y=1.1,
            orientation='h'
        )
    )
    st.plotly_chart(fig)

def calculate_regression(subset):
    if subset['PRICE'].std() == 0 or subset['UNIT_SALES'].std() == 0:
        slope = 0
        intercept = subset['UNIT_SALES'].mean() if subset['PRICE'].std() == 0 else subset['PRICE'].mean()
        r_value = p_value = std_err = 0
        line = np.full_like(subset['PRICE'], intercept)
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(subset['PRICE'], subset['UNIT_SALES'])
        line = slope * subset['PRICE'] + intercept
    return slope, intercept, r_value, p_value, std_err, line
 
# To visualize scatter plot of Price and Unit sales
def plot_price_unit_sales(store_id, upc):
    subset = get_store_upc_data(store_id, upc)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subset['PRICE'],
        y=subset['UNIT_SALES'],
        mode='markers',
        marker=dict(color='#778da9', size=10, opacity=0.7),
        text=subset.apply(lambda row: f"PRICE: {row['PRICE']}<br>UNIT_SALES: {row['UNIT_SALES']}", axis=1),
        hoverinfo='text',
        showlegend=False
    ))
    slope, intercept, r_value, p_value, std_err, line = calculate_regression(subset)

    fig.add_trace(go.Scatter(
        x=subset['PRICE'],
        y=line,
        mode='lines',
        line=dict(color='#AF3247', width=2),
        name=f'Regression Line (R²={r_value**2:.2f})'
    ))
    fig.update_layout(
        xaxis_title='PRICE',
        yaxis_title='UNIT_SALES',
        title={
            'text': 'Price vs. Unit Sales',
            'x': 0.4, 
            'y': 0.9  
        },
        legend=dict(
            x=0,
            y=1.1,
            orientation='h'
        ),
        hovermode='closest',
        width=800, 
        height=500, 
    )
    st.plotly_chart(fig)

# To visualize price distribution
def plot_price_distribution(store_id, upc):
    subset = get_store_upc_data(store_id, upc)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=subset['PRICE'],
        marker_color='#9a8c98',
        opacity=0.7,
    ))
    fig.update_layout(
        xaxis_title='PRICE',
        title={
            'text': 'Price Distribution',
            'x': 0.4, 
            'y': 0.9  
        },
    )
    st.plotly_chart(fig)


# To show info in card format
def create_card(title, content, background_color='#f0f0f0'):
    st.markdown(f"""
    <div style="background-color: {background_color}; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 12px #aaaaaa; margin: 20px 0; height: 210px;">
        <h2 style="color: #333;">{title}</h2>
        <p style="color: #555; font-size: 18px;">{content}</p>
    </div>
    """, unsafe_allow_html=True)

# To visualize price and revenue changes
def visualize_change_pct(change_percent):
    if change_percent > 0:
        return f"<span style='font-size: larger; color:green'>🔺 {change_percent:.2f} %</span>"  
    elif change_percent < 0:
        return f"<span style='font-size: larger; color:red'>🔻 {abs(change_percent):.2f} %</span>"
    else:
        return f"<span style='font-size: larger;'>No Change</span>"

# ------------------------------------------------------------------------------------------------------
#                                   Main Page and Sidebar Configuration
# ------------------------------------------------------------------------------------------------------

#-----------------------------------  Main Page --------------------------------------------------------
st.title('Pricing Desision Support System for Fast-Moving Consumer Goods (FMCG) 💲')
st.markdown("<br>", unsafe_allow_html=True)
st.header("About the App")
st.markdown("""
    **This application provides the following data-driven insights for FMCG:** 
    - Forecast next week's sales
    - Estimate price elasticity of demand
    - Recommend the optimal price to maximize revenue. The optimal price will be within **±20%** of your inputted price
""")

st.header("How To Use")
st.markdown("""
    **Please provide the following inputs in the sidebar to see store, product, and price elasticity details:**
    - Optionally, select one or more store segments to filter Store IDs
    - Select a Store ID (mandatory)
    - Optionally, select one or more product categories to filter UPCs
    - Select a UPC (mandatory)
            
    **To get next week's sales forecast and the optimal price, you MUST:**
    - Specify the price within the given range
    - Select feature and display options
""")
st.divider()


#-----------------------------------  Sidebar ------------------------------------------------------------
st.sidebar.header("Input Your Data")
store_seg_options = st.sidebar.multiselect("**Store Segment**", ['VALUE', 'MAINSTREAM', 'UPSCALE'], placeholder="Select Store Segment")
st.sidebar.markdown("<br>", unsafe_allow_html=True)
store_id = st.sidebar.selectbox("**Store ID**", get_store_id(store_seg_options), index=None, placeholder = "Select Store ID")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

forecast = False
if store_id:
    sub_category_options = st.sidebar.multiselect("**Product Category**", sorted(product['SUB_CATEGORY'].unique()), placeholder="Select Product Category")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    upc = st.sidebar.selectbox("**Universal Product Code (UPC)**", get_upcs(store_id, sub_category_options), index=None, placeholder="Select UPC")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    if store_id and upc:  
        # Adjust price slider based on historical price range
        min_price, max_price, median_price = get_price_range(store_id, upc)
        price = st.sidebar.slider("**Price**", value=median_price, min_value=min_price, max_value=max_price, step=0.01)
        price = np.round(price, 2)
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        
        feature_str = st.sidebar.selectbox("**Feature**", ["No", "Yes"], index=None, placeholder="Select Feature")
        st.sidebar.caption("Whether the product is featured in promotions or actively advertised")
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        feature = None
        if feature_str == "Yes":
            feature = 1
        elif feature_str == "No":
            feature = 0

        display_str = st.sidebar.selectbox("**Display**", ["No", "Yes"], index=None, placeholder="Select Display")
        st.sidebar.caption("Whether the product is placed on shelves or end caps where it is easily noticeable")
        display = None
        if display_str == "Yes":
            display = 1
        elif display_str == "No":
            display = 0
        if feature in [0, 1] and display in [0, 1]:
            forecast = True

# ------------------------------------------------------------------------------------------------------
#                                   Display Historical Data
# ------------------------------------------------------------------------------------------------------

if store_id and upc:
    # Show store and product details 
    store_details_df = get_store_details(store_id)
    product_details_df = get_produt_details(upc)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏪 Store Details")
        st.dataframe(store_details_df) 
    with col2:
        st.subheader("🛍️ Product Details")
        st.dataframe(product_details_df) 
    st.markdown("<br>", unsafe_allow_html=True)

    # Show price elasticity details
    elasticity_value = get_store_upc_elasticity(store_id, upc)
    st.subheader("Price Elasticity")
    st.markdown("""
        Measurement of how sensitive demand in response to price changes. 
        - **Inelastic:** Demand less sensitvie to price changes (elasticity < 1).
        - **Elastic:** Demand more sensitive to price changes (elasticity > 1).
    """)
    st.image("price_elasticity_formula1.png", caption="Price Elasticity Formula")
    show_elasticity_info(elasticity_value)
    st.markdown("<br>", unsafe_allow_html=True)

    # Visualize historical data
    st.subheader("📊 Visualizing Historical Trends: Price, Unit Sales, and Revenue")
    # Graph 1: Time series plot of unit sales and revenue 
    plot_unit_sales_revenue(store_id, upc, 'UNIT_SALES')
    # Graph 2:Time series plot of PRICE and revenue 
    plot_unit_sales_revenue(store_id, upc, 'PRICE')
    # Graph 3:Scatter plot price(x) vs unit sales(y)
    plot_price_unit_sales(store_id, upc)
    # Graph 4:Price distribution histogram
    plot_price_distribution(store_id, upc)
    st.divider()

# ------------------------------------------------------------------------------------------------------
#                                   Demand Forecasting
# ------------------------------------------------------------------------------------------------------

if forecast:
    st.header('Next Week Sales & Optimal Pricing Guide')
    st.markdown("<br>", unsafe_allow_html=True)

    user_input = []
    user_input.append({
            'STORE_ID': store_id,
            'UPC': upc,
            'PRICE': price,
            'FEATURE': feature,
            'DISPLAY': display})
    user_input_df = pd.DataFrame(user_input)
    processed_input = processing_pipeline(user_input_df)
    pred_unit_sales = forecast_unit_sales(processed_input)

    col1, col2, col3 = st.columns(3)
    with col1:
        create_card(f"{price:.2f}", 'Current/Base Price')
    with col2:
        create_card(pred_unit_sales, 'Forecasted Unit Sales in Next Week')
    with col3:
        revenue = price * pred_unit_sales
        create_card(f"{revenue:.2f}", 'Estimated Revenue in Next Week')
    
# ------------------------------------------------------------------------------------------------------
#                                   Price Optimization
# ------------------------------------------------------------------------------------------------------
  
    normalized_price_action = optimize_price(store_id, upc, feature, display, elasticity_value, pred_unit_sales)
    price_min_bound = price * 0.8
    price_max_bound = price * 1.2
    opt_price = normalized_price_action * (price_max_bound - price_min_bound) + price_min_bound
    opt_price = np.round(opt_price, 2)

    col1, col2, col3 = st.columns(3)
    with col1:
        create_card(f"{opt_price:.2f}", 'Optimal Price', '#DEEFF5')
    with col2:
        user_input_df['PRICE'] = opt_price
        processed_input = processing_pipeline(user_input_df)
        opt_unit_sales = forecast_unit_sales(processed_input)
        create_card(opt_unit_sales, 'Forecasted Unit Sales with Optimal Price', '#DEEFF5')
    with col3:
        opt_revenue = opt_price * opt_unit_sales
        create_card(f"{opt_revenue:.2f}", 'Estimated Revenue with Optimal Price', '#DEEFF5')

    # Show price and revenue changes
    price_changes_pct = (opt_price - price)/price * 100
    revenue_changes_pct = (opt_revenue - revenue)/revenue * 100
    st.markdown(f"**Price:** {visualize_change_pct(price_changes_pct)}", unsafe_allow_html=True)
    st.markdown(f"**Revenue:** {visualize_change_pct(revenue_changes_pct)}", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------------
#                                   Feature Importance for Demand Forecasting
# ------------------------------------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.subheader("**Feature Importance**")
    st.caption("""
        Table below shows the key factors (features) used in the demand forecasting model and their relative importance. 
        The higher the importance score, the more influence the factor has on the forecast.
    """)
    processed_input_transposed = processed_input.transpose()
    processed_input_transposed.columns = ['Value']
    final_df = show_feature_importance(processed_input).set_index('Feature').join(processed_input_transposed)
    st.dataframe(final_df[[ 'Value', 'Importance Score']], use_container_width=True)


# ------------------------------------------------------------------------------------------------------
#                                   Additional Info about Project
# ------------------------------------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
show_info = st.button("Learn More About This Project")
if show_info:
    st.header('About the Dataset')
    st.markdown("""
        **Weekly transaction data of FMCG**:
        - **Data period**: January 14, 2009 to January 4, 2012 (156 weeks).
        - **Coverage**: 77 Stores, 7 Product Categories, 37 UPCs 
        - **Note**: This project considers Unit Sales as synonymous with Demand(Quantity Demanded) 
    """)
    st.header('Machine Learning Implementation')
    st.markdown("""
        This application leverages advanced machine learning techniques:
        - **Demand Forecasting**: Using supervised learning with XGBoost Regression algorithm
        - **Price Elasticity Estimation**: Using double machine learning with XGBoost Regression and Linear Regression algorithms
        - **Price Optimization**: Using reinforcement learning with Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm
    """)
