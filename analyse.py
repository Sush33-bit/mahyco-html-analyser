import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import io
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Field Activity Data Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E7D32;
    }
    .insight-box {
        background-color: Black;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def parse_html_to_dataframe(html_content):
    """Parse HTML content and convert to DataFrame"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all tables
        tables = soup.find_all('table')
        
        if not tables:
            st.error("No tables found in the HTML file")
            return None
        
        # Use the first table (or you can modify this logic)
        table = tables[0]
        
        # Extract headers
        headers = []
        header_row = table.find('tr')
        if header_row:
            for th in header_row.find_all(['th', 'td']):
                headers.append(th.get_text(strip=True))
        
        # Extract data rows
        data = []
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cols = row.find_all(['td', 'th'])
            row_data = []
            for col in cols:
                row_data.append(col.get_text(strip=True))
            if row_data:  # Only add non-empty rows
                data.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=headers[:len(data[0])] if data else headers)
        
        return df
    
    except Exception as e:
        st.error(f"Error parsing HTML: {str(e)}")
        return None

def clean_and_process_data(df):
    """Clean and process the DataFrame"""
    if df is None:
        return None
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Display actual columns for debugging
    st.write("**Detected Columns:**", list(df.columns))
    
    # Try to convert numeric columns (check if they exist first)
    potential_numeric_columns = ['SRNO', 'MOBILE NO', 'KM BY KA', 'KM BY SYSTEM', 'ATTENDANCE', 
                                'GTV1 PUNCH IN', 'GTV1 PUNCH OUT', 'GTV1 TIME SPENT', 'GTV1 VILLAGE ACTIVITY COUNT',
                                'GTV2 PUNCH IN', 'GTV2 PUNCH OUT', 'GTV2 TIME SPENT', 'GTV2 VILLAGE ACTIVITY COUNT',
                                'MARKET ACTIVITY COUNT']
    
    # Also check for columns that might be numeric based on content
    for col in df.columns:
        # Check if column contains mostly numeric data
        if df[col].dtype == 'object':
            numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
            if numeric_count > len(df) * 0.5:  # If more than 50% are numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert time columns if they exist
    potential_time_columns = ['START TIME', 'END TIME']
    for col in potential_time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

def display_basic_stats(df):
    """Display basic statistics"""
    st.subheader("📈 Basic Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        # Find KA name column (might have different exact name)
        ka_col = None
        for col in df.columns:
            if 'KA' in col.upper() and 'NAME' in col.upper():
                ka_col = col
                break
        unique_kas = df[ka_col].nunique() if ka_col else 0
        st.metric("Unique KAs", unique_kas)
    
    with col3:
        # Find territory column
        territory_col = None
        for col in df.columns:
            if 'TERRITORY' in col.upper():
                territory_col = col
                break
        unique_territories = df[territory_col].nunique() if territory_col else 0
        st.metric("Territories", unique_territories)
    
    with col4:
        # Find village column
        village_col = None
        for col in df.columns:
            if 'VILLAGE' in col.upper() and 'GTV1' in col.upper():
                village_col = col
                break
        unique_villages = df[village_col].nunique() if village_col else 0
        st.metric("Villages Covered", unique_villages)

def create_geographical_analysis(df):
    """Create geographical analysis"""
    st.subheader("🗺️ Geographical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Find zone column
        zone_col = None
        for col in df.columns:
            if 'ZONE' in col.upper():
                zone_col = col
                break
        
        if zone_col:
            zone_counts = df[zone_col].value_counts()
            fig = px.pie(values=zone_counts.values, names=zone_counts.index, 
                        title="Distribution by Zone")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Zone column not found in data")
    
    with col2:
        # Find region column
        region_col = None
        for col in df.columns:
            if 'REGION' in col.upper():
                region_col = col
                break
        
        if region_col:
            region_counts = df[region_col].value_counts().head(10)
            fig = px.bar(x=region_counts.values, y=region_counts.index, 
                        orientation='h', title="Top 10 Regions by Activity")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Region column not found in data")

def create_activity_analysis(df):
    """Create activity analysis"""
    st.subheader("⚡ Activity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Find GTV1 activity column
        gtv1_activity_col = None
        for col in df.columns:
            if 'GTV1' in col.upper() and 'ACTIVITY' in col.upper() and 'COUNT' in col.upper():
                gtv1_activity_col = col
                break
        
        if gtv1_activity_col:
            fig = go.Figure(data=go.Histogram(x=df[gtv1_activity_col], 
                                            name="GTV1 Activities"))
            fig.update_layout(title="Distribution of GTV1 Village Activities",
                            xaxis_title="Activity Count",
                            yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("GTV1 activity count column not found")
    
    with col2:
        # Find GTV2 activity column
        gtv2_activity_col = None
        for col in df.columns:
            if 'GTV2' in col.upper() and 'ACTIVITY' in col.upper() and 'COUNT' in col.upper():
                gtv2_activity_col = col
                break
        
        if gtv2_activity_col:
            fig = go.Figure(data=go.Histogram(x=df[gtv2_activity_col], 
                                            name="GTV2 Activities"))
            fig.update_layout(title="Distribution of GTV2 Village Activities",
                            xaxis_title="Activity Count",
                            yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("GTV2 activity count column not found")

def create_time_analysis(df):
    """Create time-based analysis"""
    st.subheader("⏰ Time Analysis")
    
    # Find time spent columns
    gtv1_time_col = None
    gtv2_time_col = None
    
    for col in df.columns:
        if 'GTV1' in col.upper() and 'TIME' in col.upper() and 'SPENT' in col.upper():
            gtv1_time_col = col
        elif 'GTV2' in col.upper() and 'TIME' in col.upper() and 'SPENT' in col.upper():
            gtv2_time_col = col
    
    if gtv1_time_col or gtv2_time_col:
        col1, col2 = st.columns(2)
        
        with col1:
            if gtv1_time_col:
                fig = px.box(df, y=gtv1_time_col, title="GTV1 Time Spent Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("GTV1 time spent column not found")
        
        with col2:
            if gtv2_time_col:
                fig = px.box(df, y=gtv2_time_col, title="GTV2 Time Spent Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("GTV2 time spent column not found")
    else:
        st.info("No time spent columns found in the data")

def create_performance_analysis(df):
    """Create performance analysis"""
    st.subheader("🎯 Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Find attendance column
        attendance_col = None
        for col in df.columns:
            if 'ATTENDANCE' in col.upper():
                attendance_col = col
                break
        
        if attendance_col:
            fig = px.histogram(df, x=attendance_col, title="Attendance Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Attendance column not found")
    
    with col2:
        # Find KM columns
        km_by_ka_col = None
        km_by_system_col = None
        
        for col in df.columns:
            if 'KM' in col.upper() and 'KA' in col.upper() and 'SYSTEM' not in col.upper():
                km_by_ka_col = col
            elif 'KM' in col.upper() and 'SYSTEM' in col.upper():
                km_by_system_col = col
        
        if km_by_ka_col and km_by_system_col:
            fig = px.scatter(df, x=km_by_ka_col, y=km_by_system_col, 
                           title="KM by KA vs KM by System")
            # Add diagonal line
            max_val = max(df[km_by_ka_col].max(), df[km_by_system_col].max())
            fig.add_shape(type="line", x0=0, y0=0, 
                         x1=max_val, y1=max_val,
                         line=dict(dash="dash", color="red"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("KM comparison columns not found")

def create_ka_performance_ranking(df):
    """Create KA performance ranking"""
    st.subheader("🏆 KA Performance Ranking")
    
    # Find KA name column
    ka_name_col = None
    for col in df.columns:
        if 'KA' in col.upper() and 'NAME' in col.upper():
            ka_name_col = col
            break
    
    if not ka_name_col:
        st.info("KA Name column not found - cannot create performance ranking")
        return
    
    # Find relevant columns for performance calculation
    gtv1_activity_col = None
    gtv2_activity_col = None
    attendance_col = None
    km_col = None
    
    for col in df.columns:
        if 'GTV1' in col.upper() and 'ACTIVITY' in col.upper() and 'COUNT' in col.upper():
            gtv1_activity_col = col
        elif 'GTV2' in col.upper() and 'ACTIVITY' in col.upper() and 'COUNT' in col.upper():
            gtv2_activity_col = col
        elif 'ATTENDANCE' in col.upper():
            attendance_col = col
        elif 'KM' in col.upper() and 'KA' in col.upper() and 'SYSTEM' not in col.upper():
            km_col = col
    
    # Build aggregation dictionary with available columns
    agg_dict = {}
    if gtv1_activity_col:
        agg_dict[gtv1_activity_col] = 'sum'
    if gtv2_activity_col:
        agg_dict[gtv2_activity_col] = 'sum'
    if attendance_col:
        agg_dict[attendance_col] = 'mean'
    if km_col:
        agg_dict[km_col] = 'sum'
    
    if not agg_dict:
        st.info("No performance metrics found for ranking")
        return
    
    # Calculate performance metrics by KA
    ka_performance = df.groupby(ka_name_col).agg(agg_dict).round(2)
    
    # Calculate composite score based on available columns
    score_components = []
    weights = []
    
    if gtv1_activity_col in ka_performance.columns:
        score_components.append(ka_performance[gtv1_activity_col])
        weights.append(0.3)
    if gtv2_activity_col in ka_performance.columns:
        score_components.append(ka_performance[gtv2_activity_col])
        weights.append(0.3)
    if attendance_col in ka_performance.columns:
        score_components.append(ka_performance[attendance_col])
        weights.append(0.2)
    if km_col in ka_performance.columns:
        score_components.append(ka_performance[km_col])
        weights.append(0.2)
    
    # Normalize weights
    weights = [w/sum(weights) for w in weights]
    
    # Calculate composite score
    if score_components:
        composite_score = sum(comp * weight for comp, weight in zip(score_components, weights))
        ka_performance['Composite_Score'] = composite_score
        ka_performance = ka_performance.sort_values('Composite_Score', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(ka_performance.head(10), use_container_width=True)
        
        with col2:
            top_performers = ka_performance.head(5)
            fig = px.bar(x=top_performers.index, y=top_performers['Composite_Score'],
                        title="Top 5 KA Performance")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Unable to calculate composite performance score")

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    st.subheader("🔥 Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(correlation_matrix, 
                       color_continuous_scale='RdBu_r',
                       title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

def generate_insights(df):
    """Generate key insights"""
    st.subheader("💡 Key Insights")
    
    insights = []
    
    # Territory coverage insight
    territory_col = None
    for col in df.columns:
        if 'TERRITORY' in col.upper():
            territory_col = col
            break
    
    if territory_col:
        territory_coverage = df[territory_col].nunique()
        insights.append(f"📍 Coverage spans across {territory_coverage} territories")
    
    # Activity insights
    gtv1_activity_col = None
    gtv2_activity_col = None
    
    for col in df.columns:
        if 'GTV1' in col.upper() and 'ACTIVITY' in col.upper() and 'COUNT' in col.upper():
            gtv1_activity_col = col
        elif 'GTV2' in col.upper() and 'ACTIVITY' in col.upper() and 'COUNT' in col.upper():
            gtv2_activity_col = col
    
    if gtv1_activity_col:
        avg_gtv1 = df[gtv1_activity_col].mean()
        insights.append(f"📊 Average GTV1 village activities per record: {avg_gtv1:.2f}")
    
    if gtv2_activity_col:
        avg_gtv2 = df[gtv2_activity_col].mean()
        insights.append(f"📊 Average GTV2 village activities per record: {avg_gtv2:.2f}")
    
    # Attendance insight
    attendance_col = None
    for col in df.columns:
        if 'ATTENDANCE' in col.upper():
            attendance_col = col
            break
    
    if attendance_col:
        avg_attendance = df[attendance_col].mean()
        insights.append(f"👥 Average attendance rate: {avg_attendance:.2f}")
    
    # Distance insights
    km_by_ka_col = None
    km_by_system_col = None
    
    for col in df.columns:
        if 'KM' in col.upper() and 'KA' in col.upper() and 'SYSTEM' not in col.upper():
            km_by_ka_col = col
        elif 'KM' in col.upper() and 'SYSTEM' in col.upper():
            km_by_system_col = col
    
    if km_by_ka_col and km_by_system_col:
        correlation = df[km_by_ka_col].corr(df[km_by_system_col])
        insights.append(f"🚗 Correlation between KA reported KM and system KM: {correlation:.2f}")
    
    # Display insights
    if insights:
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    else:
        st.info("No specific insights could be generated with the available columns")

def main():
    st.markdown('<h1 class="main-header">📊 Field Activity Data Insights Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📁 File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your HTML file",
        type=['html', 'htm'],
        help="Upload the HTML file containing your 287 records"
    )
    
    if uploaded_file is not None:
        # Read and parse HTML
        html_content = uploaded_file.read().decode('utf-8')
        
        with st.spinner("Parsing HTML file..."):
            df = parse_html_to_dataframe(html_content)
        
        if df is not None:
            # Clean and process data
            df = clean_and_process_data(df)
            
            # Display data info
            st.success(f"✅ Successfully loaded {len(df)} records!")
            
            # Sidebar options
            st.sidebar.title("📊 Analysis Options")
            show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
            show_basic_stats = st.sidebar.checkbox("Basic Statistics", value=True)
            show_geo_analysis = st.sidebar.checkbox("Geographical Analysis", value=True)
            show_activity_analysis = st.sidebar.checkbox("Activity Analysis", value=True)
            show_time_analysis = st.sidebar.checkbox("Time Analysis", value=True)
            show_performance = st.sidebar.checkbox("Performance Analysis", value=True)
            show_ranking = st.sidebar.checkbox("KA Ranking", value=True)
            show_correlation = st.sidebar.checkbox("Correlation Analysis", value=True)
            show_insights = st.sidebar.checkbox("Key Insights", value=True)
            
            # Display selected analyses
            if show_raw_data:
                st.subheader("📄 Raw Data")
                st.dataframe(df, use_container_width=True)
                
                # Download processed data
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Processed Data as CSV",
                    data=csv,
                    file_name="processed_field_data.csv",
                    mime="text/csv"
                )
            
            if show_basic_stats:
                display_basic_stats(df)
            
            if show_geo_analysis:
                create_geographical_analysis(df)
            
            if show_activity_analysis:
                create_activity_analysis(df)
            
            if show_time_analysis:
                create_time_analysis(df)
            
            if show_performance:
                create_performance_analysis(df)
            
            if show_ranking:
                create_ka_performance_ranking(df)
            
            if show_correlation:
                create_correlation_heatmap(df)
            
            if show_insights:
                generate_insights(df)
            
        else:
            st.error("Failed to parse the HTML file. Please check the file format.")
    
    else:
        st.info("👆 Please upload your HTML file to begin analysis")
        
        # Show sample expected format
        st.subheader("📋 Expected HTML Format")
        st.write("Your HTML file should contain a table with the following columns:")
        expected_columns = [
            "SRNO", "ZONE", "REGION", "TERRITORY", "APPROVED HQ", "KA NAME", 
            "KA CODE", "ROLE", "MOBILE NO", "START TIME", "END TIME", 
            "KM BY KA", "KM BY SYSTEM", "ATTENDANCE", "GTV1 VILLAGE",
            "GTV1 PUNCH IN", "GTV1 PUNCH OUT", "GTV1 TIME SPENT", 
            "GTV1 VILLAGE ACTIVITY COUNT", "GTV2 VILLAGE", "GTV2 PUNCH IN",
            "GTV2 PUNCH OUT", "GTV2 TIME SPENT", "GTV2 VILLAGE ACTIVITY COUNT", 
            "MARKET ACTIVITY COUNT"
        ]
        st.write(", ".join(expected_columns))

if __name__ == "__main__":
    main()