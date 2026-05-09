import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. Page Configuration
st.set_page_config(
    page_title="AI HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# 2. Data Loading Function
@st.cache_data
def load_data():
    # Check if the file exists locally
    file_name = "hr_data.csv"
    if not os.path.exists(file_name):
        # Fallback for the common Databricks export name
        file_name = "HR-Employee-Attrition.csv"
        
    try:
        df = pd.read_csv(file_name)
        return df
    except FileNotFoundError:
        return None

# Load the dataset
df = load_data()

if df is None:
    st.error("❌ Error: The data file was not found!")
    st.info("Please ensure your CSV file is in the same folder as 'app.py' and named 'hr_data.csv'.")
else:
    # --- Sidebar ---
    st.sidebar.title("Dashboard Controls")
    st.sidebar.markdown("Filter the data to explore different segments.")
    
    # Department Filter
    all_depts = df['Department'].unique().tolist()
    selected_depts = st.sidebar.multiselect("Select Departments", all_depts, default=all_depts)
    
    # Gender Filter
    gender_filter = st.sidebar.multiselect("Select Gender", df['Gender'].unique(), default=df['Gender'].unique())

    # Apply Filters
    filtered_df = df[(df['Department'].isin(selected_depts)) & (df['Gender'].isin(gender_filter))]

    # --- Header ---
    st.title("📊 HR AI Analytics & Employee Attrition")
    st.markdown(f"**Analyzing data for {len(filtered_df)} employees**")
    st.divider()

    # --- Key Performance Indicators (KPIs) ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    avg_salary = filtered_df['MonthlyIncome'].mean()
    attrition_rate = (filtered_df['Attrition'] == 'Yes').mean() * 100
    
    kpi1.metric("Avg Monthly Income", f"${avg_salary:,.2f}")
    kpi2.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    kpi3.metric("Avg Age", f"{filtered_df['Age'].mean():.1f} Years")
    kpi4.metric("Avg Job Satisfaction", f"{filtered_df['JobSatisfaction'].mean():.2f} / 4")

    st.divider()

    # --- Row 1: Visualizations ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Attrition Distribution by Department")
        fig1 = px.histogram(
            filtered_df, 
            x="Department", 
            color="Attrition", 
            barmode="group",
            color_discrete_map={'Yes': '#E74C3C', 'No': '#2E86C1'}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("💰 Income vs. Performance Rating")
        fig2 = px.box(
            filtered_df, 
            x="PerformanceRating", 
            y="MonthlyIncome", 
            color="PerformanceRating",
            points="all"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Row 2: Visualizations ---
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🕒 Overtime Impact on Attrition")
        fig3 = px.sunburst(
            filtered_df, 
            path=['OverTime', 'Attrition'], 
            values='EmployeeCount',
            color='OverTime'
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("⚖️ Work-Life Balance vs. Monthly Income")
        fig4 = px.violin(
            filtered_df, 
            x="WorkLifeBalance", 
            y="MonthlyIncome", 
            color="Attrition", 
            box=True,
            points="all"
        )
        st.plotly_chart(fig4, use_container_width=True)

    # --- Data Explorer ---
    with st.expander("🔍 View Raw Filtered Data"):
        st.dataframe(filtered_df.head(100))

    st.markdown("---")
    st.caption("AI-Powered HR Analytics | Developed for University Project") 

    # --- Prediction Summary Section ---
st.header("🔮 Employee Retention Forecast")
st.markdown("Summary of employees predicted to stay vs. those at risk of leaving.")

# حساب الأعداد بناءً على البيانات
stay_count = len(filtered_df[filtered_df['Attrition'] == 'No'])
leave_count = len(filtered_df[filtered_df['Attrition'] == 'Yes'])

# عرض النتائج في شكل أعمدة ملونة
c_stay, c_leave = st.columns(2)

with c_stay:
    st.success(f"### ✅ Predicted to Stay")
    st.title(f"{stay_count}")
    st.write("Employees likely to remain with the company.")

with c_leave:
    st.error(f"### ⚠️ At Risk of Leaving")
    st.title(f"{leave_count}")
    st.write("Employees predicted to leave (Attrition).")

# إضافة رسم بياني دائري سريع للمقارنة
st.subheader("Retention Overview")
fig_pie_summary = px.pie(
    values=[stay_count, leave_count], 
    names=['Will Stay', 'Will Leave'],
    color=['Will Stay', 'Will Leave'],
    color_discrete_map={'Will Stay': '#2ECC71', 'Will Leave': '#E74C3C'},
    hole=0.5
)
st.plotly_chart(fig_pie_summary, use_container_width=True)