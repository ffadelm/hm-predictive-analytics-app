import streamlit as st
import pandas as pd
import calendar
import joblib
import plotly.express as px
from sqlalchemy import create_engine

# Setup database connection
engine = create_engine("postgresql://postgres:admin@localhost:5432/hmsales_dw")

st.set_page_config(layout="wide")
st.title("📊 H&M Sales Data Warehouse Dashboard")

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")
year_filter = st.sidebar.multiselect(
    "Select Year", 
    pd.read_sql("SELECT DISTINCT year FROM dimension_date ORDER BY year", engine)["year"]
)
category_filter = st.sidebar.multiselect(
    "Select Product Category", 
    pd.read_sql("SELECT DISTINCT category FROM dimension_product ORDER BY category", engine)["category"]
)
region_filter = st.sidebar.multiselect(
    "Select Region", 
    pd.read_sql("SELECT DISTINCT region FROM dimension_location ORDER BY region", engine)["region"]
)

# --- Main Query ---
query = "SELECT * FROM v_sales_summary"
df = pd.read_sql(query, engine)

# --- Apply Filters ---
if year_filter:
    df = df[df["year"].isin(year_filter)]
if category_filter:
    df = df[df["category"].isin(category_filter)]
if region_filter:
    df = df[df["region"].isin(region_filter)]

# --- Handle empty filtered data ---
if df.empty:
    st.warning("No data found for selected filters.")
    st.stop()

# --- KPI Section ---
st.subheader("📌 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"${df['sales'].sum():,.0f}")
col2.metric("Total Profit", f"${df['profit'].sum():,.0f}")
col3.metric("Total Orders", f"{df['row_id'].nunique():,}")
col4.metric("Avg Profit Margin", f"{df['profit_margin_pct'].mean():.2f}%")

# --- Sales by Product Category ---
st.subheader("📈 Sales by Product Category")
sales_by_category = df.groupby("category")["sales"].sum().reset_index().sort_values("sales", ascending=False)
fig_category = px.bar(sales_by_category, x="category", y="sales", color="category", text_auto=True)
st.plotly_chart(fig_category, use_container_width=True)

# --- Monthly Sales Trend ---
st.subheader("📆 Monthly Sales Trend")
selected_year = year_filter[0] if year_filter else df["year"].iloc[0]
full_months = pd.DataFrame({
    "month_number": list(range(1, 13)),
    "month_name": [calendar.month_name[i] for i in range(1, 13)],
    "year": [selected_year] * 12
})
monthly_sales = df[df["year"] == selected_year].groupby("month_name")["sales"].sum().reset_index()
monthly_sales["month_number"] = pd.to_datetime(monthly_sales["month_name"], format="%B").dt.month
merged = pd.merge(full_months, monthly_sales, on=["month_number", "month_name"], how="left")
merged["sales"] = merged["sales"].fillna(0)
merged["date_order"] = pd.to_datetime(merged["year"] + "-" + merged["month_number"].astype(str) + "-01")
merged = merged.sort_values("date_order")
merged.set_index("date_order", inplace=True)
st.line_chart(merged["sales"])

# --- Top 5 Cities by Sales ---
st.subheader("🌍 Top 5 Cities by Sales")
top_cities = (
    df.groupby(["city", "state"])["sales"]
    .sum()
    .reset_index()
    .sort_values("sales", ascending=False)
    .head(5)
)
top_cities["city_state"] = top_cities["city"] + ", " + top_cities["state"]
fig_top_cities = px.bar(top_cities, x="city_state", y="sales", color="city_state", text_auto=True)
st.plotly_chart(fig_top_cities, use_container_width=True)

# --- Average Profit Margin by Category ---
st.subheader("📦 Average Profit Margin by Category")
avg_margin = df.groupby("category")["profit_margin_pct"].mean().reset_index().sort_values("profit_margin_pct", ascending=False)
fig_margin = px.bar(avg_margin, x="category", y="profit_margin_pct", color="category", text_auto=True)
st.plotly_chart(fig_margin, use_container_width=True)

# --- Loss Analysis Section ---
st.subheader("📉 Loss Analysis")
col5, col6 = st.columns(2)
total_tx = len(df)
loss_tx = len(df[df["profit"] < 0])
loss_pct = round((loss_tx / total_tx) * 100, 2)
col5.metric("Loss Transactions", f"{loss_tx} of {total_tx}")
col6.metric("% Loss Transactions", f"{loss_pct}%")

st.markdown("#### 💥 Total Loss by Category")
loss_by_cat = df[df["profit"] < 0].groupby("category")["profit"].sum().reset_index().sort_values("profit")
fig_loss_cat = px.bar(loss_by_cat, x="category", y="profit", color="category", text_auto=True)
st.plotly_chart(fig_loss_cat, use_container_width=True)

st.markdown("#### 🗺️ Loss by Region")
loss_by_region = df[df["profit"] < 0].groupby("region")["profit"].sum().reset_index().sort_values("profit")
fig_loss_region = px.bar(loss_by_region, x="region", y="profit", color="region", text_auto=True)
st.plotly_chart(fig_loss_region, use_container_width=True)

st.markdown("#### 🎯 Discount vs Average Profit")
discount_profit = df.groupby("discount")["profit"].mean().reset_index().sort_values("discount")
st.line_chart(discount_profit.set_index("discount"))

# --- Predictive Analytics Section ---
st.subheader("🤖 Profit Prediction")

st.markdown("""
Enter the transaction details below to predict whether this transaction will result in **profit** or **loss** using the machine learning model.
""")

model = joblib.load("model.pkl")

# Capture selected category outside form to make sub-category reactive
category = st.selectbox("📁 Category", df["category"].unique(), key="category_selector")
filtered_subcategories = df[df["category"] == category]["sub_category"].unique()

with st.form("prediction_form"):
    # Row 1: Discount
    discount = st.slider("🎯 Discount (%)", 0.0, 0.9, 0.1, step=0.05)

    # Row 2: Quantity and Sub-category (sub-category reacts to selected category)
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        quantity = st.number_input("📦 Quantity", min_value=1, max_value=20, value=1)
    with row2_col2:
        sub_category = st.selectbox("📂 Sub-category", filtered_subcategories, key="sub_category_selector")

    # Row 3: Shipping Mode and Region
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        ship_mode = st.selectbox("🚚 Shipping Mode", df["ship_mode"].unique())
    with row3_col2:
        region = st.selectbox("🌍 Region", df["region"].unique())

    # Row 4: Sales
    sales = st.number_input("💵 Sales Amount ($)", min_value=1.0, value=100.0)

    submit = st.form_submit_button("🔍 Predict Profit")

if submit:
    input_dict = {
        "discount": [discount],
        "quantity": [quantity],
        "sales": [sales],
        f"category_{category}": [1],
        f"sub_category_{sub_category}": [1],
        f"ship_mode_{ship_mode}": [1],
        f"region_{region}": [1]
    }
    all_features = model.feature_names_in_
    for feature in all_features:
        if feature not in input_dict:
            input_dict[feature] = [0]
    input_df = pd.DataFrame(input_dict)
    input_df = input_df.reindex(columns=all_features, fill_value=0)
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Prediction: PROFITABLE\n\nThis transaction is likely to generate **profit** with a confidence of **{prob*100:.2f}%**.")
    else:
        st.error(f"🚨 Prediction: LOSS\n\nThis transaction is predicted to result in a **loss** with a confidence of **{prob*100:.2f}%**. Consider adjusting discount, price, or shipping method.")

    with st.expander("📋 Show Input Details"):
        st.json({
            "discount": discount,
            "quantity": quantity,
            "sales": sales,
            "category": category,
            "sub_category": sub_category,
            "ship_mode": ship_mode,
            "region": region
        })

# --- Show Raw Data ---
with st.expander("📄 Show Raw Data"):
    st.dataframe(df)
    st.download_button(
        label="Download Data as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='sales_data.csv',
        mime='text/csv'
    )
