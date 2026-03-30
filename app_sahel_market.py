import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Sahel Food Market Dynamics",
    page_icon="📈",
    layout="wide"
)

SAHEL_COUNTRIES = ["Burkina Faso", "Mali", "Niger", "Chad", "Mauritania", "Nigeria"]
DEFAULT_FOOD = "Millet"
DEFAULT_WAGE = "Wage (non-qualified labour, agricultural) - Retail"

@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["mp_price"] = pd.to_numeric(df["mp_price"], errors="coerce")
    df["mp_year"] = pd.to_numeric(df["mp_year"], errors="coerce")
    df["mp_month"] = pd.to_numeric(df["mp_month"], errors="coerce")
    df = df.dropna(subset=["adm0_name", "cm_name", "pt_name", "um_name", "mp_price", "mp_year", "mp_month"]).copy()
    df["date"] = pd.to_datetime(
        dict(year=df["mp_year"].astype(int), month=df["mp_month"].astype(int), day=1),
        errors="coerce"
    )
    df = df[df["adm0_name"].isin(SAHEL_COUNTRIES)].copy()
    df["commodity"] = df["cm_name"].astype(str)
    df["commodity"] = df["commodity"].str.replace(r"\s*-\s*Retail$", "", regex=True)
    df["commodity"] = df["commodity"].str.replace(r"\s*-\s*Wholesale$", "", regex=True)
    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    df["month_label"] = df["mp_month"].map(month_map)
    return df


def median_price_by_group(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return data.groupby(group_cols, as_index=False)["mp_price"].median()


def make_top_commodities_chart(df_kg: pd.DataFrame):
    top = df_kg["commodity"].value_counts().head(10).reset_index()
    top.columns = ["commodity", "observations"]
    fig = px.bar(
        top,
        x="observations",
        y="commodity",
        orientation="h",
        text="observations",
        title="Top 10 staple commodities by number of observations"
    )
    fig.update_layout(template="simple_white", title_x=0.5, yaxis=dict(categoryorder="total ascending"))
    return fig


def make_price_trends_chart(df_kg: pd.DataFrame, countries: list[str], commodities: list[str], year_range: tuple[int, int]):
    dff = df_kg[
        df_kg["adm0_name"].isin(countries)
        & df_kg["commodity"].isin(commodities)
        & df_kg["mp_year"].between(year_range[0], year_range[1])
    ].copy()
    agg = median_price_by_group(dff, ["date", "adm0_name", "commodity"])
    fig = px.line(
        agg,
        x="date",
        y="mp_price",
        color="adm0_name",
        facet_row="commodity" if len(commodities) > 1 else None,
        title="Staple food price trends over time",
        labels={"mp_price": "Median price", "adm0_name": "Country", "date": "Date"}
    )
    fig.update_layout(template="simple_white", title_x=0.5, height=max(450, 260 * len(commodities)))
    fig.update_yaxes(matches=None)
    return fig


def make_seasonality_chart(df_kg: pd.DataFrame, countries: list[str], commodities: list[str]):
    dff = df_kg[df_kg["adm0_name"].isin(countries) & df_kg["commodity"].isin(commodities)].copy()
    agg = median_price_by_group(dff, ["mp_month", "month_label", "adm0_name", "commodity"])
    agg["seasonal_index"] = agg.groupby(["adm0_name", "commodity"])["mp_price"].transform(lambda x: 100 * x / x.mean())
    fig = px.line(
        agg,
        x="mp_month",
        y="seasonal_index",
        color="adm0_name",
        facet_row="commodity" if len(commodities) > 1 else None,
        markers=True,
        title="Seasonal price patterns of key staples",
        labels={"mp_month": "Month", "seasonal_index": "Seasonal index (avg = 100)"}
    )
    fig.update_layout(template="simple_white", title_x=0.5, height=max(450, 260 * len(commodities)))
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    fig.update_yaxes(matches=None)
    return fig


def make_imported_vs_local_chart(df_kg: pd.DataFrame, countries: list[str], local_staples: list[str]):
    commodities = ["Rice (imported)"] + local_staples
    dff = df_kg[df_kg["adm0_name"].isin(countries) & df_kg["commodity"].isin(commodities)].copy()
    agg = median_price_by_group(dff, ["adm0_name", "commodity"])
    fig = px.bar(
        agg,
        x="adm0_name",
        y="mp_price",
        color="commodity",
        barmode="group",
        title="Imported rice compared with local staples",
        labels={"adm0_name": "Country", "mp_price": "Median price", "commodity": "Commodity"}
    )
    fig.update_layout(template="simple_white", title_x=0.5)
    return fig


def make_market_dispersion_chart(df_kg: pd.DataFrame, countries: list[str], commodity: str, year_range: tuple[int, int]):
    dff = df_kg[
        df_kg["adm0_name"].isin(countries)
        & (df_kg["commodity"] == commodity)
        & df_kg["mp_year"].between(year_range[0], year_range[1])
    ].copy()
    market_disp = dff.groupby(["adm0_name", "mkt_name"], as_index=False)["mp_price"].median()
    fig = px.box(
        market_disp,
        x="adm0_name",
        y="mp_price",
        color="adm0_name",
        title=f"Distribution of market-level median prices for {commodity}",
        labels={"adm0_name": "Country", "mp_price": "Median price"}
    )
    fig.update_layout(template="simple_white", title_x=0.5, showlegend=False)
    return fig


def make_volatility_heatmap(df_kg: pd.DataFrame, countries: list[str], commodity: str, year_range: tuple[int, int]):
    dff = df_kg[
        df_kg["adm0_name"].isin(countries)
        & (df_kg["commodity"] == commodity)
        & df_kg["mp_year"].between(year_range[0], year_range[1])
    ].copy()
    agg = (
        dff.groupby(["date", "adm0_name"], as_index=False)["mp_price"]
        .median()
        .sort_values(["adm0_name", "date"])
    )
    agg["pct_change"] = agg.groupby("adm0_name")["mp_price"].pct_change() * 100
    agg["year_month"] = agg["date"].dt.strftime("%Y-%m")
    heat = agg.pivot(index="adm0_name", columns="year_month", values="pct_change")
    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=heat.columns,
            y=heat.index,
            colorscale="RdBu",
            zmid=0,
            colorbar_title="% monthly change"
        )
    )
    fig.update_layout(
        template="simple_white",
        title=f"Monthly price volatility heatmap for {commodity}",
        title_x=0.5,
        xaxis_title="Month",
        yaxis_title="Country",
        height=420
    )
    return fig


def make_purchasing_power_chart(df: pd.DataFrame, countries: list[str], food_commodity: str, wage_series: str, selected_country: str):
    food = df[
        df["adm0_name"].isin(countries)
        & (df["cm_name"] == f"{food_commodity} - Retail")
        & (df["um_name"] == "KG")
    ].copy()
    wage = df[
        df["adm0_name"].isin(countries)
        & (df["cm_name"] == wage_series)
    ].copy()

    food_agg = food.groupby(["date", "adm0_name"], as_index=False)["mp_price"].median().rename(columns={"mp_price": "food_price"})
    wage_agg = wage.groupby(["date", "adm0_name"], as_index=False)["mp_price"].median().rename(columns={"mp_price": "daily_wage"})
    tot = pd.merge(food_agg, wage_agg, on=["date", "adm0_name"], how="inner")
    tot["kg_affordable"] = tot["daily_wage"] / tot["food_price"]
    plot_df = tot[tot["adm0_name"] == selected_country].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["food_price"], mode="lines", name=f"{food_commodity} price"))
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["kg_affordable"], mode="lines", name="KG affordable per daily wage", yaxis="y2"))
    fig.update_layout(
        template="simple_white",
        title=f"{selected_country}: {food_commodity} price and purchasing power",
        title_x=0.5,
        xaxis_title="Date",
        yaxis=dict(title="Food price"),
        yaxis2=dict(title="KG affordable per daily wage", overlaying="y", side="right"),
        height=420
    )
    return fig


def make_hotspot_chart(df_kg: pd.DataFrame, commodity: str, year_filter: int):
    dff = df_kg[(df_kg["commodity"] == commodity) & (df_kg["mp_year"] == year_filter)].copy()
    agg = (
        dff.groupby(["adm0_name", "adm1_name", "mkt_name"], as_index=False)["mp_price"]
        .median()
        .sort_values("mp_price", ascending=False)
        .head(20)
    )
    agg["location"] = agg["adm0_name"].astype(str) + " | " + agg["adm1_name"].fillna("Unknown").astype(str) + " | " + agg["mkt_name"].astype(str)
    fig = px.bar(
        agg,
        x="mp_price",
        y="location",
        orientation="h",
        color="adm0_name",
        title=f"Top 20 highest-priced markets for {commodity} ({year_filter})",
        labels={"mp_price": "Median price", "location": "Market"}
    )
    fig.update_layout(template="simple_white", title_x=0.5, yaxis=dict(categoryorder="total ascending"))
    return fig


# ---------- App starts here ----------
st.title("Sahel Food Market Dynamics and Household Food Access")
st.caption("A Streamlit mini-app for WFP-style market monitoring and stakeholder storytelling")

with st.sidebar:
    st.header("Configuration")
    uploaded = st.file_uploader("Upload WFP food prices CSV", type=["csv"])
    default_path = Path("wfp_food_prices_database_wacaro.csv")
    csv_source = uploaded if uploaded is not None else str(default_path)

    st.markdown("---")
    st.subheader("Analytical filters")

# load data after uploader
try:
    df = load_data(csv_source)
except Exception as e:
    st.error(f"Could not load the data: {e}")
    st.stop()

# subsets
retail = df[df["pt_name"].astype(str).str.contains("Retail", case=False, na=False)].copy()
kg = retail[retail["um_name"] == "KG"].copy()

available_years = sorted(kg["mp_year"].dropna().astype(int).unique().tolist())
min_year, max_year = min(available_years), max(available_years)
all_commodities = sorted(kg["commodity"].dropna().unique().tolist())
selected_default_commodities = [c for c in ["Millet", "Sorghum", "Maize"] if c in all_commodities]
selected_default_locals = [c for c in ["Millet", "Sorghum", "Maize"] if c in all_commodities]

with st.sidebar:
    countries = st.multiselect("Countries", SAHEL_COUNTRIES, default=SAHEL_COUNTRIES)
    year_range = st.slider("Year range", min_year, max_year, (max(min_year, max_year-10), max_year))
    commodities = st.multiselect("Commodities for trend/seasonality", all_commodities, default=selected_default_commodities)
    local_staples = st.multiselect("Local staples for comparison with imported rice", [c for c in all_commodities if c != "Rice (imported)"], default=selected_default_locals)
    dispersion_commodity = st.selectbox("Commodity for market dispersion", all_commodities, index=all_commodities.index("Millet") if "Millet" in all_commodities else 0)
    volatility_commodity = st.selectbox("Commodity for volatility heatmap", all_commodities, index=all_commodities.index("Millet") if "Millet" in all_commodities else 0)
    afford_country = st.selectbox("Country for purchasing power", countries or SAHEL_COUNTRIES, index=0)
    wage_candidates = sorted(df[df["cm_name"].str.contains("Wage", case=False, na=False)]["cm_name"].unique().tolist())
    wage_series = st.selectbox("Wage series", wage_candidates, index=wage_candidates.index(DEFAULT_WAGE) if DEFAULT_WAGE in wage_candidates else 0)
    hotspot_year = st.selectbox("Year for hotspot ranking", available_years, index=len(available_years)-1)
    hotspot_commodity = st.selectbox("Commodity for hotspot ranking", all_commodities, index=all_commodities.index("Millet") if "Millet" in all_commodities else 0)

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Countries", f"{df['adm0_name'].nunique():,}")
k2.metric("Markets", f"{df['mkt_name'].nunique():,}")
k3.metric("Commodities", f"{df['commodity'].nunique():,}")
k4.metric("Observations (Sahel)", f"{len(df):,}")

st.markdown("---")

# charts
col1, col2 = st.columns([1, 1])
with col1:
    st.plotly_chart(make_top_commodities_chart(kg), use_container_width=True)
with col2:
    st.plotly_chart(make_hotspot_chart(kg, hotspot_commodity, hotspot_year), use_container_width=True)

st.plotly_chart(make_price_trends_chart(kg, countries, commodities or selected_default_commodities, year_range), use_container_width=True)
st.plotly_chart(make_seasonality_chart(kg, countries, commodities or selected_default_commodities), use_container_width=True)

c3, c4 = st.columns([1, 1])
with c3:
    st.plotly_chart(make_imported_vs_local_chart(kg, countries, local_staples or selected_default_locals), use_container_width=True)
with c4:
    st.plotly_chart(make_market_dispersion_chart(kg, countries, dispersion_commodity, year_range), use_container_width=True)

c5, c6 = st.columns([1, 1])
with c5:
    st.plotly_chart(make_volatility_heatmap(kg, countries, volatility_commodity, year_range), use_container_width=True)
with c6:
    if DEFAULT_FOOD not in all_commodities:
        affordable_food = all_commodities[0]
    else:
        affordable_food = DEFAULT_FOOD
    st.plotly_chart(make_purchasing_power_chart(df, countries, affordable_food, wage_series, afford_country), use_container_width=True)

st.markdown("---")
st.subheader("Suggested stakeholder narrative")
st.markdown(
    """
1. **Food access in the Sahel is highly market-dependent.**
2. **A small number of staples drive household vulnerability.**
3. **Seasonality creates predictable lean-season pressure.**
4. **Imported staples remain structurally more expensive than local cereals.**
5. **Market fragmentation means national averages can hide local stress.**
6. **Purchasing power provides the strongest household-impact story for donors and management.**
"""
)

st.info("Tip: use this app live during your presentation to let stakeholders explore countries, commodities, and years interactively.")
