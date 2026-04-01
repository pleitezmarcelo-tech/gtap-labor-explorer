"""
tools.py
Data tools available to the Claude agent.

Supported datasets:
1. GTAP labor dashboard data
2. Diaspora GDP long-form data
"""

import pandas as pd
import numpy as np
import json
import requests
import warnings
warnings.filterwarnings("ignore")

def _wcol(df, scenario=None):
    if "metric_value" in df.columns:
        return "metric_value"
    if scenario and scenario != "baseline" and "workers_change" in df.columns:
        return "workers_change"
    if "workers_base" in df.columns:
        return "workers_base"
    if "workers_sim" in df.columns:
        return "workers_sim"
    for c in df.columns:
        if "worker" in c.lower():
            return c
    return df.columns[-1]

def _bpcol(df):
    return "birthplace" if "birthplace" in df.columns else "birthplace_label"


def _schema(df):
    if "metric_value" in df.columns and "table_name" in df.columns:
        return "diaspora"
    return "gtap"

def _apply_filters(data, filters):
    if not filters:
        return data
    for col, val in filters.items():
        if col not in data.columns:
            continue
        if isinstance(val, list):
            data = data[data[col].isin(val)]
        else:
            data = data[data[col] == val]
    return data

TOOL_DEFINITIONS = [
    {
        "name": "query_dataset",
        "description": (
            "Query the current dataset. For GTAP labor data use worker columns such as "
            "workers_base, workers_sim, or workers_change. For Diaspora GDP data use "
            "metric_value and filter by metric_name, table_name, population_group, "
            "demographic_subgroup_label, nativity_type, state_name, or geography_level."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "object",
                    "description": "Filters as key-value pairs. Example: {\"scenario\": \"JPM_sim03\"}"
                },
                "group_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to group by. Example: [\"state\", \"gtap_code\"]"
                },
                "metric": {
                    "type": "string",
                    "enum": ["workers_base", "workers_sim", "workers_change", "metric_value"],
                    "description": "Metric to aggregate. Use workers_change for GTAP simulations and metric_value for Diaspora GDP."
                },
                "top_n": {"type": "integer", "description": "Return top N rows."},
                "sort_desc": {"type": "boolean", "description": "Sort descending. Default: true"}
            },
            "required": []
        }
    },
    {
        "name": "create_map",
        "description": (
            "Create a choropleth map from the current dataset. "
            "GTAP maps use county_fips. Diaspora GDP maps use state_abbrev."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "object",
                    "description": "Filters before mapping. Example: {\"scenario\": \"JPM_sim03\"}"
                },
                "metric": {
                    "type": "string",
                    "enum": ["workers_base", "workers_sim", "workers_change", "metric_value"],
                    "description": "Column to map. Auto-detected if not specified."
                },
                "title": {"type": "string", "description": "Map title."},
                "color_scale": {
                    "type": "string",
                    "enum": ["Blues", "Reds", "Greens", "RdYlGn", "YlOrRd", "Viridis"],
                    "description": "Color scale."
                }
            },
            "required": ["title"]
        }
    },
    {
        "name": "create_chart",
        "description": "Create a bar, line, or pie chart from the dataset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie", "treemap"]
                },
                "filters": {"type": "object"},
                "x": {"type": "string", "description": "Column for x-axis."},
                "metric": {
                    "type": "string",
                    "enum": ["workers_base", "workers_sim", "workers_change", "metric_value"],
                    "description": "Metric to plot. Auto-detected if not specified."
                },
                "color": {"type": "string", "description": "Column for color grouping."},
                "title": {"type": "string"}
            },
            "required": ["chart_type", "x", "title"]
        }
    },
    {
        "name": "get_dataset_info",
        "description": (
            "Get metadata for the current dataset, including columns, categories, "
            "geography coverage, and key metrics. Always call this first when the user asks what data is available."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "download_usda_data",
        "description": "Download data from the USDA NASS Quick Stats API.",
        "input_schema": {
            "type": "object",
            "properties": {
                "commodity_desc": {"type": "string"},
                "statisticcat_desc": {"type": "string"},
                "year": {"type": "integer"},
                "agg_level": {
                    "type": "string",
                    "enum": ["NATIONAL", "STATE", "COUNTY"]
                }
            },
            "required": ["commodity_desc", "statisticcat_desc"]
        }
    }
]


def query_dataset(df, filters=None, group_by=None,
                  metric=None, top_n=None, sort_desc=True):
    try:
        data = df.copy()
        data = _apply_filters(data, filters)
        scenario = (filters or {}).get("scenario", "baseline")
        if metric is None or metric not in data.columns:
            metric = _wcol(data, scenario)

        if group_by:
            valid_cols = [c for c in group_by if c in data.columns]
            if valid_cols and metric in data.columns:
                result = data.groupby(valid_cols)[metric].sum().reset_index()
            else:
                result = data.head(20)
        else:
            if metric in data.columns:
                result = pd.DataFrame({
                    "total_" + metric: [data[metric].sum()],
                    "count_rows": [len(data)]
                })
            else:
                result = data.head(20)

        if metric in result.columns:
            result = result.sort_values(metric, ascending=not sort_desc)
        if top_n and top_n > 0:
            result = result.head(top_n)

        result_fmt = result.copy()
        for col in result_fmt.select_dtypes(include=[np.number]).columns:
            if result_fmt[col].abs().max() > 1000:
                result_fmt[col] = result_fmt[col].apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) else "")

        return {
            "success": True,
            "rows": len(result_fmt),
            "metric_used": metric,
            "data": result_fmt.to_dict(orient="records"),
            "columns": list(result_fmt.columns)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def create_map(df, filters=None, metric=None,
               title="Workers by County", color_scale=None):
    import plotly.express as px
    try:
        data = df.copy()
        data = _apply_filters(data, filters)
        scenario = (filters or {}).get("scenario", "baseline")
        if metric is None or metric not in data.columns:
            metric = _wcol(data, scenario)
        if color_scale is None:
            color_scale = "RdYlGn" if metric == "workers_change" else "Blues"

        if "county_fips" in data.columns:
            county_data = (
                data.groupby(["county_fips", "county_name", "state"])[metric]
                .sum().reset_index()
            )
            county_data = county_data[
                county_data["county_fips"].notna() &
                (county_data["county_fips"].str.len() == 5)
            ]

            kw = {"color_continuous_midpoint": 0} if metric == "workers_change" else {}
            fig = px.choropleth(
                county_data,
                geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
                locations="county_fips",
                color=metric,
                color_continuous_scale=color_scale,
                scope="usa",
                hover_data={"county_fips": False, "county_name": True,
                            "state": True, metric: ":,"},
                title=title,
                labels={metric: metric.replace("_", " ").title(),
                        "county_name": "County", "state": "State"},
                **kw
            )
        elif "state_abbrev" in data.columns:
            state_data = (
                data.groupby(["state_abbrev", "state_name"])[metric]
                .sum().reset_index()
            )
            fig = px.choropleth(
                state_data,
                locations="state_abbrev",
                locationmode="USA-states",
                color=metric,
                color_continuous_scale=color_scale,
                scope="usa",
                hover_data={"state_abbrev": False, "state_name": True, metric: ":.2f"},
                title=title,
                labels={metric: metric.replace("_", " ").title(), "state_name": "State"},
            )
        else:
            return None

        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title=metric.replace("_", " ").title(), tickformat=",d"),
            title_font_size=16, height=550
        )
        return fig
    except Exception as e:
        print(f"Map error: {e}")
        return None


def create_chart(df, chart_type, filters=None, x="gtap_code",
                 metric=None, color=None, title="Chart"):
    import plotly.express as px
    try:
        data = df.copy()
        data = _apply_filters(data, filters)
        scenario = (filters or {}).get("scenario", "baseline")
        if metric is None or metric not in data.columns:
            metric = _wcol(data, scenario)

        group_cols = [c for c in ([x, color] if color and color != x else [x])
                      if c in data.columns]
        if not group_cols or metric not in data.columns:
            return None

        agg_data = data.groupby(group_cols)[metric].sum().reset_index()
        agg_data = agg_data.sort_values(metric, ascending=False)
        colors = px.colors.qualitative.Set2
        lbl = {metric: metric.replace("_", " ").title(),
               x: x.replace("_", " ").title()}

        if chart_type == "bar":
            fig = px.bar(agg_data, x=x, y=metric,
                color=color if color and color in agg_data.columns else None,
                title=title, color_discrete_sequence=colors, labels=lbl)
            fig.update_layout(xaxis_tickangle=-45)
        elif chart_type == "line":
            fig = px.line(agg_data, x=x, y=metric,
                color=color if color and color in agg_data.columns else None,
                title=title, color_discrete_sequence=colors,
                markers=True, labels=lbl)
        elif chart_type == "pie":
            fig = px.pie(agg_data.head(12), values=metric, names=x,
                title=title, color_discrete_sequence=colors)
        elif chart_type == "treemap":
            path_cols = ([color, x] if color and color in agg_data.columns
                         and color != x else [x])
            fig = px.treemap(agg_data, path=path_cols, values=metric,
                title=title, color_discrete_sequence=colors)
        else:
            fig = px.bar(agg_data, x=x, y=metric, title=title)

        fig.update_layout(
            height=480, title_font_size=16,
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(tickformat=",d", gridcolor="#f0f0f0"),
            margin=dict(t=50, b=80)
        )
        return fig
    except Exception as e:
        print(f"Chart error: {e}")
        return None


def get_dataset_info(df):
    try:
        if _schema(df) == "diaspora":
            info = {
                "success": True,
                "info": {
                    "dataset_type": "diaspora_gdp",
                    "total_rows": len(df),
                    "columns": list(df.columns),
                    "years": sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else [],
                    "table_names": sorted(df["table_name"].dropna().unique().tolist()) if "table_name" in df.columns else [],
                    "geography_levels": sorted(df["geography_level"].dropna().unique().tolist()) if "geography_level" in df.columns else [],
                    "population_groups": sorted([x for x in df["population_group"].dropna().unique().tolist() if x]) if "population_group" in df.columns else [],
                    "nativity_types": sorted([x for x in df["nativity_type"].dropna().unique().tolist() if x]) if "nativity_type" in df.columns else [],
                    "metrics": sorted(df["metric_name"].dropna().unique().tolist()) if "metric_name" in df.columns else [],
                    "n_states": df[df["geography_level"] == "state"]["state_name"].nunique() if "state_name" in df.columns and "geography_level" in df.columns else 0,
                }
            }
            if "metric_name" in df.columns and "metric_value" in df.columns:
                top_metrics = (
                    df.groupby("metric_name")["metric_value"]
                    .agg(["count", "min", "max"])
                    .reset_index()
                    .sort_values("count", ascending=False)
                    .head(10)
                )
                info["info"]["metric_preview"] = top_metrics.to_dict(orient="records")
            return info

        bp = _bpcol(df)
        has_sims = "scenario" in df.columns and df["scenario"].nunique() > 1
        scenarios = sorted(df["scenario"].unique().tolist()) if "scenario" in df.columns else ["baseline"]

        info = {
            "success": True,
            "info": {
                "total_rows": len(df),
                "columns": list(df.columns),
                "scenarios": scenarios,
                "has_simulations": has_sims,
                "year_available": 2024,
                "n_gtap_sectors": df["gtap_code"].nunique() if "gtap_code" in df.columns else 0,
                "gtap_sectors": sorted(df["gtap_code"].unique().tolist()) if "gtap_code" in df.columns else [],
                "n_states": df["state"].nunique() if "state" in df.columns else 0,
                "n_counties": df["county_fips"].nunique() if "county_fips" in df.columns else 0,
                "skill_levels": df["skill_level"].unique().tolist() if "skill_level" in df.columns else [],
                "birthplace_categories": df[bp].unique().tolist() if bp in df.columns else [],
            }
        }

        if "workers_base" in df.columns:
            base = df[df["scenario"] == "baseline"] if "scenario" in df.columns else df
            info["info"]["total_baseline_workers"] = f"{int(base['workers_base'].sum()):,}"
            if "gtap_code" in df.columns and "gtap_sector" in df.columns:
                top10 = (base.groupby(["gtap_code", "gtap_sector"])["workers_base"]
                         .sum().reset_index()
                         .sort_values("workers_base", ascending=False).head(10))
                info["info"]["top_10_sectors"] = [
                    f"{r['gtap_code']} - {r['gtap_sector']}: {int(r['workers_base']):,}"
                    for _, r in top10.iterrows()
                ]

        if has_sims and "workers_change" in df.columns:
            sim_summary = {}
            for scen in [s for s in scenarios if s != "baseline"]:
                sf = df[df["scenario"] == scen]
                sim_summary[scen] = f"{int(sf['workers_change'].sum()):,}"
            info["info"]["total_change_by_scenario"] = sim_summary

        return info
    except Exception as e:
        return {"success": False, "error": str(e)}


def download_usda_data(api_key, commodity_desc, statisticcat_desc,
                       year=None, agg_level="STATE"):
    if not api_key or api_key.strip() in ("", "YOUR_USDA_API_KEY"):
        return {"success": False, "error": "USDA API key not configured."}
    params = {
        "key": api_key,
        "commodity_desc": commodity_desc.upper(),
        "statisticcat_desc": statisticcat_desc.upper(),
        "agg_level_desc": agg_level,
        "format": "JSON"
    }
    if year:
        params["year"] = year
    try:
        resp = requests.get(
            "https://quickstats.nass.usda.gov/api/api_GET/",
            params=params, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        if "data" not in data:
            return {"success": False, "error": "No data returned"}
        df_usda = pd.DataFrame(data["data"])
        return {
            "success": True,
            "rows_downloaded": len(df_usda),
            "preview": df_usda.head(3).to_dict(orient="records")
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_tool(tool_name, tool_input, df, usda_api_key=""):
    """Dispatch a tool call. Returns (result, is_figure)."""
    if tool_name == "query_dataset":
        return query_dataset(df, **tool_input), False
    elif tool_name == "create_map":
        return create_map(df, **tool_input), True
    elif tool_name == "create_chart":
        return create_chart(df, **tool_input), True
    elif tool_name == "get_dataset_info":
        return get_dataset_info(df), False
    elif tool_name == "download_usda_data":
        return download_usda_data(usda_api_key, **tool_input), False
    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}, False
