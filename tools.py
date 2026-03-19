"""
tools.py
Data tools available to the Claude agent.
Each function is called when Claude decides to use a tool.
"""

import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import requests
import warnings
warnings.filterwarnings("ignore")

# ── TOOL DEFINITIONS FOR CLAUDE API ──────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "query_dataset",
        "description": (
            "Query the GTAP labor dataset. Use this tool to answer questions "
            "about worker counts, sectors, regions, skill levels, birthplace, "
            "years, or counties. Returns aggregated data as a table. "
            "Available columns: gtap_code, gtap_sector, skill_level, "
            "birthplace_label, state, county_fips, county_name, year, "
            "n_in_sample, estimated_workers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "object",
                    "description": (
                        "Optional filters as key-value pairs. Keys must be "
                        "valid column names. Example: "
                        "{\"gtap_code\": \"v_f\", \"year\": 2022}"
                    )
                },
                "group_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Columns to group by for aggregation. "
                        "Example: [\"gtap_code\", \"year\"]"
                    )
                },
                "metric": {
                    "type": "string",
                    "enum": ["estimated_workers", "n_in_sample"],
                    "description": "Metric to aggregate. Default: estimated_workers"
                },
                "top_n": {
                    "type": "integer",
                    "description": "Return only top N rows by metric value. Optional."
                },
                "sort_desc": {
                    "type": "boolean",
                    "description": "Sort descending by metric. Default: true"
                }
            },
            "required": []
        }
    },
    {
        "name": "create_map",
        "description": (
            "Create a choropleth map of the United States showing estimated "
            "workers by county. Returns a Plotly figure. Use when the user "
            "asks for a map, geographic visualization, or county-level view."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "object",
                    "description": (
                        "Filters to apply before mapping. "
                        "Example: {\"gtap_code\": \"v_f\", \"year\": 2022, "
                        "\"birthplace_label\": \"Foreign born\"}"
                    )
                },
                "title": {
                    "type": "string",
                    "description": "Map title. Be descriptive."
                },
                "color_scale": {
                    "type": "string",
                    "enum": ["Blues", "Reds", "Greens", "YlOrRd", "Viridis", "Plasma"],
                    "description": "Color scale for the choropleth. Default: Blues"
                }
            },
            "required": ["title"]
        }
    },
    {
        "name": "create_chart",
        "description": (
            "Create a bar chart, line chart, or pie chart from the dataset. "
            "Use for trends over time, sector comparisons, or distributions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie", "treemap"],
                    "description": "Type of chart to create"
                },
                "filters": {
                    "type": "object",
                    "description": "Filters to apply before charting"
                },
                "x": {
                    "type": "string",
                    "description": "Column for x-axis or categories"
                },
                "color": {
                    "type": "string",
                    "description": "Column to use for color grouping. Optional."
                },
                "title": {
                    "type": "string",
                    "description": "Chart title"
                }
            },
            "required": ["chart_type", "x", "title"]
        }
    },
    {
        "name": "get_dataset_info",
        "description": (
            "Get metadata about the dataset: available sectors, years, states, "
            "skill levels, birthplace categories, total worker counts, and "
            "column descriptions. Use this first when the user asks a general "
            "question about what data is available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "download_usda_data",
        "description": (
            "Download fresh data from the USDA NASS Quick Stats API. Use when "
            "the user asks to update USDA data, check for new Census of "
            "Agriculture releases, or download specific commodity data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "commodity_desc": {
                    "type": "string",
                    "description": "Commodity to query. Example: LABOR, CORN, MILK"
                },
                "statisticcat_desc": {
                    "type": "string",
                    "description": "Statistic category. Example: WORKERS, SALES, AREA HARVESTED"
                },
                "year": {
                    "type": "integer",
                    "description": "Year to query. Leave empty for latest available."
                },
                "agg_level": {
                    "type": "string",
                    "enum": ["NATIONAL", "STATE", "COUNTY"],
                    "description": "Geographic aggregation level"
                }
            },
            "required": ["commodity_desc", "statisticcat_desc"]
        }
    }
]


# ── TOOL EXECUTION FUNCTIONS ──────────────────────────────────────────────────

def query_dataset(df, filters=None, group_by=None,
                  metric="estimated_workers", top_n=None,
                  sort_desc=True):
    try:
        data = df.copy()
        if filters:
            for col, val in filters.items():
                if col in data.columns:
                    if isinstance(val, list):
                        data = data[data[col].isin(val)]
                    else:
                        data = data[data[col] == val]

        if group_by:
            valid_cols = [c for c in group_by if c in data.columns]
            if valid_cols:
                agg_col = metric if metric in data.columns else "estimated_workers"
                result = data.groupby(valid_cols)[agg_col].sum().reset_index()
                result.columns = list(valid_cols) + [agg_col]
            else:
                result = data
        else:
            if metric in data.columns:
                result = pd.DataFrame({
                    "total": [data[metric].sum()],
                    "count_rows": [len(data)]
                })
            else:
                result = data

        sort_col = metric if metric in result.columns else result.columns[-1]
        if sort_col in result.columns:
            result = result.sort_values(sort_col, ascending=not sort_desc)

        if top_n and top_n > 0:
            result = result.head(top_n)

        for col in result.select_dtypes(include=[np.number]).columns:
            if result[col].max() > 1000:
                result[col] = result[col].apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) else "")

        return {
            "success": True,
            "rows": len(result),
            "data": result.to_dict(orient="records"),
            "columns": list(result.columns)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def create_map(df, filters=None, title="Workers by County",
               color_scale="Blues"):
    try:
        data = df.copy()
        if filters:
            for col, val in filters.items():
                if col in data.columns:
                    if isinstance(val, list):
                        data = data[data[col].isin(val)]
                    else:
                        data = data[data[col] == val]

        if "county_fips" not in data.columns:
            return None

        county_data = (
            data.groupby(["county_fips", "county_name", "state"])["estimated_workers"]
            .sum().reset_index()
        )
        county_data = county_data[
            county_data["county_fips"].notna() &
            (county_data["county_fips"].str.len() == 5)
        ]
        county_data["hover_text"] = (
            county_data["county_name"] + ", " + county_data["state"] +
            "<br>Workers: " +
            county_data["estimated_workers"].apply(lambda x: f"{int(x):,}")
        )

        fig = px.choropleth(
            county_data,
            geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
            locations="county_fips",
            color="estimated_workers",
            color_continuous_scale=color_scale,
            scope="usa",
            hover_name="hover_text",
            hover_data={"county_fips": False, "estimated_workers": False,
                        "county_name": False, "state": False},
            title=title,
            labels={"estimated_workers": "Estimated Workers"}
        )
        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Workers", tickformat=",d"),
            title_font_size=16,
            height=550
        )
        return fig
    except Exception as e:
        print(f"Map error: {e}")
        return None


def create_chart(df, chart_type, filters=None, x="gtap_code",
                 color=None, title="Chart"):
    try:
        data = df.copy()
        if filters:
            for col, val in filters.items():
                if col in data.columns:
                    if isinstance(val, list):
                        data = data[data[col].isin(val)]
                    else:
                        data = data[data[col] == val]

        group_cols = [c for c in ([x, color] if color and color != x else [x])
                      if c in data.columns]
        agg_data = data.groupby(group_cols)["estimated_workers"].sum().reset_index()
        agg_data = agg_data.sort_values("estimated_workers", ascending=False)
        colors = px.colors.qualitative.Set2

        if chart_type == "bar":
            fig = px.bar(
                agg_data, x=x, y="estimated_workers",
                color=color if color and color in agg_data.columns else None,
                title=title, color_discrete_sequence=colors,
                labels={"estimated_workers": "Estimated Workers",
                        x: x.replace("_", " ").title()}
            )
            fig.update_layout(xaxis_tickangle=-45)
        elif chart_type == "line":
            fig = px.line(
                agg_data, x=x, y="estimated_workers",
                color=color if color and color in agg_data.columns else None,
                title=title, color_discrete_sequence=colors, markers=True,
                labels={"estimated_workers": "Estimated Workers",
                        x: x.replace("_", " ").title()}
            )
        elif chart_type == "pie":
            fig = px.pie(
                agg_data.head(12), values="estimated_workers", names=x,
                title=title, color_discrete_sequence=colors
            )
        elif chart_type == "treemap":
            path_cols = ([color, x] if color and color in agg_data.columns
                         and color != x else [x])
            fig = px.treemap(
                agg_data, path=path_cols, values="estimated_workers",
                title=title, color_discrete_sequence=colors
            )
        else:
            fig = px.bar(agg_data, x=x, y="estimated_workers", title=title)

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
        return {
            "success": True,
            "info": {
                "total_rows": len(df),
                "total_workers_all_years": int(df["estimated_workers"].sum()),
                "years_available": sorted(df["year"].unique().tolist()),
                "gtap_sectors": sorted(df["gtap_code"].unique().tolist()),
                "n_gtap_sectors": df["gtap_code"].nunique(),
                "n_states": df["state"].nunique(),
                "n_counties": df["county_fips"].nunique(),
                "skill_levels": df["skill_level"].unique().tolist(),
                "birthplace_categories": df["birthplace_label"].unique().tolist(),
                "columns": list(df.columns),
                "workers_by_year": (
                    df.groupby("year")["estimated_workers"].sum()
                    .apply(lambda x: f"{int(x):,}").to_dict()
                ),
                "top_10_sectors": (
                    df.groupby(["gtap_code", "gtap_sector"])["estimated_workers"]
                    .sum().reset_index()
                    .sort_values("estimated_workers", ascending=False)
                    .head(10)
                    .apply(lambda r: f"{r['gtap_code']} - {r['gtap_sector']}: "
                           f"{int(r['estimated_workers']):,}", axis=1)
                    .tolist()
                )
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def download_usda_data(api_key, commodity_desc, statisticcat_desc,
                       year=None, agg_level="STATE"):
    if not api_key or api_key.strip() in ("", "YOUR_USDA_API_KEY"):
        return {
            "success": False,
            "error": (
                "USDA API key not configured. "
                "Get a free key at: https://quickstats.nass.usda.gov/api"
            )
        }
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
            return {"success": False, "error": "No data returned", "raw": str(data)[:300]}
        records = data["data"]
        df_usda = pd.DataFrame(records)
        return {
            "success": True,
            "rows_downloaded": len(df_usda),
            "years_available": (sorted(df_usda["year"].unique().tolist())
                                if "year" in df_usda.columns else []),
            "short_desc_available": (df_usda["short_desc"].unique().tolist()[:15]
                                     if "short_desc" in df_usda.columns else []),
            "preview": df_usda.head(3).to_dict(orient="records")
        }
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to USDA API."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_tool(tool_name, tool_input, df, usda_api_key=""):
    """Dispatch a tool call from Claude. Returns (result, is_figure)."""
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
