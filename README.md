# GTAP Labor Data Explorer

A conversational data exploration app powered by Claude. Ask questions about 
U.S. labor inputs for GTAP in natural language — Claude queries the data, 
builds county-level maps, and explains results.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your data file
Copy `gtap_complete_master.csv` to this folder, or note the full path.

### 3. Set API keys (optional — can also enter in the app sidebar)
```bash
export ANTHROPIC_API_KEY="your-key-here"
export USDA_API_KEY="your-usda-key-here"   # optional
```

Get keys:
- Anthropic: https://console.anthropic.com
- USDA NASS (free): https://quickstats.nass.usda.gov/api

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at http://localhost:8501

---

## What you can ask

**Data exploration**
- "What data is available?"
- "How many workers are in each GTAP sector in 2024?"
- "Which sectors have the highest share of foreign-born workers?"

**Maps**
- "Show a map of vegetable and fruit workers by county in 2022"
- "Map foreign-born workers in construction"
- "Which counties have the most skilled workers in manufacturing?"

**Trends and comparisons**
- "Show the trend in total agricultural workers 2021-2024"
- "Compare US-born vs foreign-born across all agricultural sectors"
- "How has the skilled/unskilled ratio changed over time?"

**USDA data updates** (requires USDA API key)
- "Check the latest available USDA hired labor data"
- "Download USDA milk sales by state"

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application and Claude agent |
| `tools.py` | Data tool functions (query, map, chart, USDA API) |
| `requirements.txt` | Python dependencies |
| `gtap_complete_master.csv` | Your GTAP labor dataset (place here) |

---

## Adding new data sources

To add a new API or data source, add a function to `tools.py` and register 
it in `TOOL_DEFINITIONS`. Claude will automatically learn to use it based 
on the description you provide.
