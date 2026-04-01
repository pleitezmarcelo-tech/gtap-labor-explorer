from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROOT_SOURCE_XLSX = ROOT / "m3_gdp_estimates.xlsx"
LOCAL_SOURCE_XLSX = ROOT / "data" / "raw" / "m3_gdp_estimates.xlsx"
LEGACY_SOURCE_XLSX = Path(
    "/Users/marcelopleitez/Library/CloudStorage/"
    "GoogleDrive-pleitez.marcelo@gmail.com/My Drive/"
    "NAID Center 2020_2/2026/Base de datos/"
    "Diaspora GDP contribution/data/final/m3_gdp_estimates.xlsx"
)

OUT_STATE = ROOT / "diaspora_state.parquet"
OUT_NATIONAL = ROOT / "diaspora_national.parquet"
OUT_WIDE = ROOT / "diaspora_wide.parquet"
OUT_LONG = ROOT / "diaspora_gdp_long.parquet"


def source_xlsx() -> Path:
    if LOCAL_SOURCE_XLSX.exists():
        return LOCAL_SOURCE_XLSX
    if ROOT_SOURCE_XLSX.exists():
        return ROOT_SOURCE_XLSX
    return LEGACY_SOURCE_XLSX


def write_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


STATE_ABBREV = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def colnum(cell_ref: str) -> int:
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n


def load_shared_strings(zf: ZipFile) -> list[str]:
    shared = []
    if "xl/sharedStrings.xml" not in zf.namelist():
        return shared
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    for si in root.findall("a:si", NS):
        shared.append("".join(t.text or "" for t in si.iterfind(".//a:t", NS)))
    return shared


def sheet_target_map(zf: ZipFile) -> dict[str, str]:
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("pr:Relationship", NS)
    }
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    targets = {}
    for sheet in wb.findall("a:sheets/a:sheet", NS):
        name = sheet.attrib["name"]
        rid = sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        target = rel_map[rid]
        if not target.startswith("xl/"):
            target = "xl/" + target
        targets[name] = target
    return targets


def read_sheet_rows(zf: ZipFile, target: str, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(zf.read(target))
    rows = []
    for row in root.findall("a:sheetData/a:row", NS):
        values = {}
        for cell in row.findall("a:c", NS):
            ref = cell.attrib["r"]
            idx = colnum(ref)
            cell_type = cell.attrib.get("t")
            value = cell.find("a:v", NS)
            inline = cell.find("a:is", NS)
            if cell_type == "s" and value is not None:
                txt = shared[int(value.text)]
            elif cell_type == "inlineStr" and inline is not None:
                txt = "".join(x.text or "" for x in inline.iterfind(".//a:t", NS))
            else:
                txt = value.text if value is not None else ""
            values[idx] = txt
        max_col = max(values) if values else 0
        rows.append([values.get(i, "") for i in range(1, max_col + 1)])
    return rows


def rows_to_df(rows: list[list[str]]) -> pd.DataFrame:
    header = []
    seen = {}
    blank_count = 0
    for raw in rows[0]:
        name = (raw or "").strip()
        if not name:
            blank_count += 1
            name = f"__blank_{blank_count}"
        if name in seen:
            seen[name] += 1
            name = f"{name}__{seen[name]}"
        else:
            seen[name] = 1
        header.append(name)
    body = rows[1:]
    return pd.DataFrame(body, columns=header)


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_state_group(df: pd.DataFrame, population_group: str) -> pd.DataFrame:
    raw = df.copy()
    df = pd.DataFrame(
        {
            "year": raw["year"],
            "state_name": raw["State name"],
            "demographic_subgroup_label": raw["Demographic subgroup label"],
            "nativity_type": raw["Nativity type"],
            "share_of_state_gdp_pct": raw["Share of state GDP (%)"],
            "total_state_gdp_billion_2023": raw["Total state GDP (billions USD, 2023)"],
        }
    )

    if "GDP contribution (billions USD, 2023)" in raw.columns:
        df["gdp_billion_2023"] = raw["GDP contribution (billions USD, 2023)"]
        if "GDP contribution (trillions USD, 2023)" in raw.columns:
            df["gdp_trillion_2023"] = raw["GDP contribution (trillions USD, 2023)"]
        else:
            df["gdp_trillion_2023"] = pd.to_numeric(
                raw["GDP contribution (billions USD, 2023)"], errors="coerce"
            ) / 1000
    else:
        # State_Latino has a shifted header:
        # the "trillions" header actually contains the billions values,
        # while the blank column to its right contains the real trillions.
        df["gdp_billion_2023"] = raw["GDP contribution (trillions USD, 2023)"]
        df["gdp_trillion_2023"] = raw["__blank_1"]

    df["population_group"] = population_group
    df["state_abbrev"] = df["state_name"].map(STATE_ABBREV)
    df = to_num(
        df,
        [
            "year",
            "gdp_billion_2023",
            "gdp_trillion_2023",
            "share_of_state_gdp_pct",
            "total_state_gdp_billion_2023",
        ],
    )
    return df


def normalize_national(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "Population group (Latino or Mexican-Origin)": "population_group",
            "Demographic subgroup label": "demographic_subgroup_label",
            "Nativity type": "nativity_type",
            "GDP contribution (billions USD, 2023)": "gdp_billion_2023",
            "GDP contribution (trillions USD, 2023)": "gdp_trillion_2023",
            "Share of US GDP (%)": "share_of_us_gdp_pct",
        }
    ).copy()
    keep = [
        "year",
        "population_group",
        "demographic_subgroup_label",
        "nativity_type",
        "gdp_billion_2023",
        "share_of_us_gdp_pct",
    ]
    if "gdp_trillion_2023" in df.columns:
        keep.insert(5, "gdp_trillion_2023")
    df = df[keep].copy()
    if "gdp_trillion_2023" not in df.columns:
        df["gdp_trillion_2023"] = pd.to_numeric(df["gdp_billion_2023"], errors="coerce") / 1000
    df = to_num(
        df,
        ["year", "gdp_billion_2023", "gdp_trillion_2023", "share_of_us_gdp_pct"],
    )
    return df


def normalize_wide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "year": "year",
            "State FIPS": "state_name_raw",
            "State name": "state_name",
            "Total state GDP (billions USD)": "total_state_gdp_billion_2023",
            "Latino Native-Born GDP (billions)": "latino_native_born_gdp_billion_2023",
            "Latino Foreign-Born GDP (billions)": "latino_foreign_born_gdp_billion_2023",
            "Latino Total GDP (billions)": "latino_total_gdp_billion_2023",
            "Latino GDP as pct of state GDP": "latino_gdp_as_pct_of_state_gdp",
            "Mexican-Origin Native-Born GDP (billions)": "mexican_native_born_gdp_billion_2023",
            "Mexican-Origin Foreign-Born GDP (billions)": "mexican_foreign_born_gdp_billion_2023",
            "Mexican-Origin Total GDP (billions)": "mexican_total_gdp_billion_2023",
            "Mexican-Origin GDP as pct of state GDP": "mexican_gdp_as_pct_of_state_gdp",
            "Mexican-Origin as pct of Latino GDP": "mexican_as_pct_of_latino_gdp",
        }
    ).copy()
    df["state_abbrev"] = df["state_name"].map(STATE_ABBREV)
    num_cols = [
        "year",
        "total_state_gdp_billion_2023",
        "latino_native_born_gdp_billion_2023",
        "latino_foreign_born_gdp_billion_2023",
        "latino_total_gdp_billion_2023",
        "latino_gdp_as_pct_of_state_gdp",
        "mexican_native_born_gdp_billion_2023",
        "mexican_foreign_born_gdp_billion_2023",
        "mexican_total_gdp_billion_2023",
        "mexican_gdp_as_pct_of_state_gdp",
        "mexican_as_pct_of_latino_gdp",
    ]
    df = to_num(df, num_cols)
    return df


def melt_long(
    state_df: pd.DataFrame, national_df: pd.DataFrame, wide_df: pd.DataFrame
) -> pd.DataFrame:
    long_parts = []

    state_metric_map = {
        "gdp_billion_2023": ("GDP contribution (billions USD, 2023)", "billions_usd"),
        "gdp_trillion_2023": ("GDP contribution (trillions USD, 2023)", "trillions_usd"),
        "share_of_state_gdp_pct": ("Share of state GDP (%)", "percent"),
        "total_state_gdp_billion_2023": ("Total state GDP (billions USD, 2023)", "billions_usd"),
    }
    for metric_name, meta in state_metric_map.items():
        tmp = state_df[
            [
                "year",
                "state_name",
                "state_abbrev",
                "population_group",
                "demographic_subgroup_label",
                "nativity_type",
                metric_name,
            ]
        ].copy()
        tmp["table_name"] = "state_group"
        tmp["geography_level"] = "state"
        tmp["metric_name"] = metric_name
        tmp["metric_label"] = meta[0]
        tmp["unit"] = meta[1]
        tmp["metric_value"] = tmp[metric_name]
        long_parts.append(tmp.drop(columns=[metric_name]))

    national_metric_map = {
        "gdp_billion_2023": ("GDP contribution (billions USD, 2023)", "billions_usd"),
        "gdp_trillion_2023": ("GDP contribution (trillions USD, 2023)", "trillions_usd"),
        "share_of_us_gdp_pct": ("Share of US GDP (%)", "percent"),
    }
    for metric_name, meta in national_metric_map.items():
        tmp = national_df[
            [
                "year",
                "population_group",
                "demographic_subgroup_label",
                "nativity_type",
                metric_name,
            ]
        ].copy()
        tmp["table_name"] = "national_summary"
        tmp["geography_level"] = "national"
        tmp["state_name"] = ""
        tmp["state_abbrev"] = ""
        tmp["metric_name"] = metric_name
        tmp["metric_label"] = meta[0]
        tmp["unit"] = meta[1]
        tmp["metric_value"] = tmp[metric_name]
        long_parts.append(tmp.drop(columns=[metric_name]))

    wide_metric_map = {
        "total_state_gdp_billion_2023": ("Total state GDP (billions USD, 2023)", "billions_usd"),
        "latino_native_born_gdp_billion_2023": ("Latino Native-Born GDP (billions)", "billions_usd"),
        "latino_foreign_born_gdp_billion_2023": ("Latino Foreign-Born GDP (billions)", "billions_usd"),
        "latino_total_gdp_billion_2023": ("Latino Total GDP (billions)", "billions_usd"),
        "latino_gdp_as_pct_of_state_gdp": ("Latino GDP as pct of state GDP", "percent"),
        "mexican_native_born_gdp_billion_2023": ("Mexican-Origin Native-Born GDP (billions)", "billions_usd"),
        "mexican_foreign_born_gdp_billion_2023": ("Mexican-Origin Foreign-Born GDP (billions)", "billions_usd"),
        "mexican_total_gdp_billion_2023": ("Mexican-Origin Total GDP (billions)", "billions_usd"),
        "mexican_gdp_as_pct_of_state_gdp": ("Mexican-Origin GDP as pct of state GDP", "percent"),
        "mexican_as_pct_of_latino_gdp": ("Mexican-Origin as pct of Latino GDP", "percent"),
    }
    for metric_name, meta in wide_metric_map.items():
        tmp = wide_df[["year", "state_name", "state_abbrev", metric_name]].copy()
        tmp["table_name"] = "state_wide"
        tmp["geography_level"] = "state"
        tmp["population_group"] = ""
        tmp["demographic_subgroup_label"] = ""
        tmp["nativity_type"] = ""
        tmp["metric_name"] = metric_name
        tmp["metric_label"] = meta[0]
        tmp["unit"] = meta[1]
        tmp["metric_value"] = tmp[metric_name]
        long_parts.append(tmp.drop(columns=[metric_name]))

    long_df = pd.concat(long_parts, ignore_index=True)
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df["metric_value"] = pd.to_numeric(long_df["metric_value"], errors="coerce")
    return long_df


def main() -> None:
    source = source_xlsx()
    with ZipFile(source) as zf:
        shared = load_shared_strings(zf)
        targets = sheet_target_map(zf)

        national_rows = read_sheet_rows(zf, targets["Summary_National"], shared)
        latino_rows = read_sheet_rows(zf, targets["State_Latino"], shared)
        mexican_rows = read_sheet_rows(zf, targets["State_Mexican"], shared)
        wide_rows = read_sheet_rows(zf, targets["State_Wide"], shared)

    national_df = normalize_national(rows_to_df(national_rows))
    state_latino = normalize_state_group(rows_to_df(latino_rows), "Latino")
    state_mexican = normalize_state_group(rows_to_df(mexican_rows), "Mexican-Origin")
    state_df = pd.concat([state_latino, state_mexican], ignore_index=True)
    wide_df = normalize_wide(rows_to_df(wide_rows))
    long_df = melt_long(state_df, national_df, wide_df)

    write_parquet_safe(state_df, OUT_STATE)
    write_parquet_safe(national_df, OUT_NATIONAL)
    write_parquet_safe(wide_df, OUT_WIDE)
    write_parquet_safe(long_df, OUT_LONG)

    print(f"Source Excel: {source}")
    print(f"Saved {OUT_STATE.name}: {len(state_df):,} rows")
    print(f"Saved {OUT_NATIONAL.name}: {len(national_df):,} rows")
    print(f"Saved {OUT_WIDE.name}: {len(wide_df):,} rows")
    print(f"Saved {OUT_LONG.name}: {len(long_df):,} rows")


if __name__ == "__main__":
    main()
