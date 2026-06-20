import re

import pandas as pd


CEMENT_REGEX_MAP = [
    (r'high.?sulfate|type.?hs',     'HS_cement'),
    (r'type.?iii|type.?3\b',        'OPC_III'),
    (r'white',                       'white_cement'),
    (r'cem.?ii|cem2|cemii',          'CEM_II'),
    (r'blast.?furnace',              'BFS_cement'),
    (r'pozzolan',                    'pozzolan_cement'),
    (r'ggbs',                        'OPC_I_GGBS'),
    (r'53.?grade|grade.?53|\b53\b',  'OPC_53'),
    (r'52[.,]?5',                    'OPC_52.5'),
    (r'42[.,]?5',                    'OPC_42.5'),
]

SP_REGEX_MAP = [
    (r'vma|viscosity.modif',                                        'VMA'),
    (r'naphthalene|sulfonat',                                       'SNF_SP'),
    (r'acrylic|acrylate|poly.acrylic.ester',                        'Other_Polymer_SP'),
    (r'polycarboxylic\s+ether|polycarboxylic.based|pce|hyperplastic','PCE_HRWRA'),
    (r'polycarboxylate|polycarboxilate|carboxylate.based',          'PCE_SP'),
    (r'hrwra|hrwr|high.range.water|high.performance.water',         'HRWRA'),
]


def load_raw_uhpc(config):
    """Load the raw UHPC csv (multi-level header) and assign meaningful column names."""
    df = pd.read_csv(config['filepath'], encoding="latin-1", header=[0, 1])
    df.columns = df.columns.get_level_values(1)
    df.columns = config['column_names']

    # Delete mix_id and research paper columns, flatten multi-level index
    df = df.iloc[1:, 1:-4].reset_index(drop=True)

    for col in df.columns:
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.isna().sum() == df[col].isna().sum():
            df[col] = converted

    return df


def filter_columns(df, config, target_col='cs_28d'):
    """Drop rows missing the target variable and drop the configured unused columns."""
    df = df.dropna(subset=[target_col])
    df = df.drop(columns=config['drop_cols'])
    return df


def clean_basic_types(df):
    """Normalize dtypes/formatting for curing_pressure, fly_ash/slag/filler/sand types and fiber2."""
    df = df.copy()

    # Extract numeric value from curing_pressure (handles cases like "1.2?MPa", "2 MPa")
    df['curing_pressure'] = df['curing_pressure'].str.extract(r'(\d+\.?\d*)')[0].astype(float)

    # Clean fly_ash_type: extract last character (F or C), strip whitespace, standardize
    df['fly_ash_type'] = df['fly_ash_type'].str.strip().str[-1]
    df['fly_ash_type'] = df['fly_ash_type'].replace({'F': 'class F', 'C': 'class C'})

    df['slag_type'] = df['slag_type'].str.strip().str.title()
    df['filler_type'] = df['filler_type'].str.strip().str.title()
    df['sand_type'] = df['sand_type'].str.strip().str.title()

    df['fiber2_amount'] = pd.to_numeric(df['fiber2_amount'], errors='coerce')
    df['fiber2_length'] = pd.to_numeric(df['fiber2_length'], errors='coerce')

    return df


def categorize_fiber(val):
    """Standardize free-text fiber type descriptions into a fixed set of categories."""
    if pd.isna(val):
        return val
    v = val.strip().lower()
    if 'hooked' in v:
        return 'Hooked Steel Fiber'
    elif 'twisted' in v:
        return 'Twisted Steel Fiber'
    elif 'straight' in v or ('steel' in v and 'fiber' in v) or ('steel' in v and 'fibre' in v):
        return 'Straight Steel Fiber'
    elif 'pe' == v or 'polyethylene' in v:
        return 'PE Fiber'
    elif 'pva' in v:
        return 'PVA Fiber'
    elif 'glass' in v:
        return 'Glass Fiber'
    elif 'carbon' in v:
        return 'Carbon Fiber'
    elif 'pp' == v:
        return 'PP Fiber'
    elif 'cellulose' in v:
        return 'Cellulose Fiber'
    elif 'wollastonit' in v:
        return 'Wollastonite Fiber'
    elif 'medium' in v or 'short' in v or 'long' in v:
        return 'Straight Steel Fiber'
    else:
        return val.strip().title()


def standardize_fiber_types(df):
    """Apply categorize_fiber to fiber1_type and fiber2_type."""
    df = df.copy()
    df['fiber1_type'] = df['fiber1_type'].apply(categorize_fiber)
    df['fiber2_type'] = df['fiber2_type'].apply(categorize_fiber)
    return df


def categorize_curing(val):
    """Standardize free-text curing method descriptions into a fixed set of categories."""
    if pd.isna(val):
        return val
    v = val.strip().lower()
    if 'autoclave' in v:
        return 'Autoclave Curing'
    elif 'steam' in v:
        return 'Steam Curing'
    elif 'hot water' in v or 'warm water' in v or 'warm bath' in v:
        return 'Hot Water Curing'
    elif 'heat' in v or 'hot' in v or 'oven' in v:
        return 'Heat Curing'
    elif 'water' in v or 'moist' in v:
        return 'Water Curing'
    elif 'air' in v:
        return 'Air Curing'
    elif 'standard' in v or 'normal' in v:
        return 'Standard Curing'
    else:
        return val.strip().title()


def standardize_curing_method(df):
    """Apply categorize_curing and fix known typo/edge-case labels."""
    df = df.copy()
    df['curing_method'] = df['curing_method'].apply(categorize_curing)
    df['curing_method'] = df['curing_method'].replace({
        'Curing  At 90 Oc': 'Heat Curing',
        'Stnadrad Curing': 'Standard Curing'
    })
    return df


def check_unique(df, col):
    """Print the unique values and cardinality of a column."""
    print(f"Unique {col} values:")
    print(list(df[col].unique()))
    print(f"Total unique values: {df[col].nunique()}")


def _resolve_cement_type(row):
    ct = str(row['cement_type']).strip() if pd.notna(row['cement_type']) else ''
    cg = str(row['cement_grade']).strip() if pd.notna(row['cement_grade']) else ''

    # special types first — grade doesn't override these
    if ct in ('HS_cement', 'OPC_III', 'CEM_II', 'white_cement',
              'pozzolan_cement', 'OPC_I_GGBS', 'BFS_cement'):
        return ct

    # grade column resolves ambiguous OPC_unknown
    if cg in ('52.5', '52.50'):  return 'OPC_52.5'
    if cg in ('42.5', '42.50'):  return 'OPC_42.5'
    if cg in ('53.0', '53'):     return 'OPC_53'

    # already graded types from cement_type
    if ct == 'OPC_52.5':  return 'OPC_52.5'
    if ct == 'OPC_42.5':  return 'OPC_42.5'
    if ct == 'OPC_53':    return 'OPC_53'

    # ungraded OPC_unknown with no grade info → OPC_I
    if ct == 'OPC_unknown':  return 'OPC_I'

    return 'Unknown'


def standardize_cement_type(df):
    """Regex-classify cement_type, then resolve ambiguous grades using cement_grade."""
    df = df.copy()
    df['cement_type'] = df['cement_type'].apply(
        lambda val: val if (pd.isna(val) or val in ('unknown_type', 'not_applicable'))
        else next((label for pat, label in CEMENT_REGEX_MAP if re.search(pat, str(val).lower())), 'OPC_unknown')
    )
    df['cement_type'] = df.apply(_resolve_cement_type, axis=1)
    return df


def standardize_sp_type(df):
    """Regex-classify superplasticizer type into a fixed set of categories."""
    df = df.copy()
    df['sp_type'] = df['sp_type'].apply(
        lambda val: val if (pd.isna(val) or val in ('unknown_type', 'not_applicable'))
        else next((label for pat, label in SP_REGEX_MAP if re.search(pat, val.lower())), 'Unspecified')
    )
    return df


def build_encoding_report(df, out_path=None):
    """Summarize the recommended encoding strategy per column, optionally saving it to csv."""
    encoding_info = []

    for col in df.columns:
        n_unique = df[col].nunique()
        dtype = df[col].dtype

        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            if n_unique == 2:
                encoding = "Binary Encoding"
            elif n_unique <= 10:
                encoding = "One-Hot Encoding"
            else:
                encoding = "Ordinal/Target Encoding"
        elif pd.api.types.is_numeric_dtype(df[col]):
            encoding = "No Encoding"
        else:
            encoding = "Check Manually"

        encoding_info.append([col, dtype, n_unique, encoding])

    encoding_df = pd.DataFrame(
        encoding_info,
        columns=["Column", "Data Type", "Unique Values", "Type of Encoding"]
    )

    if out_path:
        encoding_df.to_csv(out_path, index=False)

    return encoding_df
