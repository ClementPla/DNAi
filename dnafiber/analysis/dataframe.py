import pandas as pd


def get_label(i):
    label = ""
    while i >= 0:
        label = chr(65 + (i % 26)) + label
        i = (i // 26) - 1
    return label


def anonymise_types(df: pd.DataFrame, exceptions=None):
    if exceptions is None:
        exceptions = []
    if not isinstance(exceptions, list):
        exceptions = [exceptions]
    if df.Type.dtype.name != "category":
        df["Type"] = df["Type"].astype("category")
    current_ordered_types = df.Type.cat.categories

    type_mapping = {}
    exp_counter = 0

    # 2. Assign names based on that specific order
    for t in current_ordered_types:
        if t in exceptions:
            # Keep the name as is
            type_mapping[t] = t
        else:
            # Assign the next available Exp letter
            type_mapping[t] = f"Exp {get_label(exp_counter)}"
            exp_counter += 1

    # 3. Apply the mapping
    # We convert to string first to avoid categorical conflicts
    df["Type"] = df["Type"].astype(str).map(type_mapping)

    # 4. Re-establish the categorical order using the NEW names
    # This ensures 'Exp A' < 'Exp B' < 'siNT' < 'Exp C' etc.
    new_categories = [type_mapping[t] for t in current_ordered_types]
    df["Type"] = pd.Categorical(df["Type"], categories=new_categories, ordered=True)

    return df
