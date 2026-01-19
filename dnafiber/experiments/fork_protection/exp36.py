def map_condition_name_gt(short_name: str) -> str:
    """
    Map short-form condition names to full names.

    Examples:
        'H-siNT' -> 'Hela_siNT'
        'H-siM#1' -> 'Hela_siMMS22L#1'
        'U-siT#2' -> 'U2OS_siTONSL#2'
    """
    # Cell line mapping
    if short_name.startswith("H-"):
        cell_line = "Hela"
        rest = short_name[2:]
    elif short_name.startswith("U-"):
        cell_line = "U2OS"
        rest = short_name[2:]
    else:
        raise ValueError(f"Unknown cell line prefix: {short_name}")

    # Target mapping
    if rest.startswith("siNT"):
        target = "siNT"
        suffix = rest[4:]
    elif rest.startswith("siM"):
        target = "siMMS22L"
        suffix = rest[3:]
    elif rest.startswith("siT"):
        target = "siTONSL"
        suffix = rest[3:]
    else:
        raise ValueError(f"Unknown target in: {short_name}")

    # Handle '#1AB3' -> '#AB3' quirk
    if suffix == "#1AB3":
        suffix = "#AB3"

    return f"{cell_line}_{target}{suffix}"
