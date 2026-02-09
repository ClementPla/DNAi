def remap35_gt(short_name):
    """Map short condition names to full names.

    Examples:
        'U-siM#1' -> 'U2OS_siMMS22L#1'
        'H-siT#4' -> 'Hela_siTONSL#4'
    """
    # Cell line prefix
    if short_name.startswith("U-"):
        prefix = "U2OS_"
        rest = short_name[2:]
    elif short_name.startswith("H-"):
        prefix = "Hela_"
        rest = short_name[2:]
    else:
        raise ValueError(f"Unknown cell line prefix: {short_name}")

    # Condition mapping
    if rest == "siNT":
        condition = "siNT"
    elif rest.startswith("siM#"):
        condition = rest.replace("siM#", "siMMS22L#")
    elif rest.startswith("siT#"):
        condition = rest.replace("siT#", "siTONSL#")
    elif rest.startswith("T#"):  # Handle 'H-T#AB3' variant
        condition = rest.replace("T#", "siTONSL#")
    else:
        raise ValueError(f"Unknown condition: {rest}")

    return prefix + condition


def remap36_gt(short_name: str) -> str:
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


def remap39_gt(short_name):
    """Map short condition names to full names.

    Examples:
        'PEO1-siNT' -> 'PEO1-siNT'
        'PEO1-siT2' -> 'PEO1-siTONSL#2'
        'PE04-siNT' -> 'PEO4_siNT'
        'PEO4-siT2' -> 'PEO4_siTONSL#2'
    """
    # Cell line prefix
    if short_name.startswith("PEO1-"):
        prefix = "PEO1-"
        rest = short_name[5:]
    elif short_name.startswith("PEO4-") or short_name.startswith("PE04-"):
        prefix = "PEO4_"
        rest = short_name[5:]
    else:
        raise ValueError(f"Unknown cell line prefix: {short_name}")

    # Condition mapping
    if rest == "siNT":
        condition = "siNT"
    elif rest.startswith("siT"):
        # siT2 -> siTONSL#2, siT4 -> siTONSL#4, etc.
        number = rest[3:]  # extract "2" from "siT2"
        condition = f"siTONSL#{number}"
    elif rest.startswith("siM"):
        number = rest[3:]
        condition = f"siMMS22L#{number}"
    else:
        raise ValueError(f"Unknown condition: {rest}")

    return prefix + condition
