def map_condition_name_gt(short_name):
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
