def map_condition_name_gt(short_name):
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
