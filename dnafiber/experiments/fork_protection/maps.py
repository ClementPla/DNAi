import re


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


def remap16_pred(name: str) -> str:
    """
    Map condition names to standardized format.

    Examples:
        'sitonsl4_2_25' -> 'siTONSL4 2.25'
        'sitonsl2-1_125' -> 'siTONSL2 1.125'
        'sitonsl4_15' -> 'siTONSL4 15'
        'siNT' -> 'siNT'
    """
    if name.lower() == "sint":
        return "siNT"
    if name == "sitonsl2-2_25":
        return "siTONSL 2.5"  # Weird special case #TODO check upstream

    if name == "sitonsl2-15":
        return "siTONSL 15"  # Weird special case #TODO check upstream

    match = re.match(r"sitonsl(\d*)[-_]([\d_]+)", name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unknown pattern: {name}")

    suffix = match.group(1)  # '2', '4', or ''
    value_raw = match.group(2)  # '15', '2_25', '1_125'

    # Convert underscores to decimal: '1_125' -> '1.125'
    if "_" in value_raw:
        parts = value_raw.split("_")
        value = parts[0] + "." + "".join(parts[1:])
    else:
        value = value_raw

    return f"siTONSL{suffix} {value}"


def remap17_pred(name: str) -> str:
    """
    Map condition names, shortening combined siRNA names.

    Examples:
        'siTONSL+si53BP1' -> 'si5+si53'
        'siTONSL' -> 'siTONSL'
        'siNT' -> 'siNT'
    """
    # Abbreviations used for combined conditions
    abbrev = {
        "siTONSL": "si5",
        "si53BP1": "si53",
    }

    if "+" in name:
        parts = name.split("+")
        shortened = [abbrev.get(p, p) for p in parts]
        return "+".join(shortened)

    return name


def remap20_pred(name: str) -> str:
    """
    Map condition names, shortening combined siRNA names.
    """
    # Explicit mapping for combined conditions
    combo_map = {
        "siTONSL-D+siBRCA1": "siTONS+b1",
        "siTONSL-D+siBRCA2": "siT+B2",
    }

    return combo_map.get(name, name)


def remap25_pred(name: str) -> str:
    # Example usage
    mapping = {
        "si53BP1_12_5": "si53bp1_12.5",
        "si53BP1_7_5": "si53BP1_7.5",
        "siBRCA2_12_5": "siBRCA2_12.5",
        "siBRCA2_7_5": "siBRCA2_7.5",
    }

    return mapping.get(name, name)


def remap21_22_pred(name: str) -> str:
    """
    Map condition names, shortening combined siRNA names.
    """
    # Explicit mapping for combined conditions
    mapping = {
        "U2OS-CTL": "u2os ctl",
        "MMS22L K0-2": "mms22l ko2",
        "MMS22L K0-1": "mms22l ko1",
        "siNT +C5": "siNT+C5",
    }
    return mapping.get(name, name)
