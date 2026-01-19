def remap43_gt(short_name):
    """Map short condition names to full names.

    Examples:
        'U2OS+S1' -> 'U2OS_+S1'
        'u2os' -> 'U2OS_-S1'
        'MKO#1 +S1' -> 'U2OS_MMS22LKO#1_+S1'
        'MKO #4 ' -> 'U2OS_MMS22LKO#4_-S1'
    """
    # Normalize: uppercase and remove spaces
    normalized = short_name.upper().replace(" ", "")

    # Determine S1 status
    if "+S1" in normalized:
        s1_suffix = "_+S1"
        normalized = normalized.replace("+S1", "")
    else:
        s1_suffix = "_-S1"

    # Cell line / condition
    if normalized == "U2OS":
        condition = "U2OS"
    elif normalized.startswith("MKO#"):
        number = normalized[4:]  # extract number after MKO#
        condition = f"U2OS_MMS22LKO#{number}"
    else:
        raise ValueError(f"Unknown condition: {short_name}")

    return condition + s1_suffix


def remap42_gt(short_name):
    """Map short condition names to full names.

    Examples:
        'U2OS_+s1' -> 'U2OS_+S1'
        'U2OS' -> 'U2OS_-S1'
        'MKO#1 +S1' -> 'U2OS_MMS22LKO#1_+S1'
        'mko#6' -> 'U2OS_MMS22LKO#6_-S1'
    """
    # Normalize: uppercase and remove spaces/underscores
    normalized = short_name.upper().replace(" ", "").replace("_", "")

    # Determine S1 status
    if "+S1" in normalized:
        s1_suffix = "_+S1"
        normalized = normalized.replace("+S1", "")
    else:
        s1_suffix = "_-S1"

    # Cell line / condition
    if normalized == "U2OS":
        condition = "U2OS"
    elif normalized.startswith("MKO#"):
        number = normalized[4:]
        condition = f"U2OS_MMS22LKO#{number}"
    else:
        raise ValueError(f"Unknown condition: {short_name}")

    return condition + s1_suffix


def remap44_gt(short_name):
    """Map short condition names to full names.

    Examples:
        'U2OS siNT +s1' -> 'U2OS_siNT_+S1'
        'siT#2 +S1' -> 'U2OS_siTONSL#2_+S1'
        'siM#1 +S1' -> 'U2OS_siMMS22L#1_+S1'
        'U2OS' -> 'U2OS_siNT_-S1'
        'siT2' -> 'U2OS_siTONSL#2_-S1'
        'siM1' -> 'U2OS_siMMS22L#1_-S1'
    """
    # Normalize: uppercase and remove spaces
    normalized = short_name.upper().replace(" ", "")

    # Determine S1 status
    if "+S1" in normalized:
        s1_suffix = "_+S1"
        normalized = normalized.replace("+S1", "")
    else:
        s1_suffix = "_-S1"

    # Remove U2OS prefix if present
    normalized = normalized.replace("U2OS", "")

    # Condition mapping
    if normalized == "" or normalized == "SINT":
        condition = "siNT"
    elif normalized.startswith("SIT#"):
        number = normalized[4:]
        condition = f"siTONSL#{number}"
    elif normalized.startswith("SIT"):
        number = normalized[3:]
        condition = f"siTONSL#{number}"
    elif normalized.startswith("SIM#"):
        number = normalized[4:]
        condition = f"siMMS22L#{number}"
    elif normalized.startswith("SIM"):
        number = normalized[3:]
        condition = f"siMMS22L#{number}"
    else:
        raise ValueError(f"Unknown condition: {short_name}")

    return f"U2OS_{condition}{s1_suffix}"


def remap47_gt(short_name):
    """Map short condition names to full names.

    Examples:
        'U2OS +S1' -> 'U2OS_+S1'
        'MKO1 +S1' -> 'U2OS_MMS22LKO#1_+S1'
        'mk04 +s1' -> 'U2OS_MMS22LKO#4_+S1'
        'MK06' -> 'U2OS_MMS22LKO#6_-S1'
        'u2os' -> 'U2OS_-S1'
    """
    # Normalize: uppercase and remove spaces
    normalized = short_name.upper().replace(" ", "")

    # Determine S1 status
    if "+S1" in normalized:
        s1_suffix = "_+S1"
        normalized = normalized.replace("+S1", "")
    else:
        s1_suffix = "_-S1"

    # Normalize O/0 confusion: MK0 -> MKO
    normalized = normalized.replace("MK0", "MKO")

    # Cell line / condition
    if normalized == "U2OS":
        condition = "U2OS"
    elif normalized.startswith("MKO"):
        number = normalized[3:]
        condition = f"U2OS_MMS22LKO#{number}"
    else:
        raise ValueError(f"Unknown condition: {short_name}")

    return condition + s1_suffix
