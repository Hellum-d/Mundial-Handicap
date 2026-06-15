"""National-team name to ISO country code, for flag images in the web UI.

Codes are lowercase ISO 3166-1 alpha-2 (e.g. ``br``), with the four UK home
nations using flagcdn's subdivision codes (``gb-eng`` etc.). They feed flag
image URLs like ``https://flagcdn.com/24x18/br.png`` — actual images, because
Windows browsers do not render flag *emoji* as flags. Unmapped teams return
``None`` (no flag shown). Naming variants from the martj42 dataset are included.
"""

from __future__ import annotations

_CODES: dict[str, str] = {
    # UEFA
    "Germany": "de", "France": "fr", "Spain": "es", "England": "gb-eng",
    "Italy": "it", "Netherlands": "nl", "Portugal": "pt", "Belgium": "be",
    "Croatia": "hr", "Switzerland": "ch", "Denmark": "dk", "Poland": "pl",
    "Serbia": "rs", "Sweden": "se", "Wales": "gb-wls", "Scotland": "gb-sct",
    "Austria": "at", "Ukraine": "ua", "Czech Republic": "cz", "Czechia": "cz",
    "Russia": "ru", "Turkey": "tr", "Türkiye": "tr",
    "Republic of Ireland": "ie", "Ireland": "ie", "Northern Ireland": "gb-nir",
    "Norway": "no", "Greece": "gr", "Romania": "ro", "Hungary": "hu",
    "Slovakia": "sk", "Slovenia": "si", "Iceland": "is", "Finland": "fi",
    "Bosnia and Herzegovina": "ba", "North Macedonia": "mk", "Macedonia": "mk",
    "Albania": "al", "Bulgaria": "bg", "Montenegro": "me", "Georgia": "ge",
    "Kosovo": "xk", "Luxembourg": "lu", "Cyprus": "cy", "Estonia": "ee",
    "Latvia": "lv", "Lithuania": "lt", "Belarus": "by", "Moldova": "md",
    "Armenia": "am", "Azerbaijan": "az", "Israel": "il", "Malta": "mt",
    "Andorra": "ad", "San Marino": "sm", "Liechtenstein": "li",
    "Gibraltar": "gi", "Faroe Islands": "fo", "Kazakhstan": "kz",
    # CONMEBOL
    "Brazil": "br", "Argentina": "ar", "Uruguay": "uy", "Colombia": "co",
    "Chile": "cl", "Peru": "pe", "Paraguay": "py", "Ecuador": "ec",
    "Bolivia": "bo", "Venezuela": "ve",
    # CONCACAF
    "Mexico": "mx", "United States": "us", "USA": "us", "Costa Rica": "cr",
    "Canada": "ca", "Honduras": "hn", "Panama": "pa", "Jamaica": "jm",
    "El Salvador": "sv", "Trinidad and Tobago": "tt", "Guatemala": "gt",
    "Haiti": "ht", "Curaçao": "cw", "Curacao": "cw", "Suriname": "sr",
    "Nicaragua": "ni", "Cuba": "cu",
    # CAF
    "Senegal": "sn", "Morocco": "ma", "Tunisia": "tn", "Algeria": "dz",
    "Egypt": "eg", "Nigeria": "ng", "Cameroon": "cm", "Ghana": "gh",
    "Ivory Coast": "ci", "Côte d'Ivoire": "ci", "Mali": "ml",
    "Burkina Faso": "bf", "South Africa": "za", "DR Congo": "cd",
    "Congo DR": "cd", "Guinea": "gn", "Cape Verde": "cv", "Zambia": "zm",
    "Gabon": "ga", "Uganda": "ug", "Kenya": "ke", "Angola": "ao",
    "Mozambique": "mz", "Zimbabwe": "zw", "Ethiopia": "et", "Sudan": "sd",
    "Benin": "bj", "Togo": "tg", "Madagascar": "mg", "Mauritania": "mr",
    "Namibia": "na", "Equatorial Guinea": "gq", "Congo": "cg",
    "Tanzania": "tz", "Libya": "ly", "Sierra Leone": "sl", "Malawi": "mw",
    "Niger": "ne", "Liberia": "lr",
    # AFC
    "Japan": "jp", "South Korea": "kr", "Korea Republic": "kr", "Iran": "ir",
    "Australia": "au", "Saudi Arabia": "sa", "Qatar": "qa", "Iraq": "iq",
    "United Arab Emirates": "ae", "China": "cn", "China PR": "cn",
    "Uzbekistan": "uz", "Oman": "om", "Jordan": "jo", "Bahrain": "bh",
    "Syria": "sy", "Lebanon": "lb", "Kuwait": "kw", "India": "in",
    "Thailand": "th", "Vietnam": "vn", "Palestine": "ps", "Tajikistan": "tj",
    "Kyrgyzstan": "kg", "North Korea": "kp", "Korea DPR": "kp",
    "Indonesia": "id", "Malaysia": "my", "Philippines": "ph", "Yemen": "ye",
    "Hong Kong": "hk", "Myanmar": "mm",
    # OFC
    "New Zealand": "nz", "Fiji": "fj", "Papua New Guinea": "pg",
    "Solomon Islands": "sb", "Tahiti": "pf", "Vanuatu": "vu",
    "New Caledonia": "nc", "Samoa": "ws", "Tonga": "to", "Cook Islands": "ck",
}


def flag_code(team: str) -> str | None:
    """Return the flag image code for ``team`` (``None`` if unmapped)."""
    return _CODES.get(team)
