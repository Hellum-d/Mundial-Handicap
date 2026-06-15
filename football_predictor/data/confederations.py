"""National-team to confederation mapping for hierarchical pooling.

Used by the Dixon-Coles model to shrink each team's attack/defence strength
toward its *confederation* mean rather than the global mean (partial pooling):
a team with few recent matches then inherits the level of its confederation
(e.g. a sparsely-observed CONMEBOL side is assumed nearer the strong South
American baseline than the global average).

Coverage is deliberately broad but not exhaustive; unmapped teams fall back to
``OTHER`` and are pooled together. Common naming variants from the martj42
dataset are included so the lookup is robust.
"""

from __future__ import annotations

# confederation -> member national teams (with naming variants).
_MEMBERS: dict[str, tuple[str, ...]] = {
    "UEFA": (
        "Germany", "France", "Spain", "England", "Italy", "Netherlands",
        "Portugal", "Belgium", "Croatia", "Switzerland", "Denmark", "Poland",
        "Serbia", "Sweden", "Wales", "Scotland", "Austria", "Ukraine",
        "Czech Republic", "Czechia", "Russia", "Turkey", "Türkiye",
        "Republic of Ireland", "Northern Ireland", "Norway", "Greece",
        "Romania", "Hungary", "Slovakia", "Slovenia", "Iceland", "Finland",
        "Bosnia and Herzegovina", "North Macedonia", "Macedonia", "Albania",
        "Bulgaria", "Montenegro", "Georgia", "Kosovo", "Luxembourg", "Cyprus",
        "Estonia", "Latvia", "Lithuania", "Belarus", "Moldova", "Armenia",
        "Azerbaijan", "Israel", "Malta", "Andorra", "San Marino",
        "Liechtenstein", "Gibraltar", "Faroe Islands", "Kazakhstan",
    ),
    "CONMEBOL": (
        "Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Peru",
        "Paraguay", "Ecuador", "Bolivia", "Venezuela",
    ),
    "CONCACAF": (
        "Mexico", "United States", "USA", "Costa Rica", "Canada", "Honduras",
        "Panama", "Jamaica", "El Salvador", "Trinidad and Tobago", "Guatemala",
        "Haiti", "Curaçao", "Curacao", "Suriname", "Nicaragua", "Cuba",
    ),
    "CAF": (
        "Senegal", "Morocco", "Tunisia", "Algeria", "Egypt", "Nigeria",
        "Cameroon", "Ghana", "Ivory Coast", "Côte d'Ivoire", "Mali",
        "Burkina Faso", "South Africa", "DR Congo", "Congo DR", "Guinea",
        "Cape Verde", "Zambia", "Gabon", "Uganda", "Kenya", "Angola",
        "Mozambique", "Zimbabwe", "Ethiopia", "Sudan", "Benin", "Togo",
        "Madagascar", "Mauritania", "Namibia", "Equatorial Guinea", "Congo",
        "Tanzania", "Libya", "Sierra Leone", "Malawi", "Niger", "Liberia",
    ),
    "AFC": (
        "Japan", "South Korea", "Korea Republic", "Iran", "Australia",
        "Saudi Arabia", "Qatar", "Iraq", "United Arab Emirates", "China",
        "China PR", "Uzbekistan", "Oman", "Jordan", "Bahrain", "Syria",
        "Lebanon", "Kuwait", "India", "Thailand", "Vietnam", "Palestine",
        "Tajikistan", "Kyrgyzstan", "North Korea", "Korea DPR", "Indonesia",
        "Malaysia", "Philippines", "Yemen", "Hong Kong", "Myanmar",
    ),
    "OFC": (
        "New Zealand", "Fiji", "Papua New Guinea", "Solomon Islands",
        "Tahiti", "Vanuatu", "New Caledonia", "Samoa", "Tonga", "Cook Islands",
    ),
}

# Flattened lookup: team -> confederation.
_LOOKUP: dict[str, str] = {
    team: conf for conf, members in _MEMBERS.items() for team in members
}

OTHER = "OTHER"


def confederation_of(team: str) -> str:
    """Return the confederation for ``team``, or ``OTHER`` if unmapped."""
    return _LOOKUP.get(team, OTHER)
