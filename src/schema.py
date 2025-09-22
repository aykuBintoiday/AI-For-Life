from dataclasses import dataclass
from typing import Optional

@dataclass
class Destination:
    id: int
    name: str
    city: str
    tags: str
    category: str
    avg_cost: int
    min_days: int
    description: str
    rating: float
    lat: float
    lon: float

def unify_text(name, city, tags, category, desc):
    tags = tags.replace(";", " ")
    return f"{name} {city} {tags} {category} {desc}"
