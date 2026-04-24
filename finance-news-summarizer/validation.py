ALLOWED_SENTIMENTS = frozenset({"positive", "negative", "neutral"})

REQUIRED_KEYS = {
    "summary",
    "key_points",
    "sentiment",
    "entities",
    "why_it_matters",
}


def validate_response(response):
    """Validate the structured JSON produced by the summarizer."""
    if not isinstance(response, dict):
        return False, "output is not a JSON object"

    if set(response.keys()) != REQUIRED_KEYS:
        return False, f"keys mismatch found: {set(response.keys())}"

    if not isinstance(response["summary"], str) or not response["summary"].strip():
        return False, "summary must be a non-empty string"

    key_points = response["key_points"]
    if not isinstance(key_points, list):
        return False, "key_points must be a list"
    if not 3 <= len(key_points) <= 5:
        return False, "key_points must contain 3 to 5 items"
    if not all(isinstance(item, str) and item.strip() for item in key_points):
        return False, "all key_points items must be non-empty strings"

    if response["sentiment"] not in ALLOWED_SENTIMENTS:
        return False, f"sentiment must be one of {sorted(ALLOWED_SENTIMENTS)}"

    entities = response["entities"]
    if not isinstance(entities, list):
        return False, "entities must be a list"
    if not all(isinstance(item, str) and item.strip() for item in entities):
        return False, "all entities must be non-empty strings"

    if not isinstance(response["why_it_matters"], str) or not response[
        "why_it_matters"
    ].strip():
        return False, "why_it_matters must be a non-empty string"

    return True, "valid"
