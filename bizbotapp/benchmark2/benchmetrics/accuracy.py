def evaluate_keywords(response, expected_keywords):
    matches = sum(kw.lower() in response.lower() for kw in expected_keywords)
    return round(matches / len(expected_keywords), 2) if expected_keywords else 0.0
