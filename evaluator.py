def evaluate_result(requirements, completed_requirements, test_results):
    total = len(requirements)

    if total == 0:
        coverage = 0
    else:
        coverage = (len(completed_requirements) / total) * 100

    return {
        "requirement_coverage": round(coverage, 2),
        "completed_requirements": completed_requirements,
        "test_results": test_results
    }