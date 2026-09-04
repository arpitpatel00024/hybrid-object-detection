def fitness_function(threshold, detections):
    score = 0

    for det in detections:
        if det["confidence"] > threshold:
            score += 1

    return score