import random

def optimize(detections):
    history = []

    population = [random.uniform(0.2, 0.8) for _ in range(10)]

    for _ in range(5):
        # Fitness: count detections above threshold
        population = sorted(
            population,
            key=lambda x: sum(d["confidence"] > x for d in detections),
            reverse=True
        )

        best = population[0]
        history.append(best)

        # Selection
        population = population[:5]

        # Mutation
        while len(population) < 10:
            parent = random.choice(population)
            mutation = parent + random.uniform(-0.1, 0.1)
            mutation = max(0, min(1, mutation))
            population.append(mutation)

    return population[0], history