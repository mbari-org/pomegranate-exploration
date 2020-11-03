import numpy

# https://github.com/jmschrei/pomegranate/issues/673#issuecomment-599281188
def generate_random_distribution(observations):
    dist = {}
    random_probs = numpy.abs(numpy.random.randn(len(observations)))
    random_probs /= random_probs.sum()
    for obs, p in zip(observations, random_probs):
        dist[obs] = p
    return dist
