import matplotlib.pyplot as plt
from math import sqrt

pmf_1 = [
    (1, 0.1), # p(1 satellite) = 0.1
    (2, 0.15),
    (3, 0.25),
    (4, 0.25),
    (5, 0.15),
    (6, 0.1)
]

pmf_2 = [
    (3, 0.5),
    (4, 0.5),
]

def expected_value(pmf):
    ev = 0.0
    for val, prob in pmf:
        ev += val * prob
    return ev

print(expected_value(pmf_1))
print(expected_value(pmf_2))

def variance(pmf):
    mean = expected_value(pmf)
    var = 0.0
    for val, prob in pmf:
        var += (val-mean)**2 * prob
    return var

def standard_dev(pmf):
    return sqrt(variance(pmf))


print("PMF 1 var:", variance(pmf_1), "std:", standard_dev(pmf_1))
print("PMF 2 var:", variance(pmf_2), "std:", standard_dev(pmf_2))

def plot_pmf(pmf):
    labels = [outcome[0] for outcome in pmf ]
    probs  = [outcome[1] for outcome in pmf ]
    plt.bar(labels,probs)

plot_pmf(pmf_2)
plot_pmf(pmf_1)
plt.ylabel('probability')
plt.xlabel('number of satellites')
plt.legend(["PMF 2", "PMF 1"])
plt.show()
