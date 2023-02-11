"""
Calculates the corellation coefficient between the score and the closeness quotient
"""

import numpy as np
import matplotlib.pyplot as plt

scores = []
closenesses = []
for line in open("out.csv", "r").readlines():
    score,spread_score,packing,order,closeness = line.strip().split(",")
    scores.append(int(score))
    closenesses.append(float(spread_score))

print(np.corrcoef(scores, closenesses))

plt.scatter(scores, closenesses)
plt.xlabel("Area Score")
plt.ylabel("Spread Score")
plt.show()
