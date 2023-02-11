"""
Calculates the corellation coefficient between the score and the closeness quotient
"""

import numpy as np
import matplotlib.pyplot as plt

scores = []
closenesses = []
for line in open("out.csv", "r").readlines():
    score,waste,packing,order,closeness = line.strip().split(",")
    scores.append(int(score))
    closenesses.append(float(closeness))

print(np.corrcoef(scores, closenesses))

plt.scatter(scores, closenesses)
plt.xlabel("Area Score")
plt.ylabel("Closeness Score")
plt.show()
