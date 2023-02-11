from collections import Counter

count = Counter()
for line in open("out.csv", "r").readlines():
    score,waste,packing,order,closeness = line.strip().split(",")
    if int(score) == 0:
        count[order] += 1

for key, val in sorted(count.items(), key=lambda x: x[1], reverse=True):
    print(f"{key}: {val}")
