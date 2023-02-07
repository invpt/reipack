from collections import Counter

count = Counter()
for line in open("out.csv", "r").readlines():
    score,packing,order = line.strip().split(",")
    if int(score) < 6:
        count[order] += 1

for key, val in sorted(count.items(), key=lambda x: x[1], reverse=True):
    print(f"{key}: {val}")
