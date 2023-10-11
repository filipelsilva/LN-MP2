from enum import Enum

class Classification(Enum):
    TRUTHFULPOSITIVE  = 1
    TRUTHFULNEGATIVE  = 2
    DECEPTIVEPOSITIVE = 3
    DECEPTIVENEGATIVE = 4

def splitStringWithClassification(string):
    split = string.strip().split("\t")
    return [Classification[split[0]], split[1]]

with open("./train.txt", "r") as f:
    lines = list(map(splitStringWithClassification, f.readlines()))

print(lines)
