from enum import Enum

class Honesty(Enum):
    TRUTHFUL = 1
    DECEPTIVE = 2

    @staticmethod
    def get(string):
        if Honesty.TRUTHFUL.name in string:
            return Honesty.TRUTHFUL
        return Honesty.DECEPTIVE

class Polarity(Enum):
    POSITIVE = 1
    NEGATIVE = 2

    @staticmethod
    def get(string):
        if Polarity.POSITIVE.name in string:
            return Polarity.POSITIVE
        return Polarity.NEGATIVE

def splitStringWithClassification(string):
    split = string.strip().split("\t")
    return [[Honesty.get(split[0]), Polarity.get(split[0])], split[1]]

def getData():
    with open("./train.txt", "r") as f:
        lines = list(map(splitStringWithClassification, f.readlines()))

    return lines

def main():
    data = getData()
    pass

if __name__ == "__main__":
    main()
