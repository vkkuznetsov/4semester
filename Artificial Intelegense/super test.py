import random

random.seed(10)
print("".join(str([random.randint(-100, 100) for i in range(20)])))
