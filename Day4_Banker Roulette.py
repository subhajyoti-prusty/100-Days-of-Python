import random

names_string = input()
names = names_string.split(", ")

random_input = len(names)
random_name = random.randint(0, random_input - 1)
print(f"{names[random_name]} is going to buy the meal today!")