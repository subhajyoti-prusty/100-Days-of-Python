print("The Love Calculator is calculating your score...")
name1 = input() # What is your name?
name2 = input() # What is their name?

Name = name1 + name2
Name_In_Lowercase = Name.lower()

t = Name_In_Lowercase.count("t")
r = Name_In_Lowercase.count("r")
u = Name_In_Lowercase.count("u")
e = Name_In_Lowercase.count("e")
first = t + r + u + e

l = Name_In_Lowercase.count("l")
o = Name_In_Lowercase.count("o")
v = Name_In_Lowercase.count("v")
e = Name_In_Lowercase.count("e")
second = l + o + v + e

love_score = str(first) + str(second)
love_score = int(love_score)

if (love_score < 10) or (love_score > 90):
  print(f"Your score is {love_score}, you go together like coke and mentos.")
elif (love_score >= 40) and (love_score <= 50):
  print(f"Your score is {love_score}, you are alright together.")
else:
  print(f"Your score is {love_score}.")
  
print(f"{name1} and {name2}'s love score is {love_score}")