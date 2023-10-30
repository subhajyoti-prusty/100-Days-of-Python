# 1st input: enter height in meters e.g: 1.65
height = input()
# 2nd input: enter weight in kilograms e.g: 72
weight = input()

# Calculation part
Weight= int(weight)
Height = float(height)

bmi = Weight / Height ** 2
BMI = int(bmi)
print(BMI)