# Take a two digit number as input. Write the code such as it will give the output as the sum of both the digits of the number 
two_digit_number = input("Type a two digit number: ")

#             Method 1
# new_number = str(two_digit_number)
# index1 = new_number[0]
# index2 = new_number[1]
# final_numbre1 = int(index1)
# final_numbre2 = int(index2)
# Sum = final_numbre1 + final_numbre2
# print(Sum)

#             Method 2

print(type(two_digit_number))

first_digit = int(two_digit_number[0])

second_digit = int(two_digit_number[1])

sum = first_digit + second_digit

print(sum)