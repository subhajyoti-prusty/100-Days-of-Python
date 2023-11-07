target = int(input()) # Enter a number of players

for num in range(1,target+1):
    if num % 3 == 0 and num % 5 == 0:
        print("FizzBuzz")
    elif num % 3 == 0:
        print("Fizz")
    elif num % 5 == 0:
        print("Buzz")   
    else:
        print(str(num))
