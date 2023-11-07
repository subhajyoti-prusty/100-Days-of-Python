target = int(input()) # Enter a number between 0 and 1000

Sum_even = 0
for num in range(0,target+1):
    if (num % 2 == 0):
        Sum_even += num
        
print(Sum_even)