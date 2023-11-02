print("Thank you for choosing Python Pizza Deliveries!")

print("Enter the Size of the (S,M,L):")
size = input() # What size pizza do you want? S, M, or L
if size == 'S':
    price = 15
elif size == 'M':
    price = 20
else:
    price = 25
price = int(price)

print("Do you want pepperoni Y or N:")
add_pepperoni = input() # Do you want pepperoni? Y or N
if size == 'S':
    if add_pepperoni == 'Y' :
        price += 2
else:
    if add_pepperoni == 'Y' :
        price += 3
    
print("Do you want extra cheese Y or N:")
extra_cheese = input() # Do you want extra cheese? Y or N
if extra_cheese == 'Y':
    price += 1
else:
    pass
bill = int(price)
print(f"Your final bill is: ${bill}.")
        