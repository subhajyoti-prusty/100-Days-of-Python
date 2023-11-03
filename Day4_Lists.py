States_In_India = ["Odisha","delhi","Chhattisgarh","Goa","Punjab","Gujarat"]
print(States_In_India[0])
print(States_In_India[1])
print(States_In_India[2])
print(States_In_India[3])
print(States_In_India[4])
print(States_In_India[5])
print(States_In_India[-1])
print(States_In_India[-2])
print(States_In_India[-3])

States_In_India[1] = "Delhi"
print(States_In_India[1])
print(States_In_India)

States_In_India.append("West Bengal")
print(States_In_India)

States_In_India.extend(["Assam","Bihar","Jammu & Kashmir","Maharashtra","karnataka"])
print(States_In_India)

print(len(States_In_India))