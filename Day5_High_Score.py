student_scores = input().split()
for n in range(0, len(student_scores)):
  student_scores[n] = int(student_scores[n])

Max_Number = 0 
for max in student_scores:
    if (max > Max_Number):
        Max_Number = max

print("The highest score in the class is: ", Max_Number)
