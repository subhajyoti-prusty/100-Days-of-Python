student_heights = input().split()
for n in range(0, len(student_heights)):
  student_heights[n] = int(student_heights[n])
  
total_height = 0
count = 0
for height in student_heights:
  total_height += height
  count += 1
print(f"total height = {total_height}")
print(f"number of students = {count}")
average_height = round(total_height / count)
print(f"average height = {average_height}")