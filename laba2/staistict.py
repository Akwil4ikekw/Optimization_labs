import math as mt
nums = [1 ,5, 2, 7, 1, 9, 3, 8, 5, 9]
M_x = (sum(nums))/(len(nums)-1)
print(M_x)
Dispertion = 0
for x in nums:
    Dispertion+=(M_x-x)**2
Dispertion = Dispertion/(len(nums)-1)
print(mt.sqrt(Dispertion))