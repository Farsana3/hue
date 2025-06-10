print("FILE: ")
with open("file.txt","w") as f:
    for i in range (5):
        print("Hello",file=f)
        f.flush() # writes it immediately(even before buffer fills)


with open("file.txt","r") as f:
    data=f.read()
    print(data)

print("\nFLUSH: ")
import time

print("With Flush:")
for i in range (5):
    print(" Hello ",i,end='',flush=True)
    time.sleep(1)

print("\nWithout Flush:")
for i in range (5):
    print(" Hello ",i,end='',flush=False) #need end='' cause if not used,the default newline causes flush..won't see difference
    time.sleep(1)
print("\n")


#Arithmetic operators
print("\nARITHMETIC OPERATORS: +, -, *, /, //, %, **")
a=int(input("Enter 2 numbers: "))
b=int(input())
print("sum= ",a+b)
print("difference= ",a-b)
print("product= ",a*b)
print("quotient= ",a/b)
print("integer quotient(floor division)= ",a//b)
print("remainder= ",a%b)
print("a**b(exponentiation)= ",a**b)

#Assignment operators
print("\nASSIGNMENT OPERATORS: =, +=, -=, *=, /=, //=, %=,...")
x=int(input("Enter a number: "))
x=5
print("x=5:    x=",x)
x+=2
print("x+=2:    x=",x)
x-=2
print("x-=2:    x=",x)
x/=3
print("x/=3:    x=",x)
x*=3
print("x*=3:    x=",x)
x//=2
print("x//=3:    x=",x)
x**=3
print("x**=3:    x=",x)

#Comparison operators
print("\nCOMPARISON OPERATORS: ==, !=, >=, <=, <, >")
print("5==5: ",5==5)
print("5!=5: ",5!=5)
print("5>=3: ",5>=3)
print("5<=6: ",5<=6)
print("5<5: ",5<5)
print("7>5: ",7>5)

#Logical operators
print("\nLOGICAL OPERATORS: and, or, not")
print("True and False: ",True and False)
print("True and True: ",True and True)
print("True or False: ",True or False)
print("not False: ",not False)
print("5>=8 and 5<3: ",5>=8 and 5<3)
print("5>=8 or 2<3: ",5>=8 or 2<3)

#Bitwise operators
print("\nBITWISE OPERATORS: &, |, ^, ~, <<, >>")
print("5: 101")
print("3: 011\n")
print("5 & 3 (AND): ",5&3)
print("5 | 3 (OR): ",5|3)
print("5 ^ 3 (XOR): ",5^3)
print("~3 (NOT): ",~3)
print("5 << 1 (LEFT SHIFT): ",5<<1)
print("5 >> 1 (RIGHT SHIFT): ",5>>1)

#Special operators  
print("\nSPECIAL OPERATORS:")
    #Identity operators
print("     IDENTITY OPERATORS: is, is not,==")
a= [1, 2, 3]
b= a
c= [1, 2, 3]
print("a is b : ",a is b)       
print("a is c : ",a is c)   
print("a == c : ",a == c)       
print("a is not c : ",a is not c)  

    #Membership operator
print("     MEMBERSHIP OPERATORS: in, not in")
print("'a' in 'apple': ",'a' in 'apple')  
print("'b' in 'apple': ",'b' in 'apple') 
print("3 in [1, 2, 3, 4] : ",3 in [1, 2, 3, 4]) 
print("5 not in [1, 2, 3] : ",5 not in [1, 2, 3])
