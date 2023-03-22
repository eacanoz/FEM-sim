x = lambda n: n+2
y = 1

def verification(var):
    if callable(var):
        print('Is a function')
        return var(8)
    else:
        print('Isn\'t a function')
        return var
    

print(verification(x))
print(verification(y))