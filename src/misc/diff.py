from sympy import cos,sin,diff,symbols

A = symbols('A')
B = symbols('B')
C = symbols('C')
D = symbols('D')
d = symbols('d')
x0 = symbols('y0')
y0 = symbols('x0')

#fun = (A*sin(d)+B*cos(d))/(C*sin(d)+D*cos(d))
#print(fun.diff(d))
xp =  1/(-A*cos(d)-B*sin(d)) *(-C*cos(d)-B*sin(d)*x0+B*cos(d)*y0)
yp =  1/(-A*cos(d)-B*sin(d)) *(-C*sin(d)+A*sin(d)*x0-A*cos(d)*y0)
print(xp.diff(d))
print(yp.diff(d))


