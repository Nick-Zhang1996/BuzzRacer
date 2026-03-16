from math import radians, degrees,atan


def fun(name, right, left):
    ratio = 54/(right+left)
    offset = -radians(left-right)/2*ratio
    print(f'{name} steer_ratio={ratio},steer_offset={offset}')
    # print(degrees(radians(-right)*ratio + offset))
    # print(degrees(radians(left)*ratio + offset))


fun(13, 26.91, 23.85)
fun(16, 22.0, 22.58)
fun(11, 25.28, 22.24)

L = 98
right = degrees(atan(2*L / (11*25.4 + 73)))
left = degrees(atan(2*L / ((30-15.5)*25.4 + 73)))
print(f'20:  {right=}, {left=}')

right = degrees(atan(2*L / (11.75*25.4 + 75)))
left = degrees(atan(2*L / ((30-13)*25.4 + 75)))
print(f'21:  {right=}, {left=}')

right = degrees(atan(2*L / (11.5*25.4 + 75)))
left = degrees(atan(2*L / ((30-14.5)*25.4 + 75)))
print(f'22:  {right=}, {left=}')