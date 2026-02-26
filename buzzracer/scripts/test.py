from math import radians, degrees


def fun(name, right, left):
    ratio = 54/(right+left)
    offset = -radians(left-right)/2*ratio
    print(f'{name} steer_ratio={ratio},steer_offset={offset}')
    # print(degrees(radians(-right)*ratio + offset))
    # print(degrees(radians(left)*ratio + offset))


fun(13, 26.91, 23.85)
fun(16, 22.0, 22.58)
fun(11, 25.28, 22.24)
