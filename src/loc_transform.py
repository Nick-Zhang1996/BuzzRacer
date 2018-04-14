import math

def transform(x, y):
    # alpha = 20 degrees, verticalFOV(vFov) = 15 degrees, horizontalFOV(hFov) = 15 degrees, h = 5.4 cm
    alpha = 3
    vFov = 27.0
    hFov = 40.0
    h = 5.4

    ob = h / math.cos(math.radians(90 - alpha - vFov))
    op = math.cos(math.radians(vFov)) * ob
    bp = math.sin(math.radians(vFov)) * ob

    if y > 0 and y <= 240:
        angle = math.degrees(math.atan((240-y)/240.0*bp/op)) + 90.0 - alpha
        actualY = math.tan(math.radians(angle))*h
    else:
        angle = 90 - alpha - math.degrees(math.atan((y-240)/240*bp/op))
        actualY = math.tan(math.radians(angle))*h

    om = actualY * math.tan(math.radians(hFov))
    
    if x > 0 and x <= 320:
        actualX = -(320-x)/320.0*om
    else:
        actualX = (x-320)/320.0*om
        
    actualY = actualY + 14
    
    return actualX, actualY

x,y = transform(344, 303)
#print(y)

                   
    
