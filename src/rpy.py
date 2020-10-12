from scipy.spatial.transform import Rotation

r = Rotation.from_euler("ZYX",[180,0,90],degrees=True)
x = [-3,0,1]
a,b,c = r.apply(x,inverse=True)
print(a,b,c)
