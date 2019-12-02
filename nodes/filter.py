from scipy import signal 
import matplotlib.pyplot as plt
import pickle
# argument: order, omega(-3db)
b, a = signal.butter(1,6,'low',analog=False,fs=100)
print(b,a)
z = signal.lfilter_zi(b,a)
z = [0]

filename = "longitudinal.p"
infile = open(filename,'rb')
data = pickle.load(infile)
ori_data = data[:]
result = []
previous = data[0]
for i in range(len(data)):
    pending = data[i]
# update 
    if (i>1 and abs(pending-previous)>0.5):
        #data[i] = data[i-1]
        pending = previous
    #val, z = signal.lfilter(b,a,[data[i]],zi=z)
    val, z = signal.lfilter(b,a,[pending],zi=z)
    result.append(val)
    previous = pending

#plt.subplot(211)
plt.plot(data)
#plt.subplot(212)
plt.plot(result)
plt.show()

