from scipy import signal 
import matplotlib.pyplot as plt
import pickle
# argument: order, omegan(-3db)
b, a = signal.butter(1,3,'low',analog=False,fs=100)
print(b,a)
z = signal.lfilter_zi(b,a)
z = [0]

infile = open('velocity.p','rb')
data = pickle.load(infile)
ori_data = data[:]
result = []
for i in range(len(data)):
    #pending = data[i]
    if (i>1 and abs(data[i]-data[i-1])>0.5):
        data[i] = data[i-1]
        #pending = data[i-1]
    val, z = signal.lfilter(b,a,[data[i]],zi=z)
    #val, z = signal.lfilter(b,a,[pending],zi=z)
    result.append(val)

plt.subplot(211)
plt.plot(ori_data)
plt.subplot(212)
plt.plot(result)
plt.show()

