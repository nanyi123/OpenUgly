import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')


input_txt = 'alex.txt'
x = []
y1 = []
y2 = []

f = open(input_txt)

for line in f:
    line = line.strip('\n')
    line = line.split(' ')

    x.append(float(line[0]))
    y1.append(float(line[1]))
    y2.append(float(line[2]))

f.close

# plt.plot(x, y1,y2, marker='o', label='lost plot')
plt.xticks(x[0:len(x):2], x[0:len(x):2], rotation=45)
plt.plot(x,y1,"x-",label="accurate")
plt.plot(x,y2,"+-",label="loss")
plt.margins(0)
plt.xlabel("epoch")
plt.ylabel("loss;accurate")
plt.title("CNN for potato")
plt.tick_params(axis="both")
plt.legend(loc='upper right')

plt.show()

