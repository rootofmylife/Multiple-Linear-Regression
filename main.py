import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def computeCost(x_train1, x_train2, y_origin, b00, b11, b22): # loss funtion
    sum = 0
    for i in range(len(y_origin)):
        sum += ((b00 + b11 * x_train1[i] + b22 * x_train2[i]) - y_origin[i])**2
        
    return sum / float((2 * len(y_origin)))

def gradientDescent(x_train1, x_train2, y_origin, b0, b1, b2, learning_rate):
    db0, db1, db2 = 0, 0, 0
    for i in range(len(y_origin)):
        db0 += (b0 + b1 * x_train1[i] + b2 * x_train2[i] - y_origin[i])
        db1 += (b0 + b1 * x_train1[i] + b2 * x_train2[i] - y_origin[i]) * x_train1[i]
        db2 += (b0 + b1 * x_train1[i] + b2 * x_train2[i] - y_origin[i]) * x_train2[i]
        
    # computeGradient
    db0 = db0 / float(len(y_origin))
    db1 = db1 / float(len(y_origin))
    db2 = db2 / float(len(y_origin))

    b_0 = b0 - (learning_rate * db0)
    b_1 = b1 - (learning_rate * db1)
    b_2 = b2 - (learning_rate * db2)
    return b_0, b_1, b_2

if __name__ == '__main__':
    with open('./ex1data2.txt') as f:
        data = f.read().splitlines()

    data = [s.split(',') for s in data]

    dien_tich = [int(s[0]) for s in data] # x_train2
    sl_phong_ngu = [int(s[1]) for s in data] # x_train1
    price = [int(s[2]) for s in data] # y_value

    b0, b1, b2 = 0, 0 , 0
    lr = 0.5

    # chuẩn hóa đặc trưng
    mean_dientich = np.mean(dien_tich)
    mean_slpn = np.mean(sl_phong_ngu)
    mean_price = np.mean(price)

    std_dientich = np.std(dien_tich)
    std_slpn = np.std(sl_phong_ngu)
    std_price = np.std(price)

    dien_tich = [(s - mean_dientich) / std_dientich for s in dien_tich]
    sl_phong_ngu = [(s - mean_slpn) / std_slpn for s in sl_phong_ngu]
    price = [(s - mean_price) / std_price for s in price]

    # learning
    print('1 st Loss: ' + str(computeCost(dien_tich, sl_phong_ngu, price, b0, b1, b2)))
    for it in range(1000):
        b0, b1, b2 = gradientDescent(dien_tich, sl_phong_ngu, price, b0, b1, b2, lr)
            
    print('Last Loss: ' + str(computeCost(dien_tich, sl_phong_ngu, price, b0, b1, b2)))
    print('B0: ' + str(b0) + ' ' + 'B1: ' + str(b1) + ' ' + 'B2: ' + str(b2))

    # Graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y = np.meshgrid(np.linspace(min(dien_tich), max(dien_tich)), np.linspace(min(sl_phong_ngu), max(sl_phong_ngu)))
    z = b0 + b1 * x + b2 * y

    ax.plot_surface(x, y, z, alpha=0.4)
    ax.scatter(dien_tich, sl_phong_ngu, price, c='r', marker='o')

    ax.set_xlabel('Area')
    ax.set_ylabel('Rooms')
    ax.set_zlabel('Prices')

    plt.show()