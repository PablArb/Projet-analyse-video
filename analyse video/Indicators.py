from Modules import *

def rectangle_NB(image, extremas, rectanglewidth):
    L = len(image)
    l = len(image[0])
    marge = 4
    for key in extremas:
        xmin, ymin, xmax, ymax = int(extremas[key][0])-marge, int(extremas[key][1])-marge, int(extremas[key][2])+marge,\
                                 int(extremas[key][3])+marge
        for i in range(xmin - rectanglewidth, xmax + rectanglewidth + 1):
            for n in range(rectanglewidth + 1):
                image[(ymin - n) % L][i % l], image[(ymax + n) % L][i % l] = 255, 255
        for j in range(ymin - rectanglewidth, ymax + rectanglewidth + 1):
            for n in range(rectanglewidth + 1):
                image[j % L][(xmin - n) % l], image[j % L][(xmax + n) % l] = 255, 255
    return np.uint8(image)

def cross_color(image, positions, crosswidth):
    L = len(image)
    l = len(image[0])
    for obj in positions:
        x = int(positions[obj][0])
        y = int(positions[obj][1])
        for i in range(x - crosswidth * 10, x + crosswidth * 10 + 1):
            for n in range(y - int(crosswidth / 2), y + int(crosswidth / 2) + 1):
                image[n % L][i % l] = [0, 255, 0]
        for j in range(y - crosswidth * 10, y + crosswidth * 10 + 1):
            for n in range(x - int(crosswidth / 2), x + int(crosswidth / 2) + 1):
                image[j % L][n % l] = [0, 255, 0]
    return np.uint8(image)
