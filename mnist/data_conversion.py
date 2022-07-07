from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def convert_image_to_pixel(path):
    img = Image.open(path).resize((28,28)).convert('L')  # convert image to 8-bit grayscale
    WIDTH, HEIGHT = img.size

    data = list(img.getdata())

    data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]


    test = []
    for row in data:
        for value in row:
           test.append(round(value))

    return test

def show_digit(path):
    img = Image.open(path).resize((28,28)).convert('L')
    plt.gray()
    plt.matshow(img)
    plt.show()
