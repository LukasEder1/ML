# 2019
# Imports 
from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from data_conversion import convert_image_to_pixel, show_digit
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pygame as pg

# pygame settings
def init():
    global screen

    pg.init()
    # set canvas size
    screen = pg.display.set_mode((4 * 28, 4 * 28))
    mainloop()


# set weight
drawing = False
last_pos = None
w = 10
color = (255,255,255)


def draw(event):
    global drawing, last_pos, w

    if event.type == pg.MOUSEMOTION:
        if (drawing):
            mouse_position = pg.mouse.get_pos()
            if last_pos is not None:
                pg.draw.line(screen, color, last_pos, mouse_position, w)
            last_pos = mouse_position
    elif event.type == pg.MOUSEBUTTONUP:
        mouse_position = (0, 0)
        drawing = False
        last_pos = None
    elif event.type == pg.MOUSEBUTTONDOWN:
        drawing = True


def mainloop():
    global screen
    # can bed closed USING ESCAPE
    loop = 1
    while loop:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                loop = 0
            elif event.type == pg.KEYDOWN:
               if event.key == pg.K_ESCAPE:
                loop = 0
            draw(event)
        pg.display.flip()

    pg.image.save(screen, "image.png")
    pg.quit()


init() 

# load and split the data
mndata = MNIST('data')
mndata.gz = True
images, labels = mndata.load_training()



df = pd.DataFrame({'images': images, 'labels': labels})
images = images[:3500]
labels = labels[:3500]

X_train, X_test, y_train, y_test = train_test_split(np.array(images),
                                       np.array(labels), random_state=42,
                                       test_size=0.3, stratify=np.array(labels))

# train model
model1 = LogisticRegression(C=0.0001, class_weight=None, dual=False,
            fit_intercept=True, intercept_scaling=1, max_iter=100,
            multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
            solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

model1.fit(X_train, y_train)

# Output Scores
print(f"Training {model1.score(X_train, y_train)}")
print(f"Testing {model1.score(X_test, y_test)}")
print("\n")


# predict newly draw image
path = "image.png"
test = convert_image_to_pixel(path)

     
test = np.array([test])
print(model1.predict(test))
show_digit(path)

