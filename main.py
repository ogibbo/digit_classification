import torch
import pygame
import numpy as np
import cv2

# Loading the model using parameters from training
model = torch.load('model_weights.pth')

pygame.init()

# Creating the window
screen = pygame.display.set_mode((200,200))

# Indicator to see if drawing is done
done = False

WHITE = (255,255,255)

# Allow user to draw digit
while not done:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            done = True
        x_cord,y_cord = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed() == (1,0,0):
            pygame.draw.ellipse(screen,WHITE,(x_cord,y_cord,10,20))
        pygame.display.update()

# Saving image of digit
pygame.image.save(screen, 'digit.jpg')

# Converting to format seen during training
image_file = 'digit.jpg'
image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
image_resized = cv2.resize(image,(28,28), interpolation=cv2.INTER_LINEAR)
image_tensor = (torch.tensor(image_resized.flatten())/255)

# Pass image through model
with torch.no_grad():
            prediction = model(image_tensor)

# Output prediction
print(prediction.argmax())

pygame.quit()