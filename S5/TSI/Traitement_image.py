import cv2
import matplotlib.pyplot as plt

image = cv2.imread('images/image_gris.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# fig, axe = plt.subplots(1, 3, figsize=(10, 5))

# titles = ['original'] 
# images = [image]

# for i in range(len(images)):
#     axe[i].imshow(images[i])
#     axe[i].set_title(titles[i])
#     axe[i].axis('off')


# plt.figure(figsize=(10,5))

# # Affichage de l'image originale
# plt.subplot(1, 2, 1)  # 1 ligne, 2 colonnes, première image
# plt.imshow(image, cmap='gray')
# plt.title("Image Originale")
# plt.axis('off')

hauteur, largeur = image.shape

print(f'hauteur : {hauteur}, largeur : {largeur}')

portion = image[:10, :10]
print(portion)



plt.imshow(image, cmap='gray')
plt.title("Image en Niveaux de Gris")

plt.tight_layout()
plt.show()


# ================================================================================ #



import cv2
import matplotlib.pyplot as plt

image = cv2.imread('images/image_color.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

hauteur, lageur, _ = image.shape

print(f'hauteur : {hauteur}, largeur : {largeur}')

portion = image[:10, :10]
print(portion)

plt.imshow(image)
plt.title('original')
plt.axis(False)
plt.show()


# ================================================================================ #


import cv2
import matplotlib.pyplot as plt
import numpy as np

def adjuste_lumosité(image, valeur):
    return np.clip(image + valeur, 0, 255).astype('uint8')

image = cv2.imread('images/image_gris.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(adjuste_lumosité(image, 25), cmap='gray')
plt.title("adjusté luminosité")
plt.axis("off")

plt.show()



# ================================================================================ #


import numpy as np
import matplotlib.pyplot as plt

image_noir = np.zeros((100, 100))

print('triangle black')
plt.figure(figsize=(10, 15))
plt.subplot(1, 2, 1)
plt.imshow(image_noir, cmap='gray')
plt.axis(False)
plt.title('image noir')

image_noir[30:70, 30:70] = 255

print('triangle noir avec carre blanc')
plt.subplot(1, 2, 2)
plt.imshow(image_noir, cmap='gray')
plt.axis(False)
plt.title('image noir')

plt.show()


# ================================================================================ #


import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('images/image_color.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def image_rouge(img):
    img_copy = img.copy()
    img_copy[:,:,1], img_copy[:,:,2] = 0, 0
    return img_copy
def image_vert(img):
    img_copy = img.copy()
    img_copy[:,:,0], img_copy[:,:,2] = 0, 0
    return img_copy
def image_blue(img):
    img_copy = img.copy()
    img_copy[:,:,0], img_copy[:,:,1] = 0, 0
    return img_copy

image_rouge = image_rouge(image)
image_vert = image_vert(image)
image_blue = image_blue(image)

fig, axe = plt.subplots(1, 4, figsize=(10, 15))

titles = ['original', 'rouge', 'vert', 'blue']
images = [image, image_rouge, image_vert, image_blue]

for i in range(len(images)):
    axe[i].imshow(images[i])
    axe[i].set_title(titles[i])
    axe[i].axis(False)
    

plt.tight_layout()
plt.show()


# ================================================================================ #



import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('images/image_color.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
def augmenter_saturation(img, value=50):
    copy_image = img.copy()
    copy_image[:, :, 1] = np.clip(copy_image[:, :, 1] + value, 0, 255)
    return cv2.cvtColor(copy_image, cv2.COLOR_HSV2RGB)

def diminuer_saturation(img, value=50):
    copy_image = img.copy()
    copy_image[:, :, 1] = np.clip(copy_image[:, :, 1] - value, 0, 255)
    return cv2.cvtColor(copy_image, cv2.COLOR_HSV2RGB)
        
fig, axe = plt.subplots(1, 3, figsize=(15, 10))

titles = ['original', 'augmenter saturation', 'diminuer sturtion']
images = [image_rgb, augmenter_saturation(image), diminuer_saturation(image)]

for i in range(len(images)):
    axe[i].imshow(images[i])
    axe[i].set_title(titles[i])
    axe[i].axis(False)

plt.tight_layout()
plt.show()




# ================================================================================ #


import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('images/image_color.jpg')
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


value = image_hsv[:, :, 2]


titles = ['origin', 'hsv', 'rgb', 'gray']
images = [image, image_hsv, image_rgb, value]

fig, axe = plt.subplots(1, 4, figsize=(10, 8))


for i in range(len(images)):
    axe[i].imshow(images[i], cmap='gray')
    axe[i].set_title(titles[i])
    axe[i].axis('off')


# plt.figure(figsize=(10, 15))

# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('origin')
# plt.axis('off')


# plt.subplot(1, 2, 2)
# plt.imshow(image_hsv, cmap='hsv')
# plt.title('hsv')
# plt.axis('off')


# plt.subplot(2, 2, 1)
# plt.imshow(image_rgb, cmap='hsv')
# plt.title('rgb')
# plt.axis('off')

# plt.subplot(2, 2, 2)
# plt.imshow(value, cmap='gray')
# plt.title('value (gray)')
# plt.axis('off')


plt.tight_layout()
plt.show()


# ================================================================================ #

import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('images/image_color.jpg')

# convertiser l'mage a format rgb
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# augmentation de rouge
def augmenter_rouge(img, valeur=50):
    image_copy = image_rgb.copy()
    image_copy[:, :, 0] = np.clip(image_rgb[:, :, 0] + valeur, 0, 255)
    return image_copy

# augmentation de vert
def augmenter_vert(img, valeur=50):
    image_copy = image_rgb.copy()
    image_copy[:, :, 1] = np.clip(image_rgb[:, :, 1] + valeur, 0, 255)
    return image_copy

# augmentation de blue
def augmenter_blue(img, valeur=50):
    image_copy = image_rgb.copy()
    image_copy[:, :, 2] = np.clip(image_rgb[:, :, 2] + valeur, 0, 255)
    return image_copy

titles = ['rgb', 'filter rouge', 'filter vert', 'filter blue']
images = [image_rgb, augmenter_rouge(image_rgb), augmenter_vert(image_rgb), augmenter_blue(image_rgb)]

fig, axe = plt.subplots(2, 2, figsize=(10, 8))

axe = axe.ravel()

for i in range(len(images)):
    axe[i].imshow(images[i])
    axe[i].set_title(titles[i])
    axe[i].axis('off')

plt.tight_layout()
plt.show()


# ================================================================================ #