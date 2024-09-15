import numpy as np
import matplotlib.pyplot as plt

# Определяем изображения как 3x3 матрицы
images = {
    "minus": np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    "dot": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    "division": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
}

# Функция для отображения и сохранения изображения
def save_image(name, image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title(name)
    plt.axis('off')
    plt.savefig(f'img/{name}.png', bbox_inches='tight')
    plt.show()

# Создание изображений
for name, img in images.items():
    save_image(name, img)
