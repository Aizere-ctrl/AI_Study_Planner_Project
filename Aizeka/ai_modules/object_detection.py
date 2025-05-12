import torch
from torchvision import models, transforms  # Модели и преобразования изображений
from PIL import Image  # Работа с изображениями

# Загружаем список меток классов из ImageNet
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_labels = []
try:
    import urllib
    with urllib.request.urlopen(LABELS_URL) as f:
        imagenet_labels = [line.strip() for line in f.readlines()]  # Классы (например, "dog", "cat", "desk")
except:
    imagenet_labels = ["Class " + str(i) for i in range(1000)]  # Запасной вариант

def run(image_path):
    # Шаг 1: Подготавливаем изображение (обязательные преобразования для ResNet18)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Преобразование в тензор (0-1)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Нормализация для ImageNet
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")  # Загружаем изображение
    img_tensor = transform(image).unsqueeze(0)     # Добавляем batch размер (1, C, H, W)

    # Шаг 2: Загружаем предобученную модель ResNet18
    model = models.resnet18(pretrained=True)
    model.eval()  # Переводим модель в режим оценки

    # Шаг 3: Делаем предсказание
    with torch.no_grad():  # Без градиентов (ускорение)
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)  # Получаем индекс с наибольшим значением
        class_idx = predicted.item()
        label = imagenet_labels[class_idx].decode("utf-8") if isinstance(imagenet_labels[class_idx], bytes) else imagenet_labels[class_idx]

    return f"Detected: {label}"
