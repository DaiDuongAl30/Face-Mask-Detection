from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os

INPUT_SIZE = (224, 224)
model = load_model("mask_detector_epoch_10.model")

num_test = 200
correct = 0
Choices = ["with_mask", "without_mask"]
Incorrect = []
Name = [[]] * len(Choices)

for i in range(len(Choices)):
    Name[i] = os.listdir(os.path.expanduser('./datatest/' + Choices[i]))

for _ in range(num_test):
    choice = np.random.randint(0, len(Choices))
    img_id = np.random.randint(1, len(Name[choice]))
    img_path = './datatest/' + Choices[choice] + '/' + Name[choice][img_id]

    load_image = image.load_img(img_path, target_size=INPUT_SIZE)
    img_array = image.img_to_array(load_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    result = model.predict(img_array)
    prediction = np.argmax(result, axis=1)[0]

    predicted_label = "with_mask" if prediction == 0 else "without_mask"

    is_correct = predicted_label == Choices[choice]
    correct += is_correct

    if not is_correct:
        Incorrect.append((img_path, Choices[choice], predicted_label))

    print(f"Case {_}: {Choices[choice]} {Name[choice][img_id]} -> {'Correct' if is_correct else 'Incorrect'}")

print(f"Correct: {correct} / {num_test}")
Incorrect_num = num_test - correct
print(f"Incorrect: {Incorrect_num}")

for img_info in Incorrect:
    img_path, actual_label, predicted_label = img_info
    img = image.load_img(img_path, target_size=INPUT_SIZE)
    plt.imshow(img)
    plt.title(f"Actual: {actual_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()
