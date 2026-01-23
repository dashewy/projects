
import matplotlib.pyplot as plt
import numpy as np

def rotten_eval_visual(data, model_name,sample_size=1):
    class_names = data.class_names

    plt.figure(figsize=(12, 12))

    for images, labels in data.take(sample_size):
        preds = model_name.predict(images)

        for i in range(min(9, images.shape[0])):
            ax = plt.subplot(3, 3, i + 1)

            plt.imshow(images[i].numpy().astype("uint8"))

            pred_idx = np.argmax(preds[i])
            true_idx = np.argmax(labels[i])

            pred_label = class_names[pred_idx]
            true_label = class_names[true_idx]

            color = "green" if pred_idx == true_idx else "red"

            plt.title(
                f"Pred: {pred_label}\nTrue: {true_label}",
                color=color,
                fontsize=10
            )
            plt.axis("off")

    plt.tight_layout()
    plt.show()