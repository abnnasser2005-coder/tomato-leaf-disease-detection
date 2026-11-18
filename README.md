# 🍅 Tomato Leaf Disease Detection

Deep learning model that detects **tomato leaf diseases** from a photo of the leaf.
It compares a **baseline CNN built from scratch** against **transfer learning with MobileNetV2**, and includes an interactive **Gradio** demo where you can upload your own leaf photo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xFbJABF81JR-WBHQY7xgFZWhSr78WxJB)

![Results](results.png)

## Dataset

- **PlantVillage** (loaded with `tensorflow_datasets`), **tomato classes only**: 10 classes
- Split: **70% train / 15% validation / 15% test**
- Images resized to **160×160**

Classes: Bacterial spot · Early blight · Late blight · Leaf Mold · Septoria leaf spot · Spider mites · Target Spot · Tomato Yellow Leaf Curl Virus · Tomato mosaic virus · Healthy

## Models

| Model | Description |
|---|---|
| **Baseline CNN** | 3 Conv + MaxPooling blocks → Dropout → Dense, trained from scratch (10 epochs) |
| **MobileNetV2** | Pre-trained on ImageNet (frozen) + new classification head (8 epochs) |

Both use data augmentation (random flip, rotation, zoom) during training.

## Results

| Model | Test accuracy |
|---|---|
| Baseline CNN | ~92% (validation) |
| MobileNetV2 | see notebook (section 5) |

The notebook also shows a classification report, a confusion matrix, and predictions on unseen test images (green = correct, red = wrong).

## How to run

1. Click **Open in Colab** above
2. `Runtime → Change runtime type → T4 GPU`
3. `Runtime → Run all`
4. At the end, a Gradio link appears: open it and upload a tomato leaf photo to get a diagnosis

## Tech stack

Python · TensorFlow / Keras · TensorFlow Datasets · scikit-learn · Matplotlib · Gradio
