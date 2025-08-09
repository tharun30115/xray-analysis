# Digital X-Ray Analysis – COVID/Normal Classifier


## 📌 Overview
This project is a **deep learning–powered chest X-ray classification tool** that predicts whether a given X-ray image is **Normal** or shows signs of **COVID‑19** infection.  
It uses a **TensorFlow/Keras CNN model** trained on the publicly available **COVIDGR_1.0 dataset**, wrapped in a clean **Tkinter GUI** for easy image selection, prediction, and confidence display.

---

## ✨ Features
- **Instant predictions** with class label (COVID / Normal) and confidence percentages.
- **Model**: Custom CNN based on EfficientNetB0 architecture.
- **Clean UI** with JetBrains Mono font, clickable Docs & GitHub links.
- Works fully offline once model is downloaded (needs setup).

---

## 📊 Dataset
- **COVIDGR_1.0** – Public dataset of chest X-rays used for training.
- Preprocessed to grayscale, resized to **224×224** pixels for model input.

---

## To test it:
- Download the full .zip from the [link](https://github.com).
- Unzip, run the setup file to download all the requried packages.
- Make sure you're in the correct folder.
- Better to use a virtual python env.
```bash
python -m venv venv 
venv\Scripts\activate
```
- Install all dependencies
```bash
pip install -r setup.txt
```
- Run the gui.py file 
```bash
python gui.py
```

## Map
- A publically available **COVIDGR_1.0** was used to build and train this model, the data was slpit into training sets and validation sets.
- First training was for 50 Epochs which produced a model **model-v1** which had 60% accuracy with real world data.
- Second training was for 100 Epochs which produced a model **model-v2** which has 90% + accuracy with real world data.
- Tensorflow and keras was used to train the model.
- Gui was built using tkinter.
- Thats all for now, cya (dont forget to star the repo!!).