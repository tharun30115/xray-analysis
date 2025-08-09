import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import tensorflow as tf
import os
import webbrowser

MODEL_PATH = "model-v2.h5"
model = tf.keras.models.load_model(MODEL_PATH)
train_dir = "data/train"
class_labels = sorted(os.listdir(train_dir))
IMG_SIZE = 224

def predict_image(img_path):
    img = Image.open(img_path).convert("L")
    img = ImageOps.fit(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    return class_labels[np.argmax(preds)], preds

def select_image_handler(event=None):
    file_path = filedialog.askopenfilename(
        title="Select X-ray Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    for widget in img_box.winfo_children():
        widget.destroy()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((115, 115), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        iw, ih = img_tk.width(), img_tk.height()
        x = (250 - iw) // 2
        y = (130 - ih) // 2
        image_label = tk.Label(img_box, image=img_tk, bg="#F4F5F7")
        image_label.image = img_tk
        image_label.place(x=x, y=y, width=iw, height=ih)
        pred_class, probs = predict_image(file_path)
        result_heading.config(text=f"Output : {pred_class}")
        covid_conf = f"{probs[class_labels.index('COVID')]*100:.2f}" if 'COVID' in class_labels else "xx"
        normal_conf = f"{probs[class_labels.index('Normal')]*100:.2f}" if 'Normal' in class_labels else "xx"
        conf_details = (
            "   Confidence Levels\n"
            f"   → Covid: {covid_conf}%\n"
            f"   → Normal: {normal_conf}%"
        )
        confidence_label.config(text=conf_details)
    else:
        reset_img_box()

def reset_img_box():
    for widget in img_box.winfo_children():
        widget.destroy()
    plus_label = tk.Label(img_box, text="+", font=("JetBrains Mono", 35, "bold"),
                          fg="#212121", bg="#F4F5F7")
    plus_label.place(relx=0.5, rely=0.5, anchor="center")
    plus_label.bind("<Button-1>", select_image_handler)
    img_box.plus_label = plus_label

def open_github(event=None):
    webbrowser.open_new("https://github.com/tharun30115/xray-analysis")  

def open_docs(event=None):
    webbrowser.open_new("https://your-docs-url") 

def mono(size, weight):
    return ("JetBrains Mono", size, weight)

root = tk.Tk()
root.title("Digital X-Ray Analysis")
root.geometry("600x440")
root.resizable(False, False)
root.configure(bg="#FFFFFF")

main_frame = tk.Frame(root, bg="#FFFFFF")
main_frame.pack(fill="both", expand=True, padx=0, pady=0)

title = tk.Label(main_frame, text="Digital X-Ray analysis", font=mono(17, "bold"), bg="#FFFFFF", anchor="nw")
title.grid(row=0, column=0, sticky="nw", padx=15, pady=(14,2))

desc_text = (
    "Chest X-Ray Analysis:\n"
    "• This model is trained on COVIDGR_1.0 dataset."
    "\n• COVIDGR_1.0 dataset is publically available ."
    "\n• The model does binary prediction: Normal or COVID positive."
    "\n• Trained using tensorflow, keras and gui with tkinter."
)
desc = tk.Label(
    main_frame,
    text=desc_text,
    font=mono(10, "normal"),
    bg="#FFFFFF",
    justify="left",
    anchor="nw",
    wraplength=510,
    padx=22
)
desc.grid(row=1, column=0, sticky="nw", padx=8, pady=(0, 11))

selector_frame = tk.Frame(main_frame, bg="#FFFFFF")
selector_frame.grid(row=2, column=0, pady=(0,2), sticky="ew")
selector_frame.columnconfigure(0, weight=1)

img_box = tk.Canvas(selector_frame, width=250, height=130, bg="#F4F5F7", highlightthickness=0)
img_box.grid(row=0, column=0)
dash_color = "#BDBDBD"
img_box.create_rectangle(8, 8, 242, 122, dash=(4,2), outline=dash_color, width=2)
img_box.bind("<Button-1>", select_image_handler)
reset_img_box()

select_btn = tk.Button(selector_frame, text="Click here to select image",
    font=mono(12, "normal"), borderwidth=0, bg="#F4F5F7", fg="#616161",
    activebackground="#F4F5F7", cursor="hand2", command=select_image_handler)
select_btn.place(relx=0.5, y=140, anchor="center", width=220)

result_heading = tk.Label(main_frame, text="Output :", font=mono(15, "bold"), bg="#FFFFFF", anchor="w")
result_heading.grid(row=3, column=0, sticky="w", padx=23, pady=(18,2))

confidence_label = tk.Label(main_frame, text="   Confidence Levels\n   → Covid: xx%\n   → Normal: xx%",
    font=mono(10, "normal"), bg="#FFFFFF", anchor="w", justify="left", wraplength=540)
confidence_label.grid(row=4, column=0, sticky="w", padx=34, pady=(0,28))

footer_frame = tk.Frame(root, bg="#FFFFFF")
footer_frame.place(x=455, y=400)
docs = tk.Label(footer_frame, text="Docs", font=mono(10, "normal"), fg="#1976D2",
                bg="#FFFFFF", cursor="hand2")
docs.grid(row=0, column=0, padx=(0, 17), sticky="e")
docs.bind("<Button-1>", open_docs)
github = tk.Label(footer_frame, text="Github", font=mono(10, "normal"), fg="#1976D2",
                  bg="#FFFFFF", cursor="hand2")
github.grid(row=0, column=1, sticky="e")
github.bind("<Button-1>", open_github)

root.mainloop()