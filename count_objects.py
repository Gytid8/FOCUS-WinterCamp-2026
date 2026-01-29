import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import matplotlib
matplotlib.use('TkAgg')
if __name__ == '__main__':
    model_path = r'D:\PyCharm\Py_Projects\runs\detect\my_custom_train5\weights\best.pt'
    model = YOLO(model_path)
    img_path = r'D:\PyCharm\Py_Projects\my_data\test'
    results = model.predict(
        source=img_path,
        conf=0.15,
        imgsz=1280,
        device=0,
        workers=0
    )
    res_plotted = results[0].plot()

    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(res_rgb)
    plt.axis('off')
    plt.title(f"Detected: {len(results[0].boxes)} objects")
    plt.show()