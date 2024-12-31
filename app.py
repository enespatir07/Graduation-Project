import tkinter as tk
from tkinter import ttk, filedialog
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
from threading import Thread  # Import Thread
import os


class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Remote Worker Control")
        
        # Tam ekran açılmasını sağla
        self.root.state('zoomed')

        # Arka plan rengini ayarla
        self.root.configure(bg="#343541")  # ChatGPT'nin arka plan gri tonu

        # Uygulama simgesini değiştir
        icon_path = "icon.png"  # Simgenizin dosya yolu (PNG formatında)
        if os.path.exists(icon_path):
            icon = ImageTk.PhotoImage(file=icon_path)
            self.root.iconphoto(False, icon)
        else:
            print(f"Simge dosyası bulunamadı: {icon_path}")

        # Modern stil oluştur
        self.style = ttk.Style()
        self.style.configure("TButton", 
                             font=("Helvetica", 14), 
                             padding=10, 
                             background="white", 
                             foreground="black", 
                             borderwidth=1)
        self.style.map("Rounded.TButton", 
                       background=[("active", "#E6E6E6")])
        # Yuvarlak kenar efekti
        self.style.layout("Rounded.TButton", 
                          [("Button.background", {"sticky": "nswe", "border": "10"})])


        self.create_widgets()
        self.stop = False

    def create_widgets(self):
        # Başlık
        title_label = tk.Label(self.root, 
                               text="Remote Worker Control", 
                               font=("Bodoni", 28, "bold"), 
                               bg="#343541", 
                               fg="white")
        title_label.pack(fill=tk.X, pady=20)

        # Butonlar çerçevesi
        button_frame = tk.Frame(self.root, bg="#343541")
        button_frame.pack(pady=40)

        # Butonlar
        self.start_video_button = ttk.Button(button_frame, text="Videoyu Başlat", command=self.start_video_detection)
        self.start_video_button.grid(row=0, column=0, padx=20, pady=10)

        self.capture_photo_button = ttk.Button(button_frame, text="Fotoğraf Çek", command=self.capture_and_save_result)
        self.capture_photo_button.grid(row=0, column=1, padx=20, pady=10)

        self.upload_photo_button = ttk.Button(button_frame, text="Fotoğraf Yükle", command=self.upload_and_process_photo)
        self.upload_photo_button.grid(row=0, column=2, padx=20, pady=10)

        self.upload_video_button = ttk.Button(button_frame, text="Video Yükle", command=self.upload_and_process_video)
        self.upload_video_button.grid(row=0, column=3, padx=20, pady=10)

        # Alt bilgi
        footer_label = tk.Label(self.root, 
                                text="© 2024 Remote Worker Control", 
                                bg="#343541", 
                                fg="white", 
                                font=("Helvetica", 10))
        footer_label.pack(fill=tk.X, side=tk.BOTTOM)

    def start_video_detection(self):
        self.stop = False
        self.thread = Thread(target=self.detect_objects_in_video)
        self.thread.start()

    def detect_objects_in_video(self):
        model = YOLO('runs/train/bitirme/weights/best.pt')
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Kamera açılamadı.")
            return

        while not self.stop and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Kare alınamadı.")
                break
            
            results = model.predict(source=frame, show=False)
            annotated_frame = results[0].plot()
            cv2.imshow("Video İşleme", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop = True
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def generate_unique_filename(self, base_name, extension):
        counter = 0
        while True:
            filename = f"{base_name}_{counter}{extension}" if counter > 0 else f"{base_name}{extension}"
            if not os.path.exists(filename):
                return filename
            counter += 1

    def capture_and_save_result(self):
        model = YOLO('runs/train/bitirme/weights/best.pt')
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Kamera açılamadı.")
            return

        print("Fotoğraf çekmek için 'q' tuşuna basın.")
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("Kare alınamadı.")
                break

            cv2.imshow("Kamera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                captured_image = frame
                break

        cap.release()
        cv2.destroyAllWindows()

        temp_image_path = self.generate_unique_filename("captured_image", ".jpg")
        result_image_path = self.generate_unique_filename("result", ".jpg")

        cv2.imwrite(temp_image_path, captured_image)
        print(f"Fotoğraf kaydedildi: {temp_image_path}")

        results = model.predict(source=temp_image_path)
        annotated_frame = results[0].plot()
        cv2.imwrite(result_image_path, annotated_frame)
        print(f"Sonuç görüntüsü kaydedildi: {result_image_path}")

        # Show the result in a new window
        self.display_image(result_image_path)


    def display_image(self, image_path):
        """Displays the image in a new Tkinter Toplevel window."""
        new_window = tk.Toplevel(self.root)
        new_window.title("Sonuç Görüntüsü")

        # Load the image using PIL
        image = Image.open(image_path)
        image = image.resize((600, 400))  # Resize image for better display
        photo = ImageTk.PhotoImage(image)

        # Add the image to a label
        image_label = tk.Label(new_window, image=photo)
        image_label.image = photo  # Keep a reference to avoid garbage collection
        image_label.pack()

        # Add a close button
        close_button = ttk.Button(new_window, text="Kapat", command=new_window.destroy)
        close_button.pack(pady=10)


    def upload_and_process_photo(self):
        model = YOLO('runs/train/bitirme/weights/best.pt')
        input_image_path = filedialog.askopenfilename(
            title="Bir fotoğraf seçin",
            filetypes=(("Görüntü Dosyaları", "*.jpg *.jpeg *.png"), ("Tüm Dosyalar", "*.*"))
        )

        if not input_image_path:
            print("Hiçbir fotoğraf seçilmedi.")
            return

        print(f"Yüklenen fotoğraf: {input_image_path}")
        result_image_path = self.generate_unique_filename("result", ".jpg")

        # Process the image using YOLO
        results = model.predict(source=input_image_path, show=False)
        annotated_frame = results[0].plot()
        cv2.imwrite(result_image_path, annotated_frame)
        print(f"Sonuç görüntüsü kaydedildi: {result_image_path}")

        # Show the processed image in a new window
        self.display_image(result_image_path)



    def upload_and_process_video(self):
        model = YOLO('runs/train/bitirme/weights/best.pt')
        input_video_path = filedialog.askopenfilename(
            title="Bir video seçin",
            filetypes=(("Video Dosyaları", "*.mp4 *.avi *.mov"), ("Tüm Dosyalar", "*.*"))
        )

        if not input_video_path:
            print("Hiçbir video seçilmedi.")
            return

        print(f"Yüklenen video: {input_video_path}")
        result_video_path = self.generate_unique_filename("vid_result", ".mp4")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("Video açılamadı.")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(result_video_path, fourcc, fps, (width, height))

        print("Video işleniyor. Çıkmak için 'q' tuşuna basın.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, show=False)
            annotated_frame = results[0].plot()
            cv2.imshow("Video İşleme", annotated_frame)
            out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Sonuç videosu kaydedildi: {result_video_path}")

root = tk.Tk()
app = ObjectDetectionApp(root)
root.mainloop()
