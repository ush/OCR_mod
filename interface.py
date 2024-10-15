import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import os
import subprocess

class ImageEditorApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Демонстрационное приложение")
		
		self.root.grid_rowconfigure(0, weight=1)
		self.root.grid_rowconfigure(1, weight=1)
		self.root.grid_rowconfigure(2, weight=1)
		self.root.grid_columnconfigure(0, weight=1)
		self.root.grid_columnconfigure(1, weight=1)
		self.root.grid_columnconfigure(2, weight=1)

		self.save_path = "output/"
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)

		self.current_image_index = 0
		self.initial_image_list = []
		self.edited_image_list = []

		self.original_image_path = None
		self.transformed_image_path = None

		self.original_image_frame = tk.Frame(root, bg="black")
		self.original_image_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")
		self.original_image_label = tk.Label(self.original_image_frame, bg="white")
		self.original_image_label.pack(expand=True, fill="both")

		self.edited_image_frame = tk.Frame(root, bg="black")
		self.edited_image_frame.grid(row=0, column=2, padx=0, pady=0, sticky="nsew")
		self.edited_image_label = tk.Label(self.edited_image_frame, bg="white")
		self.edited_image_label.pack(expand=True, fill="both")

		self.progress = ttk.Progressbar(root, orient="horizontal", mode="determinate")
		self.progress.grid(row=6, column=0, columnspan=3, sticky="ew", padx=10, pady=10)

		self.metric_label = tk.Label(root, text="Метрика:")
		self.metric_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

		self.metric_output = tk.Text(root, height=4, width=50)
		self.metric_output.grid(row=1, column=1, padx=10, pady=10, columnspan=2, sticky="ew")

		self.prev_button = tk.Button(root, text="Назад", command=self.prev_image)
		self.prev_button.grid(row=2, column=0, padx=10, pady=10, sticky="w")

		self.image_counter = tk.Label(root, text="1/x")
		self.image_counter.grid(row=2, column=1, padx=10, pady=10)

		self.next_button = tk.Button(root, text="Далее", command=self.next_image)
		self.next_button.grid(row=2, column=2, padx=10, pady=10, sticky="e")

		self.choose_image_button = tk.Button(root, text="Выбрать изображения", command=self.choose_images)
		self.choose_image_button.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

		self.clear_selection_button = tk.Button(root, text="Очистить выбор", command=self.clear_selection)
		self.clear_selection_button.grid(row=4, column=2, padx=10, pady=10, sticky="ew")

		self.ocr_button = tk.Button(root, text="Посчитать метрику", command=self.get_results)
		self.ocr_button.grid(row=3, column=2, padx=10, pady=10, sticky="ew")

		self.remove_shadows_button = tk.Button(root, text="Удалить тени", command=self.remove_shadows)
		self.remove_shadows_button.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

		self.transform_button = tk.Button(root, text="Преобразовать", command=self.transform_image)
		self.transform_button.grid(row=3, column=1, padx=10, pady=10, sticky="ew")
		
		self.rotate_button = tk.Button(root, text="Повернуть", command=self.rotate_image)
		self.rotate_button.grid(row=4, column=1, padx=10, pady=10, sticky="ew")

		self.root.bind("<Configure>", self.on_resize)

	def get_width_height(self, img):
		img_width, img_height = img.size
		frame_width = 0
		frame_height = 0
		if img_width > img_height:
			frame_width = 1100
			frame_height = 700
		else:
			frame_width = 700
			frame_height = 1100
		return frame_width, frame_height

	def choose_images(self):
		file_paths = filedialog.askopenfilenames(
			title="Выберите изображения",
			filetypes=[
				("JPEG Files", "*.jpg")
			]
		)
		
		if file_paths:
			self.initial_image_list = list(file_paths)
			self.edited_image_list = list(file_paths)
			self.current_image_index = 0
			self.show_images()

	def clear_selection(self):
		self.initial_image_list = []
		self.edited_image_list = []
		self.current_image_index = 0
		self.metric_output.delete("1.0", tk.END)
		self.show_images()

	def show_images(self):
		if self.initial_image_list:
			image_path = self.initial_image_list[self.current_image_index]
			edited_image_path = self.edited_image_list[self.current_image_index]
			img = Image.open(image_path)
			edited_img = Image.open(edited_image_path)
			
			frame_width, frame_height = self.get_width_height(img)
			edited_frame_width, edited_frame_height = self.get_width_height(edited_img)
			
			self.update_original_image(img, frame_width, frame_height)
			self.update_edited_image(edited_img, edited_frame_width, edited_frame_height)

			self.image_counter.config(text=f"{self.current_image_index + 1}/{len(self.initial_image_list)}")
			self.original_image_path = image_path
			self.transformed_image_path = None
		else:
			self.original_image_label.config(image="", text="Изображение не выбрано")
			self.edited_image_label.config(image="", text="Изображение не выбрано")
			self.image_counter.config(text="0/0")

	def prev_image(self):
		if self.current_image_index > 0:
			self.current_image_index -= 1
			self.show_images()

	def next_image(self):
		if self.current_image_index < len(self.initial_image_list) - 1:
			self.current_image_index += 1
			self.show_images()

	def transform_image(self):
		if self.initial_image_list:
			image_path = self.edited_image_list[self.current_image_index]
			output_image_path = os.path.join(self.save_path, os.path.basename(image_path))
			try:
				self.progress["value"] = 0
				self.progress.update()

				self.progress["value"] = 20
				self.progress.update()
				subprocess.run('python3 grayscale.py --img_path ' + image_path + ' --out_path ' + output_image_path, shell=True)

				img = Image.open(output_image_path)
				self.progress["value"] = 60
				self.progress.update()

				frame_width, frame_height = self.get_width_height(img)
				self.update_edited_image(img, frame_width, frame_height)

				img.save(output_image_path)
				self.edited_image_list[self.current_image_index] = output_image_path

				self.progress["value"] = 100
				self.progress.update()
			except subprocess.CalledProcessError as e:
				messagebox.showerror("Ошибка", f"Ошибка преобразования: {e}")
			finally:
				self.progress["value"] = 0
				self.progress.update()
		else:
			messagebox.showwarning("Предупреждение", "Не выбрано изображение для преобразования.")
			
	def rotate_image(self):
		if self.initial_image_list:
			image_path = self.edited_image_list[self.current_image_index]
			output_image_path = os.path.join(self.save_path, os.path.basename(image_path))
			try:
				self.progress["value"] = 0
				self.progress.update()

				self.progress["value"] = 20
				self.progress.update()
				subprocess.run('python3 rotation.py --img_path ' + image_path + ' --out_path ' + output_image_path, shell=True)

				img = Image.open(output_image_path)
				self.progress["value"] = 60
				self.progress.update()

				frame_width, frame_height = self.get_width_height(img)
				self.update_edited_image(img, frame_width, frame_height)

				img.save(output_image_path)
				self.edited_image_list[self.current_image_index] = output_image_path

				self.progress["value"] = 100
				self.progress.update()
			except subprocess.CalledProcessError as e:
				messagebox.showerror("Ошибка", f"Ошибка поворота изображения: {e}")
			finally:
				self.progress["value"] = 0
				self.progress.update()
		else:
			messagebox.showwarning("Предупреждение", "Не выбрано изображение для поворота.")


	def remove_shadows(self):
		if self.original_image_path:
			image_path = self.edited_image_list[self.current_image_index]
			output_image_path = os.path.join(self.save_path, os.path.basename(image_path))

			try:
				self.progress["value"] = 0
				self.progress.update()

				self.progress["value"] = 20
				self.progress.update()
				subprocess.run('conda run -n STCGANenv python ../Models/ST-CGAN_Stacked_Conditional_Generative_Adversarial_Networks/test.py -l 750 -dt classic -i ' + image_path + ' -o ' + self.save_path + ' -rs 1024', shell=True)

				img = Image.open(output_image_path)
				self.progress["value"] = 60
				self.progress.update()

				frame_width, frame_height = self.get_width_height(img)
				self.update_edited_image(img, frame_width, frame_height)
				img.save(output_image_path)
				self.edited_image_list[self.current_image_index] = output_image_path

				self.progress["value"] = 100
				self.progress.update()
			except subprocess.CalledProcessError as e:
				messagebox.showerror("Ошибка", f"Ошибка удаления теней: {e}")
			finally:
				self.progress["value"] = 0
				self.progress.update()
		else:
			messagebox.showwarning("Предупреждение", "Не выбрано изображение для удаления теней.")
			
				
	def get_results(self):
		if self.original_image_path:
			try:
				image_path = self.edited_image_list[self.current_image_index]
				jpg_filename = os.path.basename(image_path)
				txt_filename = jpg_filename.replace('jpg', 'txt')
				output_text_path = os.path.join(self.save_path, txt_filename)

				self.progress["value"] = 0
				self.progress.update()

				self.progress["value"] = 30
				self.progress.update()
				subprocess.run('python3 metric.py --img_path ' + image_path + ' --out_path ' + output_text_path, shell=True)

				self.progress["value"] = 60
				self.progress.update()

				out_text = 0
				with open(output_text_path, 'r') as f:
					out_text = f.read()

				self.metric_output.delete("1.0", tk.END)
				self.metric_output.insert(tk.INSERT, out_text)

				self.progress["value"] = 100
				self.progress.update()
			except subprocess.CalledProcessError as e:
				messagebox.showerror("Ошибка", f"Ошибка при расчёте метрики: {e}")
			finally:
				self.progress["value"] = 0
				self.progress.update()
		else:
			messagebox.showwarning("Предупреждение", "Не выбрано изображение для расчёта метрики.")

	def on_resize(self, event):
		if self.initial_image_list:
			image_path = self.initial_image_list[self.current_image_index]
			edited_image_path = self.edited_image_list[self.current_image_index]
			img = Image.open(image_path)
			edited_img = Image.open(edited_image_path)
			
			frame_width, frame_height = self.get_width_height(img)
			edited_frame_width, edited_frame_height = self.get_width_height(edited_img)
			
			self.update_original_image(img, frame_width, frame_height)
			self.update_edited_image(edited_img, edited_frame_width, edited_frame_height)

	def update_original_image(self, img, frame_width, frame_height):
		img.thumbnail((frame_width, frame_height))
		img_tk = ImageTk.PhotoImage(img)

		self.original_image_label.config(image=img_tk)
		self.original_image_label.image = img_tk
		
	def update_edited_image(self, img, frame_width, frame_height):
		img.thumbnail((frame_width, frame_height))
		img_tk = ImageTk.PhotoImage(img)

		self.edited_image_label.config(image=img_tk)
		self.edited_image_label.image = img_tk

  
def main():
	root = tk.Tk()
	app = ImageEditorApp(root)
	root.mainloop()
	
def get_filename(address):
	parts = address.split('/')
	return parts[-1]
	
def get_filepath(address):
	parts = address.split('/')
	return '/'.join(parts[:-1])

if __name__ == "__main__":
	main()

