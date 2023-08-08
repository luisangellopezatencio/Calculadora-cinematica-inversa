import tkinter as tk
from app import App




if __name__ == "__main__":

    root = tk.Tk()
    root.wm_title("Calculador de Cinematica Inversa")
    main = App(root)
    main.pack(fill="both", expand=True)
    root.mainloop()

