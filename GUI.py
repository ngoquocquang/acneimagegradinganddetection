import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from test import main
from PIL import Image, ImageTk

filenames = []
bg = "bg.png"

class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Joint Acne Image Grading")
        self.geometry("900x375")

        PIL_image = Image.open(bg)
        img1 = PIL_image.resize((900, 375), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img1)
        self.label = tkinter.Label(self, image=img)
        self.label.image = img
        self.label.place(x=0,y=0)

        self.labelFrame = ttk.LabelFrame(self, text = "Select the Directory of Joint Ance Image that needed to grade")
        self.labelFrame.grid(column = 0, row = 0, padx = 480, pady = 300)

        self.button()

    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse Directory", command = self.fileDialog)
        self.button.grid(column = 1, row = 1)

    def fileDialog(self):
        self.filename = filedialog.askdirectory(initialdir = None, title = "Select A Directory")
        filenames= self.filename
        main(filenames)

root = Root()
root.mainloop()