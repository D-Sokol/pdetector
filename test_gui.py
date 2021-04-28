#!/usr/bin/env python3
import cv2
from argparse import Namespace
from PIL import Image, ImageTk
from threading import Thread
from time import sleep
from tkinter import Canvas, Tk, NW
from tkinter.filedialog import askopenfilename

from test import main


class StopPlaying(Exception):
    pass


def wrapper(args, writer):
    try:
        return main(args, writer)
    except (StopPlaying, RuntimeError):
        pass


class ScreenWriter:
    def __init__(self, title="Processed video"):
        self.root = Tk()
        self.root.title(title)
        self.root.protocol('WM_DELETE_WINDOW', self.quit)
        self.canvas = Canvas(self.root, height=512, width=512)
        self.canvas.pack()
        self.photo = None
        self.closed = False

    def start(self):
        self.root.mainloop()
        self.quit()

    def write(self, frame):
        if self.closed:
            raise StopPlaying
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def quit(self):
        self.closed = True
        self.root.quit()

    def release(self):
        pass


if __name__ == '__main__':
    writer = ScreenWriter()
    fname = askopenfilename()
    writer.root.update()

    config = Namespace(
        weights='./extra/checkpoints/checkpoint-180.pth',
        input=fname,
        device='cuda',
        min_proba=0.5,
        threshold=0.25,
        batch_size=64,
        verbose=False,
        lw=5,
    )

    thread = Thread(target=wrapper, args=(config, writer))

    thread.start()
    writer.start()

    try:
        thread.join()
    except StopPlaying:
        pass

