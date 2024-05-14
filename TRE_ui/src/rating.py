from matplotlib import pyplot as plt
import matplotlib as mpl
from PIL import Image


class LabelWindow:
    def __init__(self, data_path):
        self.imgs = self.read_imgs(data_path)
        self.current_path = None
        self.current_img = None
        self.current_rating = None
        self.img_ratings = []
        self.fig = None
        self.ax = None
        self.init_ui()

    @staticmethod
    def read_imgs(data_path):
        return [
            (img_path, Image.open(img_path))
            for img_path in data_path.iterdir()
            if img_path.is_file()
        ]

    def onkey(self, event):
        if event.key == 'enter':
            self.enter()
        else:
            self.rating(key=event.key)

    def enter(self):
        if self.current_img is None:
            self.current_path, self.current_img = self.imgs.pop(0)
        if self.current_rating is not None:
            self.img_ratings.append((self.current_path, self.current_rating))
            if len(self.imgs) == 0:
                plt.close()
                return
            self.current_path, self.current_img = self.imgs.pop(0)
            self.update_rating(None)

        self.ax.imshow(self.current_img)
        plt.draw()

    def rating(self, key):
        try:
            if self.current_img is None:
                return
            else:
                self.update_rating(int(key))
        except ValueError:
            return

    def update_rating(self, rating_value):
        self.current_rating = rating_value
        self.fig.suptitle(
            f"Current rating is {self.current_rating}")
        self.fig.canvas.draw_idle()

    def init_ui(self):
        mpl.rcParams['toolbar'] = 'None'
        self.ax = plt.gca()
        self.fig = plt.gcf()
        self.fig.canvas.set_window_title('Rate the images from 0 to 9.')
        self.fig.suptitle(
            "Press a number to assign a rating. Press Enter for next image.")
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        plt.show()
