from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
import pathlib


class LabelWindow:
    def __init__(self, data_path):
        self.imgs = self.read_imgs(data_path)
        self.current_path = None
        self.current_img = None
        self.current_vessel = None
        self.coords_points_colors = self.empty_coords_points_colors()
        self.coords = []
        self.fig = None
        self.ax = None
        # size of check area when clicking point
        self.epsilon = 30
        self.clicked_point = None
        # transformation data
        self.td = None
        self.init_ui()

    @staticmethod
    def read_imgs(data_path):
        return [
            (img_path, Image.open(img_path))
            for img_path in data_path.iterdir()
            if img_path.is_file()
        ]

    def onkey(self, event):
        print('key = ', event.key)
        if event.key == 'enter' or event.key == ' ':
            self.enter()
        else:
            self.select_point(key=event.key)

    def empty_coords_points_colors(self):
        # color pallete, extend if more than 6 points are required
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        return {i+1: (None, None, color) for i, color in enumerate(colors)}

    def enter(self):
        if self.current_img is None:
            self.current_path, self.current_img = self.imgs.pop(0)
        if not all(x[0] is None for x in self.coords_points_colors.values()):
            self.coords.append(
                (
                    self.current_path,
                    [x[0] for x in self.coords_points_colors.values()]
                )
            )
            if len(self.imgs) == 0:
                plt.close()
                self.write_results()
                return
            self.current_path, self.current_img = self.imgs.pop(0)
            self.update_vessel_selection(vessel_number=None)
        self.ax.imshow(self.current_img)
        self.fig.canvas.draw_idle()

    def write_results(self):
        with pathlib.Path('test.txt').open('w+') as res_file:
            res_file.write(str(self.coords[0]))

    def update_vessel_selection(self, vessel_number):
        if vessel_number is None:
            self.fig.suptitle("No vessel number selected for marking")
        else:
            self.fig.suptitle(f"Selected vessel number {vessel_number}")
            self.fig.canvas.draw_idle()
        self.current_vessel = vessel_number

    def select_point(self, key):
        try:
            if self.current_img is None:
                return
            key = int(key)
            if key not in list(self.coords_points_colors.keys()):
                return
            self.update_vessel_selection(vessel_number=key)
        except ValueError:
            return

    def onclick(self, event):
        # mark a new vessel
        if self.current_vessel is not None:
            c_new = self.td.inverted().transform((event.x, event.y))
            self.update_point(idx=self.current_vessel, coords=c_new)
            self.update_vessel_selection(vessel_number=None)
        # move an existing vessel marker
        else:
            self.clicked_point = self.get_point_under_click(event)

    def release_click(self, event):
        if event.button != 1:
            return
        self.clicked_point = None

    def move_mouse(self, event):
        if self.clicked_point is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        self.update_point(
            idx=self.clicked_point, coords=(event.xdata, event.ydata))

    def update_point(self, idx, coords):
        c, p, color = self.coords_points_colors.get(idx)
        if p is not None:
            p.remove()
        p_new, = self.ax.plot(coords[0], coords[1], f'{color}o')
        self.coords_points_colors[idx] = (coords, p_new, color)
        self.fig.canvas.draw_idle()

    def get_point_under_click(self, event):
        if all(x[0] is None for x in self.coords_points_colors.values()):
            return None
        # transData = coordinate System
        click_coords = self.td.inverted().transform([event.x, event.y])
        existing_coords = np.array(
            [x[0] for x in self.coords_points_colors.values()
             if x[0] is not None]
        )
        existing_x, existing_y = existing_coords[:, 0], existing_coords[:, 1]
        d = np.hypot(
            existing_x - click_coords[0], existing_y - click_coords[1])
        closest_point = np.argmin(d)

        if d[closest_point] >= self.epsilon:
            return None

        return closest_point + 1

    def init_ui(self):
        mpl.rcParams['toolbar'] = 'None'
        self.ax = plt.gca()
        self.fig = plt.gcf()
        self.td = self.ax.transData
        self.fig.canvas.set_window_title('Mark the visible vessels')
        self.fig.suptitle("No vessel number selected for marking")
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect(
            'button_release_event', self.release_click)
        self.fig.canvas.mpl_connect(
            'motion_notify_event', self.move_mouse)
        plt.show()
