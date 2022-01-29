# mapeditor.py
# Jeroen van den Berg 12887366
# GUI tool to create maps to run the LBM on.

from enum import Enum
from collections import deque

import tkinter.filedialog
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import colors

import numpy as np

import argparse

# Size of the undo buffer, which stores previous maps to facilitate undo.
# Should be small for large map sizes, but can be big otherwise.
UNDO_BUFFER_SIZE = 10
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0),
              (1, 1)]


class CellType(Enum):
    Air = 0
    Wall = 1
    Inlet = 2
    Outlet = 3
    Infected = 4
    Susceptible = 5


class Tool(Enum):
    Brush = 0
    Square = 1
    Line = 2
    Bucket = 3


def get_map_from_file(file):
    """
    Load the contents of the file into a map for the cellular automaton.
    """
    with open(file, 'r') as f:
        iterator = enumerate(f)

        _, firstline = next(iterator)
        width, height = [int(x) for x in firstline.strip().split(',')]

        map_array = np.zeros([height, width], dtype=int)

        for i, line in iterator:
            # i is 1 for the first line of the map, because i = 0 corresponds
            # to the line containing the dimensions of the map.
            map_array[i-1] = np.array([int(x) for x in line.strip()])

    return width, height, map_array


cmap = colors.ListedColormap(
    ["black", "forestgreen", "red", "blue", "purple", "lightcoral"])


class MapEditor:
    def __init__(self, root, map_size=None):
        if map_size is None:
            self.height = self.width = 100
        else:
            self.width, self.height = map_size

        self.map = np.full([self.height, self.width], CellType.Air.value)

        self.undo_buffer = deque()
        self.undo_buffer.append(np.copy(self.map))
        self.undo_buffer_pos = 0

        self.root = root

        self.setup_menus()
        self.setup_canvas()
        self.setup_widgets()
        self.root.bind('<Control-o>', self.open_file)
        self.root.bind('<Control-s>', self.save_file)
        self.root.bind('<Control-z>', self.undo)
        self.root.bind('<Control-y>', self.redo)

        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.filename = ""

        # We sometimes need to remember a point (when drawing lines)
        self.x1, self.y1 = 0, 0

    def setup_menus(self):
        """
        Set up the open / save as menu entries at the top of the screen.
        """
        menu_bar = tk.Menu(root)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_file,
                              accelerator="Ctrl+O")
        file_menu.add_command(label="Save as...", command=self.save_file,
                              accelerator="Ctrl+S")
        menu_bar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self.undo,
                              accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo,
                              accelerator="Ctrl+Y")
        menu_bar.add_cascade(label="Edit", menu=edit_menu)

        self.root.config(menu=menu_bar)

    def setup_canvas(self):
        """
        Do the work necessary to show the canvas.
        """
        fig = Figure(figsize=(4, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.axis('off')

        self.canvas = FigureCanvasTkAgg(fig, master=root)

        self.img = self.ax.imshow(self.map, interpolation='none', vmin=0,
                                  vmax=len(CellType), cmap=cmap)
        self.update_view()

        self.canvas.mpl_connect("button_press_event", self.handle_click)
        self.canvas.mpl_connect("motion_notify_event", self.handle_move)
        self.canvas.mpl_connect("button_release_event", self.handle_release)

        self.mouse_held = False
        self.tool = Tool.Brush.value
        self.cell_type = CellType.Air.value

    def setup_widgets(self):
        """
        Do the work necessary to show the tool / cell type selectors at the
        right hand side of the screen.
        """
        frame = ttk.Frame(self.root, relief=tk.RIDGE)
        frame.pack(side=tk.RIGHT)

        self.tk_cell_type = tk.IntVar()
        self.tk_cell_type.set(0)

        ttk.Label(frame, text="Cell type:").pack()

        for i in range(len(CellType)):
            ttk.Radiobutton(frame, text=CellType(i).name,
                            variable=self.tk_cell_type, value=i).pack()

        self.tk_tool = tk.IntVar()

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack()
        ttk.Label(frame, text="Tool:").pack()

        for i in range(len(Tool)):
            ttk.Radiobutton(frame, text=Tool(i).name,
                            variable=self.tk_tool, value=i).pack()

        self.tk_edit_radius = tk.IntVar()
        self.tk_edit_radius.set(max(1, min(self.height, self.width) // 20))

        tk.Label(frame, text="Edit radius:").pack()
        e = ttk.Entry(frame, textvariable=self.tk_edit_radius).pack()

    def open_file(self, _=None):
        """
        Open a file and load it to the canvas.
        """
        self.filename = tk.filedialog.askopenfilename()
        if not self.filename:
            return

        self.width, self.height, self.map = get_map_from_file(self.filename)

        # Draw the map to the screen
        self.img = self.ax.imshow(self.map, interpolation='none', vmin=0,
                                  vmax=len(CellType), cmap=cmap)
        self.update_view()

        self.undo_buffer = deque()
        self.undo_buffer.append(np.copy(self.map))
        self.undo_buffer_pos = 0

    def save_file(self, event=None):
        """
        Save the map that is being worked on to a file.
        """
        self.filename = tk.filedialog.asksaveasfilename()

        if self.filename:
            with open(self.filename, 'w') as file:
                file.write("{}, {}\n".format(self.width, self.height))
                for row in self.map:
                    file.write("".join([str(x) for x in row]) + "\n")

    def filter_out_of_bounds(self, row, col):
        """
        Only return cells in row, col that lie within the boundary of the map.
        """
        # Filter out of bounds rows.
        idx = np.logical_and(row >= 0, row < self.height)

        row = row[idx]
        col = col[idx]

        # Filter out of bounds cols.
        idx = np.logical_and(col >= 0, col < self.width)

        row = row[idx]
        col = col[idx]

        return row, col

    def points_in_circle(self, x, y, r):
        """
        Return cells that have a distance less than r to the point (x, y).
        """
        row, col = np.indices((2 * r, 2 * r)) - r

        idx = np.hypot(row, col) <= r

        return self.filter_out_of_bounds(row[idx] + y, col[idx] + x)

    def points_in_square(self, x, y, r):
        """
        Return cells that lie inside the square centered at (x, y) with side
        length 2r.
        """
        row, col = np.indices((2 * r, 2 * r)) - r

        return self.filter_out_of_bounds(row + y, col + x)

    def get_cells_near_line(self, x1, y1, x2,  y2, r):
        """
        Return cells that have a distance less than r to a line drawn from the
        point (x1, y1)to the point (x2, y2).
        """
        dy = y2 - y1
        dx = x2 - x1

        col = list()
        row = list()

        if abs(dx) > abs(dy):
            for x in np.arange(x1, x2 + 1, np.sign(x2 - x1)):
                y = int(y1 + dy / dx * (x - x1))
                for i in np.arange(-self.edit_radius, self.edit_radius):
                    col.append(x)
                    row.append(y + i)
        else:
            for y in np.arange(y1, y2 + 1, np.sign(y2 - y1)):
                x = int(x1 + dx / dy * (y - y1))
                for i in np.arange(-self.edit_radius, self.edit_radius):
                    row.append(y)
                    col.append(x + i)

        row = np.array(row)
        col = np.array(col)

        return self.filter_out_of_bounds(row, col)

    def bucket_fill(self):
        """
        Fills cells of the type the user's cursor is pointing at to the type
        specified by self.cell_type, starting at the cursor's location and
        then following the trail of cells of the same type.
        """
        x, y = self.x1, self.y1
        type_to_fill = self.map[y, x]

        cells_to_fill = np.zeros((self.height, self.width), bool)
        stack = [(y, x)]

        # Grid DFS
        while stack:
            y, x = stack.pop()
            for dy, dx in DIRECTIONS:
                if 0 <= y + dy < self.height \
                        and 0 <= x + dx < self.width \
                        and not cells_to_fill[y + dy, x + dx] \
                        and self.map[y + dy, x + dx] == type_to_fill:
                    cells_to_fill[y + dy, x + dx] = True
                    stack.append((y + dy, x + dx))

        self.map[cells_to_fill] = self.cell_type
        self.update_view()

    def handle_click(self, event):
        """
        The function to be fired when the user clicks on the canvas.
        """
        self.tk_edit_radius.set(max(1, self.tk_edit_radius.get()))

        if event.xdata is None or event.ydata is None:
            return

        self.x1, self.y1 = int(event.xdata), int(event.ydata)
        self.cell_type = self.tk_cell_type.get()
        self.tool = self.tk_tool.get()
        self.edit_radius = self.tk_edit_radius.get()
        self.mouse_held = True

        if self.tool == Tool.Bucket.value:
            self.bucket_fill()

        # The brush and square tool should immediately start painting.
        self.place_paint(self.x1, self.y1)

    def handle_move(self, event):
        """
        The function to be fired when the mouse is moved around on the canvas.
        """
        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.mouse_held and \
                self.tool in {Tool.Brush.value, Tool.Square.value}:
            self.place_paint(x, y)

    def place_paint(self, x, y):
        # Place paint at position (x, y) in the shape of a circle or square,
        # depending on the selected tool.
        if self.tool == Tool.Brush.value:
            # Color all cells that are closer than self.edit_radius to
            # (xdata, ydata) with the selected cell type
            row, col = self.points_in_circle(x, y, self.edit_radius)

            self.map[row, col] = self.cell_type

            self.update_view()
        elif self.tool == Tool.Square.value:
            row, col = self.points_in_square(x, y, self.edit_radius)

            self.map[row, col] = self.cell_type

            self.update_view()

    def handle_release(self, event):
        """
        The function to be fired when the mouse button is released on the
        canvas.
        """
        self.mouse_held = False

        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.tool == Tool.Line.value and self.edit_radius > 0:
            row, col = self.get_cells_near_line(
                self.x1, self.y1, x, y, self.edit_radius)

            self.map[row, col] = self.cell_type

            self.update_view()

        for _ in range(len(self.undo_buffer) - self.undo_buffer_pos - 1):
            self.undo_buffer.pop()

        if len(self.undo_buffer) == UNDO_BUFFER_SIZE:
            self.undo_buffer.popleft()
            self.undo_buffer_pos -= 1

        self.undo_buffer.append(np.copy(self.map))
        self.undo_buffer_pos += 1

    def undo(self, _=None):
        """
        Undo the user's last action
        """
        if self.undo_buffer_pos > 0:
            self.undo_buffer_pos -= 1
            self.map = np.copy(self.undo_buffer[self.undo_buffer_pos])
            self.update_view()

    def redo(self, _=None):
        """
        Redo the user's last action
        """
        if self.undo_buffer_pos < len(self.undo_buffer) - 1:
            self.undo_buffer_pos += 1
            self.map = np.copy(self.undo_buffer[self.undo_buffer_pos])
            self.update_view()

    def update_view(self):
        """
        Update the part of the GUI that can be drawn on.
        """
        self.img.set_data(self.map)
        self.canvas.draw()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--size', nargs=2, type=int, help="Map size")

    args = parser.parse_args()

    root = tk.Tk()
    MapEditor(root, map_size=args.size)
    root.winfo_toplevel().title("Map editor")
    root.mainloop()
