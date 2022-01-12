# mapeditor.py
# Jeroen van den Berg 12887366
# GUI tool to create maps to run the LBM on.

from enum import Enum
from collections import deque

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


class CellType(Enum):
    NORMAL = 0
    WALL = 1
    BOUNDARY = 2


class Tool(Enum):
    BRUSH = 0
    LINE = 1


def get_map_from_file(file):
    """
    Load the contents of the file into a map for the cellular automaton.
    """
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()

            if i == 0:
                map_size = len(line)
                map_array = np.zeros([map_size, map_size], dtype=int)

            map_array[i] = np.array([int(x) for x in line])

    return map_size, map_array


cmap = colors.ListedColormap(
    ["k", "forestgreen", "r", "gray"])


class MapEditor:
    def __init__(self, root, map_size=None):
        if map_size is None:
            self.map_size = 100
        else:
            self.map_size = int(map_size)

        self.map = np.full([self.map_size, self.map_size],
                           CellType.NORMAL.value)

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
        self.tool = Tool.BRUSH.value
        self.cell_type = CellType.NORMAL.value
        self.edit_radius = self.map_size // 10

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

        tk.Label(frame, text="Edit radius:").pack()
        e = ttk.Entry(frame, textvariable=self.tk_edit_radius).pack()

    def open_file(self, _=None):
        """
        Open a file and load it to the canvas.
        """
        self.filename = tk.filedialog.askopenfilename()
        if not self.filename:
            return

        self.map_size, self.map = get_map_from_file(self.filename)

        # The edges consist of "unknown" tiles, which are not needed in the
        # editor.
        self.map_size -= 2
        self.map = self.map[1:-1, 1:-1]

        # Draw the map to the screen
        self.img = self.ax.imshow(self.map, interpolation='none', vmin=0,
                                  vmax=len(CellType), cmap=cmap, animated=True)
        self.update_view()

        self.edit_radius = self.map_size // 10

    def save_file(self, event=None):
        """
        Save the map that is being worked on to a file.
        """
        self.filename = tk.filedialog.asksaveasfilename()

        if self.filename:
            with open(self.filename, 'w') as file:
                file.write((self.map_size + 2) *
                           str(CellType.UNKNOWN.value) + "\n")
                for row in self.map:
                    file.write(str(CellType.UNKNOWN.value))
                    file.write("".join([str(x) for x in row]))
                    file.write(str(CellType.UNKNOWN.value) + "\n")
                file.write((self.map_size + 2) *
                           str(CellType.UNKNOWN.value) + "\n")

    def filter_out_of_bounds(self, row, col):
        """
        Only return cells in row, col that lie within the boundary of the map.
        """
        # Filter out of bounds rows.
        idx = np.logical_and(row >= 0, row < self.map_size)

        row = row[idx]
        col = col[idx]

        # Filter out of bounds cols.
        idx = np.logical_and(col >= 0, col < self.map_size)

        row = row[idx]
        col = col[idx]

        return row, col

    def get_cells_near_point(self, x, y, r):
        """
        Return cells that have a distance less than r to the point (x, y).
        """
        diameter = r * 2
        row, col = np.indices((diameter, diameter)) - r

        idx = np.hypot(row, col) <= r

        row = row[idx] + y
        col = col[idx] + x

        return self.filter_out_of_bounds(row, col)

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

    def handle_move(self, event):
        """
        The function to be fired when the mouse is moved around on the canvas.
        """
        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)
        if self.mouse_held and self.tool == Tool.BRUSH.value:
            # Color all cells that are closer than self.edit_radius to
            # (xdata, ydata) with the selected cell type
            row, col = self.get_cells_near_point(x, y, self.edit_radius)

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

        if self.tool == Tool.LINE.value and self.edit_radius > 0:
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
    parser = argparse.ArgumentParser(description='Proxy operations')

    parser.add_argument('--size', help="Map size")

    args = parser.parse_args()

    root = tk.Tk()
    MapEditor(root, map_size=args.size)
    root.winfo_toplevel().title("Map editor")
    root.mainloop()
