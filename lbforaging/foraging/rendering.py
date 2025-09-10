"""
2D rendering of Dynamic Multi-Level-based Foraging (DMLBF) domain
"""

import math
import os
import sys

import numpy as np
import six
from gymnasium import error

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): special environment variable settings for Apple systems to avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)
_BLUE = (0, 0, 255)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object): # Visualization Renderer Class - responsible for creating windows and rendering all elements in the environment
    def __init__(self, world_size): # world_size: The size of the world, in the format of (number of rows, number of columns)
        display = get_display(None)
        self.rows, self.cols = world_size
        
        # set the size of grid and icon
        self.grid_size = 50
        self.icon_size = 20
        # set the size of window
        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        # Enable OpenGL blending function for translucent effects
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # Load image resources
        script_dir = os.path.dirname(__file__)
        pyglet.resource.path = [os.path.join(script_dir, "icons")] # set path of resourses
        pyglet.resource.reindex()

        self.img_apple = pyglet.resource.image("apple.png")
        self.img_agent = pyglet.resource.image("agent.png")

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top): # Set rendering boundary
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def render(self, env, return_rgb_array=False):
        """
        The main methods of rendering the environment

        Parameters:
            env: The environment object to be rendered
            return_rgb_array: Whether to return the RGB array instead of displaying the graph
        return:
            If `return_rgb_array` is set to `True`, the RGB array representation of the environment will be returned;
            Otherwise, return the status of whether the window is open (Boolean value)
        """
        glClearColor(*_WHITE, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_food(env)
        self._draw_players(env)

        time_step_label = pyglet.text.Label(
            f"Step: {env.current_step}",
            font_name="Times New Roman",
            font_size=14,
            bold=True,
            x=20,
            y=self.height - 20,
            anchor_x="left",
            anchor_y="top",
            color=(*_BLUE, 255),
        )
        time_step_label.draw()

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        # vertical lines
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,  # LEFT X
                        (self.grid_size + 1) * r + 1,  # Y
                        (self.grid_size + 1) * self.cols,  # RIGHT X
                        (self.grid_size + 1) * r + 1,  # Y
                    ),
                ),
                ("c3B", (*_BLACK, *_BLACK)),
            )

        # horizontal lines
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * c + 1,  # X
                        0,  # BOTTOM Y
                        (self.grid_size + 1) * c + 1,  # X
                        (self.grid_size + 1) * self.rows,  # TOP X
                    ),
                ),
                ("c3B", (*_BLACK, *_BLACK)),
            )
        batch.draw()

    def _draw_food(self, env):
        idxes = list(zip(*np.where(np.any(env.field > 0, axis=2))))
        visible_apples = []
        invisible_apples = []
        batch_visible = pyglet.graphics.Batch()
        batch_invisible = pyglet.graphics.Batch()

        # print(env.field)
        for row, col in idxes:
            apple = pyglet.sprite.Sprite(
                    self.img_apple,
                    (self.grid_size + 1) * col,
                    self.height - (self.grid_size + 1) * (row + 1),
                    batch=None,
                )
            apple.update(scale=self.grid_size / apple.width)
            if env.is_food_visible(row, col):
                apple.batch = batch_visible
                visible_apples.append(apple)
            else:
                apple.batch = batch_invisible
                apple.opacity = 50
                invisible_apples.append(apple)
        
        batch_visible.draw()
        batch_invisible.draw()

        for row, col in idxes:
            level_str = ",".join(map(str, env.field[row, col]))
            self._draw_level_badge(row, col, level_str)
            spawn_time = str(env.food_spawn_time[row, col])
            self._draw_time_badge(row, col, spawn_time)

    def _draw_players(self, env):
        players = []
        batch = pyglet.graphics.Batch()

        for player in env.players:
            row, col = player.position
            players.append(
                pyglet.sprite.Sprite(
                    self.img_agent,
                    (self.grid_size + 1) * col,
                    self.height - (self.grid_size + 1) * (row + 1),
                    batch=batch,
                )
            )
        for p in players:
            p.update(scale=self.grid_size / p.width)
        batch.draw()
        for p in env.players:
            level_str = ",".join(map(str, p.level))
            self._draw_level_badge(*p.position, level_str)

    def _draw_level_badge(self, row, col, level): # draw level badge
        resolution = 6
        radius = self.grid_size / 4 # radius of badge
        # calculate position of badge
        badge_x = col * (self.grid_size + 1) + (3 / 4) * (self.grid_size + 1)
        badge_y = (
            self.height
            - (self.grid_size + 1) * (row + 1)
            + (1 / 4) * (self.grid_size + 1)
        )

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_WHITE)
        circle.draw(GL_POLYGON)
        glColor3ub(*_BLACK)
        circle.draw(GL_LINE_LOOP)

        font_size = 8 if len(str(level)) > 3 else 12
        label = pyglet.text.Label(
            str(level),
            font_name="Times New Roman",
            font_size=font_size,
            bold=True,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
            color=(*_BLACK, 255),
        )
        label.draw()

    def _draw_time_badge(self, row, col, time): # draw time badge
        resolution = 20
        radius = self.grid_size / 6 # radius of badge
        # calculate position of badge
        badge_x = col * (self.grid_size + 1) + (3 / 4) * (self.grid_size + 1)
        badge_y = (
            self.height
            - (self.grid_size + 1) * (row + 1)
            + (3 / 4) * (self.grid_size + 1)
        )

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_WHITE)
        circle.draw(GL_POLYGON)
        glColor3ub(*_BLACK)
        circle.draw(GL_LINE_LOOP)

        font_size = 8 if len(str(time)) > 2 else 12
        label = pyglet.text.Label(
            str(time),
            font_name="Times New Roman",
            font_size=font_size,
            bold=True,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
            color=(*_BLACK, 200),
        )
        label.draw()

