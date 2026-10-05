

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# Configuration

DISPLAY_SIZE = 480
GRID_SIZES = (3, 4, 5)


TRANSFORMATION_COUNTS = {
    3: 6,
    4: 12,
    5: 20,
}

MAX_HINTS = 3


@dataclass
class Tile:

    tile_id: int
    home_index: int
    original_image: np.ndarray
    image: np.ndarray

    def reset(self) -> None:
        # Restore the tile to its original orientation.
        self.image = self.original_image.copy()

    def is_orientation_correct(self) -> bool:
        #eturn True when the tile has its original orientation
        return np.array_equal(self.image, self.original_image)


class TileTransformation(ABC):
    #Base class for all scramble transformations.

    @abstractmethod
    def apply(self, game: "PuzzleGame") -> None:
        # Apply the transformation to the supplied puzzle."""
        raise NotImplementedError


class SwapTransformation(TileTransformation):
    #Swap the positions of two tiles.

    def __init__(self, tile_a_id: int, tile_b_id: int) -> None:
        self.tile_a_id = tile_a_id
        self.tile_b_id = tile_b_id

    def apply(self, game: "PuzzleGame") -> None:
        game.swap_tile_ids(self.tile_a_id, self.tile_b_id)


class RotateTransformation(TileTransformation):
    #Rotate one tile clockwise by 90, 180 or 270 degrees.

    def __init__(self, tile_id: int, degrees: int) -> None:
        self.tile_id = tile_id
        self.degrees = degrees

    def apply(self, game: "PuzzleGame") -> None:
        game.rotate_tile(self.tile_id, self.degrees)


class FlipTransformation(TileTransformation):
    #Flip one tile horizontally or vertically.

    def __init__(self, tile_id: int, direction: str) -> None:
        self.tile_id = tile_id
        self.direction = direction

    def apply(self, game: "PuzzleGame") -> None:
        game.flip_tile(self.tile_id, self.direction)



# Puzzle game logic


class PuzzleGame:
    #Contains the puzzle state and all non-GUI game operations.

    def __init__(self, grid_size: int) -> None:
        if grid_size not in GRID_SIZES:
            raise ValueError("Grid size must be 3, 4 or 5.")

        self.grid_size = grid_size
        self.tiles: list[Tile] = []
        self._scramble_history: list[TileTransformation] = []

        self.moves = 0
        self.hints_used = 0
        self.score = 1000
        self.completed = False

    # Image preparation 

    @staticmethod
    def prepare_image(image: np.ndarray, size: int = DISPLAY_SIZE) -> np.ndarray:
        
        if image is None or image.size == 0:
            raise ValueError("The selected image is empty or invalid.")

        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("The selected image has invalid dimensions.")

        scale = min(size / width, size / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )

        canvas = np.zeros((size, size, 3), dtype=np.uint8)

        x_offset = (size - new_width) // 2
        y_offset = (size - new_height) // 2

        canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = resized

        return canvas

    def load_image(self, image: np.ndarray) -> None:
        # Create evenly sized tiles and scramble them.
        prepared = self.prepare_image(image)

        tile_size = DISPLAY_SIZE // self.grid_size
        self.tiles.clear()
        self._scramble_history.clear()

        tile_id = 0

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y1 = row * tile_size
                y2 = y1 + tile_size
                x1 = col * tile_size
                x2 = x1 + tile_size

                tile_image = prepared[y1:y2, x1:x2].copy()

                self.tiles.append(
                    Tile(
                        tile_id=tile_id,
                        home_index=tile_id,
                        original_image=tile_image.copy(),
                        image=tile_image.copy(),
                    )
                )

                tile_id += 1

        self.moves = 0
        self.hints_used = 0
        self.score = 1000
        self.completed = False

        self._create_and_apply_scramble()



    def _create_and_apply_scramble(self) -> None:
        """Generate all scramble transformations before applying them.

        Each tile is selected at most once by the scramble generator.
        This gives a deterministic guarantee that a tile is not targeted
        repeatedly during the initial scramble.
        """
        total_operations = TRANSFORMATION_COUNTS[self.grid_size]
        tile_count = self.grid_size * self.grid_size

        # Each swap consumes two unique tiles. Rotate and flip consume one.
        # Find all valid operation-count combinations that:
        #   1. use all three transformation types,
        #   2. contain exactly the required number of operations,
        #   3. do not require more unique tile targets than available.
        valid_distributions = []

        for swaps in range(1, total_operations):
            for rotates in range(1, total_operations):
                flips = total_operations - swaps - rotates

                if flips < 1:
                    continue

                targets_needed = (2 * swaps) + rotates + flips

                if targets_needed <= tile_count:
                    valid_distributions.append((swaps, rotates, flips))

        if not valid_distributions:
            raise RuntimeError(
                "Could not generate a valid scramble for this grid size."
            )

        swaps, rotates, flips = random.choice(valid_distributions)

        available_ids = list(range(tile_count))
        random.shuffle(available_ids)

        operations: list[TileTransformation] = []

        # Assign unique tile IDs to all transformations.
        cursor = 0

        for _ in range(swaps):
            a = available_ids[cursor]
            b = available_ids[cursor + 1]
            cursor += 2
            operations.append(SwapTransformation(a, b))

        for _ in range(rotates):
            tile_id = available_ids[cursor]
            cursor += 1
            degrees = random.choice((90, 180, 270))
            operations.append(RotateTransformation(tile_id, degrees))

        for _ in range(flips):
            tile_id = available_ids[cursor]
            cursor += 1
            direction = random.choice(("horizontal", "vertical"))
            operations.append(FlipTransformation(tile_id, direction))

        # Randomise the order so transformation types are mixed.
        random.shuffle(operations)

        self._scramble_history = operations

        # All transformations are generated first, then applied.
        for transformation in self._scramble_history:
            transformation.apply(self)

        # A scramble must not accidentally leave the puzzle solved.
        if self.is_solved():
            self._fallback_unscramble()

    def _fallback_unscramble(self) -> None:
        # Guarantee that a puzzle is visibly scrambled.
        if len(self.tiles) < 2:
            return

        self.swap_positions(0, 1)

        # If the swap somehow did not make the puzzle incorrect, rotate a tile.
        if self.is_solved():
            self.rotate_position(0, 90)

    # Tile operations 

    def find_position_by_tile_id(self, tile_id: int) -> int:
        # Return the current position of a tile ID.
        for position, tile in enumerate(self.tiles):
            if tile.tile_id == tile_id:
                return position
        raise ValueError(f"Tile ID {tile_id} does not exist.")

    def swap_tile_ids(self, tile_a_id: int, tile_b_id: int) -> None:
        #Swap two tiles by their stable IDs.
        pos_a = self.find_position_by_tile_id(tile_a_id)
        pos_b = self.find_position_by_tile_id(tile_b_id)
        self.swap_positions(pos_a, pos_b)

    def swap_positions(self, position_a: int, position_b: int) -> None:
        #Swap two tile positions.
        self.tiles[position_a], self.tiles[position_b] = (
            self.tiles[position_b],
            self.tiles[position_a],
        )

    def rotate_tile(self, tile_id: int, degrees: int) -> None:
        #Rotate a tile clockwise.
        position = self.find_position_by_tile_id(tile_id)
        self.rotate_position(position, degrees)

    def rotate_position(self, position: int, degrees: int = 90) -> None:
        #Rotate a tile at a given position clockwise.
        if degrees not in (90, 180, 270):
            raise ValueError("Rotation must be 90, 180 or 270 degrees.")

        turns = degrees // 90

        # np.rot90 uses counter-clockwise turns for positive values.
        self.tiles[position].image = np.rot90(
            self.tiles[position].image,
            k=-turns,
        ).copy()

    def flip_tile(self, tile_id: int, direction: str) -> None:
        #Flip a tile horizontally or vertically.
        position = self.find_position_by_tile_id(tile_id)
        self.flip_position(position, direction)

    def flip_position(self, position: int, direction: str) -> None:
        #Flip a tile at a given position.
        if direction == "horizontal":
            # Flip left/right.
            self.tiles[position].image = cv2.flip(
                self.tiles[position].image,
                1,
            ).copy()
        elif direction == "vertical":
            # Flip top/bottom.
            self.tiles[position].image = cv2.flip(
                self.tiles[position].image,
                0,
            ).copy()
        else:
            raise ValueError("Flip direction must be horizontal or vertical.")

    # ------------------------- Player actions -------------------------

    def player_swap(self, position_a: int, position_b: int) -> None:
       #Perform one player swap move.
        self.swap_positions(position_a, position_b)
        self._record_move()

    def player_rotate(self, position: int) -> None:
       #Perform one player rotation move.
        self.rotate_position(position, 90)
        self._record_move()

    def player_flip(self, position: int) -> None:
       #Perform one player horizontal flip move.
        self.flip_position(position, "horizontal")
        self._record_move()

    def _record_move(self) -> None:
       #update move and score counters after one player action.
        self.moves += 1
        self.score = max(0, 1000 - (self.moves * 10) - (self.hints_used * 50))

        if self.is_solved():
            self.completed = True

    # ------------------------- Status and solving -------------------------

    def is_tile_correct(self, position: int) -> bool:
        #Return whether the tile at this position is fully correct.
        tile = self.tiles[position]
        return (
            tile.home_index == position
            and tile.is_orientation_correct()
        )

    def incorrect_positions(self) -> list[int]:
        #Return positions containing incorrect tiles.
        return [
            position
            for position in range(len(self.tiles))
            if not self.is_tile_correct(position)
        ]

    def incorrect_count(self) -> int:
        #Return the number of tiles still incorrect.
        return len(self.incorrect_positions())

    def is_solved(self) -> bool:
        #Return True when every tile is in its home position and orientation.
        return self.incorrect_count() == 0

    def use_hint(self) -> tuple[int, int] | None:
        """Select one incorrect tile and return (current_position, home_position).

        The hint count is limited to MAX_HINTS.
        """
        if self.hints_used >= MAX_HINTS:
            return None

        incorrect = self.incorrect_positions()
        if not incorrect:
            return None

        current_position = random.choice(incorrect)
        home_position = self.tiles[current_position].home_index

        self.hints_used += 1
        self.score = max(0, 1000 - (self.moves * 10) - (self.hints_used * 50))

        return current_position, home_position

    def solve(self) -> None:
        #Instantly restore every tile and clear moves and score.
        # Restoring each tile to its original state is equivalent to undoing
        # all remaining scramble transformations from the player's viewpoint.
        self.tiles.sort(key=lambda tile: tile.home_index)

        for tile in self.tiles:
            tile.reset()

        self.moves = 0
        self.hints_used = 0
        self.score = 0
        self.completed = True


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------

class PuzzleApp(tk.Tk):
    #Tkinter application for the HIT137 image puzzle.

    def __init__(self) -> None:
        super().__init__()

        self.title("HIT137 Assignment 3 - Image Scramble Puzzle")
        self.geometry("1080x720")
        self.minsize(1000, 680)
        self.configure(bg="#202124")

        self.game: PuzzleGame | None = None
        self.original_image: np.ndarray | None = None

        self.original_photo: ImageTk.PhotoImage | None = None
        self.transformed_photo: ImageTk.PhotoImage | None = None

        self.selected_position: int | None = None
        self.hint_positions: tuple[int, int] | None = None

        self._build_styles()
        self._build_interface()
        self._update_controls()

    # ------------------------- GUI construction -------------------------

    def _build_styles(self) -> None:
        #Configure ttk styles.
        style = ttk.Style(self)

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "TFrame",
            background="#202124",
        )
        style.configure(
            "TLabel",
            background="#202124",
            foreground="#f1f3f4",
            font=("Segoe UI", 11),
        )
        style.configure(
            "Title.TLabel",
            background="#202124",
            foreground="#ffffff",
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "TButton",
            font=("Segoe UI", 10, "bold"),
            padding=7,
        )
        style.configure(
            "TCombobox",
            padding=5,
        )

    def _build_interface(self) -> None:
        #Create the complete application layout.
        header = ttk.Frame(self)
        header.pack(fill="x", padx=18, pady=(14, 8))

        ttk.Label(
            header,
            text="HIT137 Image Scramble Puzzle",
            style="Title.TLabel",
        ).pack(side="left")

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=18, pady=8)

        ttk.Label(controls, text="Grid size:").pack(side="left")

        self.grid_var = tk.StringVar(value="3 x 3")
        self.grid_combo = ttk.Combobox(
            controls,
            textvariable=self.grid_var,
            values=("3 x 3", "4 x 4", "5 x 5"),
            state="readonly",
            width=8,
        )
        self.grid_combo.pack(side="left", padx=(6, 14))

        self.load_button = ttk.Button(
            controls,
            text="Load Image",
            command=self.load_image,
        )
        self.load_button.pack(side="left", padx=4)

        self.hint_button = ttk.Button(
            controls,
            text="Hint",
            command=self.show_hint,
        )
        self.hint_button.pack(side="left", padx=4)

        self.solve_button = ttk.Button(
            controls,
            text="Solve",
            command=self.solve_puzzle,
        )
        self.solve_button.pack(side="left", padx=4)

        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", padx=18, pady=(0, 8))

        self.moves_var = tk.StringVar(value="Moves: 0")
        self.tiles_var = tk.StringVar(value="Tiles left: 0")
        self.score_var = tk.StringVar(value="Score: 0")
        self.hints_var = tk.StringVar(value="Hints: 0 / 3")
        self.status_var = tk.StringVar(
            value="Load a JPG, PNG or BMP image to begin."
        )

        ttk.Label(status_frame, textvariable=self.moves_var).pack(
            side="left", padx=(0, 20)
        )
        ttk.Label(status_frame, textvariable=self.tiles_var).pack(
            side="left", padx=(0, 20)
        )
        ttk.Label(status_frame, textvariable=self.score_var).pack(
            side="left", padx=(0, 20)
        )
        ttk.Label(status_frame, textvariable=self.hints_var).pack(
            side="left"
        )

        ttk.Label(
            self,
            textvariable=self.status_var,
        ).pack(fill="x", padx=18, pady=(0, 8))

        image_frame = ttk.Frame(self)
        image_frame.pack(expand=True, fill="both", padx=18, pady=8)

        left_frame = ttk.Frame(image_frame)
        left_frame.pack(side="left", expand=True, fill="both", padx=(0, 8))

        right_frame = ttk.Frame(image_frame)
        right_frame.pack(side="left", expand=True, fill="both", padx=(8, 0))

        ttk.Label(
            left_frame,
            text="Original Image",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=(0, 5))

        ttk.Label(
            right_frame,
            text="Scrambled / Play Area",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=(0, 5))

        self.original_canvas = tk.Canvas(
            left_frame,
            width=DISPLAY_SIZE,
            height=DISPLAY_SIZE,
            bg="#101114",
            highlightthickness=1,
            highlightbackground="#4a4d52",
        )
        self.original_canvas.pack()

        self.transformed_canvas = tk.Canvas(
            right_frame,
            width=DISPLAY_SIZE,
            height=DISPLAY_SIZE,
            bg="#101114",
            highlightthickness=1,
            highlightbackground="#4a4d52",
        )
        self.transformed_canvas.pack()

        # Left click, Shift + left click, and right click are mapped exactly
        # as required by the assignment.
        self.transformed_canvas.bind("<Button-1>", self.on_left_click)
        self.transformed_canvas.bind("<Button-3>", self.on_right_click)

        instruction = (
            "Controls: Left click selects/swaps. "
            "Click the same tile again to deselect. "
            "Right click rotates 90° clockwise. "
            "Shift + left click flips horizontally."
        )

        ttk.Label(
            self,
            text=instruction,
        ).pack(fill="x", padx=18, pady=(5, 14))

    # ------------------------- File loading -------------------------

    def load_image(self) -> None:
        #Open an image file and start a new puzzle round.
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            # Cancelled dialogs are handled gracefully.
            self.status_var.set("Image loading cancelled.")
            return

        try:
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError(
                    "The selected file is not a readable JPG, PNG or BMP image."
                )

            grid_size = int(self.grid_var.get()[0])

            self.game = PuzzleGame(grid_size)
            self.game.load_image(image)

            self.original_image = self.game.prepare_image(image)

            self.selected_position = None
            self.hint_positions = None

            self._display_original()
            self._render_transformed()

            filename = Path(file_path).name
            self.status_var.set(
                f"Loaded {filename}. {grid_size} x {grid_size} puzzle ready."
            )

            self._update_controls()

        except (ValueError, cv2.error, OSError) as exc:
            self.game = None
            self.original_image = None
            self._clear_canvases()
            messagebox.showerror(
                "Image Loading Error",
                str(exc),
                parent=self,
            )
            self.status_var.set("Could not load the selected image.")
            self._update_controls()

        except Exception as exc:
            self.game = None
            self.original_image = None
            self._clear_canvases()
            messagebox.showerror(
                "Unexpected Error",
                f"An unexpected error occurred:\n\n{exc}",
                parent=self,
            )
            self.status_var.set("An unexpected error occurred.")
            self._update_controls()

    # ------------------------- Rendering -------------------------

    @staticmethod
    def _cv_to_photo(image: np.ndarray) -> ImageTk.PhotoImage:
        # Convert an OpenCV BGR image into a Tkinter-compatible image.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        return ImageTk.PhotoImage(pil_image)

    def _display_original(self) -> None:
        #Display the original reference image and its grid overlay.
        if self.original_image is None:
            return

        self.original_photo = self._cv_to_photo(self.original_image)

        self.original_canvas.delete("all")
        self.original_canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.original_photo,
        )

        self._draw_grid(
            self.original_canvas,
            int(self.grid_var.get()[0]),
        )

        self._draw_hint_on_original()

    def _render_transformed(self) -> None:
        # Reassemble the current tiles and redraw the play area.
        if self.game is None:
            return

        grid = self.game.grid_size
        tile_size = DISPLAY_SIZE // grid

        assembled = np.zeros(
            (DISPLAY_SIZE, DISPLAY_SIZE, 3),
            dtype=np.uint8,
        )

        for position, tile in enumerate(self.game.tiles):
            row = position // grid
            col = position % grid

            y1 = row * tile_size
            y2 = y1 + tile_size
            x1 = col * tile_size
            x2 = x1 + tile_size

            assembled[y1:y2, x1:x2] = tile.image

        self.transformed_photo = self._cv_to_photo(assembled)

        self.transformed_canvas.delete("all")
        self.transformed_canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.transformed_photo,
        )

        self._draw_grid(self.transformed_canvas, grid)
        self._draw_selection()
        self._draw_correct_ticks()
        self._draw_hint_on_transformed()

        self._update_status()

    def _draw_grid(self, canvas: tk.Canvas, grid: int) -> None:
        # Draw faint tile boundaries.
        tile_size = DISPLAY_SIZE // grid

        for i in range(1, grid):
            offset = i * tile_size

            canvas.create_line(
                offset,
                0,
                offset,
                DISPLAY_SIZE,
                fill="#9aa0a6",
                width=1,
            )
            canvas.create_line(
                0,
                offset,
                DISPLAY_SIZE,
                offset,
                fill="#9aa0a6",
                width=1,
            )

        canvas.create_rectangle(
            0,
            0,
            DISPLAY_SIZE - 1,
            DISPLAY_SIZE - 1,
            outline="#b0b3b8",
            width=1,
        )

    def _draw_selection(self) -> None:
        # Highlight the currently selected tile.
        if self.selected_position is None or self.game is None:
            return

        grid = self.game.grid_size
        tile_size = DISPLAY_SIZE // grid

        row = self.selected_position // grid
        col = self.selected_position % grid

        x1 = col * tile_size + 3
        y1 = row * tile_size + 3
        x2 = (col + 1) * tile_size - 3
        y2 = (row + 1) * tile_size - 3

        self.transformed_canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline="#ff9800",
            width=4,
        )

    def _draw_correct_ticks(self) -> None:
       # Draw a small green tick on every correctly placed tile.
        if self.game is None:
            return

        grid = self.game.grid_size
        tile_size = DISPLAY_SIZE // grid

        for position in range(len(self.game.tiles)):
            if not self.game.is_tile_correct(position):
                continue

            row = position // grid
            col = position % grid

            x = col * tile_size + 12
            y = row * tile_size + 14

            # Small green circle.
            self.transformed_canvas.create_oval(
                x - 9,
                y - 9,
                x + 9,
                y + 9,
                fill="#2e7d32",
                outline="#ffffff",
                width=1,
            )

            # Tick mark.
            self.transformed_canvas.create_line(
                x - 5,
                y,
                x - 1,
                y + 4,
                x + 6,
                y - 5,
                fill="#ffffff",
                width=2,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
            )

    def _draw_hint_on_transformed(self) -> None:
        # Draw the blue hint circle on the selected incorrect tile.
        if self.hint_positions is None or self.game is None:
            return

        current_position, _ = self.hint_positions

        grid = self.game.grid_size
        tile_size = DISPLAY_SIZE // grid

        row = current_position // grid
        col = current_position % grid

        cx = col * tile_size + tile_size // 2
        cy = row * tile_size + tile_size // 2
        radius = max(12, tile_size // 5)

        self.transformed_canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            outline="#2196f3",
            width=4,
        )

    def _draw_hint_on_original(self) -> None:
        # Draw the blue hint circle on the tile's correct home position.
        if self.hint_positions is None or self.game is None:
            return

        _, home_position = self.hint_positions

        grid = self.game.grid_size
        tile_size = DISPLAY_SIZE // grid

        row = home_position // grid
        col = home_position % grid

        cx = col * tile_size + tile_size // 2
        cy = row * tile_size + tile_size // 2
        radius = max(12, tile_size // 5)

        self.original_canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            outline="#2196f3",
            width=4,
        )

    def _clear_canvases(self) -> None:
        # Clear both image areas.
        self.original_canvas.delete("all")
        self.transformed_canvas.delete("all")

    #  Mouse interaction 

    def _position_from_xy(self, x: int, y: int) -> int | None:
        # Convert canvas coordinates to a tile position.
        if self.game is None:
            return None

        if not (0 <= x < DISPLAY_SIZE and 0 <= y < DISPLAY_SIZE):
            # Off-image clicks are ignored.
            return None

        tile_size = DISPLAY_SIZE // self.game.grid_size
        col = x // tile_size
        row = y // tile_size

        position = row * self.game.grid_size + col

        if 0 <= position < len(self.game.tiles):
            return position

        return None

    def on_left_click(self, event: tk.Event) -> None:
        # Handle selection, swapping, and Shift + click flipping.
        if self.game is None or self.game.completed:
            return

        position = self._position_from_xy(event.x, event.y)

        if position is None:
            return

        # Clear the previous hint after the next move.
        self.hint_positions = None

        # Tk state bit 0x0001 is Shift on Windows/X11 Tk.
        shift_pressed = bool(event.state & 0x0001)

        try:
            if shift_pressed:
                # Shift + left click = horizontal flip.
                self.game.player_flip(position)
                self.selected_position = None

            elif self.selected_position is None:
                # First left click selects a tile.
                self.selected_position = position

            elif self.selected_position == position:
                # Clicking the selected tile again deselects it.
                self.selected_position = None

            else:
                # Second tile selected, swap the two.
                self.game.player_swap(self.selected_position, position)
                self.selected_position = None

            self._render_transformed()
            self._check_completion()

        except Exception as exc:
            messagebox.showerror(
                "Interaction Error",
                str(exc),
                parent=self,
            )

    def on_right_click(self, event: tk.Event) -> None:
        # Handle right click rotation.
        if self.game is None or self.game.completed:
            return

        position = self._position_from_xy(event.x, event.y)

        if position is None:
            return

        self.hint_positions = None
        self.selected_position = None

        try:
            self.game.player_rotate(position)
            self._render_transformed()
            self._check_completion()

        except Exception as exc:
            messagebox.showerror(
                "Rotation Error",
                str(exc),
                parent=self,
            )

    #  Hint and solve 

    def show_hint(self) -> None:
        # Show a hint on both images.
        if self.game is None or self.game.completed:
            return

        self.hint_positions = self.game.use_hint()

        if self.hint_positions is None:
            return

        self.selected_position = None

        self._display_original()
        self._render_transformed()
        self._update_controls()

        if self.game.hints_used >= MAX_HINTS:
            self.status_var.set(
                "Maximum of 3 hints reached. The Hint button is disabled."
            )
        else:
            self.status_var.set(
                "Hint shown. The blue circles disappear after your next move."
            )

    def solve_puzzle(self) -> None:
        # Instantly solve the current puzzle.
        if self.game is None or self.game.completed:
            return

        answer = messagebox.askyesno(
            "Solve Puzzle",
            "Solve the puzzle now? This will clear the moves and score.",
            parent=self,
        )

        if not answer:
            return

        self.game.solve()
        self.selected_position = None
        self.hint_positions = None

        self._render_transformed()
        self._update_controls()

        self.status_var.set(
            "Puzzle solved. Moves and score were cleared. Load another image to play again."
        )

    # ------------------------- State updates -------------------------

    def _check_completion(self) -> None:
        # Notify the player when the puzzle has been solved.
        if self.game is None:
            return

        if self.game.is_solved() and not self.game.completed:
            self.game.completed = True

        if self.game.completed:
            self.selected_position = None
            self.hint_positions = None

            self._render_transformed()
            self._update_controls()

            if self.game.score > 0:
                messagebox.showinfo(
                    "Puzzle Complete",
                    (
                        "Congratulations!\n\n"
                        f"Moves: {self.game.moves}\n"
                        f"Score: {self.game.score}\n"
                        "The puzzle is now locked.\n"
                        "Load another image to continue."
                    ),
                    parent=self,
                )

    def _update_status(self) -> None:
        # Update counters shown in the GUI.
        if self.game is None:
            self.moves_var.set("Moves: 0")
            self.tiles_var.set("Tiles left: 0")
            self.score_var.set("Score: 0")
            self.hints_var.set("Hints: 0 / 3")
            return

        self.moves_var.set(f"Moves: {self.game.moves}")
        self.tiles_var.set(f"Tiles left: {self.game.incorrect_count()}")
        self.score_var.set(f"Score: {self.game.score}")
        self.hints_var.set(
            f"Hints: {self.game.hints_used} / {MAX_HINTS}"
        )

    def _update_controls(self) -> None:
        # Enable or disable controls according to the current state
        has_game = self.game is not None
        active = has_game and not self.game.completed

        if active and self.game.hints_used < MAX_HINTS:
            self.hint_button.config(state="normal")
        else:
            self.hint_button.config(state="disabled")

        self.solve_button.config(
            state="normal" if active else "disabled"
        )

        # Grid size is chosen before the next image is loaded. It remains
        # selectable so a new image can start at a different grid size.
        self.grid_combo.config(state="readonly")


def main() -> None:
    # Application entry point.
    app = PuzzleApp()
    app.mainloop()


if __name__ == "__main__":
    main()
