"""
AI Sudoku Solver — Gradio app
=============================

Upload a photo of a Sudoku puzzle; the app detects the grid, recognizes the
digits with a CNN, solves the puzzle with a backtracking algorithm, and returns
both the recognized board and the solved board as rendered images.

Pipeline:
  1. Adaptive thresholding → largest contour → perspective-warp to a square grid.
  2. Slice into 81 cells, skip empties, center-bbox-and-resize each digit to 28×28.
  3. CNN digit recognition with dual-threshold confidence acceptance and a
     1-vs-7 disambiguation heuristic.
  4. Backtracking solver with row/column/3×3-box constraint check.
  5. Render the recognized and solved grids and return both.
"""

import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("SUDOKU_MODEL_PATH", "sudoku_model.keras")

CONFIDENCE_STRICT = 0.90      # accept any prediction above this
CONFIDENCE_BORDERLINE = 0.70  # accept above this if not a known confused-pair
MAX_1V7_GAP = 0.15            # if model says 7 but 1 is close, flip to 1

GRID_RENDER_SIZE = 450        # pixels for the rendered output images

model = load_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Computer-vision pipeline
# ---------------------------------------------------------------------------

def reorder_points(pts):
    """Order four corner points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape((4, 2))
    out = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    out[0] = pts[np.argmin(s)]      # top-left
    out[2] = pts[np.argmax(s)]      # bottom-right
    out[1] = pts[np.argmin(diff)]   # top-right
    out[3] = pts[np.argmax(diff)]   # bottom-left
    return out


def warp_grid(image_bgr):
    """Detect the largest 4-corner contour and warp it to a square grid."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")

    sudoku_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(sudoku_contour, True)
    approx = cv2.approxPolyDP(sudoku_contour, 0.02 * peri, True)

    if len(approx) != 4:
        raise ValueError("Could not detect a 4-corner Sudoku grid in the image.")

    pts = reorder_points(approx)
    side = int(max(np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)))
    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(gray, M, (side, side))


def split_into_cells(warped):
    """Split a square warped grid into 81 cell images, row-major."""
    side = warped.shape[0]
    cell_size = side // 9
    return [
        warped[r * cell_size : (r + 1) * cell_size, c * cell_size : (c + 1) * cell_size]
        for r in range(9)
        for c in range(9)
    ]


def is_empty_cell(cell, white_ratio_threshold=0.95):
    """A cell is treated as empty if it is mostly bright pixels."""
    cell_28 = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)
    return np.sum(cell_28 > 200) / cell_28.size > white_ratio_threshold


def preprocess_cell(cell):
    """
    Center the digit on a 28×28 canvas (MNIST-style) so the CNN sees what it
    was trained on, not raw grid cells with borders and offsets.
    """
    cell = cv2.resize(cell, (100, 100), interpolation=cv2.INTER_AREA)
    thresh = cv2.adaptiveThreshold(
        cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    digit = thresh[y : y + h, x : x + w]
    if max(h, w) <= 0:
        return None

    scale = 20.0 / max(h, w)
    digit = cv2.resize(
        digit,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - digit.shape[1]) // 2
    y_off = (28 - digit.shape[0]) // 2
    canvas[y_off : y_off + digit.shape[0], x_off : x_off + digit.shape[1]] = digit

    canvas = cv2.erode(canvas, np.ones((2, 2), np.uint8), iterations=1)
    if np.count_nonzero(canvas) < 30:
        return None

    return (canvas.astype("float32") / 255.0).reshape(1, 28, 28, 1)


def recognize_grid(cells):
    """
    Run CNN inference on every cell, applying:
      - dual confidence thresholds (strict 0.90, borderline 0.70)
      - 1-vs-7 disambiguation when the gap between top-1 (7) and 1 is small
    """
    grid = np.zeros((9, 9), dtype=int)
    for i, cell in enumerate(cells):
        row, col = i // 9, i % 9
        if is_empty_cell(cell):
            continue
        if np.mean(cell) > 250 or (np.max(cell) - np.min(cell)) < 30:
            continue

        x = preprocess_cell(cell)
        if x is None:
            continue

        prediction = model.predict(x, verbose=0)[0]
        top2 = prediction.argsort()[-2:][::-1]
        digit, second = int(top2[0]), int(top2[1])
        confidence = float(prediction[digit])

        # 1-vs-7 disambiguation: model often confuses these on raw grid cells
        if digit == 7 and second == 1 and (confidence - prediction[1]) < MAX_1V7_GAP:
            digit = 1
            confidence = float(prediction[1])

        if digit == 0:
            continue
        if confidence > CONFIDENCE_STRICT or confidence > CONFIDENCE_BORDERLINE:
            grid[row, col] = digit

    return grid


# ---------------------------------------------------------------------------
# Sudoku solver (backtracking with constraint check)
# ---------------------------------------------------------------------------

def is_valid_placement(grid, row, col, num):
    if num in grid[row]:
        return False
    if num in grid[:, col]:
        return False
    r0, c0 = 3 * (row // 3), 3 * (col // 3)
    if num in grid[r0 : r0 + 3, c0 : c0 + 3]:
        return False
    return True


def solve(grid):
    for row in range(9):
        for col in range(9):
            if grid[row, col] == 0:
                for num in range(1, 10):
                    if is_valid_placement(grid, row, col, num):
                        grid[row, col] = num
                        if solve(grid):
                            return True
                        grid[row, col] = 0
                return False
    return True


def is_grid_consistent(grid):
    """Verify the recognized grid has no row/column/box conflicts before solving."""
    for r in range(9):
        for c in range(9):
            num = grid[r, c]
            if num != 0:
                grid[r, c] = 0
                if not is_valid_placement(grid, r, c, num):
                    grid[r, c] = num
                    return False
                grid[r, c] = num
    return True


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_grid(grid, output_path):
    img_size = GRID_RENDER_SIZE
    cell_size = img_size // 9
    img = np.full((img_size, img_size, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(9):
        for j in range(9):
            num = int(grid[i, j])
            if num == 0:
                continue
            text = str(num)
            (tw, th), _ = cv2.getTextSize(text, font, 1.2, 2)
            tx = j * cell_size + (cell_size - tw) // 2
            ty = i * cell_size + (cell_size + th) // 2
            cv2.putText(img, text, (tx, ty), font, 1.2, (0, 0, 0), 2)

    for i in range(10):
        thickness = 3 if i % 3 == 0 else 1
        cv2.line(img, (0, i * cell_size), (img_size, i * cell_size), (180, 150, 110), thickness)
        cv2.line(img, (i * cell_size, 0), (i * cell_size, img_size), (180, 150, 110), thickness)

    cv2.imwrite(str(output_path), img)
    return str(output_path)


# ---------------------------------------------------------------------------
# Top-level processing
# ---------------------------------------------------------------------------

def process_image(image):
    """Full pipeline: image → recognized grid → solved grid → two rendered images."""
    if image is None:
        raise ValueError("No image provided.")

    # Gradio's numpy mode is RGB; OpenCV expects BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    warped = warp_grid(image_bgr)
    cells = split_into_cells(warped)
    recognized = recognize_grid(cells)

    if not is_grid_consistent(recognized):
        raise ValueError(
            "Recognized grid has row/column/box conflicts. "
            "Try a clearer photo or a more frontal angle."
        )

    solved = recognized.copy()
    if not solve(solved):
        raise ValueError("Recognized digits don't form a solvable puzzle.")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    return (
        render_grid(recognized, out_dir / "predicted_grid.png"),
        render_grid(solved, out_dir / "solved_grid.png"),
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def solve_uploaded_image(image):
    try:
        return process_image(image)
    except Exception as exc:  # surface errors back to the UI
        raise gr.Error(str(exc)) from exc


with gr.Blocks(title="AI Sudoku Solver") as demo:
    gr.Markdown(
        """
        # AI Sudoku Solver
        Upload a photo of a Sudoku puzzle. The pipeline detects the grid, recognizes
        the digits with a CNN, and solves the puzzle with backtracking.

        **Tips:** photograph the puzzle straight-on, with even lighting and the
        full grid visible. Skewed angles, glare, or partial crops reduce accuracy.
        """
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="📷 Upload Sudoku image")
            solve_button = gr.Button("Solve puzzle 🔎", variant="primary")
        with gr.Column():
            predicted_img = gr.Image(type="filepath", label="Recognized grid (before solving)")
            solved_img = gr.Image(type="filepath", label="Solved grid")

    solve_button.click(
        fn=solve_uploaded_image,
        inputs=input_image,
        outputs=[predicted_img, solved_img],
    )


if __name__ == "__main__":
    demo.launch()
