import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("sudoku_model.keras")  # Must upload this too

def draw_grid(grid, output_path="output.png"):
    img_size = 450
    cell_size = img_size // 9
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    for i in range(9):
        for j in range(9):
            num = grid[i][j]
            if num != 0:
                text = str(num)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = j * cell_size + (cell_size - text_size[0]) // 2
                text_y = i * cell_size + (cell_size + text_size[1]) // 2
                cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    for i in range(10):
        line_thickness = 3 if i % 3 == 0 else 1
        cv2.line(img, (0, i * cell_size), (img_size, i * cell_size), (180, 150, 110), line_thickness)
        cv2.line(img, (i * cell_size, 0), (i * cell_size, img_size), (180, 150, 110), line_thickness)
    cv2.imwrite(output_path, img)
    return output_path

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sudoku_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(sudoku_contour, True)
    approx = cv2.approxPolyDP(sudoku_contour, 0.02 * peri, True)
    if len(approx) != 4:
        raise ValueError("Grid not detected properly")

    def reorder_points(pts):
        pts = pts.reshape((4, 2))
        new_pts = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        new_pts[0] = pts[np.argmin(s)]
        new_pts[2] = pts[np.argmax(s)]
        new_pts[1] = pts[np.argmin(diff)]
        new_pts[3] = pts[np.argmax(diff)]
        return new_pts

    pts = reorder_points(approx)
    side = int(max([np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]))
    dst_pts = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(gray, M, (side, side))

    cell_size = side // 9
    cells = [warped[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] for r in range(9) for c in range(9)]

    def is_empty(cell, threshold=0.98):
        return np.sum(cell > 200) / cell.size > threshold

    def preprocess_cell(cell):
        cell = cell[5:-5, 5:-5]
        cell = cv2.resize(cell, (28, 28))
        cell = cell.astype('float32') / 255.0
        return cell.reshape(1, 28, 28, 1)

    sudoku_grid = np.zeros((9, 9), dtype=int)
    for i, cell in enumerate(cells):
        if is_empty(cell):
            continue
        input_img = preprocess_cell(cell)
        prediction = model.predict(input_img)
        confidence = np.max(prediction)
        predicted_digit = np.argmax(prediction)
        if confidence > 0.85:
            row, col = i // 9, i % 9
            sudoku_grid[row][col] = predicted_digit

    draw_grid(sudoku_grid, "predicted_grid.png")

    def is_valid(grid, row, col, num):
        if num in grid[row] or num in grid[:, col]:
            return False
        r0, c0 = 3*(row//3), 3*(col//3)
        if num in grid[r0:r0+3, c0:c0+3]:
            return False
        return True

    def solve(grid):
        for row in range(9):
            for col in range(9):
                if grid[row][col] == 0:
                    for num in range(1, 10):
                        if is_valid(grid, row, col, num):
                            grid[row][col] = num
                            if solve(grid):
                                return True
                            grid[row][col] = 0
                    return False
        return True

    solved_grid = sudoku_grid.copy()
    if not solve(solved_grid):
        raise ValueError("Could not solve the puzzle")
    
    draw_grid(solved_grid, "solved_grid.png")
    return "predicted_grid.png", "solved_grid.png"

def solve_uploaded_image(image):
    try:
        return process_image(image)
    except Exception as e:
        return None, f"Error: {str(e)}"

with gr.Blocks(title="AI Sudoku Solver") as demo:
    gr.Markdown(
        """
        # AI Sudoku Solver  
        Upload a Sudoku puzzle image and let the AI detect and solve it!
        - Digit recognition with CNN  
        - Solving using AI backtracking  
        - Clean visual outputs
        """
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="📷 Upload Sudoku Image")
            solve_button = gr.Button(" Solve Puzzle🔎")
        with gr.Column():
            predicted_img = gr.Image(type="filepath", label="Predicted Grid (Before Solving)")
            solved_img = gr.Image(type="filepath", label="Solved Sudoku Grid")

    solve_button.click(fn=solve_uploaded_image, inputs=input_image, outputs=[predicted_img, solved_img])

demo.launch()
