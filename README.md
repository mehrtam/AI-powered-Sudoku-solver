# 🧠 AI Sudoku Solver

An intelligent Sudoku-solving application powered by **Computer Vision**, **CNN-based digit recognition**, and a **backtracking algorithm**. Upload any photo of a Sudoku puzzle and get the fully solved grid in seconds!

![Demo Preview]
![solved_grid](https://github.com/user-attachments/assets/ae1c2047-7a59-4dbc-9a0f-7497887a1259)

---

## ✨ Features

- 📷 Image-based Sudoku puzzle recognition
- 🧠 Digit recognition using a trained CNN model (Keras/TensorFlow)
- 🧩 Automatic grid detection, perspective correction, and warping
- ✅ AI-based Sudoku solving using backtracking algorithm
- 🖼️ Interactive web UI built with (https://gradio.app)
- 🚀 Deployable on (https://huggingface.co/spaces)

---

## 🔍 How It Works

1. **Preprocessing**: Convert to grayscale, blur, and apply adaptive thresholding.
2. **Grid Detection**: Find the largest contour (Sudoku outline) and warp it.
3. **Digit Classification**: Use a CNN model to predict digits from each cell.
4. **Puzzle Solving**: Apply a backtracking algorithm to fill in empty cells.
5. **Result Display**: Render the predicted and solved grids as styled images.

---

## 🧪 Technologies Used

- Python
- OpenCV
- NumPy
- TensorFlow / Keras
- Matplotlib + Seaborn (for debugging/visuals)
- Gradio (UI)
- Hugging Face Spaces (deployment)

---

🧠 Model Info
The digit recognition model is a CNN trained on handwritten digits (28x28 grayscale), inspired by MNIST. You can retrain or fine-tune it for better accuracy on Sudoku-style digits.

🌍 Live Demo
🧪 Try it live on Hugging Face:
https://huggingface.co/spaces/mehrta/ai-sudoku-solver

