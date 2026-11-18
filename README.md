# Cat vs. Dog Image Classifier

A deep-learning project that classifies an input image as a **cat** or a **dog** using a Convolutional Neural Network (CNN), TensorFlow/Keras, and Streamlit.

<p align="center">
  <a href="https://catvsdogs.streamlit.app/"><strong>Try the Live Demo</strong></a>
  ·
  <a href="https://github.com/lovnishverma/catvsdogs/blob/main/dog-vs-cat-cnn-classifier.ipynb">View the Training Notebook</a>
</p>

---

## Overview

This project demonstrates an end-to-end image-classification workflow: training a CNN, saving the trained model, and exposing predictions through a simple web interface.

Upload an image through the Streamlit app to see the model's predicted class. Predictions depend on the image quality and the model's training; they should not be treated as guaranteed classifications.

## Features

- **Image classification:** Predicts whether an uploaded image contains a cat or a dog.
- **Trained Keras model:** Uses the saved `dogs_vs_cats_model.keras` model.
- **Interactive web app:** Provides a browser-based interface built with Streamlit.
- **Reproducible learning workflow:** Includes a Jupyter notebook for exploring the CNN workflow.
- **Hosted demo:** Try the deployed application without setting up the project locally.

## Live Demo

**[Launch Cat vs. Dog Classifier](https://catvsdogs.streamlit.app/)**

## Tech Stack

| Technology | Purpose |
| --- | --- |
| Python | Application and model code |
| TensorFlow / Keras | Deep-learning model and inference |
| Streamlit | Interactive web interface |
| NumPy and image-processing dependencies | Data and image preparation, as used by the app |
| Jupyter Notebook | Model experimentation and training workflow |

## Repository Structure

```text
catvsdogs/
├── app.py                          # Streamlit application
├── dog-vs-cat-cnn-classifier.ipynb # CNN notebook
├── dogs_vs_cats_model.keras        # Saved trained model
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/lovnishverma/catvsdogs.git
cd catvsdogs
```

### 2. Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually **http://localhost:8501**. Open that URL in your browser.

> **Note:** The saved model file is included in the repository. Ensure it is present in the expected path used by `app.py` before starting the app.

## Model and Training

The project includes a CNN notebook and a saved Keras model:

- **Training / exploration:** [`dog-vs-cat-cnn-classifier.ipynb`](dog-vs-cat-cnn-classifier.ipynb)
- **Saved model:** [`dogs_vs_cats_model.keras`](dogs_vs_cats_model.keras)

The notebook is the place to inspect the training pipeline, data preparation, model architecture, and evaluation. For reliable results, evaluate the model on held-out data and report metrics such as accuracy, precision, recall, and a confusion matrix. No numerical performance claims are made here because verified evaluation results are not documented in this README.

## Troubleshooting

**The app cannot find the model**

Confirm that `dogs_vs_cats_model.keras` exists in the path referenced by `app.py`. Run the application from the repository directory unless the code specifies another path.

**A Python package is missing**

Activate the virtual environment and reinstall dependencies:

```bash
python -m pip install -r requirements.txt
```

**The app fails to start**

Check the Python and TensorFlow versions supported by the dependencies in `requirements.txt`, then read the complete traceback printed in the terminal.

## Future Improvements

- Display prediction confidence with a clear explanation that confidence is not a guarantee.
- Add a curated set of sample images and provide clearer upload validation.
- Document the dataset, preprocessing steps, CNN architecture, and training configuration.
- Publish verified evaluation metrics and a confusion matrix.
- Add automated tests for image preprocessing and prediction handling.

## Contributing

Suggestions, bug reports, and improvements are welcome. Open an issue to describe a problem or propose an enhancement before submitting a pull request.

## License

No license is currently specified in this repository. Unless a license is added, reuse and redistribution are subject to the applicable default copyright rules. Add a `LICENSE` file if you intend to grant others explicit permission to use, modify, or distribute the project.

---

**Author:** [Lovnish Verma](https://github.com/lovnishverma)

**Project:** [github.com/lovnishverma/catvsdogs](https://github.com/lovnishverma/catvsdogs)
