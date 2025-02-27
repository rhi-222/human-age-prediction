# Human Age Prediction with Deep Learning

## Overview
This project utilizes **Convolutional Neural Networks (CNNs)** to predict human age from facial images. The model was developed as part of a data science bootcamp project, showcasing expertise in **computer vision, deep learning, and model optimization**.

## Technologies Used
- **Python** (TensorFlow, Keras, OpenCV, Pandas, NumPy)
- **Deep Learning** (CNNs, Transfer Learning)
- **Data Processing** (Data Augmentation, Image Normalization)
- **GPU Acceleration** (Google Colab)

## Key Achievements
- **Trained a CNN model on a large facial image dataset**, applying **transfer learning** and **data augmentation** to improve accuracy.  
- **Achieved a Mean Absolute Error (MAE) of 7.1 years**, demonstrating strong predictive performance.  
- **Utilized Google Colab's GPU acceleration**, reducing training time and enabling efficient model experimentation.  
- **Explored additional improvements**, such as EfficientNet and fine-tuning pretrained weights, to optimize accuracy.  

## How to Use
1. **Clone the repository**:
   ```bash
   git clone https://github.com/rhi-222/human-age-prediction.git
2. Install dependencies:
    ```bash
   pip install tensorflow keras opencv-python numpy pandas matplotlib

4. Run the Jupyter Notebook:
- Open `human_age_prediction.ipynb` in Jupyter Notebook or Google Colab.
- Execute each cell to preprocess images, train the CNN, and evaluate model performance.

Results & Next Steps
- The model is effective for broad age categorization (e.g., child, adult, senior) but may require improvements for precise age estimation.
- Future enhancements include fine-tuning on a larger dataset, experimenting with EfficientNet, and refining hyperparameters.
