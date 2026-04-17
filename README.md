# 🎥 Intelligent Video Surveillance System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> An advanced deep learning-based video surveillance system that automatically detects fights, theft, and violence from video footage in real-time.

<p align="center">
  <img src="docs/assets/demo.gif" alt="Demo Preview" width="600"/>
</p>

## 🌟 Features

- 🔍 *Real-time Detection* of:
  - Fights and Physical Violence
  - Theft Activities
  - Suspicious Behavior
- 🎯 *High Accuracy* using state-of-the-art deep learning models
- 🖥 *User-Friendly Interface* built with Streamlit
- 📊 *Analytics Dashboard* for incident tracking
- ⚡ *Fast Processing* with optimized inference
- 📱 *Alert System* for immediate notification
- 🎬 *Video Management* tools for clip extraction

## 🛠 Tech Stack

- *Deep Learning*: TensorFlow, OpenCV
- *Frontend*: Streamlit
- *Data Processing*: NumPy, Pandas
- *Visualization*: Matplotlib, Seaborn
- *Development*: Python 3.8+

## 📦 Installation

1. Clone the repository:
bash
git clone https://github.com/yourusername/video-surveillance-system.git
cd video-surveillance-system


2. Create and activate virtual environment:
bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows


3. Install dependencies:
bash
pip install -r requirements.txt


## 🚀 Quick Start

1. Start the Streamlit app:
bash
streamlit run src/app.py


2. Open your browser and navigate to http://localhost:8501

3. Upload a video file and start detection!

## 🎓 Model Training

1. Download the datasets:
   - [RWF-2000 Dataset](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)
   - [Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)

2. Prepare the data:
bash
python src/data/prepare_dataset.py


3. Train the model:
bash
python src/models/train.py


## 📁 Project Structure


video-surveillance-system/
├── src/
│   ├── app.py                # Streamlit application
│   ├── config.py             # Configuration settings
│   ├── data/                 # Data processing scripts
│   ├── models/               # Model architecture and training
│   └── utils/                # Helper functions
├── data/                     # Dataset storage
│   ├── raw/                  # Original dataset
│   └── processed/            # Processed frames
├── models/                   # Saved model weights
├── logs/                     # Training logs
├── tests/                    # Unit tests
└── docs/                     # Documentation


## 📊 Results

| Model         | Accuracy | F1 Score | Processing Speed |
|--------------|----------|-----------|-----------------|
| CNN + LSTM   | 92.5%    | 0.91     | 30 FPS         |
| 3D CNN       | 94.2%    | 0.93     | 25 FPS         |

## 🔧 Configuration

Modify config.py to adjust:
- Model parameters
- Detection thresholds
- Alert settings
- Video processing options

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit changes (git commit -m 'Add AmazingFeature')
4. Push to branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Rajeev Bendalam - [GitHub](https://github.com/BendalamRajeev) - [LinkedIn](https://www.linkedin.com/in/bendalam-rajeev-392170274/)

## 🙏 Acknowledgments

- RWF-2000 Dataset creators
- Real Life Violence Situations Dataset contributors
- Open-source community
