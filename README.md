# DigReco

## 🚀 Project Overview
**DigReco** (short for *Digits Recognition*) is an AI-powered application designed to recognize handwritten digits. It uses **TensorFlow** for model development and **Flask** for deployment, the project provides a simple, interactive interface to predict digits written by users.

## 🎯 Features
- **AI Model**: Built with a Convolutional Neural Network (CNN) using TensorFlow/Keras.
- **Interactive UI**: Allows users to upload an image for recognition.
- **Real-Time Prediction**: Instant feedback on digit recognition.
- **Scalable Backend**: Flask powers the server for easy deployment and API integration.
- **Customizability**: Ready for extensions to support additional datasets or recognition tasks.

## 🛠️ Tech Stack
- **Backend**: Flask (Python)
- **AI Framework**: TensorFlow/Keras
- **Frontend**: HTML/CSS/JavaScript
- **Deployment**: Flask Server

---

## 📂 Project Structure

```
DigReco/
│
├───FrontEnd/                    # Contains all frontend-related files
│   │
│   ├── index.html               # Main HTML file that provides the user interface
│   ├── scripts.js               # JavaScript for handling user interactions and API calls
│   └── styles.css               # CSS file for styling the UI
│
├───ModelServing/                # Folder for model deployment and serving
│   │
│   ├── backend.py               # Flask-based backend to handle API requests and serve predictions
│   └── tuned_handwritten_digit_model.h5  # Pre-trained model for handwritten digit recognition
│
├───ModelTraining/               # Folder for training the model
│   │
│   └── training.py              # Script for training the digit recognition model using TensorFlow/Keras
│
└───TestingData/                 # Sample testing images for validating the model
    │
    ├── 0.PNG                    # Image of digit '0'
    ├── 2.PNG                    # Image of digit '2'
    ├── 3.PNG                    # Image of digit '3'
    ├── 5-2.PNG                  # Image of mixed digits '5' and '2'
    ├── 6.PNG                    # Image of digit '6'
    ├── 7.PNG                    # Image of digit '7'
    └── 8.PNG                    # Image of digit '8'

```

---

## 🔧 Setup and Installation

### Prerequisites
- Python 3.8 or higher
- TensorFlow/Keras
- Flask

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prady00/digreco.git
   cd digreco
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Training the model** (optional, I've added a pretrained model):
   ```bash
   cd ModelTraining
   python training.py
   ```
    It will produce a model tuned_handwritten_digit_model.h5 
    You can move tuned_handwritten_digit_model.h5 to ModelServing folder for deployment or serving

5. **Start the backend**:
    ```bash
    cd ModelServing
    python backend.py
    ```
    Keep the terminal running. It exposes the model through Flask on port 5000
   
6. **Start the frontend**:
    ```bash
    cd Frontend
    python -m http.server
    ```

7. **Access the frontend app**:
   Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🧠 Model Details
- The model is a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.
- **Architecture**:
   - Input Layer: 28x28 grayscale image
   - 2 Convolutional Layers + MaxPooling
   - Flatten Layer
   - Fully Connected Dense Layers
   - Activation: ReLU, Softmax
- **Output**: Predicted digit (0-9) with confidence score.

---

## 🖥️ Usage

### From the UI
- Upload digit on the provided UI and click "Predict".
- Alternatively, upload a `.png` or `.jpg` file containing a handwritten digit.

### API Integration
Use the Flask API to send images programmatically.

**Endpoint**:
```
POST /predict
```

**Request**:
Send an image as form data under the key `file`.

**Example using cURL**:
```bash
curl -X POST -F "file=@digit.png" http://localhost:5000/predict
```

**Response**:
```json
{
   "digit": 5,
   "confidence": 0.97
}
```

---

## 🚀 Future Enhancements
- Support for multi-digit recognition.
- Integration with other digit datasets (e.g., EMNIST).
- Deploy to cloud platforms like AWS, Azure, or Heroku.
- Mobile-friendly UI for seamless usage.

---

## 👨‍💻 Contributing
We welcome contributions! Feel free to:
1. Fork the repository.
2. Create a new feature branch.
3. Submit a pull request with improvements or fixes.

---

## 📜 License
This project is licensed under the MIT License.

---

## 🤝 Contact
For queries or suggestions, reach out at:
- **Email**: pradeep.online00@gmail.com
- **LinkedIn**: [https://www.linkedin.com/in/prady00/] (https://www.linkedin.com/in/prady00/)
