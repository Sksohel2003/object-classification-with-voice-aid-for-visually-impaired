# Object Classification with Voice Aid for Visually Impaired

This Android application is designed to assist visually impaired individuals by providing real-time object detection with distance estimation and voice output. It uses the device's camera to automatically detect nearby objects and announces their names and approximate distances without any user interaction.

## 🧠 Key Features

- 📷 **Real-time Object Detection** using TensorFlow Lite
- 🔊 **Voice Output** for identified objects
- 📏 **Distance Estimation** to inform users how far the object is
- 🎙️ **Voice Commands**: Say "extract text" to switch to text extraction mode, and "stop extracting text" to return to object detection
- 🚫 **No Tap Required**: App automatically starts detection on launch

## 🛠️ Tech Stack

- **Language:** Java
- **Framework:** Android SDK
- **ML Framework:** TensorFlow Lite (TFLite)
- **Voice Output:** Android Text-to-Speech (TTS)
- **Custom Object Labels:** Based on COCO dataset with additional custom class handling
- **Distance Calculation:** Based on camera's field of view and object bounding box

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Sksohel2003/object-classification-with-voice-aid-for-visually-impaired.git
Open in Android Studio.

Build and run the app on a physical Android device with camera and microphone access.

⚠️ Note: TFLite model file and labels must be placed in the assets folder.

![image](https://github.com/user-attachments/assets/786e64b9-05aa-40c4-88d3-6ba99a6d53d0)



