import cv2
import numpy as np
import os
import sys


# Ensure we can import from src and models by adding project root to path
# This handles running the script from src/ or from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.svm_model import SVMModel
from src.feature_extraction import extract_features_from_frame, extract_feature_frame_single
def classify_frame_svm(frame, model ,classes  ):
    if frame is None: 
        return False
    display = cv2.resize(frame, (640, 480))
    try: 
        feature_vector = extract_features_from_frame(display)
        if feature_vector is None:
           label = 'no features found'
        else:
            label = classes[model.predict(feature_vector.reshape(1, -1))[0]]

    except Exception as e:
        print(f"error classifying frame: {e}")
        label = 'not classified'
    
    # Display the label on the frame
    cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Real-time Classification", display)

    return label
def path_or_frame_choice():
    print("=== Choose Mode ===")
    print("1. Live Camera")
    print("2. Image File")
    print("3. Stop")

    choice = input("enter your choice (1 or 2): ").strip()
    return choice 
def main():
    print
    # intialize the model
    svm = SVMModel()
    model_path = os.path.join(project_root, "models", "svm_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"error: Model file not found at {model_path}")
        print("please train the model first using 'src/train_svm.py' or ensure the file exists.")
        return
    else :
        print(f"model file found at {model_path}")
        print("loading model...")

    try:
        svm.load(model_path)
    except Exception as e:
        print(f"error loading model: {e}")
        return

   
    classes = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
    choise = path_or_frame_choice()

    while choise in ("1","2"):
        if choise == "1":
            # starting the camera 
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera.")
                return
            else :
                print("camera opened successfully.")

            print("starting real-time classification. press 'q' to exit.")
        #take input from user (real time loop)
            try: 
                while True:
                    #1. capture the frame 
                    ref, frame = cap.read() 
                    if not ref or frame is None:
                        print("failed to capture the frame , please reopen the camera")
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'): #very important to check keyboard settings 
                        break
                    (h, w, _) = frame.shape
                    text = 'press q to quit'
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    x = w - text_width - 10
                    y = h - text_height - 10
                    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    if not classify_frame_svm(frame, svm ,classes  ):
                        break
                    
            finally: 
                cap.release()
                cv2.destroyAllWindows()
        elif choise == "2":
            
            image_path = input("enter the path of the image: ")
            if not os.path.exists(image_path):
                print(f"Error: File not found at {image_path}")
                return
            image = cv2.imread(image_path)

            label = classify_frame_svm(image, svm ,classes  )
            print(f"predicted label: {label}")
        
        choise = path_or_frame_choice()

        




if __name__ == "__main__":
    main()
