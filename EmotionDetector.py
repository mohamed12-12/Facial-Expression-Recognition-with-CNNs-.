import cv2 
import numpy as np
from keras.models import load_model

model = load_model('Emotions8.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0) 

while(True): 

    # Capture the video frame 
    # by frame 
    if vid.isOpened():
        ret, frame = vid.read() 
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                img = cv2.resize(roi_color,(48,48))
                img = np.array(img)
                img = img / 255.0 # normalize the image
                img = img.reshape(1, 48, 48, 3) # reshape for prediction
                preds = model.predict(img)
                print(type(preds))
                print(preds)
                preds = preds.tolist()[0]
                
                pred = preds.index(max(preds))
                print(pred)
                if pred == 0:
                    label = 'Happy'
                elif pred == 1:
                    label = 'Neutral'
                else:
                    label = 'Sad'
                font = cv2.FONT_HERSHEY_SIMPLEX 
                cv2.putText(frame,  
                            f'{label}',  
                            (x, y-10),  
                            font, 1,  
                            (0, 255, 255),  
                            2,  
                            cv2.LINE_4)
            
            # Display the resulting frame 
            cv2.imshow('frame', frame) 
            
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()   



