import os
from datetime import datetime as dt
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pyrebase
import streamlit as st
import numpy as np
from deepface import DeepFace
import json

localDb = 'reports.txt'
userDb = 'users.txt'


def add_user(uid, _name, _roll):
    try:
        df_ = pd.read_csv(userDb)
    except FileNotFoundError:
        df_ = pd.DataFrame(columns=['uid', 'name', 'roll'])
    new_data = pd.DataFrame({'uid': [uid], 'name': [_name], 'roll': [_roll]})
    df_ = pd.concat([df_, new_data], ignore_index=True)
    df_.to_csv(userDb, index=False)


def add_report(box_no, id_no):
    try:
        df_ = pd.read_csv(localDb)
    except FileNotFoundError:
        df_ = pd.DataFrame(columns=['Box No.', 'Id'])
    new_data = pd.DataFrame({'Box No.': [box_no], 'Id': [id_no]})
    df_ = pd.concat([df_, new_data], ignore_index=True)
    df_.to_csv(localDb, index=False)


def retrieve_table(loc):
    try:
        df_ = pd.read_csv(loc + '.txt')
        return df_
    except FileNotFoundError:
        return None


def create_dir(path_name):
    path = path_name
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    return path_name + '/'


def get_(uid):
    df_ = retrieve_table('users')
    if df_ is not None:
        print(uid)
        print(df_.loc[df_['uid'] == uid, 'name'].values)
        _name, _roll = df_.loc[df_['uid'] == uid, 'name'].values[0], df_.loc[df_['uid'] == uid, 'roll'].values[0]
        print('-----------------', _name, _roll)
        return [_name, _roll]
    return ['sss', 'sss']


def put_(directory, label, item):
    loc = create_dir(directory) + label
    cv2.imwrite(loc, item)
    storage.child(loc).put(loc)


def process_individual_image(image):
    col1, col2 = st.columns(2)

    # Displaying Image
    col1.image(image, caption="Uploaded Image")
    placeInd = st.empty()
    placeInd.write('Processing..')

    # Reading Image
    currPath = plt.imread(image)
    currImg = currPath.copy()

    # Process Image
    currFace = DeepFace.extract_faces(currImg, target_size=(224, 224), detector_backend='retinaface',
                                      enforce_detection=False)
    r1 = currFace[0]['facial_area']
    x, y, w, h = r1['x'], r1['y'], r1['w'], r1['h']
    currFace = currImg[y:y + h, x:x + w]

    # Saving Processed Image
    put_('individuals', image.name, currFace)
    place.empty()
    col2.image(currFace, caption="Processed Image")
    st.success("Image processed and uploaded successfully.")


def detect_faces(image, dim=None, recognized_=None, special=-1):
    # Reading Group Image
    img_array = plt.imread(image)
    img_rgb = img_array.copy()

    # If dimensions not provided, detect faces and get them
    if dim is None:
        faces = DeepFace.extract_faces(img_rgb, target_size=(224, 224), detector_backend='retinaface')
        dim = [face['facial_area'] for face in faces]

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)

    i, n, my_bar = 0, len(dim), None

    # If image not yet recognized, show progress bar for extraction
    if recognized_ is None: my_bar = st.progress(0, text='Extracting Faces')

    for r in dim:
        x, y, w, h = r['x'], r['y'], r['w'], r['h']
        color = 'red' if special == i else 'blue'  # To handle better interaction

        # Drawing Rectangle
        rect = plt.Rectangle((r['x'], r['y']), r['w'], r['h'], fill=False, color=color, linewidth=3)
        ax.add_patch(rect)

        # Adding Box No.
        ax.text(x, y - 2, str(i + 1), color='g', fontsize=10, weight='bold')

        # Adding Recognized Face id (if recognized)
        if recognized_ is not None:
            ax.text(x + w - 10, y - 2, str(recognized_[i]), color='b', fontsize=10, weight='bold')

        i += 1

        # If faces not yet recognized save them for extraction
        if recognized_ is None:
            my_bar.progress(i / n, text='Extracting ' + str(i) + '/' + str(n))
            currFace = img_rgb[y:y + h, x:x + w]
            put_('extracted', str(i) + '.jpg', currFace)

    ax.axis('off')
    plt.tight_layout()

    # Saving Image
    processedDir = create_dir('recognized') if recognized_ is not None else create_dir('processed')
    path = processedDir + dt.now().strftime("%d%m%Y") + ".jpg"
    fig.savefig(path)
    storage.child(path).put(path)
    if recognized_ is None:
        st.success("Faces Extracted")
    st.experimental_rerun()

    # Saving dimensions
    with open('dim.json', 'w') as file_:
        json.dump(dim, file_)
        print('------ File Saved ------')
    return dim


def face_recognition():
    st.write("Downloading Faces...")
    files_ = storage.list_files()
    local_ind = create_dir('individuals')
    local_ext = create_dir('extracted')
    n = 0
    for file_ in files_:
        if file_.name[:12] == 'individuals/':
            filename = os.path.basename(file_.name)
            storage.child(local_ind).download(file_.name, os.path.join(local_ind, filename))
            n += 1
    for file_ in files_:
        if file_.name[:10] == 'extracted/':
            filename = os.path.basename(file_.name)
            storage.child(local_ext).download(file_.name, os.path.join(local_ext, filename))

    st.success("All individual faces downloaded")

    recArr, my_bar = [], None
    my_bar = st.progress(0, text='Recognizing Faces')
    for i in range(n):
        recon = DeepFace.find(local_ext + str(i + 1) + '.jpg', local_ind, enforce_detection=False)[0]
        if len(recon):
            recArr.append(recon.iloc[0]['identity'])
        else:
            recArr.append('~-1.')
        my_bar.progress((i + 1) / n, text='Identifying ' + str(i + 1) + '/' + str(n))

    # Getting Accuracy
    count, faces, boxes = 0, [], [i for i in range(1, 23)]
    for i in range(n):
        predicted = int(recArr[i][recArr[i].index('~') + 1:recArr[i].index('.')])  # Getting predicted value
        faces.append(predicted)                                           # Saving predicted value in faces
        count += 1 if predicted == i + 1 else 0                           # Increasing count if prediction is correct

    # Creating and saving dataframe
    df1 = pd.DataFrame({'Box No.': boxes, 'Face': faces})
    df1.to_csv('recognition.txt', index=False)
    styled_df1 = df1.style.apply(color_row, axis=1)
    st.write(styled_df1)
    st.success("Face Recognition Successful with " + str(count * 100 / n) + "% Accuracy")
    return faces


# Function to apply conditional formatting to check for correct/incorrect results
def color_row(row):
    if row['Box No.'] != row['Face']:
        return ['color: red'] * len(row)
    else:
        return ['color: green'] * len(row)


# Connecting to firebase
firebaseConfig = {
    'apiKey': "AIzaSyCrs6d7AGI_JFJ5ART8infZL-8ZP0WImII",
    'authDomain': "minor-cec64.firebaseapp.com",
    'databaseURL': "https://minor-cec64-default-rtdb.firebaseio.com",
    'projectId': "minor-cec64",
    'storageBucket': "minor-cec64.appspot.com",
    'messagingSenderId': "621717444649",
    'appId': "1:621717444649:web:e3c6b12db5d730b1906438",
    'measurementId': "G-WBX0VQ3W7C",
    'serviceAccount': 'minor-cec64-firebase-adminsdk-82yn2-f717cdeedb.json'
}

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()
auth = firebase.auth()
storage = firebase.storage()

home, image1, individual, report = st.tabs(['Home', 'Group', 'Individual', 'Report'])
with home:
    try:
        if 'status' in st.session_state.keys():
            name, roll = get_(auth.current_user['idToken'])
            st.header("Hello " + name)
        else:
            raise Exception("Hi")
    except:
        login, signup = st.tabs(['Login', 'SignUp'])
        with login:
            place = st.empty()
            with place.form(key='login'):
                email = st.text_input("Email", key='login')
                password = st.text_input("Password", type="password", key='login1')
                loginBtn = st.form_submit_button("Login")
            if loginBtn and email != '':
                user = auth.sign_in_with_email_and_password(email, password)
                print(user['localId'])
                st.session_state['status'] = True
                arr = get_(user['localId'])
                place.empty()
                st.header("Hello ", arr[0])
        with signup:
            place = st.empty()
            with place.form(key='signup'):
                co1, co2 = st.columns(2)
                name = co1.text_input("Name", key='signup3')
                roll = co2.text_input("Roll")
                email = st.text_input("Email", key='signup')
                password = st.text_input("Password", type="password", key='signup1')
                verify_password = st.text_input("Verify Password", type="password", key='signup2')
                signUpBtn = st.form_submit_button("Sign Up")
            if signUpBtn and email != '':
                try:
                    user = auth.create_user_with_email_and_password(email, password)
                    print(user)
                    print(user['localId'])
                    add_user(user['localId'], name, roll)
                    st.session_state['status'] = True
                    arr = get_(user['localId'])
                    place.empty()
                    st.header("Hello " + arr[0])
                except:
                    st.error("Invalid Credentials")

with image1:
    st.header("Today's Status")

    process, upload = st.tabs(['Process', 'Upload'])

    recognize = retrieve_table('recognition')
    with process:

        # Refresh Option
        if st.button('Refresh'):
            st.experimental_rerun()

        # Checking if today's attendance is already uploaded, processed or recognized
        status, current = '', ''
        group, recognized, processed = create_dir('group'), create_dir('recognized'), create_dir('processed')
        nowName = dt.now().strftime("%d%m%Y") + ".jpg"
        files = storage.list_files()
        for file in files:
            if file.name == group + nowName and status == '':
                status = 'Uploaded Image'
                current = file.name
            if file.name == processed + nowName and status != 'Recognized Image':
                status = 'Processed Image'
                current = file.name
            if file.name == recognized + nowName:
                status = 'Recognized Image'
                current = recognized + nowName

        if status == 'Uploaded Image':
            st.header("Image Uploaded")

            # Downloading and displaying image
            fileUpload = os.path.basename(current)
            storage.child(group).download(current, os.path.join(group, fileUpload))
            st.image(current, caption=status)

            # Processing Image
            st.write('Detecting Faces')
            dimFaces = detect_faces(current)  # Detecting and Extracting Faces from group image
            status = 'Processed Image'
        elif status == 'Processed Image':
            st.header("Image Processed")

            # Downloading and displaying image
            fileUpload = os.path.basename(current)
            storage.child(processed).download(current, os.path.join(processed, fileUpload))
            st.image(current, caption=status)

            # Recognizing Faces
            st.write('Recognizing Faces')
            recognize = face_recognition()  # Recognizing Extracted Faces
            status = 'Recognized Image'
            detect_faces(group+nowName, None, recognize)  # Saving recognized group image

        elif status == 'Recognized Image':
            st.header("Image Recognized")

            # Downloading and displaying image
            fileUpload = os.path.basename(current)
            storage.child(recognized).download(current, os.path.join(recognized, fileUpload))
            st.image(current, caption=status)

            # Button to get results
            if st.button('Get Tables'):
                if recognize is not None:
                    styled_df = recognize.style.apply(color_row, axis=1)
                    reports = retrieve_table('reports')
                    col1_, col2_ = st.columns(2)
                    col1_.write("Recognitions")
                    col1_.write(styled_df)
                    col2_.write("Reports")
                    if reports is not None:
                        col2_.write(reports)
                    else:
                        st.write("No Reports Yet")
                else:
                    face_recognition()
        else:  # No Image Uploaded Yet
            st.error("Upload an Image to Process")

    with upload:
        st.header("Upload Today's Group Image")

        # Checking status to show warnings
        files, nowName, status = storage.list_files(), dt.now().strftime("%d%m%Y") + ".jpg", ''
        for file in files:
            if file.name == 'group/' + nowName and status == '': status = 'Uploaded'
            if file.name == 'recognized/' + nowName: status = 'Recognized'
        if status == 'Uploaded':
            st.error("Today's image already uploaded! Re-uploading will replace the previous image")
        if status == 'Recognized':
            st.error("Today's image already processed! All the progress will be deleted")

        # Uploading Image
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key='2')
        submit = st.button('Submit')
        if submit:
            if uploaded_file is not None:

                # Deleting previous uploaded/processed image
                group = create_dir('group')
                file_path = 'group/' + nowName
                user = auth.sign_in_with_email_and_password('shashwatmishra156@gmail.com', '1234@As')
                token = user['idToken']
                if status:
                    storage.delete(file_path, token=token)
                if status == 'Recognized':
                    storage.delete('processed/' + nowName, token=token)
                    storage.delete('recognized/' + nowName, token=token)

                # Saving Uploaded Image locally
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                # Uploading on Firebase
                storage.child(file_path).put(file_path)
                st.success('Image Upload Successful')

                # Displaying
                st.image(uploaded_file, caption='Uploaded Image')
            else:
                st.error("Please Select a file!")

with individual:
    st.header("Upload your recent image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key='3')
    if uploaded_file is not None:
        # Detect faces and draw bounding boxes
        process_individual_image(uploaded_file)

with report:
    myId = st.selectbox('Select your Id: ', [i for i in range(1, 23)])

    recognized = create_dir('recognized')
    nowName = dt.now().strftime("%d%m%Y") + ".jpg"
    files, status, current = storage.list_files(), False, ''
    for file in files:
        if file.name == recognized + nowName:
            status = True
            current = file.name

    if status:  # Today's Image Processed

        # Downloading and displaying processed image
        fileUpload = os.path.basename(current)
        storage.child(recognized).download(current, os.path.join(recognized, fileUpload))
        st.image(current, caption="Today's Image")

        recognitions = retrieve_table('recognition')
        if recognitions is None:
            st.error("Error in finding recognitions")
        else:
            col1_, col2_ = st.columns(2)
            try:
                # Displaying box image
                extracted = create_dir('extracted')
                box = str(recognitions.loc[recognitions['Face'] == myId, 'Box No.'].values[0])
                fileName = extracted + box + '.jpg'
                fileUpload = os.path.basename(fileName)
                storage.child(extracted).download(fileName, os.path.join(extracted, fileUpload))
                col1_.image(fileName, caption="You were identified as "+box)
            except:
                col1_.error("You were not identified")
            # Displaying your image
            individual = create_dir('individuals')
            fileName = individual + '~' + str(myId) + '.jpg'
            fileUpload = os.path.basename(fileName)
            storage.child(individual).download(fileName, os.path.join(individual, fileUpload))
            col2_.image(fileName, caption="You are")

        st.header("Report Mis-recognition")
        with st.form('Report', clear_on_submit=True):
            box = st.text_input("Box No.", key='report')
            id_ = st.text_input("Your Id", key='report1')
            btnSubmit = st.form_submit_button("Submit")
        if btnSubmit:
            add_report(box, id_)
            st.success('Report Added Successfully!')

    else:
        st.error("Today's Attendance yet to be processed")
