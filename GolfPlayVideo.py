import streamlit as st
import tempfile
import copy
import cv2
import numpy as np

import matplotlib.pyplot as plt

f = st.sidebar.file_uploader("Choose video file", accept_multiple_files=False)
# @st.cache(allow_output_mutation=True)

tfile = tempfile.NamedTemporaryFile(delete=False) 
tfile.write(f.read())

cap = cv2.VideoCapture(tfile.name)
success, image = cap.read()
imgALL=[]
ret, frame = cap.read()
frame_no=cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps=cap.get(cv2.CAP_PROP_FPS)
duration = frame_no/fps




video_file = open(tfile.name,'rb')
video_bytes = video_file.read()

time = st.sidebar.slider('video time',0,20,1)

aaa=st.video(video_file,start_time=int(time))


a1=7.
b1=68.

a2=0.
b2=100.


# change zero point
def adj0(a1,a2,b1,b2):
    a2=a2-a1
    b2=b2-a1
    b1=b1-a1
    a1=a1-a1
    return a1,a2,b1,b2

def scala(a1,a2,b1,b2,scc):
    sc = scc/b1
    a2 = a2*sc
    a1 = a1*sc
    b2 = b2*sc
    b1 = b1*sc
    return a1,a2,b1,b2
    
a1,a2,b1,b2=adj0(a1,a2,b1,b2)
scc=frame_no
a1,a2,b1,b2=scala(a1,a2,b1,b2,scc)


time2 = time = st.slider(
     '',
     a2, b2, (a1,b1))

time2[0],time2[1]

        

frame_no, fps, duration

# imgALL=[]
# ii=0
# while success:
#     if ii>time2[0]:
#         success, image = cap.read()
#         imgALL.append(image)
#     elif ii>time2[1]:
#         break
#     ii
#     ii=ii+1
    
# len(imgALL)
    
# start_time = st.sidebar.slider('start_time time',0,len(imgALL),1)

# st.sidebar.header('Start time = {}'.format(start_time))

# st.header(np.shape(imgALL))

# f=plt.figure(figsize=(6,6))
# plt.imshow(imgALL[start_time])

# st.pyplot(f)
# start_time = st.sidebar.slider('start_time time',0,20,1)




