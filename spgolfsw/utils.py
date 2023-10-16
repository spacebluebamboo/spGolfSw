import torch.nn as nn
import cv2
import tempfile


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


# @st.cache(allow_output_mutation=True,suppress_st_warning=True)
def createImages(fila, nomS, events):
    """
    Given a video file location (fila) it will save as images to a folder
    Given positions in video (pos) these images from the video are saved
    pos is created based on positions of swings
    """
    fila
    tfile2 = tempfile.NamedTemporaryFile(delete=False)
    tfile2.write(fila.read())
    cap = cv2.VideoCapture(tfile2.name)

    # eventNom = [0, 1, 2, 3, 4, 5, 6, 7]
    imgALL = []
    fimg = []
    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        imgALL.append(img)
        fimg.append(e)
    #  print( np.shape(img) )
    #  cv2.imwrite(os.path.join(os.getcwd(),'_'+ nomS+'_'+"frame{:d}.jpg".format(eventNom[i])), img)     # save frame as JPG file

    cap.release()
    return imgALL, fimg
