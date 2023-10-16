import scipy.io
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# from eval import ToTensor, Normalize
# from model import EventDetector
import numpy as np
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
import copy
import sys

from classes import SampleVideo, ToTensor, Normalize, EventDetector
from utils import createImages

# ""https://github.com/tonylins/pytorch-mobilenet-v2""


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def loadStuff(uploaded_files, uploaded_filesCOPY):
    seq_length = 25
    #     input_size=120

    ds = SampleVideo(
        uploaded_files,
        #                      input_size=input_size,
        transform=transforms.Compose(
            [ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        ),
    )

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False,
    )
    try:
        save_dict = torch.load(
            "spgolfsw/models/swingnet_1800.pth.tar", map_location=torch.device("cpu")
        )
    except ValueError as e:
        print(
            "Model weights not found. "
            "Download model weights and place in 'models'"
            f" folder. See README for instructions {e}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.load_state_dict(save_dict["model_state_dict"])
    model.to(device)
    model.eval()
    print("Loaded model weights")

    print("Testing...")
    for sample in dl:
        images = sample["images"]
        # full samples do not fit into GPU memory
        # so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length :, :, :, :]
            else:
                image_batch = images[
                    :, batch * seq_length : (batch + 1) * seq_length, :, :, :
                ]
            logits = model(image_batch)
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    print("Predicted event frames: {}".format(events))

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print("Confidence: {}".format([np.round(c, 3) for c in confidence]))

    imgALL, fimg = createImages(uploaded_filesCOPY, "10", events)
    #     stra, events,imgALL, nom=[],[],[],[]
    return events, imgALL, fimg


def main():
    st.title("Golf Swing")
    # loada = st.checkbox('Load',key='AC')
    XxX = sorted(
        [(x, sys.getsizeof(globals().get(x))) for x in dir()],
        key=lambda x: x[1],
        reverse=True,
    )
    memos = np.array([(sys.getsizeof(globals().get(x))) for x in dir()])

    print("Main", XxX, "---", np.shape(XxX))
    st.write("Memory", np.sum(memos))

    # if loada:
    uploaded_files = st.sidebar.file_uploader(
        "Choose video", accept_multiple_files=False
    )

    uploaded_filesCOPY = copy.copy(uploaded_files)

    if uploaded_files:
        events, imgALL, fimg = loadStuff(uploaded_files, uploaded_filesCOPY)

    # del uploaded_files

    plota = st.checkbox("Plot", key="AC2")

    if plota:
        print(
            "PlotBox",
            sorted(
                [(x, sys.getsizeof(globals().get(x))) for x in dir()],
                key=lambda x: x[1],
                reverse=True,
            ),
        )

        imgSEL = st.selectbox("Select Image", fimg)

        numSEL = [oo for oo, x in enumerate(fimg) if x == imgSEL][0]

        f = plt.figure(figsize=(6, 6))

        plt.imshow(imgALL[numSEL])

        st.pyplot(f)


if __name__ == "__main__":
    main()
