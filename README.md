# pyaam - active appearance model

active appearance models implemented in python

## Instructions

Download MUCT dataset:

    python -m pyaam.muct

View MUCT dataset:

    ./view_data.py

Train models:

    ./train_model.py shape
    ./train_model.py patches
    ./train_model.py detector
    ./train_model.py texture
    ./train_model.py combined

View models:

    ./view_model.py shape
    ./view_model.py patches
    ./view_model.py texture
    ./view_model.py combined

View face detector on webcam:

    ./view_face.py detector

View face tracker (patches):

    ./view_face.py tracker

Face tracker using AAMs coming soon!

## References

- J. Saragih, "Non-rigid Face Tracking". In Mastering OpenCV with Practical Computer Vision Projects. PACKT, Oct 2012.
- M.B. Stegmann, "Active appearance models: Theory, extensions and cases". Master Thesis. 2nd edition. Informatics and Mathematical Modelling, Technical University of Denmark. Aug 2000.
- P. Martins, "Active Appearance Models for Facial Expression Recognition and Monocular Head Pose Estimation". MSc Thesis. Department of Electrical and Computer Engineering, University of Coimbra. June 2008.
