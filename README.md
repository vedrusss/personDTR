# personDTR
Person Detection-Tracking-ReId pipe

The project detects persons onto video, counts and tracks them, re-detects them after occlusion or out of frame movement.
The project uses external frameworks:
    - YOLOv6 to detect persons,
    - deep-person-reid to obtain person Feature Vectors used to re-identify lost-appeared detections.

Multi-object tracker is built using base trackers implemented with opencv. It's logic enforced with identification algorithms used
to find corresponded tracked object for each detected object (built a trajectory) and
to re-initialize lost tracker with corresponded detected object (in case of occlusion or out of frame disapper/appear).

1. Install
    1.1 Important: the repository must be cloned with --recursive flag. If it wasn't one should run command 
        git submodule update --init --recursive
    
    1.2 run ./install.sh

    1.3 Download detector and re-id models and put them into 'weights' folder:
        
        - detector model: https://drive.google.com/file/d/1UOjz7zbxqIovg3Pr-wMAOz7xcgUZc75u/view?usp=sharing
        
        - re-id model: https://drive.google.com/file/d/178fDv_e_O5g-ErcyTNF0O28H95AvVivY/view?usp=sharing

2. Run the demo
    
        python3 demo_vid.py -dm weights/yolov6s.pt -en osnet_x0_25 -ep weights/osnet_x0_25_imagenet.pth -i <video filepath>

