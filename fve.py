import sys
import os

sys.path.append(os.getcwd())

from torchreid.utils import FeatureExtractor

class FVE:
    def __init__(self, model_name, model_path, device='cuda'):
        self.__fve = FeatureExtractor(model_name=model_name, model_path=model_path, device=device)
        print(f"Loaded model {model_name}")
    
    def __call__(self, image):
        fv = self.__fve(image).to('cpu').numpy()[0].tolist()
        return fv
    

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <model name> <model path> <image(-s)>")
        quit()
    image_list = sys.argv[3:]

    fve = FVE(sys.argv[1], sys.argv[2])
    fp = open("features.txt", 'w')
    for image_path in image_list:
        fv = fve(image_path)  #  output (images_list len, 512)
        print(f"{os.path.basename(image_path)} : {len(fv)}")
        fp.write(f"{os.path.basename(image_path)} {str(fv[0])}")
        for el in fv[1:]: fp.write(f",{str(el)}")
        fp.write("\n")