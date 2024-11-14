from app import PICDens


def main():
    savePath = "path/to/save_dir/"
    root = "path/to/img_root_dir/"
    pixToMcmCoef = 0.42604
    speciesName = 'PS' # Pinus Sylvestris

    pD = PICDens(savePath=savePath, root=root, pixToMcmCoef=pixToMcmCoef, speciesName=speciesName)
    pD.startScan()