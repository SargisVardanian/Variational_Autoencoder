import numpy as np
import os
from cv2 import imread, resize
import pandas as pd
import subprocess


def fetch_lfw_dataset(attrs_name="lfw_attributes.txt",
                      images_name="lfw-deepfunneled",
                      raw_images_name="lfw",
                      use_raw=False,
                      dx=80, dy=80,
                      dimx=45, dimy=45
                      ):
    # download if not exists
    if (not use_raw) and not os.path.exists(images_name):
        print("images not found, downloading...")
        subprocess.run(["powershell", "Invoke-WebRequest -Uri http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz -OutFile tmp.tgz"], shell=True)
        print("extracting...")
        subprocess.run(["tar", "xvzf", "tmp.tgz"], shell=True)
        os.remove("tmp.tgz")
        print("done")
        assert os.path.exists(os.path.join(images_name, "lfw"))

    if use_raw and not os.path.exists(raw_images_name):
        print("images not found, downloading...")
        subprocess.run(["powershell", "Invoke-WebRequest -Uri http://vis-www.cs.umass.edu/lfw/lfw.tgz -OutFile tmp.tgz"], shell=True)
        subprocess.run(["powershell", "Invoke-WebRequest -Uri http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz -OutFile tmp.tgz"], shell=True)
        print("extracting...")
        subprocess.run(["tar", "xvzf", "tmp.tgz"], shell=True)
        os.remove("tmp.tgz")
        print("done")
        assert os.path.exists(os.path.join(raw_images_name, "lfw"))

    if not os.path.exists(attrs_name):
        print("attributes not found, downloading...")
        subprocess.run(["powershell", "Invoke-WebRequest -Uri http://www.cs.columbia.edu/CAVE/databases/pubfig/download/%s -OutFile %s" % (attrs_name, attrs_name)])
        print("done")

    # read attrs
    df_attrs = pd.read_csv(attrs_name, sep='\t', skiprows=1,)
    df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])
    df_attrs.imagenum = df_attrs.imagenum.astype(np.int64)

    # read photos
    dirname = raw_images_name if use_raw else images_name
    photo_ids = []
    for dirpath, dirnames, filenames in os.walk(dirname):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath, fname)
                photo_id = fname[:-4].replace('_', ' ').split()
                person_id = ' '.join(photo_id[:-1])
                photo_number = int(photo_id[-1])
                photo_ids.append({'person': person_id, 'imagenum': photo_number, 'photo_path': fpath})

    photo_ids = pd.DataFrame(photo_ids)

    # mass-merge
    # (photos now have same order as attributes)
    df = pd.merge(df_attrs, photo_ids, on=('person', 'imagenum'))

    assert len(df) == len(df_attrs), "lost some data when merging dataframes"

    # image preprocessing
    all_photos = df['photo_path'].apply(imread) \
        .apply(lambda img: img[dy:-dy, dx:-dx]) \
        .apply(lambda img: resize(img, (dimx, dimy)))

    all_photos = np.stack(all_photos.values).astype('uint8')
    all_attrs = df.drop(["photo_path", "person", "imagenum"], axis=1)

    return all_photos, all_attrs
