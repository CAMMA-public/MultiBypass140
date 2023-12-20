import glob
import pickle

bern = 'labels/bern/labels_by70_splits/labels/'
strasbourg = 'labels/strasbourg/labels_by70_splits/labels/'

files = glob.glob(f'{bern}/*/*')

for f in files:
    print('\n', f)
    

files = glob.glob(f'{strasbourg}/*/*')

for f in files:
    print('\n', f)