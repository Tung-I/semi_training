import logging
import argparse
import numpy as np
from skimage import io, morphology
from pathlib import Path

def main(args):
	data_dir = args.data_dir
	output_dir = args.output_dir
	paths = [path for path in data_dir.iterdir() if path.is_dir()]

	for idx, path in enumerate(paths):
		logging.info(f'Process {idx}.')
		if not (output_dir / Path(str(idx))).is_dir():
			(output_dir / Path(str(idx))).mkdir(parents=True)

			img_path = list((path / Path('image')).glob('*.png'))
			label_path = list((path / Path('label')).glob('*.npy'))
			img = np.asarray(io.imread(img_path[0]))
			label = np.load(label_path[0])

			label[np.where(label!=0)] = 1
			label_ero = morphology.binary_erosion(label, morphology.square(3))
			label_edge = label - label_ero
			label_012 = label + label_edge

			np.save(output_dir / Path(str(idx)) / Path('image.npy'), img)
			np.save(output_dir / Path(str(idx)) / Path('label.npy'), label_012)


def _parse_args():
	parser = argparse.ArgumentParser(description="The data preprocessing.")
	parser.add_argument('data_dir', type=Path, help='The directory of the dataset.')
	parser.add_argument('output_dir', type=Path, help='The directory of the processed data.')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', 
						level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
	args = _parse_args()
	main(args)