import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
def plot_hmm_crf_compare_accuracy():
	size = [20, 200, 2000, 20000]
	hmm = [0.31269, 0.56245, 0.81145, 0.92368]
	crf = [0.69569, 0.84256, 0.92607, 0.95716]
	t = np.arange(0., 5., 0.2)

	# red dashes, blue squares and green triangles
	line_h = plt.plot(size, hmm, label = 'hmm', color='red', linewidth=1.0, linestyle='--')
	line_c = plt.plot(size, crf, label = 'crf', color='blue', linewidth=1.0, linestyle='--')

	plt.legend(loc='upper right')

	plt.show

def plot_hmm_crf_compart_time():
	size = [20, 200, 2000, 20000]
	crf = [0.0569, 0.6640, 14.8842, 461.1556]
	hmm = [0.0003, 0.0258, 0.2238, 2.8415]
	t = np.arange(0., 5., 0.2)

	# red dashes, blue squares and green triangles
	line_h = plt.plot(size, hmm, label = 'hmm', color='red', linewidth=1.0, linestyle='--')
	line_c = plt.plot(size, crf, label = 'crf', color='blue', linewidth=1.0, linestyle='--')

	plt.legend(loc='upper right')

	plt.show()

if __name__ == "__main__":
	plot_hmm_crf_compart_time()
