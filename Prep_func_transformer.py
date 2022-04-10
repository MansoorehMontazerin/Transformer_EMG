import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal
plt.rcParams["figure.figsize"] = (15, 15)


# Defining the preprocessing functions
def low_pass_filter(data, N, f, fs=2048):
    """
    This function applies a low-pass butterworth filter to the data.
    This filter gives the positive envelope of the signal.

    Parameters
   ----------

    data:
        Data points of shape (data_samples,h_grid_flex or _extens,v_grid_flex or _extens)

    N:
        Butterworth order

    f:
        Cutoff frequency

    fs:
        Sampling frequncy

    Output:
        Filtered data
        Thesame shape as input
    """

    f = f / (fs / 2)
    data = np.abs(data)
    b, a = signal.butter(N=N, Wn=f, btype="low")
    output = signal.filtfilt(
        b, a, data,  axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1))

    return output


def encode_mu_law(x, mu):
    """
    Miu-law normalization

    Parameters
   ----------

    x:
        Data points of shape (data_samples,h_grid_flex or _extens,v_grid_flex or _extens)

    mu:
        Miu parameter
    """
    mu = mu-1
    fx = np.sign(x)*np.log(1+mu*np.abs(x))/np.log(1+mu)

    return (fx+1)/2*mu+0.5


def windows(end, window_size, skip_step):
    """
    This function creates a generator yielding windows with the length of the window_size.
    It starts to segment a sequence from the start of the data points to its end.

    """
    start = 0
    while (start + window_size) <= end:
        yield start, start + window_size
        start += skip_step


def DataGenerator(sequence_matrix, window_size, skip_step):
    """
    This function creates a data generator yielding windowed chunks of its input data (e.g. sequence_matrix).
    No padding is used here.

    """
    num_features = sequence_matrix.shape[0]  # number of sequence elements
    for start, stop in windows(num_features, window_size, skip_step):
        yield (sequence_matrix[start:stop])


def loaddata_filt(mode, window_size, skip_step, idx):
    """
    This function loads data from the data folders and then splits them to a desired window_size with a desired skip_step.
    Previously, the corresponding data to each subject, 
    each gesture and each repetition are saved in separate folders.
    The data corresponding to flexor and extensor muscle's electrodes is also separate.

    Parameters
   ----------

    idx:
        Subject index

    mode:
        train or test
        creates different data generators in different modes
    """

    xf = []
    yf = []
    xe = []
    num_gst = 66
    num_rep = 5

    for gst in range(0, num_gst):
        for rep in range(0, num_rep):
            if not(rep == 1 and gst == 33):

                # Loading flexor data
                with open('EMG_training_trans/Subj_{}/class{}{}/repetition{}/flexors.pkl'
                          .format(idx, (gst+1)//10, (gst+1) % 10, rep+1), 'rb') as f:
                    emg_sigf_final = pickle.load(f)

                # Loading extensor data
                with open('EMG_training_trans/Subj_{}/class{}{}/repetition{}/extensors.pkl'
                          .format(idx, (gst+1)//10, (gst+1) % 10, rep+1), 'rb') as f:
                    emg_sige_final = pickle.load(f)

                emg_sigf_filtered = low_pass_filter(emg_sigf_final, N=1, f=1)
                emg_sigf_clipped = np.clip(emg_sigf_filtered, 0, 0.1)
                emg_sigf_norm = encode_mu_law(emg_sigf_clipped, mu=8)-4

                emg_sige_filtered = low_pass_filter(emg_sige_final, N=1, f=1)
                emg_sige_clipped = np.clip(emg_sige_filtered, 0, 0.1)
                emg_sige_norm = encode_mu_law(emg_sige_clipped, mu=8)-4

                # Concatenating flexor and extensor data to each other
                emg_sigt_norm = np.concatenate(
                    (emg_sigf_norm, emg_sige_norm), axis=2)[:,:,1::2]

                if rep+1 in [1, 2, 3, 4] and mode == "train":

                    data_gen = DataGenerator(
                        emg_sigt_norm, window_size, skip_step)
                    for index, sample in enumerate(data_gen):
                        xf.append(sample)
                        yf.append(gst)

                elif rep+1 in [5] and mode == "test":

                    data_gen = DataGenerator(
                        emg_sigt_norm, window_size, skip_step)
                    for index, sample in enumerate(data_gen):
                        xf.append(sample)
                        yf.append(gst)

    return xf, yf, xe
