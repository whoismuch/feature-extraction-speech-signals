# Exercises for laboratory work


# Import of modules
import numpy as np
from scipy.fftpack import dct


def split_meta_line(line, delimiter=' '):
    # First you need to prepare split_meta_line function for meta line parsing
    # line format is "Speaker_ID Gender Path"

    """
    :param line: lines of metadata
    :param delimiter: delimeter
    :return: speaker_id: speaker IDs: gender: gender: file_path: path to file
    """

    ###########################################################
    # Here is your code

    # Extract speaker ID, gender, and file path
    components = line.split(delimiter)
    speaker_id = components[0]
    gender = components[1]
    file_path = ' '.join(components[2:])  # Rejoin the remaining components to form the file path

    file_path = file_path.replace("\n", "")
    ###########################################################

    return speaker_id, gender, file_path


def preemphasis(signal, pre_emphasis=0.97):
    # Here you need to preemphasis input signal with pre_emphasis coeffitient

    """
    :param signal: input signal
    :param pre_emphasis: preemphasis coeffitient
    :return: emphasized_signal: signal after pre-emphasis procedure
    """

    ###########################################################
    # signal[0]: первый отсчет аудиосигнала.
    # signal[1:]: весь аудиосигнал, начиная со второго отсчета до конца.
    # signal[:-1]: весь аудиосигнал, начиная с первого отсчета до предпоследнего.
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    ###########################################################

    return emphasized_signal


def framing(emphasized_signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01):
    # Here you need to perform framing of the input signal emphasized_signal with sample rate sample_rate
    # Please use hamming windowing

    """
    :param emphasized_signal: signal after pre-emphasis procedure
    :param sample_rate: signal sampling rate (частота дискретизации сигнала)
    :param frame_size: sliding window size (размер временного окна в секундах)
    :param frame_stride: step (размер шага между окнами в секундах)
    :return: frames: output matrix [nframes x sample_rate*frame_size]
    """

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal,
                           z)  # pad Signal to make sure that all frames have equal number of samples without
    # truncating any samples from the original signal

    ###########################################################

    # В цикле для каждого кадра:
    #   Определяются индексы начала и конца текущего окна.
    #   Из оригинального сигнала извлекается кусок, соответствующий текущему окну.
    #   Кусок сигнала умножается на оконную функцию Хэмминга для уменьшения артефактов на границах кадров.
    #   Полученный кадр добавляется в матрицу frames.

    frames = np.zeros((num_frames, frame_length))  # frames инициализируется пустой матрицей для хранения кадров

    for i in range(num_frames):
        start = frame_step * i
        end = start + frame_length
        frame = pad_signal[start:end] * np.hamming(frame_length)  # Применяем окно Хэмминга
        frames[i] = frame

    ###########################################################

    return frames


def power_spectrum(frames, NFFT=512):
    # Here you need to compute power spectum of framed signal with NFFT fft bins number

    """
    :param frames: framed signal
    :param NFFT: number of fft bins
    :return: pow_frames: framed signal power spectrum
    """

    # При применении FFT к сигналу происходит преобразование его из временной области в частотную.
    # Количество точек NFFT определяет разрешение этого преобразования:
    # чем больше NFFT, тем выше разрешение по частоте, и наоборот.

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # выполняем быстрое преобразование Фурье (FFT) для каждого
                                                        # кадра frames с использованием NFFT точек
                                                        # вычисляем амплитуду комплексных значений FFT, представляя их в виде магнитуды

    ###########################################################
    pow_frames = (1.0 / NFFT) * np.square(mag_frames)  # Квадрат амплитуд = спектр мощности для каждого кадра.
                                                     # (1.0 / NFFT) Это нормализующий коэффициент, который используется для нормализации спектра мощности на размер FFT.

    ###########################################################

    return pow_frames


def compute_fbank_filters(nfilt=40, sample_rate=16000, NFFT=512):
    # Here you need to compute fbank filters (FBs) for special case (sample_rate & NFFT)

    """
    :param nfilt: number of filters
    :param sample_rate: signal sampling rate
    :param NFFT: number of fft bins in power spectrum
    :return: fbank [nfilt x (NFFT/2+1)]
    """

    low_freq_mel = 0
    high_freq = sample_rate / 2

    ###########################################################
    # Here is your code to convert Convert Hz to Mel: 
    # high_freq -> high_freq_mel

    high_freq_mel = 2595 * np.log10(1 + high_freq / 700)  # Вычисляется верхняя граница частоты в мел-шкале

    ###########################################################

    # Используя полученные граничные значения в мел-шкале,
    # вычисляются равномерно распределенные точки в мел-шкале с помощью np.linspace()
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # equally spaced in mel scale

    ###########################################################
    # Here is your code to convert Convert Mel to Hz: 
    # mel_points -> hz_points

    # Для каждой точки в мел-шкале вычисляется соответствующая частота в герцах с помощью обратной формулы:
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    ###########################################################

    # Далее, каждая частота из результата предыдущего шага преобразуется в соответствующий индекс в массиве бинов FFT.
    # Это позволяет определить, какие частотные бины FFT будут использоваться для вычисления мел-фильтров.
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)


    # Для каждого мел-фильтра вычисляются его коэффициенты.
    # Эти коэффициенты определяют, какую часть энергии из спектра каждого бина
    # следует накапливать в результате работы соответствующего мел-фильтра.
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return fbank


def compute_fbanks_features(pow_frames, fbank):
    # You need to compute fbank features using power spectrum frames and suitable fbanks filters

    """
    :param pow_frames: framed signal power spectrum, matrix [nframes x sample_rate*frame_size]
    :param fbank: matrix of the fbank filters [nfilt x (NFFT/2+1)] where NFFT: number of fft bins in power spectrum
    :return: filter_banks_features: log mel FB energies matrix [nframes x nfilt]
    """

    ###########################################################
    # Here is your code to compute filter_banks_features

    filter_banks_features = np.dot(pow_frames, fbank.T) # вычисляем акустические признаки

    ###########################################################

    filter_banks_features = np.where(filter_banks_features == 0, np.finfo(float).eps,
                                     filter_banks_features)  # для обеспечения численной стабильности.
                                                            # Заменяем все значения, равные нулю в filter_banks_features,
                                                            # на небольшое положительное число eps, чтобы избежать ошибок при вычислении логарифма


    filter_banks_features = np.log(filter_banks_features) # Вычисляем логарифм от каждого элемента матрицы filter_banks_features

    # логарифмические энергии для каждого фильтра фильтр-банка в каждом фрейме
    return filter_banks_features


def compute_mfcc(filter_banks_features, num_ceps=20):
    # Here you need to compute MFCCs features using precomputed log mel FB energies matrix

    """
    :param filter_banks_features: log mel FB energies matrix [nframes x nfilt]
    :param num_ceps: number of cepstral components for MFCCs
    :return: mfcc: mel-frequency cepstral coefficients (MFCCs)
    """

    ###########################################################
    # Here is your code to compute mfcc features

    # Compute DCT matrix
    cep_count, freq_count = num_ceps, filter_banks_features.shape[1]
    cep_index, freq_index = np.meshgrid(np.arange(cep_count), np.arange(freq_count))
    dct_matrix = np.cos(np.pi * (cep_index + 0.5) / cep_count * (freq_index + 1))
    dct_matrix[0] *= np.sqrt(0.5)

    # Apply Discrete Cosine Transform (DCT) to filter bank energies
    mfcc = np.dot(filter_banks_features, dct_matrix)

    # Keep only the first num_ceps coefficients
    mfcc = mfcc[:, 1:num_ceps + 1]

    ###########################################################

    return mfcc


def mvn_floating(features, LC, RC, unbiased=False):
    # Here you need to do mean variance normalization of the input features

    """
    :param features: features matrix [nframes x nfeats]
    :param LC: the number of frames to the left defining the floating
    :param RC: the number of frames to the right defining the floating
    :param unbiased: biased or unbiased estimation of normalising sigma
    :return: normalised_features: normalised features matrix [nframes x nfeats]
    """

    nframes, dim = features.shape
    LC = min(LC, nframes - 1)
    RC = min(RC, nframes - 1)
    n = (np.r_[np.arange(RC + 1, nframes), np.ones(RC + 1) * nframes] - np.r_[np.zeros(LC), np.arange(nframes - LC)])[:,
        np.newaxis]
    f = np.cumsum(features, 0)
    s = np.cumsum(features ** 2, 0)
    f = (np.r_[f[RC:], np.repeat(f[[-1]], RC, axis=0)] - np.r_[np.zeros((LC + 1, dim)), f[:-LC - 1]]) / n
    s = (np.r_[s[RC:], np.repeat(s[[-1]], RC, axis=0)] - np.r_[np.zeros((LC + 1, dim)), s[:-LC - 1]]
         ) / (n - 1 if unbiased else n) - f ** 2 * (n / (n - 1) if unbiased else 1)

    ###########################################################
    # Here is your code to compute normalised features

    mean_features = np.mean(features, axis=0)
    std_features = np.std(features, axis=0)
    normalised_features = (features - mean_features) / std_features

    normalised_features[np.isnan(normalised_features)] = 0

    ###########################################################

    normalised_features[s == 0] = 0

    return normalised_features
