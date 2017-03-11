import pytest
import numpy as np

from threeML.plugins.photometry.filter_set import FilterSet
from threeML.plugins.photometry.photometric_data import PhotometryData


def test_filter_set():
    dummy_filter_name = ['g']

    dummy_trans = [4e-05, 5e-05, 7e-05, 0.00011, 0.00011999999999999999, 0.00011999999999999999, 0.00014, 0.00016,
                   0.00018, 0.00021, 0.00028, 0.0004, 0.0005200000000000001, 0.00066, 0.00081, 0.0009699999999999999,
                   0.00114, 0.00131, 0.00164, 0.0021899999999999997, 0.00309, 0.00435, 0.00595, 0.007890000000000001,
                   0.0104, 0.01374, 0.01791, 0.02376, 0.03222, 0.04448, 0.0624, 0.08871, 0.12664, 0.17814000000000002,
                   0.24211, 0.31249, 0.37765, 0.43601, 0.4913, 0.5351, 0.55703, 0.5645, 0.56767, 0.5676399999999999,
                   0.56742, 0.5672699999999999, 0.5687399999999999, 0.57765, 0.59163, 0.6038100000000001, 0.6122,
                   0.62087, 0.63134, 0.64063, 0.6491899999999999, 0.6572399999999999, 0.6613, 0.66373,
                   0.6652600000000001, 0.66714, 0.67, 0.6733100000000001, 0.67747, 0.68066, 0.6821, 0.68283, 0.68125,
                   0.67984, 0.68013, 0.6825, 0.68638, 0.69197, 0.6978300000000001, 0.7023199999999999,
                   0.7044100000000001, 0.70452, 0.70196, 0.6977800000000001, 0.69386, 0.6915100000000001, 0.69105,
                   0.69308, 0.6958300000000001, 0.69995, 0.7037, 0.70623, 0.70636, 0.7054600000000001, 0.70431, 0.70267,
                   0.70163, 0.7007800000000001, 0.70058, 0.70192, 0.703, 0.7029, 0.70262, 0.70263, 0.7016100000000001,
                   0.701]

    dummy_wave = [350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369,
                  370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,
                  390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409,
                  410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429,
                  430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449]

    fs = FilterSet(filter_names=dummy_filter_name,
                   wave_lengths=dummy_wave,
                   transmission_curves=dummy_trans,
                   magnitude_systems=['abmag'])


    assert fs.n_bands == len(dummy_filter_name)
    assert fs.waveunits == 'nm'
    assert len(fs.average_wavelength) == fs.n_bands
    assert fs.filter_names == np.array(dummy_filter_name)

    fig = fs.plot_filters()

    with pytest.raises(AssertionError):

        fs.effective_stimulus()

    def dummy_function(x):
        return x

    fs.set_model(dummy_function)

    assert fs._model_set

    es = fs.effective_stimulus()

    assert len(es) == fs.n_bands



    with pytest.raises(AssertionError):
        fs = FilterSet(filter_names=dummy_filter_name,
                       wave_lengths=dummy_wave[:1],
                       transmission_curves=dummy_trans,
                       magnitude_systems=['abmag'])


    with pytest.raises(AssertionError):
        fs = FilterSet(filter_names=dummy_filter_name,
                       wave_lengths=dummy_wave,
                       transmission_curves=dummy_trans[1:],
                       magnitude_systems=['abmag'])

    with pytest.raises(AssertionError):
        fs = FilterSet(filter_names=dummy_filter_name,
                       wave_lengths=dummy_wave,
                       transmission_curves=dummy_trans[1:],
                       magnitude_systems=['abmag','abmag'])










def test_photolike():

    pass

