from pathlib import Path
from typing import Iterable, List, Union
import numpy as np
import h5py
from speclite.filters import FilterSequence

from .filter_set import FilterSet


class PhotometericObservation(object):

    def __init__(self, band_names: List[str],
                 ab_magnitudes: Iterable[float],
                 ab_magnitude_errors: Iterable[float]

                 ) -> None:
        """

        A container for photometric data

        """

        assert len(band_names) == len(ab_magnitudes)
        assert len(ab_magnitudes) == len(ab_magnitude_errors)

        self._band_names: List[str] = band_names
        self._ab_magnitudes: Iterable[str] = ab_magnitudes
        self._ab_magnitude_errors: Iterable[str] = ab_magnitude_errors

        self._n_bands: int = len(band_names)

        d = {}
        self._internal_rep = {}
        for i, name in enumerate(self._band_names):
            d[name] = (self._ab_magnitudes[i],
                       self._ab_magnitude_errors[i])
            self._internal_rep[name] = (
                self._ab_magnitudes[i], self._ab_magnitude_errors[i])

        self.__dict__.update(d)

    def is_compatible_with_filter_set(self,
                                      filter_set: Union[FilterSet, FilterSequence]) -> bool:


        if isinstance(filter_set, FilterSet):

            for band in self._band_names:
                if band not in filter_set.names:
                    print(f"{band} not in filter set")
                    return False

        else:

            names = [fname.split("-")[1] for fname in filter_set.names]
            
            for band in self._band_names:
                if band not in names:
                    print(f"{band} not in filter set")
                    return False
            

        return True

    def get_mask_from_filter_sequence(self, filter_set: FilterSequence) -> Iterable[bool]:

        names = [fname.split("-")[1] for fname in filter_set.names]
        
        mask = np.zeros(len(filter_set), dtype = bool)

        for name in self._band_names:

            mask[names.index(name)] = True

        return mask

    
    def to_hdf5(self, file_name: str, overwrite: bool = False) -> None:
        """
        Save the data to an HDF5 file

        """

        file_name: Path = Path(file_name)

        if file_name.exists() and (not overwrite):
            raise RuntimeError(f"{file_name} already exists!")

        with h5py.File(file_name, "w") as f:
            for k, v in self.items():

                grp = f.create_group(k)
                grp.attrs["ab_magnitude"] = v[0]
                grp.attrs["ab_magnitude_err"] = v[1]

    @ classmethod
    def from_hdf5(cls, file_name: str):
        # type: (str) -> PhotometericObservation
        """
        Load an observation from an hdf5 file
        """

        output = {}

        with h5py.File(file_name, "r") as f:

            for band in f.keys():

                output[band] = (f[band].attrs["ab_magnitude"],
                                f[band].attrs["ab_magnitude_err"])
        return cls.from_dict(output)

    @ classmethod
    def from_kwargs(cls, **kwargs):
        # type: (dict) -> PhotometericObservation
        """
        Create an observation from a kwargs in the form
        (a=(mag, mag_err), b=(mag, mag_err))

        """
        return cls.from_dict(kwargs)

    @ classmethod
    def from_dict(cls, data: dict):
        # type: (dict) -> PhotometericObservation
        """
        Create an observation from a dict in the form
        data = dict(a=(mag, mag_err), b=(mag, mag_err))

        """

        mags = []
        mag_errs = []

        for k, v in data.items():
            mags.append(v[0])
            mag_errs.append(v[1])

        return cls(list(data.keys()), mags, mag_errs)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        raise RuntimeError("Cannot modify data!")

    def __delitem__(self, key):
        raise RuntimeError("Cannot modify data!")


#     def __setattr__(self, name, value):
#         if self._locked:
#             raise RuntimeError("Cannot modify data!")
#         else:
#             self[name] = value


    def __delattr__(self, name):
        if name in self:
            raise RuntimeError("Cannot modify data!")
        else:
            raise AttributeError("No such attribute: " + name)

    def __contains__(self, key):
        return key in self._internal_rep

    def __len__(self):
        return len(self._internal_rep)

    def __iter__(self):
        return iter(self._internal_rep)

    def keys(self):
        return self._internal_rep.keys()

    def items(self):
        return self._internal_rep.items()

    def __repr__(self):
        args = [f'{k} = {m} +/- {me}' for (k, m, me) in zip(
            self._band_names, self._ab_magnitudes, self._ab_magnitude_errors)]
        return 'PhotometricObservation({})'.format(', '.join(args))
