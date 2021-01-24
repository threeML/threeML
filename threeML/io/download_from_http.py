import os
import re
from builtins import object
from pathlib import Path

import requests

from threeML.config.config import threeML_config
from threeML.io.file_utils import (file_existing_and_readable,
                                   path_exists_and_is_directory,
                                   sanitize_filename)
from threeML.io.logging import setup_logger
from threeML.utils.progress_bar import tqdm

log = setup_logger(__name__)


class RemoteDirectoryNotFound(IOError):
    pass


class HTTPError(IOError):

    pass


class ApacheDirectory(object):
    """
    Allows to interact with a directory listing like the one returned by an Apache server
    """

    def __init__(self, url):

        self._request_result = requests.get(url)

        # Make sure the request was ok
        if not self._request_result.ok:

            if self._request_result.reason == "Not Found":

                raise RemoteDirectoryNotFound(
                    "Remote directory %s does not exist" % url
                )

            else:

                raise HTTPError(
                    "HTTP request failed with reason: %s" % self._request_result.reason
                )

        self._text = self._request_result.text

        # Get the listing of files and directories
        self._entries = self._get_directory_entries()

        # Now split directories and files
        self._files = []
        self._directories = []

        for entry in self._entries:

            if entry[1] == "FILE":

                self._files.append(entry[0])

            else:

                self._directories.append(entry[0])

    def _get_directory_entries(self):
        """
        List files and directories listed in the listing

        :return: a list of tuples (entry name, type (DIR or FILE))
        """

        # Get the files listed in the directory
        # A line in an Apache listing is like this:
        # <img src="/icons/unknown.gif" alt="[   ]">
        # <a href="glg_cspec_b0_bn100101988_v02.rsp">glg_cspec_b0_bn100101988_v02.rsp</a>
        #                16-Nov-2012 15:14   96K
        regexp = re.compile("<img src=.+ alt=(.+)>\s?<a href=.+>(.+)</a>.+")

        # Apache puts files in a <pre></pre> tag, so lines are ended simply with \n
        lines = self._text.split("\n")

        # Now loop over the lines and extract the file name
        entries = []

        for line in lines:

            token = re.match(regexp, line)

            if token is not None:

                # This line contains a file or a directory

                type_token, filename_token = token.groups()

                # Figure out if this is a directory or a file. A directory has a alt="[DIR]" attribute in the
                # <img> tag, a file has a alt="[   ]" or other things (if a known type)

                if type_token.upper().find("DIR") >= 0:

                    entry_type = "DIR"

                else:

                    entry_type = "FILE"

                # Append entry

                entries.append((filename_token, entry_type))

        return entries

    @property
    def files(self):

        return self._files

    @property
    def directories(self):

        return self._directories

    def download(
        self,
        remote_filename,
        destination_path: str,
        new_filename=None,
        progress=True,
        compress=False,
    ):

        assert (
            remote_filename in self.files
        ), "File %s is not contained in this directory (%s)" % (
            remote_filename,
            self._request_result.url,
        )

        destination_path: Path = sanitize_filename(
            destination_path, abspath=True)

        assert path_exists_and_is_directory(destination_path), (
            f"Provided destination {destination_path} does not exist or "
            "is not a directory"
        )

        # If no filename is specified, use the same name that the file has on the remote server

        if new_filename is None:
            new_filename: str = remote_filename.split("/")[-1]

        # Get the fully qualified path for the remote and the local file

        remote_path: str = self._request_result.url + remote_filename
        local_path: Path = destination_path / new_filename

        # Ask the server for the file, but do not download it just yet
        # (stream=True will get the HTTP header but nothing else)
        # Use stream=True for two reasons:
        # * so that the file is not downloaded all in memory before being written to the disk
        # * so that we can report progress is requested

        this_request = requests.get(remote_path, stream=True)

        # Figure out the size of the file

        file_size = int(this_request.headers["Content-Length"])

        log.debug(f"downloading {remote_filename} of size {file_size}")

        # Now check if we really need to download this file

        if compress:
            # Add a .gz at the end of the file path

            log.debug(
                f"file {remote_filename} will be downloaded and compressed")

            local_path: Path = Path(f"{local_path}.gz")

        if file_existing_and_readable(local_path):

            local_size = os.path.getsize(local_path)

            if local_size == file_size or compress:
                # if the compressed file already exists
                # it will have a smaller size

                # No need to download it again

                log.info(f"file {remote_filename} is already downloaded!")

                return local_path

        if local_path.is_file():

            first_byte = os.path.getsize(local_path)

        else:

            first_byte = 0

        # Chunk size shouldn't bee too small otherwise we are causing a bottleneck in the download speed
        chunk_size = 1024 * 10

        # If the user wants to compress the file, use gzip, otherwise the normal opener
        if compress:

            import gzip

            opener = gzip.open

        else:

            opener = open

        if threeML_config["interface"]["show_progress_bars"]:

            # Set a title for the progress bar
            bar_title = "Downloading %s" % new_filename

            total_size = int(this_request.headers.get('content-length', 0))

            bar = tqdm(
                initial=first_byte,
                unit_scale=True,
                unit_divisor=1024,
                unit="B",
                total=int(this_request.headers["Content-Length"]),
                desc=bar_title,
            )

            with opener(local_path, "wb") as f:

                for chunk in this_request.iter_content(chunk_size=chunk_size):

                    if chunk:  # filter out keep-alive new chunks

                        f.write(chunk)
                        bar.update(len(chunk))

            this_request.close()
            bar.close()

        else:

            with opener(local_path, "wb") as f:

                for chunk in this_request.iter_content(chunk_size=chunk_size):

                    if chunk:  # filter out keep-alive new chunks

                        f.write(chunk)

            this_request.close()

        return local_path

    def download_all_files(self, destination_path, progress=True, pattern=None):
        """
        Download all files in the current directory

        :param destination_path: the path for the destination directory in the local file system
        :param progress: (True or False) whether to display progress or not
        :param pattern: (default: None) If not None, only files matching this pattern (a regular expression) will be
        downloaded
        :return: list of the downloaded files as absolute paths in the local file system
        """

        local_files = []

        for file in self.files:

            if pattern is not None:

                if re.match(pattern, os.path.basename(file)) is None:

                    continue

            this_local_file = self.download(
                file, destination_path, progress=progress)

            local_files.append(this_local_file)

        return local_files
