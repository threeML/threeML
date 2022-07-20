from __future__ import print_function
import socket
import os
import requests


def internet_connection_is_active():
    """
    Check that a internet connection is working by trying contacting the following host:

    """

    timeout = 3

    if os.environ.get("http_proxy") is None:

        # No proxy

        # Host: 8.8.8.8 (google-public-dns-a.google.com)
        # OpenPort: 53/tcp
        # Service: domain (DNS/TCP)

        host = "8.8.8.8"
        port = 53

        try:

            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))

        except Exception as ex:

            print(ex.message)
            return False

        else:

            return True

    else:

        # We have a proxy. We cannot connect straight to the DNS of Google, we need to tunnel through the proxy
        # Since using raw sockets gets complicated and error prone, especially if the proxy has authentication tokens,
        # we just try to reach google with a sensible timeout
        try:

            _ = requests.get("http://google.com", timeout=timeout)

        except Exception as ex:

            print(ex.message)
            return False

        else:

            return True
