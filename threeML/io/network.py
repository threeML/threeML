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

        try:

            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, 53))
            return True
            
        except Exception as ex:

            print(ex.message)

        #if port 53 doesn't work (eg on MacOS), try port 443.
        try:
        
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, 443))
            return True
  
        except Exception as ex:

            print(ex.message)
 
    
    # Last attempt. We either have a proxy, or the above failed.
    # With a proxy, we cannot connect straight to the DNS of Google, we need to tunnel through the proxy
    # Since using raw sockets gets complicated and error prone, especially if the proxy has authentication tokens,
    # we just try to reach google with a sensible timeout
    try:

        _ = requests.get("http://google.com", timeout=timeout)
        return True

    except Exception as ex:

        print(ex.message)
        return False
