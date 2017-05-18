import socket


def internet_connection_is_active(host="8.8.8.8", port=53, timeout=3):
    """
    Check that a internet connection is working by trying contacting the following host:

    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """

    try:

        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))

    except Exception as ex:

        print(ex.message)
        return False

    else:

        return True
