"""
download url under proxy check
"""

import socket
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


http_proxy = "http://70.10.15.10:8080"
https_proxy = "http://70.10.15.10:8080"
proxyDict = {'http': http_proxy, 'https': https_proxy}


def get(url):
    cur_ip = socket.gethostbyname(socket.gethostname())

    if cur_ip.startswith("70."):
        return requests.get(url, proxies=proxyDict, verify=False)
    else:
        return requests.get(url)


if __name__ == '__main__':
    test_url = 'https://biendata.com/competition/meteorology/bj_grid/2018-05-08-15/2018-05-09-15/2k0d1d8'

    response = get(test_url)
