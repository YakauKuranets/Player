"""
Simple load testing script for the PLAYE PhotoLab backend.

This script sends concurrent requests to a specified API endpoint. It can
be used to perform basic stress testing. Requires Python 3.7+.

Usage:
    python3 load_test.py --url http://localhost:8000/api/ai/face-enhance \
        --file sample.png --token <jwt_token> --requests 50 --concurrency 5

Note: For more advanced load testing consider using dedicated tools like
locust (https://locust.io/).
"""
import argparse
import base64
import concurrent.futures
import threading
import time
from pathlib import Path
from urllib import request, error


def send_request(url: str, file_path: Path, token: str) -> float:
    """Send a POST request with an image file and return the response time."""
    data = file_path.read_bytes()
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    # Build multipart/form-data body manually
    body_lines = [
        f'--{boundary}',
        'Content-Disposition: form-data; name="file"; filename="{}"'.format(file_path.name),
        'Content-Type: image/png',
        '',
        data,
        f'--{boundary}--',
        ''
    ]
    # Join parts, ensuring bytes for file content
    body = b""
    for line in body_lines:
        if isinstance(line, bytes):
            body += line + b"\r\n"
        else:
            body += line.encode('utf-8') + b"\r\n"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': f'multipart/form-data; boundary={boundary}'
    }
    req = request.Request(url, data=body, headers=headers)
    start = time.time()
    try:
        with request.urlopen(req) as resp:
            resp.read()
    except error.HTTPError as e:
        print(f"Request failed: {e}")
    end = time.time()
    return end - start


def worker(url: str, file_path: Path, token: str, results: list, lock: threading.Lock):
    dt = send_request(url, file_path, token)
    with lock:
        results.append(dt)


def main():
    parser = argparse.ArgumentParser(description="Basic load testing script")
    parser.add_argument('--url', required=True, help='Endpoint URL')
    parser.add_argument('--file', required=True, help='Path to image file')
    parser.add_argument('--token', required=True, help='JWT token for authentication')
    parser.add_argument('--requests', type=int, default=10, help='Number of requests to send')
    parser.add_argument('--concurrency', type=int, default=2, help='Number of concurrent workers')
    args = parser.parse_args()

    file_path = Path(args.file)
    results = []
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker, args.url, file_path, args.token, results, lock)
                   for _ in range(args.requests)]
        concurrent.futures.wait(futures)

    # Print results
    if results:
        avg_time = sum(results) / len(results)
        print(f"Sent {len(results)} requests to {args.url}")
        print(f"Average response time: {avg_time:.3f} seconds")
        print(f"Min/Max: {min(results):.3f} / {max(results):.3f} seconds")


if __name__ == '__main__':
    main()