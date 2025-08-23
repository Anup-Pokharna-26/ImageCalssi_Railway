#!/usr/bin/env python3
"""
Simple Health Check CLI Tool
Usage: python healthcheck.py [URL]
"""

import sys
import requests
import time
import argparse
from urllib.parse import urlparse

def check_url_health(url, timeout=30):
    """Check if a URL is healthy (returns 200 status)"""
    try:
        # Add http:// if no protocol specified
        if not urlparse(url).scheme:
            url = f"http://{url}"
        
        print(f"Checking health of: {url}")
        start_time = time.time()
        
        response = requests.get(url, timeout=timeout)
        response_time = round((time.time() - start_time) * 1000, 2)
        
        if response.status_code == 200:
            print(f"✅ HEALTHY - Status: {response.status_code}, Response time: {response_time}ms")
            return True
        else:
            print(f"⚠️  UNHEALTHY - Status: {response.status_code}, Response time: {response_time}ms")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ TIMEOUT - Request timed out after {timeout} seconds")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ CONNECTION ERROR - Cannot connect to {url}")
        return False
    except Exception as e:
        print(f"❌ ERROR - {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Simple Health Check CLI Tool')
    parser.add_argument('url', nargs='?', default='http://localhost:5000', 
                       help='URL to check (default: http://localhost:5000)')
    parser.add_argument('--timeout', '-t', type=int, default=30,
                       help='Timeout in seconds (default: 30)')
    parser.add_argument('--continuous', '-c', action='store_true',
                       help='Run continuous health checks every 30 seconds')
    
    args = parser.parse_args()
    
    if args.continuous:
        print(f"Starting continuous health checks for {args.url}...")
        print("Press Ctrl+C to stop")
        try:
            while True:
                check_url_health(args.url, args.timeout)
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopped health checks")
    else:
        success = check_url_health(args.url, args.timeout)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
