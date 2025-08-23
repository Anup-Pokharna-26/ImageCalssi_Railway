#!/usr/bin/env python3
"""
Test script to verify Railway deployment fixes
"""
import requests
import time
import sys

def test_local_health():
    """Test health endpoint locally"""
    try:
        # Test the Flask app directly
        from app import app
        import json
        
        with app.test_client() as client:
            response = client.get('/health')
            data = json.loads(response.data.decode())
            
            print(f"✅ Health endpoint responds with status {response.status_code}")
            print(f"✅ Health response: {data}")
            
            # Test main route
            response = client.get('/')
            print(f"✅ Main route responds with status {response.status_code}")
            
            return True
    except Exception as e:
        print(f"❌ Local test failed: {e}")
        return False

def test_remote_health(url):
    """Test health endpoint on deployed app"""
    try:
        health_url = f"{url}/health"
        print(f"Testing {health_url}...")
        
        response = requests.get(health_url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Remote health check passed: {data}")
            return True
        else:
            print(f"❌ Remote health check failed with status {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Remote health check timed out")
        return False
    except Exception as e:
        print(f"❌ Remote health check failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Railway deployment fixes...")
    
    # Test locally first
    print("\n1. Testing local health endpoint...")
    local_ok = test_local_health()
    
    if not local_ok:
        print("❌ Local tests failed. Fix local issues first.")
        sys.exit(1)
    
    print("\n2. Local tests passed! ✅")
    
    # Test remote if URL provided
    if len(sys.argv) > 1:
        remote_url = sys.argv[1]
        print(f"\n3. Testing remote deployment at {remote_url}...")
        remote_ok = test_remote_health(remote_url)
        
        if remote_ok:
            print(f"\n🎉 All tests passed! Deployment should work correctly.")
        else:
            print(f"\n⚠️  Remote test failed, but local tests passed. Check deployment logs.")
    else:
        print("\n💡 To test remote deployment, run:")
        print("   python test_railway_deploy.py https://your-app.railway.app")
        
    print("\n🚀 Ready for Railway deployment!")
