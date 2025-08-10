#!/usr/bin/env python3
"""
Test script for Colab integration functionality.
This simulates how Arbor would be used in a notebook environment.
"""

import os
import sys
import tempfile
import time


def test_colab_integration():
    """Test the basic Colab integration functionality."""
    print("🧪 Testing Arbor Colab Integration")
    print("=" * 50)

    # Test 1: Import and basic functionality
    print("\n1️⃣  Testing import and basic functionality...")
    try:
        import arbor

        print("✅ Successfully imported arbor")

        # Test status when no server running
        status = arbor.status()
        if status is None:
            print("✅ Correctly reports no server running")
        else:
            print("⚠️  Server already running, shutting down first...")
            arbor.shutdown()

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test 2: Server initialization
    print("\n2️⃣  Testing server initialization...")
    try:
        # Use a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            server_info = arbor.init(
                storage_path=temp_dir,
                gpu_ids=[],  # Use CPU mode for testing
                silent=False,
            )

            print(f"✅ Server started on {server_info['host']}:{server_info['port']}")
            print(f"   Base URL: {server_info['base_url']}")

            # Test 3: Status check
            print("\n3️⃣  Testing status check...")
            status = arbor.status()
            if status:
                print("✅ Status check successful")
                print(f"   Port: {status['port']}")
                print(f"   Storage: {status['storage_path']}")
            else:
                print("❌ Status check failed")
                return False

            # Test 4: Client creation
            print("\n4️⃣  Testing client creation...")
            try:
                client = arbor.get_client()
                print("✅ OpenAI client created successfully")
            except ImportError:
                print("⚠️  OpenAI package not installed, skipping client test")
            except Exception as e:
                print(f"❌ Client creation failed: {e}")
                return False

            # Test 5: Multiple init calls (should be safe)
            print("\n5️⃣  Testing multiple init calls...")
            try:
                server_info2 = arbor.init(silent=True)
                if server_info2["port"] == server_info["port"]:
                    print("✅ Multiple init calls handled correctly")
                else:
                    print("❌ Multiple init calls created different servers")
                    return False
            except Exception as e:
                print(f"❌ Multiple init test failed: {e}")
                return False

            # Test 6: Server health (basic check)
            print("\n6️⃣  Testing server health...")
            try:
                import socket

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex(("127.0.0.1", server_info["port"]))
                    if result == 0:
                        print("✅ Server is accepting connections")
                    else:
                        print("❌ Server is not accepting connections")
                        return False
            except Exception as e:
                print(f"❌ Health check failed: {e}")
                return False

            # Test 7: Shutdown
            print("\n7️⃣  Testing shutdown...")
            try:
                arbor.shutdown()

                # Verify server is actually down
                time.sleep(1)  # Give it a moment to shut down
                status = arbor.status()
                if status is None:
                    print("✅ Server shutdown successful")
                else:
                    print("❌ Server still running after shutdown")
                    return False

            except Exception as e:
                print(f"❌ Shutdown failed: {e}")
                return False

    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        return False

    print("\n🎉 All tests passed!")
    return True


def test_environment_detection():
    """Test environment detection functions."""
    print("\n🔍 Testing Environment Detection")
    print("=" * 50)

    try:
        from arbor.colab import is_colab_environment, is_jupyter_environment

        colab_detected = is_colab_environment()
        jupyter_detected = is_jupyter_environment()

        print(f"Colab environment: {colab_detected}")
        print(f"Jupyter environment: {jupyter_detected}")

        # In a normal Python script, both should be False
        if not colab_detected and not jupyter_detected:
            print("✅ Environment detection working correctly (not in notebook)")
        else:
            print("ℹ️  Detected notebook environment")

        return True

    except Exception as e:
        print(f"❌ Environment detection failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🌳 Arbor Colab Integration Test Suite")
    print("=" * 50)

    success = True

    # Test environment detection
    if not test_environment_detection():
        success = False

    # Test main colab integration
    if not test_colab_integration():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Colab integration is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
