# test_installation.py
import pyorbbecsdk

# Check version (run: pip show pyorbbecsdk2 | grep Version)
# or check SDK version with: pyorbbecsdk.get_version()

# Initialize context and list devices
context = pyorbbecsdk.Context()
device_list = context.query_devices()
device_count = device_list.get_count()

print(f"[OK] Found {device_count} Orbbec device(s)")

if device_count > 0:
    device = device_list.get_device_by_index(0)
    device_info = device.get_device_info()
    print(f"[OK] Device name: {device_info.get_name()}")
    print(f"[OK] Serial number: {device_info.get_serial_number()}")

print("\n[OK] Installation verified successfully!")
