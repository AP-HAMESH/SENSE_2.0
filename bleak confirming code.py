import asyncio
from bleak import BleakScanner, BleakClient

DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

def handle_notify(sender, data: bytearray):
    value = int.from_bytes(data, byteorder='big')
    print(f"Notify: {value}")

async def main():
    print("Scanning for 5s...")
    devices = await BleakScanner.discover(timeout=5.0)

    target = next((d for d in devices if d.name and "SENSE2-EEG" in d.name), None)
    if not target:
        print("Device not found. Devices seen:", [d.name for d in devices])
        return

    print(f"Found {target.name} @ {target.address}")

    async with BleakClient(target.address) as client:
        print("Connected:", client.is_connected)
        await client.start_notify(DATA_CHAR_UUID, handle_notify)
        await asyncio.sleep(15)   # collect notifications for 15s
        await client.stop_notify(DATA_CHAR_UUID)

asyncio.run(main())