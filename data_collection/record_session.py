import serial
import csv
import time

GESTURE_LABELS = {
    '0': 'rest',
    '1': 'open',
    '2': 'close',
    '3': 'pinch'
}

def record_session(port, output_path, samples_per_gesture=200):
    ser = serial.Serial(port, 9600, timeout=2)
    rows = []

    for label, name in GESTURE_LABELS.items():
        input(f"\nPrepare for gesture: {name.upper()} — press Enter when ready...")
        print(f"Recording {samples_per_gesture} windows for '{name}'...")
        count = 0

        while count < samples_per_gesture:
            line = ser.readline().decode().strip()
            if line and ',' in line:
                values = list(map(int, line.split(',')))
                rows.append(values + [int(label)])
                count += 1

    ser.close()

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\nSession saved to {output_path} — {len(rows)} total windows recorded.")
