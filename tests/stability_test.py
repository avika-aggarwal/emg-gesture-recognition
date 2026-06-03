import serial
import time
import argparse

GESTURE_NAMES = {0: 'rest', 1: 'open', 2: 'close', 3: 'pinch'}

def run_stability_test(port, baud=9600, num_cycles=100, cycle_delay=0.3, verbose=False):
    print(f"\n=== Stability Test ===")
    print(f"Port: {port} | Cycles: {num_cycles} | Delay: {cycle_delay}s\n")

    try:
        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)
    except serial.SerialException as e:
        print(f"Could not open serial port: {e}")
        return None

    results = {"pass": 0, "fail": 0, "timeouts": 0,
               "latencies_ms": [], "failures": []}

    for i in range(num_cycles):
        gesture_id = i % 4
        ser.write(bytes([gesture_id]))
        t_start  = time.time()
        response = ser.readline().decode(errors='replace').strip()
        latency  = (time.time() - t_start) * 1000

        if response == f"ACK:{gesture_id}":
            results["pass"] += 1
            results["latencies_ms"].append(latency)
            if verbose:
                print(f"  [{i+1:03d}] {GESTURE_NAMES[gesture_id]:<6} OK ({latency:.1f}ms)")
        elif not response:
            results["fail"] += 1
            results["timeouts"] += 1
            results["failures"].append({"cycle": i, "gesture": gesture_id, "reason": "timeout"})
        else:
            results["fail"] += 1
            results["failures"].append({"cycle": i, "gesture": gesture_id, "reason": response})

        time.sleep(cycle_delay)

    ser.close()
    total = results["pass"] + results["fail"]
    results["pass_rate"]      = results["pass"] / total if total else 0
    results["avg_latency_ms"] = (sum(results["latencies_ms"]) /
                                  len(results["latencies_ms"])
                                  if results["latencies_ms"] else 0)

    print(f"\nPass rate: {results['pass_rate']*100:.1f}% | "
          f"Avg latency: {results['avg_latency_ms']:.1f}ms")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",    default="/dev/ttyACM0")
    parser.add_argument("--cycles",  type=int,   default=100)
    parser.add_argument("--delay",   type=float, default=0.3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_stability_test(args.port, num_cycles=args.cycles,
                       cycle_delay=args.delay, verbose=args.verbose)
