import timeit

def compute_task():
    # Replace this with the task or computation you want to measure
    result = sum(range(1, 1000000))
    return result

def measure_computational_power():
    # Number of repetitions for the task
    repetitions = 100

    # Measure time on the current machine
    local_time = timeit.timeit(compute_task, number=repetitions)

    print(f"Local machine execution time: {local_time:.5f} seconds (average over {repetitions} repetitions)")

if __name__ == "__main__":
    measure_computational_power()
