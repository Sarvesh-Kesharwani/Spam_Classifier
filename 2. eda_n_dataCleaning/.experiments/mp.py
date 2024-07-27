import time
from multiprocessing import Pool
import os

lst = [10, 15, 20, 25, 30, 35]
result = []

def fibonacci(n):
    # prints the id of process where the current instance of this function
    # is being executed currently
    # print(n, os.getpid())
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def run():
    # using multiprocessing
    start_time = time.time()

    pool = Pool(processes=8)
    result = pool.map(fibonacci, lst)
    end_time = time.time()
    time_taken = end_time - start_time
    print(result)
    print(time_taken)

if __name__ == '__main__':
    run()

# 2.6337029933929443 -- with no worker_count defined
# 2.619873523712158 -- with 5 workers
# 2.604215383529663 -- with 8 (max) workers