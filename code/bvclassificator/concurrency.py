import multiprocessing
import time
import numpy as np


global DIMENSION
DIMENSION = 5


def randomissimo():
    result = []
    for _ in range(100):
        result.append(np.random.rand())

    return result


cunc = True
iterations = 5000

if __name__ == "__main__":
    # # start_time = time.time()

    # # print("Time: {} seconds".format(time.time()-start_time))
    # # for r in result:
    # #     print(r)

    # input_matrix = np.random.randint(0, 2, (5, 5))  # Matrice casuale di 0 e 1

    # # Numero di iterazioni
    # num_iterations = 100

    # # Numero di processi paralleli
    # num_processes = multiprocessing.cpu_count()

    # # Calcola il numero di 1 in parallelo
    # total_ones = parallel_count_ones(input_matrix, num_iterations, num_processes)

    # print(f"Numero totale di 1 trovati in {num_iterations} iterazioni: {total_ones}")

    start_time = time.time()
    # a = []
    # for _ in range(100):
    #     a.append(randomissimo())
    # print(f"Time: {time.time()-start_time}")

    # print(len(a))
    # print(len(a[0]))

    if cunc:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        result = []

        for _ in range(iterations):
            result.append(pool.apply_async(randomissimo, ()))

        a = [r.get() for r in result]
        pool.close()
        pool.join()
        print(f"Time: {time.time()-start_time}")

    else:
        a = []
        for _ in range(iterations):
            a.append(randomissimo())
        print(f"Time: {time.time()-start_time}")

    print(len(a))
    print(len(a[0]))
