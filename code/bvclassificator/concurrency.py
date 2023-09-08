import multiprocessing
import time
import concurrent.futures

start = time.perf_counter()


def do_something(sec: tuple):
    a, b = sec
    print(a)
    print(b)
    return sec[0]*sec[1]


if __name__ == "__main__":

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # results = [executor.submit(do_something, 1) for _ in range(10)]
        # for f in concurrent.futures.as_completed(results):
        #    print(f.result())

        # possiamo usare executor.map per avere i risultati in ordine
        results = executor.map(do_something, [(1,2), (3,4)])
        for r in results:
            print(r)

    # process = []
    # prova = []

    # for _ in range(10):
    #     p = multiprocessing.Process(target=do_something, args = [1.5])
    #     p.start()
    #     process.append(p)

    # for p in process:
    #     prova.append(p.join())

    finish = time.perf_counter()
    print(f"Finished in {round(finish-start,2)} second(s)")
