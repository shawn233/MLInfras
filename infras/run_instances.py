import os
import logging
import time
import threading
import subprocess
from typing import List
from queue import Queue

class InstanceSupervisor(threading.Thread):

    def __init__(
        self, 
        t_id: int, 
        q: Queue,
        timeout_in_secs: int = None
    ):
        super(InstanceSupervisor, self).__init__()
        self.t_id = t_id
        self.q = q
        self.timeout = timeout_in_secs

    def run(self):
        while not self.q.empty():
            # check for existed results [not active due to display errors]
            # if (os.path.exists(f"./instances/{self.current_ins}/log.txt")):
            #     rerun = ""
            #     while rerun == "":
            #         rerun = input(
            #             f">> Instance {self.current_ins} seems already executed. Still run? [y(es)/n(o)] ")
            #     rerun = rerun.lower()
            #     if rerun == "n" or rerun == "no":
            #         logging.info(f"[thread {self.t_id}] Instance {self.current_ins} is skipped")
            #         self.current_ins += self.n_threads
            #         continue
            current_ins: int = self.q.get()
            
            ins_path = rf"./instances/{current_ins}/poc.out"
            while (not os.path.exists(ins_path)):
                # wait one minute
                logging.info(f"[thread {self.t_id}] Instance {current_ins} not found. Wait one minute")
                time.sleep(60)
            
            # run this instance
            logging.info(f"[thread {self.t_id}] Instance {current_ins} starts running ...")
            start_time = time.time()

            try:
                subprocess.run(["make", f"INSTANCE_ID={current_ins}", "run_instance"], timeout=self.timeout)
            except subprocess.TimeoutExpired as e:
                logging.warning(f"[thread {self.t_id}] Instance {current_ins} times out!")
                with open(f"./instances/{current_ins}/log.txt", "a") as f:
                    f.write(f"*timeout\n")

            duration = time.time() - start_time
            logging.info(f"[thread {self.t_id}] Instance {current_ins} "
                         f"finishes in {duration:.2f} secs ({duration / 60.:.2f} mins)!")


def main():
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s", 
        datefmt="%Y/%m/%d %H:%M:%S")

    start_ins = 0           # first instance to process
    max_ins = 200           # maximum number of instances
    n_threads = 4           # because I allocated four processors for this VM
    timeout = 80 * 60       # 80 mins in seconds

    q = Queue()
    for i in range(start_ins, max_ins): # [start_ins, max_ins)
        q.put(i)

    thread_pool: List[InstanceSupervisor] = []
    for t_id in range(n_threads):
        thread = InstanceSupervisor(t_id, q, timeout)
        thread.start()
        thread_pool.append(thread)

    for thread in thread_pool:
        thread.join()


if __name__ == "__main__":
    main()