from __future__ import print_function

import sys, time

class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 50
        self.startTime = time.time()
        self.lastIter = 0
        self.__update_amount(0)

    def animate(self, iter):
        print('\r', self, end='')
        sys.stdout.flush()
        self.lastIter = iter
        self.update_iteration(iter + 1)
    
    def increase(self):
        
        self.animate(self.lastIter + 1)
    
    def update_iteration(self, elapsed_iter):
        elapsed_iter = min(elapsed_iter, self.iterations)
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s completed in %.1f s' % (elapsed_iter, self.iterations, time.time() - self.startTime)
        #sys.stdout.flush()

    def __update_amount(self, new_amount):
        percent_done = min(int(round((new_amount / 100.0) * 100.0)), 100)
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)
