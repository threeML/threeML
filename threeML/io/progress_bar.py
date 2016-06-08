from __future__ import print_function

import sys, time
import datetime


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

        try:

            print('\r', self, end='')
            sys.stdout.flush()
            self.lastIter = iter
            self.update_iteration(iter + 1)

        except:
            # Do not crash in any case. This isn't an important operation
            pass
    
    def increase(self):
        
        self.animate(self.lastIter + 1)

    def _check_remaining_time(self, delta_t):

        # Seconds per iterations
        s_per_iter = delta_t / float(self.lastIter)

        # Seconds to go (estimate)
        s_to_go = s_per_iter * (self.iterations - self.lastIter)

        # I cast to int so it won't show decimal seconds

        return str(datetime.timedelta(seconds=int(s_to_go)))

    def update_iteration(self, elapsed_iter):

        delta_t = time.time() - self.startTime

        elapsed_iter = min(elapsed_iter, self.iterations)

        if elapsed_iter < self.iterations:

            self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
            self.prog_bar += '  %d / %s in %.1f s' % (elapsed_iter, self.iterations, delta_t)
            self.prog_bar += ' (%s remaining)' % self._check_remaining_time(delta_t)

        else:

            self.__update_amount(100)
            self.prog_bar += '  completed in %.1f s' % (time.time() - self.startTime)

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
