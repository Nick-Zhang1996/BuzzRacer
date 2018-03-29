from time import time

class execution_timer:
    
    def __init__(self, enable = False):
        self.enabled = enable
        # sectional time start time
        self.s_start = {}
        # average runtime, this is updated when a global section ends
        self.s_avg = {}
        # cumulative time consumption in one global section
        self.cul = {}
        
        # global section count, this will be used as exe count for all sections
        self.g_count = 0

        # repetition in one global scope
        self.s_rep_count = {}

        # global time counting
        self.g_start = None
        self.g_end = None
        self.g_duration_avg = None
        self.g_sample_count = 0
        # tracked variables
        self.tracked = {}
        self.tracked_count = {}


    def global_start(self):
        if not self.enabled:
            return
        self.g_start = time()
        return

    def global_end(self):
        if not self.enabled:
            return
        self.g_end = time()
        duration = self.g_end-self.g_start

        if (self.g_duration_avg is None):
            self.g_duration_avg = duration
            self.g_sample_count = 1
        else:
            self.g_duration_avg = self.g_duration_avg*self.g_sample_count+duration
            self.g_sample_count = self.g_sample_count +1
            self.g_duration_avg = self.g_duration_avg/self.g_sample_count

        for key,value in self.cul.items():
            if key in self.s_avg:
                self.s_avg[key] = self.s_avg[key]*self.g_count+value
                self.s_avg[key] = self.s_avg[key] / (self.g_count+1)
            else:
                self.s_avg[key] = value
        self.cul = {}
        self.g_count+=1
        return

        return duration
        
    def track(self, name, var):
        if not self.enabled:
            return
        if name in self.tracked:
            self.tracked[name] = self.tracked[name]*self.tracked_count[name]+var
            self.tracked_count[name] = self.tracked_count[name] + 1
            self.tracked[name] = self.tracked[name] / self.tracked_count[name]
        else:
            self.tracked[name] = var
            self.tracked_count[name] = 1
        return
        

    def start(self, name = None):
        if not self.enabled:
            return
        if name is None:
            return self.global_start()

        self.s_start[name] = time()
        return

    def end(self, name = None):
        if not self.enabled:
            return
        if name is None:
            return self.global_end()

        duration = time()-self.s_start[name] 

        if name in self.cul:
            self.cul[name] += duration
            self.s_rep_count[name] += 1
        else:
            self.cul[name] = duration
            self.s_rep_count[name] = 1

        return duration

    def s(self, n=None):
        return self.start(n)
        
    def e(self, n=None):
        return self.end(n)
        
    def summary(self):
        if not self.enabled:
            return
        # tracked variables
        print('-----Variables--------')
        for key,value in self.tracked.items():
            print(key+'\t\t'+str(value))
        print('-------Time-----------')
        #tracked times
        #note: sum_time is sum of all fractions not global time
        sum_time = sum(self.s_avg.values())
        #g_duration_avg is time between start() and end() averaged
        total_time = self.g_duration_avg
        #make sure we don't mess with the original copy
        fraction = dict(self.s_avg)
        fraction.update((x, y/total_time) for x, y in fraction.items()) 
        for key,value in fraction.items():
            print(key+'\t\t'+ "{0:.1f}".format(value*100)+' %')

        unaccounted_time = 1-sum_time/total_time
        print('avg frequency = '+"{0:.3f}".format(1/self.g_duration_avg)+'Hz')
        print('unaccounted time = '+"{0:.1f}".format(unaccounted_time*100)+' %')
        return

#sample usage        
if __name__ == '__main__':
    from time import sleep
    #Create an instance of exe_timer for all procedures you want to monitor
    # Initialize with True to enable all functions
    # When you're done analyzing, simply change the argument to False or 
    # initialize without an argument, that will cause all methods to return instantly
    t = execution_timer(True)

    # A typical scenario is to find average execution
    # time during several iterations, average exe time will be 
    # updated during each iteration
    for i in range(1,3):
        # Start global timer in the very beginning of the procedure
        t.s()

        # To track an operation, enclose it with INSTANCE.s('identifier')
        # and INSTANCE.e('identifier')
        # s and e are shorthand for start and end
        # each start() must be matched with an end() with identical identifier
        t.s('sleep2')
        sleep(0.2)
        t.e('sleep2')

        for j in range(1,3):
            t.s('sleep1')
            sleep(0.1)
            t.e('sleep1')

        # not all operations in your procedure will be timed, those not timed are called
        # unaccounted time
        sleep(0.1)

        # it is also possible to track average value of a variable, this is how you do it.
        t.track('var', 5)

        # at the end of the operation, end the global timer with a matching e()
        t.e()
    # this function prints a summary of everything.
    t.summary()
        
        
