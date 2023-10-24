from functools import wraps
import time

def timer_decorator(func):
    @wraps(func)
    def wrapper( *args, **kwargs):
        start_time = time.time()
        result = func( *args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__} executed in {end_time - start_time:.5f} seconds')
        return result

    return wrapper

@timer_decorator
def slow_function(secs=2, first='first', second='second'):
    time.sleep(secs)
    print(f'Slow function executed, secs is {secs}, second is {second}')
    return 'OK'

rst = slow_function(5)
print(f'result is {rst}')