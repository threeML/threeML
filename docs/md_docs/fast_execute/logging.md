---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# Logging and Verbosity

We use the [`logging`](https://docs.python.org/3/library/logging.html) library to manage all logs. 
The `logging` needs to be configured by the application using 3ML --e.g. a script, an interactive notebook, etc-- 
for example:

```python
# myapp.py
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
```

See the `logging` library documentation for further details.

For convenience, 3ML provides a custom logging configuration, which
can be activated with:

```python
# myapp.py
import logging
from threeML.io.logging import setup_logger
logger = setup_logger(__name__)
```

The logging behavior can be further configured with the instructions
below. Note that these 3ML-specific logging configuration only 
take effect if your logger was instantiated using 3ML's `setup_logger`.

NOTE: Until recently, 3ML configured its own logger by default. 
Now you need to call `setup_logger` if you want to recover the
previous behavior.

## Logging handlers

Logging in 3ML occurs in three different ways:

* to the console (jupyter or command line)
* a user log file
* a debug log file

The first two files record information relvant to the average 3ML user. The debug file records low level information useful when trying to solve runtime errors. 

## Configuration

The logging of 3ML is configurable in the configuration file.

```python
from threeML import threeML_config

threeML_config["logging"]
```

First, **the location of the log files can be set** before start a 3ML session in python. By default, logging to the debug file is off (**developer** switch). The **console** and **usr** switches can also be disabled by default to completely silence 3ML. Additionally, the default logging level of 3ML can be set for both the usr and console logs.

## Startup warnings

3ML checks the availability of various plugins and external libraries
during the library initialization. While these warnings are usual benign,
if it is important for your application to record these warning 
you can log them at any point by calling

```python
from threeML.io.logging import log_threeml_startup_warnings
from astromodels.utils.logging import log_astromodels_startup_warnings
log_astromodels_startup_warnings(logger)
log_threeml_startup_warnings(logger)
```

Here, `logger` can be an arbitrary `Logger`, whether it was configured by
3ML or not.

NOTE: until recently, these warnings were logged and printed by default.
Now you need to call the function above explicitely.

## Logging controls and verbosity

During a 3ML session, it may be diserable to toggle the state of the logger for certain actions. While it is possible to access the logger and handlers from ```bash threeml.io.logging``` and following the [standard practices](https://docs.python.org/3/howto/logging.html) of python logging, we provide some simple utilites to toggle the state the logger.


```python
from threeML import silence_logs, silence_warnings, activate_logs, activate_warnings, update_logging_level
import logging
log = logging.getLogger("threeML")
log.info("Hello there")
```

If we want to shut off all logging for a few operations, we can call ```python silence_logs()``` which diable all logs 

```python
log.info("Now you see me")

# logging will be silenced
silence_logs()

log.info("Now you don't")

# now we can restore the LAST state of the logs 
# before we silenced them
activate_logs()

log.info("I'm back")
```

3ML has a lot of useful warning information. Sometimes we just want to get on with out analysis and now have to see this. ```python silence_warnings()``` removes warnings while maintained all other logs

```python
log.info("I am tired of")
log.warning("being warned")

# silence warnings temporarily
silence_warnings()
log.info("so I will turn off")
log.warning("all the IMPORTANT warning info")

# and bring them back
activate_warnings()
log.warning("I hope your computer didn't crash in the mean time")

```

We can also control the level of the console log:

```python
update_logging_level("DEBUG")

log.debug("Now we can see low level logging")

update_logging_level("CRITICAL")

log.info("This is invisible")

```

## Progress bars

Some analyses take a while and it is nice to see that something is happening, or at least point to the screen when our boss walks by and we are on twitter to show that we are still working. Thus, 3ML reports the progress of various tasks with progress bars. 

These are configurable from the configuration file.


```python
threeML_config["interface"]
```

We see that it is possible to show or not show progress for all of 3ML. However, it is also possible to toggle the progress bars during a session

```python
from threeML import silence_progress_bars, activate_progress_bars, toggle_progress_bars
from threeML.utils.progress_bar import trange
```

```python
activate_progress_bars()

for i in trange(10, desc="progressive"):
    pass

# this will turn of progress for all of 3ML
silence_progress_bars()

for i in trange(10, desc="conservative"):
    pass

# toggling the progress bar simply switches the state
toggle_progress_bars()
for i in trange(10, desc="more progress"):
    pass

```

## Loud and Quiet

If you want to silence everything or turn everything on, there are toggles for this:

```python
from threeML import quiet_mode, loud_mode
```

```python
quiet_mode()

for i in trange(10, desc="invisible"):
    pass

log.info("you will not see me")

loud_mode()

for i in trange(10, desc="visible"):
    pass

log.info("here I am")
```
