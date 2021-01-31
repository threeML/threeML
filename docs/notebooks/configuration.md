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

# Configuration

The 3ML configuration sets up defaults that are read in at runtime and is a session wide state that the user can modify at any point during an analysis to alter behavior throughout 3ML.

The configuration is a hierarchical you can import and modify.

```python
from threeML import threeML_config, show_configuration

show_configuration()
```

## Local configuration


You can store a local configuration read in at the beginning of every session. By default, 3ML looking in ``` ~/.config/threeml/*.yml``` for configuration files.

* You can set your own location with the environment variable ```THREEML_CONFIG``` 
* All YAML files in this directiory are considered to be configuration files.

Multiple configuration files can be added allowing you to have readable seperation between different topics.

E.g, here are two configuration files that modify some plotting defaults and logging defaults in separate files.

```~/.config/threeml/plotting.yml```
```yaml
plugins:
  ogip:
    fit_plot:
      model_cmap: Spectral
      data_color: grey
      background_color: k
      background_mpl_kwargs:
        lw: .8
        ls: "-"


```

```~/.config/threeml/logging.yml```
```yaml
logging:
  developer: on
  startup_warnings: off
```

```python

```

Editing the configuration for your own personal analysis style is can be acheived with any standard text editor except MicroSoft Word. 


But perhaps you want to change a parameter in the middle of a session? The configuration behaves like a nested dictionary. Let's take a look at what we can configure:


Perhaps we want the default for fitting Fermi LAT LLE backgrounds to be unbinned just for this session.

```python
threeML_config["lle"]
```

```python
threeML_config["lle"]["background unbinned"] = True
threeML_config["lle"]
```

From this point on in this session, fitting LLE backgrounds will be unbinned. When we close python, the next time we start 3ML, the values set in the configuration file be loaded.

Do not worry about entering incorrect values in the file as 3ML checks both the structure and types of the parameters. If you corrupt your file, 3ML will save a copy of it and restore the default so that you can still use 3ML. 

If you have a defualt configuration you would like to add, consider opening a pull request. We would love to hear your ideas. 

```python

```
