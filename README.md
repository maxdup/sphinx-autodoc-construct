# sphinx-autodoc-construct

A Sphinx extension to automatically document your [Contruct](https://construct.readthedocs.io/en/latest/) structs.

## Installation

```bash
pip3 install sphinx-autodoc-construct
```

## Configuration

```python
# conf.py
extensions = {
    '...',
    'sphinx_autodoc_construct'
}
```

## Usage
```rst
.. automodcon:: my_package.my_module
.. autostruct:: my_package.my_module.my_struct
```

`.. automodcon::` documents all structs found in a module and `.. autostruct::` documents a specific struct.

*And here's the cool part*, once a struct is covered by `.. automodcon::` or `.. autostruct::`, you can link to it in your rst

```rst
:any:`link<my_struct>` is displayed as a hyperlink.
:ref:`mystruct<my_struct>` is displayed as a reference.
```

References can even be used in your commented struct fields

```python
from construct import *

my_struct = Struct(
    'id': Int32ul,
    'otherId': Int32ul * "index refering to :ref:`other_struct`",
)
```

