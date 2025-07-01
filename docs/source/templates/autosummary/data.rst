{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. py:data:: {{ objname }}

   {% if objname == "SURFACE_EMG__TENSOR" %}
      **Alias of:** ``Annotated[npt.NDArray[np.floating], Is[lambda x: x.ndim == 5]]``
   {% elif objname == "INPUT_CURRENT__MATRIX" %}
      **Alias of:** ``Annotated[npt.NDArray[np.floating], Is[lambda x: x.ndim == 2]]``
   {% elif objname == "SPIKE_TRAIN__MATRIX" %}
      **Alias of:** ``Annotated[npt.NDArray[np.bool_], Is[lambda x: x.ndim == 3]]``
   {% elif objname == "MUAP_SHAPE__TENSOR" %}
      **Alias of:** ``Annotated[npt.NDArray[np.floating], Is[lambda x: x.ndim == 5]]``
   {% else %}
      **Alias of:** *(see source code)*
   {% endif %}

   .. rubric:: Type Definition

   This type alias is defined using beartype validators::

       from typing import Annotated
       import numpy.typing as npt
       from beartype.vale import Is
       
       {{ objname }} = Annotated[
           npt.NDArray[np.floating],  # or np.bool_ for boolean arrays
           Is[lambda x: x.ndim == N], # where N is the required dimensions
       ]

   .. rubric:: Runtime Type Checking

   Use with beartype for automatic validation::

       from {{ module }} import {{ objname }}
       from beartype import beartype
       
       @beartype
       def process_{{ objname.lower().split('_')[0] }}_data(data: {{ objname }}) -> {{ objname }}:
           """Process data with automatic shape validation."""
           # beartype automatically validates array dimensions
           return data

   .. tip::
      🐻 **Beartype Integration**: This type uses `beartype <https://github.com/beartype/beartype>`_ validators to ensure arrays have the correct number of dimensions at runtime.

   .. note::
      📐 **Array Validation**: The `Is[lambda x: x.ndim == N]` validator automatically checks that your NumPy arrays have the expected shape for MyoGen operations. 