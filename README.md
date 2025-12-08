# simplefem: a simple example finite element library.

simplefem is a very simple finite element definition library, that is used as an example
in the DefElement documentation.

## Using simplefem

simplefem can be used to tabulate basis functions of Lagrange elements on triangles. To
create a Lagrange element on a triangle, the function `simplefem.lagrange_element` can
be used. For example, the following snippet creates a degree 3 Lagrange element on a
triangle:

```python
from simplefem import lagrange_element

e = lagrange_element(3)
```

A simplefem element can be tabulated at a set of points using the `tabulate` method. For
example, the following snippet gets the values of the basis functions of a degree 3
Lagrage element at the points (0.3, 0.1) and (1, 0):

```python
from simplefem import lagrange_element
import numpy as np

e = lagrange_element(3)

points = np.array([[0.3, 0.1], [1, 0]])
values = e.tabulate(points)
```

## Conventions
### Reference cell
The reference cell used by simplefem is the triangle with vertices at (-1, 0), (1, 0),
and (0, 1).

### Point ordering
The basis functions in simplefem are all defined using point evaluations. These points
are ordered lexicographically: for example, the points that define a degree 3 element
are arranged like this:

```
      9
     / \
    7   8
   /     \
  4   5   6
 /         \
0---1---2---3
```


## Contributing
As simplefem is a small example library, it aims to only contain the features that
are necessary for the DefElement documentation. We are therefore unlikely to accept
any pull request that adds features to simplefem. If you feel like a feature is
needed, please open an issue and we can discuss it before you work on it.
