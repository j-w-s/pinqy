#     ___________._____.____         .__
#     \__    ___/|__\_ |__  |   ____ |  |__
#       |    |   |  || __ \|  | _/ ___\|  |  \
#       |    |   |  || \_\ \  |_\  \___|   Y  \
#       |____|   |__||___  /____/\___  >___|  /
#                       \/          \/     \/

"""
vecqy: a fluent, numpy-powered vector and matrix library.

philosophy:
- immutability: operations always return a new instance, enabling a pure functional style.
- fluency: methods are chainable (e.g., `v.add(u).normalize().scale(2)`).
- performance: all core computations are delegated to numpy for c-level speed.
- ergonomics: flexible input types and operator overloading for intuitive use.
"""

from __future__ import annotations
import numpy as np
import math
from abc import ABC, abstractmethod
from typing import (
    TypeVar, Generic, Type, Union, Iterable, Tuple, cast
)

# --- type definitions for clarity and type-safety ---

# generic type for the concrete class (e.g., Vec2, Vec3)
t = TypeVar('t', bound='_BaseVec')

# flexible input types for vector-like data
vectorlike = Union[t, 'tuple', 'list', np.ndarray]

# --- abstract base class for all vectors ---

class _BaseVec(ABC, Generic[t]):
    """
    the abstract base for all vector types.
    it encapsulates a numpy array and provides the core, dimension-agnostic, fluent api.
    """
    __slots__ = ('_v',) # memory optimization: prevents creation of __dict__

    def __init__(self, data: Iterable[float], expected_dim: int):
        # the core of every vector is a numpy array. this is the source of all performance.
        self._v = np.array(data, dtype=np.float64)
        if self._v.shape != (expected_dim,):
            raise ValueError(f"expected {expected_dim} dimensions, got {self._v.size}")

    @classmethod
    def _from_np(cls: Type[t], arr: np.ndarray) -> t:
        """internal factory to create a vector from a numpy array without re-validation."""
        # this is an optimization to avoid redundant checks in internal calls.
        instance = cls.__new__(cls)
        instance._v = arr
        return instance

    @staticmethod
    def _to_np(other: vectorlike) -> np.ndarray:
        """internal helper to safely convert any vectorlike input into a numpy array."""
        if isinstance(other, _BaseVec):
            return other._v
        return np.array(other, dtype=np.float64)

    # --- fluent api methods (dimension-agnostic) ---

    def add(self: t, other: vectorlike) -> t:
        """returns a new vector that is the sum of this vector and another."""
        return self.__class__._from_np(self._v + self._to_np(other))

    def sub(self: t, other: vectorlike) -> t:
        """returns a new vector that is the difference of this vector and another."""
        return self.__class__._from_np(self._v - self._to_np(other))

    def scale(self: t, scalar: float) -> t:
        """returns a new vector scaled by a scalar value."""
        return self.__class__._from_np(self._v * scalar)

    def negate(self: t) -> t:
        """returns a new vector with all components negated."""
        return self.__class__._from_np(-self._v)

    def component_mul(self: t, other: vectorlike) -> t:
        """returns a new vector from component-wise (hadamard) product."""
        return self.__class__._from_np(self._v * self._to_np(other))

    def normalize(self: t) -> t:
        """returns a new unit vector (magnitude 1). returns a zero vector if magnitude is zero."""
        mag = self.magnitude()
        if mag == 0:
            return self.__class__.zero()
        return self.scale(1.0 / mag)

    def clamp_magnitude(self: t, max_length: float) -> t:
        """returns a new vector with its magnitude clamped to a maximum value."""
        mag_sq = self.magnitude_sq()
        if mag_sq > max_length * max_length:
            return self.normalize().scale(max_length)
        return self

    def lerp(self: t, other: t, alpha: float) -> t:
        """returns a new vector linearly interpolated between this vector and another."""
        return self.scale(1.0 - alpha).add(other.scale(alpha))

    def reflect(self: t, normal: t) -> t:
        """returns a new vector reflected across a plane defined by a normal vector."""
        # formula: r = v - 2 * (v . n) * n
        dot_product = self.dot(normal)
        return self.sub(normal.scale(2 * dot_product))

    def project(self: t, onto: t) -> t:
        """returns a new vector by projecting this vector onto another."""
        dot_product = self.dot(onto)
        mag_sq = onto.magnitude_sq()
        if mag_sq == 0:
            return self.__class__.zero()
        return onto.scale(dot_product / mag_sq)

    # --- terminal methods (return a scalar value) ---

    def dot(self, other: vectorlike) -> float:
        """calculates the dot product between this vector and another."""
        return float(np.dot(self._v, self._to_np(other)))

    def magnitude(self) -> float:
        """calculates the magnitude (length, or l2 norm) of the vector."""
        return float(np.linalg.norm(self._v))

    def magnitude_sq(self) -> float:
        """calculates the squared magnitude. faster than magnitude() as it avoids a sqrt."""
        return self.dot(self)

    def distance_to(self, other: vectorlike) -> float:
        """calculates the euclidean distance to another vector."""
        return float(np.linalg.norm(self._v - self._to_np(other)))

    def distance_sq(self, other: vectorlike) -> float:
        """calculates the squared euclidean distance to another vector. faster."""
        diff = self._v - self._to_np(other)
        return float(np.dot(diff, diff))

    def angle_between(self, other: t) -> float:
        """calculates the angle in radians between this vector and another."""
        dot_product = self.dot(other)
        mag_product = self.magnitude() * other.magnitude()
        if mag_product == 0:
            return 0.0
        # clamp to avoid floating point errors with acos
        cos_theta = np.clip(dot_product / mag_product, -1.0, 1.0)
        return float(np.arccos(cos_theta))

    # --- dunder methods for pythonic integration ---

    def __repr__(self) -> str:
        # produces a clean, readable representation
        coords = ", ".join(f"{c:.4f}" for c in self._v)
        return f"{self.__class__.__name__}({coords})"

    def __len__(self) -> int: return self._v.size
    def __getitem__(self, key: int) -> float: return self._v[key]
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__): return NotImplemented
        return np.array_equal(self._v, other._v)

    def __add__(self: t, other: vectorlike) -> t: return self.add(other)
    def __sub__(self: t, other: vectorlike) -> t: return self.sub(other)
    def __mul__(self: t, scalar: float) -> t: return self.scale(scalar)
    def __rmul__(self: t, scalar: float) -> t: return self.scale(scalar)
    def __truediv__(self: t, scalar: float) -> t: return self.scale(1.0 / scalar)
    def __neg__(self: t) -> t: return self.negate()

    # --- abstract properties and factories for subclasses ---

    @classmethod
    @abstractmethod
    def zero(cls: Type[t]) -> t: pass

# --- concrete vector implementations ---

class Vec2(_BaseVec['Vec2']):
    """a 2-dimensional vector."""
    __slots__ = () # inherits _v, no new slots needed

    def __init__(self, x: float, y: float):
        super().__init__((x, y), expected_dim=2)

    @property
    def x(self) -> float: return self._v[0]
    @property
    def y(self) -> float: return self._v[1]

    @classmethod
    def zero(cls) -> 'Vec2': return cls(0.0, 0.0)
    @classmethod
    def one(cls) -> 'Vec2': return cls(1.0, 1.0)
    @classmethod
    def unit_x(cls) -> 'Vec2': return cls(1.0, 0.0)
    @classmethod
    def unit_y(cls) -> 'Vec2': return cls(0.0, 1.0)

class Vec3(_BaseVec['Vec3']):
    """a 3-dimensional vector."""
    __slots__ = ()

    def __init__(self, x: float, y: float, z: float):
        super().__init__((x, y, z), expected_dim=3)

    @property
    def x(self) -> float: return self._v[0]
    @property
    def y(self) -> float: return self._v[1]
    @property
    def z(self) -> float: return self._v[2]

    def cross(self, other: vectorlike) -> 'Vec3':
        """returns a new vector from the cross product."""
        return Vec3._from_np(np.cross(self._v, self._to_np(other)))

    @classmethod
    def zero(cls) -> 'Vec3': return cls(0.0, 0.0, 0.0)
    @classmethod
    def one(cls) -> 'Vec3': return cls(1.0, 1.0, 1.0)
    @classmethod
    def unit_x(cls) -> 'Vec3': return cls(1.0, 0.0, 0.0)
    @classmethod
    def unit_y(cls) -> 'Vec3': return cls(0.0, 1.0, 0.0)
    @classmethod
    def unit_z(cls) -> 'Vec3': return cls(0.0, 0.0, 1.0)

# --- matrix implementation (4x4) ---

class Mat4:
    """
    an immutable, column-major 4x4 matrix for 3d transformations.
    designed for a fluent api. operations are composed via matrix multiplication.
    note: transformations are applied right-to-left. `m.translate(v).rotate(a)`
    means the rotation is applied to the object first, then the translation.
    """
    __slots__ = ('_m',)

    def __init__(self, data: Iterable[Iterable[float]]):
        # the matrix is stored as a numpy array, the source of all performance.
        self._m = np.array(data, dtype=np.float64)
        if self._m.shape != (4, 4):
            raise ValueError(f"expected a 4x4 matrix, got {self._m.shape}")

    @classmethod
    def _from_np(cls, arr: np.ndarray) -> 'Mat4':
        """internal factory for performance."""
        instance = cls.__new__(cls)
        instance._m = arr
        return instance

    # --- core factory methods ---

    @classmethod
    def identity(cls) -> 'Mat4':
        return cls._from_np(np.identity(4))

    @classmethod
    def from_translation(cls, v: vectorlike) -> 'Mat4':
        m = np.identity(4)
        m[0:3, 3] = _BaseVec._to_np(v)
        return cls._from_np(m)

    @classmethod
    def from_scale(cls, v: vectorlike) -> 'Mat4':
        s = _BaseVec._to_np(v)
        return cls._from_np(np.diag([s[0], s[1], s[2], 1.0]))

    @classmethod
    def from_rotation_x(cls, angle_rad: float) -> 'Mat4':
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return cls([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

    @classmethod
    def from_rotation_y(cls, angle_rad: float) -> 'Mat4':
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return cls([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

    @classmethod
    def from_rotation_z(cls, angle_rad: float) -> 'Mat4':
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return cls([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    @classmethod
    def look_at(cls, eye: Vec3, target: Vec3, up: Vec3) -> 'Mat4':
        """creates a view matrix."""
        f = (target - eye).normalize()
        s = f.cross(up).normalize()
        u = s.cross(f)
        m = np.identity(4)
        m[0, 0:3] = s._v
        m[1, 0:3] = u._v
        m[2, 0:3] = -f._v
        m[0:3, 3] = -np.array([s.dot(eye), u.dot(eye), -f.dot(eye)])
        return cls._from_np(m)

    @classmethod
    def perspective(cls, fov_y_rad: float, aspect: float, near: float, far: float) -> 'Mat4':
        """creates a perspective projection matrix."""
        t = math.tan(fov_y_rad / 2.0)
        m = np.zeros((4, 4))
        m[0, 0] = 1.0 / (aspect * t)
        m[1, 1] = 1.0 / t
        m[2, 2] = -(far + near) / (far - near)
        m[2, 3] = -2.0 * far * near / (far - near)
        m[3, 2] = -1.0
        return cls._from_np(m)

    # --- fluent api methods ---

    def translate(self, v: vectorlike) -> 'Mat4':
        """returns a new matrix post-multiplied by a translation matrix."""
        return self @ self.from_translation(v)

    def scale(self, v: vectorlike) -> 'Mat4':
        """returns a new matrix post-multiplied by a scale matrix."""
        return self @ self.from_scale(v)

    def rotate_x(self, angle_rad: float) -> 'Mat4':
        """returns a new matrix post-multiplied by an x-axis rotation matrix."""
        return self @ self.from_rotation_x(angle_rad)

    def rotate_y(self, angle_rad: float) -> 'Mat4':
        """returns a new matrix post-multiplied by a y-axis rotation matrix."""
        return self @ self.from_rotation_y(angle_rad)

    def rotate_z(self, angle_rad: float) -> 'Mat4':
        """returns a new matrix post-multiplied by a z-axis rotation matrix."""
        return self @ self.from_rotation_z(angle_rad)

    # --- terminal/utility methods ---

    def inverse(self) -> 'Mat4':
        """returns the inverse of the matrix. raises an exception if not invertible."""
        try:
            return Mat4._from_np(np.linalg.inv(self._m))
        except np.linalg.LinAlgError:
            raise ValueError("matrix is singular and cannot be inverted.")

    def transpose(self) -> 'Mat4':
        """returns the transpose of the matrix."""
        return Mat4._from_np(self._m.T)

    def transform_point(self, p: Vec3) -> Vec3:
        """transforms a 3d point (w=1)."""
        # add homogeneous coordinate, transform, then divide by w
        p_homogeneous = np.array([p.x, p.y, p.z, 1.0])
        res_homogeneous = self._m @ p_homogeneous
        w = res_homogeneous[3]
        if w != 0 and w != 1:
            return Vec3._from_np(res_homogeneous[0:3] / w)
        return Vec3._from_np(res_homogeneous[0:3])

    def transform_direction(self, d: Vec3) -> Vec3:
        """transforms a 3d direction vector (w=0). ignores translation."""
        d_homogeneous = np.array([d.x, d.y, d.z, 0.0])
        res_homogeneous = self._m @ d_homogeneous
        return Vec3._from_np(res_homogeneous[0:3])

    # --- dunder methods ---
    def __repr__(self) -> str:
        # produces a multi-line, aligned representation for readability
        rows = ["  [" + ", ".join(f"{c:8.3f}" for c in row) + "]" for row in self._m]
        return f"{self.__class__.__name__}(\n" + ",\n".join(rows) + "\n)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mat4): return NotImplemented
        return np.allclose(self._m, other._m)

    def __matmul__(self, other: 'Mat4') -> 'Mat4':
        """matrix-matrix multiplication using the '@' operator."""
        return Mat4._from_np(self._m @ other._m)

    def __mul__(self, v: Vec3) -> Vec3:
        """matrix-vector transformation using the '*' operator."""
        return self.transform_point(v)