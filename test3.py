#!/usr/bin/env python3

import pygame
import time
import math
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from pinqy import from_iterable as P, from_range
from vecqu import Vec3, Mat4


# --- minimal data structures ---

@dataclass(frozen=True)
class Vertex:
    pos: Vec3
    color: Tuple[int, int, int] = (255, 255, 255)


@dataclass(frozen=True)
class Triangle:
    v0: Vertex
    v1: Vertex
    v2: Vertex


@dataclass(frozen=True)
class FrameMetrics:
    frame_time: float
    triangles_rendered: int
    triangles_culled: int
    fps: float


class FastEngine:
    """stripped-down, performance-focused engine"""

    def __init__(self, width: int = 800, height: int = 600):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("fast functional 3d engine")
        self.width, self.height = width, height
        self.clock = pygame.time.Clock()

        # minimal state
        self.metrics_history = P([])
        self.camera_pos = Vec3(0, 0, 8)
        self.camera_target = Vec3(0, 0, 0)
        self.camera_up = Vec3(0, 1, 0)

        # projection
        self.fov = math.pi / 4
        self.aspect = width / height
        self.near = 0.1
        self.far = 100.0

        # pre-generate scene data once
        self.triangles = self.generate_scene()

    def generate_scene(self) -> List[Triangle]:
        """generate lots of triangles efficiently using numpy + pinqy"""
        triangles = []

        # create a field of cubes using functional generation
        positions = (from_range(0, 25)
                     .select(lambda i: (i % 5 - 2, i // 5 - 2))  # 5x5 grid
                     .select(lambda pos: Vec3(pos[0] * 3.0, 0, pos[1] * 3.0))
                     .to_list())

        # generate cube triangles for each position
        for pos in positions:
            cube_tris = self.make_cube_at(pos, 0.8)
            triangles.extend(cube_tris)

        # add some vertical elements
        for i in range(10):
            y_pos = Vec3(0, i * 2 - 5, -15 + i)
            triangles.extend(self.make_cube_at(y_pos, 0.5))

        print(f"generated {len(triangles)} triangles")
        return triangles

    def make_cube_at(self, center: Vec3, size: float) -> List[Triangle]:
        """generate cube triangles at position - optimized"""
        # cube vertices relative to center
        s = size
        verts = [
            Vertex(center + Vec3(-s, -s, s), (255, 100, 100)),  # 0
            Vertex(center + Vec3(s, -s, s), (100, 255, 100)),  # 1
            Vertex(center + Vec3(s, s, s), (100, 100, 255)),  # 2
            Vertex(center + Vec3(-s, s, s), (255, 255, 100)),  # 3
            Vertex(center + Vec3(-s, -s, -s), (255, 100, 255)),  # 4
            Vertex(center + Vec3(s, -s, -s), (100, 255, 255)),  # 5
            Vertex(center + Vec3(s, s, -s), (200, 200, 200)),  # 6
            Vertex(center + Vec3(-s, s, -s), (255, 200, 100)),  # 7
        ]

        # cube face indices - 12 triangles
        faces = [
            (0, 1, 2), (2, 3, 0),  # front
            (1, 5, 6), (6, 2, 1),  # right
            (5, 4, 7), (7, 6, 5),  # back
            (4, 0, 3), (3, 7, 4),  # left
            (3, 2, 6), (6, 7, 3),  # top
            (4, 5, 1), (1, 0, 4),  # bottom
        ]

        return [Triangle(verts[f[0]], verts[f[1]], verts[f[2]]) for f in faces]

    def fast_project(self, triangles: List[Triangle], mvp: Mat4) -> List[Tuple]:
        """vectorized projection using numpy through vecqu"""
        projected = []

        for tri in triangles:
            try:
                # project all 3 vertices
                p0 = mvp.transform_point(tri.v0.pos)
                p1 = mvp.transform_point(tri.v1.pos)
                p2 = mvp.transform_point(tri.v2.pos)

                # perspective divide and early clip
                if p0.z <= 0 or p1.z <= 0 or p2.z <= 0:
                    continue

                # screen coordinates
                screen_coords = []
                for p, color in [(p0, tri.v0.color), (p1, tri.v1.color), (p2, tri.v2.color)]:
                    x = int((p.x / p.z + 1) * self.width * 0.5)
                    y = int((1 - p.y / p.z) * self.height * 0.5)
                    screen_coords.append((x, y, color, p.z))

                projected.append(tuple(screen_coords))

            except (ZeroDivisionError, ValueError):
                continue

        return projected

    def fast_cull(self, triangles: List[Triangle], view_matrix: Mat4) -> Tuple[List[Triangle], int]:
        """optimized culling using pinqy chains"""
        camera_forward = (self.camera_target - self.camera_pos).normalize()

        # single pass: backface + distance culling
        visible = (P(triangles)
                   .where(lambda tri: self.quick_backface_test(tri, view_matrix))
                   .where(lambda tri: self.distance_test(tri))
                   .to_list())

        return visible, len(triangles) - len(visible)

    def quick_backface_test(self, tri: Triangle, view_matrix: Mat4) -> bool:
        """simplified backface culling"""
        # use triangle center for quick test
        center = Vec3(
            (tri.v0.pos.x + tri.v1.pos.x + tri.v2.pos.x) / 3,
            (tri.v0.pos.y + tri.v1.pos.y + tri.v2.pos.y) / 3,
            (tri.v0.pos.z + tri.v1.pos.z + tri.v2.pos.z) / 3
        )

        # simple dot product test with camera direction
        to_tri = center - self.camera_pos
        camera_forward = (self.camera_target - self.camera_pos).normalize()
        return to_tri.dot(camera_forward) > 0

    def distance_test(self, tri: Triangle) -> bool:
        """distance-based culling"""
        center = Vec3(
            (tri.v0.pos.x + tri.v1.pos.x + tri.v2.pos.x) / 3,
            (tri.v0.pos.y + tri.v1.pos.y + tri.v2.pos.y) / 3,
            (tri.v0.pos.z + tri.v1.pos.z + tri.v2.pos.z) / 3
        )
        return center.distance_to(self.camera_pos) < self.far * 0.8

    def update_camera(self, dt: float):
        """simple camera movement"""
        keys = pygame.key.get_pressed()
        speed = 10.0 * dt

        forward = (self.camera_target - self.camera_pos).normalize()
        right = forward.cross(self.camera_up).normalize()

        if keys[pygame.K_w]: self.camera_pos = self.camera_pos + forward * speed
        if keys[pygame.K_s]: self.camera_pos = self.camera_pos - forward * speed
        if keys[pygame.K_a]: self.camera_pos = self.camera_pos - right * speed
        if keys[pygame.K_d]: self.camera_pos = self.camera_pos + right * speed

        # simple rotation
        if keys[pygame.K_LEFT]:
            angle = -dt
            new_forward = Vec3(forward.x * math.cos(angle) - forward.z * math.sin(angle),
                               forward.y,
                               forward.x * math.sin(angle) + forward.z * math.cos(angle))
            self.camera_target = self.camera_pos + new_forward

        if keys[pygame.K_RIGHT]:
            angle = dt
            new_forward = Vec3(forward.x * math.cos(angle) - forward.z * math.sin(angle),
                               forward.y,
                               forward.x * math.sin(angle) + forward.z * math.cos(angle))
            self.camera_target = self.camera_pos + new_forward

    def render_frame(self, dt: float) -> FrameMetrics:
        """optimized single-pass rendering"""
        frame_start = time.perf_counter()

        self.screen.fill((20, 30, 60))
        self.update_camera(dt)

        # build matrices
        view_matrix = Mat4.look_at(self.camera_pos, self.camera_target, self.camera_up)
        proj_matrix = Mat4.perspective(self.fov, self.aspect, self.near, self.far)
        mvp_matrix = proj_matrix @ view_matrix

        # cull triangles
        visible_triangles, culled_count = self.fast_cull(self.triangles, view_matrix)

        # project visible triangles
        projected = self.fast_project(visible_triangles, mvp_matrix)

        # render triangles
        rendered_count = 0
        for tri_screen in projected:
            if self.screen_bounds_check(tri_screen):
                self.draw_triangle_fast(tri_screen)
                rendered_count += 1

        pygame.display.flip()

        # metrics
        frame_end = time.perf_counter()
        frame_time = frame_end - frame_start
        fps = 1.0 / frame_time if frame_time > 0 else 0

        return FrameMetrics(frame_time, rendered_count, culled_count, fps)

    def screen_bounds_check(self, tri_screen: Tuple) -> bool:
        """quick screen bounds test"""
        for pt in tri_screen:
            if -50 <= pt[0] <= self.width + 50 and -50 <= pt[1] <= self.height + 50:
                return True
        return False

    def draw_triangle_fast(self, tri_screen: Tuple):
        """minimal triangle drawing"""
        p0, p1, p2 = tri_screen

        # depth-based brightness
        avg_depth = (p0[3] + p1[3] + p2[3]) / 3
        brightness = max(0.2, min(1.0, 20.0 / avg_depth))

        # average color with brightness
        color = tuple(int(((p0[2][i] + p1[2][i] + p2[2][i]) / 3) * brightness) for i in range(3))

        points = [(p0[0], p0[1]), (p1[0], p1[1]), (p2[0], p2[1])]

        try:
            pygame.draw.polygon(self.screen, color, points)
        except:
            pass  # skip invalid triangles

    def log_performance(self, metrics: FrameMetrics):
        """minimal performance logging"""
        if len(self.metrics_history) % 60 == 0 and self.metrics_history.count() > 0:
            recent = self.metrics_history.take_last(60)
            avg_fps = recent.average(lambda m: m.fps)
            avg_frame_time = recent.average(lambda m: m.frame_time * 1000)
            avg_rendered = recent.average(lambda m: m.triangles_rendered)
            avg_culled = recent.average(lambda m: m.triangles_culled)

            print(
                f"fps: {avg_fps:.0f} | {avg_frame_time:.1f}ms | rendered: {avg_rendered:.0f} | culled: {avg_culled:.0f}")

    def run(self):
        """main loop"""
        running = True
        last_time = time.perf_counter()

        print("fast functional 3d engine")
        print("wasd = move, arrow keys = look, esc = exit")

        while running:
            current_time = time.perf_counter()
            dt = current_time - last_time
            last_time = current_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            metrics = self.render_frame(dt)
            self.metrics_history = self.metrics_history.append(metrics).take_last(180)
            self.log_performance(metrics)

            self.clock.tick(60)

        pygame.quit()


# extend pinqy if needed
def take_last(self, count: int):
    data = self._get_data()
    return self.__class__(lambda: data[-count:] if len(data) >= count else data)


from pinqy import Enumerable

if not hasattr(Enumerable, 'take_last'):
    Enumerable.take_last = take_last

if __name__ == "__main__":
    engine = FastEngine()
    engine.run()