import genesis as gs
import numpy as np
import imageio.v2 as imageio


if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False)
    plane = scene.add_entity(gs.morphs.Plane())
    so101 = scene.add_entity(gs.morphs.URDF(file='SO101/so101_new_calib.urdf'))
    cam = scene.add_camera(
        res    = (640, 480),
        pos    = (2.0, 0.0, 0.3),
        lookat = (0, 0, 0.0),
        fov    = 30,
        GUI    = False
    )
    scene.build()

frames = []
for _ in range(240):
    scene.step()
    rgb, _, _, _ = cam.render(normal=True)
    frames.append(rgb.astype(np.uint8))

imageio.mimsave("out.mp4", frames, fps=30, codec="libx264", quality=8)
