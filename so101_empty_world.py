import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
# franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
so101 = scene.add_entity(gs.morphs.URDF(file='SO101/so101_new_calib.urdf'))

scene.build()

for i in range(1000):
    scene.step()
