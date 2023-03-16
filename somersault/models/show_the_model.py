"""
This file is to display the human model into bioviz
"""
import os
import bioviz
from somersault import Models

# model_name = Models.LEG
# model_name = Models.ARM
model_name = Models.ACROBAT
# model_name = Models.UPPER_LIMB_XYZ_VARIABLES
# model_name = Models.HUMANOID_10DOF

export_model = False
background_color = (1, 1, 1) if export_model else (0.5, 0.5, 0.5)
show_gravity_vector = False if export_model else True
show_floor = False if export_model else True
show_local_ref_frame = False if export_model else True
show_global_ref_frame = False if export_model else True
show_markers = False if export_model else True
show_mass_center = False if export_model else True
show_global_center_of_mass = False if export_model else True
show_segments_center_of_mass = False if export_model else True


def print_all_camera_parameters(biorbd_viz: bioviz.Viz):
    print("Camera roll: ", biorbd_viz.get_camera_roll())
    print("Camera zoom: ", biorbd_viz.get_camera_zoom())
    print("Camera position: ", biorbd_viz.get_camera_position())
    print("Camera focus point: ", biorbd_viz.get_camera_focus_point())


biorbd_viz = bioviz.Viz(
    model_name.value,
    show_gravity_vector=False,
    show_floor=False,
    show_local_ref_frame=False,
    show_global_ref_frame=False,
    show_markers=False,
    show_mass_center=False,
    show_global_center_of_mass=False,
    show_segments_center_of_mass=False,
    mesh_opacity=1,
    background_color=(1, 1, 1),
)
biorbd_viz.set_camera_position(-8.782458942185185, 0.486269131372712, 4.362010279585766)
biorbd_viz.set_camera_roll(90)
biorbd_viz.set_camera_zoom(0.308185240948253)
biorbd_viz.set_camera_focus_point(1.624007185850899, 0.009961251074366406, 1.940316420941989)
biorbd_viz.resize(900, 900)



biorbd_viz.exec()
print_all_camera_parameters(biorbd_viz)

print("Done")
