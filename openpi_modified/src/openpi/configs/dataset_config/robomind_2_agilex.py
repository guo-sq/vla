import os

REPO_ID = []

ROOT_DIR = "/mnt/"
split = "success_episodes"

# sub_path = "oss_data/X-Humanoid/RoboMIND2.0-Agilex_lerobot/agilex"
# tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
# REPO_ID += [os.path.join(sub_path, task, split) for task in tasks if task in [
#     "fold_clothes",
# ]]

sub_path = "workspace/heyuan/test_data/RoboMIND2.0-Agilex_lerobot/agilex"
tasks = sorted(os.listdir(os.path.join(ROOT_DIR, sub_path)))
REPO_ID += [
    os.path.join(sub_path, task, split)
    for task in tasks
    if task
    in [
        "arrange_blocks_and_place_orange_in_center_with_arms",
        "close_fridge_door",
        "connect_drain_pipes_together",
        "flip_cup_and_place_on_plate_with_arms",
        "fold_clothes",
        # "fry_beef_cubes", # only 14 dim
        "insert_blue_slippers_together_front_up",
        "insert_red_and_black_pens_into_pink_holder",
        "left_arm_grabs_bread_right_arm_adds_lettuce",
        "make_breakfast_with_bread_and_shrimp",
        "move_button_from_left_to_right",
        "move_button_from_right_to_left",
        "open_pot_and_pour_water_with_arms",
        "open_pot_and_put_corn_with_arms",
        # "organize_plates_and_spoons", # only 14 dim
        "pick_red_yellow_buttons_and_place_in_box",
        "place_and_press_bell_with_both_arms",
        "place_and_stand_button_with_both_arms",
        "place_bread_and_move_sandwich_to_tray",
        "place_button_on_rack_with_both_arms",
        "place_circuit_board_on_tray",
        "place_colored_items_on_matching_plates",
        "place_corn_on_plate_with_both_arms",
        "place_milk_next_to_cup",
        "place_mug_in_plate_and_rotate_handle_right",
        "place_orange_and_plate_on_tray_with_arms",
        "place_orange_and_plate_on_tray_with_arms",
        "place_plate_and_fork_with_arms",
        "place_pomegranate_in_white_plate",
        "place_red_and_green_buttons_by_color_with_arms",
        "place_three_cups",
        "place_utensils_into_plate",
        "pour_fried_egg_into_plate_with_right_arm",
        "pour_ham_into_pot_and_cover_lid",
        "pour_meat_into_plate_with_right_arm",
        "pour_seasoning_into_cup_on_scale_with_both_arms",
        "pour_wine_from_measuring_cup_to_shaker",
        "put_bread_in_oven",
        "put_pen_and_scissors_into_bin",
        "put_scissors_in_drawer_with_both_arms",
    ]
]

REPO_ID = [repo_id for repo_id in REPO_ID if os.path.exists(os.path.join(ROOT_DIR, repo_id))]
print("-----------------------------")
print(REPO_ID)
