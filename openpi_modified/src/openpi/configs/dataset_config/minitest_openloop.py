"""Minitest dataset configuration for fast openloop evaluation in CI.

This config selects representative samples from 10 different robot types
to ensure comprehensive coverage while keeping evaluation under 10 minutes.

Robot Types Covered:
1. Agilex (from robomind_2)
2. Tianyi (from robomind_2)
3. Galaxea (from OpenGalaxea)
4. ALOHA static (from aloha)
5. Intern A1 (from intern_a1_real)
6. RDT (from rdt-ft-data)
7. AgiBot G1 (from robocoin)
8. Cobot Magic (from robocoin)
9. AIRBOT MMK2 (from robocoin)
10. Galbot G1 (from robocoin)

Data is expected to be prepared at: /mnt/workspace/openpi_minitest/
"""

ROOT_DIR = "/mnt/workspace/openpi_minitest/"

REPO_ID = [
    # 1. Agilex (from robomind_2)
    "robomind_2/agilex/fold_clothes/success_episodes",
    # 2. Tianyi (from robomind_2)
    "robomind_2/tianyi/close_drawer_under_combined_cabinet/success_episodes",
    # 3. Galaxea (from OpenGalaxea)
    "galaxea/Adjust_The_Air_Conditioner_Temperature_20250711_006",
    # 4. ALOHA static (from aloha)
    "aloha/aloha_static_coffee",
    # 5. Intern A1 (from intern_a1_real)
    "intern_a1/pickup_a_bag_of_bread_into_the_basket/set_0",
    # 6. RDT (from rdt-ft-data)
    "rdt/pick_place_water_bottle",
    # 7. AgiBot G1 (from robocoin)
    "robocoin/AgiBot-g1_box_storage_a",
    # 8. Cobot Magic (from robocoin)
    "robocoin/Cobot_Magic_move_the_cup",
    # 9. AIRBOT MMK2 (from robocoin)
    "robocoin/AIRBOT_MMK2_mobile_car",
    # 10. Galbot G1 (from robocoin)
    "robocoin/Galbot_g1_steamer_storage_baozi_a",
]

# Filter to only include existing directories
import os

REPO_ID = [repo_id for repo_id in REPO_ID if os.path.isdir(os.path.join(ROOT_DIR, repo_id))]

print(f"Minitest dataset: {len(REPO_ID)} robot types loaded")
