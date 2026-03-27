# Language command bank for the language-grounded drone navigation task.
# Maps object type name → list of natural language commands that refer to it.
# Object indices: cube=0, sphere=1, cylinder=2

COMMANDS = {
    "cube": [
        "go to the square",
        "fly to the box",
        "navigate to the cube",
        "find the red cube",
        "hover over the square",
        "move to the box",
        "head to the cube",
    ],
    "sphere": [
        "go to the circle",
        "fly to the ball",
        "navigate to the sphere",
        "find the blue ball",
        "hover over the circle",
        "move to the sphere",
        "head to the ball",
    ],
    "cylinder": [
        "go to the cylinder",
        "fly to the pillar",
        "navigate to the column",
        "find the green cylinder",
        "hover over the pillar",
        "move to the cylinder",
        "head to the column",
    ],
}

# Ordered list of object type names — index matches the scene object index
OBJECT_TYPES = list(COMMANDS.keys())  # ["cube", "sphere", "cylinder"]
