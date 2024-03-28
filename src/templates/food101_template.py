food_simple_template = [
    lambda c: f'a photo of {c}, a type of food.',
]

food_main_template = [
    lambda c: f'a photo of {c}, a type of food',
]

food_context_template = [
    lambda c, d: f'a photo of {c}, a type of food, {d}',
]

food_factor_templates = {
    "cuisines": [
        ", African cuisine",
        ", American cuisine",
        ", Asian cuisine",
        ", European cuisine",
        ", Oceanic cuisine",
    ],
    "condition": [
        "",
        ", cool",
        ", nice",
        ", weird",
    ],
}
