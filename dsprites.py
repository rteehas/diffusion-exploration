from datasets import load_dataset

LABEL_MAP = {0: "a square", 1: "an ellipse", 2: "a heart"}

def _coord_to_word(v: float) -> str:
    """Map [0,1] coordinate to 'left|center|right' or 'top|center|bottom'."""
    if v <= 1/3:
        return "left"
    if v >= 2/3:
        return "right"
    return "center"

def _coords_to_location(x: float, y: float) -> str:
    # y=0 is top, so evaluate y first
    vert = "top" if y <= 1/3 else "bottom" if y >= 2/3 else "center"
    horiz = _coord_to_word(x)
    if vert == "center" and horiz == "center":
        return "center"
    return f"{vert}-{horiz}"  # e.g. "top-left"

def make_caption(ex):
    shape = LABEL_MAP[ex['label_shape']]
    # x = ex['value_x_position']
    # y = ex['value_y_position']
    # location = _coords_to_location(x, y)
    # ex['caption_enriched'] = f"A picture of {shape} positioned in the {location}"
    ex['caption_enriched'] = f"A picture of {shape}"
    return ex

def load_dsprites():
    return load_dataset("dpdl-benchmark/dsprites")

def prepare_dsprites():
    data = load_dsprites()
    return data.map(make_caption)

# def get_validation_prompts():
#     shapes = ["a square", "an ellipse", "a heart"]
#     locations = [
#         "top-left", "top-center", "top-right",
#         "center-left", "center", "center-right",
#         "bottom-left", "bottom-center", "bottom-right"
#     ]

#     captions = [f"A picture of {s} positioned in the {loc}"
#                 for s in shapes for loc in locations]

#     return captions

def get_validation_prompts(n: int = 1):
    shapes = ["a square", "an ellipse", "a heart"]
    # locations = [
    #     "top-left", "top-center", "top-right",
    #     "center-left", "center", "center-right",
    #     "bottom-left", "bottom-center", "bottom-right"
    # ]

    # base = [f"A picture of {s} positioned in the {loc}"
    #         for s in shapes for loc in locations]

    base = [f"A picture of {s}" for s in shapes]

    return [c for c in base for _ in range(n)]
