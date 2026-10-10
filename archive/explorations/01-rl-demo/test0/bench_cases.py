from __future__ import annotations

from typing import Any, Dict, List


def _bg(elements: List[str], colors: List[str], desc: str) -> Dict[str, Any]:
    return {"id": "background", "type": "background", "spec": {"desc": desc, "elements": elements, "colors": colors}, "deps": []}


def _obj(nid: str, desc: str, noun: str, count: int = 1, colors=None, pose: str = "none", facing: str = "none", attributes=None) -> Dict[str, Any]:
    return {"id": nid, "type": "object", "spec": {"desc": desc, "noun": noun, "count": count, "colors": list(colors or []), "pose": pose, "facing": facing, "attributes": list(attributes or [])}, "deps": []}


def _ent(nid: str, desc: str, noun: str, count: int = 1, colors=None, pose: str = "standing", facing: str = "front", attributes=None) -> Dict[str, Any]:
    return {"id": nid, "type": "entity", "spec": {"desc": desc, "noun": noun, "count": count, "colors": list(colors or []), "pose": pose, "facing": facing, "attributes": list(attributes or [])}, "deps": []}


def _act(nid: str, desc: str, verb: str, participants: List[str], poses: Dict[str, str]) -> Dict[str, Any]:
    return {"id": nid, "type": "action", "spec": {"desc": desc, "verb": verb, "participants": participants, "poses": poses}, "deps": list(participants)}


def _scene(desc: str, relations: List[Dict[str, str]]) -> Dict[str, Any]:
    return {"id": "scene", "type": "scene", "spec": {"desc": desc, "relations": relations}, "deps": []}


def _rel(a: str, rel: str, b: str) -> Dict[str, str]:
    return {"a": a, "rel": rel, "b": b}


def _chk(cid: str, ctype: str, text: str, **args: Any) -> Dict[str, Any]:
    out = {"id": cid, "type": ctype, "text": text}
    out.update(args)
    return out


CASES: List[Dict[str, Any]] = [
    {
        "id": "apples_table",
        "title": "Apples and cup on a table",
        "prompt": "Three red apples on a wooden table, a blue cup to the right of the apples on the same table, plain white wall behind",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["wall", "floor"], ["white"], "plain white wall and floor"),
            _obj("red_apples", "three red apples", "apple", 3, ["red"]),
            _obj("wooden_table", "a wooden table", "table", 1, ["wooden"]),
            _obj("blue_cup", "a blue cup", "cup", 1, ["blue"]),
            _scene("apples and a cup on a table", [_rel("red_apples", "on", "wooden_table"), _rel("blue_cup", "on", "wooden_table"), _rel("blue_cup", "right_of", "red_apples")]),
        ]},
        "checks": [
            _chk("c1", "count", "exactly three apples", node="red_apples", value=3),
            _chk("c2", "color", "the apples are red", node="red_apples", value="red"),
            _chk("c3", "presence", "a wooden table is present", node="wooden_table"),
            _chk("c4", "color", "the cup is blue", node="blue_cup", value="blue"),
            _chk("c5", "right_of", "the cup is to the right of the apples", a="blue_cup", b="red_apples"),
            _chk("c6", "on", "the apples rest on the table", a="red_apples", b="wooden_table"),
        ],
    },
    {
        "id": "girl_umbrella_cat",
        "title": "Girl with umbrella and cat",
        "prompt": "A girl in a yellow dress holding a red umbrella, standing to the left of a black cat that sits on green grass, blue sky",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["sky", "grass"], ["blue", "green"], "blue sky and green grass"),
            _ent("girl", "a girl in a yellow dress", "girl", 1, ["yellow"], "holding", "right"),
            _obj("red_umbrella", "a red umbrella", "umbrella", 1, ["red"]),
            _act("hold_umbrella", "the girl holding the red umbrella", "holding", ["girl", "red_umbrella"], {"girl": "holding"}),
            _ent("black_cat", "a black cat sitting", "cat", 1, ["black"], "sitting", "left"),
            _scene("girl with umbrella left of a sitting cat", [_rel("girl", "left_of", "black_cat")]),
        ]},
        "checks": [
            _chk("c1", "color", "the girl wears yellow", node="girl", value="yellow"),
            _chk("c2", "color", "the umbrella is red", node="red_umbrella", value="red"),
            _chk("c3", "interaction", "the girl is holding the umbrella", node="hold_umbrella"),
            _chk("c4", "color", "the cat is black", node="black_cat", value="black"),
            _chk("c5", "pose", "the cat is sitting", node="black_cat", value="sitting"),
            _chk("c6", "left_of", "the girl is to the left of the cat", a="girl", b="black_cat"),
        ],
    },
    {
        "id": "dogs_park",
        "title": "Dogs in a park",
        "prompt": "Two brown dogs running on green grass, a red ball near the dogs, a tall green tree to the right of the dogs, blue sky with a sun",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["sky", "sun", "grass"], ["blue", "green"], "blue sky with a sun over green grass"),
            _ent("brown_dogs", "two brown dogs running", "dog", 2, ["brown"], "running", "right"),
            _obj("red_ball", "a red ball", "ball", 1, ["red"]),
            _obj("green_tree", "a tall green tree", "tree", 1, ["green"], attributes=["tall"]),
            _scene("dogs running near a ball with a tree on the right", [_rel("red_ball", "near", "brown_dogs"), _rel("green_tree", "right_of", "brown_dogs")]),
        ]},
        "checks": [
            _chk("c1", "count", "exactly two dogs", node="brown_dogs", value=2),
            _chk("c2", "color", "the dogs are brown", node="brown_dogs", value="brown"),
            _chk("c3", "pose", "the dogs are running", node="brown_dogs", value="running"),
            _chk("c4", "color", "the ball is red", node="red_ball", value="red"),
            _chk("c5", "right_of", "the tree is to the right of the dogs", a="green_tree", b="brown_dogs"),
            _chk("c6", "near", "the ball is near the dogs", a="red_ball", b="brown_dogs"),
        ],
    },
    {
        "id": "boy_bicycle",
        "title": "Boy on a bicycle",
        "prompt": "A boy riding a blue bicycle on a gray road, a white dog running to the left of the bicycle, two orange cones to the right of the bicycle, blue sky",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["sky", "road"], ["blue", "gray"], "blue sky over a gray road"),
            _ent("boy", "a boy", "boy", 1, [], "sitting", "right"),
            _obj("blue_bicycle", "a blue bicycle", "bicycle", 1, ["blue"]),
            _act("ride_bicycle", "the boy riding the blue bicycle", "riding", ["boy", "blue_bicycle"], {"boy": "sitting"}),
            _ent("white_dog", "a white dog running", "dog", 1, ["white"], "running", "right"),
            _obj("orange_cones", "two orange cones", "cone", 2, ["orange"]),
            _scene("boy riding with a dog on the left and cones on the right", [_rel("white_dog", "left_of", "ride_bicycle"), _rel("orange_cones", "right_of", "ride_bicycle")]),
        ]},
        "checks": [
            _chk("c1", "interaction", "the boy is riding the bicycle", node="ride_bicycle"),
            _chk("c2", "color", "the bicycle is blue", node="blue_bicycle", value="blue"),
            _chk("c3", "color", "the dog is white", node="white_dog", value="white"),
            _chk("c4", "count", "exactly two cones", node="orange_cones", value=2),
            _chk("c5", "left_of", "the dog is left of the bicycle", a="white_dog", b="ride_bicycle"),
            _chk("c6", "right_of", "the cones are right of the bicycle", a="orange_cones", b="ride_bicycle"),
        ],
    },
    {
        "id": "kitchen_cups",
        "title": "Cups and bottle in a kitchen",
        "prompt": "Four white cups on a wooden table, a green bottle to the left of the cups on the table, a yellow banana in front of the bottle, kitchen wall behind",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["wall", "floor"], ["beige"], "kitchen wall and floor"),
            _obj("white_cups", "four white cups", "cup", 4, ["white"]),
            _obj("wooden_table", "a wooden table", "table", 1, ["wooden"]),
            _obj("green_bottle", "a green bottle", "bottle", 1, ["green"]),
            _obj("yellow_banana", "a yellow banana", "banana", 1, ["yellow"]),
            _scene("cups, bottle and banana on a table", [_rel("white_cups", "on", "wooden_table"), _rel("green_bottle", "on", "wooden_table"), _rel("green_bottle", "left_of", "white_cups"), _rel("yellow_banana", "in_front_of", "green_bottle")]),
        ]},
        "checks": [
            _chk("c1", "count", "exactly four cups", node="white_cups", value=4),
            _chk("c2", "color", "the cups are white", node="white_cups", value="white"),
            _chk("c3", "color", "the bottle is green", node="green_bottle", value="green"),
            _chk("c4", "left_of", "the bottle is left of the cups", a="green_bottle", b="white_cups"),
            _chk("c5", "color", "the banana is yellow", node="yellow_banana", value="yellow"),
            _chk("c6", "presence", "a wooden table is present", node="wooden_table"),
        ],
    },
    {
        "id": "beach",
        "title": "Beach scene",
        "prompt": "A woman in a red dress lying on yellow sand, a blue umbrella to the right of the woman, three white seagulls flying above her, sea and sky in the background",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["sky", "sea", "sand"], ["blue", "yellow"], "sky, sea and yellow sand"),
            _ent("woman", "a woman in a red dress lying", "woman", 1, ["red"], "lying", "right"),
            _obj("blue_umbrella", "a blue umbrella", "umbrella", 1, ["blue"]),
            _ent("white_seagulls", "three white seagulls flying", "bird", 3, ["white"], "flying", "left"),
            _scene("woman lying on the beach with an umbrella and seagulls", [_rel("blue_umbrella", "right_of", "woman"), _rel("white_seagulls", "above", "woman")]),
        ]},
        "checks": [
            _chk("c1", "color", "the woman wears red", node="woman", value="red"),
            _chk("c2", "pose", "the woman is lying down", node="woman", value="lying"),
            _chk("c3", "color", "the umbrella is blue", node="blue_umbrella", value="blue"),
            _chk("c4", "count", "exactly three seagulls", node="white_seagulls", value=3),
            _chk("c5", "right_of", "the umbrella is right of the woman", a="blue_umbrella", b="woman"),
            _chk("c6", "above", "the seagulls are above the woman", a="white_seagulls", b="woman"),
        ],
    },
    {
        "id": "farm",
        "title": "Farm animals",
        "prompt": "Two black cows standing on the left, a brown horse standing on the right facing left, a red barn behind the animals, green field and blue sky",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["sky", "grass"], ["blue", "green"], "blue sky and green grass"),
            _ent("black_cows", "two black cows standing", "cow", 2, ["black"], "standing", "right"),
            _ent("brown_horse", "a brown horse standing facing left", "horse", 1, ["brown"], "standing", "left"),
            _obj("red_barn", "a red barn", "house", 1, ["red"]),
            _scene("cows on the left, horse on the right, barn behind", [_rel("black_cows", "left_of", "brown_horse"), _rel("red_barn", "behind", "black_cows"), _rel("red_barn", "behind", "brown_horse")]),
        ]},
        "checks": [
            _chk("c1", "count", "exactly two cows", node="black_cows", value=2),
            _chk("c2", "color", "the cows are black", node="black_cows", value="black"),
            _chk("c3", "color", "the horse is brown", node="brown_horse", value="brown"),
            _chk("c4", "left_of", "the cows are left of the horse", a="black_cows", b="brown_horse"),
            _chk("c5", "color", "the barn is red", node="red_barn", value="red"),
            _chk("c6", "behind", "the barn is behind the animals", a="red_barn", b="brown_horse"),
        ],
    },
    {
        "id": "classroom",
        "title": "Classroom",
        "prompt": "A teacher standing next to a green board, three students sitting on wooden chairs to the left of the teacher, a wooden desk in front of the teacher, classroom wall",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["wall", "floor"], ["beige"], "classroom wall and floor"),
            _ent("teacher", "a teacher standing", "teacher", 1, ["blue"], "standing", "left"),
            _obj("green_board", "a green board", "board", 1, ["green"]),
            _ent("students", "three students", "student", 3, ["red"], "sitting", "right"),
            _obj("wooden_chairs", "three wooden chairs", "chair", 3, ["wooden"]),
            _act("students_sitting", "three students sitting on wooden chairs", "sitting_on", ["students", "wooden_chairs"], {"students": "sitting"}),
            _obj("wooden_desk", "a wooden desk", "table", 1, ["wooden"]),
            _scene("teacher by the board with seated students", [_rel("teacher", "near", "green_board"), _rel("students_sitting", "left_of", "teacher"), _rel("wooden_desk", "in_front_of", "teacher")]),
        ]},
        "checks": [
            _chk("c1", "presence", "a teacher is present", node="teacher"),
            _chk("c2", "color", "the board is green", node="green_board", value="green"),
            _chk("c3", "count", "exactly three students", node="students", value=3),
            _chk("c4", "interaction", "the students sit on chairs", node="students_sitting"),
            _chk("c5", "left_of", "the students are left of the teacher", a="students_sitting", b="teacher"),
            _chk("c6", "presence", "a wooden desk is present", node="wooden_desk"),
        ],
    },
    {
        "id": "picnic",
        "title": "Picnic",
        "prompt": "A red blanket on green grass, a wicker basket on the blanket, two yellow bananas and one green apple on the blanket, a brown dog sitting to the right of the blanket, blue sky",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["sky", "grass"], ["blue", "green"], "blue sky and green grass"),
            _obj("red_blanket", "a red blanket", "blanket", 1, ["red"]),
            _obj("wicker_basket", "a wicker basket", "basket", 1, ["tan"]),
            _obj("yellow_bananas", "two yellow bananas", "banana", 2, ["yellow"]),
            _obj("green_apple", "one green apple", "apple", 1, ["green"]),
            _ent("brown_dog", "a brown dog sitting", "dog", 1, ["brown"], "sitting", "left"),
            _scene("picnic blanket with basket, fruit and a dog", [_rel("wicker_basket", "on", "red_blanket"), _rel("yellow_bananas", "on", "red_blanket"), _rel("green_apple", "on", "red_blanket"), _rel("brown_dog", "right_of", "red_blanket")]),
        ]},
        "checks": [
            _chk("c1", "color", "the blanket is red", node="red_blanket", value="red"),
            _chk("c2", "presence", "a basket is present", node="wicker_basket"),
            _chk("c3", "count", "exactly two bananas", node="yellow_bananas", value=2),
            _chk("c4", "color", "the apple is green", node="green_apple", value="green"),
            _chk("c5", "pose", "the dog is sitting", node="brown_dog", value="sitting"),
            _chk("c6", "right_of", "the dog is right of the blanket", a="brown_dog", b="red_blanket"),
        ],
    },
    {
        "id": "night_camp",
        "title": "Night camp",
        "prompt": "An orange tent on the left, a campfire in the middle, a man sitting to the right of the campfire facing left, a moon and stars in the night sky, pine trees far behind",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["night", "moon", "stars", "forest", "ground"], ["navy"], "night sky with moon and stars, pine trees, dark ground"),
            _obj("orange_tent", "an orange tent", "tent", 1, ["orange"]),
            _obj("campfire", "a campfire", "campfire", 1, ["orange"]),
            _ent("man", "a man sitting facing left", "man", 1, ["green"], "sitting", "left"),
            _scene("tent, campfire and a seated man at night", [_rel("orange_tent", "left_of", "campfire"), _rel("man", "right_of", "campfire")]),
        ]},
        "checks": [
            _chk("c1", "color", "the tent is orange", node="orange_tent", value="orange"),
            _chk("c2", "presence", "a campfire is present", node="campfire"),
            _chk("c3", "pose", "the man is sitting", node="man", value="sitting"),
            _chk("c4", "left_of", "the tent is left of the campfire", a="orange_tent", b="campfire"),
            _chk("c5", "right_of", "the man is right of the campfire", a="man", b="campfire"),
        ],
    },
    {
        "id": "fruit_bowl",
        "title": "Fruit bowl",
        "prompt": "A blue bowl on a wooden table with five red apples inside the bowl, a silver spoon to the left of the bowl on the table, plain white wall behind",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["wall", "floor"], ["white"], "plain white wall and floor"),
            _obj("blue_bowl", "a blue bowl", "bowl", 1, ["blue"]),
            _obj("wooden_table", "a wooden table", "table", 1, ["wooden"]),
            _obj("red_apples_five", "five red apples", "apple", 5, ["red"]),
            _obj("silver_spoon", "a silver spoon", "spoon", 1, ["silver"]),
            _scene("bowl of apples and a spoon on a table", [_rel("blue_bowl", "on", "wooden_table"), _rel("red_apples_five", "inside", "blue_bowl"), _rel("silver_spoon", "left_of", "blue_bowl"), _rel("silver_spoon", "on", "wooden_table")]),
        ]},
        "checks": [
            _chk("c1", "color", "the bowl is blue", node="blue_bowl", value="blue"),
            _chk("c2", "count", "exactly five apples", node="red_apples_five", value=5),
            _chk("c3", "color", "the apples are red", node="red_apples_five", value="red"),
            _chk("c4", "inside", "the apples are inside the bowl", a="red_apples_five", b="blue_bowl"),
            _chk("c5", "left_of", "the spoon is left of the bowl", a="silver_spoon", b="blue_bowl"),
            _chk("c6", "presence", "a wooden table is present", node="wooden_table"),
        ],
    },
    {
        "id": "birds_wire",
        "title": "Birds on a wire",
        "prompt": "Six black birds sitting on a long gray wire, a yellow sun in the sky, a white house below the wire, blue sky",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["sky", "sun", "ground"], ["blue"], "blue sky with a yellow sun"),
            _ent("black_birds", "six black birds sitting", "bird", 6, ["black"], "sitting", "left"),
            _obj("gray_wire", "a long gray wire", "wire", 1, ["gray"], attributes=["long"]),
            _act("birds_on_wire", "six black birds sitting on the wire", "sitting_on", ["black_birds", "gray_wire"], {"black_birds": "sitting"}),
            _obj("white_house", "a white house", "house", 1, ["white"]),
            _scene("birds on a wire above a house", [_rel("white_house", "below", "birds_on_wire")]),
        ]},
        "checks": [
            _chk("c1", "count", "exactly six birds", node="black_birds", value=6),
            _chk("c2", "color", "the birds are black", node="black_birds", value="black"),
            _chk("c3", "interaction", "the birds sit on the wire", node="birds_on_wire"),
            _chk("c4", "color", "the house is white", node="white_house", value="white"),
            _chk("c5", "below", "the house is below the wire", a="white_house", b="birds_on_wire"),
        ],
    },
    {
        "id": "living_room",
        "title": "Living room",
        "prompt": "A gray cat lying on a red sofa, a brown dog sitting on the floor to the left of the sofa, a yellow lamp behind the sofa, living room wall",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["wall", "floor"], ["beige"], "living room wall and floor"),
            _ent("gray_cat", "a gray cat lying", "cat", 1, ["gray"], "lying", "left"),
            _obj("red_sofa", "a red sofa", "sofa", 1, ["red"]),
            _act("cat_on_sofa", "a gray cat lying on the red sofa", "lying_on", ["gray_cat", "red_sofa"], {"gray_cat": "lying"}),
            _ent("brown_dog", "a brown dog sitting", "dog", 1, ["brown"], "sitting", "left"),
            _obj("yellow_lamp", "a yellow lamp", "lamp", 1, ["yellow"]),
            _scene("cat on a sofa, dog on the floor, lamp behind", [_rel("brown_dog", "left_of", "cat_on_sofa"), _rel("yellow_lamp", "behind", "cat_on_sofa")]),
        ]},
        "checks": [
            _chk("c1", "color", "the cat is gray", node="gray_cat", value="gray"),
            _chk("c2", "interaction", "the cat lies on the sofa", node="cat_on_sofa"),
            _chk("c3", "color", "the sofa is red", node="red_sofa", value="red"),
            _chk("c4", "pose", "the dog is sitting", node="brown_dog", value="sitting"),
            _chk("c5", "left_of", "the dog is left of the sofa", a="brown_dog", b="cat_on_sofa"),
            _chk("c6", "color", "the lamp is yellow", node="yellow_lamp", value="yellow"),
        ],
    },
    {
        "id": "market",
        "title": "Market stall",
        "prompt": "Three green watermelons on the left, four orange pumpkins on the right, a wooden crate between them, a woman standing behind the crate holding a basket, blue sky",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["sky", "ground"], ["blue"], "blue sky over the ground"),
            _obj("green_watermelons", "three green watermelons", "melon", 3, ["green"]),
            _obj("orange_pumpkins", "four orange pumpkins", "pumpkin", 4, ["orange"]),
            _obj("wooden_crate", "a wooden crate", "crate", 1, ["wooden"]),
            _ent("woman", "a woman standing", "woman", 1, ["purple"], "holding", "front"),
            _obj("basket", "a basket", "basket", 1, ["tan"]),
            _act("woman_basket", "the woman holding a basket", "holding", ["woman", "basket"], {"woman": "holding"}),
            _scene("watermelons left, pumpkins right, crate in the middle, woman behind", [_rel("green_watermelons", "left_of", "wooden_crate"), _rel("orange_pumpkins", "right_of", "wooden_crate"), _rel("woman_basket", "behind", "wooden_crate")]),
        ]},
        "checks": [
            _chk("c1", "count", "exactly three watermelons", node="green_watermelons", value=3),
            _chk("c2", "count", "exactly four pumpkins", node="orange_pumpkins", value=4),
            _chk("c3", "color", "the pumpkins are orange", node="orange_pumpkins", value="orange"),
            _chk("c4", "left_of", "the watermelons are left of the crate", a="green_watermelons", b="wooden_crate"),
            _chk("c5", "right_of", "the pumpkins are right of the crate", a="orange_pumpkins", b="wooden_crate"),
            _chk("c6", "interaction", "the woman holds a basket", node="woman_basket"),
        ],
    },
    {
        "id": "pond",
        "title": "Pond at sunset",
        "prompt": "Two white swans swimming on a blue pond, a green frog sitting on a lily pad to the right of the swans, tall reeds on the left of the swans, sunset sky",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["sunset", "sky", "water"], ["orange", "blue"], "sunset sky over blue pond water"),
            _ent("white_swans", "two white swans swimming", "swan", 2, ["white"], "swimming", "left"),
            _ent("green_frog", "a green frog sitting", "frog", 1, ["green"], "sitting", "left"),
            _obj("lily_pad", "a lily pad", "pad", 1, ["green"]),
            _act("frog_on_pad", "a green frog sitting on a lily pad", "sitting_on", ["green_frog", "lily_pad"], {"green_frog": "sitting"}),
            _obj("reeds", "tall reeds", "reeds", 1, ["green"], attributes=["tall"]),
            _scene("swans with a frog on the right and reeds on the left", [_rel("frog_on_pad", "right_of", "white_swans"), _rel("reeds", "left_of", "white_swans")]),
        ]},
        "checks": [
            _chk("c1", "count", "exactly two swans", node="white_swans", value=2),
            _chk("c2", "color", "the swans are white", node="white_swans", value="white"),
            _chk("c3", "color", "the frog is green", node="green_frog", value="green"),
            _chk("c4", "interaction", "the frog sits on a lily pad", node="frog_on_pad"),
            _chk("c5", "right_of", "the frog is right of the swans", a="frog_on_pad", b="white_swans"),
            _chk("c6", "left_of", "the reeds are left of the swans", a="reeds", b="white_swans"),
        ],
    },
    {
        "id": "snow",
        "title": "Snow day",
        "prompt": "A white snowman with a black hat in the middle, a red sled to the right of the snowman, two children in blue jackets standing to the left of the snowman, snowy field and gray sky",
        "hint": {"canvas": {"width": 1280, "height": 768}, "nodes": [
            _bg(["sky", "snow"], ["gray", "white"], "gray sky over a snowy field"),
            _obj("snowman", "a white snowman with a black hat", "snowman", 1, ["white", "black"]),
            _obj("red_sled", "a red sled", "sled", 1, ["red"]),
            _ent("children", "two children in blue jackets standing", "child", 2, ["blue"], "standing", "right"),
            _scene("snowman in the middle, sled right, children left", [_rel("red_sled", "right_of", "snowman"), _rel("children", "left_of", "snowman")]),
        ]},
        "checks": [
            _chk("c1", "color", "the snowman is white", node="snowman", value="white"),
            _chk("c2", "color", "the sled is red", node="red_sled", value="red"),
            _chk("c3", "count", "exactly two children", node="children", value=2),
            _chk("c4", "color", "the children wear blue", node="children", value="blue"),
            _chk("c5", "right_of", "the sled is right of the snowman", a="red_sled", b="snowman"),
            _chk("c6", "left_of", "the children are left of the snowman", a="children", b="snowman"),
        ],
    },
]


def case_by_id(case_id: str) -> Dict[str, Any]:
    for case in CASES:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)
