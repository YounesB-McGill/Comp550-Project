import json


def add_class_json(class_name: str) -> str:
    return json.dumps({
        "intents": [{"intent": "create_class"}],
        "entities": [{"value": class_name}],
        "output": {"text": [f"I created a class called {class_name}."]}
    })


def add_attribute(class_name: str, attribute_name: str) -> str:
    return json.dumps({
        "intents": [{"intent": "add_attribute"}],
        "entities": [{"value": class_name}, {"value": attribute_name}],
        "output": [{"text": f"{class_name} now has the attribute {attribute_name}."}]
    })


def create_composition(whole_class_name: str, part_class_name: str) -> str:
    return json.dumps({
        "intents": [{"intent": "create_composition"}],
        "entities": [{"value": whole_class_name}, {"value": part_class_name}],
        "output": {"text": [f"{whole_class_name} is now composed of {part_class_name}."]},
        "context": {"varContainer": whole_class_name, "varPart": part_class_name}
    })


def create_association(class_name1: str, class_name2: str) -> str:
    return json.dumps({
        "intents": [{"intent": "create_association"}],
        "entities": [{"value": class_name1}, {"value": class_name2}],
        "output": [{"text": f"A {class_name1} has many {class_name2}s."}],
    })


def create_inheritance(child: str, parent: str) -> str:
    return json.dumps({
        "intents": [{"intent": "create_inheritance"}],
        "entities": [{"value": child}, {"value": parent}],
        "output": {"text": [f"{child} is a subclass of {parent}."]}
    })


def return_error_to_user(error_msg: str) -> str:
    return json.dumps({
        "intents": [{"intent": "return_error_to_user"}],
        "output": {"text": error_msg}
    })
