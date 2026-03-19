"""
Tool schemas for evaluation actions.
Provides both OpenAI function-calling format and MCP tool format.
"""

from mcp import types

# OpenAI function-calling format
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_scene_objects",
            "description": "list all receptacles and their positions in the scene.",
            "parameters": {
                "type": "object",
                "required": [],
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nav_to",
            "description": "navigate to a receptacle",
            "parameters": {
                "type": "object",
                "required": ["receptacle_name"],
                "properties": {
                    "receptacle_name": {
                        "type": "string",
                        "description": "the name of the receptacle to perform navigation",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "walk_around",
            "description": "walk around the current receptacle to get all objects on top of it.",
            "parameters": {
                "type": "object",
                "required": [],
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gaze_at",
            "description": "gaze at an object for manipulation in your current marker_id space.",
            "parameters": {
                "type": "object",
                "required": ["marker_id"],
                "properties": {
                    "marker_id": {
                        "type": "string",
                        "description": "the marker id of the object to gaze",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_object_by_category",
            "description": "detect objects of a category in your view and show their names.",
            "parameters": {
                "type": "object",
                "required": ["target_category"],
                "properties": {
                    "target_category": {
                        "type": "string",
                        "description": "the category of the object to detect",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_receptacles",
            "description": "highlight all receptacle objects in your view.",
            "parameters": {
                "type": "object",
                "required": [],
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pick",
            "description": "pick up a specific object in your view by marker id, the object will become your inventory",
            "parameters": {
                "type": "object",
                "required": ["marker_id"],
                "properties": {
                    "marker_id": {
                        "type": "string",
                        "description": "id of the marker on the object to pick up",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place",
            "description": "place your inventory on top of a receptacle surface.",
            "parameters": {
                "type": "object",
                "required": ["marker_id"],
                "properties": {
                    "marker_id": {
                        "type": "string",
                        "description": "id of the marker on the receptacle surface to place your inventory",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open",
            "description": "open the door of an articulated object",
            "parameters": {
                "type": "object",
                "required": ["marker_id"],
                "properties": {
                    "marker_id": {
                        "type": "string",
                        "description": "marker id of the receptacle door to open",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close",
            "description": "close the door of an articulated object",
            "parameters": {
                "type": "object",
                "required": ["marker_id"],
                "properties": {
                    "marker_id": {
                        "type": "string",
                        "description": "marker id of the receptacle door to close",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "you must call this tool to finish the episode when you think you have completed all the instructions.",
            "parameters": {
                "type": "object",
                "required": [],
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask",
            "description": "ask the user a question to get more information about the task.",
            "parameters": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "the question to ask the user",
                    }
                },
            },
        },
    },
]

# MCP tool format
MCP_TOOLS = [
    types.Tool(
        name="check_scene_objects",
        description="list all receptacles and their positions in the scene.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="nav_to",
        description="navigate to a receptacle",
        inputSchema={
            "type": "object",
            "properties": {
                "receptacle_name": {
                    "type": "string",
                    "description": "the name of the receptacle to perform navigation",
                }
            },
            "required": ["receptacle_name"],
        },
    ),
    types.Tool(
        name="walk_around",
        description="walk around the current receptacle to get all objects on top of it.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="gaze_at",
        description="gaze at an object for manipulation in your current marker_id space.",
        inputSchema={
            "type": "object",
            "properties": {
                "marker_id": {
                    "type": "string",
                    "description": "the marker id of the object to gaze",
                }
            },
            "required": ["marker_id"],
        },
    ),
    types.Tool(
        name="show_object_by_category",
        description="detect objects of a category in your view and show their names.",
        inputSchema={
            "type": "object",
            "properties": {
                "target_category": {
                    "type": "string",
                    "description": "the category of the object to detect",
                }
            },
            "required": ["target_category"],
        },
    ),
    types.Tool(
        name="show_receptacles",
        description="highlight all receptacle objects in your view.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="pick",
        description="pick up a specific object in your view by marker id, the object will become your inventory",
        inputSchema={
            "type": "object",
            "properties": {
                "marker_id": {
                    "type": "string",
                    "description": "id of the marker on the object to pick up",
                },
            },
            "required": ["marker_id"],
        },
    ),
    types.Tool(
        name="place",
        description="place your inventory on top of a receptacle surface.",
        inputSchema={
            "type": "object",
            "properties": {
                "marker_id": {
                    "type": "string",
                    "description": "id of the marker on the receptacle surface to place your inventory",
                },
            },
            "required": ["marker_id"],
        },
    ),
    types.Tool(
        name="open",
        description="open the door of an articulated object",
        inputSchema={
            "type": "object",
            "properties": {
                "marker_id": {
                    "type": "string",
                    "description": "marker id of the receptacle door to open",
                },
            },
            "required": ["marker_id"],
        },
    ),
    types.Tool(
        name="close",
        description="close the door of an articulated object",
        inputSchema={
            "type": "object",
            "properties": {
                "marker_id": {
                    "type": "string",
                    "description": "marker id of the receptacle door to close",
                },
            },
            "required": ["marker_id"],
        },
    ),
    types.Tool(
        name="finish",
        description="you must call this tool to finish the episode when you think you have completed all the instructions.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="ask",
        description="ask the user a question to get more information about the task.",
        inputSchema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "the question to ask the user",
                }
            },
            "required": ["question"],
        },
    ),
]
