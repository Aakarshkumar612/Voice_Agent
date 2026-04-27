def book_demo(name, date, time="10:00 AM"):
    return {"status": "confirmed", "message": f"Demo booked for {name} on {date} at {time}.", "booking_id": f"DEMO-{abs(hash(name+date)) % 10000:04d}"}

def check_slot_availability(date):
    slots = ["9:00 AM", "11:00 AM", "2:00 PM", "4:00 PM"]
    return {"date": date, "available_slots": slots, "message": f"Available: {', '.join(slots)}"}

def get_current_time():
    from datetime import datetime
    now = datetime.now()
    return {"time": now.strftime("%I:%M %p"), "date": now.strftime("%B %d, %Y")}

TOOL_REGISTRY = {"book_demo": book_demo, "check_slot_availability": check_slot_availability, "get_current_time": get_current_time}

def dispatch(tool_name, args):
    fn = TOOL_REGISTRY.get(tool_name)
    return str(fn(**args)) if fn else f"Unknown tool: {tool_name}"

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "book_demo",
            "description": "Book a product demo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Customer name"},
                    "date": {"type": "string", "description": "Demo date"},
                    "time": {"type": "string", "description": "Demo time"}
                },
                "required": ["name", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_slot_availability",
            "description": "Check available demo slots on a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date to check availability"}
                },
                "required": ["date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current date and time.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]
