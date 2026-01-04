# app/action/corridors.py

def corridor_from_toe_angle(toe_deg):
    """
    Defines the expected hip corridor from BFC toe angle.
    """
    if toe_deg >= 60:
        return "SIDE_ON", (45, 90)
    if toe_deg >= 30:
        return "HYBRID", (25, 70)
    return "FRONT_ON", (0, 40)


def within_corridor(angle, corridor):
    low, high = corridor
    return low <= angle <= high

