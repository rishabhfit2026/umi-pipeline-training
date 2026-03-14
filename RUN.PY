import numpy as np

# ==========================================================
# 🛑 DEFINE SAFE WORKSPACE (CHANGE TO YOUR REAL M750 LIMITS)
# ==========================================================

M750_LIMITS = {
    "x_min": -400,   # mm
    "x_max":  400,
    "y_min": -400,
    "y_max":  400,
    "z_min":   50,
    "z_max":  600,

    "rx_min": -180,  # degrees
    "rx_max":  180,
    "ry_min": -180,
    "ry_max":  180,
    "rz_min": -180,
    "rz_max":  180,
}


# ==========================================================
# ✅ CHECK IF COMMAND IS SAFE
# ==========================================================

def is_safe_m750_command(cmd):
    x, y, z = cmd["pos_delta_mm"]
    rx, ry, rz = cmd["rot_delta_deg"]

    return (
        M750_LIMITS["x_min"] <= x <= M750_LIMITS["x_max"] and
        M750_LIMITS["y_min"] <= y <= M750_LIMITS["y_max"] and
        M750_LIMITS["z_min"] <= z <= M750_LIMITS["z_max"] and
        M750_LIMITS["rx_min"] <= rx <= M750_LIMITS["rx_max"] and
        M750_LIMITS["ry_min"] <= ry <= M750_LIMITS["ry_max"] and
        M750_LIMITS["rz_min"] <= rz <= M750_LIMITS["rz_max"]
    )


# ==========================================================
# 🔧 CLAMP COMMAND INSIDE SAFE RANGE
# ==========================================================

def clamp_m750_command(cmd):
    x, y, z = cmd["pos_delta_mm"]
    rx, ry, rz = cmd["rot_delta_deg"]

    x = np.clip(x, M750_LIMITS["x_min"], M750_LIMITS["x_max"])
    y = np.clip(y, M750_LIMITS["y_min"], M750_LIMITS["y_max"])
    z = np.clip(z, M750_LIMITS["z_min"], M750_LIMITS["z_max"])

    rx = np.clip(rx, M750_LIMITS["rx_min"], M750_LIMITS["rx_max"])
    ry = np.clip(ry, M750_LIMITS["ry_min"], M750_LIMITS["ry_max"])
    rz = np.clip(rz, M750_LIMITS["rz_min"], M750_LIMITS["rz_max"])

    cmd["pos_delta_mm"] = [float(x), float(y), float(z)]
    cmd["rot_delta_deg"] = [float(rx), float(ry), float(rz)]

    return cmd


# ==========================================================
# 🤖 SAFE EXECUTION WRAPPER
# ==========================================================

def execute_m750_safe(robot_execute_fn, command):
    """
    Wraps your M750 execution with safety check.
    """

    safe = is_safe_m750_command(command)

    if not safe:
        print("⚠️ Command outside safe range — clamping.")
        command = clamp_m750_command(command)

    robot_execute_fn(command)
