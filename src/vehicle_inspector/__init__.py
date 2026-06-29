"""Vehicle Inspector — vision-based car damage detection and multi-task inspection."""

__version__ = "0.1.0"

# Fixed CarDD damage class order, used across data, models, eval, and reporting.
DAMAGE_CLASSES = (
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "lamp_broken",
    "tire_flat",
)
