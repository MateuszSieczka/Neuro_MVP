class GlobalOscillator:
    """
    Global phase pacemaker (e.g., Theta rhythm).
    Synchronizes k-WTA evaluation across all network layers.
    """

    def __init__(self, base_period: int = 100) -> None:
        self.base_period = base_period
        self.current_phase: float = 0.0

    def tick(self, ne_level: float, sero_level: float) -> bool:
        """
        Advances the global clock.
        Returns True if a phase reset (k-WTA evaluation) should occur.
        """
        alpha = self.base_period * 0.5  # Serotonin lengthens window
        beta = self.base_period * 0.5  # Noradrenaline shortens window

        dyn_period = max(1, int(self.base_period + alpha * sero_level - beta * ne_level))

        self.current_phase += 1.0 / dyn_period

        if self.current_phase >= 1.0:
            self.current_phase -= 1.0
            return True
        return False

    def reset(self) -> None:
        self.current_phase = 0.0