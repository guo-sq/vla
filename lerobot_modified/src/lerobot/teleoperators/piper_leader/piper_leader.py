import logging
from typing import Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import MotorCalibration
from lerobot.teleoperators import Teleoperator

from .config_piper_leader import PiperLeaderConfig
from lerobot.robots.piper.piper_sdk_interface import PiperSDKInterface

logger = logging.getLogger(__name__)


class PiperLeader(Teleoperator):
    """
    Piper robot arm as a leader device for teleoperation.
    Reads joint positions via Piper SDK and provides standard LeRobot interface.
    """
    
    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig):
        super().__init__(config)
        self.config = config
        self.sdk: PiperSDKInterface | None = None

    @property
    def action_features(self) -> dict[str, type]:
        """Return 7-DOF action features (6 joints + 1 gripper)."""
        return {f"joint_{i}.pos": float for i in range(7)}

    @property
    def feedback_features(self) -> dict[str, type]:
        """Return 7-DOF feedback features (same as action features)."""
        return {f"joint_{i}.pos": float for i in range(7)}

    @property
    def is_connected(self) -> bool:
        """Check if SDK interface is active."""
        return self.sdk is not None

    def connect(self, calibrate: bool = True) -> None:
        """Initialize connection to Piper arm via CAN bus."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Connecting to Piper leader on {self.config.port}")
        self.sdk = PiperSDKInterface(port=self.config.port)

        if not self.is_calibrated and calibrate:
            logger.info("Calibration data not found or incomplete")
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected successfully")

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration file exists for this robot ID."""
        return self.calibration_fpath.is_file()

    def calibrate(self) -> None:
        """
        Calibrate by storing position limits from Piper SDK.
        Unlike SO100, no manual movement needed - limits are read from robot memory.
        """
        if self.is_calibrated:
            user_input = input(
                f"\nCalibration exists for {self.id}. Press ENTER to use it, or 'c' to recalibrate: "
            )
            if user_input.strip().lower() != "c":
                self._load_calibration()
                return

        logger.info(f"\nCalibrating {self}...")
        
        # Piper SDK already read hardware limits during __init__
        self.calibration = {}
        for i in range(7):
            motor_name = f"joint_{i}"
            self.calibration[motor_name] = MotorCalibration(
                id=i,
                drive_mode=0,  # Standard direction
                homing_offset=0,  # Piper handles homing internally
                range_min=self.sdk.min_pos[i],
                range_max=self.sdk.max_pos[i],
            )

        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Apply runtime configuration (Piper SDK already sets position mode)."""
        if not self.sdk:
            raise DeviceNotConnectedError("SDK not initialized")
        logger.debug(f"{self} configured")

    def get_action(self) -> dict[str, float]:
        """
        Read current joint positions from Piper.
        Returns mapped positions in -100% to 100% range (matches Piper follower).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # SDK returns fully mapped status dict
        status = self.sdk.get_status()
        
        # Filter to only action features
        action = {key: status[key] for key in self.action_features.keys()}
        
        logger.debug(f"{self} action: {action}")
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Move leader arm to target position via standard joint control.

        Raises:
            KeyError: If any of the 7 required joint keys is missing. Missing
                keys used to silently default to 0.0, which could command the
                arm to jump to 0° — a physical safety hazard.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        expected_keys = [f"joint_{i}.pos" for i in range(7)]
        missing = [k for k in expected_keys if k not in feedback]
        if missing:
            raise KeyError(
                f"send_feedback missing required keys: {missing}. "
                f"Got keys: {sorted(feedback.keys())}"
            )
        positions = [feedback[f"joint_{i}.pos"] for i in range(7)]
        self.sdk.set_joint_positions(positions)

    def disconnect(self) -> None:
        """Close connection to Piper arm."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        self.sdk.disconnect()
        self.sdk = None
        logger.info(f"{self} disconnected")