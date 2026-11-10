import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import TrafficMonitoringWindow


def main() -> None:
    """Entry point for the Smart Urban Traffic Monitoring & ANPR System."""
    app = QApplication(sys.argv)
    window = TrafficMonitoringWindow(initial_cam_key="3")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
