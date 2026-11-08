"""Check sensor parsing so a stale or missing reading cannot become a zero."""
import unittest

from benchmarks.android.compare_genshin import parse_temperatures, temperature_summary


class TemperatureTests(unittest.TestCase):
    def test_battery_units_and_fresh_hal_override(self):
        battery = "  temperature: 339\n  USB powered: true\n  level: 86\n"
        thermal = (
            "Thermal Status: 1\nCached temperatures:\n"
            "Temperature{mValue=40.1, mType=0, mName=AP, mStatus=1}\n"
            "Current temperatures from HAL:\n"
            "Temperature{mValue=37.1, mType=0, mName=AP, mStatus=0}\n"
            "Temperature{mValue=36.9, mType=3, mName=SKIN, mStatus=1}\n")
        row = parse_temperatures(battery, thermal)
        self.assertEqual(row["battery_c"], 33.9)
        self.assertEqual(row["AP_c"], 37.1)
        self.assertEqual(row["SKIN_c"], 36.9)
        self.assertEqual(row["USB powered"], "true")

    def test_missing_sample_does_not_fabricate_cooling(self):
        rows = [{"AP_c": 35.0}, {"error": "unavailable"}, {"AP_c": 37.0}, {"AP_c": 36.0}]
        summary = temperature_summary(rows)
        self.assertEqual(summary["AP_c"], dict(start=35, end=36, peak=37, delta=1))
        self.assertNotIn("battery_c", summary)
        self.assertEqual(parse_temperatures("", ""), {})


if __name__ == "__main__":
    unittest.main()
