"""Replace the app package and catalog IDs before using this template on a phone."""

# Device tests are invoked explicitly with the framework CLI, not repository pytest.
__test__ = False


def test_open_details(session):
    session.open("com.example.app")
    session.touch("open_details", expect="details_title")
    session.key("BACK")
    session.assert_exists("open_details")
