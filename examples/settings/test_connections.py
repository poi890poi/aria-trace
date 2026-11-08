"""Read the Samsung Settings connections page in English or Traditional Chinese."""

from swiftest import AnyOf, Node, open_app, touch, assert_exists, keyevent

# Run explicitly with the device CLI. Repository-wide pytest runs should stay offline.
__test__ = False

connections = AnyOf(
    Node(resource_id="android:id/title", text="連接"),
    Node(resource_id="android:id/title", text="Connections"),
)
wifi = Node(resource_id="android:id/title", text="Wi-Fi")


def test_connections():
    open_app("com.android.settings")
    touch(connections, expect=wifi)
    keyevent("BACK")
    assert_exists(connections)
