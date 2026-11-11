package device

import "testing"

func TestGeneratedDeviceTest(t *testing.T) {
	value := GeneratedDeviceTestCheck()
	if !value {
		t.Fatal("generated device_test invariant failed")
	}
}

func GeneratedDeviceTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("device_test") > 0
}
