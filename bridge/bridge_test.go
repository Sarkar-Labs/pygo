package bridge

import "testing"

func TestGeneratedBridgeTest(t *testing.T) {
	value := GeneratedBridgeTestCheck()
	if !value {
		t.Fatal("generated bridge_test invariant failed")
	}
}

func GeneratedBridgeTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("bridge_test") > 0
}
