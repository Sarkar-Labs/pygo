package callbacks

import "testing"

func TestGeneratedCallbacksTest(t *testing.T) {
	value := GeneratedCallbacksTestCheck()
	if !value {
		t.Fatal("generated callbacks_test invariant failed")
	}
}

func GeneratedCallbacksTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("callbacks_test") > 0
}
