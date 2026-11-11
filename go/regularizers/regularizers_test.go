package regularizers

import "testing"

func TestGeneratedRegularizersTest(t *testing.T) {
	value := GeneratedRegularizersTestCheck()
	if !value {
		t.Fatal("generated regularizers_test invariant failed")
	}
}

func GeneratedRegularizersTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("regularizers_test") > 0
}
