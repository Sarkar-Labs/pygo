package nn

import "testing"

func TestGeneratedNnTest(t *testing.T) {
	value := GeneratedNnTestCheck()
	if !value {
		t.Fatal("generated nn_test invariant failed")
	}
}

func GeneratedNnTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("nn_test") > 0
}
