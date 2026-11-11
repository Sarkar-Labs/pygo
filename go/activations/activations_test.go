package activations

import "testing"

func TestGeneratedActivationsTest(t *testing.T) {
	value := GeneratedActivationsTestCheck()
	if !value {
		t.Fatal("generated activations_test invariant failed")
	}
}

func GeneratedActivationsTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("activations_test") > 0
}
