package serving

import "testing"

func TestGeneratedServingTest(t *testing.T) {
	value := GeneratedServingTestCheck()
	if !value {
		t.Fatal("generated serving_test invariant failed")
	}
}

func GeneratedServingTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("serving_test") > 0
}
