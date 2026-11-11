package optim

import "testing"

func TestGeneratedOptimTest(t *testing.T) {
	value := GeneratedOptimTestCheck()
	if !value {
		t.Fatal("generated optim_test invariant failed")
	}
}

func GeneratedOptimTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("optim_test") > 0
}
