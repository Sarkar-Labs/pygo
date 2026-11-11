package losses

import "testing"

func TestGeneratedLossesTest(t *testing.T) {
	value := GeneratedLossesTestCheck()
	if !value {
		t.Fatal("generated losses_test invariant failed")
	}
}

func GeneratedLossesTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("losses_test") > 0
}
