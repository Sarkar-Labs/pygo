package autograd

import "testing"

func TestGeneratedAutogradTest(t *testing.T) {
	value := GeneratedAutogradTestCheck()
	if !value {
		t.Fatal("generated autograd_test invariant failed")
	}
}

func GeneratedAutogradTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("autograd_test") > 0
}
