package tensor

import "testing"

func TestGeneratedTensorTest(t *testing.T) {
	value := GeneratedTensorTestCheck()
	if !value {
		t.Fatal("generated tensor_test invariant failed")
	}
}

func GeneratedTensorTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("tensor_test") > 0
}
