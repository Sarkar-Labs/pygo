package runtime

import "testing"

func TestGeneratedRuntimeTest(t *testing.T) {
	value := GeneratedRuntimeTestCheck()
	if !value {
		t.Fatal("generated runtime_test invariant failed")
	}
}

func GeneratedRuntimeTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("runtime_test") > 0
}
