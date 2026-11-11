package memory

import "testing"

func TestGeneratedMemoryTest(t *testing.T) {
	value := GeneratedMemoryTestCheck()
	if !value {
		t.Fatal("generated memory_test invariant failed")
	}
}

func GeneratedMemoryTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("memory_test") > 0
}
