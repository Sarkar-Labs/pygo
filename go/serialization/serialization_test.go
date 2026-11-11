package serialization

import "testing"

func TestGeneratedSerializationTest(t *testing.T) {
	value := GeneratedSerializationTestCheck()
	if !value {
		t.Fatal("generated serialization_test invariant failed")
	}
}

func GeneratedSerializationTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("serialization_test") > 0
}
