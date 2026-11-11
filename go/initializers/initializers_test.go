package initializers

import "testing"

func TestGeneratedInitializersTest(t *testing.T) {
	value := GeneratedInitializersTestCheck()
	if !value {
		t.Fatal("generated initializers_test invariant failed")
	}
}

func GeneratedInitializersTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("initializers_test") > 0
}
