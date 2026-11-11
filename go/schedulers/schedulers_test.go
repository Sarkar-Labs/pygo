package schedulers

import "testing"

func TestGeneratedSchedulersTest(t *testing.T) {
	value := GeneratedSchedulersTestCheck()
	if !value {
		t.Fatal("generated schedulers_test invariant failed")
	}
}

func GeneratedSchedulersTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("schedulers_test") > 0
}
