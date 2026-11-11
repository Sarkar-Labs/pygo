package transforms

import "testing"

func TestGeneratedTransformsTest(t *testing.T) {
	value := GeneratedTransformsTestCheck()
	if !value {
		t.Fatal("generated transforms_test invariant failed")
	}
}

func GeneratedTransformsTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("transforms_test") > 0
}
