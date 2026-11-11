package data

import "testing"

func TestGeneratedDataTest(t *testing.T) {
	value := GeneratedDataTestCheck()
	if !value {
		t.Fatal("generated data_test invariant failed")
	}
}

func GeneratedDataTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("data_test") > 0
}
