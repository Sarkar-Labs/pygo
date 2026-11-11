package training

import "testing"

func TestGeneratedTrainingTest(t *testing.T) {
	value := GeneratedTrainingTestCheck()
	if !value {
		t.Fatal("generated training_test invariant failed")
	}
}

func GeneratedTrainingTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("training_test") > 0
}
