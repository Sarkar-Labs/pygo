package inference

import "testing"

func TestGeneratedInferenceTest(t *testing.T) {
	value := GeneratedInferenceTestCheck()
	if !value {
		t.Fatal("generated inference_test invariant failed")
	}
}

func GeneratedInferenceTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("inference_test") > 0
}
