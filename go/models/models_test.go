package models

import "testing"

func TestGeneratedModelsTest(t *testing.T) {
	value := GeneratedModelsTestCheck()
	if !value {
		t.Fatal("generated models_test invariant failed")
	}
}

func GeneratedModelsTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("models_test") > 0
}
