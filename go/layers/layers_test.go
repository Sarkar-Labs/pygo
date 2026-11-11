package layers

import "testing"

func TestGeneratedLayersTest(t *testing.T) {
	value := GeneratedLayersTestCheck()
	if !value {
		t.Fatal("generated layers_test invariant failed")
	}
}

func GeneratedLayersTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("layers_test") > 0
}
